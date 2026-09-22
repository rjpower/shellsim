"""Common asyncio behavior expected from shellsim's cooperative Python runtime.

The suite intentionally tests observable behavior rather than event-loop internals. Shellsim may
schedule coroutines with deterministic green threads, but programs that make progress on CPython
must not deadlock merely because a sibling coroutine owns the corresponding wakeup.
"""

import asyncio


def test_async_functions_are_lazy_and_return_values_through_await():
    events = []

    async def inner(value):
        events.append(("inner", value))
        return value * 2

    async def outer():
        events.append(("outer", "start"))
        result = await inner(21)
        events.append(("outer", "done"))
        return result

    coroutine = outer()
    assert events == []
    assert asyncio.run(coroutine) == 42
    assert events == [("outer", "start"), ("inner", 21), ("outer", "done")]


def test_sleep_zero_yields_and_gather_preserves_input_order():
    events = []

    async def worker(name, steps):
        for step in range(steps):
            events.append((name, step))
            await asyncio.sleep(0)
        return name

    async def main():
        return await asyncio.gather(worker("a", 2), worker("b", 2))

    assert asyncio.run(main()) == ["a", "b"]
    assert events == [("a", 0), ("b", 0), ("a", 1), ("b", 1)]


def test_create_task_allows_a_sibling_to_wake_an_event_waiter():
    events = []

    async def main():
        ready = asyncio.Event()

        async def consumer():
            events.append("consumer waiting")
            await ready.wait()
            events.append("consumer awake")
            return "consumed"

        async def producer():
            events.append("producer running")
            ready.set()

        consumer_task = asyncio.create_task(consumer())
        producer_task = asyncio.create_task(producer())
        await producer_task
        return await consumer_task

    assert asyncio.run(main()) == "consumed"
    assert events == ["consumer waiting", "producer running", "consumer awake"]


def test_current_task_is_stable_across_direct_awaits():
    async def nested(expected):
        assert asyncio.current_task() is expected

    async def main():
        task = asyncio.current_task()
        await nested(task)
        assert asyncio.current_task() is task

    asyncio.run(main())


def test_event_state_is_visible_and_clearable():
    async def main():
        event = asyncio.Event()
        assert not event.is_set()
        event.set()
        assert event.is_set()
        await event.wait()
        event.clear()
        assert not event.is_set()

    asyncio.run(main())


def test_queue_is_fifo_and_blocked_get_is_woken_by_put():
    async def main():
        queue = asyncio.Queue()

        async def consumer():
            return [await queue.get(), await queue.get()]

        task = asyncio.create_task(consumer())
        await asyncio.sleep(0)
        await queue.put("first")
        await queue.put("second")
        return await task

    assert asyncio.run(main()) == ["first", "second"]


def test_bounded_queue_applies_backpressure_and_tracks_work():
    events = []

    async def main():
        queue = asyncio.Queue(maxsize=1)
        assert queue.maxsize == 1
        assert queue.empty()
        await queue.put("first")
        assert queue.full()

        async def producer():
            events.append("put waiting")
            await queue.put("second")
            events.append("put complete")

        producer_task = asyncio.create_task(producer())
        await asyncio.sleep(0)
        assert events == ["put waiting"]

        assert await queue.get() == "first"
        queue.task_done()
        await producer_task
        assert events == ["put waiting", "put complete"]

        join_task = asyncio.create_task(queue.join())
        await asyncio.sleep(0)
        assert not join_task.done()
        assert queue.get_nowait() == "second"
        queue.task_done()
        await join_task
        assert queue.empty()

    asyncio.run(main())


def test_queue_nowait_errors_and_specialized_ordering():
    async def main():
        queue = asyncio.Queue(maxsize=1)
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
        else:
            raise AssertionError("empty queue must raise QueueEmpty")

        queue.put_nowait("full")
        try:
            queue.put_nowait("overflow")
        except asyncio.QueueFull:
            pass
        else:
            raise AssertionError("full queue must raise QueueFull")

        priority = asyncio.PriorityQueue()
        await priority.put((2, "second"))
        await priority.put((1, "first"))
        assert await priority.get() == (1, "first")
        assert await priority.get() == (2, "second")

        lifo = asyncio.LifoQueue()
        await lifo.put("first")
        await lifo.put("second")
        assert await lifo.get() == "second"
        assert await lifo.get() == "first"

    asyncio.run(main())


def test_cancelled_queue_waiters_do_not_consume_items_or_capacity():
    async def main():
        queue = asyncio.Queue(maxsize=1)
        await queue.put("existing")
        cancelled_put = asyncio.create_task(queue.put("cancelled"))
        await asyncio.sleep(0)
        cancelled_put.cancel()
        try:
            await cancelled_put
        except asyncio.CancelledError:
            pass

        assert await queue.get() == "existing"
        queue.task_done()
        await queue.put("replacement")
        assert await queue.get() == "replacement"
        queue.task_done()

        cancelled_get = asyncio.create_task(queue.get())
        await asyncio.sleep(0)
        cancelled_get.cancel()
        try:
            await cancelled_get
        except asyncio.CancelledError:
            pass
        await queue.put("delivered")
        assert await queue.get() == "delivered"
        queue.task_done()
        await queue.join()

    asyncio.run(main())


def test_lock_serializes_tasks_across_await_points():
    events = []

    async def main():
        lock = asyncio.Lock()

        async def worker(name):
            async with lock:
                events.append((name, "enter"))
                await asyncio.sleep(0)
                events.append((name, "exit"))

        await asyncio.gather(worker("a"), worker("b"))

    asyncio.run(main())
    assert events == [
        ("a", "enter"),
        ("a", "exit"),
        ("b", "enter"),
        ("b", "exit"),
    ]


def test_task_exceptions_propagate_to_awaiters_and_gather():
    async def fail():
        await asyncio.sleep(0)
        raise ValueError("boom")

    async def main():
        task = asyncio.create_task(fail())
        try:
            await task
        except ValueError as error:
            assert str(error) == "boom"
        else:
            raise AssertionError("awaiting a failed task must raise its exception")

        results = await asyncio.gather(fail(), return_exceptions=True)
        assert len(results) == 1
        assert isinstance(results[0], ValueError)

    asyncio.run(main())


def test_wait_for_times_out_and_cancels_the_child_task():
    events = []

    async def waits_forever():
        try:
            await asyncio.Event().wait()
        finally:
            events.append("cleaned")

    async def main():
        try:
            await asyncio.wait_for(waits_forever(), 0.001)
        except TimeoutError:
            return "timed out"
        return "unexpected"

    assert asyncio.run(main()) == "timed out"
    assert events == ["cleaned"]


def test_task_cancellation_runs_finally_and_reports_cancelled_state():
    events = []

    async def worker():
        try:
            await asyncio.Event().wait()
        finally:
            events.append("cancelled")

    async def main():
        task = asyncio.create_task(worker())
        await asyncio.sleep(0)
        assert task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("awaiting a cancelled task must raise CancelledError")
        assert task.done()
        assert task.cancelled()

    asyncio.run(main())
    assert events == ["cancelled"]


def test_async_context_managers_and_iterators_follow_protocols():
    events = []

    class Context:
        async def __aenter__(self):
            events.append("enter")
            return "value"

        async def __aexit__(self, kind, value, traceback):
            events.append("exit")

    class Values:
        def __init__(self):
            self.current = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.current == 3:
                raise StopAsyncIteration
            value = self.current
            self.current += 1
            await asyncio.sleep(0)
            return value

    async def main():
        async with Context() as value:
            assert value == "value"
            observed = []
            async for item in Values():
                observed.append(item)
        return observed

    assert asyncio.run(main()) == [0, 1, 2]
    assert events == ["enter", "exit"]


def test_async_context_manager_can_suppress_an_exception():
    events = []

    class Context:
        async def __aenter__(self):
            events.append("enter")

        async def __aexit__(self, kind, value, traceback):
            events.append(kind is not None)
            return True

    async def main():
        async with Context():
            raise ValueError("suppressed")
        events.append("continued")

    asyncio.run(main())
    assert events == ["enter", True, "continued"]


def test_async_subprocess_communicate_captures_output_and_status():
    async def main():
        process = await asyncio.create_subprocess_exec(
            "sh",
            "-c",
            "printf stdout; printf stderr >&2; exit 3",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        return process.returncode, stdout, stderr

    assert asyncio.run(main()) == (3, b"stdout", b"stderr")


def test_async_subprocess_streams_handle_duplex_backpressure():
    data = b"abcdefgh" * 20000

    async def main():
        process = await asyncio.create_subprocess_exec(
            "cat",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate(data)
        return process.returncode, stdout, stderr

    status, stdout, stderr = asyncio.run(main())
    assert status == 0
    assert stdout == data
    assert stderr is None


def test_async_subprocess_line_reads_and_wait_are_cooperative():
    async def main():
        process = await asyncio.create_subprocess_shell(
            "printf 'first\\nsecond\\n'",
            stdout=asyncio.subprocess.PIPE,
        )
        first = await process.stdout.readline()
        remainder = await process.stdout.read()
        status = await process.wait()
        return first, remainder, status

    assert asyncio.run(main()) == (b"first\n", b"second\n", 0)


def test_async_subprocess_timeout_can_cancel_wait_without_deadlock():
    async def main():
        process = await asyncio.create_subprocess_exec("sleep", "10")
        try:
            await asyncio.wait_for(process.wait(), 1)
        except TimeoutError:
            process.kill()
        return await process.wait()

    assert asyncio.run(main()) == -9


def test_async_subprocess_writer_close_flushes_buffered_input():
    data = b"close-after-write" * 1000

    async def main():
        process = await asyncio.create_subprocess_exec(
            "cat",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
        )
        process.stdin.write(data)
        process.stdin.close()
        await process.stdin.wait_closed()
        output = await process.stdout.read()
        status = await process.wait()
        return output, status

    assert asyncio.run(main()) == (data, 0)


def test_wait_for_none_disables_the_timeout():
    async def value():
        await asyncio.sleep(0)
        return 42

    assert asyncio.run(asyncio.wait_for(value(), None)) == 42


def test_timeout_context_cancels_the_body_and_runs_cleanup():
    events = []

    async def main():
        guard = None
        try:
            async with asyncio.timeout(0.001) as guard:
                try:
                    await asyncio.Event().wait()
                finally:
                    events.append("cleaned")
        except TimeoutError:
            pass
        else:
            raise AssertionError("expired context must raise TimeoutError")
        assert guard.expired()
        assert asyncio.current_task().cancelling() == 0

    asyncio.run(main())
    assert events == ["cleaned"]


def test_timeout_can_be_disabled_rescheduled_and_set_at_a_deadline():
    async def main():
        loop = asyncio.get_running_loop()
        async with asyncio.timeout(None) as guard:
            assert guard.when() is None
            guard.reschedule(loop.time() + 1)
            assert guard.when() is not None
            await asyncio.sleep(0)
        assert not guard.expired()

        async with asyncio.timeout_at(loop.time() + 1):
            await asyncio.sleep(0)

        try:
            async with asyncio.timeout(None) as rescheduled:
                rescheduled.reschedule(loop.time())
                await asyncio.Event().wait()
        except TimeoutError:
            assert rescheduled.expired()
        else:
            raise AssertionError("rescheduled timeout must expire")

    asyncio.run(main())


def test_timeout_does_not_transform_external_cancellation():
    async def worker():
        async with asyncio.timeout(10):
            await asyncio.Event().wait()

    async def main():
        task = asyncio.create_task(worker())
        await asyncio.sleep(0)
        task.cancel()
        outcome = None
        try:
            await task
        except asyncio.CancelledError:
            outcome = "cancelled"
        except TimeoutError:
            outcome = "timeout"
        assert outcome == "cancelled"

    asyncio.run(main())


def test_wait_supports_completion_modes_and_non_destructive_timeouts():
    async def delayed(value, delay):
        await asyncio.sleep(delay)
        return value

    async def main():
        slow = asyncio.create_task(delayed("slow", 2))
        fast = asyncio.create_task(delayed("fast", 1))
        done, pending = await asyncio.wait({slow, fast}, return_when=asyncio.FIRST_COMPLETED)
        assert len(done) == 1
        assert next(iter(done)).result() == "fast"
        assert pending == {slow}

        done, pending = await asyncio.wait({slow}, timeout=0)
        assert done == set()
        assert pending == {slow}
        assert not slow.cancelled()
        return await slow

    assert asyncio.run(main()) == "slow"


def test_as_completed_yields_results_in_completion_order():
    async def delayed(value, delay):
        await asyncio.sleep(delay)
        return value

    async def main():
        results = []
        for result in asyncio.as_completed([delayed("last", 0.003), delayed("first", 0.001), delayed("middle", 0.002)]):
            results.append(await result)
        return results

    assert asyncio.run(main()) == ["first", "middle", "last"]


def test_shield_keeps_the_inner_task_alive_when_its_waiter_is_cancelled():
    events = []

    async def main():
        release = asyncio.Event()

        async def inner():
            await release.wait()
            events.append("inner finished")
            return 42

        inner_task = asyncio.create_task(inner())

        async def outer():
            return await asyncio.shield(inner_task)

        outer_task = asyncio.create_task(outer())
        await asyncio.sleep(0)
        outer_task.cancel()
        try:
            await outer_task
        except asyncio.CancelledError:
            pass
        release.set()
        assert await inner_task == 42
        return inner_task.cancelled()

    assert asyncio.run(main()) is False
    assert events == ["inner finished"]


def test_task_group_names_and_task_introspection():
    async def worker(value):
        assert asyncio.current_task() in asyncio.all_tasks()
        await asyncio.sleep(0)
        return value

    async def main():
        async with asyncio.TaskGroup() as group:
            first = group.create_task(worker(1), name="first")
            second = group.create_task(worker(2), name="second")
            assert first.get_name() == "first"
            second.set_name("renamed")
            assert second.get_name() == "renamed"
        return first.result() + second.result()

    assert asyncio.run(main()) == 3


def test_task_callbacks_and_cancellation_introspection():
    observed = []

    async def value():
        return 42

    async def main():
        completed = asyncio.create_task(value())
        completed.add_done_callback(lambda task: observed.append(task.result()))
        assert await completed == 42
        assert completed.exception() is None

        cancelled = asyncio.create_task(value())
        cancelled.add_done_callback(lambda task: observed.append(task.cancelled()))
        assert cancelled.cancel()
        try:
            await cancelled
        except asyncio.CancelledError:
            pass
        await asyncio.sleep(0)
        assert cancelled.cancelling() == 1
        assert cancelled.uncancel() == 0

    asyncio.run(main())
    assert observed == [42, True]


def test_future_results_callbacks_and_identity_follow_asyncio_ordering():
    events = []

    async def main():
        loop = asyncio.get_running_loop()
        assert asyncio.get_event_loop() is loop
        future = loop.create_future()
        assert asyncio.isfuture(future)
        assert asyncio.ensure_future(future) is future
        assert future.get_loop() is loop
        assert not future.done()

        future.add_done_callback(lambda completed: events.append(completed.result()))
        loop.call_soon(future.set_result, 42)
        assert await future == 42
        assert future.done()
        assert not future.cancelled()
        assert future.result() == 42
        assert future.exception() is None
        assert events == [42]

        future.add_done_callback(lambda completed: events.append("late"))
        assert events == [42]
        await asyncio.sleep(0)
        assert events == [42, "late"]

    asyncio.run(main())


def test_future_exceptions_invalid_states_and_callback_removal():
    events = []

    async def main():
        loop = asyncio.get_running_loop()
        pending = loop.create_future()
        try:
            pending.result()
        except asyncio.InvalidStateError:
            pass
        else:
            raise AssertionError("pending future result must be unavailable")

        def callback(completed):
            events.append(completed.result())

        pending.add_done_callback(callback)
        assert pending.remove_done_callback(callback) == 1
        error = ValueError("future failed")
        loop.call_soon(pending.set_exception, error)
        try:
            await pending
        except ValueError as observed:
            assert observed is error
        else:
            raise AssertionError("future exception must cross await")
        assert pending.exception() is error
        assert events == []

        try:
            pending.set_result(1)
        except asyncio.InvalidStateError:
            pass
        else:
            raise AssertionError("completed future must reject a new result")

        invalid = loop.create_future()
        try:
            invalid.set_exception("not an exception")
        except TypeError:
            pass
        else:
            raise AssertionError("future must reject a non-exception failure")

    asyncio.run(main())


def test_task_cancellation_propagates_to_its_awaited_future():
    events = []

    async def main():
        loop = asyncio.get_running_loop()
        future = loop.create_future()

        async def waiter():
            try:
                await future
            finally:
                events.append("cleaned")

        task = asyncio.create_task(waiter())
        await asyncio.sleep(0)
        assert task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("cancelled future waiter must be cancelled")
        assert future.cancelled()
        assert task.cancelled()
        try:
            future.result()
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("cancelled future result must raise")

        timed = loop.create_future()
        loop.call_soon(timed.cancel)
        try:
            await asyncio.wait_for(timed, 1)
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("wait_for must preserve future cancellation")

    asyncio.run(main())
    assert events == ["cleaned"]


def test_task_cancellation_propagates_through_nested_awaits_but_not_shield():
    events = []

    async def child(name):
        try:
            await asyncio.Event().wait()
        finally:
            events.append(name + " cleaned")

    async def main():
        nested_child = asyncio.create_task(child("nested"))

        async def nested_parent():
            await nested_child

        nested = asyncio.create_task(nested_parent())
        await asyncio.sleep(0)
        assert nested.cancel()
        try:
            await nested
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("nested cancellation must reach the parent")
        assert nested_child.cancelled()

        async def direct_parent():
            await child("direct")

        direct = asyncio.create_task(direct_parent())
        await asyncio.sleep(0)
        direct.cancel()
        try:
            await direct
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("cancellation must cross a direct coroutine await")

        shielded_child = asyncio.create_task(child("shielded"))

        async def shielded_parent():
            await asyncio.shield(shielded_child)

        shielded = asyncio.create_task(shielded_parent())
        await asyncio.sleep(0)
        assert shielded.cancel()
        try:
            await shielded
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("shielded parent must still be cancelled")
        assert not shielded_child.done()
        shielded_child.cancel()
        try:
            await shielded_child
        except asyncio.CancelledError:
            pass

    asyncio.run(main())
    assert events == ["nested cleaned", "direct cleaned", "shielded cleaned"]


def test_gather_is_a_future_and_preserves_child_cancellation_semantics():
    events = []

    async def child(name):
        try:
            await asyncio.Event().wait()
        finally:
            events.append(name)

    async def main():
        first = asyncio.create_task(child("first"))
        second = asyncio.create_task(child("second"))
        group = asyncio.gather(first, second)
        assert asyncio.isfuture(group)
        await asyncio.sleep(0)
        assert group.cancel()
        try:
            await group
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("cancelled gather must raise CancelledError")
        assert not group.cancelled()
        assert first.cancelled()
        assert second.cancelled()

        cancelled = asyncio.create_task(child("returned"))
        returned = asyncio.gather(cancelled, return_exceptions=True)
        await asyncio.sleep(0)
        cancelled.cancel()
        results = await returned
        assert len(results) == 1
        assert isinstance(results[0], asyncio.CancelledError)

    asyncio.run(main())
    assert events == ["first", "second", "returned"]


def test_gather_failure_leaves_siblings_running_and_preserves_duplicate_results():
    events = []

    async def fail():
        await asyncio.sleep(0)
        raise ValueError("failed")

    async def finish():
        await asyncio.sleep(0.001)
        events.append("finished")
        return 9

    async def main():
        sibling = asyncio.create_task(finish())
        group = asyncio.gather(fail(), sibling)
        try:
            await group
        except ValueError as error:
            assert str(error) == "failed"
        else:
            raise AssertionError("gather must propagate its first child failure")
        assert not sibling.cancelled()
        assert await sibling == 9

        duplicate = asyncio.create_task(asyncio.sleep(0, result=5))
        assert await asyncio.gather(duplicate, duplicate) == [5, 5]

        assert await asyncio.shield(asyncio.sleep(0, result=11)) == 11

    asyncio.run(main())
    assert events == ["finished"]


def test_wait_returns_on_failure_without_cancelling_pending_tasks():
    events = []

    async def fail():
        await asyncio.sleep(0)
        raise ValueError("failed")

    async def finish():
        await asyncio.sleep(0.002)
        events.append("finished")
        return 7

    async def main():
        failed = asyncio.create_task(fail())
        pending_task = asyncio.create_task(finish())
        done, pending = await asyncio.wait({failed, pending_task}, return_when=asyncio.FIRST_EXCEPTION)
        assert done == {failed}
        assert pending == {pending_task}
        assert isinstance(failed.exception(), ValueError)
        assert await pending_task == 7

        timed = asyncio.create_task(finish())
        done, pending = await asyncio.wait({timed}, timeout=0)
        assert done == set()
        assert pending == {timed}
        assert await timed == 7

        surviving = asyncio.create_task(finish())
        waiting = asyncio.create_task(asyncio.wait({surviving}))
        await asyncio.sleep(0)
        waiting.cancel()
        try:
            await waiting
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("cancelled wait must raise CancelledError")
        assert not surviving.cancelled()
        assert await surviving == 7

    asyncio.run(main())
    assert events == ["finished", "finished", "finished"]


def test_callback_handles_are_fifo_cancellable_and_timer_backed():
    events = []

    async def main():
        loop = asyncio.get_running_loop()
        loop.call_soon(events.append, "soon first")
        cancelled = loop.call_soon(events.append, "cancelled")
        cancelled.cancel()
        assert cancelled.cancelled()
        loop.call_soon_threadsafe(events.append, "soon second")
        await asyncio.sleep(0)
        assert events == ["soon first", "soon second"]

        cancelled_timer = loop.call_later(0.001, events.append, "cancelled timer")
        assert cancelled_timer.when() >= loop.time()
        cancelled_timer.cancel()
        loop.call_later(0.001, events.append, "later")
        loop.call_at(loop.time() + 0.002, events.append, "at")
        await asyncio.sleep(0.003)
        assert events == ["soon first", "soon second", "later", "at"]

    asyncio.run(main())


def test_semaphore_and_condition_coordinate_waiters_in_fifo_order():
    events = []

    async def main():
        semaphore = asyncio.Semaphore(1)
        condition = asyncio.Condition()
        ready = False

        async def serialized(name):
            async with semaphore:
                events.append(name + " enter")
                await asyncio.sleep(0)
                events.append(name + " exit")

        async def consumer():
            async with condition:
                await condition.wait_for(lambda: ready)
                events.append("condition ready")

        async def producer():
            nonlocal ready
            await asyncio.sleep(0)
            async with condition:
                ready = True
                condition.notify_all()

        await asyncio.gather(serialized("a"), serialized("b"), consumer(), producer())

    asyncio.run(main())
    assert events.index("a enter") < events.index("a exit")
    assert events.index("a exit") < events.index("b enter")
    assert events.index("b enter") < events.index("b exit")
    assert events.count("condition ready") == 1
