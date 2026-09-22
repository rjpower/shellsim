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
