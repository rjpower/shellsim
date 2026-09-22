"""Deterministic cooperative subset of asyncio for shellsim."""

import subprocess

from _asyncio import (
    _is_coroutine,
    _monotonic_ns,
    _process_poll,
    _process_read,
    _process_write,
    _step,
    _wait_resources,
)


_running_loop = None

ALL_COMPLETED = "ALL_COMPLETED"
FIRST_COMPLETED = "FIRST_COMPLETED"
FIRST_EXCEPTION = "FIRST_EXCEPTION"
TimeoutError = TimeoutError


class CancelledError(Exception):
    pass


class QueueEmpty(Exception):
    pass


class QueueFull(Exception):
    pass


class InvalidStateError(Exception):
    pass


class Handle:
    def __init__(self, callback, args):
        self._callback = callback
        self._args = args
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def cancelled(self):
        return self._cancelled

    def _run(self):
        if self._cancelled:
            return
        try:
            self._callback(*self._args)
        except Exception:
            pass


class TimerHandle(Handle):
    def __init__(self, when, callback, args):
        Handle.__init__(self, callback, args)
        self._when = when

    def when(self):
        return self._when


class _Sleep:
    def __init__(self, delay):
        self.delay = delay


class _EventWait:
    def __init__(self, event):
        self.event = event


class _QueueGet:
    def __init__(self, queue):
        self.queue = queue


class _QueuePut:
    def __init__(self, queue, item):
        self.queue = queue
        self.item = item


class _LockWait:
    def __init__(self, lock):
        self.lock = lock


class _SemaphoreWait:
    def __init__(self, semaphore):
        self.semaphore = semaphore


class _ConditionWait:
    def __init__(self, condition):
        self.condition = condition


class _ResourceWait:
    def __init__(self, token):
        self.token = token


class _WaitFor:
    def __init__(self, task, timeout):
        self.task = task
        self.timeout = timeout


class _TimeoutWait:
    def __init__(self, parent, child):
        self.parent = parent
        self.child = child
        self.timed_out = False


class Future:
    def __init__(self, *, loop=None):
        if loop is None:
            loop = get_running_loop()
        self._loop = loop
        self._done = False
        self._cancelled = False
        self._result = None
        self._exception = None
        self._waiters = []
        self._callbacks = []

    def get_loop(self):
        return self._loop

    def done(self):
        return self._done

    def cancelled(self):
        return self._cancelled

    def result(self):
        if not self._done:
            raise InvalidStateError("Result is not ready")
        if self._cancelled:
            raise CancelledError()
        if self._exception is not None:
            raise self._exception
        return self._result

    def exception(self):
        if not self._done:
            raise InvalidStateError("Exception is not set")
        if self._cancelled:
            raise CancelledError()
        return self._exception

    def add_done_callback(self, callback, *, context=None):
        if self._done:
            self._loop.call_soon(callback, self)
        else:
            self._callbacks.append(callback)

    def remove_done_callback(self, callback):
        previous = len(self._callbacks)
        self._callbacks = [item for item in self._callbacks if item is not callback]
        return previous - len(self._callbacks)

    def cancel(self, msg=None):
        if self._done:
            return False
        self._done = True
        self._cancelled = True
        self._loop._run_done_callbacks(self)
        self._loop._wake_waiters(self)
        return True

    def set_result(self, result):
        if self._done:
            raise InvalidStateError("invalid state")
        self._done = True
        self._result = result
        self._loop._run_done_callbacks(self)
        self._loop._wake_waiters(self)

    def set_exception(self, exception):
        if self._done:
            raise InvalidStateError("invalid state")
        if not isinstance(exception, BaseException):
            raise TypeError("invalid exception object")
        self._done = True
        self._exception = exception
        self._loop._run_done_callbacks(self)
        self._loop._wake_waiters(self)


class Task:
    def __init__(self, coroutine, loop, name=None):
        if not _is_coroutine(coroutine):
            raise TypeError("a coroutine was expected")
        self._coroutine = coroutine
        self._loop = loop
        self._done = False
        self._cancelled = False
        self._result = None
        self._exception = None
        self._waiters = []
        self._send_value = None
        self._started = False
        self._waiting_on = None
        self._name = name
        self._callbacks = []
        self._cancel_requests = 0
        self._logical_task = self

    def done(self):
        return self._done

    def cancelled(self):
        return self._cancelled

    def result(self):
        if not self._done:
            raise RuntimeError("task is not complete")
        if self._exception is not None:
            raise self._exception
        return self._result

    def exception(self):
        if not self._done:
            raise RuntimeError("task is not complete")
        if self._cancelled:
            raise CancelledError()
        return self._exception

    def get_coro(self):
        return self._coroutine

    def get_name(self):
        return self._name

    def set_name(self, name):
        self._name = str(name)

    def cancelling(self):
        return self._cancel_requests

    def uncancel(self):
        if self._cancel_requests:
            self._cancel_requests -= 1
        return self._cancel_requests

    def add_done_callback(self, callback, context=None):
        if self._done:
            self._loop.call_soon(callback, self)
        else:
            self._callbacks.append(callback)

    def remove_done_callback(self, callback):
        previous = len(self._callbacks)
        self._callbacks = [item for item in self._callbacks if item is not callback]
        return previous - len(self._callbacks)

    def cancel(self, msg=None):
        if self._done:
            return False
        self._cancel_requests += 1
        self._loop._cancel(self)
        return True


class Event:
    def __init__(self):
        self._set = False
        self._waiters = []

    def is_set(self):
        return self._set

    def set(self):
        self._set = True
        waiters = self._waiters
        self._waiters = []
        for task in waiters:
            task._loop._schedule(task, True)

    def clear(self):
        self._set = False

    async def wait(self):
        if self._set:
            return True
        return await _EventWait(self)


class Queue:
    def __init__(self, maxsize=0):
        self._maxsize = maxsize
        self._items = []
        self._getters = []
        self._putters = []
        self._unfinished_tasks = 0
        self._finished = Event()
        self._finished.set()

    @property
    def maxsize(self):
        return self._maxsize

    def empty(self):
        return len(self._items) == 0

    def full(self):
        return self._maxsize > 0 and len(self._items) >= self._maxsize

    def qsize(self):
        return len(self._items)

    def _put(self, item):
        self._items.append(item)

    def _get(self):
        return self._items.pop(0)

    def _next_getter(self):
        while self._getters:
            task = self._getters.pop(0)
            if not task.done():
                return task
        return None

    def _next_putter(self):
        while self._putters:
            task, item = self._putters.pop(0)
            if not task.done():
                return task, item
        return None

    def _put_item(self, item):
        self._unfinished_tasks += 1
        self._finished.clear()
        getter = self._next_getter()
        if getter is None:
            self._put(item)
        else:
            getter._loop._schedule(getter, item)

    def _wake_putter(self):
        putter = self._next_putter()
        if putter is None:
            return
        task, item = putter
        self._put_item(item)
        task._loop._schedule(task, None)

    def _get_item(self):
        item = self._get()
        self._wake_putter()
        return item

    def put_nowait(self, item):
        if self.full():
            raise QueueFull()
        self._put_item(item)

    async def put(self, item):
        if self.full():
            return await _QueuePut(self, item)
        self._put_item(item)

    def get_nowait(self):
        if not self._items:
            raise QueueEmpty()
        return self._get_item()

    async def get(self):
        if self._items:
            return self._get_item()
        return await _QueueGet(self)

    def task_done(self):
        if self._unfinished_tasks <= 0:
            raise ValueError("task_done() called too many times")
        self._unfinished_tasks -= 1
        if self._unfinished_tasks == 0:
            self._finished.set()

    async def join(self):
        if self._unfinished_tasks:
            await self._finished.wait()


class PriorityQueue(Queue):
    def _put(self, item):
        self._items.append(item)
        self._items.sort()


class LifoQueue(Queue):
    def _get(self):
        return self._items.pop()


class Lock:
    def __init__(self):
        self._locked = False
        self._waiters = []

    def locked(self):
        return self._locked

    async def acquire(self):
        if not self._locked:
            self._locked = True
            return True
        return await _LockWait(self)

    def release(self):
        if not self._locked:
            raise RuntimeError("Lock is not acquired")
        if self._waiters:
            task = self._waiters.pop(0)
            task._loop._schedule(task, True)
        else:
            self._locked = False

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, kind, value, traceback):
        self.release()


class Semaphore:
    def __init__(self, value=1):
        if value < 0:
            raise ValueError("Semaphore initial value must be >= 0")
        self._value = value
        self._waiters = []

    def locked(self):
        return self._value == 0

    async def acquire(self):
        if self._value > 0:
            self._value -= 1
            return True
        return await _SemaphoreWait(self)

    def release(self):
        if self._waiters:
            task = self._waiters.pop(0)
            task._loop._schedule(task, True)
        else:
            self._value += 1

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, kind, value, traceback):
        self.release()


class BoundedSemaphore(Semaphore):
    def __init__(self, value=1):
        super().__init__(value)
        self._bound_value = value

    def release(self):
        if not self._waiters and self._value >= self._bound_value:
            raise ValueError("BoundedSemaphore released too many times")
        super().release()


class Condition:
    def __init__(self, lock=None):
        self._lock = lock if lock is not None else Lock()
        self._waiters = []

    def locked(self):
        return self._lock.locked()

    async def acquire(self):
        return await self._lock.acquire()

    def release(self):
        self._lock.release()

    async def wait(self):
        if not self.locked():
            raise RuntimeError("cannot wait on un-acquired lock")
        self.release()
        try:
            return await _ConditionWait(self)
        finally:
            await self.acquire()

    async def wait_for(self, predicate):
        result = predicate()
        while not result:
            await self.wait()
            result = predicate()
        return result

    def notify(self, n=1):
        if not self.locked():
            raise RuntimeError("cannot notify on un-acquired lock")
        for _ in range(min(n, len(self._waiters))):
            task = self._waiters.pop(0)
            task._loop._schedule(task, True)

    def notify_all(self):
        self.notify(len(self._waiters))

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, kind, value, traceback):
        self.release()


class TaskGroup:
    def __init__(self):
        self._tasks = []
        self._entered = False

    async def __aenter__(self):
        self._entered = True
        return self

    def create_task(self, coroutine, name=None, context=None):
        if not self._entered:
            raise RuntimeError("TaskGroup has not been entered")
        task = create_task(coroutine, name=name)
        self._tasks.append(task)
        return task

    async def __aexit__(self, kind, value, traceback):
        self._entered = False
        if kind is not None:
            for task in self._tasks:
                task.cancel()
            await gather(*self._tasks, return_exceptions=True)
            return False
        if not self._tasks:
            return False
        done, pending = await wait(self._tasks, return_when=FIRST_EXCEPTION)
        failed = []
        for task in self._tasks:
            if task not in done:
                continue
            if not task.cancelled() and task.exception() is not None:
                failed.append(task.exception())
        if failed:
            for task in pending:
                task.cancel()
            await gather(*pending, return_exceptions=True)
            raise failed[0]
        await gather(*pending)
        return False


class StreamReader:
    def __init__(self, process, fd):
        self._process = process
        self._fd = fd

    async def read(self, n=-1):
        while True:
            ready, value, token = _process_read(self._process._handle, self._fd, n)
            if ready:
                return value
            await _ResourceWait(token)

    async def readline(self):
        result = b""
        while True:
            part = await self.read(1)
            if not part:
                return result
            result += part
            if part == b"\n":
                return result


class StreamWriter:
    def __init__(self, process):
        self._process = process
        self._buffer = b""
        self._closing = False

    def write(self, data):
        if self._closing:
            raise RuntimeError("write to closing subprocess stream")
        if not isinstance(data, bytes):
            raise TypeError("subprocess stdin data must be bytes")
        self._buffer += data

    async def drain(self):
        while self._buffer:
            ready, written, token = _process_write(self._process._handle, self._buffer)
            if ready:
                self._buffer = self._buffer[written:]
            else:
                await _ResourceWait(token)

    def close(self):
        if not self._closing:
            self._closing = True
            if not self._buffer:
                self._process._popen.stdin.close()

    def is_closing(self):
        return self._closing

    async def wait_closed(self):
        await self.drain()
        if not self._process._popen.stdin.closed:
            self._process._popen.stdin.close()
        return None

    def write_eof(self):
        self.close()

    def can_write_eof(self):
        return True


class Process:
    def __init__(self, popen):
        self._popen = popen
        self._handle = popen._handle
        self.pid = popen.pid
        self.stdin = StreamWriter(self) if popen.stdin is not None else None
        self.stdout = StreamReader(self, 1) if popen.stdout is not None else None
        self.stderr = StreamReader(self, 2) if popen.stderr is not None else None

    @property
    def returncode(self):
        return self._popen.poll()

    async def wait(self):
        status = _process_poll(self._handle)
        while status is None:
            await _ResourceWait(("child", self.pid))
            status = _process_poll(self._handle)
        self._popen.returncode = status
        return status

    async def communicate(self, input=None):
        stdout_task = create_task(self.stdout.read()) if self.stdout is not None else None
        stderr_task = create_task(self.stderr.read()) if self.stderr is not None else None
        if input is not None:
            if self.stdin is None:
                raise ValueError("stdin was not opened with PIPE")
            self.stdin.write(input)
            await self.stdin.drain()
        if self.stdin is not None:
            self.stdin.close()
        status_task = create_task(self.wait())
        stdout = await stdout_task if stdout_task is not None else None
        stderr = await stderr_task if stderr_task is not None else None
        await status_task
        return stdout, stderr

    def send_signal(self, signal):
        self._popen.send_signal(signal)

    def terminate(self):
        self._popen.terminate()

    def kill(self):
        self._popen.kill()


class _Loop:
    def __init__(self):
        self._ready = []
        self._timers = []
        self._resource_waiters = []
        self._tasks = []
        self._current = None

    def create_task(self, coroutine, name=None):
        task = Task(coroutine, self, name)
        self._tasks.append(task)
        self._ready.append(task)
        return task

    def create_future(self):
        return Future(loop=self)

    def _create_awaited_task(self, coroutine, parent):
        task = self.create_task(coroutine)
        task._logical_task = parent._logical_task
        return task

    def time(self):
        return _monotonic_ns() / 1000000000

    def call_soon(self, callback, *args, context=None):
        handle = Handle(callback, args)
        self._ready.append(handle)
        return handle

    def call_soon_threadsafe(self, callback, *args, context=None):
        return self.call_soon(callback, *args, context=context)

    def call_at(self, when, callback, *args, context=None):
        handle = TimerHandle(when, callback, args)
        self._timers.append((int(when * 1000000000), handle))
        return handle

    def call_later(self, delay, callback, *args, context=None):
        return self.call_at(self.time() + delay, callback, *args, context=context)

    def _schedule(self, task, value):
        if not task._done:
            self._resource_waiters = [
                waiter for waiter in self._resource_waiters if waiter[0] is not task
            ]
            task._send_value = (True, value)
            self._ready.append(task)

    def _schedule_failure(self, task, error):
        if not task._done:
            self._resource_waiters = [
                waiter for waiter in self._resource_waiters if waiter[0] is not task
            ]
            task._send_value = (False, error)
            self._ready.append(task)

    def _cancel(self, task):
        if isinstance(task._waiting_on, Future) and task._waiting_on.cancel():
            return
        if not task._started:
            task._done = True
            task._cancelled = True
            task._exception = CancelledError()
            self._wake_waiters(task)
            self._run_done_callbacks(task)
        else:
            self._schedule_failure(task, CancelledError())

    def _sleep(self, task, delay):
        if delay <= 0:
            self._schedule(task, None)
        else:
            self._timers.append((_monotonic_ns() + int(delay * 1000000000), task))

    def _wake_due_timers(self):
        now = _monotonic_ns()
        pending = []
        for timer_deadline, timer in self._timers:
            if isinstance(timer, TimerHandle) and timer.cancelled():
                continue
            if timer_deadline <= now:
                if isinstance(timer, _TimeoutWait):
                    if timer.parent._waiting_on is timer:
                        timer.timed_out = True
                        timer.child.cancel()
                elif isinstance(timer, TimerHandle):
                    self._ready.append(timer)
                else:
                    self._schedule(timer, None)
            else:
                pending.append((timer_deadline, timer))
        self._timers = pending

    def _next_timer_token(self):
        active = [
            item
            for item in self._timers
            if not isinstance(item[1], TimerHandle) or not item[1].cancelled()
        ]
        if not active:
            return None
        deadline = active[0][0]
        for timer_deadline, timer in active:
            if timer_deadline < deadline:
                deadline = timer_deadline
        return ("timer", deadline)

    def _wait_for_resources(self):
        reasons = [waiter[1] for waiter in self._resource_waiters]
        timer = self._next_timer_token()
        if timer is not None:
            reasons.append(timer)
        if not reasons:
            raise RuntimeError("asyncio deadlock: no runnable tasks")
        _wait_resources(reasons)
        self._wake_due_timers()
        waiters = self._resource_waiters
        self._resource_waiters = []
        for task, token in waiters:
            self._schedule(task, None)

    def _wake_waiters(self, future):
        waiters = future._waiters
        future._waiters = []
        for waiter in waiters:
            if isinstance(waiter, _TimeoutWait):
                waiter.parent._waiting_on = None
                if waiter.timed_out:
                    self._schedule_failure(waiter.parent, TimeoutError())
                elif future._cancelled:
                    self._schedule_failure(waiter.parent, CancelledError())
                elif future._exception is not None:
                    self._schedule_failure(waiter.parent, future._exception)
                else:
                    self._schedule(waiter.parent, future._result)
            elif future._cancelled:
                if waiter._waiting_on is future:
                    waiter._waiting_on = None
                self._schedule_failure(waiter, CancelledError())
            elif future._exception is not None:
                if waiter._waiting_on is future:
                    waiter._waiting_on = None
                self._schedule_failure(waiter, future._exception)
            else:
                if waiter._waiting_on is future:
                    waiter._waiting_on = None
                self._schedule(waiter, future._result)

    def _run_done_callbacks(self, future):
        callbacks = future._callbacks
        future._callbacks = []
        for callback in callbacks:
            self.call_soon(callback, future)

    def _finish(self, task, result):
        task._done = True
        task._result = result
        self._wake_waiters(task)
        self._run_done_callbacks(task)

    def _fail(self, task, error):
        task._done = True
        task._exception = error
        task._cancelled = isinstance(error, CancelledError)
        self._wake_waiters(task)
        self._run_done_callbacks(task)

    def _block_on_future(self, task, awaited):
        if awaited._done:
            if awaited._cancelled:
                self._schedule_failure(task, CancelledError())
            elif awaited._exception is not None:
                self._schedule_failure(task, awaited._exception)
            else:
                self._schedule(task, awaited._result)
        else:
            if isinstance(awaited, Future):
                task._waiting_on = awaited
            awaited._waiters.append(task)

    def _dispatch_yield(self, task, awaited):
        if isinstance(awaited, (Task, Future)):
            self._block_on_future(task, awaited)
        elif isinstance(awaited, _Sleep):
            self._sleep(task, awaited.delay)
        elif isinstance(awaited, _EventWait):
            if awaited.event.is_set():
                self._schedule(task, True)
            else:
                awaited.event._waiters.append(task)
        elif isinstance(awaited, _QueueGet):
            if awaited.queue._items:
                self._schedule(task, awaited.queue._get_item())
            else:
                awaited.queue._getters.append(task)
        elif isinstance(awaited, _QueuePut):
            if awaited.queue.full():
                awaited.queue._putters.append((task, awaited.item))
            else:
                awaited.queue._put_item(awaited.item)
                self._schedule(task, None)
        elif isinstance(awaited, _LockWait):
            awaited.lock._waiters.append(task)
        elif isinstance(awaited, _SemaphoreWait):
            awaited.semaphore._waiters.append(task)
        elif isinstance(awaited, _ConditionWait):
            awaited.condition._waiters.append(task)
        elif isinstance(awaited, _WaitFor):
            if awaited.task._done:
                self._block_on_future(task, awaited.task)
            else:
                timeout = _TimeoutWait(task, awaited.task)
                task._waiting_on = timeout
                awaited.task._waiters.append(timeout)
                self._timers.append(
                    (_monotonic_ns() + int(awaited.timeout * 1000000000), timeout)
                )
        elif isinstance(awaited, _ResourceWait):
            self._resource_waiters.append((task, awaited.token))
        elif _is_coroutine(awaited):
            self._block_on_future(task, self._create_awaited_task(awaited, task))
        else:
            raise TypeError("object cannot be used in 'await' expression")

    def run(self, coroutine):
        root = self.create_task(coroutine)
        while not root.done():
            if not self._ready:
                self._wake_due_timers()
            if not self._ready:
                self._wait_for_resources()
            task = self._ready.pop(0)
            if isinstance(task, Handle):
                task._run()
                continue
            if task._done:
                continue
            task._started = True
            self._current = task
            status, value = _step(task._coroutine, task._send_value)
            self._current = None
            task._send_value = None
            if status == 1:
                self._finish(task, value)
            elif status == 2:
                self._fail(task, value)
            else:
                self._dispatch_yield(task, value)
        return root.result()


def get_running_loop():
    if _running_loop is None:
        raise RuntimeError("no running event loop")
    return _running_loop


def get_event_loop():
    return get_running_loop()


def create_task(coroutine, name=None, context=None):
    return get_running_loop().create_task(coroutine, name)


def ensure_future(awaitable):
    return _ensure_task(awaitable)


def isfuture(value):
    return isinstance(value, (Future, Task))


def current_task(loop=None):
    if loop is None:
        loop = get_running_loop()
    if loop._current is None:
        return None
    return loop._current._logical_task


def all_tasks(loop=None):
    if loop is None:
        loop = get_running_loop()
    return {task for task in loop._tasks if not task.done()}


def _ensure_task(awaitable):
    if isinstance(awaitable, (Task, Future)):
        return awaitable
    return create_task(awaitable)


async def sleep(delay, result=None):
    await _Sleep(delay)
    return result


async def gather(*coroutines, return_exceptions=False):
    tasks = [_ensure_task(coroutine) for coroutine in coroutines]
    results = []
    for task in tasks:
        try:
            results.append(await task)
        except Exception as error:
            if not return_exceptions:
                raise
            results.append(error)
    return results


async def wait_for(awaitable, timeout):
    task = _ensure_task(awaitable)
    if timeout is None:
        return await task
    return await _WaitFor(task, timeout)


class Timeout:
    def __init__(self, when):
        self._when = when
        self._task = None
        self._timer = None
        self._cancelling = 0
        self._expired = False

    def when(self):
        return self._when

    def expired(self):
        return self._expired

    def reschedule(self, when):
        if self._task is None:
            raise RuntimeError("Timeout has not been entered")
        if self._expired:
            raise RuntimeError("Cannot change state of expired Timeout")
        if self._timer is not None:
            self._timer.cancel()
        self._when = when
        self._timer = self._start_timer()

    def _start_timer(self):
        if self._when is None:
            return None

        async def expire():
            delay = self._when - get_running_loop().time()
            await sleep(delay)
            self._expired = True
            self._task.cancel()

        return create_task(expire())

    async def __aenter__(self):
        if self._task is not None:
            raise RuntimeError("Timeout has already been entered")
        self._task = current_task()
        self._cancelling = self._task.cancelling()
        self._timer = self._start_timer()
        return self

    async def __aexit__(self, kind, value, traceback):
        if self._timer is not None:
            self._timer.cancel()
        if not self._expired:
            return False
        remaining = self._task.uncancel()
        if remaining <= self._cancelling and isinstance(value, CancelledError):
            raise TimeoutError()
        return False


def timeout(delay):
    if delay is None:
        return Timeout(None)
    return Timeout(get_running_loop().time() + delay)


def timeout_at(when):
    return Timeout(when)


def _wait_condition(tasks, return_when):
    done = {task for task in tasks if task.done()}
    if return_when == FIRST_COMPLETED:
        return bool(done)
    if return_when == FIRST_EXCEPTION:
        for task in done:
            if not task.cancelled() and task.exception() is not None:
                return True
        return len(done) == len(tasks)
    return len(done) == len(tasks)


async def wait(awaitables, timeout=None, return_when=ALL_COMPLETED):
    if return_when not in (ALL_COMPLETED, FIRST_COMPLETED, FIRST_EXCEPTION):
        raise ValueError("invalid return_when value")
    tasks = []
    for awaitable in awaitables:
        if awaitable not in tasks:
            tasks.append(awaitable)
    if not tasks:
        raise ValueError("Set of Tasks/Futures is empty")
    for task in tasks:
        if not isinstance(task, (Task, Future)):
            raise TypeError("Passing coroutines is forbidden, use tasks explicitly")
    changed = Event()

    def task_completed(task):
        changed.set()

    observed = [task for task in tasks if not task.done()]
    for task in observed:
        task.add_done_callback(task_completed)

    async def wait_until_ready():
        while not _wait_condition(tasks, return_when):
            changed.clear()
            if _wait_condition(tasks, return_when):
                break
            await changed.wait()

    try:
        if timeout is None:
            await wait_until_ready()
        else:
            try:
                await wait_for(wait_until_ready(), timeout)
            except TimeoutError:
                pass
    finally:
        for task in observed:
            task.remove_done_callback(task_completed)
    done = {task for task in tasks if task.done()}
    return done, set(tasks) - done


def as_completed(awaitables, timeout=None):
    tasks = []
    for awaitable in awaitables:
        task = _ensure_task(awaitable)
        if task not in tasks:
            tasks.append(task)
    completed = Queue()
    deadline = None
    if timeout is not None:
        deadline = _monotonic_ns() + int(timeout * 1000000000)

    def task_completed(task):
        completed.put_nowait(task)

    for task in tasks:
        task.add_done_callback(task_completed)

    async def next_result():
        if deadline is None:
            task = await completed.get()
        else:
            remaining = (deadline - _monotonic_ns()) / 1000000000
            task = await wait_for(completed.get(), max(0, remaining))
        return task.result()

    for task in tasks:
        yield next_result()


async def shield(awaitable):
    return await _ensure_task(awaitable)


async def create_subprocess_exec(program, *args, stdin=None, stdout=None, stderr=None,
                                 cwd=None, env=None, start_new_session=False, limit=65536):
    if limit <= 0:
        raise ValueError("limit must be positive")
    popen = subprocess.Popen(
        [program] + list(args),
        stdin=stdin,
        stdout=stdout,
        stderr=stderr,
        cwd=cwd,
        env=env,
        start_new_session=start_new_session,
    )
    return Process(popen)


async def create_subprocess_shell(command, stdin=None, stdout=None, stderr=None,
                                  cwd=None, env=None, start_new_session=False, limit=65536):
    if limit <= 0:
        raise ValueError("limit must be positive")
    popen = subprocess.Popen(
        command,
        stdin=stdin,
        stdout=stdout,
        stderr=stderr,
        cwd=cwd,
        env=env,
        shell=True,
        start_new_session=start_new_session,
    )
    return Process(popen)


def run(coroutine):
    global _running_loop
    if _running_loop is not None:
        raise RuntimeError("asyncio.run() cannot be called from a running event loop")
    loop = _Loop()
    _running_loop = loop
    try:
        return loop.run(coroutine)
    finally:
        _running_loop = None
