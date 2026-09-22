"""Deterministic cooperative subset of asyncio for shellsim."""

from _asyncio import _is_coroutine, _step


_running_loop = None


class _Sleep:
    def __init__(self, delay):
        self.delay = delay


class _EventWait:
    def __init__(self, event):
        self.event = event


class _QueueGet:
    def __init__(self, queue):
        self.queue = queue


class _LockWait:
    def __init__(self, lock):
        self.lock = lock


class Task:
    def __init__(self, coroutine, loop):
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

    def empty(self):
        return len(self._items) == 0

    def qsize(self):
        return len(self._items)

    def put_nowait(self, item):
        if self._getters:
            task = self._getters.pop(0)
            task._loop._schedule(task, item)
        else:
            self._items.append(item)

    async def put(self, item):
        self.put_nowait(item)

    def get_nowait(self):
        if not self._items:
            raise RuntimeError("queue is empty")
        return self._items.pop(0)

    async def get(self):
        if self._items:
            return self._items.pop(0)
        return await _QueueGet(self)


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


class _Loop:
    def __init__(self):
        self._ready = []
        self._timers = []
        self._time = 0.0

    def create_task(self, coroutine):
        task = Task(coroutine, self)
        self._ready.append(task)
        return task

    def _schedule(self, task, value):
        task._send_value = value
        self._ready.append(task)

    def _sleep(self, task, delay):
        if delay <= 0:
            self._schedule(task, None)
        else:
            self._timers.append((self._time + delay, task))

    def _wake_next_timers(self):
        deadline = self._timers[0][0]
        for timer_deadline, task in self._timers:
            if timer_deadline < deadline:
                deadline = timer_deadline
        self._time = deadline
        pending = []
        for timer_deadline, task in self._timers:
            if timer_deadline <= deadline:
                self._schedule(task, None)
            else:
                pending.append((timer_deadline, task))
        self._timers = pending

    def _finish(self, task, result):
        task._done = True
        task._result = result
        waiters = task._waiters
        task._waiters = []
        for waiter in waiters:
            self._schedule(waiter, result)

    def _block_on_task(self, task, awaited):
        if awaited._done:
            self._schedule(task, awaited.result())
        else:
            awaited._waiters.append(task)

    def _dispatch_yield(self, task, awaited):
        if isinstance(awaited, Task):
            self._block_on_task(task, awaited)
        elif isinstance(awaited, _Sleep):
            self._sleep(task, awaited.delay)
        elif isinstance(awaited, _EventWait):
            if awaited.event.is_set():
                self._schedule(task, True)
            else:
                awaited.event._waiters.append(task)
        elif isinstance(awaited, _QueueGet):
            if awaited.queue._items:
                self._schedule(task, awaited.queue._items.pop(0))
            else:
                awaited.queue._getters.append(task)
        elif isinstance(awaited, _LockWait):
            awaited.lock._waiters.append(task)
        elif _is_coroutine(awaited):
            self._block_on_task(task, self.create_task(awaited))
        else:
            raise TypeError("object cannot be used in 'await' expression")

    def run(self, coroutine):
        root = self.create_task(coroutine)
        while not root.done():
            if not self._ready:
                if self._timers:
                    self._wake_next_timers()
                else:
                    raise RuntimeError("asyncio deadlock: no runnable tasks")
            task = self._ready.pop(0)
            done, value = _step(task._coroutine, task._send_value)
            task._send_value = None
            if done:
                self._finish(task, value)
            else:
                self._dispatch_yield(task, value)
        return root.result()


def get_running_loop():
    if _running_loop is None:
        raise RuntimeError("no running event loop")
    return _running_loop


def create_task(coroutine):
    return get_running_loop().create_task(coroutine)


async def sleep(delay, result=None):
    await _Sleep(delay)
    return result


async def gather(*coroutines, return_exceptions=False):
    tasks = []
    for coroutine in coroutines:
        if isinstance(coroutine, Task):
            tasks.append(coroutine)
        else:
            tasks.append(create_task(coroutine))
    results = []
    for task in tasks:
        try:
            results.append(await task)
        except Exception as error:
            if not return_exceptions:
                raise
            results.append(error)
    return results


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
