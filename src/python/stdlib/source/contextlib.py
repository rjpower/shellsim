"""Utilities for ``with`` and ``async with`` statements.

The behavior follows the documented contracts of CPython's ``contextlib``: generator-based
context managers, ``suppress``, ``nullcontext``, ``closing``, the exit stacks, the ``redirect_*``
helpers and ``chdir``. ``redirect_stdout`` rebinds ``sys.stdout`` to the modeled stream object the
caller supplies, and ``chdir`` changes the process's VFS working directory.
"""

import abc
import os
import sys
from functools import wraps

__all__ = [
    "asynccontextmanager", "contextmanager", "closing", "nullcontext", "AbstractContextManager",
    "AbstractAsyncContextManager", "AsyncExitStack", "ContextDecorator", "ExitStack",
    "redirect_stdout", "redirect_stderr", "suppress", "aclosing", "chdir", "AsyncContextDecorator",
]


class AbstractContextManager(abc.ABC):
    """A class with ``__enter__`` and ``__exit__``; ``__class_getitem__`` is a no-op alias."""

    __class_getitem__ = classmethod(lambda cls, item: cls)

    def __enter__(self):
        return self

    @abc.abstractmethod
    def __exit__(self, exc_type, exc_value, traceback):
        return None

    @classmethod
    def __subclasshook__(cls, C):
        if cls is AbstractContextManager:
            return abc._check_methods(C, "__enter__", "__exit__") if hasattr(abc, "_check_methods") else _check_methods(C, "__enter__", "__exit__")
        return NotImplemented


class AbstractAsyncContextManager(abc.ABC):
    """A class with ``__aenter__`` and ``__aexit__``."""

    __class_getitem__ = classmethod(lambda cls, item: cls)

    async def __aenter__(self):
        return self

    @abc.abstractmethod
    async def __aexit__(self, exc_type, exc_value, traceback):
        return None

    @classmethod
    def __subclasshook__(cls, C):
        if cls is AbstractAsyncContextManager:
            return _check_methods(C, "__aenter__", "__aexit__")
        return NotImplemented


def _check_methods(C, *methods):
    mro = getattr(C, "__mro__", (C,))
    for method in methods:
        for base in mro:
            namespace = getattr(base, "__dict__", None)
            if namespace is not None and method in namespace:
                if namespace[method] is None:
                    return NotImplemented
                break
        else:
            return NotImplemented
    return True


class ContextDecorator:
    """A context manager that can also decorate a function, entering around each call."""

    def _recreate_cm(self):
        return self

    def __call__(self, func):
        @wraps(func)
        def inner(*args, **kwds):
            with self._recreate_cm():
                return func(*args, **kwds)

        return inner


class AsyncContextDecorator:
    """An async context manager that can also decorate a coroutine function."""

    def _recreate_cm(self):
        return self

    def __call__(self, func):
        @wraps(func)
        async def inner(*args, **kwds):
            async with self._recreate_cm():
                return await func(*args, **kwds)

        return inner


class _GeneratorContextManagerBase:
    def __init__(self, func, args, kwds):
        self.gen = func(*args, **kwds)
        self.func, self.args, self.kwds = func, args, kwds
        doc = getattr(func, "__doc__", None)
        if doc is None:
            doc = type(self).__doc__
        self.__doc__ = doc

    def _recreate_cm(self):
        return self.__class__(self.func, self.args, self.kwds)


class _GeneratorContextManager(_GeneratorContextManagerBase, AbstractContextManager, ContextDecorator):
    """The context manager a `contextmanager` function returns for one call."""

    def __enter__(self):
        del self.args, self.kwds, self.func
        try:
            return next(self.gen)
        except StopIteration:
            raise RuntimeError("generator didn't yield") from None

    def __exit__(self, typ, value, traceback):
        if typ is None:
            try:
                next(self.gen)
            except StopIteration:
                return False
            raise RuntimeError("generator didn't stop")
        if value is None:
            value = typ()
        try:
            self.gen.throw(value)
        except StopIteration as exc:
            # The generator handled the exception and returned, which suppresses it.
            return exc is not value
        except RuntimeError as exc:
            if exc is value:
                return False
            if isinstance(value, StopIteration) and exc.__cause__ is value:
                return False
            raise
        except BaseException as exc:
            # Re-raising the same exception lets the `with` statement propagate the original.
            if exc is not value:
                raise
            return False
        raise RuntimeError("generator didn't stop after throw()")


class _AsyncGeneratorContextManager(_GeneratorContextManagerBase, AbstractAsyncContextManager,
                                    AsyncContextDecorator):
    """The async context manager an `asynccontextmanager` function returns for one call."""

    async def __aenter__(self):
        del self.args, self.kwds, self.func
        try:
            return await anext(self.gen)
        except StopAsyncIteration:
            raise RuntimeError("generator didn't yield") from None

    async def __aexit__(self, typ, value, traceback):
        if typ is None:
            try:
                await anext(self.gen)
            except StopAsyncIteration:
                return False
            raise RuntimeError("generator didn't stop")
        if value is None:
            value = typ()
        try:
            await self.gen.athrow(value)
        except StopAsyncIteration as exc:
            return exc is not value
        except RuntimeError as exc:
            if exc is value:
                return False
            if isinstance(value, (StopIteration, StopAsyncIteration)) and exc.__cause__ is value:
                return False
            raise
        except BaseException as exc:
            if exc is not value:
                raise
            return False
        raise RuntimeError("generator didn't stop after athrow()")


def contextmanager(func):
    """Turn a generator function that yields once into a context-manager factory.

    ```python
    @contextmanager
    def tag(name):
        print(f"<{name}>")
        yield name
        print(f"</{name}>")
    ```
    """

    @wraps(func)
    def helper(*args, **kwds):
        return _GeneratorContextManager(func, args, kwds)

    return helper


def asynccontextmanager(func):
    """Turn an async generator function that yields once into an async context-manager factory."""

    @wraps(func)
    def helper(*args, **kwds):
        return _AsyncGeneratorContextManager(func, args, kwds)

    return helper


class closing(AbstractContextManager):
    """Return `thing` from `__enter__` and call `thing.close()` on exit."""

    def __init__(self, thing):
        self.thing = thing

    def __enter__(self):
        return self.thing

    def __exit__(self, *exc_info):
        self.thing.close()


class aclosing(AbstractAsyncContextManager):
    """Return `thing` from `__aenter__` and await `thing.aclose()` on exit."""

    def __init__(self, thing):
        self.thing = thing

    async def __aenter__(self):
        return self.thing

    async def __aexit__(self, *exc_info):
        await self.thing.aclose()


class _RedirectStream(AbstractContextManager):
    _stream = None

    def __init__(self, new_target):
        self._new_target = new_target
        self._old_targets = []

    def __enter__(self):
        self._old_targets.append(getattr(sys, self._stream))
        setattr(sys, self._stream, self._new_target)
        return self._new_target

    def __exit__(self, exctype, excinst, exctb):
        setattr(sys, self._stream, self._old_targets.pop())


class redirect_stdout(_RedirectStream):
    """Rebind ``sys.stdout`` to ``new_target`` inside the block."""

    _stream = "stdout"


class redirect_stderr(_RedirectStream):
    """Rebind ``sys.stderr`` to ``new_target`` inside the block."""

    _stream = "stderr"


class suppress(AbstractContextManager):
    """Suppress the listed exception types, and their subclasses, raised in the `with` body."""

    def __init__(self, *exceptions):
        self._exceptions = exceptions

    def __enter__(self):
        pass

    def __exit__(self, exctype, excinst, exctb):
        if exctype is None:
            return False
        if issubclass(exctype, self._exceptions):
            return True
        if issubclass(exctype, BaseExceptionGroup):
            match, rest = excinst.split(self._exceptions)
            if rest is None:
                return True
            raise rest
        return False


class nullcontext(AbstractContextManager, AbstractAsyncContextManager):
    """A context manager that does nothing and returns `enter_result` from `__enter__`."""

    def __init__(self, enter_result=None):
        self.enter_result = enter_result

    def __enter__(self):
        return self.enter_result

    def __exit__(self, *excinfo):
        pass

    async def __aenter__(self):
        return self.enter_result

    async def __aexit__(self, *excinfo):
        pass


class _BaseExitStack:
    """Callbacks run in reverse order of registration when the stack exits. Each receives the
    exception still pending at that point; a true result suppresses it, and an exception raised
    by a callback replaces it for the callbacks that remain."""

    @staticmethod
    def _create_exit_wrapper(cm, cm_exit):
        return lambda exc_type, exc, tb: cm_exit(cm, exc_type, exc, tb)

    @staticmethod
    def _create_cb_wrapper(callback, /, *args, **kwds):
        def _exit_wrapper(exc_type, exc, tb):
            callback(*args, **kwds)

        return _exit_wrapper

    def __init__(self):
        self._exit_callbacks = []

    def pop_all(self):
        """Move every registered callback to a new stack of the same type and return it."""
        new_stack = type(self)()
        new_stack._exit_callbacks = self._exit_callbacks
        self._exit_callbacks = []
        return new_stack

    def push(self, exit):
        """Register a context manager's `__exit__`, or a callable with the same signature."""
        _cb_type = type(exit)
        try:
            exit_method = _cb_type.__exit__
        except AttributeError:
            self._push_exit_callback(exit)
        else:
            self._push_cm_exit(exit, exit_method)
        return exit

    def enter_context(self, cm):
        """Enter `cm` and register its `__exit__`; return what `__enter__` returned."""
        cls = type(cm)
        try:
            _enter = cls.__enter__
            _exit = cls.__exit__
        except AttributeError:
            raise TypeError(
                "'%s' object does not support the context manager protocol" % cls.__name__
            ) from None
        result = _enter(cm)
        self._push_cm_exit(cm, _exit)
        return result

    def callback(self, callback, /, *args, **kwds):
        """Register `callback(*args, **kwds)` to run on exit; it cannot suppress exceptions."""
        _exit_wrapper = self._create_cb_wrapper(callback, *args, **kwds)
        _exit_wrapper.__wrapped__ = callback
        self._push_exit_callback(_exit_wrapper)
        return callback

    def _push_cm_exit(self, cm, cm_exit):
        _exit_wrapper = self._create_exit_wrapper(cm, cm_exit)
        self._push_exit_callback(_exit_wrapper, True)

    def _push_exit_callback(self, callback, is_sync=True):
        self._exit_callbacks.append((is_sync, callback))


class ExitStack(_BaseExitStack, AbstractContextManager):
    """Combine a dynamic number of context managers and cleanup callbacks."""

    def __enter__(self):
        return self

    def __exit__(self, *exc_details):
        received_exc = exc_details[0] is not None
        pending = exc_details[1]
        suppressed_exc = False
        pending_raise = False
        while self._exit_callbacks:
            is_sync, cb = self._exit_callbacks.pop()
            try:
                if cb(None if pending is None else type(pending), pending, None):
                    suppressed_exc = True
                    pending_raise = False
                    pending = None
            except BaseException as new_exc:
                pending_raise = True
                pending = new_exc
        if pending_raise:
            raise pending
        return received_exc and suppressed_exc

    def close(self):
        """Run every registered callback now."""
        self.__exit__(None, None, None)


class AsyncExitStack(_BaseExitStack, AbstractAsyncContextManager):
    """An ``ExitStack`` that also accepts async context managers and coroutine callbacks."""

    @staticmethod
    def _create_async_exit_wrapper(cm, cm_exit):
        return lambda exc_type, exc, tb: cm_exit(cm, exc_type, exc, tb)

    @staticmethod
    def _create_async_cb_wrapper(callback, /, *args, **kwds):
        async def _exit_wrapper(exc_type, exc, tb):
            await callback(*args, **kwds)

        return _exit_wrapper

    async def enter_async_context(self, cm):
        cls = type(cm)
        try:
            _enter = cls.__aenter__
            _exit = cls.__aexit__
        except AttributeError:
            raise TypeError(
                "'%s' object does not support the asynchronous context manager protocol"
                % cls.__name__
            ) from None
        result = await _enter(cm)
        self._push_async_cm_exit(cm, _exit)
        return result

    def push_async_exit(self, exit):
        _cb_type = type(exit)
        try:
            exit_method = _cb_type.__aexit__
        except AttributeError:
            self._push_exit_callback(exit, False)
        else:
            self._push_async_cm_exit(exit, exit_method)
        return exit

    def push_async_callback(self, callback, /, *args, **kwds):
        _exit_wrapper = self._create_async_cb_wrapper(callback, *args, **kwds)
        _exit_wrapper.__wrapped__ = callback
        self._push_exit_callback(_exit_wrapper, False)
        return callback

    async def aclose(self):
        await self.__aexit__(None, None, None)

    def _push_async_cm_exit(self, cm, cm_exit):
        _exit_wrapper = self._create_async_exit_wrapper(cm, cm_exit)
        self._push_exit_callback(_exit_wrapper, False)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_details):
        received_exc = exc_details[0] is not None
        pending = exc_details[1]
        suppressed_exc = False
        pending_raise = False
        while self._exit_callbacks:
            is_sync, cb = self._exit_callbacks.pop()
            try:
                exc_type = None if pending is None else type(pending)
                if is_sync:
                    cb_suppress = cb(exc_type, pending, None)
                else:
                    cb_suppress = await cb(exc_type, pending, None)
                if cb_suppress:
                    suppressed_exc = True
                    pending_raise = False
                    pending = None
            except BaseException as new_exc:
                pending_raise = True
                pending = new_exc
        if pending_raise:
            raise pending
        return received_exc and suppressed_exc


class chdir(AbstractContextManager):
    """Change the working directory inside the block and restore it afterwards."""

    def __init__(self, path):
        self.path = path
        self._old_cwd = []

    def __enter__(self):
        self._old_cwd.append(os.getcwd())
        os.chdir(self.path)

    def __exit__(self, *excinfo):
        os.chdir(self._old_cwd.pop())
