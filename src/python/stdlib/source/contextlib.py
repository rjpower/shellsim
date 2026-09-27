"""Context-manager utilities: `contextmanager`, `suppress`, `nullcontext`, `closing`, and
`ExitStack`.

The behavior follows the documented contracts of CPython's `contextlib`. The asynchronous
variants, `redirect_stdout`, `chdir`, and using a `contextmanager` object as a decorator are not
provided. A function decorated with `contextmanager` keeps the wrapper's own `__name__`, because
shellsim functions do not accept attribute assignment.
"""


class _GeneratorContextManager:
    """The context manager a `contextmanager` function returns for one call."""

    def __init__(self, func, args, kwds):
        self._generator = func(*args, **kwds)

    def __enter__(self):
        try:
            return next(self._generator)
        except StopIteration:
            raise RuntimeError("generator didn't yield") from None

    def __exit__(self, kind, value, traceback):
        if kind is None:
            try:
                next(self._generator)
            except StopIteration:
                return False
            raise RuntimeError("generator didn't stop")
        if value is None:
            value = kind()
        try:
            self._generator.throw(value)
        except StopIteration as stop:
            # The generator handled the exception and returned, which suppresses it.
            return stop is not value
        except BaseException as error:
            # Re-raising the same exception lets the `with` statement propagate the original.
            if error is value:
                return False
            raise
        raise RuntimeError("generator didn't stop after throw()")


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

    def helper(*args, **kwds):
        return _GeneratorContextManager(func, args, kwds)

    return helper


class suppress:
    """Suppress the listed exception types, and their subclasses, raised in the `with` body."""

    def __init__(self, *exceptions):
        self._exceptions = exceptions

    def __enter__(self):
        pass

    def __exit__(self, kind, value, traceback):
        return kind is not None and issubclass(kind, self._exceptions)


class nullcontext:
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


class closing:
    """Return `thing` from `__enter__` and call `thing.close()` on exit."""

    def __init__(self, thing):
        self.thing = thing

    def __enter__(self):
        return self.thing

    def __exit__(self, *excinfo):
        self.thing.close()


class ExitStack:
    """Combine a dynamic number of context managers and cleanup callbacks.

    Callbacks run in reverse order of registration when the stack exits. Each receives the
    exception still pending at that point; a true result suppresses it, and an exception raised
    by a callback replaces it for the callbacks that remain.
    """

    def __init__(self):
        self._callbacks = []

    def __enter__(self):
        return self

    def enter_context(self, cm):
        """Enter `cm` and register its `__exit__`; return what `__enter__` returned."""
        kind = type(cm)
        enter = getattr(kind, "__enter__", None)
        exit = getattr(kind, "__exit__", None)
        if enter is None or exit is None:
            raise TypeError(f"'{kind.__name__}' object does not support the context manager protocol")
        result = enter(cm)
        self._callbacks.append(lambda kind, value, traceback: exit(cm, kind, value, traceback))
        return result

    def push(self, exit):
        """Register a context manager's `__exit__`, or a callable with the same signature."""
        method = getattr(type(exit), "__exit__", None)
        if method is None:
            self._callbacks.append(exit)
        else:
            self._callbacks.append(lambda kind, value, traceback: method(exit, kind, value, traceback))
        return exit

    def callback(self, callback, /, *args, **kwds):
        """Register `callback(*args, **kwds)` to run on exit; it cannot suppress exceptions."""

        def call(kind, value, traceback):
            callback(*args, **kwds)

        self._callbacks.append(call)
        return callback

    def pop_all(self):
        """Move every registered callback to a new `ExitStack` and return it."""
        stack = ExitStack()
        stack._callbacks = self._callbacks
        self._callbacks = []
        return stack

    def close(self):
        """Run every registered callback now."""
        self.__exit__(None, None, None)

    def __exit__(self, kind, value, traceback):
        pending = value
        while self._callbacks:
            callback = self._callbacks.pop()
            try:
                if callback(None if pending is None else type(pending), pending, None):
                    pending = None
            except BaseException as error:
                pending = error
        if pending is not None and pending is not value:
            raise pending
        return value is not None and pending is None
