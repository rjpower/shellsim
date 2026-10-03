"""Iteration helpers the VM runs as bytecode instead of native loops.

A native builtin such as ``list(gen)`` must not drive a generator or ``__next__`` method from a
Rust loop: items it accumulated would live in host memory the heap cannot see. Running the loop
here makes every item an ordinary instruction whose result lands in a metered list.
"""


def materialize(iterable):
    return [item for item in iterable]


def aiter(iterable):
    """The async iterator of `iterable`, from its type's `__aiter__`."""
    # Async generators expose the protocol on the object itself, as `async for` relies on.
    try:
        method = iterable.__aiter__
    except AttributeError:
        raise TypeError(
            "'" + type(iterable).__name__ + "' object is not an async iterable"
        ) from None
    return method()


async def _anext_or_default(awaitable, default):
    try:
        return await awaitable
    except StopAsyncIteration:
        return default


def anext(iterator, *default):
    """An awaitable for the next item of an async iterator, or `default` when it is exhausted."""
    if len(default) > 1:
        raise TypeError("anext expected at most 2 arguments, got " + str(len(default) + 1))
    try:
        method = iterator.__anext__
    except AttributeError:
        raise TypeError(
            "'" + type(iterator).__name__ + "' object is not an async iterator"
        ) from None
    awaitable = method()
    if not default:
        return awaitable
    return _anext_or_default(awaitable, default[0])
