"""Iteration helpers the VM runs as bytecode instead of native loops.

A native builtin such as ``list(gen)`` must not drive a generator or ``__next__`` method from a
Rust loop: items it accumulated would live in host memory the heap cannot see. Running the loop
here makes every item an ordinary instruction whose result lands in a metered list.
"""


def materialize(iterable):
    return [item for item in iterable]
