# Python in shellsim

Shellsim implements Python source directly in its deterministic, resource-bounded environment.
The goal is to run ordinary Python used in agent tasks without granting access to host Python,
native extensions, files, processes, networking, environment variables, locale, or time.

The language runtime is intended to be mostly complete. Library compatibility follows a harder
boundary: a module is included when shellsim can provide a coherent, useful implementation of its
ordinary behavior. Missing modules and APIs fail at import or attribute lookup instead of exposing
plausible but inconsistent stubs.

## Run Python

The `python`, `python3`, and `python3.14` commands run source from `-c`, stdin, or a VFS file:

```sh
shellsim -c 'python3 -c "print(sum(x*x for x in range(5)))"'
shellsim --root ./project -c 'python3.14 /work/main.py'
shellsim-python ./project/main.py -- arg1
shellsim-python ./project/tests --pytest
```

`shellsim-python` imports the selected trusted project into `/work` before execution. It supports
`--entry`, `--pytest`, `--root`, resource limits, and `--json`. Simulated imports remain confined
to frozen modules and the virtual filesystem.

Virtual-filesystem packages execute their `__init__.py` before child modules, expose stable
`__name__`, `__package__`, and `__file__` values, cache module identity, and resolve leading-dot
imports within the current package. Imports beyond the top-level package fail explicitly.

The runtime supports functions and closures, classes and descriptors, exceptions and context
managers, comprehensions and lazy generators, arbitrary-precision integers, mutable containers,
f-strings, VFS imports, and the common language protocols needed by real scripts. Generators
support `yield`, `yield from`, `send`, `throw`, and `close`, including suspension in `try` and
`with` regions. Generator shutdown uses a deliberately simple bounded drain through pending
cleanup code.

`complex` is a native arena type. Its arithmetic, string parsing, `repr`, and error messages
follow CPython 3.14, including the mixed-mode rules for real operands. Ordering, floor division,
modulo, `int()`, `float()`, `round()`, and `math` functions reject complex values with
`TypeError`. Format specifications on complex values are not implemented. NumPy `complex128`
arrays store these values directly, and `numpy.complex128` is the builtin `complex` type.

The core collection surface includes mutable sets and immutable `frozenset` values with mixed
comparison and set algebra. VFS-backed text and binary files support read, write, append, and
their `+` update variants with a shared seekable cursor.

Function calls support positional, variadic, keyword-only, and keyword-variadic parameters,
including bounded `*iterable` and `**mapping` expansion. Duplicate keywords, non-string mapping
keys, and non-mapping `**` operands are rejected explicitly. List, tuple, and set displays expand
`*iterable` items in place.

Text formatting uses one protocol for f-strings and `str.format`, including conversions,
alignment, width and precision, and decimal, binary, octal, and hexadecimal integer formats.
It also handles signed fixed-point output, general floating-point precision, and decimal comma
grouping. The `__import__` builtin uses the simulated loader for absolute imports; relative
`__import__` calls are explicitly unsupported. The `exec`
builtin accepts one source string and executes it in the simulated namespace. Code objects and
explicit globals or locals mappings are not supported.
The frozen `functools` module provides `reduce` and positional and keyword argument binding with
`partial`.

User-defined exception subclasses preserve inherited constructor arguments, including compatible
`args`, `str`, and `repr` behavior.

Builtin operations raise ordinary Python exceptions with CPython 3.14's types, hierarchy, and
messages, so `except LookupError` catches a missing dictionary key and an uncaught error exits with
status 1 and a traceback. A missing name raises `NameError`, a module outside the standard library
raises `ModuleNotFoundError`, a failed `from module import name` raises `ImportError`, and a missing
attribute of a module, class, or instance raises `AttributeError`. Argument-binding errors name the
function by `__name__`; CPython uses the qualified name for nested functions and methods.

CPython behavior that shellsim does not model fails differently. An unimplemented builtin such as
`eval`, an unimplemented standard-library module such as `threading`, or a missing method of a
builtin value stops the program with exit status 2 and an "unsupported by minimal shim" diagnostic.
These failures cannot be caught, so an `except ImportError` fallback cannot mistake a shellsim gap
for functionality that is really absent.

`async def`, `await`, `async with`, and `async for` run on a deterministic cooperative scheduler.
The bundled `asyncio` surface includes task creation and introspection, `gather`, `wait`,
`as_completed`, `shield`, futures, deterministic callback and timer scheduling, virtual-time sleeps
and timeout contexts, task groups, events, bounded FIFO, priority, and LIFO queues with work
tracking, locks, semaphores, conditions, and modeled subprocesses. `create_subprocess_exec` and
`create_subprocess_shell` provide byte-oriented stdin, stdout, and stderr streams with cooperative
pipe backpressure, `wait`, `communicate`, signals, cancellation, and virtual-time timeouts. Task
failures cross `await` with their Python exception type, and cancellation or timeout resumes the
coroutine so `finally` cleanup runs. Task groups cancel unfinished siblings on failure and raise
the first child error directly because the runtime does not yet model exception groups. Direct
task cancellation follows nested awaits, while `shield` and `wait` preserve child tasks according
to their asyncio contracts. `asyncio.run` cancels and awaits unfinished tasks, permits their
asynchronous cleanup, shuts down its bounded loop facilities, and closes the loop even when the
main coroutine fails. Suspended coroutines and generators retain their own active exception state,
so cleanup and bare reraises cannot interfere across tasks. The VM parks its logical Python process
on the union of the timers, child states, and pipes awaited by its coroutines, then retries only
inside the simulation when one becomes ready. Native async generators, async sockets,
executors, host threads, text-mode subprocess streams, and custom event loops are not exposed;
synchronous calls made inside a coroutine retain their usual blocking behavior.

The bounded `pytest` runner supports ordinary and yield fixtures, fixture dependencies, literal
`@pytest.mark.parametrize` cases, `tmp_path`/`tmpdir`, skip markers, `pytest.raises` with exception
tuples and `match`, and explicit test files. Fixture scopes and dynamic parametrization remain outside this small runner. `unittest`
supports straightforward test classes. Unsupported syntax and runner features produce an error
rather than a false passing result.

## Runtime model

```text
source -> UTF-8 lexer -> AST parser -> bytecode compiler -> metered stack VM
                                                        -> native modules
                                                        -> modeled capabilities
```

The bytecode is shellsim's internal semantic format, not CPython bytecode. Values are immediate
scalars or typed objects in an interpreter-owned arena. User-visible identity, mutation, type
lookup, descriptors, and operator slots are modeled explicitly. CPU is charged per instruction
and native loop; source, recursion, calls, allocation, iteration, and output are bounded.
Native operations may use conservative constant or linear estimates. Accounting aims for stable
order-of-magnitude costs, not exact host allocation sizes or execution time.

Native modules exchange a type-erased `PyValue` and recover checked views such as `PyNumber`,
`PyString`, `PyList`, or `PyArray`. They receive only the capabilities declared by `PyRuntime`.
A pure algorithm cannot acquire VFS, process, clock, or network access accidentally.

## Adding a module

Use a frozen Python module when the behavior composes naturally from supported Python. Use a Rust
native module for compact algorithms, interpreter-owned objects, or an explicit modeled
capability. Native definitions use declarative function, method, getter, type, and value tables.
A getter is a read-only data descriptor such as `int.real` or `ndarray.shape`: instance lookup
calls it, lookup through the type object returns the descriptor, and assignment raises
`AttributeError`.

For either form:

1. define the useful supported surface and the explicit unsupported frontier;
2. validate arguments and reject unknown keywords;
3. meter loops and reserve result storage before allocation using a simple proportional estimate;
4. keep mutable interpreter layouts behind checked runtime views;
5. add differential tests against the matching CPython behavior where deterministic.

Do not add an importable placeholder for a module whose central contract is absent. For example,
an empty `sqlite3` namespace is less useful than a clear import failure because callers otherwise
cannot tell which database semantics are real.

The minimal NumPy design follows these rules in [numpy.md](numpy.md).
