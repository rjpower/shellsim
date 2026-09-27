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
`with` regions. `throw` raises the exception at the suspended `yield`, so the generator's
`except`, `finally`, and `with` blocks handle it, and `close` raises `GeneratorExit` there. `yield
from` passes `send`, `throw`, and `close` through to the subiterator's methods of those names and
evaluates to the value the subiterator returns. A builtin iterator has none of them: sending it a
value other than `None` raises `AttributeError`, an exception thrown at it is raised in the
delegating generator, and the expression evaluates to `None`.

`for` loops, `yield from`, `iter`, `next`, and `itertools.islice` advance an iterator returned by
a class's `__iter__` one `__next__` call at a time, so infinite iterators and side effects in
`__next__` behave as in CPython. `map`, `filter`, `zip`, `enumerate`, and `itertools.chain`
still consume their whole input before returning.

`complex` is a native arena type. Its arithmetic, string parsing, `repr`, and error messages
follow CPython 3.14, including the mixed-mode rules for real operands. Ordering, floor division,
modulo, `int()`, `float()`, `round()`, and `math` functions reject complex values with
`TypeError`. Complex format specifications follow CPython: the width applies to the whole
number, and zero padding, `=` alignment and `%` are rejected.

The core collection surface includes mutable sets and immutable `frozenset` values with mixed
comparison and set algebra. VFS-backed text and binary files support read, write, append, and
their `+` update variants with a shared seekable cursor.

Function calls support positional-only (`/`), positional, variadic, keyword-only, and
keyword-variadic parameters, including bounded `*iterable` and `**mapping` expansion. Duplicate
keywords, non-string mapping keys, and non-mapping `**` operands are rejected explicitly. List,
tuple, and set displays expand `*iterable` items in place.

The `...` literal evaluates to the `Ellipsis` singleton, so `obj[..., 0]` passes the tuple
`(Ellipsis, 0)` to `__getitem__`. Binary-operator and rich-comparison methods may return
`NotImplemented` to decline. The runtime then tries the reflected method and then the default:
identity for `==` and `!=`, and `TypeError` for ordering and arithmetic. As in CPython 3.14,
testing `NotImplemented` for truth raises `TypeError`. Lists, tuples, dictionaries, and sets compare
elements, find keys, and answer `in` with `x is y or x == y`, so a class's `__eq__` holds inside
containers and an equal key of another type, such as `0.5` for `Fraction(1, 2)`, finds a dict entry.
`object` provides `__hash__`, `__eq__`, and `__ne__`, so a class that defines `__eq__` can keep
identity hashing with `__hash__ = object.__hash__`.

Classes record their defining module as `__module__` and report `__bases__`, `__mro__`, and
`mro()`, including native bases such as `int` and the exception hierarchy; `repr` qualifies the
class name with its module. Class attributes, special methods included, may be assigned or deleted
after the class statement, and the change applies to existing instances and subclasses. `del
obj.attr` and `delattr` delete through the descriptor protocol: a class's `__delattr__`, a
descriptor's `__delete__`, then the instance's own attribute. Classes and functions have no
`__qualname__` or `__doc__`, so a nested class's `repr` omits its enclosing scope. Builtin
functions, native methods and bound methods report `__name__`, and bound methods expose
`__self__` and `__func__`.

A class may subclass `int` or `tuple`. Its instances carry the builtin value, so arithmetic,
comparison, hashing, indexing, iteration, `len`, `repr` and `json.dumps` act on that value unless
the class overrides them. The subclass inherits the native methods such as `bit_length`,
`to_bytes`, `count` and `index`, and `from_bytes` called on the subclass returns an instance of
it. `int.__new__(cls, ...)` and `tuple.__new__(cls, ...)` build a subclass instance from a user
`__new__`. Subclassing any other builtin type, such as `list`, `dict` or `str`, fails with exit
status 2 rather than producing an instance that lacks the builtin behavior.

Calling a class whose MRO defines `__new__` calls it with the class and the arguments, then
calls `__init__` only if the result is an instance of the class. `object.__new__` creates a plain
instance and, like CPython's, rejects arguments that neither an overriding `__new__` nor an
`__init__` would accept. Attribute lookup on a class continues through its builtin ancestors, so
`C.__init__ is object.__init__` for a class without its own, and `Exception.__init__(self, message)`
initializes a user exception. User `__new__` methods of exception classes and dataclasses are not
called yet.

Augmented assignment updates mutable operands in place, as in CPython: `list +=` extends with any
iterable, `set |=` and its siblings mutate the set, `dict |=` updates the mapping, and user classes
may define `__iadd__` and the other in-place methods. Other operands fall back to the binary
operator.

Text formatting uses one protocol for f-strings, `format()` and `str.format`, including conversions,
alignment, width and precision, and decimal, binary, octal, and hexadecimal integer formats.
It also handles signed fixed-point output, general floating-point precision, and decimal comma
grouping. `from module import *` binds the names in a list or tuple `__all__`, or else the
names without a leading underscore, and only at module level. The `__import__` builtin uses the
simulated loader for absolute imports; relative
`__import__` calls are explicitly unsupported. The `exec`
builtin accepts one source string and executes it in the simulated namespace. Code objects and
explicit globals or locals mappings are not supported.
The frozen `functools` module provides `reduce` and positional and keyword argument binding with
`partial`. The frozen `contextlib` module provides `contextmanager`, `suppress`, `nullcontext`,
`closing`, and `ExitStack`. The frozen `inspect` module provides `signature`, `Signature`, and
`Parameter` for Python functions, bound methods, classes with a Python `__init__`, and instances
with a Python `__call__`; builtin callables have no signature and raise `ValueError`. The frozen `operator` module provides CPython's operator functions, `itemgetter`,
`attrgetter`, and `methodcaller`.

`math` includes `isclose`, the hyperbolic functions and their inverses, and `gamma` and
`lgamma`. The hyperbolic functions use Rust's standard library and the gamma functions the `libm`
crate, so their last bit can differ from CPython's. Poles and out-of-domain arguments raise
`ValueError`, and finite arguments whose result overflows raise `OverflowError`. `round()` and
`math.floor`, `ceil`, and `trunc` defer to a class's `__round__`, `__floor__`, `__ceil__`, and
`__trunc__`, and other `math` functions convert instances through `__float__`.

The frozen `fractions` module provides `Fraction`, built from integers, fractions, rational or
decimal strings, and floats or other objects with `as_integer_ratio`. Arithmetic with integers and
fractions is exact and metered like other integer work; mixing in a float or complex number gives a
float or complex result. Fractions hash like equal integers and floats and support rounding,
`limit_denominator`, and `from_float`. `Decimal` operands, the `numbers` ABCs, and format
specifications are not supported.

The frozen `warnings` module implements `warn`, `warn_explicit`, `filterwarnings`, `simplefilter`,
`resetwarnings`, `catch_warnings` with `record=True`, and CPython's `default`, `once`, `module`,
`always`, `ignore`, and `error` actions. A warning names the line executing `stacklevel` frames up.
Every frame reports the script path, as tracebacks do, and `module=` filters match `__main__`.

Exceptions keep their constructor arguments in `args`, which is writable, and derive `str` and
`repr` from them as CPython does: a `KeyError` shows its key's `repr`, and a generator's return
value becomes the `value` of the `StopIteration` that ends it. User-defined exception subclasses
inherit the same behavior. Builtin exception instances do not accept other attributes, and
`OSError`'s `errno`, `strerror`, and `filename`, `with_traceback`, and `add_note` are not
modeled.

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

The NumPy and SciPy implementations follow these rules; see [numpy.md](numpy.md) and
[scipy.md](scipy.md).
