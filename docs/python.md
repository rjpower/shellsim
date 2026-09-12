# Python in shellsim

Shellsim implements a deterministic, resource-bounded Python 3.14 subset in Rust. Its purpose is
to run ordinary Python embedded in agent tasks without granting access to host Python, native
extensions, the host filesystem, processes, network, environment, locale, or clock. Compatibility
is defined by accepted source behavior and differential tests. Unsupported syntax and APIs fail
explicitly.

The interpreter is intentionally not CPython-compatible at the ABI or bytecode level. It consumes
Python source and uses shellsim-owned data structures throughout.

## Execution pipeline

`python`, `python3`, and `python3.14` all enter `src/python/mod.rs`. The supported entrypoints are
`-c`, stdin, VFS script files and shebangs, the persistent shell-owned REPL, and the bounded
`pytest` and `unittest` runners.

For host-side experiments, `shellsim-python PATH` creates a fresh environment, imports a file's
containing project into `/work`, and runs the file. A directory discovers `test_*.py` files by
default; `--entry`, `--pytest`, `--root`, resource-limit flags, and `--json` select other modes.
Host ingestion rejects symlinks and is complete before the interpreter starts, so it does not give
simulated code ambient filesystem access.

```text
source -> lexer -> AST parser -> semantic bytecode compiler -> metered stack VM
                                                            |
                                                            +-> native modules
                                                            +-> modeled capability traits
```

The lexer and parser are UTF-8 and indentation aware. The compiler emits a typed internal
instruction enum rather than CPython opcodes. The VM charges CPU per instruction and bounds
source size, nesting, calls, allocation, iteration, and output. Product code never delegates
unsupported input to a host interpreter.

Ordinary invocations receive fresh Python globals, modules, and heap state. Intended machine
effects, such as VFS writes and resource consumption, persist in the surrounding `Environment`.
The REPL is the explicit exception and retains its `ReplState` between shell actions.

## Values and identity

Every Python value crosses runtime and native-module boundaries as a 16-byte `PyValue`:

```text
PyValue { payload: u64, aux: [u8; 7], tag: ValueTag }

ValueTag
  None | Bool | Int | Float | SmallString
  Object(ObjectId) | Native(closed interpreter handle)
```

`None`, booleans, bounded integers, floats, and UTF-8 strings up to fifteen bytes are immediate.
Long strings, immutable bytes, mutable byte arrays, arbitrary-precision integers, mutable values,
exceptions, classes, and other values requiring distinct identity live in the invocation arena.
Each arena entry has a semantic `TypeId`, optional attributes, and a typed payload. Byte sequences
are never routed through UTF-8 storage: `PyBytes` exposes an owned, checked octet snapshot and
`PyByteArray` provides snapshot-and-commit mutation.

Physical tags do not define Python types. `type_id(value)` maps immediate storage to canonical
builtin types and arena values to the type in their object header. `PyKind` exists only to obtain a
checked native view such as `PyNumber`, `PyString`, `PyBytes`, `PyList`, or `PyDict`.

There is no separate identity field. Arena values use `ObjectId`; immediate values are canonical
by representation, with floats compared by exact bits for identity. Container equality and
membership check identity before equality, preserving reflexive behavior for a stored NaN.

## Types, descriptors, and operators

The runtime bootstraps canonical `object`, `type`, and builtin type objects. User types carry direct
bases, a metered C3 MRO, a metaclass, attributes, layout, and cached protocol slots. Supported
class construction includes metaclass selection, `__prepare__`, `type.__new__`, `__set_name__`,
`__init_subclass__`, and metaclass `__init__` and `__call__`.

Attribute lookup follows Python's descriptor order:

1. data descriptors through the MRO;
2. the instance dictionary;
3. ordinary attributes and non-data descriptors through the MRO;
4. descriptor binding through `__get__`;
5. a missing-attribute result.

Python functions, native methods, `property`, `staticmethod`, and `classmethod` use this path.
Zero-argument `super()` uses the function's captured defining class and the receiver's C3 MRO.
`int` subclasses have an integer instance layout while preserving their user-defined type.

Important dunder methods populate cached slots for calls, construction, attributes, display,
truth, iteration, arithmetic, comparison, and containment. Bytecode arithmetic asks the operand
types for the appropriate slot. Numeric slots cast both erased operands to `PyNumber`, covering
immediate integers, heap big integers, floats, booleans, and `int` subclass payloads without VM
tag-specific arithmetic branches. This includes reflected operations, floor division, remainder,
bitwise operations, and unary positive, negative, and invert. `sum()` uses the same addition
dispatch instead of a private numeric fast path.

## Native Python APIs

Native functions and methods use one erased ABI:

```rust
fn(&mut dyn PyRuntime, CallArgs) -> PyResult<PyValue>
fn(&mut dyn PyRuntime, PyValue, CallArgs) -> PyResult<PyValue>
```

Implementations immediately cast values to checked views and return structured `PyError` values.
Mutable views take metered snapshots and commit only after an operation succeeds. They never hold
arena borrows across allocation or Python calls.

`ModuleDef`, `FunctionDef`, `ValueDef`, `NativeTypeDef`, and `MethodDef` provide declarative module
and type tables. Most modules receive only `PyRuntime`. Narrow traits expose modeled state where
required: `time` receives the virtual clock, `os` receives the simulated environment, and the
private subprocess core receives a logical-process runner. A module
must not inspect VM stacks, heap payload variants, or `Environment` directly.

Filesystem access is split into two explicit boundaries in `python/filesystem.rs`. `PyFilesystem`
provides separate bounded text and byte I/O, predicates, directory mutation, renaming, and globbing
to the private frozen-module facade. `PyModuleLoader` performs VFS-only source discovery for
imports. Both own path policy, resource charging, quota translation, and mutation-time
synchronization; `vm.rs` contains no VFS operations and neither boundary can reach the host
filesystem.

The registry contains bounded native slices of modules such as `argparse`, `bisect`, `dataclasses`,
`enum`, `functools`, `heapq`, `itertools`, `math`, `pytest`, `re`, `string`, `subprocess`, `sys`,
`time`, `typing`, and `unittest`. A separate closed frozen-source registry bundles Python
implementations with `include_str!` and executes them through the ordinary compiler and module
namespace. `abc`, `base64`, `codecs`, `collections`, `csv`, `datetime`, `glob`, `hashlib`, `io`,
`json`, `logging`, `os`, `pathlib`, `struct`, `subprocess`, `tempfile`, `uuid`, and `zlib` use this
path. Source modules may import small private native
cores for algorithms or modeled capabilities that Python cannot implement directly. Filesystem
modules share `_shellsim_vfs`; `collections` imports only native `defaultdict`; `json`, `hashlib`,
and `zlib` delegate their bounded codec, digest, or checksum primitives. `subprocess` delegates
only direct argv execution to `_shellsim_subprocess`: `run`, `call`, `check_call`, and
`check_output` execute registered commands and VFS scripts as synchronous logical children.
Captured and inherited streams, bytes/text input and output, cwd, replacement environments,
return codes, checks, and virtual-clock timeouts are modeled. `shell=True` invokes shellsim's own
shell and cannot select a host shell. Live `Popen` children remain unsupported until the process
scheduler and descriptor table can represent pipes and overlapping execution.

Prefer frozen Python for module policy, composition, and ordinary object behavior. Add a private
native primitive only for a modeled capability, an algorithm that must meter host allocation
before it occurs, or representation-level behavior such as byte codecs. This keeps module APIs
expressed in the same object protocols as user programs and keeps the VM independent of stdlib
names.

## Supported behavior and frontiers

The useful current language surface includes containers, arbitrary-precision integer arithmetic,
control flow, functions, closures, defaults and `*args`, comprehensions, classes, inheritance,
descriptors, constrained metaclasses, exceptions, context managers, decorators, f-strings,
suspended generators, VFS imports, and common builtins. Basic string/list/dict/set APIs and
`map`, `filter`, `reversed`, `getattr`, and `hasattr` use native descriptors or erased runtime
protocols.

Iterators are deliberately bounded. Some APIs that are lazy in CPython materialize a metered
snapshot before returning an iterator. Generator expressions are currently eager, and generators
cannot suspend across cleanup regions. These choices are deterministic and fail on resource
limits, but their side-effect timing can differ from CPython.

Other explicit frontiers include:

- async functions, async iterators, async fixtures, and structural pattern matching;
- generator `send`, `throw`, `close`, and `yield from`;
- custom exception subclasses and complete attribute interception;
- the complete buffer protocol, multidimensional slicing, attribute deletion, hashing slots, and
  dict-view semantics;
- `exec`, `eval`, `compile`, code objects, pickle, weak references, and garbage collection;
- native extensions, arbitrary import hooks, host-backed modules, and full pytest/unittest.

The checked stdlib probe set covers 19 named APIs, not entire modules. The TaskTrove mini corpus
currently supports 99 of 100 checked cases, with async behavior as the recorded frontier. A
portable CPython-basic sample passes 61 of 64 isolated behaviors; the remaining probes cover the
three deeper object/generator items named above. Raw CPython `Lib/test` files are not a useful
file-level target because they depend heavily on CPython's private test harness and internals.

## Extending Python support

For a native function or method:

1. Add a `FunctionDef` or `MethodDef` in the relevant `src/python/stdlib` module.
2. Accept and return only erased `PyValue` values at the boundary.
3. Cast locally to checked views and call `PyRuntime` protocols for comparison, iteration,
   attributes, calls, allocation, and mutation.
4. Add the narrowest runtime operation only when several implementations need a semantic protocol;
   do not expose heap representations.
5. Charge work and reserve worst-case growth before host allocation or mutation.
6. Return a structured Python error for invalid input and a resource error for exhaustion.
7. Add a Python-source compatibility test that also runs on CPython, plus Rust tests for resource
   stops or simulator-only state.

For language behavior, keep parsing, compilation, and execution separate. Add syntax tests at the
parser/compiler layer, semantic behavior through source fixtures, and an explicit rejection test
for the unsupported edge. Any filesystem, process, clock, environment, or network requirement
must be implemented against a narrow modeled capability before Python code can observe it.
