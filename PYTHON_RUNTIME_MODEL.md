# Python runtime model

Status: implemented native-module and initial object-model foundation, 2026-09-11

Shellsim implements a bounded Python subset over one erased value representation. The VM,
builtins, and standard-library modules exchange `PyValue`; an implementation locally casts a
value to a checked typed view such as `PyList`, `PyDict`, or `PyNumber`. A view is a small handle
into the invocation arena, not a second owned representation.

## Values and objects

Immediate scalar values live in `PyValue`. Mutable values and values with identity live in the
arena and are named by `ObjectId`.

```text
PyValue
  None | Bool | Int | Float | String
  Object(ObjectId)

Object
  List | Tuple | Dict | Set
  Function | Class | Instance | Iterator | Generator | Module
  explicitly modeled native payloads
```

`PyKind` describes this storage/runtime shape. It is not Python's eventual `type()` and subclass
model. `PyValue::cast` validates a value and returns a typed view. Collection views store only an
`ObjectId`; their read and mutation methods take a runtime argument so they cannot retain arena
borrows across allocation or Python protocol calls.

The first numeric view is `PyNumber::{Int(i64), Float(f64)}`. All numeric coercion belongs there.
Checked integer overflow is an explicit error. A future `BigInt(ObjectId)` representation can be
added to `PyNumber` without teaching every module about heap layout.

## Types and inheritance

User classes retain their direct bases, a metered C3 method-resolution order, their attribute
dictionary, instance layout, and metaclass. Instance and class attribute lookup follow the same
MRO. Inconsistent or duplicate base graphs are rejected while the class is created.

Ordinary instances use the object layout. An `int` subclass uses an integer payload stored inside
the instance object, preserving its class identity while numeric protocols expose the payload to
`PyNumber`, arithmetic, comparison, truth, representation, and conversion. Numeric operations
deliberately produce base integers. Other builtin layouts can follow the same representation.

`type`, `isinstance`, and `issubclass` use this graph. Classes derived from `type` may serve as
explicit metaclasses, metaclass identity propagates to subclasses, and metaclass attributes
participate in class lookup. Custom metaclass `__new__`, `__init__`, and `__call__` hooks are an
explicit unsupported frontier until descriptor invocation and class construction share one call
path. Builtin constructors currently also act as their builtin type values; a distinct
`BuiltinType` representation is the next cleanup before expanding builtin subclass layouts.

## Native calls

All native functions have one erased ABI:

```text
fn(&mut dyn PyRuntime, CallArgs) -> PyResult<PyValue>
```

`CallArgs` binds positional and keyword arguments and produces Python-shaped argument errors.
`PyResult<T>` is generic over a locally useful success type and always carries a structured
`PyError` on failure. Type erasure occurs at the `PyValue` call boundary; typed results inside an
implementation prevent repeated unchecked matching.

`ModuleDef`, `FunctionDef`, and `ValueDef` describe native modules declaratively. An imported
native module carries its `ModuleDef` directly; adding one registry entry does not require a VM
module enum or a second attribute switch. Modules may use only `PyRuntime` operations for
allocation, metering, and object inspection. Pure modules such as `json` and `math` receive no
filesystem, process, network, environment, or clock capability.

Runtime services are split by purpose. General value protocols such as comparison, display,
iteration, calls, and typed allocation live on `PyRuntime`. Modeled ambient state is available
only through explicit narrow traits: `time` receives `PyClock`, while `os` receives the read-only
`PyEnvironment`. Both are implemented by the simulated interpreter and cannot reach host state.

Native errors retain a `PyErrorKind` until the VM call boundary. The VM turns ordinary kinds into
catchable Python exceptions, preserves explicit pytest control exceptions, propagates callable
exit status, and leaves resource failures attached to the shared resource stop reason.

Native object methods use the erased ABI
`fn(&mut dyn PyRuntime, PyValue, CallArgs) -> PyResult<PyValue>`. `NativeTypeDef` owns a table of
`MethodDef`s, and attribute lookup creates a bound native method containing the receiver and
descriptor. `re.Pattern` and `re.Match` are the first converted types; their implementations use
checked `PyRegex` and `PyMatch` views plus metered payload snapshots.

## Invariants

- The VM, builtins, and modules exchange only `PyValue`.
- A typed view validates but never duplicates the underlying object.
- Arena-backed views do not retain references into the arena.
- Object mutation and result allocation go through `PyRuntime` and are metered first.
- Modules do not inspect VM stacks or access `Interp`.
- Numeric conversion and promotion go through `PyNumber`.
- Python truth, representation, equality, ordering, hashing, iteration, calls, attributes, and
  indexing have one runtime protocol implementation.
- Unsupported syntax, conversions, module names, and capabilities fail explicitly.
- Parsing and native loops bound depth, allocation, output, and work before performing unbounded
  host operations.

## Current module layout

The declarative registry now owns `argparse`, `bisect`, `collections`, `dataclasses`, `enum`,
`functools`, `heapq`, `itertools`, `json`, `math`, `os`, `pytest`, `re`, `string`, `subprocess`,
`sys`, `time`, `typing`, and `unittest`. This describes the supported shellsim slices of those
modules, not their complete CPython APIs. `subprocess` is deliberately an empty importable
frontier and never receives a process capability.

Collection algorithms use checked `PySequence` or `PyList` views and runtime comparison. Mutating
algorithms operate on metered snapshots and replace the list only after success. Iterator modules
return `PyIterator` handles instead of eagerly materializing potentially infinite input.
`collections.defaultdict` and higher-order functions use checked `PyCallable` values. JSON walks
only erased values and typed collection views, so it can parse and construct ordinary Python
objects without knowing arena layout.

The remaining legacy method boundary covers mutable argument parsers, raises contexts, streams,
and the environment marker. Argument parsers should move next using a checked payload view with a
metered snapshot-and-commit mutation API. User-defined descriptors and zero-argument `super()`
remain future object-model slices. Adding module-specific raw heap inspection to `PyRuntime`
would recreate the coupling this layout removes.
