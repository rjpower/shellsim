# Python runtime model

Status: implemented, 2026-09-11

Shellsim implements a bounded Python subset over one erased value representation. The VM,
builtins, and standard-library modules exchange `PyValue`; an implementation locally casts a
value to a checked typed view such as `PyList`, `PyDict`, or `PyNumber`. A view is a small handle
into the invocation arena, not a second owned representation.

## Values and objects

Every `PyValue` is a 16-byte tagged value. `None`, booleans, bounded integers, floats, and UTF-8
strings of at most fifteen bytes are stored directly. Mutable values, long strings,
arbitrary-precision integers, exceptions, and values with distinct identity live in the arena and
are named by `ObjectId`. Every arena allocation has a common `TypeId` and
attribute-dictionary header.

```text
PyValue { payload: u64, aux: [u8; 7], tag: ValueTag }
ValueTag
  None | Bool | Int | Float | SmallString(0..15 bytes)
  Object(ObjectId) | Native(closed runtime handle)

HeapObject { TypeId, attributes, payload }
Object payload
  String | BigInt | Exception | List | Tuple | Dict | Set
  Function | Class | Instance | Iterator | Generator | Module
  explicitly modeled native payloads
```

`PyKind` describes a checked native view, not semantic type identity. `TypeId` and the runtime
type registry drive `type()`, subclass tests, descriptors, and slots. `PyValue::cast` validates a
value and returns a typed view. Collection views store only an
`ObjectId`; their read and mutation methods take a runtime argument so they cannot retain arena
borrows across allocation or Python protocol calls.

Identity follows the representation boundary. `None` and both booleans are canonical singletons.
Arena objects compare their `ObjectId`, including mutable collections, user instances, classes,
and builtin-subclass instances. Immediate integers and strings are canonical by value, while
floats are canonical by their exact bits. This is equivalent to interning every immediate value:
it preserves reflexive and assignment identity without requiring a pointer or side token. Python
code must still use `==` for scalar value comparison because whether separate computations reuse
an immutable object is implementation-dependent.

Container protocols check identity before equality. A NaN therefore remains unequal to itself but
is identical to the same immediate bit-pattern, so `x in [x]` and `[x] == [x]` behave like CPython.
This differs from Monty's current equality-only fallback for pointer-free floats. If shellsim later
exposes operations that can construct distinct immutable objects with the same representation,
those values will need boxing or an identity token rather than silently weakening assignment
identity.

The numeric view is `PyNumber::{Int(i64), BigInt(String), Float(f64)}`. The decimal bigint snapshot
keeps modules independent of heap layout. Integer literals and checked arithmetic promote to
heap-backed arbitrary precision, and JSON and argparse construct large integers through the same
erased runtime operation. Conversion to `f64` is checked before math and time modules use it.
Builtin numeric types share native `add`, `subtract`, and `multiply` functions stored directly in
their `PyType` slot tables. Those functions cast both erased operands to `PyNumber`; the bytecode
handler only invokes the left slot, then the right reflected slot. String, list, and tuple slots
use their corresponding checked views through the same ABI.

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
participate in class lookup. Metaclass selection checks compatibility across bases.
`__prepare__`, `__new__`, `__init__`, and `__call__` run through the common callable path.
`type.__new__` is the shared class allocator for ordinary class statements, custom metaclasses,
and direct metaclass calls.

Builtin classes use canonical `BuiltinType` values, separate from builtin functions. `type(x)`
therefore returns stable type objects for `None`, scalar values, core collections, functions, and
modules. Calling `int`, `float`, `str`, `bool`, `list`, `tuple`, `dict`, and `set` dispatches through
the same type representation. `object()` instances, arbitrary `dict()` input forms, and builtin
subclass layouts other than `int` remain explicit compatibility frontiers.

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

Native and builtin object methods use the erased ABI
`fn(&mut dyn PyRuntime, PyValue, CallArgs) -> PyResult<PyValue>`. `NativeTypeDef` owns a table of
`MethodDef`s, and attribute lookup creates a bound native method containing the receiver and
descriptor. Regex and match objects, argument parsers, raises contexts, streams, and environment
markers all use registered method descriptors plus checked native views. Core string, list,
dictionary, set, and property methods use the same mechanism; the VM has no method-kind switch.

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

String, collection, property, unittest, argparse, regex, stream, environment, and pytest behavior
all use ordinary native descriptors. User-defined descriptors, standard descriptors,
`__set_name__`, cached protocol slots, captured `__class__`, and zero-argument `super()` use the
shared object model. Adding module-specific raw heap inspection to `PyRuntime` would recreate the
coupling this layout removes.

Pure object-model compatibility suites live as Python files under
`tests/fixtures/python/object_model`. Shellsim's bounded pytest runner collects their top-level,
zero-argument `test_*` functions; a thin Rust harness installs the files into the VFS and checks
process results. The same source is run under CPython pytest when that optional reference is
available. Rust tests remain responsible for simulator-only observations such as resource stop
reasons, VFS construction, output bytes, and exit status.
