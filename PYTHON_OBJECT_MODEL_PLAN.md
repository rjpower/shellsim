# Python object-model implementation plan

Status: implemented, 2026-09-11.

Implemented in the current rollout: bootstrapped type registry, universal `type_id`, common heap
headers, C3 lookup, cached user slots, Python and native descriptors, standard descriptors,
`__set_name__`, behaviorally captured class ownership for zero-argument `super`, registered native
methods for regex, argparse, pytest, unittest, streams and environment values, compatible
metaclass selection, `__prepare__`, `__init_subclass__`, metaclass `__init__`/`__call__`, and
metered heap-backed big integers used by literals, arithmetic, JSON, argparse, and `PyNumber`.

This plan replaced shellsim's former mixture of VM builtin tags, heap object variants, and
module-specific method dispatch with one metered Python object model. Each milestone preserves
the simulation boundary: no Python value or protocol operation gains ambient host capabilities.

## Design target

Every Python value has a semantic `TypeId`. Immediate values retain compact payloads; values that
need mutable state or distinct identity use an `ObjectId` into the modeled heap. `ValueTag` is the
physical storage discriminator and must remain separate from the Python-level `TypeId`.

```text
#[repr(C)]
PyValue {
    payload: u64,
    aux: [u8; 7],
    tag: ValueTag,
}

ValueTag = None | Bool | Int | Float | SmallString | Object | Native

HeapObject {
    type_id: TypeId,
    attributes: optional instance dictionary,
    payload: ObjectPayload,
}

PyType {
    name: String,
    bases: Vec<TypeId>,
    mro: Vec<TypeId>,
    metaclass: TypeId,
    attributes: type dictionary,
    layout: PyLayout,
    slots: TypeSlots,
}
```

The target representation is 16 bytes. `Int` and `Float` use the payload directly. `Object` uses
it for an `ObjectId`. `SmallString` uses the payload plus `aux` for up to fifteen inline bytes;
longer strings are heap objects. `Native` uses a closed interpreter-owned handle, never a host
pointer exposed to simulated code. The exact bit packing remains private to the runtime, and
native modules continue to exchange the safe `PyValue` wrapper rather than inspecting fields.

The runtime uses this exact 16-byte representation. All semantic typing goes through
`type_id(value)`, and runtime-owned string construction chooses inline or metered heap storage.
Native modules cannot allocate an unmetered long string.

There is deliberately no per-value identity field. Immediate identity is canonical by encoded
value (`Float` compares exact bits); heap-backed values compare `ObjectId`. Assignment therefore
preserves identity without allocating, while independently created long strings, big integers,
collections, and instances may retain distinct identities.

`type_id(value)` maps storage to semantics:

```text
None/Bool/Int/Float/SmallString -> fixed builtin TypeId
Object(id)                     -> heap[id].type_id
Native(handle)                 -> registered interpreter-owned TypeId
```

Immediate immutable values are canonical by representation. Integers and strings use value
identity and floats use exact IEEE bits. Heap values use `ObjectId`. Container protocols test
identity before equality.

## Milestones

### 1. Bootstrap builtin types

- Introduce stable `TypeId` values and a `TypeRegistry` owned by the Python runtime state.
- Bootstrap `object` and `type`, including `type(object) is type`, `type(type) is type`, and the
  required base/MRO cycle.
- Register scalar, collection, function, module, iterator, exception, and native payload types.
- Represent builtin names as references to registered type objects, not builtin functions.

Acceptance: builtin type identity, representation, construction, `isinstance`, and `issubclass`
all query the registry.

### 2. Make semantic typing universal

- Add the single `type_id(value)` operation.
- Retain `PyKind` only as a checked storage-shape API for native module casts.
- Remove semantic type decisions based directly on `PyKind`, `Builtin`, or heap variants.

Acceptance: `type()`, class predicates, descriptor binding, and protocol lookup start with
`type_id(value)`.

### 3. Generalize heap objects

- Give every heap allocation a type header and optional attribute dictionary.
- Move variant-specific data behind `ObjectPayload`.
- Keep typed runtime views so native modules never inspect `ObjectPayload` directly.
- Meter headers, dictionaries, and payload growth before host allocation.

Acceptance: ordinary instance attributes and native payload attributes use the same object header.

### 4. Implement descriptors and slot caches

- Implement data-descriptor, instance-dictionary, and non-data-descriptor lookup order.
- Make Python functions and native `MethodDef`s descriptor values; attach erased native binary
  functions directly to builtin numeric and sequence slots.
- Add `TypeSlots` for call/new/init, attribute access, representation, truth/hash, iteration,
  numeric operators, comparison, and containment.
- Populate slots from builtin/native definitions and user dunder methods at class creation.
- Invalidate and rebuild affected caches when a type dictionary changes.

Acceptance: bound Python and native methods use the same descriptor path; normal calls and
operators dispatch through cached slots.

### 5. Migrate remaining native object methods

- Move argument parsers, streams, environment markers, and raises/context-manager objects to
  checked runtime views and ordinary native descriptors.
- Use metered snapshot-and-commit mutation for mutable native payloads.
- Delete their former VM method switches after equivalent tests pass.

Acceptance: adding a native type or method requires a type definition and runtime view, not a VM
match arm.

### 6. Add `super()` and class cells

- Capture the defining class on functions created by class bodies and expose it as the immutable
  `__class__` cell while their method frame is active.
- Implement explicit and zero-argument `super` over the receiver type's C3 MRO.
- Bind attributes returned through `super` with the original receiver.

Acceptance: cooperative single and multiple inheritance fixtures match CPython.

### 7. Add standard descriptors

- Implement `property`, `staticmethod`, and `classmethod` as descriptor types.
- Call `__set_name__` after class creation.
- Support user-defined `__get__`, `__set__`, and `__delete__` within the bounded call path.

Acceptance: descriptor precedence and binding fixtures match CPython, including data versus
non-data descriptor behavior.

### 8. Complete class and metaclass construction

- Resolve the winning metaclass from explicit keywords and bases.
- Call `metaclass.__prepare__`, execute the body in its namespace, and call the metaclass with
  `(name, bases, namespace, keywords)`.
- Implement `type.__new__`/`type.__init__`, compatible layout selection, slot installation,
  `__set_name__`, and `__init_subclass__`.
- Enable custom metaclass `__new__`, `__init__`, and `__call__` through the common protocol path.

Acceptance: supported metaclass hooks observe Python ordering and unsupported namespace or layout
capabilities fail explicitly.

### 9. Extend compact scalar storage

- Add heap-backed arbitrary-precision integers and promote checked `i64` overflow to them.
- Add an inline small-string representation with heap-backed long strings.
- Route both through existing numeric/string protocols and preserve immediate canonical identity.
- Meter parsing, arithmetic, conversion, hashing, and output before unbounded work.

Acceptance: large-integer arithmetic and long-string behavior match CPython within configured
resource limits, while common integers and short strings remain pointer-free.

## Validation after each milestone

- Add unit coverage for lookup/slot/layout invariants.
- Add integration or differential coverage for observable Python behavior.
- Run a narrow affected test while iterating.
- Run `./infra/pre-commit.py --all-files --fix`, `./infra/pre-commit.py --all-files`, and
  `./infra/ci/run_tests.py` before handoff.

## Explicit non-goals

- Host execution or delegation to CPython from product code.
- Ambient filesystem, process, network, environment, locale, or clock access.
- Unmetered arbitrary-precision work or container growth.
- Binary compatibility with CPython's `PyObject` layout.
