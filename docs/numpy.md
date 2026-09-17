# Minimal NumPy architecture

## Decision

Implement a useful NumPy surface over one deliberately simple array contract. Prefer predictable,
generic loops to specialized kernels. Shellsim does not need BLAS, SIMD, dtype-specific storage,
or NumPy's C ABI to cover the array operations commonly found in small agent tasks.

An array is an opaque interpreter object with:

```text
Array
  storage: shared flat sequence of 16-byte PyValue scalars
  shape:   non-negative dimension lengths
  strides: signed storage offsets, one per dimension
  offset:  storage offset of logical element [0, ..., 0]
```

The native module sees only a checked `PyArray` handle and runtime operations for creating arrays
and views, inspecting layout, and getting or setting an indexed value. It does not inspect heap
objects. Contiguous arrays use row-major strides. Reshape and transpose return views when their
layout permits it; operations may make a simple contiguous copy when they cannot express a result
as a view.

Array elements are ordinary inline `PyValue`s, not arena objects. NumPy registers opaque
`numpy.bool_`, `numpy.int64`, and `numpy.float64` value kinds when the interpreter is built. A
registration supplies a constructor and protocol slots; only the NumPy module can pack or unpack
its payload. The VM stores the payload and a registered-kind number, then dispatches every
operation through the registered functions. It has no NumPy-specific scalar tags, coercions, or
arithmetic branches.

## Kernel model

All kernels iterate logical indices and access values through the contract. Matrix multiplication
is the literal three-loop algorithm:

```text
for row in left rows
  for column in right columns
    total = 0
    for inner in shared dimension
      total += left.at([row, inner]) * right.at([inner, column])
```

Elementwise operations use the same iterator plus right-aligned broadcasting. Reductions vary one
axis while retaining the same indexed access. This is intentionally slow. It keeps behavior small,
auditable, deterministic, and uniformly resource-metered.

## Compatibility target

The first coherent slice includes:

- construction with `array`, `asarray`, `zeros`, `ones`, `full`, and `arange`;
- like-constructors, `linspace`, `eye`, and `identity`;
- `shape`, `ndim`, `size`, `T`, signed-stride slices on the first axis, basic indexing,
  integer-array gathers, boolean expressions and masks, assignment, iteration, `tolist`, `copy`,
  `reshape`, `transpose`, `squeeze`, `expand_dims`, `swapaxes`, `broadcast_to`, `flatten`, and
  `ravel`;
- elementwise `+`, `-`, `*`, `/`, rich comparisons, unary operators, common floating-point
  functions, `minimum`, `maximum`, `clip`, and `where`, all with scalar and array broadcasting;
- `concatenate`, `stack`, `vstack`, and `hstack`;
- `sum`, `prod`, `mean`, `min`, `max`, `var`, `std`, `median`, `all`, `any`, `dot`, and
  `argmin`, `argmax`, cumulative sums and products, `inner`, `outer`, and two-dimensional
  `matmul`;
- module functions that delegate to the corresponding array behavior.

Numeric storage is type-erased but numeric behavior is not accidental. Integer elements are
signed 64-bit values and arithmetic wraps at 64 bits instead of promoting to Python's unbounded
integer. Float elements are IEEE-754 doubles. Array access returns the corresponding registered
NumPy scalar; `tolist` converts elements back to ordinary Python scalars. The `dtype` argument
accepts `bool`, `int`, `int64`, `float`, and `float64` names and type objects.

Explicitly unsupported in the first slice are NumPy's C ABI, buffers, structured/object/string
dtypes, masked arrays, multidimensional slice syntax, mixed advanced indices inside a tuple,
axis tuples and `keepdims` in reductions, batched matrix multiplication, `tensordot`, generalized
`einsum`, linear algebra factorizations, random distributions, persistence formats, and
performance guarantees. `einsum` is omitted because a coherent implementation requires label
parsing, diagonal selection, contraction, output ordering, broadcasting, and ellipsis handling;
a special-case spelling of matrix multiplication would not be an intelligible boundary.
Unsupported arguments and operations fail rather than silently changing meaning.

## Safety and accounting

Shape products, offsets, and output sizes use checked arithmetic. Allocation is reserved before
storage growth. Every element visit and scalar operator call consumes CPU fuel. Arrays share only
interpreter-owned storage and cannot expose a host pointer or native extension boundary.

Integration tests retain checked shellsim expectations for views and aliasing, registered scalar
identity and overflow, broadcasting, gathers and masks, statistics, matrix multiplication,
shape and stride views, joins, selection kernels, invalid shapes and axes, incompatible operands,
and resource exhaustion.
