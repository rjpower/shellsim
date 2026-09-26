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

Real array elements are ordinary inline `PyValue`s, not arena objects. NumPy registers each real
scalar kind from one local descriptor table. A row supplies the dtype, numeric class, width,
aliases, and a scalar representation. Real rows carry a constructor and shared protocol slots, and
only the NumPy module can pack or unpack their payload. The VM
stores the payload and a registered-kind number, then dispatches every operation through the
registered functions. It has no NumPy-specific scalar tags, coercions, or arithmetic branches.

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

The supported surface includes:

- construction with `array`, `asarray`, `zeros`, `ones`, `full`, and `arange`;
- like-constructors, `linspace`, `eye`, and `identity`;
- `shape`, `ndim`, `size`, `T`, signed-stride slices on the first axis, basic indexing,
  integer-array gathers, boolean expressions and masks, full-coordinate and advanced-index
  assignment, iteration, `tolist`, `copy`,
  `reshape`, `transpose`, `squeeze`, `expand_dims`, `swapaxes`, `broadcast_to`, `flatten`, and
  `ravel`;
- elementwise `+`, `-`, `*`, `/`, rich comparisons, unary operators, common floating-point
  functions, `minimum`, `maximum`, `clip`, and `where`, all with scalar and array broadcasting;
- `concatenate`, `stack`, `vstack`, and `hstack`;
- `sum`, `prod`, `mean`, `min`, `max`, `var`, `std`, `median`, `all`, `any`, `dot`, and
  `argmin`, `argmax`, cumulative sums and products, `inner`, `outer`, and two-dimensional
  `matmul`;
- module functions that delegate to the corresponding array behavior.

Numeric storage is type-erased but numeric behavior is not accidental. The supported dtype table
is:

| Class | Canonical dtypes | Exported aliases | String codes |
| --- | --- | --- | --- |
| Boolean | `bool_` | `bool` | `bool`, `bool_`, `?` |
| Signed integer | `int8`, `int16`, `int32`, `int64` | `byte`, `short`, `intc`, `int_`, `intp`, `longlong` | `i1`, `i2`, `i4`, `i8` |
| Unsigned integer | `uint8`, `uint16`, `uint32`, `uint64` | `ubyte`, `ushort`, `uintc`, `uint`, `uintp`, `ulonglong` | `u1`, `u2`, `u4`, `u8` |
| Floating point | `float32`, `float64` | `single`, `double` | `f4`, `f8` |
| Complex | `complex128` | `cdouble` | `complex`, `c16` |

Integer arithmetic wraps at its result width instead of promoting to Python's unbounded integer.
Conversions reject values outside the requested integer dtype. Promotion chooses the wider type
within one integer class; mixed signed and unsigned operands use the smallest signed type that can
represent both, or `float64` when none can. `float32` combined with an integer wider than 16 bits
promotes to `float64`. Python scalars are weak operands: integer scalars preserve an integer or
floating array dtype when the value fits, and floating scalars preserve a floating array dtype.
An out-of-range Python integer is rejected instead of wrapped. Integer `sum`, `prod`, `cumsum`,
and `cumprod` widen inputs below 64 bits to `int64` or `uint64`; `float32` floating reductions
remain `float32`. Floating unary functions preserve `float32` inputs and use `float64` for integer
inputs because `float16` is outside the supported dtype boundary.

Array access returns the corresponding registered NumPy scalar; `tolist` converts elements back
to ordinary Python scalars, including the full `uint64` range.

`complex128` is the one dtype without a registered value kind. Its table row has a complex scalar
representation. Each element is a builtin `complex` arena object, and `numpy.complex128` and
`numpy.cdouble` are the builtin `complex` type, so indexing returns an ordinary `complex`. Complex
input infers `complex128`, and complex dominates promotion. `+`, `-`, `*`, `/`, negation, `==`,
`!=`, `sum`, `prod`, `mean`, cumulative sums and products, `dot`, `inner`, `outer`, `matmul`,
`where`, `all`, `any`, `array_equal`, and `.npy` files with descriptor `<c16` are supported.
`abs` returns `float64`. `real`, `imag`, and `conjugate` (also `conj`) are available as functions,
array attributes, and array methods. Real dtypes return themselves as their real part, a zero array
as their imaginary part, and an unchanged conjugate. `real` and `imag` return copies rather than
writable views because an element has no addressable component storage.

Operations that need a real value domain check the dtype once at entry and raise `TypeError` for
complex operands: ordering comparisons, `min`, `max`, `argmin`, `argmax`, `argsort`, `minimum`,
`maximum`, `clip`, `var`, `std`, `median`, `percentile`, `allclose`, `linspace`, `isnan`,
`isinf`, `sign`, bitwise invert, and every transcendental or rounding function. NumPy defines
several of these for complex input; shellsim rejects them rather than approximating them. Casting
complex values to a real dtype also raises `TypeError` instead of discarding the imaginary part
with a warning.

Explicitly unsupported are `float16`, `complex64`, NumPy's C ABI, buffers,
structured/object/string dtypes, masked arrays, arrays above 64 dimensions, multidimensional slice
syntax, partial basic-index assignment, mixed advanced indices inside a tuple, axis tuples and
`keepdims` in reductions, batched matrix multiplication, `tensordot`, generalized `einsum`, linear
algebra factorizations, random distributions, persistence formats, and performance guarantees.
`einsum` is omitted because a coherent implementation requires label parsing, diagonal selection,
contraction, output ordering, broadcasting, and ellipsis handling; a special-case spelling of
matrix multiplication would not be an intelligible boundary. Unsupported arguments and operations
fail rather than silently changing meaning.

## Safety and accounting

Shape products, offsets, and output sizes use checked arithmetic. Allocation is reserved before
storage growth. Every element visit and scalar operator call consumes CPU fuel. Arrays share only
interpreter-owned storage and cannot expose a host pointer or native extension boundary.

Integration tests retain checked shellsim expectations for views and aliasing, registered scalar
identity and overflow, broadcasting, gathers and masks, statistics, matrix multiplication,
shape and stride views, joins, selection kernels, invalid shapes and axes, incompatible operands,
and resource exhaustion.
