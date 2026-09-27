# NumPy in shellsim

shellsim ships a NumPy 2.5 work-alike for simulated Python programs. It targets the observable
behavior of NumPy 2.5.3: values, dtypes, printed text, error types and messages, warnings, and
seeded random streams. It does not provide NumPy's C ABI, and it never runs host NumPy.

## Architecture

The package has two layers.

- **Native core** (`src/python/stdlib/numpy/`). Arrays, dtypes, scalar types, ufuncs,
  reductions, indexing, shape operations, sorting, products, printing digits, and the kernels
  behind `linalg`, `fft` and `random`. The core is exported as native modules: `_numpy` holds
  the core, and area modules such as `_numpy_reduce`, `_numpy_shape` and `_numpy_linalg` hold
  one family of functions each.
- **Frozen Python** (`src/python/stdlib/source/numpy/`). The `numpy` package star-imports the
  native modules and adds the parts NumPy itself writes in Python: `errstate`, statistics and
  `nan*` functions, set operations, histograms, `arrayprint`, `numpy.linalg`, `numpy.fft`,
  `numpy.random`, `numpy.strings`, `numpy.testing`, and `numpy.lib` file I/O. Their argument
  handling and error messages match NumPy's.

Larger submodules (`fft`, `random`, `testing`, `lib` and the file I/O functions) load on first
access through a module-level `__getattr__`, as in NumPy.

### Storage

An array is a view over storage owned by the VM:

```text
Array
  storage: packed little-endian element bytes, or traced Python references for dtype=object
  dtype:   an index into a constant dtype table, plus a width for str dtypes
  shape:   dimension lengths
  strides: signed byte strides, one per dimension
  offset:  byte offset of element [0, ..., 0]
```

Native code reads and writes storage only through runtime closures that cannot re-enter Python.
Views share storage, so slicing, transposition and `reshape` (when the layout allows) alias the
source as in NumPy. New arrays follow NumPy's memory-order rules: constructors and copies accept
`order` with C, F, A and K semantics, and ufunc outputs follow their operands' layout.

### Dtypes

A dtype is a `#[repr(u8)]` kind plus a character width for `str`. Metadata comes from a constant
table indexed by the kind, so lookups never scan. Kernels are monomorphized per element type, so
arithmetic happens at the dtype's own width: `int8` wraps at 8 bits, and `float32` rounds to
single precision after every operation. `float16` is stored as IEEE binary16 and computed in
`f32`, rounding on every store, as NumPy's half-precision loops do.

| Class | Dtypes |
| --- | --- |
| Boolean | `bool` |
| Signed integer | `int8`, `int16`, `int32`, `int64` |
| Unsigned integer | `uint8`, `uint16`, `uint32`, `uint64` |
| Floating point | `float16`, `float32`, `float64` |
| Complex | `complex64`, `complex128` |
| Text | `str` (`<U{n}`, UCS-4) |
| Object | `object` |

Promotion follows NEP 50: Python scalars are weak operands, and an out-of-range Python integer
raises `OverflowError` instead of wrapping. Every computed dtype is native-endian.

Storage is always little-endian. A multi-byte numeric dtype may still be big-endian (`>i4`). The
byte order is a descriptor attribute that applies only where NumPy exposes raw bytes:
`tobytes`, `frombuffer`, `byteswap`, and `.npy` files. Computation reads native storage.

### Scalars

NumPy scalars are registered value kinds carrying one element and its dtype. The abstract
hierarchy (`generic`, `number`, `integer`, `floating`, ...) is registered, so `isinstance` checks
work, and `float64` and `complex128` derive from Python's `float` and `complex`. Scalar
arithmetic shares the ufuncs' element semantics, so it wraps, promotes, and reports
floating-point errors as NumPy does.

Array and scalar operators follow NEP 13's opt-out: when the other operand's type sets
`__array_ufunc__ = None`, the operator returns `NotImplemented` and Python calls that operand's
reflected method, which is how `pytest.approx` compares arrays.

## Compatibility contract

The portable suites in `tests/python/numpy/` are the contract. Each file states literal
expectations that were checked against NumPy 2.5.3 on CPython 3.14.4, the versions pinned in
`tests/python/scientific-requirements.txt`. Cargo runs every suite under shellsim's pytest (see
`tests/python/scientific_suites.rs`). Re-checking the suites against real NumPy is optional: build
the pinned environment with `infra/scientific-reference.py` and export its interpreter as
`SHELLSIM_SCIENTIFIC_PYTHON`.

The suites cover construction, dtypes, scalars, indexing, ufuncs, `errstate`, reductions, shape
operations, sorting and sets, string and object arrays, printing, `linalg`, `fft`, `random`, file
I/O, and `numpy.testing`. `tests/python/numpy.rs` holds shellsim-only checks: deliberate
differences, the unsupported frontier, and resource limits.

## Supported surface

- **Construction and conversion.** `array` and the `as*array` family, the `*_like`
  constructors, `arange`, `linspace` (including complex bounds), `logspace`, `geomspace`, `eye`,
  `diag`, `meshgrid`, `fromiter`, `frombuffer`, `astype` with NumPy's casting rules, and
  `result_type`, `can_cast`, `issubdtype`, `isdtype` and `mintypecode`.
- **Indexing.** Basic indexing returns views. Integer-array and boolean-mask indexing follow
  NumPy's placement rules for combined advanced indices. Assignment works through every index
  form, broadcasting the value as NumPy does. `select`, `extract`, `place`, `putmask`, `nonzero`,
  `where` and `copyto` choose and fill elements by condition.
- **Ufuncs.** The arithmetic, comparison, logical, bitwise, rounding, exponential,
  trigonometric and hyperbolic ufuncs, with `out=`, `where=`, `dtype=`, `casting=`, `order=`,
  and `reduce`/`accumulate`. Operators share the ufunc table. `np.errstate` and `np.seterr`
  control whether floating-point errors are ignored, warned about, or raised. `vectorize` applies
  a Python function element by element, including the `signature=` form.
- **Reductions and statistics.** `sum`, `prod`, `min`, `max`, `argmin`, `argmax`, `all`, `any`,
  `cumsum`, `cumprod`, `cumulative_sum`, `cumulative_prod`, `mean`, `var`, `std`, `average`,
  `median`, quantiles, and the `nan*` variants. A reduction visits the reduced axes in memory
  order, as NumPy's iterator does, and floating-point (and complex) `add` groups its pairwise
  sums the way NumPy's inner loop does, so sums and means of contiguous, transposed, and strided
  arrays match NumPy bit for bit along any axis. A reduction that casts its input, such as
  `sum(dtype=np.float64)` of `float32` data, may differ in the last bits because shellsim does
  not split it into NumPy's 8192-element cast buffers. `sum`, `prod`, `max`/`amax`,
  `min`/`amin`, `any`, `all`, and `ufunc.reduce`/`accumulate` support `where=` masks; `mean`,
  `var` and `std` do not yet.
- **Shape.** Reshaping, transposition and axis moves, joining and splitting (including `block`),
  repetition, flips (`flip`, `fliplr`, `flipud`), rolls, `pad`, triangles and diagonals, `kron`,
  and broadcasting helpers, with NumPy's view-or-copy semantics.
- **Sorting and sets.** `sort`, `argsort`, `lexsort`, `partition`, `argpartition`,
  `searchsorted`, `bincount`, `unique` and its variants, set operations, `histogram` and
  `digitize`.
- **Products and linear algebra.** `dot`, `vdot`, `vecdot`, `cross`, `inner`, `outer`, `matmul`
  and `@`, `tensordot`, and `numpy.linalg` for real arrays: `inv`, `solve`, `det`, `slogdet`,
  `eigh`, `eigvalsh`, `svd`, `qr`, `cholesky`, `lstsq`, `pinv`, `matrix_rank`, `matrix_power` and
  `norm`.
- **Whole-array helpers.** `diff`, `gradient`, `interp`, `trapezoid`, `convolve`, `correlate`,
  `angle`, `polyfit` and `polyval` (least-squares fit and Horner evaluation, not the
  `numpy.polynomial` package, which is absent).
- **FFT.** `numpy.fft`: `fft`, `ifft`, `rfft`, `irfft`, `hfft`, `ihfft`, their two- and
  n-dimensional forms (`fft2`, `fftn`, `rfft2`, `rfftn`, ...), `fftfreq`, `rfftfreq`,
  `fftshift`, and `ifftshift`, with `n`/`s`, `axis`/`axes`, `norm`, and `out`. Every transform
  length runs in O(n log n) (a radix-2 transform for powers of two, Bluestein's algorithm on
  top of it otherwise); results agree with NumPy to about `1e-15 * log2(n)` relative to the
  largest output magnitude, tighter than NumPy's own single-precision loops.
- **Random.** `numpy.random` implements `SeedSequence`, `MT19937`, `PCG64`, `Generator`, and the
  legacy `RandomState` from published algorithms. Both draw uniforms, integers, normals,
  exponentials, and gamma, chi-square, F, Student's t, binomial and Poisson variates, and choose,
  shuffle and permute. Seeded streams match NumPy 2.5.3 bit for bit, except for the ziggurat
  draws described under "Deliberate differences".
- **Text.** `str` arrays, `numpy.strings`, and object arrays with Python-level element operations.
- **Printing.** Scalar and array `repr`/`str`, `array2string`, `array_repr`, `array_str`,
  `format_float_positional`/`_scientific`, and `set_printoptions`/`get_printoptions`/
  `printoptions` reproduce NumPy's output byte for byte: shortest round-trip digits (via Rust's
  correctly-rounded float formatting), the positional/scientific notation switch, summarization,
  line wrapping, and dtype/shape suffixes.
- **File I/O.** `save`, `load`, `savez`, `savez_compressed`, `savetxt`, `loadtxt`, and
  `genfromtxt` through the simulated `open` and the virtual filesystem. `.npy` and `.npz` files
  follow NumPy's formats byte for byte, so each implementation reads the other's files.
- **Testing.** `numpy.testing` assertions with NumPy's comparison rules and messages.

### Object arrays in `.npy` files

shellsim has no `pickle` module, so `np.save` writes object arrays with a protocol 4 pickler
that reproduces the bytes of CPython's C pickler, including memo and frame layout. It handles
elements that are `None`, bools, numbers, strings, bytes, NumPy scalars and arrays, and lists,
tuples and dicts of these; other elements raise `NotImplementedError`. `np.load` with
`allow_pickle=True` uses a restricted unpickler that resolves only NumPy's array and scalar
reconstructors and a few builtins. Any other global raises `UnpicklingError`, so loading a file
never imports a module or calls arbitrary code.

## Deliberate differences

- **Sorting.** Every sort is a stable merge sort, whatever `kind` requests. NumPy's default
  sorts are not stable, so the order they give equal elements is unspecified; the stable order is
  one NumPy may produce. `partition` and `argpartition` fully sort each lane rather than running
  introselect, which still satisfies NumPy's documented contract (the `kth` element lands where a
  full sort would put it, with smaller elements before it and larger ones after) but leaves the
  unindexed elements in sorted, not arbitrary, order.
- **`frombuffer`.** Arrays cannot share storage with a Python `bytes` or `bytearray`, so
  `frombuffer` copies. The copy is read-only, so a write that NumPy would pass through to a
  `bytearray` fails instead of silently diverging.
- **Pickle memo.** Short strings have no identity in shellsim, so the pickler memoizes equal
  strings by value. CPython memoizes by identity and writes computed duplicates again. Both
  files load to equal arrays.
- **Unseeded random streams.** Simulated programs have no host entropy, so an unseeded
  `SeedSequence` uses fixed entropy and produces the same stream on every run.
- **Ziggurat draws.** `Generator`'s normal and exponential draws (the latter with the default
  `method="zig"`) compute their ziggurat tables from Marsaglia and Tsang's recursion. NumPy's
  tables differ in the last few bits, so these draws, and the gamma, chi-square, F and Student's
  t draws built on them, agree with NumPy to about `1e-14` relative. They consume the same raw
  words, so later draws stay aligned.
- **`str` and `object` elements** box to plain Python `str` values and the stored objects, not to
  `np.str_` or `np.object_` instances.
- **Byte order.** Views between byte orders, big-endian `str` dtypes, and `tobytes` of object
  arrays are rejected.
- **Product summation.** Products sum each dot product in index order. NumPy passes
  floating-point products to BLAS, whose blocked kernels add in a different order that depends
  on the CPU, so their last bits can differ.
- **`correlate`/`convolve`.** Both compute in `float64` (or `complex128`) regardless of the
  input dtype, then cast the result back. Integer inputs whose products fall outside the 53-bit
  exactly representable range may round differently than NumPy's native-width accumulation.

## Unsupported frontier

These fail explicitly with shellsim's unsupported-operation error or `NotImplementedError`:

- structured, void, bytes (`S`), datetime and timedelta, and `longdouble` dtypes;
- masked arrays, `np.memmap`, `mmap_mode=`, and `genfromtxt` with `names=` or mixed column types;
- complex input to `numpy.linalg`, and `eig` and `eigvals`;
- `where=` masks in `mean`, `var`, and `std` (the `sum`, `prod`, `max`, `min`, `any`, `all`, and
  `ufunc.reduce`/`accumulate` reductions they build on support `where=`);
- `linspace` with `axis=`, `meshgrid` with `sparse=True`, `ndarray.resize`, and `ndarray.view`
  with a different item size.

`einsum`, `np.matrix`, `numpy.ma`, `numpy.polynomial`, and `numpy.emath` are absent; accessing
them raises `AttributeError`.

## Safety and accounting

Shape products, offsets, and sizes use checked arithmetic, and arrays are limited to 64
dimensions. Every operation that allocates element storage reserves memory first and charges CPU
in proportion to the elements it touches; products and linear algebra charge their multiply-adds
before they start. Costs are deliberately coarse. File I/O reads and writes through the
simulated filesystem in bounded chunks, so a `.npy` file costs about twice its payload at its
peak. Arrays share only interpreter-owned storage and cannot expose a host pointer.
