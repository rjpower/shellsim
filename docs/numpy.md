# NumPy in shellsim

shellsim ships a NumPy subset for simulated Python programs. It never runs host NumPy and does not
provide NumPy's C ABI.

## Layout

- **Native core** (`src/python/stdlib/numpy/`). The array engine and every per-element loop:
  - arrays, dtypes and scalar types;
  - ufuncs and reductions;
  - indexing and broadcasting;
  - sorting and searching;
  - reshape, transpose and concatenate, and `dot` and `matmul`;
  - the dense factorizations behind `numpy.linalg`, and the per-draw random samplers.

  They are exported as private native modules: `_numpy`, `_numpy_reduce`, `_numpy_shape`,
  `_numpy_sort`, `_numpy_products`, `_numpy_linalg`, `_numpy_random` and `_numpy_io`.
- **Frozen Python** (`src/python/stdlib/source/numpy/`). Everything composed from those
  primitives, with helpers grouped by topic:
  - `_creation`: ranges, grids and constructors;
  - `_shapes`: stacking, splitting, padding and axis helpers;
  - `_math`: calculus, interpolation, convolution, polynomials and `vectorize`;
  - `_products`: inner, outer and tensor products, and `einsum`;
  - `_sets`: set operations;
  - `_stats`: means, quantiles, histograms and the `nan*` functions;
  - `_errors`: `errstate`, `finfo` and `iinfo`;
  - `_io`: `.npy`, `.npz` and text files;
  - `_printing`: array text.

  The public submodules `numpy.linalg`, `numpy.fft`, `numpy.random`, `numpy.testing` and
  `numpy.exceptions` are one file each. `numpy.fft` is pure Python: radix-2 butterflies, plus
  Bluestein's algorithm for other lengths.

An array is a view (shape, strides, offset) over storage owned by the interpreter: packed
little-endian element bytes, or traced Python references for `dtype=object`. Slicing,
transposition and `reshape` return views when the layout allows, as in NumPy. Kernels are
monomorphized per element type, so `int8` wraps at 8 bits and `float32` rounds to single
precision after every operation. `float16` computes in `f32` and rounds on every store.

| Class | Dtypes |
| --- | --- |
| Boolean | `bool` |
| Signed integer | `int8`, `int16`, `int32`, `int64` |
| Unsigned integer | `uint8`, `uint16`, `uint32`, `uint64` |
| Floating point | `float16`, `float32`, `float64` |
| Complex | `complex64`, `complex128` |
| Text | `str` (`<U{n}`) |
| Object | `object` |

Promotion follows NEP 50: Python scalars are weak operands, and an out-of-range Python integer
raises `OverflowError`. NumPy scalar types form the usual hierarchy (`generic`, `number`,
`integer`, `floating`, ...), and `float64` and `complex128` derive from Python's `float` and
`complex`.

## Supported subset

- **Construction.** `array`, the `as*array` family, `zeros`, `ones`, `empty`, `full` and their
  `*_like` forms, `arange`, `linspace`, `logspace`, `geomspace`, `eye`, `identity`, `diag`,
  `meshgrid`, `fromiter`, `frombuffer`, the index helpers (`r_`, `c_`, `mgrid`, `ogrid`, `ix_`,
  `indices`), `astype`, `result_type`, `can_cast` and `issubdtype`.
- **Indexing.** Basic indexing returns views. Integer-array and boolean-mask indexing and
  assignment follow NumPy's rules, as do `where`, `nonzero`, `select`, `extract`, `take`, `put`
  and `copyto`.
- **Ufuncs.** The arithmetic, comparison, logical, bitwise, rounding, exponential,
  trigonometric and hyperbolic ufuncs, with `out=`, `where=`, `dtype=` and `casting=`, and the
  ufunc methods `reduce` and `accumulate`. `errstate` and `seterr` choose whether floating-point
  errors are ignored, warned about or raised.
- **Reductions and statistics.** `sum`, `prod`, `min`, `max`, `argmin`, `argmax`, `all`, `any`,
  `cumsum`, `cumprod`, `mean`, `var`, `std`, `average`, `median`, `percentile`, `quantile`, `cov`,
  `corrcoef`, `histogram` and the `nan*` variants.
- **Shape.** Reshaping, transposition and axis moves, joining and splitting (including `block`),
  `repeat`, `tile`, flips, `roll`, `pad`, triangles and diagonals, `kron` and broadcasting
  helpers.
- **Sorting and sets.** `sort`, `argsort`, `lexsort`, `partition`, `argpartition`,
  `searchsorted`, `bincount`, `digitize`, `unique` and the set operations.
- **Products and linear algebra.**
  - Products: `dot`, `vdot`, `inner`, `outer`, `matmul` and `@`, `tensordot` and `einsum`.
  - `numpy.linalg`, for real and complex arrays: `inv`, `solve`, `det`, `slogdet`, `eig`, `eigvals`, `eigh`,
    `eigvalsh`, `svd`, `svdvals`, `qr`, `cholesky`, `lstsq`, `pinv`, `matrix_rank`,
    `matrix_power` and `norm`.
- **Numerical helpers.** `diff`, `gradient`, `interp`, `trapezoid`, `convolve`, `correlate`,
  `polyfit`, `polyval`, `clip`, `isclose`, `allclose` and `vectorize`.
- **FFT.** `fft`, `ifft`, `rfft`, `irfft`, `fft2`, `ifft2`, `fftn`, `ifftn`, `fftfreq`,
  `rfftfreq`, `fftshift` and `ifftshift`.
- **Random.**
  - `default_rng` and `Generator`, over one `PCG64` bit generator.
  - `Generator` draws: `random`, `integers`, `uniform`, `normal`, `exponential`, `gamma`, `beta`,
    `chisquare`, `standard_t`, `f`, `lognormal`, `binomial` and `poisson`. It also has `choice`,
    `shuffle` and `permutation`.
  - The legacy functions `seed`, `rand`, `randn`, `randint`, `random`, `choice`, `shuffle` and
    `permutation`, plus `RandomState`.
- **Printing.** Array `repr` and `str`, `array2string`, `array_repr`, `array_str`,
  `format_float_positional`, `format_float_scientific`, and `set_printoptions`,
  `get_printoptions` and `printoptions`. The supported options are `precision`, `threshold`,
  `edgeitems`, `linewidth`, `suppress`, `sign` and `floatmode`.
- **File I/O.** `save`, `load`, `savez`, `savez_compressed`, `savetxt`, `loadtxt` and
  `genfromtxt`, through the virtual filesystem. `.npy` and `.npz` files use NumPy's format.
- **Testing.** `numpy.testing`: `assert_allclose`, `assert_array_equal`,
  `assert_array_almost_equal`, `assert_almost_equal`, `assert_equal`, `assert_array_less`,
  `assert_raises` and `assert_warns`.

## Compatibility contract

Values agree with NumPy 2.5.3 within floating-point tolerance. Shapes, dtypes, exception types
and warning categories match. Printed arrays match NumPy's layout for ordinary arrays. Error
messages are shellsim's own.

Some results differ from NumPy by design:
- **Summation order.** Floating-point sums use plain pairwise summation, and products sum each
  dot product in index order. The last bits can therefore differ from NumPy's.
- **Random streams.** Seeded runs are reproducible within shellsim but draw different numbers
  from NumPy's. Simulated programs have no host entropy, so an unseeded generator starts from a
  fixed seed.
- **Sorting.** Every sort is stable, whatever `kind` requests.
- **`frombuffer`.** Arrays cannot share storage with a Python `bytes` or `bytearray`, so
  `frombuffer` returns a read-only copy.
- **Elements of text and object arrays** are returned as plain Python `str` values and the
  stored objects.
- **Complex linear algebra.** These small-array paths use a real block representation for
  solves and eigenproblems, Householder reflections for QR, and one-sided Jacobi sweeps for SVD.
  Solves check their residual and make up to two correction passes. Euclidean and Frobenius norms
  scale before squaring. These methods do more work than the real kernels and cannot recover
  digits lost to input conditioning; general complex eigenvectors remain especially sensitive
  near repeated or defective eigenvalues. They are intended for small scientific calculations.

The portable suites in `tests/python/numpy/` state this contract. Cargo runs them under shellsim
(see `tests/python/scientific_suites.rs`). To re-check them against real NumPy, build the pinned
environment with `infra/scientific-reference.py` and export its interpreter as
`SHELLSIM_SCIENTIFIC_PYTHON`. `tests/python/numpy.rs` holds the shellsim-only checks: the
frontier and resource limits.

## Unsupported frontier

- **Unsupported dtypes and options** raise `NotImplementedError`. This covers:
  - structured, void, bytes (`S`), datetime and timedelta dtypes;
  - `numpy.linalg.qr(mode="raw")` with complex input;
  - `ndarray.resize`, and `view` with a different item size.
- **Object arrays in files.** Saving or loading object arrays in `.npy` or `.npz` files raises
  `ValueError`.
- **Missing modules.** `numpy.lib`, `numpy.strings`, `numpy.ma` and `numpy.polynomial` are
  absent. Importing one raises `ModuleNotFoundError`.
- **Missing names.** `np.matrix` and `np.longdouble` are absent, so using one raises
  `AttributeError`.

## Safety and accounting

Shape products, offsets and sizes use checked arithmetic, and arrays are limited to 64
dimensions. Every native operation that allocates element storage reserves memory first. It
charges CPU in proportion to the elements it touches; products and factorizations charge their
multiply-adds before they start. Python compositions are charged through the primitives they
call and the interpreter's own loop metering. File I/O goes through the simulated filesystem in
bounded chunks.
