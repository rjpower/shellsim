# SciPy in shellsim

shellsim ships `scipy.special`, `scipy.stats` and `scipy.linalg` for simulated Python programs.
It targets the observable behavior of SciPy 1.18.1 with NumPy 2.5.3: values, result dtypes,
error types and messages, and warnings. It never runs host SciPy. The other SciPy subpackages
raise `NotImplementedError` when imported.

## Architecture

- **Ufuncs.** Each `scipy.special` function is an entry in NumPy's ufunc table (see
  [numpy.md](numpy.md)), so broadcasting, `out=`, `where=`, `dtype=`, casting and scalar results
  behave as they do for NumPy's ufuncs. The native module `_scipy_special` exports them.
- **Kernels** (`src/python/stdlib/scipy/special/`). The kernels are an independent
  implementation of standard published algorithms, not a port of SciPy's Cephes- or
  Boost.Math-derived code: the Lanczos approximation for `gamma`/`gammaln` (`gamma.rs`), a
  series-plus-continued-fraction regularized incomplete gamma function (`igam.rs`) and a
  continued-fraction regularized incomplete beta function (`ibeta.rs`) that every other
  distribution function is built on, Peter Acklam's rational approximation refined by Halley's
  method for `ndtri` (`erf.rs`), and Euler-Maclaurin summation for the Hurwitz and Riemann zeta
  functions (`zeta.rs`). Matching SciPy's *values* (to about `1e-13` relative error) does not
  require matching SciPy's *algorithms*, so results can differ from SciPy's by a few ULPs; see
  "Deliberate differences" below for the specific, measured gaps.
- **Frozen Python** (`src/python/stdlib/source/scipy/`). `scipy.special` star-imports the
  native module and adds the functions SciPy writes in Python: `zeta`, `comb`, `perm`,
  `factorial`, `factorial2`, `factorialk`, `logsumexp`, `softmax` and `log_softmax`. They
  follow SciPy's code, so their argument handling and messages match. `scipy.special._ufuncs`
  holds every ufunc, including the private `_binom_pmf`, `_binom_cdf`, `_binom_sf`, `_binom_ppf`
  and `_binom_isf` that `scipy.stats` calls. The `scipy` package loads subpackages on first
  access, as SciPy does.
- **`scipy.stats`** is frozen Python that reimplements SciPy's observable behavior directly:
  a shared `rv_generic` base with `rv_continuous` and `rv_discrete` subclasses, the summary
  statistics, and the correlation and hypothesis tests, computing with NumPy and
  `scipy.special`. Each distribution supplies `_pdf`/`_pmf` and `_cdf` (and, where a closed form
  exists, `_ppf`, `_stats`, `_entropy` and `_rvs`); generic methods derived from whichever of
  those a subclass defines cover the rest, the way SciPy's do. As in SciPy, `inspect.signature`
  of `_pdf` and `_cdf` gives a distribution's shape parameters. Each statistic declares SciPy's
  full signature, including `axis`, `nan_policy` and `keepdims`, rather than having a decorator
  add them.
- **`scipy.linalg`** is frozen Python that follows SciPy's modules over a native module,
  `_scipy_linalg`, which plays the parts of SciPy's compiled code: the C++ `_batched_linalg`
  loops behind `solve`, `inv`, `det`, `lu`, `cholesky` and `qr`, the `expm` kernel, and the f2py
  wrappers in `scipy.linalg.lapack` and `scipy.linalg.blas`. The LAPACK and BLAS routines are
  ports of the ones SciPy's OpenBLAS runs (see below). `svd`, `eigh` and the functions built on
  them (`lstsq`, `pinv`, `pinvh`, `polar`, `null_space`, `orth` and `subspace_angles`) use
  shellsim's `numpy.linalg`.

### Rounding in `scipy.linalg`

SciPy's wheels bundle OpenBLAS, which selects kernels for the CPU when it loads, replaces some
LAPACK routines with its own, and runs the rest from reference LAPACK on its kernels. The same
matrix can therefore round differently on different machines. shellsim ports the kernels and
routines that SciPy 1.18.1's OpenBLAS 0.3.31 uses on the reference machine, an AMD Zen 2 CPU for
which OpenBLAS picks its Haswell kernels and starts 16 threads. With them, LU, Cholesky and
triangular factorizations, the solves and inverses built on them, and `det` agree with SciPy's
to the last bit within these limits:

- OpenBLAS switches to blocked drivers above order 33 for `getrf`, 32 for `potrf`, 64 for
  `lauum` and 256 for `trsm`. shellsim keeps the unblocked kernels at every order, so larger
  results can differ in the last bits.
- From order 17, OpenBLAS runs `potri`, behind `inv(a, assume_a="pos")`, on its threaded
  drivers, which shellsim does not model.
- OpenBLAS splits several right-hand sides among its threads, and a column solved alone takes
  a different kernel. shellsim splits them as the 16 threads of the reference machine do, so
  up to 16 right-hand sides are each solved alone.
- `float32` input follows the double-precision kernels. OpenBLAS's single-precision kernels
  block by eight columns and 32 elements, so `float32` results can differ in the last bit from
  about order 5.

### Loop selection

SciPy registers a `float32` loop and a `float64` loop for each function, and NumPy selects the
first loop that every input casts to safely. `float32`, `float16` and 8- and 16-bit integer
inputs therefore compute in single precision. Wider integers and `float64` compute in double
precision. `logit` registers its `float64` loop first, so only `float32` input selects single
precision. Every single-precision loop here computes in `f64` and rounds the final result, since
a `float32` result correctly rounded from a `float64` computation meets this crate's accuracy
target without a separate single-precision code path; see "Deliberate differences" for where
this differs from SciPy's own `float32` loops in the last place.

`bdtr` and `bdtrc` also have a loop that takes the trial count `n` as an integer. When a call
selects a loop with a floating-point `n`, it issues SciPy's `DeprecationWarning`.

As in SciPy, the kernels never raise floating-point errors, and `np.errstate` does not affect
them. Domain errors return NaN and poles return infinities.

## Supported surface

- **Error function.** `erf`, `erfc`, `erfinv`, `erfcinv`, `ndtr`, `log_ndtr`, `ndtri`.
- **Gamma and beta.** `gamma`, `rgamma`, `gammaln`, `loggamma`, `psi` (and `digamma`), `beta`,
  `betaln`, `poch`, `binom`, `zeta`.
- **Incomplete gamma and beta.** `gammainc`, `gammaincc`, `gammaincinv`, `gammainccinv`,
  `betainc`, `betaincc`, `betaincinv`.
- **Distributions.** `stdtr`, `stdtrit`, `chdtr`, `chdtrc`, `chdtri`, `fdtr`, `fdtrc`, `fdtri`,
  `pdtr`, `pdtrc`, `pdtrik`, `bdtr`, `bdtrc`.
- **Logistic and information theory.** `expit`, `logit`, `log_expit`, `xlogy`, `xlog1py`,
  `entr`, `rel_entr`, `kl_div`.
- **Other.** `expm1`, `log1p`, `boxcox`, `inv_boxcox`, `comb`, `perm`, `factorial`,
  `factorial2`, `factorialk`, `logsumexp`, `softmax`, `log_softmax`.
- **Distributions (`scipy.stats`).** `norm`, `t`, `chi2`, `f`, `uniform`, `expon`, `binom` and
  `poisson`, and user subclasses of `rv_continuous` and `rv_discrete`. They provide densities or
  mass functions, distribution and survival functions, their inverses and logarithms, `stats`,
  `moment`, `entropy`, `median`, `mean`, `var`, `std`, `interval`, `support`, `nnlf`, `rvs`
  from NumPy's seeded streams, and frozen distributions. `norm`, `uniform` and `expon` fit by
  their closed-form estimates, and discrete `expect` sums its series as SciPy's does.
- **Statistics (`scipy.stats`).** `describe`, `moment`, `skew`, `kurtosis`, `mode`, `sem`,
  `zscore`, `zmap`, `trim_mean`, `rankdata` and `entropy`, with `axis`, `nan_policy` and
  `keepdims` where SciPy accepts them.
- **Correlation and tests (`scipy.stats`).** `pearsonr`, `spearmanr`, `linregress`,
  `ttest_1samp`, `ttest_ind`, `ttest_ind_from_stats`, `ttest_rel`, `chisquare`,
  `power_divergence`, and `chi2_contingency` with the rest of `scipy.stats.contingency`
  (`margins`, `expected_freq` and `association`). Results carry SciPy's attributes, the
  `pearsonr` and t-test results provide `confidence_interval`, and warnings use SciPy's
  categories and messages.
- **Linear systems (`scipy.linalg`).** `solve` with every `assume_a` structure, `solve_triangular`,
  `solve_banded`, `solve_circulant`, `inv`, `det`, `lstsq` with each LAPACK driver, `pinv` and
  `pinvh`, over stacked matrices where SciPy batches them.
- **Decompositions (`scipy.linalg`).** `lu`, `lu_factor`, `lu_solve`, `cholesky`, `cho_factor`,
  `cho_solve`, `qr` in every mode and with pivoting, `svd`, `svdvals`, `eigh` and `eigvalsh`
  with generalized problems and subsets, `polar`, `diagsvd`, `orth`, `null_space`,
  `subspace_angles` and `orthogonal_procrustes`.
- **Matrix functions and helpers (`scipy.linalg`).** `expm`, `coshm`, `sinhm`, `tanhm`,
  `khatri_rao`, `norm`, `bandwidth`, `issymmetric`, `ishermitian`, the special matrices
  (`toeplitz`, `circulant`, `hankel`, `hadamard`, `leslie`, `block_diag`, `companion`,
  `helmert`, `hilbert`, `invhilbert`, `pascal`, `invpascal`, `fiedler`, `fiedler_companion`,
  `convolution_matrix` and `dft`), `LinAlgError` and `LinAlgWarning`.
- **LAPACK and BLAS wrappers.** The `s` and `d` forms of `getrf`, `getrs`, `gecon`, `getri`,
  `trtrs`, `trtri`, `potrf`, `potrs`, `potri`, `gtsv`, `gbsv`, `lange` and `nrm2`, with f2py's
  arguments, argument checks and Fortran-ordered outputs, and `get_lapack_funcs`,
  `get_blas_funcs`, `find_best_blas_type` and `find_best_lapack_type`.

## Compatibility contract

The portable suites in `tests/python/scipy/` are the contract. Their literal expectations were
checked against the SciPy and NumPy versions pinned in `tests/python/scientific-requirements.txt`.
Cargo runs them under shellsim's pytest alongside the NumPy suites; see
[numpy.md](numpy.md#compatibility-contract) for re-checking them against real SciPy.
`tests/python/scipy.rs` holds shellsim-only checks for the unsupported frontier and resource
limits.

The exact `scipy.linalg` expectations were measured on the reference machine described above;
on a CPU for which OpenBLAS selects other kernels, SciPy itself rounds the order-12 and
order-70 cases differently. `scipy.special`'s kernels are architecture-independent (plain `f64`
arithmetic, no vendored SIMD sums), so its expectations do not depend on the reference machine.

## Deliberate differences

- **`bdtr` and `bdtrc` warnings.** SciPy warns once for each element with a floating-point `n`;
  shellsim warns once per call. Under the default filters, both show a single warning.
- **Precision of the iterative kernels.** shellsim's incomplete gamma, incomplete beta and
  Hurwitz zeta functions (`igam.rs`, `ibeta.rs`, `zeta.rs`) are independent series and
  continued-fraction implementations, not ports of SciPy's Cephes or Boost.Math code, so they do
  not reproduce SciPy's rounding bit-for-bit. Every function everywhere in `scipy.special`
  targets, and in spot checks over its domain meets, about `1e-13` relative error against SciPy
  1.18.1; most arguments agree to within a few `f64` ULPs (for example `betainc(5, 200, 0.03)`
  differs by about `3e-15` relatively, `stdtrit(7, 0.01)` by about `1e-16`). `gamma` and `binom`
  on small integer arguments, and `poch` on integer `k`, use an exact product instead of the
  general log-domain path and so agree with SciPy exactly. The `_binom_ppf` discrete search can
  differ from SciPy's at the extreme tail (below about `1e-300`) where `cdf` underflows almost
  to zero over several consecutive `k`; there is no simple rule for which `k` either
  implementation's search lands on, so `tests/python/scipy/test_special.py` only pins the exact
  value down to where both agree.
- **Single-precision loops.** SciPy's `float32` loops for the Boost- and xsf-templated
  functions (`erfinv`, `betainc`, `betaincc`, `betaincinv`, `stdtr`, `stdtrit`, `fdtr`, `fdtrc`,
  `fdtri`, `pdtrik`, the binomial functions, `expit`, `logit`, `log_expit`, `xlogy` and
  `xlog1py`) round every intermediate step to single precision; shellsim computes every function
  in `f64` and rounds only the final result (see "Loop selection" above). Both are within
  `float32` precision of the true value, but can differ from each other in the last one or two
  bits, for example `expit` of a `float32` array disagreeing with SciPy in the last mantissa bit
  at one of eight sample points checked during development.
- **`scipy.stats` result objects** unpack, index and compare like SciPy's named tuples, but
  they are not `tuple` instances, because shellsim cannot subclass `tuple`.
- **`scipy.stats` warning locations.** Each statistic calls its axis and NaN handling from a
  thin wrapper instead of a decorator, so a warning whose `stacklevel` SciPy chose for its
  decorated call, such as the precision-loss `RuntimeWarning` from `skew`, is attributed one
  frame deeper than in SciPy.
- **`scipy.stats` log and complement methods.** `sf` computes `1 - cdf(x)`, `isf` computes
  `ppf(1 - q)`, and `logpdf`/`logpmf`, `logcdf` and `logsf` take the logarithm of the
  corresponding non-log method, rather than SciPy's separate closed forms for each. Values agree
  with SciPy's well within the tested tolerances, but deep in a tail, where SciPy's dedicated
  formulas avoid cancellation, shellsim's can lose precision that SciPy's would not.
- **Generic distribution methods.** A user subclass of `rv_continuous` or `rv_discrete` that
  defines only `_cdf` gets a generic `_pdf`/`_pmf` (a central difference for continuous
  distributions, a first difference of `_cdf` for discrete ones), a generic `_ppf` (bisection
  on `_cdf`), and generic `_stats`/`_entropy`, the way SciPy derives the methods it can from
  whichever of `_pdf`, `_cdf` and `_ppf` a subclass provides. The specific numerical methods are
  shellsim's own, not SciPy's, so results can differ from SciPy's in the last several digits
  even though both approximate the same quantity.
- **Eigenvalues and singular values.** `eigh`, `eigvalsh`, `svd` and `svdvals` use Jacobi
  methods where SciPy calls LAPACK's `syevr` and `gesdd`. Their values agree with SciPy's to
  about `1e-15` relatively, and eigenvectors and singular vectors can differ in sign. `lstsq`
  computes its solution from the SVD for every driver, and `pinv`, `pinvh`, `polar`,
  `null_space`, `orth` and `subspace_angles` build on the SVD or `eigh`, so their results can
  differ from SciPy's in the last bits, and a zero singular value can be exactly zero where
  SciPy's is near `1e-16`.
- **`lstsq.default_lapack_driver`** does not exist, because shellsim's functions cannot hold
  attributes. The default driver is still `gelsd`.
- **Output layout of batched results.** The f2py wrappers, and the functions that return their
  results, return Fortran-ordered arrays as SciPy's do. For stacked input, `eigh` and
  `qr(mode="raw")` return C-ordered arrays where SciPy's are not contiguous.

## Unsupported frontier

These fail explicitly with shellsim's unsupported-operation error or `NotImplementedError`:

- complex input to functions that have complex loops in SciPy, such as `erf`, `gamma`,
  `loggamma`, `psi`, `xlogy` and `zeta`. Functions without complex loops raise SciPy's
  `TypeError`;
- `reduce` and `accumulate` of `scipy.special` ufuncs;
- complex `n` in `factorial`, `factorial2` and `factorialk` with `extend="complex"`, which
  reaches the unsupported complex `gamma`;
- `nan_policy='omit'` with NaN input in `describe` and `spearmanr`, which SciPy computes with
  masked arrays;
- the resampling `method` objects (`PermutationMethod`, `MonteCarloMethod` and
  `BootstrapMethod`) and `chi2_contingency(..., method=...)`;
- `rv_discrete(values=...)`, which needs `__new__` to return a different class;
- distribution methods that SciPy computes by numerical integration or root finding, such as
  the generic `fit`, continuous `expect`, and `moment` orders beyond those `stats` provides.
  They import `scipy.integrate` or `scipy.optimize` when called and fail there;
- the other names in SciPy's `scipy.stats.__all__`, such as `gamma` or `mannwhitneyu`;
- complex input to `scipy.linalg` functions, and the complex (`c` and `z`) LAPACK and BLAS
  routines;
- the other names in SciPy's `scipy.linalg.__all__`, such as `eig`, `schur`, `logm`, `sqrtm` and
  `solveh_banded`, and the LAPACK and BLAS routines not listed above, such as `dgeev` or
  `ddot`;
- importing any subpackage other than `scipy.special`, `scipy.stats` and `scipy.linalg`.

Other `scipy.special` functions and `scipy.stats` names are absent, so accessing them raises
`AttributeError`.

## Safety and accounting

Every element is charged a flat CPU cost. Series, continued fractions and root finders can need
from a few to millions of iterations, depending on the arguments. The Hurwitz zeta sum for a
negative non-integer `q`, for example, runs about `|q|` terms, as SciPy's does. So the kernels
also count their iterations, and each element is charged for the iterations it ran.

An element runs under an allowance of iterations. If it exhausts the allowance, shellsim charges
the work done and evaluates the element again with twice the allowance. Work per element is
therefore bounded by the CPU limit rather than by a fixed cap. Kernels are pure functions, so a
repeated evaluation gives the same result, and the repeated work is at most the work of the
final evaluation.

`scipy.stats` is Python code over NumPy and `scipy.special`, so the interpreter and those
kernels meter its work.

`scipy.linalg` charges each factorization, solve, inverse and matrix function for its work
before running it: cubic in the matrix order, times the number of stacked matrices, plus the
right-hand sides of a solve. `expm` adds one matrix product per squaring. Working copies are
reserved against the memory limit first. The Jacobi methods behind `svd` and `eigh` are charged
per sweep by `numpy.linalg`.
