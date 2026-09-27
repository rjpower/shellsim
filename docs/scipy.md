# SciPy in shellsim

shellsim ships `scipy.special`, `scipy.stats` and `scipy.linalg` for simulated Python programs.
It targets the observable behavior of SciPy 1.18.1 with NumPy 2.5.3: values, result dtypes,
error types and messages, and warnings. It never runs host SciPy. The other SciPy subpackages
raise `NotImplementedError` when imported.

## Architecture

- **Ufuncs.** Each `scipy.special` function is an entry in NumPy's ufunc table (see
  [numpy.md](numpy.md)), so broadcasting, `out=`, `where=`, `dtype=`, casting and scalar results
  behave as they do for NumPy's ufuncs. The native module `_scipy_special` exports them.
- **Kernels** (`src/python/stdlib/scipy/special/`). The kernels are ports of the code SciPy 1.18
  runs. Most functions come from Cephes, through SciPy's xsf library. `erfinv`, the incomplete
  beta function and its inverse, the Student's t, F and binomial distributions, and `pdtrik`
  come from Boost.Math, including its polynomial evaluation and root finders, so that results
  round as SciPy's do. `NOTICE.md` lists the ported components and reproduces their licenses.
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
precision. `expit`, `logit`, `log_expit`, `xlogy` and `xlog1py` round every step to single
precision, as xsf's templates do. The other single-precision loops compute in `f64` and round
the result, as SciPy's Cephes-based loops do; see the deliberate differences for the Boost-based
ones.

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

The reference is SciPy's x86-64 build. There Boost evaluates its Lanczos sums with SSE2
instructions, which shellsim reproduces; SciPy's ARM builds round those sums differently, so
their Boost-based results can differ from shellsim's and from each other in the last place.
The exact `scipy.linalg` expectations were measured on the reference machine described above;
on a CPU for which OpenBLAS selects other kernels, SciPy itself rounds the order-12 and
order-70 cases differently.

## Deliberate differences

- **`bdtr` and `bdtrc` warnings.** SciPy warns once for each element with a floating-point `n`;
  shellsim warns once per call. Under the default filters, both show a single warning.
- **`bdtr` and `bdtrc` precision.** shellsim evaluates both through Boost's incomplete beta
  function, where SciPy uses Cephes' `incbet`. The results agree less closely as `n` grows; at
  `n = 1000` their relative difference reaches about `7e-13`.
- **Single precision in the Boost-based functions.** For `float32` input, SciPy runs Boost's
  `erfinv`, `betainc`, `betaincc`, `betaincinv`, `stdtr`, `stdtrit`, `fdtr`, `fdtrc`, `fdtri`,
  `pdtrik` and binomial functions in single precision. shellsim computes them in double
  precision and rounds the result, which is at least as accurate but differs from SciPy's
  `float32` result in the last place for about half of all arguments.
- **Gamma functions inside the Boost ports.** Where Boost's incomplete beta, t, F and Poisson
  code calls Boost's own `tgamma`, `lgamma`, `erfc` and incomplete gamma functions, shellsim
  calls their Cephes ports. `erfinv` and the binomial functions agree with SciPy exactly; for
  `betainc`, `betaincinv`, `stdtr`, `stdtrit`, `fdtr` and `fdtri`, about 5% to 40% of
  results differ in the last few places, by at most about `5e-14` relatively. `pdtrik` solves
  for a root of a function that can be very flat, which magnifies the difference to about
  `1e-11`.
- **`betaincc` in extended precision.** SciPy calls Boost's `ibetac` with Boost's default
  policy, which on x86-64 computes in 80-bit `long double`. shellsim computes Boost's algorithm
  in double precision, so most results differ in the last place, and results below about
  `1e-280` lose up to half their digits or underflow to zero where SciPy's do not. The
  `pearsonr` p-value goes through `betaincc`, so it can differ from SciPy's in the last place.
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
