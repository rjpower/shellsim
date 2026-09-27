# SciPy in shellsim

shellsim ships `scipy.special` and `scipy.stats` for simulated Python programs. It targets the
observable behavior of SciPy 1.18.1 with NumPy 2.5.3: values, result dtypes, error types and
messages, and warnings. It never runs host SciPy. The other SciPy subpackages raise
`NotImplementedError` when imported.

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
- **`scipy.stats`** is frozen Python ported from SciPy's own modules: the `rv_continuous` and
  `rv_discrete` framework, `_axis_nan_policy`, `_stats_py`, `_entropy` and `contingency`,
  computing with NumPy and `scipy.special`. SciPy writes the statistics against the array API;
  the port keeps their NumPy branch. SciPy uses `inspect` to find a distribution's shape
  parameters from the signatures of `_pdf` and `_cdf`, and to add `axis`, `nan_policy` and
  `keepdims` to its statistics. shellsim has no `inspect`, so the port reads parameters through
  a private module, `_shellsim_introspect`, and each statistic declares SciPy's full signature
  itself.

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
- importing any subpackage other than `scipy.special` and `scipy.stats`.

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
