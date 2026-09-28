# SciPy in shellsim

shellsim ships a small SciPy subset for simulated Python programs: `scipy.special`, `stats`,
`linalg`, `integrate`, `optimize`, `interpolate` and `spatial.distance`. It never runs host SciPy.
Each public module is one frozen Python file under `src/python/stdlib/source/scipy/`. Only
`scipy.special` has its own native kernels; everything else is Python over NumPy.

`import scipy` names the supported subpackages and imports each one on first attribute access, so
both `import scipy.stats` and `scipy.stats` after `import scipy` work.

## Supported subset

- **`scipy.special`.**
  - Native kernels: `gamma`, `gammaln`, `loggamma`, `digamma`/`psi`, `erf`, `erfc`, `erfinv`,
    `erfcinv`, `ndtr`, `ndtri` and `log_ndtr`; the regularized incomplete gamma and beta functions
    (`gammainc`, `gammaincc`, `betainc`, `betaincc`) and their inverses; and `zeta`. They are
    entries in NumPy's ufunc table, so they broadcast and accept `out=`, `where=` and `dtype=`.
  - Python compositions: `beta`, `betaln`, `binom`, `comb`, `factorial`, `expit`, `logit`, `xlogy`,
    `xlog1py`, `entr`, `rel_entr`, `kl_div`, `boxcox`, `inv_boxcox`, `logsumexp`, `softmax`,
    `expm1` and `log1p`.
  - The kernels implement published algorithms: the Lanczos approximation, series and continued
    fractions for the incomplete functions, Acklam's approximation refined by Halley's method for
    `ndtri`, and Euler-Maclaurin summation for `zeta`.
- **`scipy.stats`.**
  - Distributions: `norm`, `t`, `chi2`, `f`, `expon`, `uniform`, `binom` and `poisson`.
  - Methods: `pdf`/`pmf`, `logpdf`/`logpmf`, `cdf`, `logcdf`, `sf`, `logsf`, `ppf`, `isf`,
    `mean`, `var`, `std`, `median`, `interval` and `rvs`.
  - Continuous distributions take `loc` and `scale`, and discrete ones take `loc`. Calling a
    distribution freezes its parameters, as in `norm(0, 2).pdf(x)`. `rvs` draws through
    `numpy.random.default_rng(random_state)`.
  - Statistics: `describe`, `mode`, `moment`, `skew`, `kurtosis`, `sem`, `zscore`, `zmap`,
    `trim_mean` and `rankdata`.
  - Tests: `pearsonr`, `spearmanr`, `linregress`, `ttest_1samp`, `ttest_ind`, `ttest_rel`,
    `chisquare` and `chi2_contingency`.
  - Results unpack like tuples and expose named fields.
- **`scipy.linalg`.**
  - Functions: `solve`, `solve_triangular`, `solve_banded`, `inv`, `det`, `lu`, `lu_factor`,
    `lu_solve`, `cholesky`, `cho_factor`, `cho_solve`, `qr`, `svd`, `svdvals`, `eig`, `eigvals`,
    `eigh`, `eigvalsh`, `lstsq`, `pinv`, `norm`, `expm`, `null_space` and `orth`.
  - Special matrices: `toeplitz`, `circulant`, `block_diag` and `hilbert`.
  - Everything is built on the `numpy.linalg` kernels, and input is real (float64).
- **`scipy.integrate`.**
  - Quadrature: adaptive Gauss-Kronrod `quad`, which handles infinite limits and warns
    `IntegrationWarning` when it misses the tolerance, and `dblquad`.
  - Sampled data: `trapezoid`, `cumulative_trapezoid` and `simpson`.
  - ODE solvers: `solve_ivp` with the RK45 method and `t_eval`, and `odeint`.
- **`scipy.optimize`.**
  - Minimization: `minimize` with Nelder-Mead or BFGS, and `minimize_scalar` (Brent, or bounded).
  - Root finding: `root_scalar`, `brentq`, `bisect` and `newton`.
  - Fitting: `curve_fit`, by Levenberg-Marquardt.
  - Results are `OptimizeResult` objects.
- **`scipy.interpolate`.** `interp1d` (kinds up to cubic), `CubicSpline`, `PchipInterpolator`,
  `pchip_interpolate`, `CubicHermiteSpline` and `PPoly`.
- **`scipy.spatial.distance`.**
  - Metrics: `euclidean`, `sqeuclidean`, `cityblock`, `chebyshev`, `minkowski`, `cosine`,
    `correlation`, `hamming`, `jaccard`, `canberra`, `braycurtis`, `mahalanobis`, `seuclidean`
    and `jensenshannon`.
  - Batch distances: `pdist` and `cdist`, taking a metric name or a callable.
  - Conversion: `squareform`.

## Compatibility contract

Results agree with SciPy 1.18.1 within numerical tolerance. Shapes, dtypes, exception types and
warning categories match. Error and warning messages are shellsim's own. Optimizers and
integrators reach the same solutions, not the same iteration counts or diagnostic fields.

The portable suites in `tests/python/scipy/` state this contract. Cargo runs them under shellsim
(see `tests/python/scientific_suites.rs`). To re-check them against real SciPy, build the pinned
environment with `infra/scientific-reference.py` and export its interpreter as
`SHELLSIM_SCIENTIFIC_PYTHON`. `tests/python/scipy.rs` holds the shellsim-only checks: the
unsupported frontier and CPU accounting.

## Unsupported frontier

- **Other subpackages.** Importing any other subpackage, such as `scipy.sparse`, `signal`, `fft`,
  `ndimage` or `cluster`, raises `ModuleNotFoundError`. An `except ImportError` fallback therefore
  works.
- **Other public names.** Names outside the subset above raise `AttributeError` or `ImportError`.
  This covers other distributions such as `stats.gamma`, `scipy.linalg.lapack`, `KDTree` and
  `BSpline`.
- **Unsupported options** of supported functions raise `NotImplementedError`. Examples:
  - `solve_ivp` methods other than RK45;
  - `minimize` bounds and constraints;
  - `qr(pivoting=True)`;
  - `interp1d` spline orders above cubic.
- **Complex input** to `scipy.linalg` and to the `scipy.special` kernels raises
  `NotImplementedError`.
- **Special-function ufunc methods.** The `reduce` and `accumulate` methods of the
  `scipy.special` ufuncs are not supported.

## Safety and accounting

Each `scipy.special` element is charged for the iterations its kernel runs. An element starts
with an allowance of iterations; if it exhausts the allowance, shellsim charges the work done and
evaluates the element again with twice the allowance. Work is therefore bounded by the CPU limit
rather than by a fixed iteration cap. Kernels are pure functions, so the repeated work is at most
the work of the final evaluation.

The rest of SciPy is Python. The interpreter meters its loops, and the NumPy kernels it calls
charge for their own work (see [numpy.md](numpy.md)).
