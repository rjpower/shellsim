//! Shellsim-specific `scipy` behavior that the portable suites in `tests/python/scipy` cannot
//! check against SciPy: the explicit unsupported frontier and CPU metering of kernels whose
//! work depends on their arguments or grows faster than their input.

use shellsim::{python, Environment, Limits};

fn run(cpu: u64, source: &str) -> (i32, String) {
    let mut environment = Environment::with_limits(Limits {
        cpu,
        ..Limits::default()
    });
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let source = format!("import numpy as np\nfrom scipy import special\n{source}");
    let status = python::run_python(
        &mut environment,
        &["python3.14".into(), "-c".into(), source],
        Vec::new(),
        &mut stdout,
        &mut stderr,
    );
    (status, String::from_utf8_lossy(&stderr).into_owned())
}

#[test]
fn unsupported_scipy_features_fail_explicitly() {
    for (source, expected) in [
        (
            "special.erf(np.array([1j]))",
            "complex input to scipy.special.erf is not supported by shellsim's SciPy",
        ),
        (
            "special.gammainc.reduce(np.array([1.0, 2.0]))",
            "reduce and accumulate of scipy.special.gammainc are not supported by shellsim's SciPy",
        ),
        (
            "special.factorial(1 + 1j, extend='complex')",
            "complex input to scipy.special.gamma is not supported by shellsim's SciPy",
        ),
        ("import scipy.sparse", "ModuleNotFoundError"),
        // No public rv_continuous/rv_discrete subclassing, fitting or moment-above-variance
        // framework: `scipy.stats` distributions are plain classes exposing only the kept
        // methods (see src/python/stdlib/source/scipy/stats/_distributions.py), so anything
        // else fails through Python's own attribute lookup.
        (
            "from scipy import stats\nstats.t.fit([1.0, 2.0])",
            "AttributeError",
        ),
        (
            "from scipy import stats\nstats.t.moment(5, 7)",
            "AttributeError",
        ),
        (
            "from scipy import stats\nstats.rv_discrete",
            "AttributeError",
        ),
        ("from scipy import stats\nstats.gamma", "AttributeError"),
        (
            "from scipy import stats\nstats.describe([1, np.nan], nan_policy='omit')",
            "describe(..., nan_policy='omit') with NaN input is not supported by shellsim's SciPy",
        ),
        (
            "from scipy import stats\nstats.chi2_contingency([[1, 2], [3, 4]], method=1)",
            "chi2_contingency(..., method=...) is not supported by shellsim's SciPy",
        ),
        (
            "import scipy.linalg as sl\nsl.solveh_banded",
            "AttributeError",
        ),
        (
            "import scipy.linalg as sl\nsl.solve(np.eye(2) + 1j, np.ones(2))",
            "complex input to scipy.linalg.solve is not supported by shellsim's SciPy",
        ),
        (
            "import scipy.linalg as sl\nsl.eigh(np.eye(2) * 1j)",
            "complex input to scipy.linalg.eigh is not supported by shellsim's SciPy",
        ),
        (
            "from scipy.linalg import lapack\nlapack.zgetrf",
            "NotImplementedError: LAPACK routine zgetrf is not supported by shellsim's SciPy",
        ),
        (
            "from scipy.linalg import lapack\nlapack.dgeev",
            "NotImplementedError: LAPACK routine dgeev is not supported by shellsim's SciPy",
        ),
        (
            "from scipy.linalg import blas\nblas.ddot",
            "NotImplementedError: BLAS routine ddot is not supported by shellsim's SciPy",
        ),
        ("from scipy.spatial import KDTree", "ImportError"),
        (
            "from scipy.spatial import distance\ndistance.dice",
            "AttributeError",
        ),
        ("from scipy.interpolate import BSpline", "ImportError"),
        (
            "from scipy.interpolate import CubicSpline\nCubicSpline([0, 1, 2], [1, -1, 1]).roots()",
            "NotImplementedError: PPoly.roots is not supported by shellsim's SciPy",
        ),
        (
            "from scipy.spatial import distance\ndistance.pdist([[1.0], [2.0]], 'yule')",
            "ValueError",
        ),
    ] {
        let (status, stderr) = run(Limits::default().cpu, source);
        assert_ne!(status, 0, "{source} unexpectedly succeeded");
        assert!(stderr.contains(expected), "{source}: {stderr:?}");
    }
}

#[test]
fn iterative_kernels_are_charged_for_the_iterations_they_run() {
    let cpu = 20_000_000;
    // Incomplete beta evaluations at small parameters converge in a few iterations.
    let light = "a = np.full(2000, 2.0)\nassert special.betainc(a, 3.0, 0.5)[0] == 0.6875";
    assert_eq!(run(cpu, light), (0, String::new()));
    // At a = b = 3e10 each takes about 17,700 iterations, which the same budget cannot cover.
    let heavy = "a = np.full(2000, 3e10)\nspecial.betainc(a, a, 0.5)";
    assert_eq!(run(cpu, heavy), (137, String::new()));
    // The Hurwitz zeta sum for a negative q runs about |q| terms, here 4e15, as SciPy's does.
    let unbounded = "special.zeta(2.0, -4e15 + 0.5)";
    assert_eq!(run(cpu, unbounded), (137, String::new()));
}

#[test]
fn linear_solves_are_charged_for_their_cubic_work() {
    let cpu = 35_000_000;
    // Order 200 factors twice (`getrf`, then `gecon` for the ill-conditioning check `solve`
    // always runs), each about 16 million multiply-adds, plus a cheap `getrs`: about 32 million.
    let small =
        "import scipy.linalg as sl\nassert sl.solve(np.eye(200) * 2.0, np.ones(200))[0] == 0.5";
    assert_eq!(run(cpu, small), (0, String::new()));
    // Order 400 takes about 256 million the same way, which the same budget cannot cover.
    let large = "import scipy.linalg as sl\nsl.solve(np.eye(400) * 2.0, np.ones(400))";
    assert_eq!(run(cpu, large), (137, String::new()));
}
