//! Portable NumPy suites executed unchanged by shellsim's pytest.
//!
//! Each `tests/python/numpy/test_*.py` file states literal expectations that were checked against
//! the versions pinned in `tests/python/scientific-requirements.txt`. Cargo always runs the
//! suites under shellsim, so CI enforces those expectations offline. Re-checking them against
//! real NumPy is an optional step: build the pinned environment with
//! `infra/scientific-reference.py` and export its interpreter as `SHELLSIM_SCIENTIFIC_PYTHON`.
//! Tests never resolve or install packages themselves.
//!
//! Suites whose behavior lands in a later phase of the typed-array migration stay ignored with a
//! reason naming that phase; the reference check still covers them.

use std::path::{Path, PathBuf};
use std::process::Command;

use shellsim::Environment;

const REFERENCE_PYTHON: &str = "SHELLSIM_SCIENTIFIC_PYTHON";

fn suite_directory() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/python/numpy")
}

fn assert_numpy_suite(file: &str) {
    let source = std::fs::read(suite_directory().join(file)).expect("read NumPy suite");
    let simulated_path = format!("/tests/numpy/{file}");
    let mut environment = Environment::new();
    environment
        .vfs
        .put_file(&simulated_path, source, 0o644)
        .expect("install NumPy suite");

    let (outcome, stdout, stderr) =
        environment.run_script_capture(&format!("python3.14 -m pytest {simulated_path}"));
    assert_eq!(
        outcome.exit_status,
        0,
        "shellsim suite {file} failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&stdout),
        String::from_utf8_lossy(&stderr),
    );
    assert!(stderr.is_empty(), "{}", String::from_utf8_lossy(&stderr));
}

/// Run every suite, including pending ones, under the pinned reference interpreter when one is
/// configured. Without `SHELLSIM_SCIENTIFIC_PYTHON` this test checks nothing.
#[test]
fn reference_python_passes_every_suite() {
    let Some(python) = std::env::var_os(REFERENCE_PYTHON) else {
        return;
    };
    let output = Command::new(&python)
        .args(["-m", "pytest", "-q", "-p", "no:cacheprovider"])
        .arg(suite_directory())
        .output()
        .unwrap_or_else(|error| panic!("could not run {python:?}: {error}"));
    assert!(
        output.status.success(),
        "reference NumPy rejected the suites:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}

macro_rules! numpy_suites {
    ($($(#[$attribute:meta])* $name:ident => $file:literal;)*) => {
        $(
            #[test]
            $(#[$attribute])*
            fn $name() {
                assert_numpy_suite($file);
            }
        )*

        /// Every suite file on disk is registered above, so a new file cannot be silently
        /// skipped by Cargo.
        #[test]
        fn every_suite_file_is_registered() {
            let registered = [$($file),*];
            let mut on_disk = std::fs::read_dir(suite_directory())
                .expect("list NumPy suites")
                .map(|entry| entry.expect("read suite entry").file_name())
                .filter_map(|name| name.into_string().ok())
                .filter(|name| name.starts_with("test_") && name.ends_with(".py"))
                .collect::<Vec<_>>();
            on_disk.sort();
            let mut registered = registered.map(String::from).to_vec();
            registered.sort();
            assert_eq!(on_disk, registered);
        }
    };
}

numpy_suites! {
    #[ignore = "pending: typed NumPy core (phase 2)"]
    construction => "test_construction.py";
    #[ignore = "pending: typed NumPy core (phase 2)"]
    dtypes => "test_dtypes.py";
    #[ignore = "pending: NumPy scalar protocol (phase 3)"]
    scalars => "test_scalars.py";
    #[ignore = "pending: typed NumPy core (phase 2)"]
    indexing => "test_indexing.py";
    #[ignore = "pending: NumPy ufunc table (phase 3)"]
    ufuncs => "test_ufuncs.py";
    #[ignore = "pending: NumPy errstate and warnings (phase 3)"]
    errstate => "test_errstate.py";
    #[ignore = "pending: NumPy surface completion (phase 4)"]
    reductions => "test_reductions.py";
    #[ignore = "pending: NumPy surface completion (phase 4)"]
    shape => "test_shape.py";
    #[ignore = "pending: NumPy surface completion (phase 4)"]
    sorting => "test_sorting.py";
    #[ignore = "pending: NumPy surface completion (phase 4)"]
    strings_objects => "test_strings_objects.py";
    #[ignore = "pending: NumPy surface completion (phase 4)"]
    printing => "test_printing.py";
    #[ignore = "pending: NumPy surface completion (phase 4)"]
    linalg => "test_linalg.py";
    #[ignore = "pending: NumPy surface completion (phase 4)"]
    fft => "test_fft.py";
    #[ignore = "pending: NumPy surface completion (phase 4)"]
    random => "test_random.py";
    #[ignore = "pending: NumPy surface completion (phase 4)"]
    io => "test_io.py";
    #[ignore = "pending: NumPy surface completion (phase 4)"]
    testing => "test_testing.py";
}
