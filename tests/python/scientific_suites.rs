//! Portable NumPy and SciPy suites executed unchanged by shellsim's pytest.
//!
//! Each `tests/python/{numpy,scipy}/test_*.py` file states literal expectations that were checked
//! against the versions pinned in `tests/python/scientific-requirements.txt`. Cargo always runs
//! the suites under shellsim, so CI enforces those expectations offline. Re-checking them against
//! real NumPy and SciPy is an optional step: build the pinned environment with
//! `infra/scientific-reference.py` and export its interpreter as `SHELLSIM_SCIENTIFIC_PYTHON`.
//! Tests never resolve or install packages themselves.

use std::path::{Path, PathBuf};
use std::process::Command;

use shellsim::Environment;

const REFERENCE_PYTHON: &str = "SHELLSIM_SCIENTIFIC_PYTHON";

const SUITE_DIRECTORIES: [&str; 2] = ["numpy", "scipy"];

fn suite_directory(package: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/python")
        .join(package)
}

fn assert_suite(package: &str, file: &str) {
    let source = std::fs::read(suite_directory(package).join(file)).expect("read suite");
    let simulated_path = format!("/tests/{package}/{file}");
    let mut environment = Environment::new();
    environment
        .vfs
        .put_file(&simulated_path, source, 0o644)
        .expect("install suite");

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
        .args([
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            // numpy/ and scipy/ both hold a test_linalg.py; importlib mode imports each file
            // by path so the two basenames do not collide.
            "--import-mode=importlib",
        ])
        .args(SUITE_DIRECTORIES.map(suite_directory))
        .output()
        .unwrap_or_else(|error| panic!("could not run {python:?}: {error}"));
    assert!(
        output.status.success(),
        "the reference environment rejected the suites:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}

macro_rules! suites {
    ($module:ident, $package:literal, { $($(#[$attribute:meta])* $name:ident => $file:literal;)* }) => {
        mod $module {
            use super::*;

            $(
                #[test]
                $(#[$attribute])*
                fn $name() {
                    assert_suite($package, $file);
                }
            )*

            /// Every suite file on disk is registered above, so a new file cannot be silently
            /// skipped by Cargo.
            #[test]
            fn every_suite_file_is_registered() {
                let registered = [$($file),*];
                let mut on_disk = std::fs::read_dir(suite_directory($package))
                    .expect("list suites")
                    .map(|entry| entry.expect("read suite entry").file_name())
                    .filter_map(|name| name.into_string().ok())
                    .filter(|name| name.starts_with("test_") && name.ends_with(".py"))
                    .collect::<Vec<_>>();
                on_disk.sort();
                let mut registered = registered.map(String::from).to_vec();
                registered.sort();
                assert_eq!(on_disk, registered);
            }
        }
    };
}

suites!(numpy, "numpy", {
    construction => "test_construction.py";
    dtypes => "test_dtypes.py";
    scalars => "test_scalars.py";
    indexing => "test_indexing.py";
    ufuncs => "test_ufuncs.py";
    errstate => "test_errstate.py";
    reductions => "test_reductions.py";
    shape => "test_shape.py";
    sorting => "test_sorting.py";
    strings_objects => "test_strings_objects.py";
    printing => "test_printing.py";
    linalg => "test_linalg.py";
    fft => "test_fft.py";
    random => "test_random.py";
    io => "test_io.py";
    testing => "test_testing.py";
});

suites!(scipy, "scipy", {
    special => "test_special.py";
    stats => "test_stats.py";
    linalg => "test_linalg.py";
    integrate => "test_integrate.py";
    interpolate => "test_interpolate.py";
    spatial => "test_spatial.py";
});
