//! End-to-end tests for importing trusted host project inputs into a fresh simulated VFS.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use shellsim::Environment;

static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);

struct TestDirectory(PathBuf);

impl TestDirectory {
    fn new() -> Self {
        let sequence = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "shellsim-python-test-{}-{sequence}",
            std::process::id()
        ));
        std::fs::create_dir(&path).expect("create test project");
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn file_mode_mounts_sibling_modules_and_forwards_arguments() {
    let project = TestDirectory::new();
    std::fs::write(
        project.path().join("helper.py"),
        "def answer():\n    return 42\n",
    )
    .expect("write helper");
    std::fs::write(
        project.path().join("main.py"),
        "import helper\nimport sys\nprint(helper.answer(), sys.argv[1])\n",
    )
    .expect("write script");

    let output = Command::new(env!("CARGO_BIN_EXE_shellsim-python"))
        .arg(project.path().join("main.py"))
        .arg("sample")
        .output()
        .expect("run shellsim-python");

    assert!(output.status.success(), "{:?}", output.stderr);
    assert_eq!(output.stdout, b"42 sample\n");
    assert!(output.stderr.is_empty());
}

#[test]
fn mounted_packages_resolve_relative_and_parenthesized_imports() {
    let project = TestDirectory::new();
    let package = project.path().join("sample");
    std::fs::create_dir(&package).expect("create package");
    std::fs::write(package.join("__init__.py"), "").expect("write package initializer");
    std::fs::write(package.join("values.py"), "left = 20\nright = 22\n")
        .expect("write package values");
    std::fs::write(
        package.join("answer.py"),
        "from .values import (left, right,)\nvalue = left + right\n",
    )
    .expect("write package module");
    std::fs::write(
        project.path().join("main.py"),
        "from sample.answer import value\nprint(value)\n",
    )
    .expect("write script");

    let output = Command::new(env!("CARGO_BIN_EXE_shellsim-python"))
        .arg(project.path().join("main.py"))
        .output()
        .expect("run shellsim-python");

    assert!(output.status.success(), "{:?}", output.stderr);
    assert_eq!(output.stdout, b"42\n");
    assert!(output.stderr.is_empty());
}

#[test]
fn directory_mode_discovers_python_tests() {
    let project = TestDirectory::new();
    std::fs::write(
        project.path().join("test_sample.py"),
        "def test_value():\n    assert 2 + 2 == 4\n",
    )
    .expect("write test");

    let output = Command::new(env!("CARGO_BIN_EXE_shellsim-python"))
        .arg(project.path())
        .output()
        .expect("run shellsim-python");

    assert!(output.status.success(), "{:?}", output.stderr);
    assert_eq!(output.stdout, b"/work/test_sample.py::test_value PASSED\n");
    assert!(output.stderr.is_empty());
}

#[test]
fn json_mode_reports_unsupported_behavior() {
    let project = TestDirectory::new();
    std::fs::write(project.path().join("main.py"), "import threading\n").expect("write script");

    let output = Command::new(env!("CARGO_BIN_EXE_shellsim-python"))
        .arg("--json")
        .arg(project.path().join("main.py"))
        .output()
        .expect("run shellsim-python");
    let report: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("parse JSON report");

    assert_eq!(output.status.code(), Some(2));
    assert_eq!(report["outcome"]["exit_status"], 2);
    assert_eq!(report["mounted_files"], 1);
    assert!(report["unsupported"][0]
        .as_str()
        .is_some_and(|value| value.contains("threading")));
}

#[test]
fn project_ingestion_obeys_the_vfs_disk_limit() {
    let project = TestDirectory::new();
    std::fs::write(
        project.path().join("main.py"),
        "print('this source is deliberately larger than one byte')\n",
    )
    .expect("write script");

    let output = Command::new(env!("CARGO_BIN_EXE_shellsim-python"))
        .args(["--disk", "1"])
        .arg(project.path().join("main.py"))
        .output()
        .expect("run shellsim-python");

    assert_eq!(output.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&output.stderr).contains("larger than the configured VFS"));
    assert!(output.stdout.is_empty());
}

#[test]
fn project_ingestion_reports_skipped_dependency_directories() {
    let project = TestDirectory::new();
    std::fs::create_dir(project.path().join(".venv")).expect("create skipped directory");
    std::fs::write(project.path().join(".venv/config"), "host-only").expect("write skipped file");
    std::fs::write(project.path().join("main.py"), "print('safe')\n").expect("write script");
    let mut environment = Environment::new();

    let report =
        shellsim::host_ingest::mount_host_tree_report(&mut environment, project.path(), "/work")
            .expect("mount project");

    assert_eq!(report.files, 1);
    assert_eq!(report.skipped_directories, [".venv"]);
    assert!(environment.vfs.exists("/", "/work/main.py"));
    assert!(!environment.vfs.exists("/", "/work/.venv/config"));
}

#[cfg(unix)]
#[test]
fn project_ingestion_rejects_host_symlinks() {
    use std::os::unix::fs::symlink;

    let project = TestDirectory::new();
    std::fs::write(project.path().join("main.py"), "print('safe')\n").expect("write script");
    symlink("main.py", project.path().join("alias.py")).expect("create symlink");

    let output = Command::new(env!("CARGO_BIN_EXE_shellsim-python"))
        .arg(project.path().join("main.py"))
        .output()
        .expect("run shellsim-python");

    assert_eq!(output.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&output.stderr).contains("refusing host symlink"));

    let mut environment = Environment::new();
    let error = shellsim::host_ingest::mount_host_tree(&mut environment, project.path(), "/work")
        .unwrap_err();
    assert!(error.contains("refusing host symlink"));
    assert!(!environment.vfs.exists("/", "/work/main.py"));
}
