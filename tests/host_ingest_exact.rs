//! Exact package imports keep virtual-environment files that project imports deliberately skip.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use shellsim::host_ingest::{mount_host_tree_report, mount_host_tree_report_exact};
use shellsim::Environment;

static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);

struct TestDirectory(PathBuf);

impl TestDirectory {
    fn new() -> Self {
        let sequence = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "shellsim-exact-host-import-test-{}-{sequence}",
            std::process::id()
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn exact_package_import_keeps_venv_and_default_project_import_skips_it() {
    let source = TestDirectory::new();
    let scripts = source.path().join("work/.venv/bin");
    fs::create_dir_all(&scripts).unwrap();
    fs::write(scripts.join("pytest"), b"guest script").unwrap();

    let mut packages = Environment::new();
    let report = mount_host_tree_report_exact(&mut packages, source.path(), "/").unwrap();
    assert_eq!(report.files, 1);
    assert_eq!(
        packages.vfs.read("/", "/work/.venv/bin/pytest").unwrap(),
        b"guest script"
    );

    let mut project = Environment::new();
    let report = mount_host_tree_report(&mut project, source.path(), "/").unwrap();
    assert_eq!(report.skipped_directories, [".venv"]);
    assert!(!project.vfs.exists("/", "/work/.venv/bin/pytest"));
}

#[test]
fn exact_package_import_rejects_git_metadata_without_vfs_mutation() {
    let source = TestDirectory::new();
    fs::create_dir(source.path().join(".git")).unwrap();
    fs::write(source.path().join("package.py"), b"value = 1\n").unwrap();
    let mut environment = Environment::new();
    assert!(mount_host_tree_report_exact(&mut environment, source.path(), "/work").is_err());
    assert!(!environment.vfs.exists("/", "/work/package.py"));
}
