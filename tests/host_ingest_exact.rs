//! Exact package imports keep virtual-environment files that project imports deliberately skip.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use shellsim::host_ingest::{mount_host_tree_report, mount_package_tree};
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
    let report = mount_package_tree(&mut packages, source.path(), "/", &[]).unwrap();
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
    assert!(mount_package_tree(&mut environment, source.path(), "/work", &[]).is_err());
    assert!(!environment.vfs.exists("/", "/work/package.py"));
}

#[test]
fn package_conflicts_roll_back_and_identical_files_preserve_identity() {
    let source = TestDirectory::new();
    fs::write(source.path().join("a-new"), b"new").unwrap();
    fs::write(source.path().join("z-existing"), b"same").unwrap();
    let mut environment = Environment::new();
    environment
        .vfs
        .put_file("/work/z-existing", b"different".to_vec(), 0o644)
        .unwrap();
    assert!(mount_package_tree(&mut environment, source.path(), "/work", &[]).is_err());
    assert!(!environment.vfs.exists("/", "/work/a-new"));
    assert_eq!(
        environment.vfs.read("/", "/work/z-existing").unwrap(),
        b"different"
    );
    environment
        .vfs
        .put_file("/work/z-existing", b"same".to_vec(), 0o644)
        .unwrap();
    let inode = environment
        .vfs
        .metadata("/", "/work/z-existing", false)
        .unwrap()
        .inode;
    mount_package_tree(&mut environment, source.path(), "/work", &[]).unwrap();
    assert_eq!(
        environment
            .vfs
            .metadata("/", "/work/z-existing", false)
            .unwrap()
            .inode,
        inode
    );
    environment
        .vfs
        .chmod("/", "/work/z-existing", 0o600)
        .unwrap();
    assert!(mount_package_tree(&mut environment, source.path(), "/work", &[]).is_err());
}

#[test]
fn package_import_rejects_links_and_preserves_existing_directories() {
    let source = TestDirectory::new();
    fs::create_dir(source.path().join("data")).unwrap();
    fs::write(source.path().join("data/value"), b"value").unwrap();
    let mut environment = Environment::new();
    environment.vfs.put_dir("/work/data", 0o700).unwrap();
    mount_package_tree(&mut environment, source.path(), "/work", &[]).unwrap();
    assert_eq!(
        environment
            .vfs
            .metadata("/", "/work/data", false)
            .unwrap()
            .mode,
        0o700
    );
    environment.vfs.symlink("/", "/work", "/linked").unwrap();
    assert!(mount_package_tree(&mut environment, source.path(), "/linked", &[]).is_err());
}

#[test]
fn only_declared_original_builtin_destinations_can_be_replaced() {
    let source = TestDirectory::new();
    fs::write(source.path().join("make"), b"guest make").unwrap();
    let mut environment = Environment::new();
    assert!(mount_package_tree(&mut environment, source.path(), "/usr/bin", &[]).is_err());
    mount_package_tree(
        &mut environment,
        source.path(),
        "/usr/bin",
        &["/usr/bin/make".into()],
    )
    .unwrap();
    assert_eq!(
        environment.vfs.read("/", "/usr/bin/make").unwrap(),
        b"guest make"
    );
    fs::write(source.path().join("make"), b"replacement").unwrap();
    assert!(mount_package_tree(
        &mut environment,
        source.path(),
        "/usr/bin",
        &["/usr/bin/make".into()]
    )
    .is_err());
}
