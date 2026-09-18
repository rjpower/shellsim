//! End-to-end coverage for the host-Git import spike.
//!
//! Tests construct repositories with the reference Git executable. One is aggressively packed
//! and verified to contain a delta before Shellsim imports it; another exercises `.git` file and
//! common-directory discovery through a linked worktree.

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
            "shellsim-git-cli-import-test-{}-{sequence}",
            std::process::id()
        ));
        std::fs::create_dir(&path).expect("create test repository directory");
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

fn git(repository: &Path, arguments: &[&str]) -> Vec<u8> {
    let output = Command::new("git")
        .args(arguments)
        .current_dir(repository)
        .env("GIT_AUTHOR_DATE", "1700000000 +0000")
        .env("GIT_COMMITTER_DATE", "1700000000 +0000")
        .output()
        .expect("run reference Git");
    assert!(
        output.status.success(),
        "git {arguments:?} failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    output.stdout
}

fn initialize(repository: &Path) {
    git(repository, &["init", "-b", "main"]);
    git(repository, &["config", "user.name", "Import Test"]);
    git(repository, &["config", "user.email", "import@example.com"]);
}

fn version(number: usize) -> String {
    (0..2_000)
        .map(|line| {
            format!("line {line:04}: stable content for delta compression, version {number}\n")
        })
        .collect()
}

#[test]
fn imports_head_history_from_a_delta_compressed_pack() {
    let repository = TestDirectory::new();
    initialize(repository.path());

    for (number, message) in [(1, "first"), (2, "second"), (3, "third")] {
        std::fs::write(repository.path().join("data.txt"), version(number))
            .expect("write versioned content");
        git(repository.path(), &["add", "data.txt"]);
        git(repository.path(), &["commit", "-m", message]);
    }
    git(
        repository.path(),
        &["repack", "-adf", "--depth=50", "--window=50"],
    );
    git(repository.path(), &["prune-packed"]);

    let pack_dir = repository.path().join(".git/objects/pack");
    let index = std::fs::read_dir(&pack_dir)
        .expect("read pack directory")
        .map(|entry| entry.expect("read pack entry").path())
        .find(|path| path.extension().is_some_and(|extension| extension == "idx"))
        .expect("packed repository has an index");
    let verification = Command::new("git")
        .args(["verify-pack", "-v"])
        .arg(&index)
        .output()
        .expect("verify generated pack");
    assert!(verification.status.success());
    assert!(
        String::from_utf8_lossy(&verification.stdout)
            .lines()
            .any(|line| line.split_whitespace().count() == 7),
        "fixture pack did not contain a delta"
    );

    let mut environment = Environment::new();
    let report =
        shellsim::host_ingest::mount_host_tree_report(&mut environment, repository.path(), "/work")
            .expect("import packed repository");
    let git = report.git.as_ref().expect("Git history report");

    assert_eq!(report.files, 1);
    assert_eq!(git.commits, 3);
    assert_eq!(git.blobs, 3);
    assert_eq!(git.tree_entries, 3);
    assert!(!git.truncated_history);
    assert_ne!(git.source_head, git.imported_head);

    environment.cwd = "/work".to_string();
    let (outcome, stdout, stderr) = environment.run_script_capture(
        "git branch --show-current; git log --format=%s; git status --porcelain",
    );
    assert_eq!(
        outcome.exit_status,
        0,
        "{}",
        String::from_utf8_lossy(&stderr)
    );
    assert_eq!(stdout, b"main\nthird\nsecond\nfirst\n");
    assert!(stderr.is_empty());

    let (outcome, stdout, stderr) = environment.run_script_capture("git show HEAD~1:data.txt");
    assert_eq!(
        outcome.exit_status,
        0,
        "{}",
        String::from_utf8_lossy(&stderr)
    );
    assert_eq!(stdout, version(2).as_bytes());
    assert!(stderr.is_empty());
}

#[test]
fn imports_a_linked_worktree_without_copying_its_gitfile() {
    let directory = TestDirectory::new();
    let repository = directory.path().join("primary");
    let linked = directory.path().join("linked");
    std::fs::create_dir(&repository).expect("create primary repository");
    initialize(&repository);
    std::fs::write(repository.join("note.txt"), "from primary\n").expect("write tracked file");
    git(&repository, &["add", "note.txt"]);
    git(&repository, &["commit", "-m", "root"]);
    git(
        &repository,
        &[
            "worktree",
            "add",
            "-b",
            "side",
            linked.to_str().expect("UTF-8 linked-worktree path"),
        ],
    );
    assert!(linked.join(".git").is_file());

    let mut environment = Environment::new();
    let report = shellsim::host_ingest::mount_host_tree_report(&mut environment, &linked, "/work")
        .expect("import linked worktree");
    let git = report.git.as_ref().expect("Git history report");

    assert_eq!(report.files, 1);
    assert_eq!(git.commits, 1);
    assert!(environment.vfs.is_dir("/", "/work/.git"));
    environment.cwd = "/work".to_string();
    let (outcome, stdout, stderr) = environment.run_script_capture(
        "git branch --show-current; git log --format=%s; git status --porcelain",
    );
    assert_eq!(
        outcome.exit_status,
        0,
        "{}",
        String::from_utf8_lossy(&stderr)
    );
    assert_eq!(stdout, b"side\nroot\n");
    assert!(stderr.is_empty());
}
