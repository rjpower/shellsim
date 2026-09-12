//! Integration coverage for the deterministic, VFS-only Git porcelain subset.
//!
//! Tests exercise Git through the normal shell dispatcher and inspect only simulated files. They
//! cover the index/blob boundary because cached content must not change when the working file does.

use shellsim::Environment;

fn run(env: &mut Environment, command: &str) -> (i32, String, String) {
    let (outcome, stdout, stderr) = env.run_script_capture(command);
    (
        outcome.exit_status,
        String::from_utf8_lossy(&stdout).into_owned(),
        String::from_utf8_lossy(&stderr).into_owned(),
    )
}

#[test]
fn stages_commits_and_reports_working_tree_changes() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "git init").0, 0);
    env.vfs
        .put_file("/note.txt", b"first\n".to_vec(), 0o644)
        .unwrap();

    assert_eq!(run(&mut env, "git status --short").1, "?? note.txt\n");
    assert_eq!(run(&mut env, "git add note.txt").0, 0);
    assert_eq!(run(&mut env, "git status --short").1, "A  note.txt\n");
    assert_eq!(run(&mut env, "git commit -m initial").0, 0);
    assert_eq!(run(&mut env, "git status --short").1, "");

    env.vfs
        .put_file("/note.txt", b"second\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(run(&mut env, "git status --porcelain").1, " M note.txt\n");
    let diff = run(&mut env, "git diff");
    assert_eq!(diff.0, 0, "{}", diff.2);
    assert!(diff.1.contains("-first\n+second\n"), "{}", diff.1);
}

#[test]
fn cached_diff_reads_staged_blobs_not_the_working_file() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "git init").0, 0);
    env.vfs.put_file("/data", b"one\n".to_vec(), 0o644).unwrap();
    assert_eq!(run(&mut env, "git add data; git commit -m one").0, 0);

    env.vfs.put_file("/data", b"two\n".to_vec(), 0o644).unwrap();
    assert_eq!(run(&mut env, "git add data").0, 0);
    env.vfs
        .put_file("/data", b"three\n".to_vec(), 0o644)
        .unwrap();

    let cached = run(&mut env, "git diff --cached");
    assert_eq!(cached.0, 0, "{}", cached.2);
    assert!(cached.1.contains("-one\n+two\n"), "{}", cached.1);
    assert!(!cached.1.contains("three"), "{}", cached.1);

    let working = run(&mut env, "git diff");
    assert!(working.1.contains("-two\n+three\n"), "{}", working.1);
}

#[test]
fn resolves_head_and_rejects_unsupported_operations() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "git init project").0, 0);
    env.vfs
        .put_file("/project/file", b"value".to_vec(), 0o644)
        .unwrap();
    let committed = run(&mut env, "cd project; git add file; git commit -m snapshot");
    assert_eq!(committed.0, 0, "{}", committed.2);

    let head = run(&mut env, "git rev-parse HEAD");
    assert_eq!(head.0, 0, "{}", head.2);
    assert_eq!(head.1.trim().len(), 40);
    assert!(head.1.trim().bytes().all(|byte| byte.is_ascii_hexdigit()));

    let unsupported = run(&mut env, "git clone nowhere");
    assert_eq!(unsupported.0, 2);
    assert!(unsupported.2.contains("unsupported subcommand"));
}

#[test]
fn branches_switch_worktrees_and_preserve_independent_history() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "git init").0, 0);
    env.vfs
        .put_file("/file", b"main\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(run(&mut env, "git add file; git commit -m initial").0, 0);
    let initial = run(&mut env, "git rev-parse HEAD").1.trim().to_string();

    assert_eq!(run(&mut env, "git switch -c feature").0, 0);
    env.vfs
        .put_file("/file", b"feature\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(run(&mut env, "git add file; git commit -m feature").0, 0);
    let feature = run(&mut env, "git rev-parse HEAD").1.trim().to_string();
    assert_ne!(initial, feature);

    assert_eq!(run(&mut env, "git switch main").0, 0);
    assert_eq!(env.vfs.read("/", "/file").unwrap(), b"main\n");
    assert_eq!(run(&mut env, "git rev-parse --abbrev-ref HEAD").1, "main\n");
    assert_eq!(run(&mut env, "git branch --show-current").1, "main\n");
    assert_eq!(run(&mut env, "git branch").1, "  feature\n* main\n");
    assert_eq!(
        run(&mut env, "git log --oneline").1,
        format!("{} initial\n", &initial[..7])
    );
    let shown = run(&mut env, "git show feature");
    assert_eq!(shown.0, 0, "{}", shown.2);
    assert!(shown.1.contains("feature"), "{}", shown.1);
    assert!(shown.1.contains("-main\n+feature\n"), "{}", shown.1);

    env.vfs
        .put_file("/file", b"changed\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(run(&mut env, "git diff --name-only").1, "file\n");
    assert!(run(&mut env, "git status")
        .1
        .starts_with("On branch main\n"));
}

#[test]
fn restore_and_reset_move_index_worktree_and_head_coherently() {
    let mut env = Environment::new();
    assert_eq!(run(&mut env, "git init").0, 0);
    env.vfs.put_file("/file", b"one\n".to_vec(), 0o644).unwrap();
    assert_eq!(run(&mut env, "git add file; git commit -m one").0, 0);
    let first = run(&mut env, "git rev-parse HEAD").1.trim().to_string();

    env.vfs.put_file("/file", b"two\n".to_vec(), 0o644).unwrap();
    assert_eq!(run(&mut env, "git restore file").0, 0);
    assert_eq!(env.vfs.read("/", "/file").unwrap(), b"one\n");

    env.vfs.put_file("/file", b"two\n".to_vec(), 0o644).unwrap();
    assert_eq!(run(&mut env, "git add file; git commit -m two").0, 0);
    env.vfs
        .put_file("/staged", b"remove me\n".to_vec(), 0o644)
        .unwrap();
    env.vfs
        .put_file("/untracked", b"keep me\n".to_vec(), 0o644)
        .unwrap();
    assert_eq!(run(&mut env, "git add staged").0, 0);
    assert_eq!(run(&mut env, &format!("git reset --hard {first}")).0, 0);
    assert_eq!(run(&mut env, "git rev-parse HEAD").1.trim(), first);
    assert_eq!(env.vfs.read("/", "/file").unwrap(), b"one\n");
    assert!(!env.vfs.exists("/", "/staged"));
    assert_eq!(env.vfs.read("/", "/untracked").unwrap(), b"keep me\n");
    assert_eq!(run(&mut env, "git status --short").1, "?? untracked\n");
    assert_eq!(run(&mut env, "git log -1 --oneline").1.lines().count(), 1);
}
