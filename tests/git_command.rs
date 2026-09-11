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
