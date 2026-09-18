//! Integration coverage for atomic, VFS-only `patch` application.

use shellsim::Environment;

fn run(env: &mut Environment, script: &str) -> (i32, String) {
    let (outcome, _, stderr) = env.run_script_capture(script);
    (
        outcome.exit_status,
        String::from_utf8_lossy(&stderr).into_owned(),
    )
}

#[test]
fn unified_patch_updates_adds_and_deletes_files_atomically() {
    let mut env = Environment::new();
    env.vfs
        .put_file("/old.txt", b"-- marker\nkeep\n".to_vec(), 0o755)
        .unwrap();
    env.vfs
        .put_file("/gone.txt", b"gone\n".to_vec(), 0o644)
        .unwrap();
    let script = r#"patch -p1 <<'PATCH'
--- a/old.txt
+++ b/old.txt
@@ -1,2 +1,2 @@
--- marker
+new
 keep
--- /dev/null
+++ b/new.txt
@@ -0,0 +1 @@
+created
--- a/gone.txt
+++ /dev/null
@@ -1 +0,0 @@
-gone
PATCH"#;
    assert_eq!(run(&mut env, script).0, 0);
    assert_eq!(env.vfs.read("/", "/old.txt").unwrap(), b"new\nkeep\n");
    assert_eq!(env.vfs.metadata("/", "/old.txt", true).unwrap().mode, 0o755);
    assert_eq!(env.vfs.read("/", "/new.txt").unwrap(), b"created\n");
    assert!(!env.vfs.exists("/", "/gone.txt"));
}

#[test]
fn agent_patch_format_uses_exact_context_and_rolls_back_every_file_on_failure() {
    let mut env = Environment::new();
    env.vfs
        .put_file("/one", b"before\n".to_vec(), 0o644)
        .unwrap();
    env.vfs
        .put_file("/two", b"actual\n".to_vec(), 0o644)
        .unwrap();
    let script = r#"apply_patch <<'PATCH'
*** Begin Patch
*** Update File: one
@@
-before
+after
*** Update File: two
@@
-expected
+changed
*** End Patch
PATCH"#;
    let (status, stderr) = run(&mut env, script);
    assert_eq!(status, 1);
    assert!(stderr.contains("context was not found"), "{stderr}");
    assert_eq!(env.vfs.read("/", "/one").unwrap(), b"before\n");
    assert_eq!(env.vfs.read("/", "/two").unwrap(), b"actual\n");
}

#[test]
fn agent_patch_can_create_and_delete_files_without_host_access() {
    let mut env = Environment::new();
    env.vfs
        .put_file("/delete-me", b"value\n".to_vec(), 0o644)
        .unwrap();
    let script = r#"apply_patch <<'PATCH'
*** Begin Patch
*** Add File: nested/new.py
+print("hello")
*** Delete File: delete-me
*** End Patch
PATCH"#;
    assert_eq!(run(&mut env, script).0, 0);
    assert_eq!(
        env.vfs.read("/", "/nested/new.py").unwrap(),
        b"print(\"hello\")\n"
    );
    assert!(!env.vfs.exists("/", "/delete-me"));
}
