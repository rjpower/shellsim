//! Compatibility tests for file and directory comparison commands.

use shellsim::interp::{Environment, Interp};

fn run(environment: &mut Interp, source: &str) -> (i32, Vec<u8>, String) {
    let (outcome, stdout, stderr) = environment.run_script_capture(source);
    (
        outcome.exit_status,
        stdout,
        String::from_utf8_lossy(&stderr).into_owned(),
    )
}

#[test]
fn diff_uses_lcs_for_insertions_and_supports_unified_output() {
    let mut environment = Environment::new();
    let (status, stdout, stderr) = run(
        &mut environment,
        "printf 'a\\nb\\nc\\n' > old; printf 'a\\nx\\nb\\nc\\n' > new; diff -u old new",
    );
    assert_eq!(status, 1, "{stderr}");
    assert_eq!(
        stdout,
        b"--- old\n+++ new\n@@ -1,3 +1,4 @@\n a\n+x\n b\n c\n"
    );
}

#[test]
fn recursive_diff_and_brief_mode_report_differences() {
    let mut environment = Environment::new();
    let (status, stdout, stderr) = run(
        &mut environment,
        "mkdir -p a/sub b/sub; echo one > a/sub/f; echo two > b/sub/f; diff -rq a b",
    );
    assert_eq!(status, 1, "{stderr}");
    assert_eq!(stdout, b"Files a/sub/f and b/sub/f differ\n");
}

#[test]
fn recursive_diff_reports_only_in_directories_at_their_parent() {
    let mut environment = Environment::new();
    let (status, stdout, stderr) = run(
        &mut environment,
        "mkdir -p a/nested a/left-empty b/nested b/nested/right-empty; diff -r a b",
    );
    assert_eq!(status, 1, "{stderr}");
    assert_eq!(
        stdout,
        b"Only in a: left-empty\nOnly in b/nested: right-empty\n"
    );
}

#[test]
fn diff_whitespace_controls_compare_keys_but_print_original_lines() {
    let mut environment = Environment::new();
    let (status, stdout, stderr) = run(
        &mut environment,
        "printf 'alpha  beta\\nleft  old\\n' > old; printf 'alpha beta\\nleft new\\n' > new; diff -ub old new",
    );
    assert_eq!(status, 1, "{stderr}");
    assert_eq!(
        stdout,
        b"--- old\n+++ new\n@@ -1,2 +1,2 @@\n alpha  beta\n-left  old\n+left new\n"
    );

    let (status, stdout, stderr) = run(
        &mut environment,
        "printf 'ab c\\n' > old; printf 'a bc\\n' > new; diff --ignore-all-space -q old new; diff --ignore-space-change -q old new",
    );
    assert_eq!(status, 1, "{stderr}");
    assert_eq!(stdout, b"Files old and new differ\n");
}
