//! Compatibility strategy for `find`: ordinary predicate composition and explicit frontiers.

use shellsim::interp::{Environment, Interp};
use shellsim::{Limits, StopReason};

fn run(environment: &mut Interp, source: &str) -> (i32, Vec<u8>, String) {
    let (outcome, stdout, stderr) = environment.run_script_capture(source);
    (
        outcome.exit_status,
        stdout,
        String::from_utf8_lossy(&stderr).into_owned(),
    )
}

#[test]
fn grouped_alternatives_implicit_and_and_negation_compose() {
    let mut environment = Environment::new();
    let (status, stdout, stderr) = run(
        &mut environment,
        "mkdir -p src/nested; touch src/a.rs src/b.py src/c.txt src/nested/d.rs; find src \\( -name '*.rs' -o -name '*.py' \\) -type f",
    );
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, b"src/a.rs\nsrc/b.py\nsrc/nested/d.rs\n");

    let (status, stdout, stderr) = run(&mut environment, "find src -type f ! -name '*.rs' -print");
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, b"src/b.py\nsrc/c.txt\n");
}

#[test]
fn depth_path_and_nul_printing_match_common_find_usage() {
    let mut environment = Environment::new();
    let (status, stdout, stderr) = run(
        &mut environment,
        "mkdir -p root/a/b; touch root/top root/a/mid root/a/b/deep; find root -mindepth 1 -maxdepth 2 -path 'root/a*' -print0",
    );
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, b"root/a\0root/a/b\0root/a/mid\0");

    let (status, stdout, stderr) = run(
        &mut environment,
        "mkdir -p /work/target/project/src; touch /work/target/project/src/a.rs; cd /work/target/project; find . -type f ! -path '*/target/*'",
    );
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, b"./src/a.rs\n");

    let (status, stdout, stderr) = run(
        &mut environment,
        "cd /; mkdir root-entry; find . -mindepth 1 -maxdepth 1 -name root-entry",
    );
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, b"./root-entry\n");
}

#[test]
fn unsupported_or_invalid_predicates_fail_before_walking() {
    let mut environment = Environment::new();
    for (source, expected) in [
        ("find . -printf '%p\\n'", "unsupported predicate"),
        ("find . \\( -name x", "missing ')'"),
        ("find . -maxdepth nope", "invalid argument"),
    ] {
        let (status, stdout, stderr) = run(&mut environment, source);
        assert_eq!(status, 2, "{source}: {stderr}");
        assert!(stdout.is_empty(), "{source}");
        assert!(stderr.contains(expected), "{source}: {stderr}");
    }
}

#[test]
fn find_output_obeys_the_environment_limit() {
    let mut environment = Environment::with_limits(Limits {
        output: 16,
        ..Limits::unlimited()
    });
    let (outcome, _, _) = environment.run_script_capture(
        "mkdir -p directory; touch directory/one directory/two directory/three; find directory -type f",
    );
    assert_eq!(outcome.exit_status, 137);
    assert_eq!(outcome.stop_reason, Some(StopReason::OutputLimitExceeded));
}

#[test]
fn metadata_predicates_and_delete_cover_common_cleanup_usage() {
    let mut environment = Environment::new();
    let (status, stdout, stderr) = run(
        &mut environment,
        "mkdir -p root/empty root/full; : > root/zero; printf data > root/full/data; chmod 600 root/full/data; find root -empty -print; find root -type f -size 4c -perm 600; find root/empty -delete; test ! -e root/empty",
    );
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, b"root/empty\nroot/zero\nroot/full/data\n");
}

#[test]
fn size_units_round_nonempty_files_up() {
    let mut environment = Environment::new();
    let (status, stdout, stderr) = run(
        &mut environment,
        "mkdir root; : > root/zero; printf x > root/one; find root -type f -size 1; find root -type f -size -1",
    );
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, b"root/one\nroot/zero\n");
}

#[test]
fn failed_delete_is_an_operational_error() {
    let mut environment = Environment::new();
    let (status, stdout, stderr) = run(
        &mut environment,
        "mkdir -p root/child; find root -maxdepth 0 -delete",
    );
    assert_eq!(status, 1);
    assert!(stdout.is_empty());
    assert!(stderr.contains("cannot delete 'root'"), "{stderr}");
    assert!(stderr.contains("Directory not empty"), "{stderr}");
}

#[test]
fn virtual_age_predicates_use_elapsed_simulated_time() {
    let mut environment = Environment::new();
    let (status, stdout, stderr) = run(
        &mut environment,
        "mkdir root; touch root/old; sleep 61; touch root/new; find root -type f -mmin +0; find root -type f -mmin 0",
    );
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, b"root/old\nroot/new\n");
}

#[test]
fn exec_dispatches_modeled_commands_immediately_or_in_a_batch() {
    let mut environment = Environment::new();
    let (status, stdout, stderr) = run(
        &mut environment,
        r"mkdir root; touch root/a root/b; find root -type f -exec printf '<%s>' {} \;; echo; find root -type f -exec printf '[%s]' {} +; echo; find root -type f -exec false \; -print",
    );
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, b"<root/a><root/b>\n[root/a][root/b]\n");
}

#[test]
fn malformed_exec_forms_fail_during_expression_parsing() {
    let mut environment = Environment::new();
    for source in [
        "find . -exec echo {}",
        "find . -exec +",
        "find . -exec echo +",
    ] {
        let (status, stdout, stderr) = run(&mut environment, source);
        assert_eq!(status, 2, "{source}: {stderr}");
        assert!(stdout.is_empty(), "{source}");
        assert!(stderr.contains("find:"), "{source}: {stderr}");
    }
}
