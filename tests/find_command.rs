//! Compatibility strategy: exercise ordinary find predicate composition and explicit frontiers.

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
        ("find . -exec echo {} ';'", "unsupported predicate"),
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
