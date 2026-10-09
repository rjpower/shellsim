//! Real verifier scripts use nullglob so absent test files produce no iterations.

use shellsim::Environment;

fn check(source: &str, expected: &str) {
    let mut environment = Environment::new();
    let (outcome, stdout, stderr) = environment.run_script_capture(source);
    assert_eq!(outcome.exit_status, 0, "{:?}", stderr);
    assert!(stderr.is_empty(), "{:?}", stderr);
    assert_eq!(stdout, expected.as_bytes());
}

#[test]
fn nullglob_removes_unmatched_patterns_and_preserves_quoted_words() {
    check(
        r#"shopt -s nullglob; for item in /absent/*.txt '/absent/*.txt' /absent/\*.txt literal[; do printf '<%s>\n' "$item"; done"#,
        "</absent/*.txt>\n</absent/*.txt>\n<literal[>\n",
    );
}

#[test]
fn nullglob_matches_existing_files_and_can_be_disabled() {
    check(
        r#"mkdir /cases; touch /cases/b.txt /cases/a.txt; shopt -s nullglob; for file in /cases/*.txt /cases/*.missing; do echo "$file"; done; shopt -u nullglob; for file in /cases/*.missing; do echo "$file"; done"#,
        "/cases/a.txt\n/cases/b.txt\n/cases/*.missing\n",
    );
}

#[test]
fn nullglob_is_process_local_and_queryable() {
    check(
        r#"shopt -q nullglob; echo $?; (shopt -s nullglob; shopt -q nullglob; echo $?; for file in /absent/*; do echo wrong; done); shopt -q nullglob; echo $?"#,
        "1\n0\n1\n",
    );
}
