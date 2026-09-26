//! Integration coverage for the typed `/usr/bin/expr` command: POSIX precedence, GNU extras,
//! and the documented error/status contract.

use shellsim::{Environment, Limits};

fn run(source: &str) -> (i32, String, String) {
    let mut environment = Environment::new();
    let (outcome, stdout, stderr) = environment.run_script_capture(source);
    (
        outcome.exit_status,
        String::from_utf8_lossy(&stdout).into_owned(),
        String::from_utf8_lossy(&stderr).into_owned(),
    )
}

#[test]
fn arithmetic_respects_posix_precedence() {
    assert_eq!(run("expr 2 + 3 '*' 4"), (0, "14\n".into(), String::new()));
    assert_eq!(
        run("expr '(' 2 + 3 ')' '*' 4"),
        (0, "20\n".into(), String::new())
    );
    assert_eq!(run("expr 7 % 3"), (0, "1\n".into(), String::new()));
}

#[test]
fn exit_status_reflects_null_or_zero_result() {
    let (status, stdout, _) = run("expr 1 - 1");
    assert_eq!((status, stdout.as_str()), (1, "0\n"));
    let (status, stdout, _) = run("expr 5 + 5");
    assert_eq!((status, stdout.as_str()), (0, "10\n"));
}

#[test]
fn string_and_integer_comparisons() {
    assert_eq!(run("expr 3 '<' 10"), (0, "1\n".into(), String::new()));
    assert_eq!(run("expr abc '<' abd"), (0, "1\n".into(), String::new()));
    assert_eq!(run("expr abc = abc"), (0, "1\n".into(), String::new()));
}

#[test]
fn or_and_and_short_circuit_on_falsy_operands() {
    assert_eq!(
        run("expr '' '|' fallback"),
        (0, "fallback\n".into(), String::new())
    );
    assert_eq!(run("expr foo '&' bar"), (0, "foo\n".into(), String::new()));
    assert_eq!(run("expr '' '&' bar"), (1, "0\n".into(), String::new()));
}

#[test]
fn colon_returns_capture_or_match_length() {
    assert_eq!(
        run(r"expr hello : 'h\(.*\)o'"),
        (0, "ell\n".into(), String::new())
    );
    assert_eq!(run("expr hello : hel"), (0, "3\n".into(), String::new()));
    assert_eq!(run("expr hello : xyz"), (1, "0\n".into(), String::new()));
}

#[test]
fn gnu_extensions_match_substr_index_length() {
    assert_eq!(
        run(r"expr match hello 'h\(.*\)o'"),
        (0, "ell\n".into(), String::new())
    );
    assert_eq!(run("expr length hello"), (0, "5\n".into(), String::new()));
    assert_eq!(run("expr index hello lo"), (0, "3\n".into(), String::new()));
    assert_eq!(
        run("expr substr hello 2 3"),
        (0, "ell\n".into(), String::new())
    );
}

#[test]
fn plus_escapes_a_keyword_as_a_literal_string() {
    assert_eq!(run("expr + length"), (0, "length\n".into(), String::new()));
}

#[test]
fn division_by_zero_is_a_syntax_level_error() {
    let (status, _, stderr) = run("expr 1 / 0");
    assert_eq!(status, 2);
    assert!(stderr.contains("division by zero"), "{stderr}");
}

#[test]
fn non_integer_operands_reject_arithmetic() {
    let (status, _, stderr) = run("expr abc + 1");
    assert_eq!(status, 2);
    assert!(stderr.contains("non-integer argument"), "{stderr}");
}

#[test]
fn integer_overflow_is_reported_not_wrapped() {
    let (status, _, stderr) = run("expr 9223372036854775807 + 1");
    assert_eq!(status, 2);
    assert!(stderr.contains("too large"), "{stderr}");
}

#[test]
fn unbalanced_parentheses_are_a_syntax_error() {
    let (status, _, stderr) = run("expr '(' 1 + 1");
    assert_eq!(status, 2);
    assert!(stderr.contains("syntax error"), "{stderr}");
}

#[test]
fn cpu_budget_bounds_a_large_expr_invocation() {
    let mut environment = Environment::with_limits(Limits {
        cpu: 40,
        ..Limits::unlimited()
    });
    let (outcome, _, _) = environment.run_script_capture("expr 1 + 1");
    assert_eq!(outcome.exit_status, 137);
}
