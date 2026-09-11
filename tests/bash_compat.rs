use shellsim::Environment;

fn run(source: &str) -> (i32, String, String) {
    let mut env = Environment::new();
    let (outcome, out, err) = env.run_script_capture(source);
    (
        outcome.exit_status,
        String::from_utf8_lossy(&out).into_owned(),
        String::from_utf8_lossy(&err).into_owned(),
    )
}

#[test]
fn brace_expansion_and_here_strings() {
    assert_eq!(
        run("printf '%s ' file{1..3}.txt; cat <<< \"hello world\""),
        (
            0,
            "file1.txt file2.txt file3.txt hello world\n".into(),
            String::new()
        )
    );
}

#[test]
fn arithmetic_commands_and_c_style_for_loops() {
    assert_eq!(
        run("sum=0; for ((i=0; i<4; i++)); do ((sum += i)); done; echo $sum"),
        (0, "6\n".into(), String::new())
    );
}

#[test]
fn standard_paths_and_environment_utilities() {
    assert_eq!(
        run("FOO=outer; /usr/bin/env FOO=inner printenv FOO; echo $FOO; uname -m"),
        (0, "inner\nouter\nx86_64\n".into(), String::new())
    );
}

#[test]
fn common_bash_guard_idioms() {
    assert_eq!(
        run("set -uo pipefail; false | true; echo $?; command -v awk; command echo ok; [[ abc =~ ^a ]] && echo match"),
        (0, "1\n/usr/bin/awk\nok\nmatch\n".into(), String::new())
    );
}

#[test]
fn malformed_compound_syntax_never_executes_a_partial_ast() {
    for source in [
        "echo partial; if true; then echo no",
        "echo partial; for item in one; do echo no",
        "echo partial; while true; do echo no",
        "echo partial; (echo no",
        "echo partial; echo 'no",
    ] {
        let (status, out, err) = run(source);
        assert_eq!(status, 2, "wrong status for {source:?}");
        assert_eq!(out, "", "partial syntax tree executed for {source:?}");
        assert!(
            err.starts_with("shellsim: syntax error:"),
            "missing parse diagnostic for {source:?}: {err:?}"
        );
    }
}

#[test]
fn valid_compound_syntax_still_executes() {
    assert_eq!(
        run("if true; then for item in one two; do (echo \"$item\"); done; fi"),
        (0, "one\ntwo\n".into(), String::new())
    );
}

#[test]
fn completed_background_jobs_are_queryable_and_waitable() {
    assert_eq!(
        run("false & jobs; wait %1; echo $?"),
        (0, "[1] Done false\n1\n".into(), String::new())
    );
    assert_eq!(
        run("alias ll='ls -l'"),
        (
            2,
            String::new(),
            "shellsim: builtin is not supported\n".into()
        )
    );
}

#[test]
fn python_repl_persists_and_returns_to_shell() {
    let mut env = Environment::new();
    let (_, entered, _) = env.run_script_capture("python");
    assert!(String::from_utf8_lossy(&entered).ends_with(">>> "));
    assert!(env.in_python_repl());

    let (_, assigned, _) = env.run_script_capture("x = 4");
    assert_eq!(String::from_utf8_lossy(&assigned), ">>> ");
    let (_, evaluated, _) = env.run_script_capture("x + 3");
    assert_eq!(String::from_utf8_lossy(&evaluated), "7\n>>> ");

    env.run_script_capture("exit()");
    assert!(!env.in_python_repl());
    let (_, shell_out, _) = env.run_script_capture("echo shell");
    assert_eq!(String::from_utf8_lossy(&shell_out), "shell\n");
}
