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
fn path_resolves_only_executable_vfs_scripts() {
    assert_eq!(
        run(
            "mkdir /tools; printf '%s\n' '#!/bin/sh' 'printf path:$1' > /tools/hello; chmod +x /tools/hello; PATH=/tools command -v hello; PATH=/tools hello world",
        ),
        (0, "/tools/hello\npath:world".into(), String::new())
    );
    let denied = run("printf 'echo no' > /not-executable; /not-executable");
    assert_eq!(denied.0, 126);
    assert!(denied.2.contains("permission denied"), "{}", denied.2);
    let interpreter =
        run("printf '%s\n' '#!/host/interpreter' 'echo no' > /bad; chmod +x /bad; /bad");
    assert_eq!(interpreter.0, 126);
    assert!(
        interpreter.2.contains("unsupported script interpreter"),
        "{}",
        interpreter.2
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
fn child_shell_boundaries_isolate_local_state_but_share_files() {
    assert_eq!(
        run("x=parent; (x=child; cd /tmp; printf saved > child-file); printf '%s:%s:' \"$x\" \"$PWD\"; cat /tmp/child-file"),
        (0, "parent:/:saved".into(), String::new())
    );
    assert_eq!(
        run("X=outer; value=$(cd /tmp; X=inner; printf captured); printf '%s:%s:%s\n' \"$X\" \"$PWD\" \"$value\""),
        (0, "outer:/:captured\n".into(), String::new())
    );
    assert_eq!(
        run("printf value | read piped; echo ${piped-unset}; sh -c 'cd /tmp; X=inner'; printf '%s:%s\n' \"${X-unset}\" \"$PWD\""),
        (0, "unset\nunset:/\n".into(), String::new())
    );
}

#[test]
fn redirections_use_ordered_process_descriptors() {
    assert_eq!(
        run("sh -c 'printf out; missing-command' > /both 2>&1; cat /both"),
        (
            0,
            "outmissing-command: command not found\n".into(),
            String::new()
        )
    );
    assert_eq!(
        run("sh -c 'printf out; missing-command' 2>&1 > /only-out; cat /only-out"),
        (
            0,
            "missing-command: command not found\nout".into(),
            String::new()
        )
    );
    assert_eq!(
        run("{ readlink /proc/self/fd/1; } > /target; cat /target"),
        (0, "/target\n".into(), String::new())
    );
}

#[test]
fn missing_input_redirection_fails_before_running_the_command() {
    let (status, out, err) = run("cat < /missing; echo $?");
    assert_eq!(status, 0);
    assert_eq!(out, "1\n");
    assert!(err.contains("No such file or directory: /missing"), "{err}");
}

#[test]
fn descriptor_close_and_failed_redirect_setup_are_explicit() {
    assert_eq!(
        run("printf hidden >&-; echo visible"),
        (
            0,
            "visible\n".into(),
            "shellsim: printf: bad file descriptor\n".into()
        )
    );
    let (status, out, err) = run("echo hidden 1>&9; echo $?");
    assert_eq!(status, 0);
    assert_eq!(out, "1\n");
    assert!(err.contains("InvalidFd"), "{err}");

    let mut env = Environment::new();
    let (_, _, err) = env.run_script_capture("printf value > /created > /missing/result");
    assert!(!env.vfs.lexists("/", "/created"));
    assert!(String::from_utf8_lossy(&err).contains("No such file or directory"));
}

#[test]
fn recursive_shell_functions_stop_at_the_continuation_limit() {
    let (status, out, err) = run("recurse() { recurse nested; }; recurse outer; echo unreachable");
    assert_eq!(status, 2);
    assert_eq!(out, "");
    assert!(
        err.contains("shell continuation frame limit exceeded"),
        "{err}"
    );
}

#[test]
fn logical_process_ids_back_jobs_wait_and_ps() {
    assert_eq!(
        run("printf '%s:%s:%s\n' \"$$\" \"$BASHPID\" \"$PPID\"; (printf '%s:%s:%s\n' \"$$\" \"$BASHPID\" \"$PPID\")"),
        (0, "1234:1234:0\n1234:1235:1234\n".into(), String::new())
    );
    assert_eq!(
        run("sleep 0 & printf '%s\n' \"$!\"; jobs -p; wait \"$!\"; jobs -p"),
        (0, "1235\n1235\n".into(), String::new())
    );
    let (status, output, error) = run("(ps -ef)");
    assert_eq!(status, 0, "{error}");
    assert!(output.contains(" 1234 "), "{output}");
    assert!(output.contains(" 1235 "), "{output}");
}

#[test]
fn nested_shells_remain_under_outer_timeout() {
    assert_eq!(
        run("timeout 1 sh -c 'sleep 2'; echo $?"),
        (0, "124\n".into(), String::new())
    );
}

#[test]
fn nested_shells_yield_to_other_logical_processes() {
    assert_eq!(
        run("(sleep 1; printf ready > /tmp/marker) & bash -c 'sleep 2; cat /tmp/marker'; wait"),
        (0, "ready".into(), String::new())
    );
    let (status, stdout, stderr) = run("bash -c 'echo partial; if true; then echo never'");
    assert_eq!(status, 2);
    assert!(stdout.is_empty());
    assert!(stderr.contains("expected `fi`"), "{stderr}");
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
