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
fn random_is_bounded_deterministic_and_assignment_reseeds_it() {
    let source = "RANDOM=7; printf '%s ' \"$RANDOM\" \"$RANDOM\"; RANDOM=7; printf '%s\\n' \"$RANDOM\"; for ((i=0; i<20; i++)); do (( RANDOM >= 0 && RANDOM <= 32767 )) || exit 9; done";
    let first = run(source);
    let second = run(source);
    assert_eq!(first, second);
    assert_eq!(first.0, 0, "{}", first.2);
    let values = first.1.split_whitespace().collect::<Vec<_>>();
    assert_eq!(values.len(), 3);
    assert_eq!(values[0], values[2]);
    assert_ne!(values[0], values[1]);
}

#[test]
fn standard_paths_and_environment_utilities() {
    assert_eq!(
        run("FOO=outer; /usr/bin/env FOO=inner printenv FOO; echo $FOO; uname -m"),
        (0, "inner\nouter\nx86_64\n".into(), String::new())
    );
}

#[test]
fn path_resolves_only_executable_vfs_entries() {
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
    assert_eq!(
        run("mkdir /tools; printf '%s\n' '#!/bin/sh' 'sleep 2' 'cat /tmp/ready' > /tools/yielding; chmod +x /tools/yielding; (sleep 1; printf scheduled > /tmp/ready) & PATH=/tools:/usr/bin yielding; wait"),
        (0, "scheduled".into(), String::new())
    );
    assert_eq!(
        run("mkdir /tools; printf '%s\n' '#!/bin/sh' 'cat' > /tools/read-input; chmod +x /tools/read-input; printf piped | PATH=/tools:/usr/bin read-input"),
        (0, "piped".into(), String::new())
    );
    assert_eq!(
        run("mkdir -p /work/tools; cd /work; printf '%s\n' '#!/bin/sh' 'printf relative' > tools/hello; chmod +x tools/hello; PATH=tools hello"),
        (0, "relative".into(), String::new())
    );
    assert_eq!(
        run("cd /work; printf '%s\n' '#!/bin/sh' 'printf current' > hello; chmod +x hello; PATH=:/usr/bin hello"),
        (0, "current".into(), String::new())
    );
}

#[test]
fn command_v_finds_shell_builtins_without_path_entries() {
    assert_eq!(
        run("PATH=/missing command -v local; PATH=/missing command -V local"),
        (0, "local\nlocal is a shell builtin\n".into(), String::new())
    );
    assert_eq!(run("PATH=/missing command -v cat").0, 1);
    assert_eq!(
        run("unset PATH; command -v cat; PATH='' command -v cat"),
        (1, "/usr/bin/cat\n".into(), String::new())
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
fn getopts_handles_clusters_arguments_errors_and_explicit_operands() {
    let source = r#"
set -- -ab value rest
while getopts "ab:" option; do printf '%s:%s:%s\n' "$option" "$OPTARG" "$OPTIND"; done
printf 'done:%s\n' "$OPTIND"
OPTIND=1
while getopts ":x:" option -xinline -z; do printf '%s:%s:%s\n' "$option" "$OPTARG" "$OPTIND"; done
OPTIND=1
getopts "ab" option -ab; printf 'reset:%s:%s\n' "$option" "$OPTIND"
OPTIND=1
getopts "ab" option -ab; printf 'reset:%s:%s\n' "$option" "$OPTIND"
"#;
    assert_eq!(
        run(source),
        (
            0,
            "a::1\nb:value:3\ndone:3\nx:inline:2\n?:z:3\nreset:a:1\nreset:a:1\n".into(),
            String::new()
        )
    );
}

#[test]
fn double_bracket_covers_predicates_boolean_logic_and_regex_captures() {
    assert_eq!(
        run(
            "mkdir -p /work/logs; empty=; level=ERROR; line='[FATAL] Exception occurred: app.Bad'; [[ -d /work/logs && -z \"$empty\" ]] && echo ready; [[ \"$level\" == ERROR || \"$level\" == FATAL ]] && echo level; [[ $line =~ \\[(ERROR|FATAL)\\] ]] && printf '%s:%s\\n' \"${BASH_REMATCH[0]}\" \"${BASH_REMATCH[1]}\"; [[ ! ( 1 -gt 2 || x != x ) ]] && echo grouped",
        ),
        (
            0,
            "ready\nlevel\n[FATAL]:FATAL\ngrouped\n".into(),
            String::new(),
        )
    );
}

#[test]
fn input_process_substitution_preserves_bytes_and_parent_shell_mutation() {
    assert_eq!(
        run(
            "mapfile -t rows < <(printf 'one\\ntwo\\n'); printf '%s:%s:%s\\n' \"${rows[0]}\" \"${rows[1]}\" \"${#rows[@]}\"; mapfile -d '' names < <(printf 'a\\0b\\0'); printf '%s:%s:%s\\n' \"${names[0]}\" \"${names[1]}\" \"${#names[@]}\"",
        ),
        (0, "one:two:2\na:b:2\n".into(), String::new())
    );

    assert_eq!(
        run("cat <(printf data); printf streamed | tee >(cat > /captured) > /dev/null; cat /captured"),
        (0, "datastreamed".into(), String::new())
    );
}

#[test]
fn failed_process_substitution_preparation_removes_temporary_files() {
    let mut environment = Environment::new();
    let (outcome, _, _) = environment.run_script_capture("cat >(cat > /captured) <(if)");
    assert_ne!(outcome.exit_status, 0);
    assert!(environment
        .vfs
        .walk("/tmp")
        .iter()
        .all(|path| !path.contains(".shellsim-process-substitution-")));
}

#[test]
fn extended_redirection_operators_match_common_shell_idioms() {
    assert_eq!(
        run("missing-one &> /log; missing-two &>> /log; cat /log"),
        (
            0,
            "missing-one: command not found\nmissing-two: command not found\n".into(),
            String::new(),
        )
    );
    assert_eq!(
        run(
            "missing-command |& cat; printf copied | cat 3<&0 <&3; printf kept >| /file; cat /file"
        ),
        (
            0,
            "missing-command: command not found\ncopiedkept".into(),
            String::new(),
        )
    );
    assert_eq!(
        run("printf original > /rw; read value 3<>/rw <&3; printf %s \"$value\""),
        (0, "original".into(), String::new())
    );
}

#[test]
fn multi_digit_descriptors_support_common_flock_guards() {
    assert_eq!(
        run("(flock -x 200; printf locked) 200>/tmp/lock; test -f /tmp/lock"),
        (0, "locked".into(), String::new())
    );
    let (status, _, stderr) = run("flock -x 200");
    assert_eq!(status, 1);
    assert!(stderr.contains("bad file descriptor"), "{stderr}");
}

#[test]
fn exit_traps_run_once_at_script_completion_and_preserve_status() {
    assert_eq!(
        run("trap 'printf cleanup:$?' EXIT; printf body:; false"),
        (1, "body:cleanup:1".into(), String::new())
    );
    assert_eq!(
        run("trap 'printf no' EXIT; trap - EXIT; printf kept"),
        (0, "kept".into(), String::new())
    );
}

#[test]
fn test_negates_unary_file_predicates() {
    assert_eq!(
        run("[ ! -f /missing ] && echo absent; touch /present; test ! -d /present && echo file"),
        (0, "absent\nfile\n".into(), String::new())
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
        "echo partial; [[ value == value",
        "echo partial; cat < <(printf no",
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
        run("alias ll='echo alias'; ll works; alias ll; unalias ll; ll fails"),
        (
            127,
            "alias works\nalias ll='echo alias'\n".into(),
            "ll: command not found\n".into()
        )
    );
}

#[test]
fn aliases_are_process_local_and_recursion_is_bounded() {
    assert_eq!(
        run("alias say='echo'; alias outer='say'; outer hello; (unalias say; say child); say parent"),
        (
            0,
            "hello\nparent\n".into(),
            "say: command not found\n".into()
        )
    );
    let recursive = run("alias a=b; alias b=a; a");
    assert_eq!(recursive.0, 2);
    assert!(recursive.2.contains("recursive alias"), "{}", recursive.2);
    let compound = run("alias bad='echo one; echo two'");
    assert_eq!(compound.0, 2);
    assert!(
        compound.2.contains("simple-command aliases"),
        "{}",
        compound.2
    );
}

#[test]
fn directory_stack_is_process_local_and_bounded() {
    assert_eq!(
        run("mkdir -p /work/a /work/b; cd /work; pushd a; pushd ../b; dirs -p; popd; pwd; pushd; pwd; dirs -c; dirs"),
        (
            0,
            concat!(
                "/work/a /work\n",
                "/work/b /work/a /work\n",
                "/work/b\n/work/a\n/work\n",
                "/work/a /work\n",
                "/work/a\n",
                "/work /work/a\n",
                "/work\n",
                "/work\n",
            )
            .into(),
            String::new()
        )
    );
    assert_eq!(
        run("mkdir /a; pushd /a >/dev/null; (pushd /tmp >/dev/null; dirs); dirs"),
        (0, "/tmp /a /\n/a /\n".into(), String::new())
    );
    let empty = run("popd");
    assert_eq!(empty.0, 1);
    assert!(empty.2.contains("directory stack empty"), "{}", empty.2);
}

#[test]
fn mapfile_populates_bounded_indexed_arrays() {
    assert_eq!(
        run("printf 'zero\\none\\ntwo\\nthree\\n' > /tmp/lines; mapfile -t -s 1 -n 2 rows < /tmp/lines; printf '<%s>|<%s>|%s\\n' \"${rows[0]}\" \"${rows[1]}\" \"${#rows[@]}\"; mapfile -t -O 2 rows <<< tail; printf '%s:%s\\n' \"${rows[2]}\" \"${#rows[@]}\""),
        (0, "<one>|<two>|2\ntail:3\n".into(), String::new())
    );
    assert_eq!(
        run("printf 'a,b,' > /tmp/records; mapfile -t -d , values < /tmp/records; printf '%s-%s-%s\\n' \"${values[0]}\" \"${values[1]}\" \"${#values[@]}\""),
        (0, "a-b-2\n".into(), String::new())
    );
    let invalid = run("mapfile -C callback rows");
    assert_eq!(invalid.0, 2);
    assert!(invalid.2.contains("unsupported option"), "{}", invalid.2);
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
fn redirections_preserve_append_saved_fds_and_here_documents() {
    assert_eq!(
        run("printf first > /file; printf second >> /file; exec 3< /file; cat <&3; cat <<EOF\nheredoc\nEOF\n"),
        (0, "firstsecondheredoc\n".into(), String::new())
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
fn source_and_eval_execute_inline_continuation_frames() {
    assert_eq!(
        run("printf '%s\n' 'sleep 2' 'cat /tmp/marker' 'from_source=yes' > /tmp/body; (sleep 1; printf ready > /tmp/marker) & source /tmp/body; eval 'from_eval=yes'; printf ':%s:%s' \"$from_source\" \"$from_eval\"; wait"),
        (0, "ready:yes:yes".into(), String::new())
    );
    let (status, stdout, stderr) =
        run("printf 'echo partial; if true; then echo never' > /tmp/bad; source /tmp/bad");
    assert_eq!(status, 2);
    assert!(stdout.is_empty());
    assert!(stderr.contains("expected `fi`"), "{stderr}");
}

#[test]
fn return_exits_the_nearest_sourced_script() {
    assert_eq!(
        run("printf '%s\n' 'printf before' 'return 7' 'printf after' > /tmp/body; . /tmp/body; printf ':%s' \"$?\""),
        (0, "before:7".into(), String::new())
    );

    let outside = run("return 4");
    assert_eq!(outside.0, 1);
    assert!(outside.2.contains("can only be used"), "{}", outside.2);
}

#[test]
fn command_substitutions_use_scheduled_captured_children() {
    assert_eq!(
        run("(sleep 1; printf ready > /tmp/marker) & value=$(sleep 2; cat /tmp/marker); printf '<%s>' \"$value\"; wait"),
        (0, "<ready>".into(), String::new())
    );
    assert_eq!(
        run("printf '<%s>:<%s>' \"$(printf 'a b\\n\\n')\" `printf legacy`; env | rg '^__SHELLSIM_COMMAND_SUBSTITUTION_'"),
        (1, "<a b>:<legacy>".into(), String::new())
    );
    assert_eq!(
        run("result=$(false); printf '%s' $?"),
        (0, "1".into(), String::new())
    );
    assert_eq!(
        run("(sleep 1; printf 'one two' > /tmp/items) & for item in $(sleep 2; cat /tmp/items); do printf '[%s]' \"$item\"; done; wait"),
        (0, "[one][two]".into(), String::new())
    );
    assert_eq!(
        run("case \"$(printf ready)\" in $(printf 'r*')) printf case;; esac; printf redirected > \"$(printf /tmp/output)\"; cat /tmp/output; cat <<EOF\n$(printf heredoc)\nEOF"),
        (0, "caseredirectedheredoc\n".into(), String::new())
    );
    assert_eq!(
        run("(sleep 1; printf 2 > /tmp/start) & (( value = $(sleep 2; cat /tmp/start) + 1 )); printf '%s:%s' \"$value\" $?; wait"),
        (0, "3:0".into(), String::new())
    );
    assert_eq!(
        run("(sleep 1; printf 1 > /tmp/start) & for (( i=$(sleep 2; cat /tmp/start); i<3; i=$(printf \"$((i+1))\") )); do printf '%s' \"$i\"; done; wait"),
        (0, "12".into(), String::new())
    );
}

#[test]
fn native_command_adapters_launch_scheduled_children() {
    assert_eq!(
        run("(sleep 1; printf ready > /tmp/marker) & env TEST=child bash -c 'sleep 2; cat /tmp/marker; printf :$TEST'; printf ':%s' \"${TEST:-}\"; wait"),
        (0, "ready:child:".into(), String::new())
    );
    assert_eq!(
        run("(sleep 1; printf ready > /tmp/marker) & printf 'one\\ntwo\\n' | xargs -n 1 sh -c 'sleep 2; printf \"<$1>:\"; cat /tmp/marker' _; wait"),
        (0, "<one>:ready<two>:ready".into(), String::new())
    );
    assert_eq!(
        run("printf 'a\\0b c\\0' | xargs -0 -n 1 printf '<%s>'"),
        (0, "<a><b c>".into(), String::new())
    );
    assert_eq!(
        run("(sleep 1; printf ready > /tmp/marker) & command bash -c 'sleep 2; cat /tmp/marker'; command cd /tmp; printf ':%s' \"$PWD\"; wait"),
        (0, "ready:/tmp".into(), String::new())
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

#[test]
fn empty_double_quoted_word_is_one_empty_field() {
    // Only an empty quoted array or `"$@"` expansion disappears; `""` and `"$empty"` remain.
    assert_eq!(
        run("f(){ printf '%s,' \"$#\"; }; e=; a=(); set --; \
             f \"\" x; f \"\"; f \"$e\"; f $e; f \"$@\"; f \"${a[@]}\"; f \"x${a[@]}\"; \
             printf '[%s]' \"\" x"),
        (0, "2,1,1,0,0,0,1,[][x]".into(), String::new())
    );
}
