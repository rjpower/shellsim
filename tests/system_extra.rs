//! Compatibility checks for shell state and virtual process/network diagnostics.
//!
//! All observations come from shellsim's process, descriptor, resource, and listener stores.

use shellsim::{
    vfs::{NativeProgram, NodeKind},
    Environment,
};

fn run(environment: &mut Environment, source: &str) -> (i32, String, String) {
    let (outcome, stdout, stderr) = environment.run_script_capture(source);
    (
        outcome.exit_status,
        String::from_utf8_lossy(&stdout).into_owned(),
        String::from_utf8_lossy(&stderr).into_owned(),
    )
}

#[test]
fn shell_state_builtins_have_deterministic_process_local_behavior() {
    let mut environment = Environment::new();
    let (status, stdout, stderr) = run(
        &mut environment,
        "readonly answer=42; unset answer; umask; umask 077; umask; hash echo; hash -t echo; shopt -s expand_aliases; shopt -q expand_aliases; ulimit -n",
    );
    assert_eq!(status, 0, "{stderr}");
    assert!(stderr.contains("readonly variable"));
    assert_eq!(stdout, "0022\n0077\n/usr/bin/echo\n1024\n");
}

#[test]
fn envsubst_reads_child_environment_and_large_pipe() {
    let mut environment = Environment::new();
    environment
        .vfs
        .write("/", "/work/template", &vec![b'x'; 128 * 1024], 0o644)
        .unwrap();
    let (status, stdout, stderr) = run(
        &mut environment,
        "export ITEM=ready; printf '$ITEM ${ITEM}\\n' | envsubst; cat /work/template | envsubst | wc -c",
    );
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, "ready ready\n131072\n");
}

#[test]
fn native_env_preserves_inherited_stdin_and_child_only_overrides() {
    let mut environment = Environment::new();
    let image = environment.vfs.metadata("/", "/usr/bin/env", true).unwrap();
    assert!(matches!(
        image.kind,
        NodeKind::NativeExecutable(NativeProgram::Env)
    ));
    let (status, stdout, stderr) = run(
        &mut environment,
        "printf payload | env -i /usr/bin/cat; env -i KEY=value; env -C /work pwd; pwd",
    );
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, "payloadKEY=value\n/work\n/\n");
}

#[test]
fn umask_accepts_zero_with_any_octal_padding() {
    let mut environment = Environment::new();
    let (status, stdout, stderr) = run(&mut environment, "umask 0; umask; umask 000; umask");
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, "0000\n0000\n");
}

#[test]
fn exec_stops_the_enclosing_shell_after_the_replacement_command() {
    let mut environment = Environment::new();
    let (status, stdout, stderr) = run(
        &mut environment,
        "printf before; exec printf replaced; printf after",
    );
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, "beforereplaced");
}

#[test]
fn descriptor_only_exec_persists_redirections_and_umask_applies() {
    let mut environment = Environment::new();
    let (status, stdout, stderr) = run(
        &mut environment,
        "umask 077; exec > /captured; printf persisted",
    );
    assert_eq!(status, 0, "{stderr}");
    assert!(stdout.is_empty());
    assert_eq!(
        environment.vfs.read("/", "/captured").unwrap(),
        b"persisted"
    );
    assert_eq!(
        environment
            .vfs
            .metadata("/", "/captured", true)
            .unwrap()
            .mode
            & 0o777,
        0o600
    );
}

#[test]
fn readonly_rejects_plain_and_temporary_assignments() {
    let mut environment = Environment::new();
    let (status, stdout, stderr) = run(
        &mut environment,
        "readonly value=kept; value=changed; printf '%s' \"$value\"",
    );
    assert_eq!(status, 0);
    assert_eq!(stdout, "kept");
    assert_eq!(stderr.matches("readonly variable").count(), 1);

    let mut environment = Environment::new();
    let (status, stdout, stderr) = run(
        &mut environment,
        "readonly value=kept; value=temp printf hidden",
    );
    assert_eq!(status, 1);
    assert!(stdout.is_empty());
    assert_eq!(stderr.matches("readonly variable").count(), 1);
}

#[test]
fn process_queries_use_the_modeled_process_table() {
    let mut environment = Environment::new();
    for command in ["ps", "pgrep", "lsof", "ss", "netstat"] {
        let node = environment
            .vfs
            .metadata("/", &format!("/usr/bin/{command}"), true)
            .unwrap();
        assert!(
            matches!(
                node.kind,
                NodeKind::NativeExecutable(NativeProgram::Registered(_))
            ),
            "{command} did not use the native System boundary"
        );
    }
    let (status, stdout, stderr) = run(
        &mut environment,
        "ps -p $$ -o pid,ppid,stat,comm; ps $$ -o pid,comm; pgrep -x bash; lsof -p $$",
    );
    assert_eq!(status, 0, "{stderr}");
    assert!(
        stdout.contains("PID PPID STAT COMMAND\n1234 0 R bash\n"),
        "{stdout}"
    );
    assert!(stdout.contains("PID COMMAND\n1234 bash\n"), "{stdout}");
    assert!(stdout.contains("1234\n"), "{stdout}");
    assert!(stdout.contains("COMMAND PID USER FD TYPE"), "{stdout}");
}

#[test]
fn socket_queries_list_only_virtual_listeners() {
    let mut environment = Environment::new();
    environment.net.listen("127.0.0.1:8080");
    let (status, stdout, stderr) = run(&mut environment, "ss -lnt; netstat -lnt");
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout.matches("127.0.0.1:8080").count(), 2);
}

#[test]
fn hostname_is_machine_state_not_a_child_shell_assignment() {
    let mut environment = Environment::new();
    let (status, stdout, stderr) = run(
        &mut environment,
        "hostname; hostname workshop; hostname; uname -n; printf '%s\\n' \"$HOSTNAME\"",
    );
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, "sandbox\nworkshop\nworkshop\nsandbox\n");
    let (status, stdout, stderr) = run(
        &mut environment,
        &format!("hostname {}; hostname", "x".repeat(256)),
    );
    assert_eq!(status, 0);
    assert_eq!(stdout, "workshop\n");
    assert!(stderr.contains("hostname: invalid argument"), "{stderr}");
}

#[test]
fn nohup_delegates_without_host_process_or_terminal_access() {
    let mut environment = Environment::new();
    assert_eq!(
        run(&mut environment, "nohup printf ok"),
        (0, "ok".into(), String::new())
    );
}

#[test]
fn nohup_ignores_hangup_across_exec() {
    let mut environment = Environment::new();
    let (status, stdout, stderr) = run(
        &mut environment,
        "nohup sh -c 'kill -HUP $$; echo survived'; echo status:$?; nohup; echo status:$?",
    );
    assert_eq!(
        (status, stdout.as_str()),
        (0, "survived\nstatus:0\nstatus:125\n")
    );
    assert_eq!(stderr, "nohup: missing operand\n");
}

#[test]
fn env_and_shell_images_exec_in_the_launching_process() {
    let mut environment = Environment::new();
    let (status, stdout, stderr) = run(
        &mut environment,
        "env sleep 5 & echo bg=$!; sh -c 'ps -o pid,ppid,args'; kill $!",
    );
    assert_eq!(status, 0, "{stderr}");
    assert!(stdout.starts_with("bg=1235\n"), "{stdout}");
    assert!(stdout.contains("\n1235 1234 sleep 5\n"), "{stdout}");
    assert!(
        stdout.contains(" 1234 ps -o pid,ppid,args\n"),
        "sh -c did not exec its only command: {stdout}"
    );
    // Every image that ran under one PID is complete once that process exits.
    assert!(environment
        .invocations
        .events()
        .iter()
        .all(|event| event.status.is_some()));
}

#[test]
fn pkill_and_killall_signal_other_virtual_processes() {
    let mut environment = Environment::new();
    let (status, stdout, stderr) = run(
        &mut environment,
        "sleep 5 & pkill sleep; wait $!; echo pkill:$?; \
         sleep 5 & killall -KILL sleep; wait $!; echo killall:$?; \
         killall missing; echo missing:$?; pgrep pgrep; echo self:$?",
    );
    assert_eq!(
        (status, stdout.as_str()),
        (0, "pkill:143\nkillall:137\nmissing:1\nself:1\n")
    );
    assert_eq!(stderr, "missing: no process found\n");
}

#[test]
fn nice_validates_its_adjustment_and_execs_the_command() {
    let mut environment = Environment::new();
    let (status, stdout, stderr) = run(
        &mut environment,
        "nice; nice -n 5 printf ok; nice -10 printf ' ok'; echo; nice -n x true; echo status:$?",
    );
    assert_eq!((status, stdout.as_str()), (0, "0\nok ok\nstatus:125\n"));
    assert_eq!(stderr, "nice: invalid adjustment 'x'\n");
}

#[test]
fn exec_builtin_replaces_the_shell_process_image() {
    let mut environment = Environment::new();
    // The replacement keeps the PID, inherits pending redirections and stdin, skips builtins
    // and the EXIT trap, and never returns to the rest of the program.
    let (status, stdout, stderr) = run(
        &mut environment,
        "echo pid=$$; exec env X=1 sh -c 'echo \"x=$X pid=$$\"'; echo unreachable",
    );
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, "pid=1234\nx=1 pid=1234\n");

    let (status, stdout, stderr) = run(
        &mut environment,
        "echo hi | exec cat; sh -c 'trap \"echo trap\" EXIT; exec printf \"%s\\n\" image'; \
         sh -c '{ exec cat; } > /captured <<< data; echo unreachable'; cat /captured; \
         sh -c 'exec xargs echo' <<< 'a b'",
    );
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, "hi\nimage\ndata\na b\n");
}

#[test]
fn exec_builtin_failures_leave_the_shell_to_exit() {
    let mut environment = Environment::new();
    let (status, stdout, stderr) = run(
        &mut environment,
        "sh -c 'trap \"echo trap\" EXIT; exec missing; echo unreachable'; echo s=$?; \
         sh -c 'exec -z true'; echo s=$?; sh -c 'exec -a name true'; echo s=$?",
    );
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, "trap\ns=127\ns=2\ns=2\n");
    assert!(stderr.contains("exec: missing: not found"), "{stderr}");
    assert!(stderr.contains("exec: -z: invalid option"), "{stderr}");
    assert!(stderr.contains("exec: -a: unsupported option"), "{stderr}");
}
