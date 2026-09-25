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
