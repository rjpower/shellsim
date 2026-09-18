//! Compatibility and isolation tests for Python subprocesses over logical process execution.
//!
//! These tests use only registered commands and VFS scripts. They intentionally probe captured
//! and inherited streams plus process-local cwd/environment state at the public Python API.

use shellsim::{process::MAX_PROCESSES, Environment};

use super::support::run_python_text_in as run;

#[test]
fn run_captures_bytes_and_text_and_can_inherit_output() {
    let mut environment = Environment::new();
    let result = run(
        &mut environment,
        r#"
import subprocess
binary = subprocess.run(["printf", "hello"], capture_output=True)
text = subprocess.run(("printf", "world"), capture_output=True, text=True)
print(binary.returncode, binary.stdout, binary.stderr)
print(text.stdout)
subprocess.run(["printf", "inherited"])
"#,
    );
    assert_eq!(
        result,
        (0, "0 b'hello' b''\nworld\ninherited".into(), String::new())
    );
}

#[test]
fn input_wrappers_and_failure_status_match_the_synchronous_api() {
    let mut environment = Environment::new();
    let result = run(
        &mut environment,
        r#"
import subprocess
from subprocess import CalledProcessError
print(subprocess.check_output(["cat"], input=b"bytes"))
print(subprocess.check_output(["cat"], input="text", text=True))
print(subprocess.call(["false"]))
try:
    subprocess.check_call(["false"])
except CalledProcessError as error:
    print(str(error))
"#,
    );
    assert_eq!(result.0, 0, "{}", result.2);
    assert_eq!(
        result.1,
        "b'bytes'\ntext\n1\nCommand ['false'] returned non-zero exit status 1\n"
    );
}

#[test]
fn cwd_environment_and_filesystem_effects_cross_only_modeled_boundaries() {
    let mut environment = Environment::new();
    let result = run(
        &mut environment,
        r#"
import os
import subprocess
from subprocess import TimeoutExpired
subprocess.check_call(["mkdir", "made"], cwd="/work", env={"ONLY": "child"})
print(subprocess.check_output(["pwd"], cwd="/work", text=True).strip())
print(subprocess.check_output(["printenv", "ONLY"], env={"ONLY": "child"}, text=True).strip())
print(os.getcwd(), os.getenv("ONLY"), os.path.isdir("/work/made"))
"#,
    );
    assert_eq!(
        result,
        (0, "/work\nchild\n/ None True\n".into(), String::new())
    );
}

#[test]
fn python_can_launch_a_python_child_over_the_same_process_layer() {
    let mut environment = Environment::new();
    let result = run(
        &mut environment,
        r#"
import subprocess
output = subprocess.check_output(
    ["python3.14", "-c", "import os\nprint(os.getcwd(), os.getenv('CHILD'))"],
    cwd="/work",
    env={"CHILD": "yes"},
    text=True,
)
print(output.strip())
"#,
    );
    assert_eq!(result, (0, "/work yes\n".into(), String::new()));
}

#[test]
fn popen_children_overlap_and_poll_without_advancing_virtual_time() {
    let mut environment = Environment::new();
    let result = run(
        &mut environment,
        r#"
import subprocess
import time
slow = subprocess.Popen(["sleep", "2"])
fast = subprocess.Popen(["sleep", "1"])
print(slow.pid != fast.pid, slow.poll(), time.monotonic())
print(slow.wait(), fast.poll(), time.monotonic())
"#,
    );
    assert_eq!(
        result,
        (0, "True None 0.0\n0 0 2.0\n".into(), String::new())
    );
}

#[test]
fn popen_poll_observes_without_dispatching_the_child() {
    let mut environment = Environment::new();
    let result = run(
        &mut environment,
        r#"
import os
import subprocess
process = subprocess.Popen(["touch", "/polled"])
print(process.poll(), os.path.exists("/polled"))
print(process.wait(), os.path.exists("/polled"))
"#,
    );
    assert_eq!(result, (0, "None False\n0 True\n".into(), String::new()));
}

#[test]
fn popen_communicate_drains_duplex_pipes_larger_than_capacity() {
    let mut environment = Environment::new();
    let result = run(
        &mut environment,
        r#"
import subprocess
data = b"abcdefgh" * 20000
process = subprocess.Popen(["cat"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE)
stdout, stderr = process.communicate(data)
print(process.returncode, len(stdout), stdout == data, stderr)
"#,
    );
    assert_eq!(result, (0, "0 160000 True b''\n".into(), String::new()));
}

#[test]
fn popen_exposes_binary_and_text_pipe_streams() {
    let mut environment = Environment::new();
    let result = run(
        &mut environment,
        r#"
import subprocess
binary = subprocess.Popen(["printf", "abcdef"], stdout=subprocess.PIPE)
print(binary.stdout.read(2), binary.stdout.read(), binary.wait())
text = subprocess.Popen(["cat"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
print(text.stdin.write("one\ntwo\n"), text.stdin.flush())
text.stdin.close()
print(text.stdout.readline().strip(), text.stdout.read().strip(), text.wait())
print(text.stdin.closed, text.stdout.readable(), text.stdout.fileno())
"#,
    );
    assert_eq!(result.0, 0, "{}", result.2);
    assert_eq!(
        result.1,
        "b'ab' b'cdef' 0\n8 None\none two 0\nTrue True 1\n"
    );
}

#[test]
fn popen_timeout_keeps_child_live_and_terminate_reports_negative_signal() {
    let mut environment = Environment::new();
    let result = run(
        &mut environment,
        r#"
import subprocess
import time
from subprocess import TimeoutExpired
process = subprocess.Popen(["sleep", "10"])
try:
    process.wait(timeout=1)
except TimeoutExpired:
    print("timeout", process.poll(), time.monotonic())
process.terminate()
print(process.wait(), time.monotonic())
"#,
    );
    assert_eq!(
        result,
        (0, "timeout None 1.0\n-15 1.0\n".into(), String::new())
    );
    assert_eq!(environment.clock.pending_len(), 0);
}

#[test]
fn popen_start_new_session_creates_a_modeled_process_group() {
    let mut environment = Environment::new();
    let result = run(
        &mut environment,
        r#"
import subprocess
process = subprocess.Popen(
    ["sh", "-c", "cat /proc/self/status"],
    stdout=subprocess.PIPE,
    text=True,
    start_new_session=True,
)
stdout, stderr = process.communicate()
identity = [line for line in stdout.split("\n") if line.startswith("NSpgid:") or line.startswith("NSsid:")]
print(process.pid, identity)
"#,
    );
    assert_eq!(
        result,
        (
            0,
            "1235 ['NSpgid:\\t1235', 'NSsid:\\t1235']\n".into(),
            String::new()
        )
    );
}

#[test]
fn communicate_retry_preserves_partial_input_progress() {
    let mut environment = Environment::new();
    let result = run(
        &mut environment,
        r#"
import subprocess
from subprocess import TimeoutExpired
process = subprocess.Popen(["sleep", "10"], stdin=subprocess.PIPE,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
try:
    process.communicate(b"x" * 100000, timeout=1)
except TimeoutExpired:
    print("timed out", process.poll())
process.kill()
stdout, stderr = process.communicate()
print(process.returncode, stdout, stderr)
"#,
    );
    assert_eq!(result.0, 0, "{}", result.2);
    assert_eq!(result.1, "timed out None\n-9 b'' b''\n");
}

#[test]
fn timeout_and_host_capability_requests_fail_explicitly() {
    let mut environment = Environment::new();
    let result = run(
        &mut environment,
        r#"
import subprocess
for action in [
    lambda: subprocess.run(["sleep", "10"], timeout=1),
]:
    try:
        action()
    except Exception as error:
        print(str(error))
shell = subprocess.run("printf modeled-shell", shell=True, capture_output=True, text=True)
print(shell.stdout)
missing = subprocess.run(["/opt/not-a-host-command"], capture_output=True, text=True)
print(missing.returncode, missing.stderr)
"#,
    );
    assert_eq!(result.0, 0, "{}", result.2);
    assert!(result.1.contains("timed out"), "{}", result.1);
    assert!(result.1.contains("modeled-shell"), "{}", result.1);
    assert!(
        result
            .1
            .contains("127 /opt/not-a-host-command: command not found"),
        "{}",
        result.1
    );
}

#[test]
fn subprocess_has_child_identity_and_is_reaped_after_completion() {
    let mut environment = Environment::new();
    let result = run(
        &mut environment,
        r#"
import subprocess
status = subprocess.check_output(["cat", "/proc/self/status"], text=True)
print("Pid:\t1235" in status, "PPid:\t1234" in status)
"#,
    );
    assert_eq!(result, (0, "True True\n".into(), String::new()));
    assert!(environment.processes.get(1_235).is_none());
}

#[test]
fn stderr_modes_are_bounded_and_deterministic() {
    let mut environment = Environment::new();
    let result = run(
        &mut environment,
        r#"
import subprocess
captured = subprocess.run(["sh", "-c", "printf out; /missing"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
merged = subprocess.run(["sh", "-c", "printf out; /missing"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
discarded = subprocess.run(["sh", "-c", "printf out; /missing"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print(captured.stdout, captured.stderr)
print(merged.stdout, merged.stderr)
print(discarded.stdout, discarded.stderr)
"#,
    );
    assert_eq!(result.0, 0, "{}", result.2);
    assert_eq!(
        result.1,
        "b'out' b'/missing: command not found\\n'\nb'out/missing: command not found\\n' None\nNone None\n"
    );
}

#[test]
fn logical_process_exhaustion_fails_before_command_execution() {
    let mut environment = Environment::new();
    for index in 1..MAX_PROCESSES {
        environment
            .processes
            .spawn(
                1_234,
                shellsim::process::ChildPlacement::Inherit,
                &format!("occupied-{index}"),
                "/",
                Default::default(),
            )
            .expect("fill logical process table");
    }
    let result = run(
        &mut environment,
        "import subprocess\nsubprocess.run(['touch', '/must-not-exist'])",
    );
    assert_ne!(result.0, 0);
    assert!(
        result.2.contains("logical process limit exceeded"),
        "{}",
        result.2
    );
    assert!(!environment.vfs.exists("/", "/must-not-exist"));
    assert_eq!(environment.processes.iter().count(), MAX_PROCESSES);
}
