//! Compatibility and isolation tests for Python subprocesses over logical process execution.
//!
//! These tests use only registered commands and VFS scripts. They intentionally probe captured
//! and inherited streams plus process-local cwd/environment state at the public Python API.

use shellsim::{process::MAX_PROCESSES, python, Environment};

fn run(environment: &mut Environment, source: &str) -> (i32, String, String) {
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let status = python::run_python(
        environment,
        &["python3.14".into(), "-c".into(), source.into()],
        Vec::new(),
        &mut stdout,
        &mut stderr,
    );
    (
        status,
        String::from_utf8_lossy(&stdout).into_owned(),
        String::from_utf8_lossy(&stderr).into_owned(),
    )
}

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
            .spawn(1_234, &format!("occupied-{index}"), "/", Default::default())
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
