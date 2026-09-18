//! Shared runners for Python integration modules.

use shellsim::{python, Environment};

pub type Captured = (i32, Vec<u8>, Vec<u8>);
pub type CapturedText = (i32, String, String);

pub fn run_python(source: &str) -> Captured {
    run_python_in(&mut Environment::new(), source)
}

pub fn run_python_in(environment: &mut Environment, source: &str) -> Captured {
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let status = python::run_python(
        environment,
        &["python3.14".into(), "-c".into(), source.into()],
        Vec::new(),
        &mut stdout,
        &mut stderr,
    );
    (status, stdout, stderr)
}

pub fn run_python_text(source: &str) -> CapturedText {
    let (status, stdout, stderr) = run_python(source);
    (
        status,
        String::from_utf8(stdout).expect("Python stdout is UTF-8"),
        String::from_utf8(stderr).expect("Python stderr is UTF-8"),
    )
}

pub fn run_python_text_in(environment: &mut Environment, source: &str) -> CapturedText {
    let (status, stdout, stderr) = run_python_in(environment, source);
    (
        status,
        String::from_utf8(stdout).expect("Python stdout is UTF-8"),
        String::from_utf8(stderr).expect("Python stderr is UTF-8"),
    )
}

pub fn run_shell(source: &str) -> Captured {
    let mut environment = Environment::new();
    let (outcome, stdout, stderr) = environment.run_script_capture(source);
    (outcome.exit_status, stdout, stderr)
}
