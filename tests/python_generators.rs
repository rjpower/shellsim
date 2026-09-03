use std::process::Command;

fn run(source: &str) -> (i32, Vec<u8>, Vec<u8>) {
    let mut environment = shellsim::Environment::new();
    let argv = vec!["python3.14".into(), "-c".into(), source.into()];
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let status = shellsim::python::run_python(
        &mut environment,
        &argv,
        Vec::new(),
        &mut stdout,
        &mut stderr,
    );
    (status, stdout, stderr)
}

#[test]
fn generator_frames_suspend_and_resume_in_order() {
    let source = r#"def values(limit):
    current = 0
    while current < limit:
        yield current * 2
        current += 1

items = values(3)
print(next(items), next(items))
print(list(items))
print(next(items, "done"))
"#;
    let (status, stdout, stderr) = run(source);
    assert_eq!(status, 0, "{}", String::from_utf8_lossy(&stderr));
    assert_eq!(stdout, b"0 2\n[4]\ndone\n");
    assert!(stderr.is_empty());

    if let Ok(reference) = Command::new("python3").arg("-c").arg(source).output() {
        assert_eq!(stdout, reference.stdout);
        assert_eq!(stderr, reference.stderr);
        assert_eq!(status, reference.status.code().unwrap_or(1));
    }
}

#[test]
fn generator_closure_keeps_lexical_state_between_yields() {
    let source = r#"def make(step):
    value = 1
    def sequence():
        nonlocal value
        yield value
        value += step
        yield value
    return sequence

items = make(4)()
print(next(items), next(items), next(items, None))
"#;
    let (status, stdout, stderr) = run(source);
    assert_eq!(status, 0, "{}", String::from_utf8_lossy(&stderr));
    assert_eq!(stdout, b"1 5 None\n");
    assert!(stderr.is_empty());
}

#[test]
fn generator_resource_usage_is_metered() {
    let source = r#"def endless():
    value = 0
    while True:
        yield value
        value += 1

for value in endless():
    print(value)
"#;
    let (status, _stdout, _stderr) = run(source);
    assert_eq!(status, 137);
}
