use std::process::Command;

use shellsim::Environment;

fn run(source: &str) -> (i32, Vec<u8>, Vec<u8>) {
    let mut environment = Environment::new();
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
fn comprehensions_support_clauses_filters_and_container_kinds() {
    let source = r#"numbers = [1, 2, 3, 4]
print([n * 2 for n in numbers if n % 2 == 0])
print([(left, right) for left in [1, 2] for right in [3, 4] if left < right])
print({n * n for n in numbers if n > 2})
print({str(n): n * n for n in numbers if n != 2})"#;
    let (status, stdout, stderr) = run(source);
    assert_eq!(status, 0, "{}", String::from_utf8_lossy(&stderr));
    assert_eq!(
        stdout,
        b"[4, 8]\n[(1, 3), (1, 4), (2, 3), (2, 4)]\n{9, 16}\n{'1': 1, '3': 9, '4': 16}\n"
    );
    assert!(stderr.is_empty());
}

#[test]
fn comprehension_bindings_are_isolated_and_closures_are_lexical() {
    let source = r#"item = 99
def shifted(offset):
    return [item + offset for item in range(3)]
print(shifted(10), item)
print([value for value in range(3)], item)"#;
    let (status, stdout, stderr) = run(source);
    assert_eq!(status, 0, "{}", String::from_utf8_lossy(&stderr));
    assert_eq!(stdout, b"[10, 11, 12] 99\n[0, 1, 2] 99\n");
    assert!(stderr.is_empty());
}

#[test]
fn generator_expressions_are_metered_iterables_with_comprehension_scope() {
    let source = r#"item = 50
total = sum(value * value for value in range(5) if value % 2)
print(total, item)
print(list(value + 1 for value in [1, 2, 3]))"#;
    let (status, stdout, stderr) = run(source);
    assert_eq!(status, 0, "{}", String::from_utf8_lossy(&stderr));
    assert_eq!(stdout, b"10 50\n[2, 3, 4]\n");
    assert!(stderr.is_empty());

    if let Ok(reference) = Command::new("python3").arg("-c").arg(source).output() {
        assert_eq!(status, reference.status.code().unwrap_or(1));
        assert_eq!(stdout, reference.stdout);
        assert_eq!(stderr, reference.stderr);
    }
}
