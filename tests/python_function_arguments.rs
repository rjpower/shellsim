use shellsim::Environment;

fn run(source: &str) -> (i32, Vec<u8>, Vec<u8>) {
    let mut environment = Environment::new();
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let argv = vec!["python3.14".into(), "-c".into(), source.into()];
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
fn defaults_are_evaluated_at_definition_and_shared_when_mutable() {
    let source = r#"seed = 1
def combine(left, right=seed, total=seed + 2):
    return left + right + total
seed = 100
print(combine(3), combine(3, 4), combine(left=3, total=9, right=8))

def append(value, bucket=[]):
    bucket.append(value)
    return bucket
print(append(1), append(2))

increment = lambda value, amount=2: value + amount
print(increment(5), increment(5, amount=4))"#;
    let (status, stdout, stderr) = run(source);
    assert_eq!(status, 0, "{}", String::from_utf8_lossy(&stderr));
    assert_eq!(stdout, b"7 10 20\n[1, 2] [1, 2]\n7 9\n");
    assert!(stderr.is_empty());
}

#[test]
fn argument_binding_reports_python_style_errors() {
    for (source, expected) in [
        (
            "def f(a, b=2):\n    return a + b\nprint(f())",
            "missing required argument",
        ),
        (
            "def f(a, b=2):\n    return a + b\nprint(f(1, 2, 3))",
            "takes 2 positional arguments",
        ),
        (
            "def f(a, b=2):\n    return a + b\nprint(f(1, c=3))",
            "unexpected keyword argument",
        ),
        (
            "def f(a, b=2):\n    return a + b\nprint(f(1, a=3))",
            "multiple values for argument",
        ),
    ] {
        let (status, _stdout, stderr) = run(source);
        assert_eq!(status, 2, "{}", String::from_utf8_lossy(&stderr));
        assert!(
            String::from_utf8_lossy(&stderr).contains(expected),
            "expected {expected:?} in {}",
            String::from_utf8_lossy(&stderr)
        );
    }
}

#[test]
fn list_and_tuple_sequence_operations_create_new_values() {
    let source = r#"left = [1, 2]
right = left + [3]
repeated = ("x",) * 3
print(left, right, repeated)
right.append(4)
print(left, right)"#;
    let (status, stdout, stderr) = run(source);
    assert_eq!(status, 0, "{}", String::from_utf8_lossy(&stderr));
    assert_eq!(
        stdout,
        b"[1, 2] [1, 2, 3] ('x', 'x', 'x')\n[1, 2] [1, 2, 3, 4]\n"
    );
    assert!(stderr.is_empty());
}

#[test]
fn variadic_positional_arguments_are_bound_as_a_tuple() {
    let source = r#"def total(prefix, *values):
    return prefix + sum(values)
print(total(10), total(1, 2, 3, 4))
print((lambda *values: sum(values))(5, 6))"#;
    let (status, stdout, stderr) = run(source);
    assert_eq!(status, 0, "{}", String::from_utf8_lossy(&stderr));
    assert_eq!(stdout, b"10 10\n11\n");
    assert!(stderr.is_empty());
}

#[test]
fn keyword_only_arguments_use_the_ordinary_argument_binder() {
    let source = r#"def configure(prefix, *, window, scale=2):
    return prefix + window * scale
def collect(prefix, *values, suffix):
    return prefix + sum(values) + suffix
print(configure(1, window=3))
print(collect(1, 2, 3, suffix=4))
print((lambda *, value=5: value)(value=7))"#;
    let (status, stdout, stderr) = run(source);
    assert_eq!(status, 0, "{}", String::from_utf8_lossy(&stderr));
    assert_eq!(stdout, b"7\n10\n7\n");
    assert!(stderr.is_empty());

    let (_, _, stderr) = run("def f(*, value): return value\nf(1)");
    assert!(String::from_utf8_lossy(&stderr).contains("takes 0 positional arguments"));
}
