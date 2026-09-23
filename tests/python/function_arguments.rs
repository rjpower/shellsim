//! Failure behavior that requires assertions on shellsim's Python diagnostics.

use super::support::run_python;

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
        let (status, stdout, stderr) = run_python(source);
        assert_eq!(status, 2, "{}", String::from_utf8_lossy(&stderr));
        assert!(stdout.is_empty());
        assert!(
            String::from_utf8_lossy(&stderr).contains(expected),
            "expected {expected:?} in {}",
            String::from_utf8_lossy(&stderr)
        );
    }
}

#[test]
fn keyword_only_arguments_reject_positional_values() {
    let (status, stdout, stderr) = run_python("def f(*, value): return value\nf(1)");
    assert_eq!(status, 2);
    assert!(stdout.is_empty());
    assert!(String::from_utf8_lossy(&stderr).contains("takes 0 positional arguments"));
}

#[test]
fn varargs_start_a_new_keyword_only_default_sequence() {
    let source =
        "def f(a=1, *args, scale=2):\n    return a, args, scale\nprint(f(3, 4, 5, scale=6))";
    let (status, stdout, stderr) = run_python(source);

    assert_eq!(status, 0, "{}", String::from_utf8_lossy(&stderr));
    assert_eq!(stdout, b"(3, (4, 5), 6)\n");
    assert!(stderr.is_empty());
}
