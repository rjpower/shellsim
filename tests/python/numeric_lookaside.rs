//! Numeric lookaside tests keep exact builtins fast without bypassing general Python protocols.

use super::support::run_python;

#[test]
fn exact_numbers_preserve_arithmetic_comparison_and_overflow() {
    let source = r#"print(2 + 3, 7 - 11, 6 * 7, -7 // 3, -7 % 3)
print(1.5 + 2, 5 / 2, 2 ** 5, 7 << 2, 12 & 10)
print(3 < 4, 4 <= 4, 5 > 8, True == 1, False != 0)
print(9223372036854775807 + 1)"#;

    let (status, stdout, stderr) = run_python(source);
    assert_eq!(status, 0, "{}", String::from_utf8_lossy(&stderr));
    assert_eq!(
        stdout,
        b"5 -4 42 -3 2\n3.5 2.5 32 28 8\nTrue True False True False\n9223372036854775808\n"
    );
    assert!(stderr.is_empty());
}

#[test]
fn exact_division_by_zero_retains_the_python_error() {
    let (status, stdout, stderr) = run_python("1 // 0");
    assert_eq!(status, 2);
    assert!(stdout.is_empty());
    assert!(String::from_utf8_lossy(&stderr).contains("ZeroDivisionError"));
}

#[test]
fn non_exact_operands_retain_slots_and_reflected_operations() {
    let source = r#"class Left:
    def __add__(self, other):
        return 40 + other

class Right:
    def __radd__(self, other):
        return other + 40
    def __gt__(self, other):
        return True

print(Left() + 2)
print(2 + Right())
print(2 < Right())
print(3 * "ab")"#;

    let (status, stdout, stderr) = run_python(source);
    assert_eq!(status, 0, "{}", String::from_utf8_lossy(&stderr));
    assert_eq!(stdout, b"42\n42\nTrue\nababab\n");
    assert!(stderr.is_empty());
}
