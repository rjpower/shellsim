//! Numeric parser failures and shellsim-specific integer boundaries.

use super::support::run_python;

#[test]
fn arbitrary_precision_integers_cross_host_index_boundaries_safely() {
    let source = r#"x = 9223372036854775808
print(x)
print(9223372036854775807 + 1)
print(-(-9223372036854775808))
print(x * x)
print(-x // 3, -x % 3)
print(x == 9223372036854775808, x > 1)
print(9007199254740993 == 9007199254740992.0, 9007199254740993 > 9007199254740992.0)
print(abs(-x), sum([x, x]), int(x), float(x) > 9e18)
print([1] * -9223372036854775809, -9223372036854775809 * (1,))
"#;
    assert_eq!(run_python(source), (0, b"9223372036854775808\n9223372036854775808\n9223372036854775808\n85070591730234615865843651857942052864\n-3074457345618258603 1\nTrue True\nFalse True\n9223372036854775808 18446744073709551616 9223372036854775808 True\n[] ()\n".to_vec(), Vec::new()));
}

#[test]
fn malformed_decimal_literals_fail_loudly() {
    for source in [
        "print(1__2)",
        "print(1_.2)",
        "print(1._2)",
        "print(1e)",
        "print(1e+_2)",
        "print(.5_)",
    ] {
        let (status, stdout, stderr) = run_python(source);
        assert_ne!(status, 0, "accepted invalid numeric source: {source}");
        assert!(
            stdout.is_empty(),
            "wrote output for invalid source: {source}"
        );
        assert!(
            !stderr.is_empty(),
            "failed silently for invalid source: {source}"
        );
    }
}
