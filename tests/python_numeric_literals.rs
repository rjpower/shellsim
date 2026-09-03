use std::process::Command;

use shellsim::Environment;

fn run(source: &str) -> (i32, Vec<u8>, Vec<u8>) {
    let mut environment = Environment::new();
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let status = shellsim::python::run_python(
        &mut environment,
        &["python3.14".into(), "-c".into(), source.into()],
        Vec::new(),
        &mut stdout,
        &mut stderr,
    );
    (status, stdout, stderr)
}

#[test]
fn decimal_float_literals_and_arithmetic_match_cpython() {
    let source = r#"values = [1.2, .5, 1., 1_000.50_0, 1_2e-1, 1_2E+1]
print(values)
print(1.2 + .5, 1.2 * 2, 5.0 / 2, 1e2 == 100)
"#;
    let simulated = run(source);
    let reference = Command::new("python3.14").arg("-c").arg(source).output();
    if let Ok(reference) = reference {
        assert_eq!(
            simulated,
            (
                reference.status.code().unwrap_or(1),
                reference.stdout,
                reference.stderr
            )
        );
    }
    assert_eq!(
        simulated,
        (
            0,
            b"[1.2, 0.5, 1.0, 1000.5, 1.2, 120.0]\n1.7 2.4 2.5 True\n".to_vec(),
            Vec::new()
        )
    );
}

#[test]
fn integer_overflow_and_malformed_decimal_literals_fail_loudly() {
    for source in [
        "print(9223372036854775808)",
        "print(1__2)",
        "print(1_.2)",
        "print(1._2)",
        "print(1e)",
        "print(1e+_2)",
        "print(.5_)",
    ] {
        let (status, stdout, stderr) = run(source);
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

#[test]
fn exponent_underflow_and_overflow_follow_python_float_semantics() {
    let source = "print(1e9999, 1e-9999)";
    let simulated = run(source);
    assert_eq!(simulated, (0, b"inf 0.0\n".to_vec(), Vec::new()));
}
