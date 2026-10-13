//! Deeply nested guest input must fail explicitly instead of overflowing the host stack.
//!
//! A stack overflow aborts the whole host process, including a Python process that embeds
//! shellsim. Each case runs on a thread with a deliberately small stack, so a recursion that
//! is neither bounded nor moved onto a heap segment aborts this test binary on every host.

use shellsim::Environment;

/// Small enough that an unguarded recursion of a few hundred debug-build frames overflows.
const SMALL_STACK: usize = 256 * 1024;
const DEEP: usize = 20_000;

fn run_on_small_stack(source: String) -> (i32, String, String) {
    std::thread::Builder::new()
        .stack_size(SMALL_STACK)
        .spawn(move || {
            let mut environment = Environment::new();
            let (outcome, stdout, stderr) = environment.run_script_capture(&source);
            (
                outcome.exit_status,
                String::from_utf8_lossy(&stdout).into_owned(),
                String::from_utf8_lossy(&stderr).into_owned(),
            )
        })
        .unwrap()
        .join()
        .unwrap()
}

#[test]
fn ordinary_script_fits_the_small_test_stack() {
    let (status, stdout, _) = run_on_small_stack("echo ok; printf '1\\n'".into());
    assert_eq!((status, stdout.as_str()), (0, "ok\n1\n"));
}

/// Inputs nested past every limit, with the stderr fragment that names the explicit failure.
fn rejected_cases() -> Vec<(&'static str, String, &'static str)> {
    vec![
        (
            "shell groups",
            "{ ".repeat(DEEP) + "true; " + &"} ".repeat(DEEP),
            "nested too deeply",
        ),
        (
            "shell if",
            "if true; then ".repeat(DEEP) + "true; " + &"fi; ".repeat(DEEP),
            "nested too deeply",
        ),
        (
            "shell && chain",
            "true && ".repeat(DEEP) + "true",
            "nested too deeply",
        ),
        (
            "shell time chain",
            "time ".repeat(DEEP) + "true",
            "nested too deeply",
        ),
        (
            "parameter expansion",
            "echo ".to_string() + &"${x:-".repeat(DEEP) + "a" + &"}".repeat(DEEP),
            "nested too deeply",
        ),
        (
            "test negation",
            "[ ".to_string() + &"! ".repeat(DEEP) + "x ]",
            "nested too deeply",
        ),
        (
            "expr parentheses",
            "expr ".to_string() + &"\\( ".repeat(DEEP) + "1 " + &"\\) ".repeat(DEEP),
            "nested too deeply",
        ),
        (
            "find parentheses",
            "find / ".to_string() + &"\\( ".repeat(DEEP) + "-name x " + &"\\) ".repeat(DEEP),
            "nested too deeply",
        ),
        (
            "find implicit -a chain",
            "find / ".to_string() + &"-name x ".repeat(DEEP),
            "nested too deeply",
        ),
        (
            "awk negation",
            format!("awk 'BEGIN {{ print {}1 }}'", "!".repeat(DEEP)),
            "nested too deeply",
        ),
        (
            "awk blocks",
            format!("awk 'BEGIN {{ {}{} }}'", "{".repeat(DEEP), "}".repeat(DEEP)),
            "nested too deeply",
        ),
        (
            "awk + chain",
            format!("awk 'BEGIN {{ print {}1 }}'", "1+".repeat(DEEP)),
            "nested too deeply",
        ),
    ]
}

#[test]
fn deep_nesting_fails_explicitly_on_a_small_stack() {
    for (name, source, expected) in rejected_cases() {
        let (status, _, stderr) = run_on_small_stack(source);
        assert_ne!(status, 0, "{name} succeeded");
        assert!(
            stderr.contains(expected),
            "{name}: expected {expected:?} in stderr {stderr:?}"
        );
    }
}

/// Nesting just inside the limits still works when the stack is small, so the limits are not a
/// substitute for stack growth and derived drops of bounded trees fit.
#[test]
fn nesting_within_limits_succeeds_on_a_small_stack() {
    let cases = [
        ("{ ".repeat(400) + "echo ok; " + &"} ".repeat(400), "ok\n"),
        ("true && ".repeat(900) + "echo ok", "ok\n"),
        (
            "echo ".to_string() + &"${x:-".repeat(900) + "ok" + &"}".repeat(900),
            "ok\n",
        ),
        (
            format!("awk 'BEGIN {{ print {}1 }}'", "1+".repeat(900)),
            "901\n",
        ),
    ];
    for (source, expected) in cases {
        let (status, stdout, stderr) = run_on_small_stack(source.clone());
        assert_eq!(
            (status, stdout.as_str()),
            (0, expected),
            "{}…: {stderr}",
            &source[..40]
        );
    }
}
