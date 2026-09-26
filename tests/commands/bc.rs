//! Integration coverage for the typed `/usr/bin/bc` command: arbitrary-precision arithmetic,
//! control flow, file operands, and the documented unsupported frontier.

use shellsim::{Environment, Limits};

fn run(source: &str) -> (i32, String, String) {
    let mut environment = Environment::new();
    let (outcome, stdout, stderr) = environment.run_script_capture(source);
    (
        outcome.exit_status,
        String::from_utf8_lossy(&stdout).into_owned(),
        String::from_utf8_lossy(&stderr).into_owned(),
    )
}

#[test]
fn arithmetic_and_scale_from_stdin() {
    assert_eq!(
        run("printf '1 + 2\\n' | bc"),
        (0, "3\n".into(), String::new())
    );
    assert_eq!(
        run("printf 'scale = 5\\n22/7\\n' | bc"),
        (0, "3.14285\n".into(), String::new())
    );
}

#[test]
fn assignment_statements_print_nothing() {
    assert_eq!(
        run("printf 'a = 5\\na\\n' | bc"),
        (0, "5\n".into(), String::new())
    );
}

#[test]
fn control_flow_loops_and_conditionals() {
    assert_eq!(
        run("printf 'i = 0\\nwhile (i < 5) { i += 1 }\\ni\\n' | bc"),
        (0, "5\n".into(), String::new())
    );
    assert_eq!(
        run("printf 'for (i = 0; i < 3; i++) i\\n' | bc"),
        (0, "0\n1\n2\n".into(), String::new())
    );
    assert_eq!(
        // A bc string literal prints raw with no added newline (see
        // `string_literals_print_raw_bytes_with_no_added_newline`), so "yes" has none here.
        run("printf 'if (1 < 2) \"yes\"\\n' | bc"),
        (0, "yes".into(), String::new())
    );
}

#[test]
fn builtin_functions_sqrt_length_scale() {
    assert_eq!(
        run("printf 'scale = 4\\nsqrt(2)\\n' | bc"),
        (0, "1.4142\n".into(), String::new())
    );
    assert_eq!(
        run("printf 'length(12345)\\n' | bc"),
        (0, "5\n".into(), String::new())
    );
    assert_eq!(
        run("printf 'scale = 3\\nscale(1.5)\\n' | bc"),
        (0, "1\n".into(), String::new())
    );
}

#[test]
fn program_reads_file_operands_before_stdin() {
    assert_eq!(
        run("printf 'a = 3\\n' > /prog.bc\nprintf 'a + 1\\n' | bc /prog.bc"),
        (0, "4\n".into(), String::new())
    );
}

#[test]
fn divide_by_zero_reports_a_runtime_error_and_continues() {
    let (status, stdout, stderr) = run("printf '1 / 0\\n2 + 2\\n' | bc");
    assert_eq!(status, 0);
    assert_eq!(stdout, "4\n");
    assert!(stderr.to_lowercase().contains("divide by zero"), "{stderr}");
}

#[test]
fn non_decimal_bases_are_explicitly_rejected() {
    let (_, _, stderr) = run("printf 'ibase = 16\\n' | bc");
    assert!(stderr.contains("base 10"), "{stderr}");
}

#[test]
fn define_and_arrays_are_explicitly_unsupported() {
    let (status, _, stderr) = run("printf 'define f(x) { return x }\\n' | bc");
    assert_eq!(status, 1);
    assert!(stderr.contains("not supported"), "{stderr}");

    let (status, _, stderr) = run("printf 'a[0] = 1\\n' | bc");
    assert_eq!(status, 1);
    assert!(
        stderr.contains("not supported") || stderr.contains("array"),
        "{stderr}"
    );
}

#[test]
fn math_library_flag_is_explicitly_rejected() {
    let (status, _, stderr) = run("printf '1\\n' | bc -l");
    assert_eq!(status, 2);
    assert!(stderr.contains("-l"), "{stderr}");
}

#[test]
fn oversize_numbers_are_rejected_with_a_bounded_diagnostic() {
    // Route the oversize literal through a VFS file rather than a shell argv/pipeline so the
    // test exercises bc's own digit cap, not an unrelated pipeline argv-size limit.
    let mut environment = Environment::new();
    let digits = "9".repeat(200_001);
    environment
        .vfs
        .write("/", "/big.bc", format!("{digits}\n").as_bytes(), 0o644)
        .unwrap();
    let (outcome, _, stderr) = environment.run_script_capture("bc /big.bc");
    assert_eq!(outcome.exit_status, 1);
    let stderr = String::from_utf8_lossy(&stderr);
    assert!(stderr.contains("digit"), "{stderr}");
}

#[test]
fn long_results_wrap_at_seventy_columns() {
    let (_, stdout, _) = run("printf '10^100\\n' | bc");
    assert!(stdout.contains("\\\n"), "{stdout}");
}

#[test]
fn line_wrap_matches_gnu_bcs_68_digits_per_continuation_line() {
    // GNU bc 1.07.1's default BC_LINE_LENGTH (70) counts the trailing `\` and `\n`, so each
    // continuation line carries 68 digits; verified directly against `echo '2^1000' | bc`.
    let (status, stdout, _) = run("printf '2^1000\\n' | bc");
    assert_eq!(status, 0);
    let first_line = stdout.split("\\\n").next().unwrap();
    assert_eq!(first_line.len(), 68, "{stdout}");
}

#[test]
fn string_literals_print_raw_bytes_with_no_added_newline() {
    // Within the shell's single quotes `\\n` is two literal backslash-n chars; printf's own
    // escape processing turns bare `\n` into a real newline before bc ever sees the program.
    // bc's string literals do not interpret escapes and print no trailing newline of their own,
    // so the two backslash-n characters survive into the output verbatim, back to back with the
    // next statement's output.
    let (status, stdout, stderr) = run("printf '\"hi\\\\n\"\\n\"a\"\\n1\\n' | bc");
    assert_eq!(status, 0);
    assert_eq!(stdout, "hi\\na1\n");
    assert_eq!(stderr, "");
}

#[test]
fn negative_exponent_is_computed_as_a_reciprocal() {
    assert_eq!(
        run("printf '2^-1\\n' | bc"),
        (0, "0\n".into(), String::new())
    );
    assert_eq!(
        run("printf 'scale=3\\n2^-2\\n' | bc"),
        (0, ".250\n".into(), String::new())
    );
}

#[test]
fn divide_by_zero_message_matches_gnu_bc_format() {
    let (status, stdout, stderr) = run("printf '1/0\\n2+2\\n' | bc");
    assert_eq!(status, 0);
    assert_eq!(stdout, "4\n");
    assert_eq!(stderr, "Runtime error (func=(main)): Divide by zero\n");
}

#[test]
fn cpu_budget_bounds_an_infinite_bc_loop() {
    let mut environment = Environment::with_limits(Limits {
        cpu: 200,
        ..Limits::unlimited()
    });
    let (outcome, _, _) = environment.run_script_capture("printf 'while (1) { }\\n' | bc");
    assert_eq!(outcome.exit_status, 137);
}
