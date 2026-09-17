//! Compatibility strategy: exercise ordinary composed programs and explicit failure boundaries.
//! Host utilities are not invoked; exact outputs are stable contracts for the simulated commands.

use shellsim::interp::{Environment, Interp};
use shellsim::{Limits, StopReason};

fn run(source: &str) -> (i32, String, String) {
    let mut environment = Interp::new();
    let (outcome, stdout, stderr) = environment.run_script_capture(source);
    (
        outcome.exit_status,
        String::from_utf8_lossy(&stdout).into_owned(),
        String::from_utf8_lossy(&stderr).into_owned(),
    )
}

#[test]
fn grep_pattern_files_filename_modes_filters_and_context_compose() {
    assert_eq!(
        run("printf 'alpha\\n' > patterns; printf 'alpha\\nbeta\\n' > one; printf 'beta\\n' > two; grep -H -f patterns one two; grep -L alpha one two"),
        (0, "one:alpha\ntwo\n".into(), String::new())
    );
    let (status, stdout, stderr) = run(
        "mkdir -p src vendor; printf 'zero\\nmatch\\nafter\\n' > src/a.rs; printf 'match\\n' > src/a.txt; printf 'match\\n' > vendor/b.rs; grep -rn -A1 --include='*.rs' --exclude-dir=vendor match .",
    );
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, "/src/a.rs:2:match\n/src/a.rs-3-after\n");
    assert!(stderr.is_empty(), "{stderr}");
}

#[test]
fn grep_rejects_incoherent_option_combinations_and_invalid_text() {
    let (status, _, stderr) = run("printf x | grep -A1 -o x");
    assert_eq!(status, 2);
    assert!(stderr.contains("cannot be combined"), "{stderr}");

    let mut environment = Environment::new();
    environment
        .vfs
        .write("/", "bad", &[0xff, b'\n'], 0o644)
        .unwrap();
    let (outcome, _, stderr) = environment.run_script_capture("grep x bad");
    assert_eq!(outcome.exit_status, 2);
    assert!(String::from_utf8_lossy(&stderr).contains("not valid UTF-8"));

    let (status, _, stderr) = run("printf 'a\\0b' | grep a");
    assert_eq!(status, 2);
    assert!(stderr.contains("binary input is not supported"), "{stderr}");

    let (status, _, stderr) = run("printf a | grep --color=auto a");
    assert_eq!(status, 2);
    assert!(stderr.contains("unsupported color mode"), "{stderr}");

    assert_eq!(
        run("printf '' > empty; printf 'x\\n' | grep -f empty"),
        (1, String::new(), String::new())
    );
}

#[test]
fn sed_handles_delimiters_addresses_ranges_and_common_commands() {
    assert_eq!(
        run("printf 'a;b\\nstart\\nmid\\nend\\n' | sed -n 's/;/x/p; /start/,/end/p'"),
        (0, "axb\nstart\nmid\nend\n".into(), String::new())
    );
    assert_eq!(
        run("printf 'a\\nb\\nc\\n' | sed '2i before; 2c changed; 3q'"),
        (0, "a\nbefore\nchanged\nc\n".into(), String::new())
    );
    assert_eq!(
        run("printf 'abc\\n' | sed 'y/ac/AC/;='"),
        (0, "1\nAbC\n".into(), String::new())
    );
    assert_eq!(
        run("printf 'a\\nb\\nc\\n' | sed '1,2c changed; /changed/!s/c/C/'"),
        (0, "changed\nC\n".into(), String::new())
    );
}

#[test]
fn sed_script_files_and_in_place_writes_preserve_mode() {
    let mut environment = Environment::new();
    environment
        .vfs
        .write("/", "script.sed", b"s/a/b/g\n", 0o644)
        .unwrap();
    environment.vfs.write("/", "tool", b"aa\n", 0o755).unwrap();
    let (outcome, stdout, stderr) =
        environment.run_script_capture("sed -i -f script.sed tool; cat tool");
    assert_eq!(
        outcome.exit_status,
        0,
        "{}",
        String::from_utf8_lossy(&stderr)
    );
    assert_eq!(stdout, b"bb\n");
    assert_eq!(
        environment.vfs.metadata("/", "tool", false).unwrap().mode,
        0o755
    );
}

#[test]
fn awk_supports_basic_control_flow_arrays_and_functions() {
    assert_eq!(
        run("awk 'BEGIN { for (i = 1; i <= 3; i++) { squares[i] = i * i } total = 0; for (key in squares) { if (key == 2) continue; total += squares[key] } print total, substr(\"hello\", 2, 3), index(\"abc\", \"b\") }'"),
        (0, "10 ell 2\n".into(), String::new())
    );
    assert_eq!(
        run("awk 'BEGIN { value = \"a,b,c\"; n = split(value, parts, \",\"); changed = gsub(/,/, \"-\", value); print n, parts[2], value, changed, match(\"abc\", /b/), RSTART, RLENGTH }'"),
        (0, "3 b a-b-c 2 2 2 1\n".into(), String::new())
    );
    assert_eq!(
        run("printf 'a1\\na2\\n' > one; printf 'b1\\nb2\\n' > two; awk '{ print; nextfile }' one two"),
        (0, "a1\nb1\n".into(), String::new())
    );
    assert_eq!(
        run("printf 'alpha\\ngem\\n' | awk '$0 ~ /^a/ { print } $0 !~ /a$/ { print \"no-a\" }'"),
        (0, "alpha\nno-a\n".into(), String::new())
    );
}

#[test]
fn awk_validates_the_whole_program_before_execution() {
    let (status, stdout, stderr) = run("awk 'BEGIN { print \"must not print\"; unknown(1) }'");
    assert_eq!(status, 2);
    assert!(stdout.is_empty());
    assert!(stderr.contains("unsupported function"), "{stderr}");

    let (status, stdout, stderr) = run("awk 'BEGIN { printf \"%q\", 1 }'");
    assert_eq!(status, 2);
    assert!(stdout.is_empty());
    assert!(stderr.contains("unsupported printf conversion"), "{stderr}");

    let (status, stdout, stderr) = run("awk 'BEGIN { print 1 > \"out\" }'");
    assert_eq!(status, 2);
    assert!(stdout.is_empty());
    assert!(stderr.contains("redirection"), "{stderr}");

    let (status, stdout, stderr) = run("printf 'one\\n' | awk -F '[' '{ print $1 }'");
    assert_eq!(status, 2);
    assert!(stdout.is_empty());
    assert!(stderr.contains("invalid field separator"), "{stderr}");
}

#[test]
fn awk_command_line_special_variables_affect_execution() {
    assert_eq!(
        run("printf 'a,b\\n' | awk -v FS=, -v OFS=: '{ print $1, $2 }'"),
        (0, "a:b\n".into(), String::new())
    );
}

#[test]
fn awk_loops_consume_modeled_cpu_fuel() {
    let mut environment = Environment::with_limits(Limits {
        cpu: 500,
        ..Limits::unlimited()
    });
    let (outcome, stdout, _) =
        environment.run_script_capture("awk 'BEGIN { while (1) { value++ } }'");
    assert_eq!(outcome.exit_status, 137);
    assert_eq!(outcome.stop_reason, Some(StopReason::CpuExhausted));
    assert_eq!(outcome.usage.cpu_used, 500);
    assert!(stdout.is_empty());
}
