//! Compatibility tests for bounded `grep` behavior and explicit unsupported boundaries.

use shellsim::{Environment, Limits, StopReason};

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
fn pattern_files_filename_modes_filters_and_context_compose() {
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
fn rejects_incoherent_options_and_invalid_text() {
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

    let mut environment = Environment::new();
    environment
        .vfs
        .write("/", "patterns", &[0xff], 0o644)
        .unwrap();
    let (outcome, _, stderr) = environment.run_script_capture("grep -f patterns input");
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
fn pattern_file_growth_obeys_memory_limits() {
    let mut environment = Environment::with_limits(Limits {
        memory: 1024,
        ..Limits::unlimited()
    });
    environment
        .vfs
        .write("/", "patterns", &vec![b'a'; 2048], 0o644)
        .unwrap();
    let (outcome, _, _) = environment.run_script_capture("grep -f patterns input");
    assert_eq!(outcome.exit_status, 137);
    assert_eq!(outcome.stop_reason, Some(StopReason::MemoryExhausted));
}
