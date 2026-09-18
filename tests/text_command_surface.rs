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

    let (status, _, stderr) = run("printf x | sed -n '0p'");
    assert_eq!(status, 2);
    assert!(stderr.contains("at least 1"), "{stderr}");
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
fn sed_rejects_invalid_script_files_and_occurrence_numbers() {
    let mut environment = Environment::new();
    environment
        .vfs
        .write("/", "bad.sed", &[0xff], 0o644)
        .unwrap();
    let (outcome, _, stderr) = environment.run_script_capture("sed -f bad.sed");
    assert_eq!(outcome.exit_status, 2);
    assert!(String::from_utf8_lossy(&stderr).contains("not valid UTF-8"));

    let (status, _, stderr) = run("printf a | sed 's/a/b/99999999999999999999'");
    assert_eq!(status, 2);
    assert!(
        stderr.contains("invalid substitution occurrence"),
        "{stderr}"
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
    assert_eq!(
        run("awk 'BEGIN { printf \"%.4f\\n\", log(8) / log(2) }'"),
        (0, "3.0000\n".into(), String::new())
    );
}

#[test]
fn awk_validates_the_whole_program_before_execution() {
    for (source, expected) in [
        (
            "awk 'BEGIN { print \"must not print\"; unknown(1) }'",
            "unsupported function",
        ),
        (
            "awk 'BEGIN { printf \"%q\", 1 }'",
            "unsupported printf conversion",
        ),
        ("awk 'BEGIN { print 1 > \"out\" }'", "redirection"),
        (
            "printf 'one\\n' | awk -F '[' '{ print $1 }'",
            "invalid field separator",
        ),
        ("awk 'BEGIN { getline value }'", "unsupported keyword"),
        (
            "awk 'BEGIN { NR = 4 }'",
            "assignment to 'NR' is not supported",
        ),
        (
            "awk -v NF=4 'BEGIN { print 1 }'",
            "assignment to 'NF' is not supported",
        ),
    ] {
        let (status, stdout, stderr) = run(source);
        assert_eq!(status, 2, "{source}: {stderr}");
        assert!(stdout.is_empty(), "{source}");
        assert!(stderr.contains(expected), "{source}: {stderr}");
    }
}

#[test]
fn awk_rejects_invalid_utf8_program_files() {
    let mut environment = Environment::new();
    environment
        .vfs
        .write("/", "bad.awk", &[0xff], 0o644)
        .unwrap();
    let (outcome, _, stderr) = environment.run_script_capture("awk -f bad.awk");
    assert_eq!(outcome.exit_status, 2);
    assert!(String::from_utf8_lossy(&stderr).contains("not valid UTF-8"));
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

#[test]
fn generated_text_and_field_growth_obey_resource_limits() {
    for source in [
        "awk 'BEGIN { $1000000000 = 1 }'",
        "awk 'BEGIN { value = \"a\"; while (1) value = value value }'",
        "printf a | sed 's/a/&&&&&&&&&&/g; s/a/&&&&&&&&&&/g; s/a/&&&&&&&&&&/g; s/a/&&&&&&&&&&/g; s/a/&&&&&&&&&&/g'",
    ] {
        let mut environment = Environment::with_limits(Limits {
            memory: 128 * 1024,
            ..Limits::unlimited()
        });
        let (outcome, _, _) = environment.run_script_capture(source);
        assert_eq!(outcome.exit_status, 137, "{source}");
        assert_eq!(outcome.stop_reason, Some(StopReason::MemoryExhausted));
    }

    let mut environment = Environment::with_limits(Limits {
        output: 128,
        ..Limits::unlimited()
    });
    let (outcome, stdout, _) =
        environment.run_script_capture("awk 'BEGIN { printf \"%1000000000s\", \"x\" }'");
    assert_eq!(outcome.exit_status, 137);
    assert_eq!(outcome.stop_reason, Some(StopReason::OutputLimitExceeded));
    assert!(stdout.is_empty());

    let mut environment = Environment::with_limits(Limits {
        output: 128,
        ..Limits::unlimited()
    });
    let (outcome, _, _) = environment.run_script_capture(
        "printf 'abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz\\n' | sed -n 'p;p;p'",
    );
    assert_eq!(outcome.exit_status, 137);
    assert_eq!(outcome.stop_reason, Some(StopReason::OutputLimitExceeded));

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

#[test]
fn recursive_listing_labels_directories_relatively_and_skips_hidden_ones() {
    // An agent running `ls -R` inside a repository must not be handed the contents of `.git`.
    let (status, stdout, stderr) = run(
        "mkdir -p pkg/.cache pkg/sub && touch pkg/top pkg/.cache/junk pkg/sub/leaf && ls -R pkg",
    );
    assert_eq!((status, stderr.as_str()), (0, ""));
    assert_eq!(stdout, "pkg:\nsub\ntop\n\npkg/sub:\nleaf\n");

    let (status, stdout, _) = run(
        "mkdir -p pkg/.cache pkg/sub && touch pkg/top pkg/.cache/junk pkg/sub/leaf && ls -R -A pkg",
    );
    assert_eq!(status, 0);
    assert_eq!(
        stdout,
        "pkg:\n.cache\nsub\ntop\n\npkg/.cache:\njunk\n\npkg/sub:\nleaf\n"
    );
}

#[test]
fn listing_writes_one_name_per_line() {
    // Simulated output is never a terminal, so `ls` uses the format real `ls` uses for a pipe.
    let (status, stdout, stderr) = run("mkdir -p d/b && touch d/a d/c && ls d");
    assert_eq!((status, stderr.as_str()), (0, ""));
    assert_eq!(stdout, "a\nb\nc\n");
    assert_eq!(
        run("mkdir -p d/b && touch d/a d/c && ls d | wc -l").1,
        "3\n"
    );
}

#[test]
fn a_long_listing_describes_a_named_file_as_well_as_a_directory() {
    let (status, stdout, stderr) = run("touch f && chmod +x f && ls -l f");
    assert_eq!((status, stderr.as_str()), (0, ""));
    assert!(stdout.starts_with("-rwxr-xr-x "), "{stdout}");
    assert!(stdout.trim_end().ends_with(" f"), "{stdout}");
}
