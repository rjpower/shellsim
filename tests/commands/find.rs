//! Compatibility strategy for `find`: ordinary predicate composition and explicit frontiers.
//!
//! `find` runs as a resumable native process image (see `src/program/find.rs`): every top-level
//! invocation below is dispatched through `commands::dispatch`'s native-process path, which spawns
//! it as its own child rather than running it in the calling shell, so these tests also cover
//! `-exec`/`-execdir` spawning real virtual children instead of nested synchronous dispatch.

use shellsim::interp::{Environment, Interp};
use shellsim::{Limits, StopReason};

fn run(environment: &mut Interp, source: &str) -> (i32, Vec<u8>, String) {
    let (outcome, stdout, stderr) = environment.run_script_capture(source);
    (
        outcome.exit_status,
        stdout,
        String::from_utf8_lossy(&stderr).into_owned(),
    )
}

#[test]
fn grouped_alternatives_implicit_and_and_negation_compose() {
    let mut environment = Environment::new();
    let (status, stdout, stderr) = run(
        &mut environment,
        "mkdir -p src/nested; touch src/a.rs src/b.py src/c.txt src/nested/d.rs; find src \\( -name '*.rs' -o -name '*.py' \\) -type f",
    );
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, b"src/a.rs\nsrc/b.py\nsrc/nested/d.rs\n");

    let (status, stdout, stderr) = run(&mut environment, "find src -type f ! -name '*.rs' -print");
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, b"src/b.py\nsrc/c.txt\n");
}

#[test]
fn depth_path_and_nul_printing_match_common_find_usage() {
    let mut environment = Environment::new();
    let (status, stdout, stderr) = run(
        &mut environment,
        "mkdir -p root/a/b; touch root/top root/a/mid root/a/b/deep; find root -mindepth 1 -maxdepth 2 -path 'root/a*' -print0",
    );
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, b"root/a\0root/a/b\0root/a/mid\0");

    let (status, stdout, stderr) = run(
        &mut environment,
        "mkdir -p /work/target/project/src; touch /work/target/project/src/a.rs; cd /work/target/project; find . -type f ! -path '*/target/*'",
    );
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, b"./src/a.rs\n");

    let (status, stdout, stderr) = run(
        &mut environment,
        "cd /; mkdir root-entry; find . -mindepth 1 -maxdepth 1 -name root-entry",
    );
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, b"./root-entry\n");
}

#[test]
fn unsupported_or_invalid_predicates_fail_before_walking() {
    let mut environment = Environment::new();
    for (source, expected) in [
        ("find . -printf '%p\\n'", "unsupported predicate"),
        ("find . \\( -name x", "missing ')'"),
        ("find . -maxdepth nope", "invalid argument"),
    ] {
        let (status, stdout, stderr) = run(&mut environment, source);
        assert_eq!(status, 2, "{source}: {stderr}");
        assert!(stdout.is_empty(), "{source}");
        assert!(stderr.contains(expected), "{source}: {stderr}");
    }
}

#[test]
fn find_output_obeys_the_environment_limit() {
    let mut environment = Environment::with_limits(Limits {
        output: 16,
        ..Limits::unlimited()
    });
    let (outcome, _, _) = environment.run_script_capture(
        "mkdir -p directory; touch directory/one directory/two directory/three; find directory -type f",
    );
    assert_eq!(outcome.exit_status, 137);
    assert_eq!(outcome.stop_reason, Some(StopReason::OutputLimitExceeded));
}

#[test]
fn metadata_predicates_and_delete_cover_common_cleanup_usage() {
    let mut environment = Environment::new();
    let (status, stdout, stderr) = run(
        &mut environment,
        "mkdir -p root/empty root/full; : > root/zero; printf data > root/full/data; chmod 600 root/full/data; find root -empty -print; find root -type f -size 4c -perm 600; find root/empty -delete; test ! -e root/empty",
    );
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, b"root/empty\nroot/zero\nroot/full/data\n");
}

#[test]
fn size_units_round_nonempty_files_up() {
    let mut environment = Environment::new();
    let (status, stdout, stderr) = run(
        &mut environment,
        "mkdir root; : > root/zero; printf x > root/one; find root -type f -size 1; find root -type f -size -1",
    );
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, b"root/one\nroot/zero\n");
}

#[test]
fn failed_delete_is_an_operational_error() {
    let mut environment = Environment::new();
    let (status, stdout, stderr) = run(
        &mut environment,
        "mkdir -p root/child; find root -maxdepth 0 -delete",
    );
    assert_eq!(status, 1);
    assert!(stdout.is_empty());
    assert!(stderr.contains("cannot delete 'root'"), "{stderr}");
    assert!(stderr.contains("Directory not empty"), "{stderr}");
}

#[test]
fn virtual_age_predicates_use_elapsed_simulated_time() {
    let mut environment = Environment::new();
    let (status, stdout, stderr) = run(
        &mut environment,
        "mkdir root; touch root/old; sleep 61; touch root/new; find root -type f -mmin +0; find root -type f -mmin 0",
    );
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, b"root/old\nroot/new\n");
}

#[test]
fn exec_dispatches_modeled_commands_immediately_or_in_a_batch() {
    let mut environment = Environment::new();
    let (status, stdout, stderr) = run(
        &mut environment,
        r"mkdir root; touch root/a root/b; find root -type f -exec printf '<%s>' {} \;; echo; find root -type f -exec printf '[%s]' {} +; echo; find root -type f -exec false \; -print",
    );
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, b"<root/a><root/b>\n[root/a][root/b]\n");
}

#[test]
fn exec_children_are_separate_processes_not_the_calling_shell() {
    let mut environment = Environment::new();
    let (status, _stdout, stderr) = run(
        &mut environment,
        r"mkdir root; touch root/a; find root -type f -exec true \;",
    );
    assert_eq!(status, 0, "{stderr}");
    let events = environment.invocations.events();
    let find_event = events
        .iter()
        .find(|event| event.argv.first().map(String::as_str) == Some("find"))
        .expect("find invocation recorded");
    assert_ne!(find_event.pid, 1_234, "find must run as its own process");
    let exec_event = events
        .iter()
        .find(|event| event.argv == ["true"])
        .expect("-exec child invocation recorded");
    assert_ne!(
        exec_event.pid, 1_234,
        "-exec child must not run as the shell process"
    );
    assert_ne!(
        exec_event.pid, find_event.pid,
        "-exec child must not run as find's own process"
    );
}

#[test]
fn exec_batches_report_failure_without_stopping_the_walk() {
    let mut environment = Environment::new();
    let (status, stdout, stderr) = run(
        &mut environment,
        r"mkdir root; touch root/a root/b; find root -type f -exec sh -c 'exit 1' {} +",
    );
    assert_eq!(status, 1, "{stderr}");
    assert!(stdout.is_empty(), "{stdout:?}");
}

#[test]
fn print_output_precedes_a_later_exec_childs_output() {
    let mut environment = Environment::new();
    let (status, stdout, stderr) = run(
        &mut environment,
        r"mkdir root; touch root/a; find root -type f -print -exec echo child \;",
    );
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, b"root/a\nchild\n");
}

#[test]
fn exec_children_resume_after_blocking_on_a_full_pipe() {
    // 400 matches, each spawning its own `echo` child that writes a 200-byte line: well over the
    // 64 KiB default pipe capacity, so `wc -l` on the other end must repeatedly drain the pipe
    // while several `-exec` children in turn block on `IoWait::PipeWritable` and resume.
    let mut environment = Environment::new();
    let filler = "a".repeat(200);
    let (status, stdout, stderr) = run(
        &mut environment,
        &format!(
            r"mkdir many; touch many/f{{1..400}}; find many -type f -exec echo {filler} \; | wc -l"
        ),
    );
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(String::from_utf8_lossy(&stdout).trim(), "400");
}

#[test]
fn cpu_exhaustion_during_a_large_walk_stops_and_reports_the_limit() {
    let mut environment = Environment::with_limits(Limits {
        cpu: 400,
        ..Limits::unlimited()
    });
    let (outcome, _stdout, _stderr) = environment
        .run_script_capture("mkdir big; for i in $(seq 1 500); do touch big/f$i; done; find big");
    assert_eq!(outcome.exit_status, 137);
    assert_eq!(outcome.stop_reason, Some(StopReason::CpuExhausted));
}

#[test]
fn ok_and_okdir_are_rejected_as_unsupported() {
    let mut environment = Environment::new();
    for source in ["find . -ok echo {} \\;", "find . -okdir echo {} \\;"] {
        let (status, stdout, stderr) = run(&mut environment, source);
        assert_eq!(status, 2, "{source}: {stderr}");
        assert!(stdout.is_empty(), "{source}");
        assert!(
            stderr.contains("interactive terminal"),
            "{source}: {stderr}"
        );
    }
}

#[test]
fn execdir_runs_in_the_matchs_directory_with_a_relative_name() {
    let mut environment = Environment::new();
    let (status, stdout, stderr) = run(
        &mut environment,
        r"mkdir -p root/sub; touch root/sub/file; find root -type f -execdir pwd \; -execdir echo {} \;",
    );
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, b"/root/sub\n./file\n");
}

#[test]
fn execdir_batching_is_rejected_as_unsupported() {
    let mut environment = Environment::new();
    let (status, stdout, stderr) = run(&mut environment, r"find . -execdir echo {} +");
    assert_eq!(status, 2);
    assert!(stdout.is_empty());
    assert!(stderr.contains("execdir"), "{stderr}");
}

#[test]
fn exec_plus_packs_matches_like_gnu_find() {
    // GNU find's `-exec ... +` packs as many matches as fit under its argument-byte budget into
    // one invocation. 300 short names are nowhere near that budget, so a real system's `find`
    // (and this native image, sized to match) runs exactly one `echo`, producing one `wc -l` line.
    let mut environment = Environment::new();
    environment.vfs.mkdir_all("/", "/many").unwrap();
    for i in 0..300 {
        environment
            .vfs
            .write("/", &format!("/many/f_long_name_number_{i}"), b"", 0o644)
            .unwrap();
    }
    let (status, stdout, stderr) = run(
        &mut environment,
        "find many -name 'f_*' -exec echo {} + | wc -l",
    );
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(String::from_utf8_lossy(&stdout).trim(), "1");
}

#[test]
fn exec_batches_flush_early_once_the_argument_byte_cap_is_reached() {
    // 300 names of just over 500 bytes each add up to comfortably more than the 128 KiB batch
    // cap, so this must flush more than one child; confirm the `+` form still runs every match
    // exactly once across those children. Files are seeded directly through the VFS so the test
    // does not need a single enormous shell command line to create them.
    let mut environment = Environment::new();
    environment.vfs.mkdir_all("/", "/many").unwrap();
    let mut expected_names = Vec::new();
    for i in 0..300 {
        let name = format!("f_{i}_{}", "x".repeat(500));
        environment
            .vfs
            .write("/", &format!("/many/{name}"), b"", 0o644)
            .unwrap();
        expected_names.push(format!("many/{name}"));
    }
    let (status, stdout, stderr) = run(&mut environment, "find many -type f -exec echo {} +");
    assert_eq!(status, 0, "{stderr}");
    let printed = String::from_utf8_lossy(&stdout);
    let mut names: Vec<&str> = printed.split_whitespace().collect();
    names.sort_unstable();
    names.dedup();
    expected_names.sort_unstable();
    assert_eq!(
        names, expected_names,
        "every match must be echoed exactly once"
    );
    let echo_invocations = environment
        .invocations
        .events()
        .iter()
        .filter(|event| event.argv.first().map(String::as_str) == Some("echo"))
        .count();
    assert!(
        echo_invocations > 1,
        "expected the batch cap to force more than one echo child, got {echo_invocations}"
    );
}

#[test]
fn delete_removes_nested_directories_depth_first() {
    let mut environment = Environment::new();
    let (status, stdout, stderr) = run(
        &mut environment,
        "mkdir -p root/a/b; touch root/a/b/f; find root -delete; test -e root",
    );
    assert_eq!(status, 1, "{stderr}"); // `test -e root` is false: everything was removed.
    assert!(stdout.is_empty());
}

#[test]
fn malformed_exec_forms_fail_during_expression_parsing() {
    let mut environment = Environment::new();
    for source in [
        "find . -exec echo {}",
        "find . -exec +",
        "find . -exec echo +",
    ] {
        let (status, stdout, stderr) = run(&mut environment, source);
        assert_eq!(status, 2, "{source}: {stderr}");
        assert!(stdout.is_empty(), "{source}");
        assert!(stderr.contains("find:"), "{source}: {stderr}");
    }
}
