//! Integration coverage for the bounded, VFS-only `make` command.
//!
//! The tests install Makefiles into the simulated VFS and invoke the command through the normal
//! shell dispatcher. They never read or execute a host Makefile.

use shellsim::{Environment, Limits, StopReason};

fn run_make(env: &mut Environment, makefile: &str, args: &str) -> (i32, String, String) {
    env.vfs
        .put_file("/Makefile", makefile.as_bytes().to_vec(), 0o644)
        .unwrap();
    let command = if args.is_empty() {
        "make".to_string()
    } else {
        format!("make {args}")
    };
    let (outcome, stdout, stderr) = env.run_script_capture(&command);
    (
        outcome.exit_status,
        String::from_utf8_lossy(&stdout).into_owned(),
        String::from_utf8_lossy(&stderr).into_owned(),
    )
}

#[test]
fn builds_prerequisites_in_order_and_expands_variables() {
    let mut env = Environment::new();
    env.vfs
        .put_file("/input", b"payload\n".to_vec(), 0o644)
        .unwrap();
    let (status, stdout, stderr) = run_make(
        &mut env,
        "NAME = generated\nOUT = result\nall: $(OUT)\n\t@echo done $(NAME)\nresult: input\n\t@cat input > result\n",
        "",
    );
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, "done generated\n");
    assert_eq!(env.vfs.read_string("/", "/result").unwrap(), "payload\n");
}

#[test]
fn backslashes_in_comments_do_not_start_line_continuations() {
    let mut env = Environment::new();
    let (status, stdout, stderr) = run_make(
        &mut env,
        "# commented flags \\\nFLAGS=-O2 # explanatory comment \\\nall:\n\t@printf '%s' $(FLAGS)\n",
        "",
    );
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, "-O2");
}

#[test]
fn expands_variables_inside_target_and_prerequisite_names() {
    let mut env = Environment::new();
    let (status, stdout, stderr) = run_make(
        &mut env,
        "SUFFIX=.txt\nNAME=result\nall: $(NAME)$(SUFFIX)\n\t@cat result.txt\nresult$(SUFFIX):\n\t@printf ready > $@\n",
        "",
    );
    assert_eq!(status, 0, "{stderr}");
    assert!(stdout.ends_with("ready"));
    assert_eq!(env.vfs.read_string("/", "/result.txt").unwrap(), "ready");
}

#[test]
fn continued_recipe_runs_in_one_shell_process() {
    let mut env = Environment::new();
    let (status, stdout, stderr) = run_make(
        &mut env,
        "all:\n\t@VALUE=ready; \\\n\tprintf '%s' \"$VALUE\"\n",
        "",
    );
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, "ready");
}

#[test]
fn recipe_dash_prefix_ignores_only_that_command_failure() {
    let mut env = Environment::new();
    let (status, stdout, stderr) = run_make(&mut env, "all:\n\t-@false\n\t@printf done\n", "");
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, "done");
}

#[test]
fn later_dependency_only_rule_extends_existing_target() {
    let mut env = Environment::new();
    env.vfs.write("/", "/input", b"source", 0o644).unwrap();
    let (status, stdout, stderr) = run_make(
        &mut env,
        "result:\n\t@printf ready > result\nresult: input\nall: result\n\t@cat result\n",
        "all",
    );
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, "ready");
}

#[test]
fn builds_a_file_target_and_skips_it_when_up_to_date() {
    let mut env = Environment::new();
    env.vfs
        .put_file("/input", b"payload\n".to_vec(), 0o644)
        .unwrap();
    let makefile = "result: input\n\t@cat input > result\n";
    let first = run_make(&mut env, makefile, "");
    assert_eq!(first.0, 0, "{}", first.2);
    assert_eq!(env.vfs.read_string("/", "/result").unwrap(), "payload\n");
    let second = run_make(&mut env, makefile, "");
    assert_eq!(second.0, 0, "{}", second.2);
    assert_eq!(second.1, "make: 'result' is up to date.\n");
}

#[test]
fn recipes_run_as_scheduled_children() {
    let mut env = Environment::new();
    env.vfs
        .put_file(
            "/Makefile",
            b"result:\n\t@sleep 2; cat /tmp/marker > result\n".to_vec(),
            0o644,
        )
        .unwrap();
    let (outcome, stdout, stderr) =
        env.run_script_capture("(sleep 1; printf ready > /tmp/marker) & make; cat result; wait");
    assert_eq!(
        outcome.exit_status,
        0,
        "{}",
        String::from_utf8_lossy(&stderr)
    );
    assert_eq!(stdout, b"ready");
    assert_eq!(env.clock.monotonic_ns(), 2_000_000_000);

    env.vfs
        .put_file("/Makefile", b"broken:\n\t@false\n".to_vec(), 0o644)
        .unwrap();
    let (outcome, _, _) = env.run_script_capture("make broken");
    assert_eq!(outcome.exit_status, 2);
}

#[test]
fn supports_dry_run_and_explicit_file_directory() {
    let mut env = Environment::new();
    env.vfs.put_dir("/work/project", 0o755).unwrap();
    env.vfs
        .put_file(
            "/work/project/build.mk",
            b"all:\n\t@printf dry > output\n".to_vec(),
            0o644,
        )
        .unwrap();
    let (outcome, stdout, stderr) = env.run_script_capture("make -n -C /work/project -f build.mk");
    assert_eq!(outcome.exit_status, 0, "{:?}", stderr);
    assert_eq!(stdout, b"printf dry > output\n");
    assert!(!env.vfs.exists("/work/project", "output"));
}

#[test]
fn rejects_unsupported_options_and_syntax() {
    let mut env = Environment::new();
    // `-j N` is accepted for compatibility; recipes still run one at a time.
    let jobs = run_make(&mut env, "all:\n\t@printf ok\n", "-j2");
    assert_eq!(jobs.0, 0, "{}", jobs.2);
    assert_eq!(jobs.1, "ok");

    let unsupported = run_make(&mut env, "all:\n\t@true\n", "--bogus-option");
    assert_eq!(unsupported.0, 2);
    assert!(unsupported.2.contains("unsupported"));

    let syntax = run_make(&mut env, "include other.mk\n", "");
    assert_eq!(syntax.0, 2);
    assert!(syntax.2.contains("missing ':'") || syntax.2.contains("unsupported"));
}

#[test]
fn makefile_parsing_is_metered() {
    let mut env = Environment::with_limits(Limits {
        cpu: 10,
        ..Limits::unlimited()
    });
    let (outcome, _, stderr) = env.run_script_capture("make");
    assert_eq!(outcome.stop_reason, Some(StopReason::CpuExhausted));
    assert!(stderr.is_empty() || String::from_utf8_lossy(&stderr).contains("resource"));
}

#[test]
fn echoes_recipe_lines_before_their_output_unless_silenced() {
    let mut env = Environment::new();
    let (status, stdout, stderr) = run_make(
        &mut env,
        "all:\n\tprintf 'shown\\n'\n\t@printf 'quiet\\n'\n",
        "",
    );
    assert_eq!(status, 0, "{stderr}");
    // The non-`@` line is echoed before it runs; the `@` line never is.
    assert_eq!(stdout, "printf 'shown\\n'\nshown\nquiet\n");
}

#[test]
fn global_silent_flag_suppresses_echo_like_at_prefix() {
    let mut env = Environment::new();
    let (status, stdout, stderr) = run_make(&mut env, "all:\n\tprintf done\n", "-s");
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, "done");
}

#[test]
fn recipe_failure_reports_gnu_style_diagnostic_and_stops() {
    let mut env = Environment::new();
    let (status, stdout, stderr) = run_make(&mut env, "all:\n\t@false\n\t@printf never\n", "");
    assert_eq!(status, 2);
    assert!(stdout.is_empty(), "{stdout}");
    assert_eq!(stderr, "make: *** [Makefile:2: all] Error 1\n");
}

#[test]
fn keep_going_continues_past_a_failed_target() {
    let mut env = Environment::new();
    let (status, stdout, stderr) = run_make(
        &mut env,
        "all: broken ok\nbroken:\n\t@false\nok:\n\t@printf ran\n",
        "-k",
    );
    assert_eq!(status, 2);
    assert!(stdout.contains("ran"), "{stdout}");
    assert!(
        stderr.contains("*** [Makefile:3: broken] Error 1"),
        "{stderr}"
    );
}

#[test]
fn without_keep_going_a_failure_skips_later_prerequisites() {
    let mut env = Environment::new();
    let (status, stdout, stderr) = run_make(
        &mut env,
        "all: broken ok\nbroken:\n\t@false\nok:\n\t@printf ran\n",
        "",
    );
    assert_eq!(status, 2);
    assert!(!stdout.contains("ran"), "{stdout}");
    assert!(stderr.contains("Error 1"), "{stderr}");
}

#[test]
fn missing_rule_reports_gnu_style_diagnostic() {
    let mut env = Environment::new();
    let (status, _, stderr) = run_make(&mut env, "all: missing.txt\n\t@true\n", "");
    assert_eq!(status, 2);
    assert_eq!(
        stderr,
        "make: *** No rule to make target 'missing.txt'.  Stop.\n"
    );
}

#[test]
fn rebuild_decisions_see_output_from_earlier_sibling_recipes() {
    let mut env = Environment::new();
    // Build /shared and /b with the second strictly newer than the first.
    let setup = env.run_script_capture("printf old > /shared; sleep 1; printf oldb > /b");
    assert_eq!(setup.0.exit_status, 0);
    // `a` runs first and rewrites `shared` after another full second, so by the time `b`'s
    // rebuild decision is made `shared` is newer than `b`. A planner that decided both targets
    // up front, before any recipe ran, would still see the original (older) `shared` mtime here
    // and wrongly leave `b` alone.
    env.vfs
        .put_file(
            "/Makefile",
            b"all: a b\na:\n\t@sleep 1; printf newshared > shared\nb: shared\n\t@printf newb > b\n"
                .to_vec(),
            0o644,
        )
        .unwrap();
    let (outcome, _, stderr) = env.run_script_capture("make");
    assert_eq!(
        outcome.exit_status,
        0,
        "{}",
        String::from_utf8_lossy(&stderr)
    );
    assert_eq!(env.vfs.read_string("/", "/b").unwrap(), "newb");
}

#[test]
fn recipes_run_as_processes_distinct_from_the_shell() {
    let mut env = Environment::new();
    env.vfs
        .put_file("/Makefile", b"all:\n\t@printf ran\n".to_vec(), 0o644)
        .unwrap();
    let (outcome, stdout, stderr) = env.run_script_capture("make");
    assert_eq!(
        outcome.exit_status,
        0,
        "{}",
        String::from_utf8_lossy(&stderr)
    );
    assert_eq!(stdout, b"ran");
    assert!(env
        .invocations
        .events()
        .iter()
        .any(|event| event.pid != 1_234 && event.argv == ["sh", "-c", "printf ran"]));
}

#[test]
fn recipe_count_is_bounded() {
    let mut env = Environment::new();
    let mut makefile = String::from("all: t0\n");
    for i in 0..20_100 {
        makefile.push_str(&format!("t{i}:\n\t@true\n"));
    }
    let (status, _, stderr) = run_make(&mut env, &makefile, "");
    assert_eq!(status, 2);
    assert!(stderr.contains("limit exceeded"), "{stderr}");
}

#[test]
fn circular_dependency_is_reported_and_dropped_rather_than_hanging() {
    let mut env = Environment::new();
    let (status, _, stderr) = run_make(&mut env, "a: b\n\t@true\nb: a\n\t@true\n", "a");
    assert_eq!(status, 0, "{stderr}");
    assert!(stderr.contains("Circular"), "{stderr}");
}

#[test]
fn pattern_rules_are_rejected_as_unsupported() {
    let mut env = Environment::new();
    let (status, _, stderr) = run_make(&mut env, "%.o: %.c\n\t@true\n", "");
    assert_eq!(status, 2);
    assert!(stderr.contains("unsupported"), "{stderr}");
}

#[test]
fn recipes_inherit_makes_stdin_and_follow_gnu_export_rules() {
    // Matches host GNU make 4.x on the same Makefile: a makefile-only variable (`X`) is invisible
    // to recipes unless explicitly exported; a variable that came from the environment (`Y`) is
    // exported automatically; `export Z=3` exports a makefile-only variable; and a command-line
    // override (`make X=9`) is always exported, taking precedence over the makefile value.
    let mut env = Environment::new();
    env.vfs
        .put_file(
            "/Makefile",
            b"X=1\nexport Z=3\nall: dep\n\t@echo \"x=$X y=$Y z=$Z\"; read l; echo \"stdin=$l\"\ndep:\n\t@echo dep\n"
                .to_vec(),
            0o644,
        )
        .unwrap();

    let (outcome, stdout, stderr) = env.run_script_capture("echo input | Y=2 make");
    assert_eq!(
        outcome.exit_status,
        0,
        "{}",
        String::from_utf8_lossy(&stderr)
    );
    assert_eq!(stdout, b"dep\nx= y=2 z=3\nstdin=input\n");

    let (outcome, stdout, stderr) = env.run_script_capture("echo input | Y=2 make X=9 all");
    assert_eq!(
        outcome.exit_status,
        0,
        "{}",
        String::from_utf8_lossy(&stderr)
    );
    assert_eq!(stdout, b"dep\nx=9 y=2 z=3\nstdin=input\n");
}

#[test]
fn unexport_hides_an_environment_variable_from_recipes() {
    let mut env = Environment::new();
    env.vfs
        .put_file(
            "/Makefile",
            b"unexport W\nall:\n\t@printf \"w=[$W]\"\n".to_vec(),
            0o644,
        )
        .unwrap();
    let (outcome, stdout, stderr) = env.run_script_capture("W=5 make");
    assert_eq!(
        outcome.exit_status,
        0,
        "{}",
        String::from_utf8_lossy(&stderr)
    );
    assert_eq!(stdout, b"w=[]");
}
