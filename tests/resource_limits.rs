use shellsim::{Environment, Limits, StopReason};

#[test]
fn cpu_fuel_stops_shell_loops() {
    let mut env = Environment::with_limits(Limits {
        cpu: 500,
        ..Limits::unlimited()
    });
    let (outcome, _, _) = env.run_script_capture("while true; do :; done");
    assert_eq!(outcome.exit_status, 137);
    assert_eq!(outcome.stop_reason, Some(StopReason::CpuExhausted));
    assert_eq!(outcome.usage.cpu_used, 500);
}

#[test]
fn command_memory_is_a_released_working_set() {
    let mut env = Environment::with_limits(Limits {
        memory: 12 * 1024,
        ..Limits::unlimited()
    });
    let (outcome, _, _) = env.run_script_capture("sort <<EOF\nb\na\nEOF");
    assert_eq!(outcome.stop_reason, Some(StopReason::MemoryExhausted));
    assert_eq!(outcome.usage.memory_current, 0);
    assert_eq!(outcome.exit_status, 137);
}

#[test]
fn redirected_write_reports_disk_full_without_creating_a_file() {
    let mut env = Environment::with_limits(Limits {
        disk: 256 + 2,
        ..Limits::unlimited()
    });
    let (outcome, _, stderr) = env.run_script_capture("printf 123 > /result");
    assert_eq!(outcome.exit_status, 1);
    assert!(!env.vfs.lexists("/", "/result"));
    assert!(String::from_utf8_lossy(&stderr).contains("No space left on device"));
}

#[test]
fn removing_a_base_directory_does_not_create_free_disk_quota() {
    let mut env = Environment::with_limits(Limits {
        disk: 256 + 2,
        ..Limits::unlimited()
    });
    let (outcome, _, stderr) = env.run_script_capture("rmdir /tmp; printf 123 > /result");

    assert_eq!(outcome.exit_status, 1);
    assert!(!env.vfs.lexists("/", "/result"));
    assert!(String::from_utf8_lossy(&stderr).contains("No space left on device"));
}

#[test]
fn output_limit_is_reported_separately() {
    let mut env = Environment::with_limits(Limits {
        output: 5,
        ..Limits::unlimited()
    });
    let (outcome, stdout, _) = env.run_script_capture("echo 12345");
    assert_eq!(outcome.stop_reason, Some(StopReason::OutputLimitExceeded));
    assert_eq!(outcome.usage.output_bytes, 5);
    assert_eq!(outcome.exit_status, 137);
    assert_eq!(stdout, b"12345");
}

#[test]
fn native_child_output_uses_the_same_limit() {
    let mut env = Environment::with_limits(Limits {
        output: 1,
        ..Limits::unlimited()
    });
    let (outcome, stdout, _) = env.run_script_capture("env pwd");
    assert_eq!(outcome.stop_reason, Some(StopReason::OutputLimitExceeded));
    assert_eq!(outcome.usage.output_bytes, 1);
    assert_eq!(outcome.exit_status, 137);
    assert_eq!(stdout, b"/");
}

#[test]
fn command_trace_contains_resource_deltas() {
    let mut env = Environment::new();
    let (outcome, stdout, _) = env.run_script_capture("printf 'b\\na\\n' | sort");
    assert_eq!(stdout, b"a\nb\n");
    assert_eq!(outcome.command_usage.len(), 2);
    assert!(outcome.command_usage.iter().all(|entry| entry.cpu > 0));
    assert!(outcome.usage.memory_peak >= 16 * 1024);
}

#[test]
fn nested_commands_do_not_double_charge_output() {
    let mut env = Environment::with_limits(Limits {
        output: 7,
        ..Limits::unlimited()
    });
    let (outcome, stdout, _) = env.run_script_capture("printf 'a b' | xargs -n 1 echo");
    // Three bytes in the internal pipe plus four emitted by the nested echo invocations.
    assert_eq!(outcome.usage.output_bytes, 7);
    assert_eq!(outcome.stop_reason, None);
    assert_eq!(stdout, b"a\nb\n");
}

#[test]
fn reused_environment_preserves_shell_and_filesystem_state() {
    let mut env = Environment::new();
    let (first, _, _) =
        env.run_script_capture("name=agent; mkdir /demo; printf saved > /demo/value; cd /demo");
    let (second, stdout, _) =
        env.run_script_capture("printf '%s:%s:' \"$name\" \"$PWD\"; cat value");

    assert_eq!(first.exit_status, 0);
    assert_eq!(second.exit_status, 0);
    assert_eq!(stdout, b"agent:/demo:saved");
    assert!(second.usage.cpu_used > first.usage.cpu_used);
    assert_eq!(env.cwd, "/demo");
}

#[test]
fn action_stdin_is_explicit_and_shell_state_still_persists() {
    let mut env = Environment::new();
    env.run_script_capture("name=agent; cd /work");
    let (outcome, stdout, stderr) = env.run_script_capture_with_stdin(
        "cat > input; printf '%s:%s:' \"$name\" \"$PWD\"; cat input",
        b"payload\n",
    );

    assert_eq!(outcome.exit_status, 0);
    assert_eq!(stdout, b"agent:/work:payload\n");
    assert!(stderr.is_empty());
    assert_eq!(env.cwd, "/work");
}

#[test]
fn conventional_workspace_directories_exist_in_every_environment() {
    let env = Environment::new();

    for path in ["/root", "/tmp", "/work"] {
        assert!(
            env.vfs.is_dir("/", path),
            "missing conventional directory {path}"
        );
    }
    assert_eq!(env.vfs.disk_used(), 0);
}

#[test]
fn exit_is_sticky_across_shell_actions() {
    let mut env = Environment::new();
    let (first, _, _) = env.run_script_capture("exit 7");
    let cpu_after_exit = first.usage.cpu_used;
    let (second, stdout, _) = env.run_script_capture("echo should-not-run");

    assert!(env.is_terminated());
    assert_eq!(first.exit_status, 7);
    assert_eq!(second.exit_status, 7);
    assert_eq!(second.usage.cpu_used, cpu_after_exit);
    assert!(stdout.is_empty());
}

#[test]
fn exhaustion_is_sticky_across_shell_actions() {
    let mut env = Environment::with_limits(Limits {
        cpu: 150,
        ..Limits::unlimited()
    });
    let (first, _, _) = env.run_script_capture("true; true");
    let (second, stdout, _) = env.run_script_capture("echo should-not-run");

    assert_eq!(first.stop_reason, Some(StopReason::CpuExhausted));
    assert_eq!(second.stop_reason, Some(StopReason::CpuExhausted));
    assert_eq!(second.usage.cpu_used, 150);
    assert!(stdout.is_empty());
}

/// Run `source` under a small memory budget and require it to stop on memory, not abort.
fn assert_stops_on_memory(source: &str) {
    const MEMORY: u64 = 1024 * 1024;
    let mut env = Environment::with_limits(Limits {
        cpu: 200_000_000,
        memory: MEMORY,
        ..Limits::unlimited()
    });
    let (outcome, _, _) = env.run_script_capture(source);
    assert_eq!(
        outcome.stop_reason,
        Some(StopReason::MemoryExhausted),
        "{source}"
    );
    assert_eq!(outcome.exit_status, 137, "{source}");
    assert!(outcome.usage.memory_peak <= MEMORY, "{source}");
}

#[test]
fn shell_state_and_expansions_are_charged_before_they_are_retained() {
    for source in [
        "x=a; while :; do x=$x$x; done",
        "x=$(printf %01000d 0); while :; do x=${x//0/00}; done",
        "i=0; while :; do eval \"v$i=0123456789012345678901234567890123456789\"; i=$((i+1)); done",
        "a=(); while :; do a+=(0123456789012345678901234567890123456789); done",
        "f() { local v=$1$1; f \"$v\"; }; f 0123456789",
        "printf '%*d' 2000000000 1",
        "printf '%.*f' 2000000000 1",
        "awk 'BEGIN { for (i = 0; ; i++) a[i] = i }'",
        "awk 'BEGIN { s = sprintf(\"%200000s\", \"\"); gsub(/ /, \"a \", s); split(s, a) }'",
    ] {
        assert_stops_on_memory(source);
    }
}

#[test]
fn sparse_indexed_arrays_store_only_assigned_elements() {
    let mut env = Environment::with_limits(Limits {
        memory: 1024 * 1024,
        ..Limits::unlimited()
    });
    let (outcome, stdout, _) = env.run_script_capture(
        "a[9999999999]=1; a[3]=x; a+=(y); echo ${#a[@]} ${!a[@]} ${a[9999999999]} ${a[10000000000]}",
    );
    assert_eq!(outcome.exit_status, 0);
    assert_eq!(stdout, b"3 3 9999999999 10000000000 1 y\n");
}

#[test]
fn command_usage_has_one_entry_per_command() {
    let mut env = Environment::new();
    let (outcome, _, _) =
        env.run_script_capture("for i in 1 2 3 4 5; do printf x; :; done | sort >/dev/null");
    let commands = outcome
        .command_usage
        .iter()
        .map(|usage| usage.command.as_str())
        .collect::<Vec<_>>();
    assert_eq!(commands.iter().filter(|name| **name == "printf").count(), 1);
    assert!(commands.len() <= 3, "{commands:?}");
}

#[test]
fn a_lone_bracket_does_not_list_the_directory() {
    // `[` without a closing `]` is not a glob pattern, so a test loop does no directory work.
    let mut env = Environment::new();
    let script = "mkdir d; cd d; for i in $(seq 2000); do : > f$i; done";
    let (outcome, _, _) = env.run_script_capture(script);
    assert_eq!(outcome.exit_status, 0);
    let (lone, _, _) = env.run_script_capture("i=0; while [ $i -lt 200 ]; do i=$((i+1)); done");
    let (glob, _, _) = env.run_script_capture("for i in $(seq 200); do echo f1* >/dev/null; done");
    assert_eq!(lone.exit_status, 0);
    assert!(
        glob.usage.cpu_used > lone.usage.cpu_used + 200 * 2000,
        "glob {} lone {}",
        glob.usage.cpu_used,
        lone.usage.cpu_used
    );
}
