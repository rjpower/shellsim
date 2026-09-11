use shellsim::{python, Environment, Limits, StopReason};

fn run_with_limits(
    source: &str,
    limits: Limits,
) -> (i32, Vec<u8>, Vec<u8>, shellsim::resources::Usage) {
    let mut environment = Environment::with_limits(limits);
    let argv = vec!["python3.14".into(), "-c".into(), source.into()];
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let status = python::run_python(
        &mut environment,
        &argv,
        Vec::new(),
        &mut stdout,
        &mut stderr,
    );
    let usage = environment.resources.outcome(status, 0, 0).usage;
    (status, stdout, stderr, usage)
}

#[test]
fn python_print_and_stream_writes_are_hard_output_bounded() {
    let limits = Limits {
        output: 7,
        ..Limits::unlimited()
    };
    let (status, stdout, stderr, usage) = run_with_limits(
        "import sys; print('x' * 100); sys.stderr.write('y' * 100)",
        limits,
    );
    assert_eq!(status, 137);
    assert_eq!(usage.output_bytes, 7);
    assert_eq!(stdout, b"xxxxxxx");
    assert!(
        stderr.is_empty(),
        "a stopped VM must not append after the cap"
    );
}

#[test]
fn python_command_output_is_not_double_charged() {
    let mut environment = Environment::with_limits(Limits {
        output: 5,
        ..Limits::unlimited()
    });
    let (outcome, stdout, stderr) =
        environment.run_script_capture("python3.14 -c 'print(\"x\" * 100)'");
    assert_eq!(outcome.stop_reason, Some(StopReason::OutputLimitExceeded));
    assert_eq!(outcome.usage.output_bytes, 5);
    assert_eq!(stdout, b"xxxxx");
    assert!(stderr.is_empty());
}

#[test]
fn regex_substitution_reserves_before_constructing_result() {
    let (status, stdout, stderr, _usage) = run_with_limits(
        "import re; print(re.sub('x', 'y' * 10000, 'x' * 1000))",
        Limits {
            memory: 32 * 1024,
            ..Limits::unlimited()
        },
    );
    assert_eq!(status, 137);
    assert!(stdout.is_empty());
    assert!(stderr.is_empty());
}

#[test]
fn native_sorting_consumes_cpu_fuel() {
    let (status, stdout, stderr, usage) = run_with_limits(
        "print(sorted(range(100, 0, -1)))",
        Limits {
            cpu: 500,
            ..Limits::unlimited()
        },
    );
    assert_eq!(status, 137);
    assert_eq!(usage.cpu_used, 500);
    assert!(stdout.is_empty());
    assert!(stderr.is_empty());
}

#[test]
fn iterable_materialization_is_metered_before_host_growth() {
    let (status, stdout, stderr, usage) = run_with_limits(
        "print(list(range(100000)))",
        Limits {
            memory: 32 * 1024,
            ..Limits::unlimited()
        },
    );
    assert_eq!(status, 137);
    assert!(usage.memory_peak <= 32 * 1024);
    assert!(stdout.is_empty());
    assert!(stderr.is_empty());
}

#[test]
fn json_dumps_has_a_preallocation_bound() {
    let (status, stdout, stderr, usage) = run_with_limits(
        "import json; print(json.dumps('x' * 10000))",
        Limits {
            memory: 32 * 1024,
            ..Limits::unlimited()
        },
    );
    assert_eq!(status, 137);
    assert!(usage.memory_peak <= 32 * 1024);
    assert!(stdout.is_empty());
    assert!(stderr.is_empty());
}

#[test]
fn json_loads_reserves_parser_and_object_memory() {
    let (status, stdout, stderr, usage) = run_with_limits(
        "import json; json.loads('[0,0,0,0,0,0,0,0,0,0]' * 1000)",
        Limits {
            memory: 32 * 1024,
            ..Limits::unlimited()
        },
    );
    assert_eq!(status, 137);
    assert!(usage.memory_peak <= 32 * 1024);
    assert!(stdout.is_empty());
    assert!(stderr.is_empty());
}
