//! Resource-accounting boundaries for Python operations reachable from simulated input.

use shellsim::{python, Environment, Limits, StopReason};
use std::io::Write;

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
fn direct_list_growth_does_not_require_a_full_container_snapshot() {
    let (status, stdout, stderr, usage) = run_with_limits(
        "items = []\nfor value in range(3000):\n    items.append(value)\nprint(len(items))",
        Limits {
            memory: 128 * 1024,
            ..Limits::unlimited()
        },
    );
    assert_eq!(
        status,
        0,
        "stderr={} usage={usage:?}",
        String::from_utf8_lossy(&stderr)
    );
    assert_eq!(stdout, b"3000\n");
    assert!(usage.memory_peak <= 128 * 1024);
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
    let payload = format!("[{}]", vec!["0"; 4_000].join(","));
    let source = format!("import json; json.loads({payload:?})");
    let (status, stdout, stderr, usage) = run_with_limits(
        &source,
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
fn arbitrary_precision_arithmetic_is_metered_before_growth() {
    let (status, stdout, stderr, usage) = run_with_limits(
        "value = 9223372036854775807\nfor _ in range(100):\n    value = value * value",
        Limits {
            memory: 64 * 1024,
            ..Limits::unlimited()
        },
    );
    assert_eq!(status, 137);
    assert!(usage.memory_peak <= 64 * 1024);
    assert!(stdout.is_empty());
    assert!(stderr.is_empty());
}

#[test]
fn string_join_reserves_its_result_before_host_growth() {
    let (status, stdout, stderr, usage) = run_with_limits(
        "separator = 'x' * 1000\nseparator.join(['a'] * 200)",
        Limits {
            memory: 40 * 1024,
            ..Limits::unlimited()
        },
    );
    assert_eq!(status, 137);
    assert!(usage.memory_peak <= 40 * 1024);
    assert!(stdout.is_empty());
    assert!(stderr.is_empty());
}

#[test]
fn repeated_large_ascii_string_indexing_stays_within_cpu_budget() {
    let source =
        "value = 'x' * 10000\nfor _ in range(1000):\n    assert value[9999] == 'x'\nprint('ok')";
    let (status, stdout, stderr, usage) = run_with_limits(
        source,
        Limits {
            cpu: 500_000,
            ..Limits::unlimited()
        },
    );
    assert_eq!(
        status,
        0,
        "stderr={} usage={usage:?}",
        String::from_utf8_lossy(&stderr)
    );
    assert_eq!(stdout, b"ok\n");
    assert!(usage.cpu_used <= 500_000);
}

#[test]
fn repeated_large_ascii_string_slicing_stays_within_cpu_budget() {
    let source = "value = 'x' * 100000\nfor offset in range(1000):\n    assert value[offset:offset + 10] == 'x' * 10\nprint('ok')";
    let (status, stdout, stderr, usage) = run_with_limits(
        source,
        Limits {
            cpu: 500_000,
            ..Limits::unlimited()
        },
    );
    assert_eq!(
        status,
        0,
        "stderr={} usage={usage:?}",
        String::from_utf8_lossy(&stderr)
    );
    assert_eq!(stdout, b"ok\n");
    assert!(usage.cpu_used <= 500_000);
}

#[test]
fn bytes_repetition_reserves_before_allocating() {
    let (status, stdout, stderr, usage) = run_with_limits(
        "b'x' * 1000000",
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
fn zlib_rejects_expansion_before_materializing_the_result() {
    let mut encoder = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
    encoder.write_all(&vec![0; 1024 * 1024]).unwrap();
    let compressed = encoder.finish().unwrap();
    let literal = compressed
        .iter()
        .map(|byte| format!("\\x{byte:02x}"))
        .collect::<String>();
    let source = format!("import zlib\nzlib.decompress(b'{literal}')");
    let (status, stdout, stderr, usage) = run_with_limits(
        &source,
        Limits {
            memory: 512 * 1024,
            ..Limits::unlimited()
        },
    );
    assert_eq!(status, 137);
    assert!(usage.memory_peak <= 512 * 1024);
    assert!(stdout.is_empty());
    assert!(stderr.is_empty());
}

#[test]
fn temporary_objects_and_cycles_use_a_bounded_working_set() {
    let source = "index = 0\nwhile index < 5000:\n    value = 'x' * 1024\n    cycle = []\n    cycle.append(cycle)\n    index += 1\nprint(len(value))";
    let (status, stdout, stderr, usage) = run_with_limits(
        source,
        Limits {
            cpu: 20_000_000,
            memory: 512 * 1024,
            ..Limits::unlimited()
        },
    );
    assert_eq!(
        status,
        0,
        "stderr={} usage={usage:?}",
        String::from_utf8_lossy(&stderr)
    );
    assert_eq!(stdout, b"1024\n");
    assert_eq!(usage.memory_current, 0);
    assert!(usage.memory_peak <= 512 * 1024);
}

#[test]
fn completed_python_processes_release_owned_memory() {
    let mut environment = Environment::with_limits(Limits {
        cpu: 2_000_000,
        memory: 256 * 1024,
        ..Limits::unlimited()
    });
    for _ in 0..20 {
        let (outcome, stdout, stderr) =
            environment.run_script_capture("python3.14 -c 'print(\"x\" * 4096)' >/dev/null");
        assert_eq!(
            outcome.exit_status,
            0,
            "{}",
            String::from_utf8_lossy(&stderr)
        );
        assert!(stdout.is_empty());
        assert_eq!(outcome.usage.memory_current, 0);
    }
}
