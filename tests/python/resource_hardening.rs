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
fn percent_format_width_is_rejected_before_host_allocation() {
    let (status, stdout, stderr, usage) = run_with_limits(
        "print('%1000000s' % 'x')",
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
fn format_spec_width_is_rejected_before_host_allocation() {
    let (status, stdout, stderr, usage) = run_with_limits(
        "print('{value:1000000s}'.format(value='x'))",
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
fn exec_source_consumes_cpu_before_parsing() {
    let (status, stdout, stderr, usage) = run_with_limits(
        "exec('pass\\n' * 10000)",
        Limits {
            cpu: 1000,
            ..Limits::unlimited()
        },
    );
    assert_eq!(status, 137);
    assert_eq!(usage.cpu_used, 1000);
    assert!(stdout.is_empty());
    assert!(stderr.is_empty());
}

#[test]
fn math_comb_reserves_result_before_large_multiplications() {
    let (status, stdout, stderr, usage) = run_with_limits(
        "import math\nprint(math.comb(100000, 50000))",
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
fn combinatorial_iterators_reserve_before_materializing_results() {
    let (status, stdout, stderr, usage) = run_with_limits(
        "import itertools\nlist(itertools.product(range(100), range(100)))",
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

#[test]
fn grouped_formats_reserve_before_growth() {
    let (status, stdout, stderr, _) = run_with_limits(
        "print(f'{1.5:01000000,.2f}')",
        Limits {
            memory: 64 * 1024,
            ..Limits::unlimited()
        },
    );
    assert_eq!(status, 137);
    assert!(stdout.is_empty());
    assert!(stderr.is_empty());
}

#[test]
fn complex_values_are_metered_heap_allocations() {
    // The same list of inline floats fits; each complex adds a metered arena object.
    let limits = Limits {
        memory: 1024 * 1024,
        ..Limits::unlimited()
    };
    let (status, stdout, _, _) = run_with_limits(
        "values = [float(i) for i in range(20000)]\nprint(len(values))",
        limits,
    );
    assert_eq!((status, stdout), (0, b"20000\n".to_vec()));
    let (status, stdout, stderr, usage) = run_with_limits(
        "values = [complex(i, i) for i in range(20000)]\nprint(len(values))",
        limits,
    );
    assert_eq!(status, 137);
    assert!(usage.memory_peak <= 1024 * 1024);
    assert!(stdout.is_empty());
    assert!(stderr.is_empty());
}

#[test]
fn complex_arrays_reserve_element_storage_before_allocation() {
    let limits = Limits {
        memory: 64 * 1024,
        ..Limits::unlimited()
    };
    let (status, stdout, _, _) = run_with_limits(
        "import numpy as np\nprint(np.zeros(100, dtype=complex).sum())",
        limits,
    );
    assert_eq!((status, stdout), (0, b"0j\n".to_vec()));
    let (status, stdout, stderr, usage) = run_with_limits(
        "import numpy as np\nnp.zeros(100000, dtype=complex)",
        limits,
    );
    assert_eq!(status, 137);
    assert!(usage.memory_peak <= 64 * 1024);
    assert!(stdout.is_empty());
    assert!(stderr.is_empty());
}

#[test]
fn set_algebra_consumes_cpu_per_membership_test() {
    // Building the 200-member set costs about half this budget. Each operation below compares
    // all 200 members with 200 operand items and must stop at the limit.
    let limits = Limits {
        cpu: 40_000,
        ..Limits::unlimited()
    };
    let setup = "members = set(range(200))\n";
    let (status, stdout, _, _) = run_with_limits(&format!("{setup}print('built')"), limits);
    assert_eq!((status, stdout), (0, b"built\n".to_vec()));
    for operation in [
        "members.intersection(range(200, 400))",
        "members.difference(range(200, 400))",
        "members.symmetric_difference(range(200, 400))",
        "members.difference_update(range(200, 400))",
        "members.isdisjoint(range(200, 400))",
    ] {
        let (status, stdout, stderr, usage) =
            run_with_limits(&format!("{setup}{operation}\nprint('done')"), limits);
        assert_eq!(status, 137, "{operation}");
        assert_eq!(usage.cpu_used, 40_000, "{operation}");
        assert!(stdout.is_empty(), "{operation}");
        assert!(stderr.is_empty(), "{operation}");
    }
}

#[test]
fn container_methods_reserve_before_materializing_iterables() {
    let limits = Limits {
        memory: 32 * 1024,
        ..Limits::unlimited()
    };
    for operation in [
        "dict.fromkeys(range(100000))",
        "set().update(range(100000))",
        "{0}.symmetric_difference(range(100000))",
        "{0}.issubset(range(100000))",
    ] {
        let (status, stdout, stderr, usage) =
            run_with_limits(&format!("{operation}\nprint('done')"), limits);
        assert_eq!(status, 137, "{operation}");
        assert!(usage.memory_peak <= 32 * 1024, "{operation}");
        assert!(stdout.is_empty(), "{operation}");
        assert!(stderr.is_empty(), "{operation}");
    }
}

#[test]
fn bytes_methods_reserve_before_building_results() {
    // The heap models each byte as one value slot, so the 10,000-byte setup uses about half of
    // this budget. Each operation below would build a result of at least one megabyte.
    let limits = Limits {
        memory: 512 * 1024,
        ..Limits::unlimited()
    };
    let setup = "part = b'x' * 10000\n";
    let (status, stdout, _, _) = run_with_limits(&format!("{setup}print(len(part))"), limits);
    assert_eq!((status, stdout), (0, b"10000\n".to_vec()));
    for operation in [
        "(b'a' * 100).replace(b'', part)",
        "part.replace(b'x', b'y' * 100)",
        "b''.join([part] * 100)",
        "bytearray(b',' * 20000).split(b',')",
    ] {
        let (status, stdout, stderr, usage) =
            run_with_limits(&format!("{setup}{operation}\nprint('done')"), limits);
        assert_eq!(status, 137, "{operation}");
        assert!(usage.memory_peak <= 512 * 1024, "{operation}");
        assert!(stdout.is_empty(), "{operation}");
        assert!(stderr.is_empty(), "{operation}");
    }
}
