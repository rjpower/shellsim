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
    // The list's 720 KB of values fits beside the interpreter's startup heap and collector
    // slack, but a second copy of the list would not.
    let (status, stdout, stderr, usage) = run_with_limits(
        "items = []\nfor value in range(30000):\n    items.append(value)\nprint(len(items))",
        Limits {
            memory: 1024 * 1024,
            ..Limits::unlimited()
        },
    );
    assert_eq!(
        status,
        0,
        "stderr={} usage={usage:?}",
        String::from_utf8_lossy(&stderr)
    );
    assert_eq!(stdout, b"30000\n");
    assert!(usage.memory_peak <= 1024 * 1024);
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
fn bytes_join_reserves_its_parts_and_result() {
    // A thousand references to one 100 kB value would join into 100 MB.
    let (status, stdout, stderr, usage) = run_with_limits(
        "data = bytes(100_000)\nb''.join([b'ok'])\nprint('small')\nb''.join([data] * 1000)",
        Limits {
            memory: 2 * 1024 * 1024,
            ..Limits::unlimited()
        },
    );
    assert_eq!(status, 137, "{}", String::from_utf8_lossy(&stderr));
    assert!(usage.memory_peak <= 2 * 1024 * 1024);
    assert_eq!(stdout, b"small\n");
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
    // The first actions create shell variables such as `PIPESTATUS`, which stay charged as
    // retained shell state; each Python process must release everything it owned.
    let mut retained = Vec::new();
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
        retained.push(outcome.usage.memory_current);
    }
    assert!(retained[0] < 1024, "{retained:?}");
    assert!(
        retained[2..].iter().all(|bytes| *bytes == retained[2]),
        "{retained:?}"
    );
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
    // `import numpy` loads the Python half of the package, which peaks near 5 MiB while the
    // largest module is parsed; the million-element array needs 16 MB.
    let limits = Limits {
        memory: 8 * 1024 * 1024,
        ..Limits::unlimited()
    };
    let (status, stdout, _, _) = run_with_limits(
        "import numpy as np\nprint(np.zeros(100, dtype=complex).sum())",
        limits,
    );
    assert_eq!((status, stdout), (0, b"0j\n".to_vec()));
    let (status, stdout, stderr, usage) = run_with_limits(
        "import numpy as np\nnp.zeros(1_000_000, dtype=complex)",
        limits,
    );
    assert_eq!(status, 137);
    assert!(usage.memory_peak <= 8 * 1024 * 1024);
    assert!(stdout.is_empty());
    assert!(stderr.is_empty());
}

#[test]
fn numpy_linalg_charges_cubic_work_before_factoring() {
    let limits = Limits {
        cpu: 5_000_000,
        ..Limits::unlimited()
    };
    let (status, stdout, stderr, _) = run_with_limits(
        "import numpy as np\nprint(np.linalg.inv(np.eye(20)).trace())",
        limits,
    );
    assert_eq!(
        (status, stdout),
        (0, b"20.0\n".to_vec()),
        "{}",
        String::from_utf8_lossy(&stderr)
    );
    // Inverting 300x300 costs 2 * 300^3 units, which the call charges before factoring.
    let (status, stdout, stderr, usage) = run_with_limits(
        "import numpy as np\na = np.eye(300)\nprint('built')\nnp.linalg.inv(a)",
        limits,
    );
    assert_eq!(status, 137);
    assert_eq!(usage.cpu_used, 5_000_000);
    assert_eq!(stdout, b"built\n");
    assert!(stderr.is_empty());
}

#[test]
fn complex_numpy_linalg_uses_the_same_cpu_boundary() {
    let (status, stdout, stderr, usage) = run_with_limits(
        "import numpy as np\na = np.eye(300, dtype=complex)\nprint('built')\nnp.linalg.inv(a)",
        Limits {
            cpu: 5_000_000,
            ..Limits::unlimited()
        },
    );
    assert_eq!(status, 137);
    assert_eq!(usage.cpu_used, 5_000_000);
    assert_eq!(stdout, b"built\n");
    assert!(stderr.is_empty());
}

#[test]
fn numpy_jacobi_eigensolver_charges_each_sweep() {
    let limits = Limits {
        cpu: 5_000_000,
        ..Limits::unlimited()
    };
    // Each sweep over a 120x120 matrix costs 120^3 units, so a few sweeps exhaust the budget
    // even though the call charges only quadratic setup work up front.
    let (status, stdout, stderr, usage) = run_with_limits(
        "import numpy as np\na = np.arange(14400.0).reshape(120, 120) % 7\na = a + a.T\n\
         print('built')\nnp.linalg.eigvalsh(a)",
        limits,
    );
    assert_eq!(status, 137);
    assert_eq!(usage.cpu_used, 5_000_000);
    assert_eq!(stdout, b"built\n");
    assert!(stderr.is_empty());
}

#[test]
fn numpy_fft_charges_transform_work_before_running() {
    let limits = Limits {
        cpu: 5_000_000,
        ..Limits::unlimited()
    };
    let (status, stdout, stderr, _) = run_with_limits(
        "import numpy as np\nprint(np.fft.fft(np.ones(8)).real[0])",
        limits,
    );
    assert_eq!(
        (status, stdout),
        (0, b"8.0\n".to_vec()),
        "{}",
        String::from_utf8_lossy(&stderr)
    );
    // `numpy.fft` is frozen Python built from ordinary vectorized NumPy calls (moveaxis, take,
    // reshape, exp, ...), each already metered per element by the native ufunc/array machinery
    // they go through; a 64x4096 batch's log2(4096) = 12 butterfly stages charge for tens of
    // millions of complex-array elements well before the transform finishes.
    let (status, stdout, stderr, usage) = run_with_limits(
        "import numpy as np\na = np.zeros((64, 4096))\nprint('built')\nnp.fft.fft(a)",
        limits,
    );
    assert_eq!(status, 137);
    assert_eq!(usage.cpu_used, 5_000_000);
    assert_eq!(stdout, b"built\n");
    assert!(stderr.is_empty());
}

#[test]
fn numpy_sort_charges_n_log_n_work_before_sorting() {
    let limits = Limits {
        cpu: 5_000_000,
        ..Limits::unlimited()
    };
    let (status, stdout, stderr, _) = run_with_limits(
        "import numpy as np\nprint(np.sort(np.array([3, 1, 2])).tolist())",
        limits,
    );
    assert_eq!(
        (status, stdout),
        (0, b"[1, 2, 3]\n".to_vec()),
        "{}",
        String::from_utf8_lossy(&stderr)
    );
    // Sorting charges n * ceil(log2(n)) units per lane before any comparison runs, so two
    // million elements exhausts the budget even though building the input is much cheaper.
    let (status, stdout, stderr, usage) = run_with_limits(
        "import numpy as np\na = np.arange(2_000_000)\nprint('built')\nnp.sort(a)",
        limits,
    );
    assert_eq!(status, 137);
    assert_eq!(usage.cpu_used, 5_000_000);
    assert_eq!(stdout, b"built\n");
    assert!(stderr.is_empty());
}

#[test]
fn numpy_random_reserves_and_charges_before_drawing() {
    let limits = Limits {
        cpu: 5_000_000,
        memory: 16 * 1024 * 1024,
        ..Limits::unlimited()
    };
    let (status, stdout, stderr, _) = run_with_limits(
        "import numpy as np\nprint(np.random.default_rng(1).random(1000).shape)",
        limits,
    );
    assert_eq!(
        (status, stdout),
        (0, b"(1000,)\n".to_vec()),
        "{}",
        String::from_utf8_lossy(&stderr)
    );
    // Ten million doubles need 80 MB, which the fill reserves before drawing.
    let (status, stdout, stderr, usage) = run_with_limits(
        "import numpy as np\nrng = np.random.default_rng(1)\nprint('seeded')\nrng.random(10_000_000)",
        limits,
    );
    assert_eq!(status, 137);
    assert!(usage.memory_peak <= 16 * 1024 * 1024);
    assert_eq!(stdout, b"seeded\n");
    assert!(stderr.is_empty());
    // One million draws fit in memory but prepay a million CPU units at once.
    let (status, stdout, stderr, usage) = run_with_limits(
        "import numpy as np\nrng = np.random.default_rng(1)\nprint('seeded')\nrng.random(1_000_000)",
        Limits {
            cpu: 800_000,
            ..limits
        },
    );
    assert_eq!(status, 137);
    assert_eq!(usage.cpu_used, 800_000);
    assert_eq!(stdout, b"seeded\n");
    assert!(stderr.is_empty());
}

#[test]
fn numpy_loadtxt_stops_at_the_memory_limit() {
    let (status, stdout, stderr, usage) = run_with_limits(
        "import io\nimport numpy as np\ntext = '1 2 3\\n' * 5000\nprint(np.loadtxt(io.StringIO(text[:60])).shape)\nnp.loadtxt(io.StringIO(text * 100))",
        Limits {
            cpu: 50_000_000,
            memory: 8 * 1024 * 1024,
            ..Limits::unlimited()
        },
    );
    assert_eq!(status, 137, "{}", String::from_utf8_lossy(&stderr));
    assert!(usage.memory_peak <= 8 * 1024 * 1024);
    assert_eq!(stdout, b"(10, 3)\n");
    assert!(stderr.is_empty());
}

#[test]
fn numpy_byte_copies_reserve_memory_before_copying() {
    // Bytes are charged at their length, and each copy reserves its host buffer before it is
    // made. Peak use is about 20 MB after `frombuffer` and 26 MB during `tobytes`.
    let limits = Limits {
        cpu: 50_000_000,
        memory: 22 * 1024 * 1024,
        ..Limits::unlimited()
    };
    let (status, stdout, stderr, usage) = run_with_limits(
        "import numpy as np\ndata = bytes(6_000_000)\nvalues = np.frombuffer(data, np.uint8)\nprint('copied')\nvalues.tobytes()",
        limits,
    );
    assert_eq!(status, 137, "{}", String::from_utf8_lossy(&stderr));
    assert!(usage.memory_peak <= 22 * 1024 * 1024);
    assert_eq!(stdout, b"copied\n");
    assert!(stderr.is_empty());
}

#[test]
fn set_algebra_consumes_cpu_per_membership_test() {
    // Membership uses the hash index, so 200 members against 200 operand items fit the
    // budget, while 200,000 operand items are each charged and stop at the limit.
    let limits = Limits {
        cpu: 200_000,
        ..Limits::unlimited()
    };
    let setup = "members = set(range(200))\n";
    for operation in [
        "members.intersection(range(200, 400))",
        "members.difference(range(200, 400))",
        "members.symmetric_difference(range(200, 400))",
        "members.difference_update(range(200, 400))",
        "members.isdisjoint(range(200, 400))",
    ] {
        let (status, stdout, stderr, _) =
            run_with_limits(&format!("{setup}{operation}\nprint('done')"), limits);
        assert_eq!(
            (status, stdout),
            (0, b"done\n".to_vec()),
            "{operation}: {}",
            String::from_utf8_lossy(&stderr)
        );
        let large = operation.replace("range(200, 400)", "range(200, 200_200)");
        let (status, stdout, stderr, usage) =
            run_with_limits(&format!("{setup}{large}\nprint('done')"), limits);
        assert_eq!(status, 137, "{large}");
        assert_eq!(usage.cpu_used, 200_000, "{large}");
        assert!(stdout.is_empty(), "{large}");
        assert!(stderr.is_empty(), "{large}");
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

#[test]
fn fraction_growth_stops_at_the_memory_limit() {
    let limits = Limits {
        memory: 4 * 1024 * 1024,
        ..Limits::unlimited()
    };
    // Squaring doubles the digits of both terms, so the loop reaches the limit quickly.
    let (status, stdout, stderr, _usage) = run_with_limits(
        "from fractions import Fraction\nx = Fraction(1, 3)\nprint('ready')\nwhile True:\n    x = x * x",
        limits,
    );
    assert_eq!(status, 137);
    assert_eq!(stdout, b"ready\n");
    assert!(stderr.is_empty(), "{}", String::from_utf8_lossy(&stderr));
}

#[test]
fn rejection_sampling_uses_modeled_cpu_fuel() {
    let (status, stdout, stderr, usage) = run_with_limits(
        // The gamma sampler discards a first draw outside (1e-7, 0.9999999), so a source that
        // always returns 0.99999999 never leaves its rejection loop.
        "import random\nclass NeverAccept(random.Random):\n    def random(self):\n        return 0.99999999\nprint('ready')\nNeverAccept(1).gammavariate(2.0, 1.0)",
        Limits {
            cpu: 100_000,
            ..Limits::unlimited()
        },
    );
    assert_eq!(status, 137);
    assert_eq!(usage.cpu_used, 100_000);
    assert_eq!(stdout, b"ready\n");
    assert!(stderr.is_empty());
}

#[test]
fn big_integer_work_is_charged_by_operand_words() {
    // `1 << 4_000_000` has 62,501 words; squaring it is charged about n * sqrt(n), and long
    // division by a half-size divisor about the product of the word counts.
    let limits = Limits {
        cpu: 50_000_000,
        ..Limits::unlimited()
    };
    let (status, stdout, stderr, usage) = run_with_limits(
        "x = 1 << 4_000_000\ny = x * x\nprint(y.bit_length())",
        limits,
    );
    assert_eq!(status, 0, "{}", String::from_utf8_lossy(&stderr));
    assert_eq!(stdout, b"8000001\n");
    assert!(usage.cpu_used > 15_000_000, "{usage:?}");

    let (status, stdout, _, usage) = run_with_limits(
        "x = 1 << 4_000_000\ny = x // ((1 << 2_000_000) + 1)\nprint('done')",
        limits,
    );
    assert_eq!(status, 137);
    assert_eq!(usage.cpu_used, 50_000_000);
    assert!(stdout.is_empty());
}

#[test]
fn dict_and_set_deletion_cost_is_constant_per_member() {
    let limits = Limits {
        cpu: 5_000_000,
        ..Limits::unlimited()
    };
    let (status, stdout, stderr, _) = run_with_limits(
        "d = dict.fromkeys(range(100_000))\nfor key in range(100_000):\n    del d[key]\ns = set(range(100_000))\nwhile s:\n    s.pop()\nprint(len(d), len(s))",
        limits,
    );
    assert_eq!(status, 0, "{}", String::from_utf8_lossy(&stderr));
    assert_eq!(stdout, b"0 0\n");
}

#[test]
fn heap_keys_are_indexed_by_hash() {
    // Long strings, tuples and user objects live on the heap; each lookup must compare only the
    // keys that share its hash rather than every key in the dict.
    let limits = Limits {
        cpu: 20_000_000,
        ..Limits::unlimited()
    };
    let (status, stdout, stderr, _) = run_with_limits(
        "class K:\n    pass\nd = {}\nfor i in range(50_000):\n    d['key-' * 10 + str(i)] = i\n    d[(i, str(i))] = i\n    d[K()] = i\nprint(len(d), d['key-' * 10 + '7'], d[(9, '9')])",
        limits,
    );
    assert_eq!(status, 0, "{}", String::from_utf8_lossy(&stderr));
    assert_eq!(stdout, b"150000 7 9\n");
}

#[test]
fn adversarial_substring_search_is_linear() {
    let limits = Limits {
        cpu: 20_000_000,
        ..Limits::unlimited()
    };
    let (status, stdout, stderr, _) = run_with_limits(
        "s = 'a' * 1_000_000\nneedle = 'a' * 100_000 + 'b'\nfor i in range(20):\n    assert s.find(needle) == -1 and s.count(needle) == 0\nprint(s.rfind('a' * 3), s.count('aa'))",
        limits,
    );
    assert_eq!(status, 0, "{}", String::from_utf8_lossy(&stderr));
    assert_eq!(stdout, b"999997 500000\n");
}

#[test]
fn sorting_a_large_list_uses_n_log_n_comparisons() {
    let limits = Limits {
        cpu: 20_000_000,
        ..Limits::unlimited()
    };
    let (status, stdout, stderr, _) = run_with_limits(
        "values = [(i * 7919) % 100_003 for i in range(100_000)]\nprint(sorted(values, key=lambda v: -v)[0], sorted(values)[-1])\nvalues.sort(reverse=True)\nprint(values[0])",
        limits,
    );
    assert_eq!(status, 0, "{}", String::from_utf8_lossy(&stderr));
    assert_eq!(stdout, b"100002 100002\n100002\n");
}

#[test]
fn quadratic_copies_are_charged_by_bytes_moved() {
    let limits = Limits {
        cpu: 20_000_000,
        ..Limits::unlimited()
    };
    for program in [
        "l = []\nfor i in range(1_000_000):\n    l.insert(0, i)",
        "s = ''\nfor i in range(1_000_000):\n    s += 'abcdefgh'",
        "import math\nprint(math.gcd(7 ** 1_000_000, 3 ** 1_000_000))",
    ] {
        let (status, _, _, usage) = run_with_limits(program, limits);
        assert_eq!(status, 137, "{program}");
        assert_eq!(usage.cpu_used, limits.cpu, "{program}");
    }
}

#[test]
fn generator_fed_builtins_are_bounded_by_the_memory_limit() {
    // `list()` drains the generator from a bytecode loop, so each item lands in a metered heap
    // list instead of host scratch that the generator's own instructions could release.
    let (status, stdout, stderr, usage) = run_with_limits(
        "def g():\n    while True:\n        yield 1\nlist(g())",
        Limits {
            memory: 1024 * 1024,
            ..Limits::unlimited()
        },
    );
    assert_eq!(status, 137);
    assert!(usage.memory_peak <= 1024 * 1024);
    assert!(stdout.is_empty());
    assert!(stderr.is_empty());
}

#[test]
fn join_over_a_generator_is_bounded_by_the_memory_limit() {
    let (status, stdout, stderr, usage) = run_with_limits(
        "def g():\n    while True:\n        yield 'x'\n''.join(g())",
        Limits {
            memory: 1024 * 1024,
            ..Limits::unlimited()
        },
    );
    assert_eq!(status, 137);
    assert!(usage.memory_peak <= 1024 * 1024);
    assert!(stdout.is_empty());
    assert!(stderr.is_empty());
}

#[test]
fn native_scratch_survives_nested_python_frames() {
    // `repr(list)` accumulates its output as host scratch while each element's Python `__repr__`
    // runs its own instructions. Those instructions must not refund the enclosing reservation.
    let (status, stdout, stderr, usage) = run_with_limits(
        "class A:\n    def __repr__(self):\n        return 'x' * 1000\nrepr([A()] * 20000)",
        Limits {
            memory: 4 * 1024 * 1024,
            ..Limits::unlimited()
        },
    );
    assert_eq!(status, 137);
    assert!(usage.memory_peak <= 4 * 1024 * 1024);
    assert!(stdout.is_empty());
    assert!(stderr.is_empty());
}

#[test]
fn deeply_nested_sequence_ordering_raises_recursion_error() {
    // Comparing 100k-deep tuples used to recurse on the host stack; the bound surfaces as the
    // same exception CPython raises.
    let (status, stdout, stderr, _) = run_with_limits(
        "x = ()\ny = ()\nfor i in range(100000):\n    x = (x,)\n    y = (y,)\n\
         try:\n    x < y\nexcept RecursionError:\n    print('bounded')\n\
         print((1, 2) < (1, 3), [[1]] < [[2]], sorted([(2, 1), (1, 2)]))",
        Limits::unlimited(),
    );
    assert_eq!(status, 0);
    assert_eq!(stdout, b"bounded\nTrue True [(1, 2), (2, 1)]\n");
    assert!(stderr.is_empty());
}

#[test]
fn rendering_a_self_referential_exception_raises_recursion_error() {
    let (status, stdout, stderr, _) = run_with_limits(
        "e = ValueError(1)\ne.args = (e,)\n\
         try:\n    str(e)\nexcept RecursionError:\n    print('str bounded')\n\
         class E(Exception):\n    pass\nf = E()\nf.args = (f,)\n\
         try:\n    str(f)\nexcept RecursionError:\n    print('user bounded')\n\
         print(str(ValueError(ValueError(1))), repr(KeyError(KeyError('k'))))",
        Limits::unlimited(),
    );
    assert_eq!(status, 0);
    assert_eq!(
        stdout,
        b"str bounded\nuser bounded\n1 KeyError(KeyError('k'))\n"
    );
    assert!(stderr.is_empty());
}

#[test]
fn int_from_bytes_is_linear_in_the_input() {
    // A megabyte of input used to round-trip through a decimal string, which is quadratic.
    let (status, stdout, stderr, usage) = run_with_limits(
        "print(int.from_bytes(b'\\xff' * 1_000_000, 'big').bit_length())\n\
         print(int.from_bytes(b'\\x01\\x00', 'little'), int.from_bytes(b'\\xff', 'big', signed=True))",
        Limits {
            cpu: 20_000_000,
            ..Limits::unlimited()
        },
    );
    assert_eq!(status, 0);
    assert_eq!(stdout, b"8000000\n1 -1\n");
    assert!(stderr.is_empty());
    assert!(usage.cpu_used < 20_000_000);
}

#[test]
fn full_svd_and_complete_qr_reserve_their_square_factors() {
    // A 4000 x 200 input has a 4000 x 4000 `U` or `Q` of 128 MB, which must be reserved before
    // it is built instead of appearing as host memory the model never saw.
    for source in [
        "import numpy as np\nnp.linalg.svd(np.zeros((4000, 200)))",
        "import numpy as np\nnp.linalg.qr(np.zeros((4000, 200)), mode='complete')",
    ] {
        let (status, stdout, stderr, usage) = run_with_limits(
            source,
            Limits {
                memory: 64 * 1024 * 1024,
                ..Limits::unlimited()
            },
        );
        assert_eq!(status, 137, "{source}");
        assert!(usage.memory_peak <= 64 * 1024 * 1024, "{source}");
        assert!(stdout.is_empty());
        assert!(stderr.is_empty());
    }
    // The thin factorizations the solvers use stay small and correct.
    let (status, stdout, stderr, usage) = run_with_limits(
        "import numpy as np\na = np.arange(12.0).reshape(4, 3)\n\
         u, s, vt = np.linalg.svd(a, full_matrices=False)\n\
         print(np.allclose(u @ np.diag(s) @ vt, a), u.shape, vt.shape)\n\
         x, *_ = np.linalg.lstsq(a, np.arange(4.0), rcond=None)\nprint(np.allclose(a @ x, np.arange(4.0)))",
        Limits {
            memory: 16 * 1024 * 1024,
            ..Limits::unlimited()
        },
    );
    assert_eq!(status, 0, "{}", String::from_utf8_lossy(&stderr));
    assert_eq!(stdout, b"True (4, 3) (3, 3)\nTrue\n");
    assert!(usage.memory_peak <= 16 * 1024 * 1024);
}

#[test]
fn over_long_paths_are_rejected_before_any_node_is_created() {
    // A 20k-component path used to cost a gigabyte of host memory for the parent directories.
    let (status, stdout, stderr, usage) = run_with_limits(
        "import os\ntry:\n    os.makedirs('/' + '/'.join(['d'] * 3000))\nexcept OSError:\n    print('OSError')\n\
         try:\n    open('/' + 'n' * 300, 'w')\nexcept OSError:\n    print('OSError')\n\
         os.makedirs('/ok/' + '/'.join(['d'] * 100))\nprint(os.path.isdir('/ok/d/d'))",
        Limits {
            memory: 16 * 1024 * 1024,
            ..Limits::unlimited()
        },
    );
    assert_eq!(status, 0, "{}", String::from_utf8_lossy(&stderr));
    assert_eq!(stdout, b"OSError\nOSError\nTrue\n");
    assert!(usage.memory_peak <= 16 * 1024 * 1024);
}

#[test]
fn leaving_an_except_handler_releases_the_handled_exception() {
    // `continue`, `break` and `return` out of a handler used to leave the exception on the VM's
    // stack, so a long loop grew without bound and every collection walked the whole stack.
    let (status, stdout, stderr, usage) = run_with_limits(
        "def f():\n    try:\n        raise KeyError(1)\n    except KeyError:\n        return 1\n\
         total = 0\nfor i in range(20000):\n    try:\n        raise ValueError(i)\n    except ValueError:\n        total += f()\n        continue\n\
         print(total)",
        Limits {
            memory: 4 * 1024 * 1024,
            cpu: 50_000_000,
            ..Limits::unlimited()
        },
    );
    assert_eq!(status, 0, "{}", String::from_utf8_lossy(&stderr));
    assert_eq!(stdout, b"20000\n");
    assert!(usage.memory_peak <= 2 * 1024 * 1024);
}

#[test]
fn repeated_eval_does_not_retain_code_caches() {
    let (status, stdout, stderr, usage) = run_with_limits(
        "total = 0\nfor i in range(20000):\n    total += eval('1 + 1')\nprint(total)",
        Limits {
            memory: 4 * 1024 * 1024,
            cpu: 50_000_000,
            ..Limits::unlimited()
        },
    );
    assert_eq!(status, 0, "{}", String::from_utf8_lossy(&stderr));
    assert_eq!(stdout, b"40000\n");
    assert!(usage.memory_peak <= 1024 * 1024);
}

#[test]
fn string_scans_charge_cpu_proportional_to_their_length() {
    // Substring search, equality, ordering, strip and affix tests over a 10 MB string each scan
    // the whole value; a loop of them must run out of CPU rather than host time.
    for operation in [
        "'q' in s",
        "s == t",
        "s < t",
        "s.strip()",
        "s.startswith('q')",
    ] {
        let (status, stdout, _, usage) = run_with_limits(
            &format!("s = 'x' * 10_000_000\nt = 'x' * 10_000_000\nfor _ in range(100000):\n    {operation}"),
            Limits {
                cpu: 20_000_000,
                memory: 256 * 1024 * 1024,
                ..Limits::unlimited()
            },
        );
        assert_eq!(status, 137, "{operation}");
        assert_eq!(usage.cpu_used, 20_000_000, "{operation}");
        assert!(stdout.is_empty());
    }
}

#[test]
fn set_subset_tests_are_linear_in_the_smaller_set() {
    let (status, stdout, stderr, usage) = run_with_limits(
        "a = set(range(20000))\nb = set(range(40000))\nfor _ in range(50):\n    assert a <= b and not b <= a and a < b\nprint('ok')",
        Limits {
            cpu: 50_000_000,
            ..Limits::unlimited()
        },
    );
    assert_eq!(status, 0, "{}", String::from_utf8_lossy(&stderr));
    assert_eq!(stdout, b"ok\n");
    assert!(usage.cpu_used < 50_000_000);
}

#[test]
fn front_end_memory_is_reserved_before_a_large_source_is_parsed() {
    // A megabyte of statements expands to hundreds of megabytes of tokens and syntax tree. The
    // reservation happens before lexing so the limit stops the program instead of the host.
    let (status, stdout, stderr, usage) = run_with_limits(
        "exec('x = 1\\n' * 200000)\nprint(x)",
        Limits {
            memory: 32 * 1024 * 1024,
            ..Limits::unlimited()
        },
    );
    assert_eq!(status, 137, "{}", String::from_utf8_lossy(&stderr));
    assert!(usage.memory_peak <= 32 * 1024 * 1024);
    assert!(stdout.is_empty());
    let (status, stdout, stderr, _) = run_with_limits(
        "exec('x = 1\\n' * 2000)\nprint(x)",
        Limits {
            memory: 32 * 1024 * 1024,
            ..Limits::unlimited()
        },
    );
    assert_eq!(status, 0, "{}", String::from_utf8_lossy(&stderr));
    assert_eq!(stdout, b"1\n");
}
