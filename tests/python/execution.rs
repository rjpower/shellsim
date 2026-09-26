//! Shell entrypoint behavior and small end-to-end Python compatibility checks.

use super::support::run_shell;

#[test]
fn python_314_is_registered_and_reports_the_emulated_version() {
    assert_eq!(
        run_shell("command -v python3.14; python3.14 --version"),
        (
            0,
            b"/usr/bin/python3.14\nPython 3.14.0\n".to_vec(),
            Vec::new()
        )
    );
}

#[test]
fn bytecode_vm_covers_arithmetic_argv_and_environment() {
    assert_eq!(
        run_shell(
            "export SHELLSIM_TOKEN=present; python3.14 -c 'import os; import sys; print(1 + 2 * 3, 5 / 2, -7 // 3, -7 % 3); print(sys.argv[-1]); print(os.getenv(\"SHELLSIM_TOKEN\", \"missing\"))' final"
        ),
        (
            0,
            b"7 2.5 -3 2\nfinal\npresent\n".to_vec(),
            Vec::new()
        )
    );
}

#[test]
fn python_chdir_is_process_local_to_the_python_command() {
    assert_eq!(
        run_shell(
            "mkdir -p /work/project; cd /work; pwd; python3.14 -c 'import os; os.chdir(\"project\"); print(os.getcwd())'; pwd"
        ),
        (
            0,
            b"/work\n/work/project\n/work\n".to_vec(),
            Vec::new()
        )
    );
}

#[test]
fn containers_comparisons_and_short_circuiting_use_python_protocols() {
    assert_eq!(
        run_shell(
            "python3.14 -c 'a = [1, \"x\"]; b = a; print(a, (1,), {\"a\": 2}, {}); print(a is b, 1 < 2 < 3, 1 < 2 > 3); print(0 and missing, 1 or missing, not [], \"x\" in \"xyz\", 2 in [1, 2]); print({\"a\": 2}[\"a\"], {1, 2} == {2, 1})'"
        ),
        (
            0,
            b"[1, 'x'] (1,) {'a': 2} {}\nTrue True False\n0 1 True True True\n2 True\n"
                .to_vec(),
            Vec::new()
        )
    );
}

#[test]
fn string_indexing_preserves_ascii_unicode_and_negative_indices() {
    assert_eq!(
        run_shell(
            "python3.14 -c 'value = \"aé☃z\"; print(value[0], value[1], value[-2], value[-1])'"
        ),
        (0, "a é ☃ z\n".as_bytes().to_vec(), Vec::new())
    );
}

#[test]
fn string_slicing_preserves_ascii_unicode_and_extended_slice_semantics() {
    assert_eq!(
        run_shell(
            "python3.14 -c 'ascii = \"0123456789\" * 4; unicode = \"aé☃z🙂q\"; print(ascii[3:17], ascii[-12:-2], ascii[20:5:-4], ascii[4:4]); print(unicode[1:5], unicode[::-2], unicode[-5:-1:2])'"
        ),
        (
            0,
            "34567890123456 8901234567 0628 \né☃z🙂 qzé éz\n"
                .as_bytes()
                .to_vec(),
            Vec::new(),
        )
    );
}

#[test]
fn sequence_ordering_uses_element_protocols_and_sets_use_subset_ordering() {
    let source = r#"class Key:
    def __init__(self, value):
        self.value = value
    def __eq__(self, other):
        return self.value == other.value
    def __lt__(self, other):
        return self.value < other.value
print((Key(1), 9) < (Key(2), 0))
print([Key(2)] > [Key(1)])
print({1} < {1, 2}, {1, 2} <= {2, 1}, {1, 2} >= {1})
print(sorted({1, 2, 3} - {2, 4}))"#;
    assert_eq!(
        run_shell(&format!("python3.14 <<'PY'\n{source}\nPY")),
        (
            0,
            b"True\nTrue\nTrue True True\n[1, 3]\n".to_vec(),
            Vec::new(),
        )
    );
}

#[test]
fn script_execution_defines_file_without_exposing_the_host() {
    assert_eq!(
        run_shell("printf 'print(__file__)\\n' > /work/program.py; python /work/program.py"),
        (0, b"/work/program.py\n".to_vec(), Vec::new())
    );
}

#[test]
fn system_exit_uses_process_status_and_base_exception_hierarchy() {
    assert_eq!(
        run_shell(
            "python - <<'PY'\ntry:\n    raise SystemExit(4)\nexcept Exception:\n    print('wrong')\nexcept BaseException:\n    print('base')\nraise SystemExit(7)\nPY"
        ),
        (7, b"base\n".to_vec(), Vec::new())
    );
    assert_eq!(
        run_shell("python -c 'raise SystemExit(\"clear boundary\")'"),
        (1, Vec::new(), b"clear boundary\n".to_vec())
    );
}

#[test]
fn venv_exposes_offline_python_and_pip_entrypoints() {
    assert_eq!(
        run_shell(
            "printf 'numpy==2.1.3\\n' > /work/requirements.txt; python -m venv /tmp/example; /tmp/example/bin/pip install --no-cache-dir -r /work/requirements.txt; /tmp/example/bin/python -c 'import numpy; print(numpy.__version__)'"
        ),
        (0, b"2.0.0-shellsim\n".to_vec(), Vec::new())
    );
}

#[test]
fn common_text_predicates_lines_and_collection_copies_match_python() {
    let source = r#"print('ABC 12'.isupper(), 'abc 12'.islower(), ''.isupper())
print('a\r\nb\nc\r'.splitlines(), 'a\r\nb\n'.splitlines(True))
print('42'.zfill(5), '-42'.zfill(5), '+42'.zfill(2))
items = [[1]]
items_copy = items.copy()
items_copy.append([2])
mapping = {'a': items}
mapping_copy = mapping.copy()
mapping_copy['b'] = 2
values = {1, 2}
values_copy = values.copy()
values_copy.add(3)
print(len(items), len(items_copy), mapping_copy['a'] is items, sorted(mapping_copy.keys()))
print(sorted(values), sorted(values_copy))"#;
    assert_eq!(
        run_shell(&format!("python3.14 <<'PY'\n{source}\nPY")),
        (
            0,
            b"True True False\n['a', 'b', 'c'] ['a\\r\\n', 'b\\n']\n00042 -0042 +42\n1 2 True ['a', 'b']\n[1, 2] [1, 2, 3]\n".to_vec(),
            Vec::new(),
        )
    );
}

#[test]
fn chr_builds_unicode_scalars_and_rejects_out_of_range_values() {
    assert_eq!(
        run_shell("python -c 'print(chr(65), chr(0x1f642))'"),
        (0, "A 🙂\n".as_bytes().to_vec(), Vec::new())
    );
    let (status, _, stderr) = run_shell("python -c 'chr(0x110000)' ");
    assert_eq!(status, 1);
    assert!(
        String::from_utf8_lossy(&stderr)
            .ends_with("ValueError: chr() arg not in range(0x110000)\n"),
        "{}",
        String::from_utf8_lossy(&stderr)
    );
}
