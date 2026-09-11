use std::process::Command;

use shellsim::{Environment, Limits, StopReason};

const TASK_ORDERING_BOOTSTRAP: &[u8] = include_bytes!("fixtures/python/task_ordering_bootstrap.py");

fn run_shell(source: &str) -> (i32, Vec<u8>, Vec<u8>) {
    let mut environment = Environment::new();
    let (outcome, stdout, stderr) = environment.run_script_capture(source);
    (outcome.exit_status, stdout, stderr)
}

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
fn indented_control_flow_functions_and_iteration_run_as_bytecode() {
    let source = r#"def fib(n):
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)

total = 0
for value in [1, 2, 3, 4]:
    if value == 3:
        continue
    total = total + value
else:
    total = total + 10

for value in [1, 2, 3]:
    if value == 2:
        break
else:
    total = 999

count = 3
while count:
    total = total + count
    count = count - 1

print(fib(8), total)"#;
    let mut environment = Environment::new();
    let argv = vec!["python3.14".into(), "-c".into(), source.into()];
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let status = shellsim::python::run_python(
        &mut environment,
        &argv,
        Vec::new(),
        &mut stdout,
        &mut stderr,
    );
    assert_eq!(status, 0, "{}", String::from_utf8_lossy(&stderr));
    assert_eq!(stdout, b"21 23\n");
    assert!(stderr.is_empty());
}

#[test]
fn mutable_collection_methods_preserve_aliasing() {
    let source = r#"data = {}
data["a"] = 1
same = data
same.setdefault("b", 2)
items = []
for item in data.items():
    items.append(item)
values = [1]
values.extend([2, 3])
last = values.pop()
members = {1}
members.add(2)
members.update([2, 3])
members.discard(99)
parts = "  a=b c  ".strip().split()
pair = parts[0].split("=", 1)
print(data is same, data["b"], items)
print(values, last, len(members), 3 in members)
print(parts, pair, "hello".startswith("he"), "x\n".rstrip("\n"))"#;
    let mut environment = Environment::new();
    let argv = vec!["python3.14".into(), "-c".into(), source.into()];
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let status = shellsim::python::run_python(
        &mut environment,
        &argv,
        Vec::new(),
        &mut stdout,
        &mut stderr,
    );
    assert_eq!(status, 0, "{}", String::from_utf8_lossy(&stderr));
    assert_eq!(
        stdout,
        b"True 2 [('a', 1), ('b', 2)]\n[1, 2] 3 3 True\n['a=b', 'c'] ['a', 'b'] True x\n"
    );
    assert!(stderr.is_empty());
}

#[test]
fn unpacking_augmented_assignment_and_iterator_builtins_are_generic() {
    let source = r#"left, right = 1, 2
numbers = [10, 20]
total = 0
for index, (number, offset) in enumerate(zip(numbers, range(1, 3)), 5):
    total += index + number + offset
scores = {"x": 1}
scores["x"] += right
print(left, right, total, scores["x"])
print(tuple(numbers), list((3, 4)), len(set([1, 1, 2])))
print(any([False, 1]), all([True, 1]), bool([]), bool([0]))"#;
    let mut environment = Environment::new();
    let argv = vec!["python3.14".into(), "-c".into(), source.into()];
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let status = shellsim::python::run_python(
        &mut environment,
        &argv,
        Vec::new(),
        &mut stdout,
        &mut stderr,
    );
    assert_eq!(status, 0, "{}", String::from_utf8_lossy(&stderr));
    assert_eq!(
        stdout,
        b"1 2 44 3\n(10, 20) [3, 4] 2\nTrue True False True\n"
    );
    assert!(stderr.is_empty());
}

#[test]
fn keyword_calls_and_json_dumps_follow_the_same_call_protocol() {
    let source = r#"import json
def combine(a, b):
    return a * 10 + b

payload = {"z": [1, True, None], "a": "x"}
print(combine(b=2, a=3))
print(json.dumps(payload))
print(json.dumps(payload, separators=(",", ":"), sort_keys=True))"#;
    let mut environment = Environment::new();
    let argv = vec!["python3.14".into(), "-c".into(), source.into()];
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let status = shellsim::python::run_python(
        &mut environment,
        &argv,
        Vec::new(),
        &mut stdout,
        &mut stderr,
    );
    assert_eq!(status, 0, "{}", String::from_utf8_lossy(&stderr));
    assert_eq!(
        stdout,
        b"32\n{\"z\": [1, true, null], \"a\": \"x\"}\n{\"a\":\"x\",\"z\":[1,true,null]}\n"
    );
    assert!(stderr.is_empty());
}

#[test]
fn json_loads_builds_python_values_and_round_trips_in_order() {
    let source = r#"import json
value = json.loads('{"z": [1, true, null, "x"], "a": -2.5}')
print(value["z"][0], value["z"][1], value["z"][2], value["z"][3], value["a"])
print(json.dumps(value, separators=(",", ":"), sort_keys=True))"#;
    let mut environment = Environment::new();
    let argv = vec!["python3.14".into(), "-c".into(), source.into()];
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let status = shellsim::python::run_python(
        &mut environment,
        &argv,
        Vec::new(),
        &mut stdout,
        &mut stderr,
    );
    assert_eq!(status, 0, "{}", String::from_utf8_lossy(&stderr));
    assert_eq!(
        stdout,
        b"1 True None x -2.5\n{\"a\":-2.5,\"z\":[1,true,null,\"x\"]}\n"
    );
    assert!(stderr.is_empty());
    if let Ok(reference) = Command::new("python3.14").arg("-c").arg(source).output() {
        assert_eq!(status, reference.status.code().unwrap_or(1));
        assert_eq!(stdout, reference.stdout);
        assert_eq!(stderr, reference.stderr);
    }
}

#[test]
fn json_loads_rejects_invalid_input_and_unbounded_integers() {
    for source in [
        "import json; json.loads('{')",
        "import json; json.loads('9223372036854775808')",
        "import json; json.loads(1)",
    ] {
        let mut environment = Environment::new();
        let argv = vec!["python3.14".into(), "-c".into(), source.into()];
        let mut stdout = Vec::new();
        let mut stderr = Vec::new();
        let status = shellsim::python::run_python(
            &mut environment,
            &argv,
            Vec::new(),
            &mut stdout,
            &mut stderr,
        );
        assert_ne!(status, 0, "source unexpectedly succeeded: {source}");
        assert!(stdout.is_empty());
        assert!(!stderr.is_empty());
    }
}

#[test]
fn native_module_errors_preserve_python_exception_kinds() {
    assert_eq!(
        run_shell(
            "python3.14 -c 'import json; import math\ntry:\n    json.loads(\"{\")\nexcept ValueError:\n    print(\"json value\")\ntry:\n    math.sqrt(-1)\nexcept ValueError:\n    print(\"math value\")\ntry:\n    math.sqrt(\"x\")\nexcept TypeError:\n    print(\"math type\")'"
        ),
        (
            0,
            b"json value\nmath value\nmath type\n".to_vec(),
            Vec::new()
        )
    );
}

#[test]
fn lambdas_and_keyed_sorted_preserve_stability() {
    let source = r#"items = [("b", 2), ("a", 3), ("a", 1)]
print(sorted(items, key=lambda item: item[0]))
print(sorted(items, key=lambda item: item[1], reverse=True))
items.sort(key=lambda item: item[1])
print(items)"#;
    let mut environment = Environment::new();
    let argv = vec!["python3.14".into(), "-c".into(), source.into()];
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let status = shellsim::python::run_python(
        &mut environment,
        &argv,
        Vec::new(),
        &mut stdout,
        &mut stderr,
    );
    assert_eq!(status, 0, "{}", String::from_utf8_lossy(&stderr));
    assert_eq!(
        stdout,
        b"[('a', 3), ('a', 1), ('b', 2)]\n[('a', 3), ('b', 2), ('a', 1)]\n[('a', 1), ('b', 2), ('a', 3)]\n"
    );
    assert!(stderr.is_empty());
}

#[test]
fn collections_defaultdict_uses_normal_factories_and_mapping_protocols() {
    let source = r#"import json
from collections import defaultdict

groups = defaultdict(list)
for name, value in [("x", 1), ("y", 2), ("x", 3)]:
    groups[name].append(value)
print(groups["x"], list(groups.keys()))
print(json.dumps(groups, separators=(",", ":"), sort_keys=True))"#;
    let mut environment = Environment::new();
    let argv = vec!["python3.14".into(), "-c".into(), source.into()];
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let status = shellsim::python::run_python(
        &mut environment,
        &argv,
        Vec::new(),
        &mut stdout,
        &mut stderr,
    );
    assert_eq!(status, 0, "{}", String::from_utf8_lossy(&stderr));
    assert_eq!(stdout, b"[1, 3] ['x', 'y']\n{\"x\":[1,3],\"y\":[2]}\n");
    assert!(stderr.is_empty());
}

#[test]
fn nonlocal_updates_the_nearest_persistent_lexical_scope() {
    let source = r#"def make_counter(start):
    value = start
    def next_value():
        nonlocal value
        value += 1
        return value
    return next_value

counter = make_counter(40)
print(counter(), counter())"#;
    let mut environment = Environment::new();
    let argv = vec!["python3.14".into(), "-c".into(), source.into()];
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let status = shellsim::python::run_python(
        &mut environment,
        &argv,
        Vec::new(),
        &mut stdout,
        &mut stderr,
    );
    assert_eq!(status, 0, "{}", String::from_utf8_lossy(&stderr));
    assert_eq!(stdout, b"41 42\n");
    assert!(stderr.is_empty());
}

#[test]
fn user_classes_bind_methods_and_keep_instance_attributes() {
    let source = r#"class DSU:
    def __init__(self):
        self.parent = {}

    def find(self, value):
        if value not in self.parent:
            self.parent[value] = value
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left, right):
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root

first = DSU()
second = DSU()
first.union("a", "b")
print(first.find("b"), second.find("b"), first.parent is second.parent)"#;
    let mut environment = Environment::new();
    let argv = vec!["python3.14".into(), "-c".into(), source.into()];
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let status = shellsim::python::run_python(
        &mut environment,
        &argv,
        Vec::new(),
        &mut stdout,
        &mut stderr,
    );
    assert_eq!(status, 0, "{}", String::from_utf8_lossy(&stderr));
    assert_eq!(stdout, b"a b False\n");
    assert!(stderr.is_empty());
}

#[test]
fn user_class_inheritance_uses_c3_attribute_lookup() {
    assert_eq!(
        run_shell(
            "python3.14 -c 'class A:\n    def __init__(self, value):\n        self.value = value\n    def source(self):\n        return \"A\"\nclass B(A):\n    pass\nclass C(A):\n    def source(self):\n        return \"C\"\nclass D(B, C):\n    pass\nd = D(7)\nprint(d.value, d.source(), D.source(d))'"
        ),
        (0, b"7 C C\n".to_vec(), Vec::new())
    );

    let (status, _stdout, stderr) = run_shell(
        "python3.14 -c 'class X:\n    pass\nclass Y:\n    pass\nclass A(X, Y):\n    pass\nclass B(Y, X):\n    pass\nclass Invalid(A, B):\n    pass'",
    );
    assert_ne!(status, 0);
    assert!(String::from_utf8_lossy(&stderr)
        .contains("cannot create a consistent method resolution order"));
}

#[test]
fn int_subclasses_preserve_identity_and_use_numeric_protocols() {
    assert_eq!(
        run_shell(
            "python3.14 -c 'class UserId(int):\n    def next_id(self):\n        return self + 1\nvalue = UserId(12)\nzero = UserId()\nprint(value, int(value), value.next_id())\nprint(value == 12, value > 3, bool(value), bool(zero))'"
        ),
        (0, b"12 12 13\nTrue True True False\n".to_vec(), Vec::new())
    );
}

#[test]
fn type_predicates_follow_user_mro_and_builtin_layouts() {
    assert_eq!(
        run_shell(
            "python3.14 -c 'class Root(object):\n    pass\nclass UserId(int):\n    pass\nclass Child(UserId):\n    pass\nvalue = Child(4)\nprint(type(value) is Child, type(Child) is type)\nprint(isinstance(value, Child), isinstance(value, UserId), isinstance(value, int), isinstance(value, object))\nprint(issubclass(Child, UserId), issubclass(Child, int), issubclass(Child, object), issubclass(bool, int))\nprint(isinstance(Root(), Root))'"
        ),
        (
            0,
            b"True True\nTrue True True True\nTrue True True True\nTrue\n".to_vec(),
            Vec::new()
        )
    );
}

#[test]
fn constrained_metaclasses_preserve_class_identity() {
    assert_eq!(
        run_shell(
            "python3.14 -c 'class Meta(type):\n    label = \"model\"\nclass X(metaclass=Meta):\n    pass\nclass Child(X):\n    pass\nprint(type(int) is type, type(Meta) is type, type(X) is Meta, type(Child) is Meta)\nprint(isinstance(X, Meta), issubclass(Meta, type), isinstance(X(), X), X.label)'"
        ),
        (
            0,
            b"True True True True\nTrue True True model\n".to_vec(),
            Vec::new()
        )
    );

    let (status, _stdout, stderr) =
        run_shell("python3.14 -c 'class Unsafe(type):\n    def __call__(self):\n        pass'");
    assert_ne!(status, 0);
    assert!(String::from_utf8_lossy(&stderr)
        .contains("custom metaclass construction hooks are not implemented"));
}

#[test]
fn reduced_tasktrove_task_ordering_fixture_matches_cpython() {
    let mut environment = Environment::new();
    environment
        .vfs
        .put_file(
            "/app/task_ordering_bootstrap.py",
            TASK_ORDERING_BOOTSTRAP.to_vec(),
            0o644,
        )
        .unwrap();
    let (outcome, stdout, stderr) =
        environment.run_script_capture("python3.14 /app/task_ordering_bootstrap.py");
    assert_eq!(
        outcome.exit_status,
        0,
        "{}",
        String::from_utf8_lossy(&stderr)
    );
    assert_eq!(stdout, b"['all', 'compile', 'test']\n");
    assert!(stderr.is_empty());

    if let Ok(reference) = Command::new("python3.14")
        .arg("tests/fixtures/python/task_ordering_bootstrap.py")
        .output()
    {
        assert_eq!(outcome.exit_status, reference.status.code().unwrap_or(1));
        assert_eq!(stdout, reference.stdout);
        assert_eq!(stderr, reference.stderr);
    }
}

#[test]
fn vfs_modules_keep_module_globals_and_lexical_closures() {
    let helper = br#"from __future__ import annotations

offset: int = 7

def add(value: int) -> int:
    return value + offset

def make_adder(left: int) -> object:
    def add_right(right: int) -> int:
        return left + right
    return add_right
"#;
    let main = br#"from __future__ import annotations
import helper as helpers
from helper import add as plus
add_ten = helpers.make_adder(10)
print(helpers.add(5), plus(6), add_ten(3))
"#;
    let mut environment = Environment::new();
    environment
        .vfs
        .put_file("/app/helper.py", helper.to_vec(), 0o644)
        .unwrap();
    environment
        .vfs
        .put_file("/app/main.py", main.to_vec(), 0o644)
        .unwrap();
    let (outcome, stdout, stderr) = environment.run_script_capture("python3.14 /app/main.py");
    assert_eq!(
        outcome.exit_status,
        0,
        "{}",
        String::from_utf8_lossy(&stderr)
    );
    assert_eq!(stdout, b"12 13 13\n");
    assert!(stderr.is_empty());
}

#[test]
fn computed_string_allocation_is_bounded_before_allocation() {
    let mut environment = Environment::with_limits(Limits {
        memory: 40 * 1024,
        ..Limits::unlimited()
    });
    let (outcome, stdout, _) =
        environment.run_script_capture("python3.14 -c 'print(\"x\" * 1000000)'");
    assert_eq!(outcome.exit_status, 137);
    assert_eq!(outcome.stop_reason, Some(StopReason::MemoryExhausted));
    assert!(stdout.is_empty());
}

#[test]
fn python_loops_consume_fuel_per_bytecode_instruction() {
    let mut environment = Environment::with_limits(Limits {
        cpu: 500,
        ..Limits::unlimited()
    });
    let source = "while True:\n    pass\n";
    let argv = vec!["python3.14".into(), "-c".into(), source.into()];
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let status = shellsim::python::run_python(
        &mut environment,
        &argv,
        Vec::new(),
        &mut stdout,
        &mut stderr,
    );
    assert_eq!(status, 137);
    assert_eq!(
        environment.outcome(status).stop_reason,
        Some(StopReason::CpuExhausted)
    );
    assert!(stdout.is_empty());
}

#[test]
fn executes_python_scripts_from_the_virtual_filesystem() {
    assert_eq!(
        run_shell(
            "printf '%s\\n' 'import sys' 'print(sys.argv[0], sys.argv[1])' > /tool.py; python3.14 /tool.py value"
        ),
        (0, b"/tool.py value\n".to_vec(), Vec::new())
    );
    assert_eq!(
        run_shell(
            "printf '%s\\n' '#!/usr/bin/env python3' 'import sys' 'print(sys.argv[1])' > /tool.py; chmod +x /tool.py; /tool.py shebang"
        ),
        (0, b"shebang\n".to_vec(), Vec::new())
    );
}

/// When the development host has CPython 3.14, compare the same source and argv directly. The
/// checked expectations above remain authoritative on builders where that executable is absent.
#[test]
fn differential_scalar_cases_match_cpython_314_when_available() {
    let cases: &[(&str, &[&str])] = &[
        ("print(1 + 2 * 3, 5 / 2, -7 // 3, -7 % 3)", &[]),
        (
            "a = [1, \"x\"]; b = a; print(a, (1,), {\"a\": 2}, {}); print(a is b, 1 < 2 < 3, 1 < 2 > 3); print(0 and missing, 1 or missing, not [], \"x\" in \"xyz\", 2 in [1, 2]); print({\"a\": 2}[\"a\"], {1, 2} == {2, 1})",
            &[],
        ),
        (
            "import sys; print(sys.argv[0]); print(sys.argv[-1])",
            &["last"],
        ),
        ("print(\"café\\nline\")", &[]),
    ];

    for (source, arguments) in cases {
        let reference = match Command::new("python3.14")
            .arg("-c")
            .arg(source)
            .args(*arguments)
            .output()
        {
            Ok(output) => output,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return,
            Err(error) => panic!("failed to run CPython 3.14 reference: {error}"),
        };
        let quoted_source = source.replace('\'', "'\\''");
        let shell_arguments = arguments
            .iter()
            .map(|argument| format!(" '{argument}'"))
            .collect::<String>();
        let simulated = run_shell(&format!("python3.14 -c '{quoted_source}'{shell_arguments}"));
        assert_eq!(
            simulated.0,
            reference.status.code().unwrap_or(1),
            "{source}"
        );
        assert_eq!(simulated.1, reference.stdout, "{source}");
        assert_eq!(simulated.2, reference.stderr, "{source}");
    }
}
