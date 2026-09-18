//! Operational compatibility checks that need Rust-side output or host-reference assertions.

use std::process::Command;

use shellsim::Environment;

use super::support::run_shell;

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
fn ordered_mapping_index_preserves_numeric_keys_and_deletion_order() {
    let source = r#"values = {1: "int", 2: "two", 3: "three"}
values[True] = "bool"
values[1.0] = "float"
removed = values.pop(2)
values[4] = "four"
long_key = "key-longer-than-inline-storage"
values[long_key] = "long"
print(len(values), values[1], removed, values[3], values[long_key])
print(values.keys())"#;
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
        b"4 float two three long\n[1, 3, 4, 'key-longer-than-inline-storage']\n"
    );
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
print(json.dumps(payload, separators=(",", ":"), sort_keys=True))
print(json.dumps({"snowman": "☃"}))
print(json.dumps({"snowman": "☃"}, ensure_ascii=False))"#;
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
        b"32\n{\"z\": [1, true, null], \"a\": \"x\"}\n{\"a\":\"x\",\"z\":[1,true,null]}\n{\"snowman\": \"\\u2603\"}\n{\"snowman\": \"\xe2\x98\x83\"}\n"
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
fn json_loads_rejects_invalid_input_types_and_syntax() {
    for source in ["import json; json.loads('{')", "import json; json.loads(1)"] {
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
fn json_preserves_arbitrary_precision_integers() {
    let source = "import json; value = json.loads('922337203685477580812345'); print(value); print(json.dumps(value))";
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
        b"922337203685477580812345\n922337203685477580812345\n"
    );
    assert!(stderr.is_empty());
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
fn callable_sentinel_iter_is_lazy_and_stops_before_the_sentinel() {
    let source = r#"
from io import BytesIO
stream = BytesIO(b'abcdef')
chunks = iter(lambda: stream.read(2), b'')
print(stream.tell())
print(list(chunks))
print(stream.tell())
"#;
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
    assert_eq!(stdout, b"0\n[b'ab', b'cd', b'ef']\n6\n");
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
