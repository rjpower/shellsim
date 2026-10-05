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
        b"4 float two three long\ndict_keys([1, 3, 4, 'key-longer-than-inline-storage'])\n"
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

#[test]
fn wildcard_imports_bind_all_or_public_names() {
    let mut environment = Environment::new();
    for (path, source) in [
        ("/public.py", "value = 7\n_private = 9\n"),
        (
            "/explicit.py",
            "__all__ = ['_private']\n_private = 11\nother = 12\n",
        ),
        ("/bad.py", "__all__ = [7]\n"),
        ("/missing.py", "__all__ = ['absent']\n"),
        ("/unordered.py", "__all__ = {'value'}\nvalue = 1\n"),
    ] {
        environment
            .vfs
            .put_file(path, source.as_bytes().to_vec(), 0o644)
            .unwrap();
    }
    // Expected output recorded from CPython 3.14 with the same modules.
    let source = r#"from public import *
assert value == 7 and "_private" not in dir()
from explicit import *
assert _private == 11 and "other" not in dir()
for module in ["bad", "missing", "unordered"]:
    try:
        exec(f"from {module} import *")
    except (TypeError, AttributeError) as error:
        print(type(error).__name__, error)
from math import *
print(cos(0), floor(2.5))
"#;
    assert_eq!(
        super::support::run_python_text_in(&mut environment, source),
        (
            0,
            "TypeError Item in bad.__all__ must be str, not int\n\
             AttributeError module 'missing' has no attribute 'absent'\n\
             TypeError 'set' object does not support indexing\n\
             1.0 2\n"
                .into(),
            String::new()
        )
    );
}

/// `globals()` inside an imported module is backed by that module's own lexical scope (see
/// `heap::NamespaceTarget::Scope`), distinct from the flat table backing the top-level script's
/// own `globals()` (`NamespaceTarget::Repl`). `module.__dict__`, `vars(module)` and module-level
/// `locals()` view the same scope. `tests/python/test_language.py` runs entirely inside one `-c`
/// script, so it never exercises the scope-backed path; a real `import` is needed to reach it.
#[test]
fn globals_of_an_imported_module_is_the_modules_own_live_scope() {
    let mut environment = Environment::new();
    environment
        .vfs
        .put_file(
            "/counter.py",
            b"count = 0\n\n\
              def bump():\n    \
                  globals()[\"count\"] = globals()[\"count\"] + 1\n    \
                  return globals()[\"count\"]\n\n\
              class Marker:\n    \
                  same_module = \"count\" in globals()\n\n\
              locals()[\"from_locals\"] = \"module\"\n\
              seen_from_locals = from_locals\n"
                .to_vec(),
            0o644,
        )
        .unwrap();
    let source = r#"import counter
print(counter.bump())
print(counter.bump())
print(counter.count)
print(counter.Marker.same_module)
print("count" in globals(), "counter" in globals())
print(counter.seen_from_locals, counter.__dict__["count"], vars(counter) == counter.__dict__)
counter.__dict__["extra"] = 5
print(counter.extra, "bump" in vars(counter))
"#;
    assert_eq!(
        super::support::run_python_text_in(&mut environment, source),
        (
            0,
            "1\n2\n2\nTrue\nFalse True\nmodule 2 True\n5 True\n".into(),
            String::new()
        )
    );
}

/// Namespaces live on name-keyed storage, so binding a key that is not a string raises
/// `TypeError` where CPython's dict-backed namespaces would accept it.
#[test]
fn namespace_views_reject_non_string_keys_on_write() {
    let source = r#"class Box:
    pass

for view in (globals(), vars(Box())):
    try:
        view[1] = "one"
    except TypeError:
        print("TypeError")
"#;
    assert_eq!(
        super::support::run_python_text(source),
        (0, "TypeError\nTypeError\n".into(), String::new())
    );
}

/// Locals of a function without nested scopes live in the VM frame, not in a heap scope, so
/// they must stay rooted across collections and still be visible to `locals()`, `eval` and
/// nested frames, while closures, `nonlocal` and comprehension walruses keep working through
/// the heap scopes of the functions that need them.
#[test]
fn frame_held_locals_survive_collection_and_scope_protocols() {
    let source = r#"def churn(n):
    keep = [i for i in range(8)]
    first = keep[:]
    for i in range(n):
        garbage = [i] * 64
        if i % 1000 == 0:
            keep = keep + [i]
    return first, keep[-1], len(keep)

print(churn(20000))

def snapshot(a, b=2):
    c = a + b
    del b
    return sorted(locals().items()), eval("a + c")

print(snapshot(1))

def outer():
    count = 0
    def bump():
        nonlocal count
        count += 1
        return count
    squares = [(last := i * i) for i in range(4)]
    return bump(), bump(), squares, last

print(outer())

def recurse(depth):
    local = depth * 2
    if depth == 0:
        return [local]
    return recurse(depth - 1) + [local]

print(recurse(5))

def gen(n):
    total = 0
    for i in range(n):
        total += i
        yield total

print(list(gen(5)))
"#;
    assert_eq!(
        super::support::run_python_text(source),
        (
            0,
            "([0, 1, 2, 3, 4, 5, 6, 7], 19000, 28)\n\
             ([('a', 1), ('c', 3)], 4)\n\
             (1, 2, [0, 1, 4, 9], 9)\n\
             [0, 2, 4, 6, 8, 10]\n\
             [0, 1, 3, 6, 10]\n"
                .into(),
            String::new()
        )
    );
}
