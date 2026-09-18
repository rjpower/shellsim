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
    assert_eq!(status, 2);
    assert!(
        String::from_utf8_lossy(&stderr).contains("not in range"),
        "{}",
        String::from_utf8_lossy(&stderr)
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
            "python3.14 -c 'class Root(object):\n    pass\nclass UserId(int):\n    pass\nclass Child(UserId):\n    pass\nvalue = Child(4)\nprint(type(value) is Child, type(Child) is type)\nprint(isinstance(value, Child), isinstance(value, UserId), isinstance(value, int), isinstance(value, object))\nprint(issubclass(Child, UserId), issubclass(Child, int), issubclass(Child, object), issubclass(bool, int))\nprint(isinstance(Root(), Root))\nprint(isinstance(value, (str, int)), isinstance(value, (str, bytes)))\nprint(issubclass(Child, (str, int)), issubclass(Child, (str, bytes)))'"
        ),
        (
            0,
            b"True True\nTrue True True True\nTrue True True True\nTrue\nTrue False\nTrue False\n".to_vec(),
            Vec::new()
        )
    );
}

#[test]
fn builtin_classes_have_canonical_type_identity_and_constructors() {
    assert_eq!(
        run_shell(
            "python3.14 -c 'import math\nprint(type(None), type(1) is int, type(1.5) is float, type(\"\") is str)\nprint(type([]) is list, type(()) is tuple, type({}) is dict, type(set()) is set)\nprint(type(len), type(math), isinstance(1.5, float), issubclass(float, object))\nprint(float(), float(\"1.5\"), str(), dict({\"a\": 1}), list((1, 2)))'"
        ),
        (
            0,
            b"<class 'NoneType'> True True True\nTrue True True True\n<class 'function'> <class 'module'> True True\n0.0 1.5  {'a': 1} [1, 2]\n"
                .to_vec(),
            Vec::new()
        )
    );
}

#[test]
fn immediate_scalar_identity_uses_canonical_value_representations() {
    assert_eq!(
        run_shell(
            "python3.14 -c 'integer = 1000\nsame_integer = integer\nnumber = 1.5\nsame_number = number\ntext = \"value\"\nsame_text = text\nnan = float(\"nan\")\nprint(None is None, True is True, False is False)\nprint(integer is same_integer, number is same_number, text is same_text)\nprint(nan is nan, nan == nan, nan in [nan], [nan] == [nan])'"
        ),
        (
            0,
            b"True True True\nTrue True True\nTrue False True True\n".to_vec(),
            Vec::new()
        )
    );
}

#[test]
fn strings_share_one_type_across_inline_and_heap_storage() {
    assert_eq!(
        run_shell(
            "python3.14 -c 'short = \"123456789012345\"\nlong = \"1234567890123456\"\nsame = long\nother = \"1234567890123456\"\nprint(type(short) is str, type(long) is str)\nprint(long is same, long is other, long == other)\nprint(long + other, long * 2)'",
        ),
        (
            0,
            b"True True\nTrue False True\n12345678901234561234567890123456 12345678901234561234567890123456\n"
                .to_vec(),
            Vec::new(),
        )
    );
}

#[test]
fn descriptors_and_zero_argument_super_share_method_binding() {
    assert_eq!(
        run_shell(
            "python3.14 -c 'class Base:\n    label = \"base\"\n    def __init__(self, value):\n        self._value = value\n    @property\n    def value(self):\n        return self._value\n    @value.setter\n    def value(self, value):\n        self._value = value\n    @staticmethod\n    def add(left, right):\n        return left + right\n    @classmethod\n    def class_label(cls):\n        return cls.label\n    def describe(self):\n        return \"Base\"\nclass Child(Base):\n    label = \"child\"\n    def describe(self):\n        return super().describe() + \" Child\"\nvalue = Child(4)\nprint(value.value, Child.add(2, 3), value.add(3, 4))\nvalue.value = 9\nunbound = Child.describe\nprint(value.value, Child.class_label(), value.class_label(), value.describe(), unbound(value))'"
        ),
        (
            0,
            b"4 5 7\n9 child child Base Child Base Child\n".to_vec(),
            Vec::new()
        )
    );
}

#[test]
fn user_descriptors_follow_precedence_and_receive_set_name() {
    assert_eq!(
        run_shell(
            "python3.14 -c 'class Field:\n    def __set_name__(self, owner, name):\n        self.public_name = name\n    def __get__(self, instance, owner):\n        if instance is None:\n            return self\n        return instance._stored\n    def __set__(self, instance, value):\n        instance._stored = value\nclass Record:\n    value = Field()\nrecord = Record()\nrecord.value = 12\nprint(record.value, Record.value.public_name)\nrecord.__dict_shadow = 99\nprint(record.value)'"
        ),
        (0, b"12 value\n12\n".to_vec(), Vec::new())
    );
}

#[test]
fn user_protocol_slots_dispatch_cached_dunder_methods() {
    assert_eq!(
        run_shell(
            "python3.14 -c 'class NumberBox:\n    def __init__(self, value):\n        self.value = value\n    def __call__(self, amount):\n        return self.value + amount\n    def __add__(self, other):\n        return self.value + other\n    def __eq__(self, other):\n        return self.value == other\n    def __lt__(self, other):\n        return self.value < other\n    def __contains__(self, item):\n        return item == self.value\n    def __bool__(self):\n        return self.value != 0\n    def __str__(self):\n        return \"box\"\n    def __repr__(self):\n        return \"NumberBox\"\n    def __iter__(self):\n        return [self.value, self.value + 1]\nbox = NumberBox(4)\nzero = NumberBox(0)\nprint(box(3), box + 2, box == 4, box != 4, box != 5, box < 8, 4 in box)\nprint(bool(box), bool(zero), not zero)\nprint(str(box), repr(box), list(box))'"
        ),
        (
            0,
            b"7 6 True False True True True\nTrue False True\nbox NumberBox [4, 5]\n".to_vec(),
            Vec::new()
        )
    );
}

#[test]
fn reflected_binary_slots_follow_the_rhs_type() {
    assert_eq!(
        run_shell(
            "python3.14 -c 'class Right:\n    def __radd__(self, left):\n        return left + 10\n    def __rsub__(self, left):\n        return left - 10\n    def __rmul__(self, left):\n        return left * 10\nvalue = Right()\nclass Count(int):\n    pass\nprint(2 + value, 20 - value, 3 * value)\nprint(Count(2) * \"ab\", Count(2) * [1], (1,) * Count(2))'",
        ),
        (0, b"12 10 30\nabab [1, 1] (1, 1)\n".to_vec(), Vec::new())
    );
}

#[test]
fn user_length_controls_truth_when_bool_is_absent() {
    assert_eq!(
        run_shell(
            "python - <<'PY'\nclass Sized:\n    def __init__(self, length):\n        self.length = length\n    def __len__(self):\n        return self.length\nprint(bool(Sized(2)), bool(Sized(0)))\nPY"
        ),
        (0, b"True False\n".to_vec(), Vec::new())
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

    assert_eq!(
        run_shell(
            "python3.14 -c 'class CallableMeta(type):\n    def __call__(cls, value):\n        return value + 1\nclass X(metaclass=CallableMeta):\n    pass\nprint(X(4))'"
        ),
        (0, b"5\n".to_vec(), Vec::new())
    );

    assert_eq!(
        run_shell(
            r#"python3.14 -c 'events = []
class Meta(type):
    def __prepare__(name, bases):
        events.append("prepare")
        return {"prepared": 7}
    def __init__(cls, name, bases, namespace):
        events.append("meta_init")
class Base:
    def __init_subclass__(cls):
        events.append("init_subclass")
class Child(Base, metaclass=Meta):
    body = 8
print(events, Child.prepared, Child.body)'"#
        ),
        (
            0,
            b"['prepare', 'init_subclass', 'meta_init'] 7 8\n".to_vec(),
            Vec::new()
        )
    );
}

#[test]
fn custom_metaclass_new_uses_the_shared_type_allocator() {
    assert_eq!(
        run_shell(
            r#"python3.14 -c 'events = []
class Meta(type):
    def __new__(mcls, name, bases, namespace):
        events.append("new:" + name)
        namespace["created_by"] = "Meta"
        return super().__new__(mcls, name, bases, namespace)
    def __init__(cls, name, bases, namespace):
        events.append("init:" + name)
class Item(metaclass=Meta):
    pass
Dynamic = Meta("Dynamic", (object,), {"answer": 42})
print(events, Item.created_by, type(Item) is Meta)
print(Dynamic.answer, Dynamic.created_by, type(Dynamic) is Meta)'"#,
        ),
        (
            0,
            b"['new:Item', 'init:Item', 'new:Dynamic', 'init:Dynamic'] Meta True\n42 Meta True\n"
                .to_vec(),
            Vec::new(),
        )
    );
}

#[test]
fn class_body_functions_capture_their_defining_class() {
    assert_eq!(
        run_shell(
            "python3.14 -c 'class Owner:\n    def defining_class(self):\n        return __class__\nprint(Owner().defining_class() is Owner)'",
        ),
        (0, b"True\n".to_vec(), Vec::new())
    );
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

#[test]
fn recursive_python_calls_stop_on_the_owned_frame_limit() {
    let (status, stdout, stderr) = run_shell(
        "python3.14 -c 'def recurse():\n    recurse()\nrecurse()\nprint(\"unreachable\")'",
    );
    assert_eq!(status, 2);
    assert!(stdout.is_empty());
    assert!(
        String::from_utf8_lossy(&stderr).contains("maximum recursion depth exceeded"),
        "{}",
        String::from_utf8_lossy(&stderr)
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
