//! Execution-representation tests cover slot-backed lexical scopes and lazy builtin iterators.

use shellsim::{python, Environment, Limits};

fn run(source: &str, limits: Limits) -> (i32, Vec<u8>, Vec<u8>, shellsim::resources::Usage) {
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
fn indexed_locals_remain_visible_to_closures_and_nonlocal_stores() {
    let source = r#"def outer():
    value = 1
    def read():
        return value
    def write():
        nonlocal value
        value = value + 2
    write()
    return read()

print(outer())"#;
    let (status, stdout, stderr, _) = run(source, Limits::unlimited());
    assert_eq!(status, 0, "{}", String::from_utf8_lossy(&stderr));
    assert_eq!(stdout, b"3\n");
    assert!(stderr.is_empty());
}

#[test]
fn ranges_are_reusable_and_sequence_iterators_observe_appends() {
    let source = r#"numbers = range(1, 6, 2)
print(list(numbers), list(numbers), len(numbers), numbers[-1], numbers == range(1, 7, 2))
values = [1, 2]
seen = []
for value in values:
    seen.append(value)
    if value == 1:
        values.append(3)
print(seen)"#;
    let (status, stdout, stderr, _) = run(source, Limits::unlimited());
    assert_eq!(status, 0, "{}", String::from_utf8_lossy(&stderr));
    assert_eq!(stdout, b"[1, 3, 5] [1, 3, 5] 3 5 True\n[1, 2, 3]\n");
    assert!(stderr.is_empty());
}

#[test]
fn a_large_range_uses_constant_memory_until_values_are_consumed() {
    let source = r#"total = 0
for value in range(1000000000):
    total += value
    if value == 3:
        break
print(total)"#;
    let (status, stdout, stderr, usage) = run(
        source,
        Limits {
            memory: 64 * 1024,
            ..Limits::unlimited()
        },
    );
    assert_eq!(status, 0, "{}", String::from_utf8_lossy(&stderr));
    assert_eq!(stdout, b"6\n");
    assert!(usage.memory_peak <= 64 * 1024);
}

#[test]
fn instance_shapes_preserve_late_attributes_and_dictionary_fallback() {
    let mut source = String::from(
        "class Item:\n    pass\ndef read_left(item):\n    return item.left\nfirst = Item()\nsecond = Item()\nfirst.left = 1\nfirst.right = 2\nsecond.right = 3\nsecond.left = 4\nprint(read_left(first))\nfirst.extra = 6\nprint(read_left(first))\nfirst.right = 5\n",
    );
    for index in 0..40 {
        source.push_str(&format!("first.field_{index} = {index}\n"));
    }
    source.push_str("print(read_left(first))\nprint(first.left, first.right, second.left, second.right, first.field_0, first.field_39)\n");

    let (status, stdout, stderr, _) = run(&source, Limits::unlimited());
    assert_eq!(status, 0, "{}", String::from_utf8_lossy(&stderr));
    assert_eq!(stdout, b"1\n1\n1\n1 5 4 3 0 39\n");
    assert!(stderr.is_empty());
}

#[test]
fn divergent_instance_attributes_remain_memory_bounded() {
    let mut source = String::from("class Item:\n    pass\nitem = Item()\n");
    for index in 0..2_000 {
        source.push_str(&format!("item.field_{index} = {index}\n"));
    }
    let (status, stdout, stderr, usage) = run(
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
fn attribute_lookaside_preserves_non_data_descriptor_precedence() {
    let source = r#"class Item:
    def label(self):
        return "class"

def read_label(item):
    return item.label

item = Item()
item.label = "instance"
print(read_label(item), read_label(item))
item.extra = 1
print(read_label(item))"#;
    let (status, stdout, stderr, _) = run(source, Limits::unlimited());
    assert_eq!(status, 0, "{}", String::from_utf8_lossy(&stderr));
    assert_eq!(stdout, b"instance instance\ninstance\n");
    assert!(stderr.is_empty());
}
