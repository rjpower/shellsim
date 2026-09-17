//! Differential coverage for ordinary syntax found in unchanged TaskTrove Python sources.

use shellsim::{python, Environment};

fn run(source: &str) -> (i32, String, String) {
    let mut environment = Environment::default();
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let status = python::run_python(
        &mut environment,
        &["python3.14".into(), "-c".into(), source.into()],
        Vec::new(),
        &mut stdout,
        &mut stderr,
    );
    (
        status,
        String::from_utf8(stdout).expect("UTF-8 stdout"),
        String::from_utf8(stderr).expect("UTF-8 stderr"),
    )
}

#[test]
fn multiline_strings_line_continuations_and_comment_first_suites_execute() {
    let source = r#"
def documented():
    # A comment may precede the first real statement.
    """A multiline
    docstring."""
    return 20 + \
        22

print(documented())
"#;
    assert_eq!(run(source), (0, "42\n".into(), String::new()));
}

#[test]
fn bitwise_conditional_and_slice_expressions_match_python_precedence() {
    let source = r#"
values = [0, 1, 2, 3, 4, 5]
print(1 | 2 & 6 ^ 1)
print(~5)
print(-17 // 5, -17 % 5)
print(values[1:5:2], values[::-1], "café"[1:3])
print("yes" if values[:2] == [0, 1] else "no")
"#;
    assert_eq!(
        run(source),
        (
            0,
            "3\n-6\n-4 3\n[1, 3] [5, 4, 3, 2, 1, 0] af\nyes\n".into(),
            String::new()
        )
    );
}

#[test]
fn tuple_subscripts_work_for_ordinary_mapping_keys() {
    let source = r#"
values = {(1, 2): "pair", (1,): "single"}
print(values[1, 2], values[1,])
"#;
    assert_eq!(run(source), (0, "pair single\n".into(), String::new()));
}

#[test]
fn all_arithmetic_bytecodes_use_type_slots() {
    let source = r#"
class Number:
    def __init__(self, value):
        self.value = value

    def __floordiv__(self, other):
        return Number(self.value // other.value)

    def __mod__(self, other):
        return Number(self.value % other.value)

    def __neg__(self):
        return Number(-self.value)

    def __invert__(self):
        return Number(~self.value)

    def __abs__(self):
        return Number(abs(self.value))

class Addend:
    def __init__(self, value):
        self.value = value

    def __add__(self, other):
        return Addend(self.value + other.value)

    def __radd__(self, other):
        return Addend(other + self.value)

left = Number(17)
right = Number(5)
print((left // right).value, (left % right).value)
print((-left).value, (~left).value, abs(Number(-9)).value)
print(sum([Addend(2), Addend(3)]).value)
"#;
    assert_eq!(
        run(source),
        (0, "3 2\n-17 -18 9\n5\n".into(), String::new())
    );
}

#[test]
fn bitwise_operators_dispatch_through_user_type_slots() {
    let source = r#"
class Mask:
    def __init__(self, value):
        self.value = value

    def __or__(self, other):
        return Mask(self.value | other.value)

print((Mask(4) | Mask(2)).value)
"#;
    assert_eq!(run(source), (0, "6\n".into(), String::new()));
}

#[test]
fn raw_fstrings_preserve_backslashes() {
    assert_eq!(
        run(r#"name = "item"
print(rf"^{name}-?(\d+)$")
"#),
        (0, "^item-?(\\d+)$\n".into(), String::new())
    );
}

#[test]
fn common_fstring_conversions_and_numeric_formats_are_supported() {
    let source = r#"
value = 12.34567
print(f"{value:.2f} {value:08.3f} {42:05d}")
print(f"{'x':>3} {['a']!r}")
"#;
    assert_eq!(
        run(source),
        (0, "12.35 0012.346 00042\n  x ['a']\n".into(), String::new())
    );
}

#[test]
fn adjacent_string_literals_are_folded_before_execution() {
    let source = r#"
name = "world"
print("hello, " f"{name}" "!")
print(b"ab" b"cd")
"#;
    assert_eq!(
        run(source),
        (0, "hello, world!\nb'abcd'\n".into(), String::new())
    );
    assert!(run("print(b'x' 'y')")
        .2
        .contains("cannot mix bytes and nonbytes"));
}

#[test]
fn dictionary_unpacking_uses_source_order_and_last_value_wins() {
    let source = r#"
base = {"a": 1, "b": 2}
print({**base, "b": 3, **{"c": 4}})
"#;
    assert_eq!(
        run(source),
        (0, "{'a': 1, 'b': 3, 'c': 4}\n".into(), String::new())
    );
    assert!(run("print({**[1, 2]})").2.contains("must be a mapping"));
}

#[test]
fn compact_language_wins_match_python_semantics() {
    let source = r#"
a = b = 0x10 + 0o10 + 0b10
a ^= 3
b **= 2
if a == 25: a += 1; b //= 2
print(a, b)
print(-2 ** 2, 2 ** -2, 2 ** 3 ** 2)

class Power:
    def __pow__(self, other):
        return 40 + other

print(Power() ** 2)

state = 1
def bump():
    global state
    state += 1
bump()
values = {'keep': 1, 'remove': 2}
del values['remove']
print(state, values)

class Bag:
    def __init__(self):
        self.deleted = None
    def __delitem__(self, key):
        self.deleted = key
bag = Bag()
del bag['item']
print(bag.deleted)
"#;
    assert_eq!(
        run(source),
        (
            0,
            "26 338\n-4 0.25 512\n42\n2 {'keep': 1}\nitem\n".into(),
            String::new()
        )
    );
}
