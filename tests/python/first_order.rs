//! Differential coverage for ordinary syntax found in unchanged TaskTrove Python sources.

use super::support::run_python_text as run;

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
fn crlf_blank_lines_do_not_change_python_indentation() {
    let source = "class Value:\r\n    def get(self):\r\n        first = 20\r\n\r\n        second = 22\r\n        return first + second\r\n\r\nprint(Value().get())\r\n";
    assert_eq!(run(source), (0, "42\n".into(), String::new()));
}

#[test]
fn bitwise_conditional_and_slice_expressions_match_python_precedence() {
    let source = r#"
values = [0, 1, 2, 3, 4, 5]
print(1 | 2 & 6 ^ 1)
print(3 << 4, 65 >> 2)
print(~5)
print(-17 // 5, -17 % 5)
print(values[1:5:2], values[::-1], "café"[1:3])
print("yes" if values[:2] == [0, 1] else "no")
"#;
    assert_eq!(
        run(source),
        (
            0,
            "3\n48 16\n-6\n-4 3\n[1, 3] [5, 4, 3, 2, 1, 0] af\nyes\n".into(),
            String::new()
        )
    );
}

#[test]
fn set_bitwise_operators_match_set_algebra() {
    let left = "{1, 2, 3}";
    let right = "{3, 4}";
    let source = format!(
        "print(sorted({left} & {right}))\nprint(sorted({left} | {right}))\nprint(sorted({left} ^ {right}))\nprint(sorted({left} - {right}))"
    );
    assert_eq!(
        run(&source),
        (
            0,
            "[3]\n[1, 2, 3, 4]\n[1, 2, 4]\n[1, 2]\n".into(),
            String::new()
        )
    );
}

#[test]
fn round_preserves_python_numeric_types_and_ties_to_even() {
    let source = r#"
print(round(2.5), round(3.5), round(-2.5))
print(round(2.675, 2), round(1.25, ndigits=1), round(-1.25, 1))
print(round(2500, -3), round(3500, -3), round(123456789012345678901, -4))
print(isinstance(round(3.5), int), isinstance(round(3.5, 0), float))
"#;
    assert_eq!(
        run(source),
        (
            0,
            "2 4 -2\n2.67 1.2 -1.2\n2000 4000 123456789012345680000\nTrue True\n".into(),
            String::new()
        )
    );
}

#[test]
fn named_expressions_escape_only_synthetic_comprehension_scopes() {
    let source = r#"
module_values = [(module_last := value) for value in [1, 2]]
print(module_values, module_last)

def collect(values):
    selected = [(last := value * 2) for value in values if (seen := value) > 0]
    nested = [[(product := left * right) for right in [2]] for left in [3]]
    return selected, last, seen, nested, product

print(collect([-1, 2, 3]))

global_value = 0
def set_global():
    global global_value
    return [(global_value := value) for value in [4, 5]]
set_global()
print(global_value)

def outer():
    captured = 0
    def inner():
        nonlocal captured
        return [(captured := value) for value in [6, 7]]
    inner()
    return captured
print(outer())
"#;
    assert_eq!(
        run(source),
        (
            0,
            "[1, 2] 2\n([4, 6], 6, 3, [[6]], 6)\n5\n7\n".into(),
            String::new()
        )
    );
}

#[test]
fn named_expressions_in_class_comprehensions_are_rejected() {
    let (_, _, stderr) = run("class Invalid:\n    values = [(bound := value) for value in [1]]\n");
    assert!(stderr
        .contains("assignment expression within a comprehension cannot be used in a class body"));
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
fn common_numeric_and_callable_builtins_match_python() {
    let source = r#"
print(ord("A"), ord(b"z"))
print(bin(10), oct(10), hex(255), hex(-16))
print(divmod(17, 5), divmod(-17, 5))
print(callable(print), callable(1))
"#;
    assert_eq!(
        run(source),
        (
            0,
            "65 122\n0b1010 0o12 0xff -0x10\n(3, 2) (-4, 3)\nTrue False\n".into(),
            String::new()
        )
    );
}

#[test]
fn common_string_search_split_and_alignment_methods_match_python() {
    let source = r#"
value = "bananas"
print(value.find("na"), value.rfind("na"), value.index("ana"), value.rindex("a"))
print(value.count("a"), value.partition("na"), value.rpartition("na"))
print("a,b,c".rsplit(",", 1), "x".center(5, "-"))
print("abc".find("", 4), "abc".count("", 4))
print("  a  b  ".rsplit(None, 0), "  a  b  ".rsplit(None, 1))
"#;
    assert_eq!(
        run(source),
        (
            0,
            "2 4 1 5\n3 ('ba', 'na', 'nas') ('bana', 'na', 's')\n['a,b', 'c'] --x--\n-1 0\n['  a  b'] ['  a', 'b']\n".into(),
            String::new()
        )
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

    def __lshift__(self, other):
        return Mask(self.value << other.value)

print((Mask(4) | Mask(2)).value, (Mask(3) << Mask(2)).value)
"#;
    assert_eq!(run(source), (0, "6 12\n".into(), String::new()));
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
print(f"{31:#x} {31:#06x} {-10:#06X} {5:#b} {9:#o}")
print(f"{'x':>3} {['a']!r}")
"#;
    assert_eq!(
        run(source),
        (
            0,
            "12.35 0012.346 00042\n0x1f 0x001f -0X00A 0b101 0o11\n  x ['a']\n".into(),
            String::new()
        )
    );
}

#[test]
fn frozenset_is_immutable_and_preserves_left_operand_type() {
    let source = r#"
frozen = frozenset([1, 2, 2])
mutable = {2, 3}
print(frozen, frozen == {1, 2}, isinstance(frozen | mutable, frozenset))
print(isinstance(mutable | frozen, set), sorted(frozen.union([4])))
print('immutable', hasattr(frozen, 'add'))
"#;
    assert_eq!(
        run(source),
        (
            0,
            "frozenset({1, 2}) True True\nTrue [1, 2, 4]\nimmutable False\n".into(),
            String::new(),
        )
    );
}

#[test]
fn percent_formatting_justification_and_pow_cover_common_agent_output() {
    let source = r#"
print("%s %.2f %+d %04d %%" % ("value", 1.25, 3, 7))
print("%r %.3s %#x" % ([1], "abcdef", 31))
print("%(name)s=%(value)03d" % {"name": "count", "value": 5})
print("x".ljust(3, "-"), "x".rjust(3, "-"))
print(pow(2, 10), pow(2, -2))
"#;
    assert_eq!(
        run(source),
        (
            0,
            "value 1.25 +3 0007 %\n[1] abc 0x1f\ncount=005\nx-- --x\n1024 0.25\n".into(),
            String::new()
        )
    );
}

#[test]
fn dir_lists_current_globals_and_object_attributes() {
    let source = r#"
class Record:
    kind = "record"
    def __init__(self):
        self.value = 42

record = Record()
print("record" in dir(), "kind" in dir(Record), "value" in dir(record))
"#;
    assert_eq!(run(source), (0, "True True True\n".into(), String::new()));
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
