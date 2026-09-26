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
fn format_specs_support_alignment_string_width_and_signed_precision() {
    let source = r#"
x = "cat"
print("|{x:>10}|{x:5s}|{x:20s}|{x:s}|{value:+.2f}|".format(x=x, value=1.25))
print(f"|{x:^7s}|{-1.25:+.2f}|{42:>5d}|{1.5:>5}|")
print("{:.4g} {:+.1} {:+.4} {:+.6}".format(12.34567, 1.25, 1.25, 1.25))
"#;
    assert_eq!(
        run(source),
        (
            0,
            "|       cat|cat  |cat                 |cat|+1.25|\n|  cat  |-1.25|   42|  1.5|\n12.35 +1e+00 +1.25 +1.25\n".into(),
            String::new()
        )
    );
}

#[test]
fn general_float_format_switches_at_python_exponent_thresholds() {
    let source = r#"
for value in (0.0, 1.0, 12.34567, 12345.67, 0.000012345):
    print("{:.4g} {:+.4}".format(value, value))
"#;
    assert_eq!(
        run(source),
        (
            0,
            "0 +0.0\n1 +1.0\n12.35 +12.35\n1.235e+04 +1.235e+04\n1.234e-05 +1.234e-05\n".into(),
            String::new()
        )
    );
}

#[test]
fn percentage_formatting_accepts_precision_sign_and_alignment() {
    let source = r#"
print("|{:.1%}|{:+.2%}|{:>10.1%}|".format(0.125, 0.125, -0.125))
print(f"{0.5:.0%} {float('inf'):.1%} {float('nan'):+.2%}")
"#;
    assert_eq!(
        run(source),
        (
            0,
            "|12.5%|+12.50%|    -12.5%|\n50% inf% +nan%\n".into(),
            String::new()
        )
    );
}

#[test]
fn imaginary_literals_and_fractional_negative_powers_produce_complex_values() {
    let source = r#"
print(1j, 2j, 1 + 1j, 1j * 1j)
values = [1j, (2j), 1 + 3j]
print(values[0], values[1], values[2])
value = (-1) ** 0.5
print(abs(value.real) < 1e-12, round(value.imag, 6))
"#;
    assert_eq!(
        run(source),
        (
            0,
            "1j 2j (1+1j) (-1+0j)\n1j 2j (1+3j)\nTrue 1.0\n".into(),
            String::new()
        )
    );
}

#[test]
fn exec_runs_source_in_the_simulated_namespace() {
    let source = r#"
exec("answer = 6 * 7")
print(answer)
print(exec("answer += 1"), answer)
try:
    exec(42)
except TypeError:
    print("TypeError")
"#;
    assert_eq!(
        run(source),
        (0, "42\nNone 43\nTypeError\n".into(), String::new())
    );
}

#[test]
fn exec_rejects_unimplemented_syntax_without_host_fallback() {
    let (status, stdout, stderr) = run("exec('match 1:\\n    case 1: pass')");
    assert_eq!(status, 2);
    assert!(stdout.is_empty());
    assert!(stderr.contains("unsupported by minimal shim"));
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

#[test]
fn import_builtin_and_grouping_execute() {
    let source = r#"
math = __import__('math')
assert math.sqrt(4) == 2
import numpy.random
assert __import__('numpy.random') is __import__('numpy')
assert __import__('numpy.random', fromlist=['RandomState']) is numpy.random
print(f'{1234567:,.0f}', f'{-1234567:,.2f}', f'{1234567:,d}')
print(f'{12345.0:,.0}', f'{1234.5:>14,.2f}')
try:
    assert False
except AssertionError as error:
    assert str(error) == ''
print('ok')
"#;
    assert_eq!(
        run(source),
        (
            0,
            "1,234,567 -1,234,567.00 1,234,567\n1e+04       1,234.50\nok\n".into(),
            String::new()
        )
    );
}

#[test]
fn bare_assertion_error_is_not_duplicated() {
    let (status, _, error) = run("assert False");
    assert_ne!(status, 0);
    assert!(error.contains("AssertionError"));
    assert!(!error.contains("AssertionError: AssertionError"));
}

#[test]
fn import_builtin_rejects_relative_and_unavailable_imports() {
    for (source, expected) in [
        ("__import__('math', level=1)", "relative __import__"),
        (
            "__import__('missing_shellsim_module')",
            "missing_shellsim_module",
        ),
        ("__import__(42)", "str"),
    ] {
        let (status, _, error) = run(source);
        assert_ne!(status, 0);
        assert!(error.contains(expected), "{error}");
    }
}
