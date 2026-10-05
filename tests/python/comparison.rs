//! Rich comparison through the VM's single entry point: slot order, container ordering and
//! the fallbacks CPython defines when no slot answers.

use super::support::run_python_text as run;

#[test]
fn each_comparison_operator_calls_its_own_slot() {
    // `<=` asks `__le__`, never `__eq__` plus `__lt__`; `!=` without `__ne__` negates `__eq__`;
    // a strict subclass that overrides the reflected slot answers first.
    let source = r#"
class Base:
    def __init__(self, v): self.v = v
    def __eq__(self, o): return ("eq", self.v == o.v)[1]
    def __lt__(self, o): return ("lt", self.v < o.v)[1]
    def __le__(self, o): return "le-called"
    def __gt__(self, o): return "base-gt"

class Sub(Base):
    def __gt__(self, o): return "sub-gt"

print(Base(1) <= Base(2))
print(Base(1) != Base(1), Base(1) != Base(2))
print(Base(1) < Sub(2))
print(Base(1) < Base(2))
"#;
    assert_eq!(
        run(source),
        (
            0,
            "le-called\nFalse True\nsub-gt\nTrue\n".into(),
            String::new()
        )
    );
}

#[test]
fn sequences_order_by_the_first_unequal_item_then_by_length() {
    let source = r#"
class AlwaysEqual:
    def __init__(self, v): self.v = v
    def __eq__(self, o): return True
    def __lt__(self, o): return self.v < o.v

print([1, 2] < [1, 3], [1] < [1, 0], [1, 0] < [1], (1, 2) <= (1, 2))
print((AlwaysEqual(1),) < (AlwaysEqual(2),))
print((AlwaysEqual(5), 1) < (AlwaysEqual(2), 2))
try:
    [1] < (1,)
except TypeError as error:
    print(error)
try:
    [object()] < [object()]
except TypeError as error:
    print(error)
"#;
    assert_eq!(
        run(source),
        (
            0,
            "True True False True\nFalse\nTrue\n'<' not supported between instances of 'list' \
             and 'tuple'\n'<' not supported between instances of 'object' and 'object'\n"
                .into(),
            String::new()
        )
    );
}

#[test]
fn sorting_user_objects_keeps_handles_bounded_and_order_stable() {
    // Every `__lt__` call creates handles; a sort of many objects must release them as it goes.
    let source = r#"
class Item:
    def __init__(self, key, tag): self.key, self.tag = key, tag
    def __lt__(self, other): return self.key < other.key

items = [Item((i * 7919) % 97, i) for i in range(4000)]
items.sort()
print(all(a.key <= b.key for a, b in zip(items, items[1:])))
print(all(a.tag < b.tag for a, b in zip(items, items[1:]) if a.key == b.key))
print(sorted([3.0, 1.5, 1.0])[0], sorted(["b", "a", "c"], reverse=True))
"#;
    assert_eq!(
        run(source),
        (0, "True\nTrue\n1.0 ['c', 'b', 'a']\n".into(), String::new())
    );
}
