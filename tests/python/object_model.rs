//! Object-model compatibility checks requiring exact process output.

use shellsim::Environment;

use super::support::run_shell;

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
