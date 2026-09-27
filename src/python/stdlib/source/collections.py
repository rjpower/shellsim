"""Common container types implemented over ordinary Python protocols."""

import sys
from keyword import iskeyword as _iskeyword

from _collections import defaultdict


class Counter:
    def __init__(self, iterable=None):
        self._counts = {}
        if iterable is not None:
            self.update(iterable)

    def __getitem__(self, key):
        return self._counts.get(key, 0)

    def __setitem__(self, key, value):
        self._counts[key] = value

    def __delitem__(self, key):
        del self._counts[key]

    def __contains__(self, key):
        return key in self._counts

    def __iter__(self):
        return iter(self._counts.keys())

    def __len__(self):
        return len(self._counts)

    def __repr__(self):
        return "Counter(" + repr(self._counts) + ")"

    def update(self, iterable=None):
        if iterable is None:
            return
        if isinstance(iterable, dict):
            for key, value in iterable.items():
                self._counts[key] = self[key] + value
        else:
            for key in iterable:
                self._counts[key] = self[key] + 1

    def subtract(self, iterable=None):
        if iterable is None:
            return
        if isinstance(iterable, dict):
            for key, value in iterable.items():
                self._counts[key] = self[key] - value
        else:
            for key in iterable:
                self._counts[key] = self[key] - 1

    def elements(self):
        result = []
        for key, count in self._counts.items():
            if count > 0:
                result.extend([key] * count)
        return result

    def most_common(self, n=None):
        values = sorted(self._counts.items(), key=lambda item: item[1], reverse=True)
        if n is None:
            return values
        return values[:n]

    def total(self):
        return sum(self._counts.values())

    def keys(self):
        return self._counts.keys()

    def values(self):
        return self._counts.values()

    def items(self):
        return self._counts.items()

    def clear(self):
        self._counts = {}

    def copy(self):
        return Counter(self._counts)

    def __add__(self, other):
        result = Counter()
        for key in self:
            value = self[key] + other[key]
            if value > 0:
                result[key] = value
        for key in other:
            if key not in self and other[key] > 0:
                result[key] = other[key]
        return result

    def __sub__(self, other):
        result = Counter()
        for key in self:
            value = self[key] - other[key]
            if value > 0:
                result[key] = value
        return result


class deque:
    def __init__(self, iterable=None, maxlen=None):
        self._items = []
        self.maxlen = maxlen
        if iterable is not None:
            self.extend(iterable)

    def __len__(self):
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    def __getitem__(self, index):
        return self._items[index]

    def __setitem__(self, index, value):
        self._items[index] = value

    def __repr__(self):
        return "deque(" + repr(self._items) + ")"

    def _trim_right(self):
        if self.maxlen is not None:
            while len(self._items) > self.maxlen:
                self._items.pop()

    def _trim_left(self):
        if self.maxlen is not None:
            while len(self._items) > self.maxlen:
                self._items.pop(0)

    def append(self, value):
        self._items.append(value)
        self._trim_left()

    def appendleft(self, value):
        self._items = [value] + self._items
        self._trim_right()

    def extend(self, iterable):
        for value in iterable:
            self.append(value)

    def extendleft(self, iterable):
        for value in iterable:
            self.appendleft(value)

    def pop(self):
        return self._items.pop()

    def popleft(self):
        return self._items.pop(0)

    def clear(self):
        self._items = []

    def copy(self):
        return deque(self._items, self.maxlen)

    def count(self, value):
        return self._items.count(value)

    def remove(self, value):
        self._items.remove(value)

    def reverse(self):
        self._items.reverse()

    def rotate(self, amount=1):
        if len(self._items) == 0:
            return
        if amount > 0:
            for _ in range(amount % len(self._items)):
                self.appendleft(self._items.pop())
        else:
            for _ in range((-amount) % len(self._items)):
                self.append(self._items.pop(0))


class _tuplegetter:
    """The read-only attribute for one namedtuple field.

    It reads through subscription, so a subclass that overrides ``__getitem__`` also changes
    what its fields return, which CPython's C descriptor does not do.
    """

    def __init__(self, index, doc):
        self._index = index
        self.__doc__ = doc

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        return instance[self._index]

    def __set__(self, instance, value):
        raise AttributeError("can't set attribute")

    def __delete__(self, instance):
        raise AttributeError("can't delete attribute")

    def __repr__(self):
        return f"_tuplegetter({self._index}, {self.__doc__!r})"


def _quoted_names(names):
    """``'a'``, ``'a' and 'b'`` or ``'a', 'b', and 'c'``, as CPython lists missing arguments."""

    quoted = [repr(name) for name in names]
    if len(quoted) <= 2:
        return " and ".join(quoted)
    return ", ".join(quoted[:-1]) + ", and " + quoted[-1]


def _bind_fields(typename, fields, defaults, args, kwargs):
    """The field values for one ``__new__`` call, with CPython's argument errors."""

    count = len(fields)
    values = dict(zip(fields, args))
    for name, value in kwargs.items():
        if name not in fields:
            raise TypeError(f"{typename}.__new__() got an unexpected keyword argument {name!r}")
        if name in values:
            raise TypeError(f"{typename}.__new__() got multiple values for argument {name!r}")
        values[name] = value
    if len(args) > count:
        required = count - len(defaults)
        if defaults:
            takes = f"from {required + 1} to {count + 1} positional arguments"
        elif count == 0:
            takes = "1 positional argument"
        else:
            takes = f"{count + 1} positional arguments"
        raise TypeError(f"{typename}.__new__() takes {takes} but {len(args) + 1} were given")
    missing = [name for name in fields if name not in values and name not in defaults]
    if missing:
        noun = "argument" if len(missing) == 1 else "arguments"
        raise TypeError(
            f"{typename}.__new__() missing {len(missing)} required positional {noun}: "
            + _quoted_names(missing)
        )
    return [values[name] if name in values else defaults[name] for name in fields]


def _field_names(typename, field_names, rename):
    """Validated field names, with invalid ones replaced by ``_index`` when ``rename``."""

    if isinstance(field_names, str):
        field_names = field_names.replace(",", " ").split()
    names = [str(name) for name in field_names]
    if rename:
        seen = set()
        for index, name in enumerate(names):
            if not name.isidentifier() or _iskeyword(name) or name.startswith("_") or name in seen:
                names[index] = f"_{index}"
            seen.add(name)
    for name in [typename] + names:
        if not name.isidentifier():
            raise ValueError(f"Type names and field names must be valid identifiers: {name!r}")
        if _iskeyword(name):
            raise ValueError(f"Type names and field names cannot be a keyword: {name!r}")
    seen = set()
    for name in names:
        if name.startswith("_") and not rename:
            raise ValueError(f"Field names cannot start with an underscore: {name!r}")
        if name in seen:
            raise ValueError(f"Encountered duplicate field name: {name!r}")
        seen.add(name)
    return tuple(names)


def namedtuple(typename, field_names, *, rename=False, defaults=None, module=None):
    """A ``tuple`` subclass named ``typename`` whose items are also named attributes."""

    typename = str(typename)
    fields = _field_names(typename, field_names, rename)
    field_defaults = {}
    if defaults is not None:
        defaults = tuple(defaults)
        if len(defaults) > len(fields):
            raise TypeError("Got more default values than field names")
        field_defaults = dict(zip(fields[len(fields) - len(defaults) :], defaults))
    if module is None:
        module = sys._getframemodulename(1) or "__main__"

    def __new__(_cls, *args, **kwargs):
        values = _bind_fields(typename, fields, field_defaults, args, kwargs)
        return tuple.__new__(_cls, values)

    def _make(cls, iterable):
        result = tuple.__new__(cls, iterable)
        if len(result) != len(fields):
            raise TypeError(f"Expected {len(fields)} arguments, got {len(result)}")
        return result

    def _replace(self, /, **changes):
        result = self._make([changes.pop(name, value) for name, value in zip(fields, self)])
        if changes:
            raise TypeError(f"Got unexpected field names: {list(changes)!r}")
        return result

    def __repr__(self):
        items = ", ".join(f"{name}={value!r}" for name, value in zip(fields, self))
        return f"{self.__class__.__name__}({items})"

    def _asdict(self):
        return dict(zip(fields, self))

    def __getnewargs__(self):
        return tuple(self)

    namespace = {
        # The field tuple's repr without quotes, so one field reads `P(x,)`.
        "__doc__": typename + repr(fields).replace("'", ""),
        "__module__": module,
        "__slots__": (),
        "_fields": fields,
        "_field_defaults": field_defaults,
        "__new__": __new__,
        "_make": classmethod(_make),
        "_replace": _replace,
        "__replace__": _replace,
        "__repr__": __repr__,
        "_asdict": _asdict,
        "__getnewargs__": __getnewargs__,
        "__match_args__": fields,
    }
    for index, name in enumerate(fields):
        namespace[name] = _tuplegetter(index, f"Alias for field number {index}")
    return type(typename, (tuple,), namespace)


_NAMED_TUPLE_PROHIBITED = frozenset(
    {
        "__new__",
        "__init__",
        "__slots__",
        "__getnewargs__",
        "_fields",
        "_field_defaults",
        "_make",
        "_replace",
        "_asdict",
        "_source",
    }
)
_NAMED_TUPLE_SPECIAL = frozenset({"__module__", "__name__", "__qualname__", "__annotations__"})


def _namedtuple_from_pairs(typename, fields, module):
    """The class ``typing.NamedTuple(typename, fields)`` builds from ``(name, type)`` pairs."""

    return namedtuple(typename, [name for name, _ in fields], module=module)


def _namedtuple_from_class(typename, fields, namespace, module):
    """The class ``class typename(typing.NamedTuple)`` defines in ``module``.

    ``fields`` lists the class body's annotated names in order. A field the body also assigns
    takes that value as its default; the body's other attributes are copied onto the class.
    """

    defaults = {}
    for name in fields:
        if name in namespace:
            defaults[name] = namespace[name]
        elif defaults:
            plural = "s" if len(defaults) > 1 else ""
            raise TypeError(
                f"Non-default namedtuple field {name} cannot follow default field{plural} "
                + ", ".join(defaults)
            )
    result = namedtuple(typename, fields, defaults=list(defaults.values()), module=module)
    for key, value in namespace.items():
        if key in _NAMED_TUPLE_PROHIBITED:
            raise AttributeError("Cannot overwrite NamedTuple attribute " + key)
        if key not in _NAMED_TUPLE_SPECIAL and key not in fields:
            setattr(result, key, value)
    return result
