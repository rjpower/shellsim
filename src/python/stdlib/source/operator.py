"""Standard operators as functions, following CPython 3.14's ``operator`` module.

Every function delegates to the ordinary operator syntax, so user-defined dunder methods and
builtin type rules apply exactly as they do in expressions.
"""


def lt(a, b):
    return a < b


def le(a, b):
    return a <= b


def eq(a, b):
    return a == b


def ne(a, b):
    return a != b


def ge(a, b):
    return a >= b


def gt(a, b):
    return a > b


def not_(a):
    return not a


def truth(a):
    return bool(a)


def is_(a, b):
    return a is b


def is_not(a, b):
    return a is not b


def is_none(a):
    return a is None


def is_not_none(a):
    return a is not None


# Captured before this module rebinds the names to its own functions.
_builtin_abs = abs


def abs(a):  # noqa: A001
    return _builtin_abs(a)


def add(a, b):
    return a + b


def and_(a, b):
    return a & b


def floordiv(a, b):
    return a // b


def index(a):
    if isinstance(a, int):
        return int(a)
    method = getattr(type(a), "__index__", None)
    if method is None:
        raise TypeError(f"'{type(a).__name__}' object cannot be interpreted as an integer")
    return method(a)


def inv(a):
    return ~a


invert = inv


def lshift(a, b):
    return a << b


def mod(a, b):
    return a % b


def mul(a, b):
    return a * b


def matmul(a, b):
    return a @ b


def neg(a):
    return -a


def or_(a, b):
    return a | b


def pos(a):
    return +a


def pow(a, b):  # noqa: A001
    return a**b


def rshift(a, b):
    return a >> b


def sub(a, b):
    return a - b


def truediv(a, b):
    return a / b


def xor(a, b):
    return a ^ b


def _is_sequence(value):
    return isinstance(value, (str, bytes, bytearray, list, tuple)) or hasattr(value, "__getitem__")


def concat(a, b):
    if not _is_sequence(a):
        raise TypeError(f"'{type(a).__name__}' object can't be concatenated")
    return a + b


def contains(a, b):
    return b in a


def countOf(a, b):  # noqa: N802
    count = 0
    for item in a:
        if item is b or item == b:
            count += 1
    return count


def delitem(a, b):
    del a[b]


def getitem(a, b):
    return a[b]


def indexOf(a, b):  # noqa: N802
    for position, item in enumerate(a):
        if item is b or item == b:
            return position
    raise ValueError("sequence.index(x): x not in sequence")


def setitem(a, b, c):
    a[b] = c


def length_hint(obj, default=0):
    try:
        return len(obj)
    except TypeError:
        return default


def call(obj, /, *args, **kwargs):
    return obj(*args, **kwargs)


def iadd(a, b):
    a += b
    return a


def iand(a, b):
    a &= b
    return a


def iconcat(a, b):
    if not _is_sequence(a):
        raise TypeError(f"'{type(a).__name__}' object can't be concatenated")
    a += b
    return a


def ifloordiv(a, b):
    a //= b
    return a


def ilshift(a, b):
    a <<= b
    return a


def imod(a, b):
    a %= b
    return a


def imul(a, b):
    a *= b
    return a


def imatmul(a, b):
    a @= b
    return a


def ior(a, b):
    a |= b
    return a


def ipow(a, b):
    a **= b
    return a


def irshift(a, b):
    a >>= b
    return a


def isub(a, b):
    a -= b
    return a


def itruediv(a, b):
    a /= b
    return a


def ixor(a, b):
    a ^= b
    return a


class attrgetter:
    """Return a callable that fetches the given dotted attribute paths from its operand."""

    def __init__(self, attr, *attrs):
        names = (attr, *attrs)
        for name in names:
            if not isinstance(name, str):
                raise TypeError("attribute name must be a string")
        self._paths = tuple(tuple(name.split(".")) for name in names)
        self._names = names

    def __call__(self, obj):
        values = []
        for path in self._paths:
            value = obj
            for part in path:
                value = getattr(value, part)
            values.append(value)
        if len(values) == 1:
            return values[0]
        return tuple(values)

    def __repr__(self):
        return f"operator.attrgetter({', '.join(repr(name) for name in self._names)})"


class itemgetter:
    """Return a callable that fetches the given items from its operand."""

    def __init__(self, item, *items):
        self._items = (item, *items)

    def __call__(self, obj):
        if len(self._items) == 1:
            return obj[self._items[0]]
        return tuple(obj[item] for item in self._items)

    def __repr__(self):
        return f"operator.itemgetter({', '.join(repr(item) for item in self._items)})"


class methodcaller:
    """Return a callable that calls the named method on its operand."""

    def __init__(self, name, /, *args, **kwargs):
        if not isinstance(name, str):
            raise TypeError("method name must be a string")
        self._name = name
        self._args = args
        self._kwargs = kwargs

    def __call__(self, obj):
        return getattr(obj, self._name)(*self._args, **self._kwargs)

    def __repr__(self):
        arguments = [repr(self._name)]
        arguments.extend(repr(argument) for argument in self._args)
        arguments.extend(f"{key}={value!r}" for key, value in self._kwargs.items())
        return f"operator.methodcaller({', '.join(arguments)})"


__lt__ = lt
__le__ = le
__eq__ = eq
__ne__ = ne
__ge__ = ge
__gt__ = gt
__not__ = not_
__abs__ = abs
__add__ = add
__and__ = and_
__floordiv__ = floordiv
__index__ = index
__inv__ = inv
__invert__ = invert
__lshift__ = lshift
__mod__ = mod
__mul__ = mul
__matmul__ = matmul
__neg__ = neg
__or__ = or_
__pos__ = pos
__pow__ = pow
__rshift__ = rshift
__sub__ = sub
__truediv__ = truediv
__xor__ = xor
__concat__ = concat
__contains__ = contains
__delitem__ = delitem
__getitem__ = getitem
__setitem__ = setitem
__call__ = call
__iadd__ = iadd
__iand__ = iand
__iconcat__ = iconcat
__ifloordiv__ = ifloordiv
__ilshift__ = ilshift
__imod__ = imod
__imul__ = imul
__imatmul__ = imatmul
__ior__ = ior
__ipow__ = ipow
__irshift__ = irshift
__isub__ = isub
__itruediv__ = itruediv
__ixor__ = ixor
