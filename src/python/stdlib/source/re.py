"""Regular expressions over the ``_re`` native matcher.

Patterns are compiled and matched natively; this module supplies the public constants, the
``error`` type a bad pattern raises, and the helpers CPython implements on top of the matcher.
"""

import _re

__all__ = [
    "match", "fullmatch", "search", "sub", "subn", "split", "findall", "finditer", "compile",
    "purge", "escape", "error", "PatternError", "Pattern", "Match", "A", "I", "L", "M", "S",
    "X", "U", "ASCII", "IGNORECASE", "LOCALE", "MULTILINE", "DOTALL", "VERBOSE", "UNICODE",
    "NOFLAG", "RegexFlag",
]

class RegexFlag(int):
    """An `re` flag: an int whose repr names the flags it combines, as CPython's IntFlag does."""

    _names = {}

    def __new__(cls, value, name=None):
        self = int.__new__(cls, value)
        if name is not None:
            RegexFlag._names[int(value)] = name
        return self

    def _combine(self, value):
        return RegexFlag(value)

    def __or__(self, other):
        return self._combine(int(self) | int(other))

    __ror__ = __or__

    def __and__(self, other):
        return self._combine(int(self) & int(other))

    __rand__ = __and__

    def __xor__(self, other):
        return self._combine(int(self) ^ int(other))

    __rxor__ = __xor__

    def __invert__(self):
        return self._combine(~int(self) & 0x1FF)

    def __repr__(self):
        value = int(self)
        if value == 0:
            return "re.NOFLAG"
        names = []
        remainder = value
        for bit in sorted(RegexFlag._names):
            if bit and value & bit == bit:
                names.append("re." + RegexFlag._names[bit])
                remainder &= ~bit
        if remainder:
            names.append(hex(remainder))
        return "|".join(names)

    __str__ = __repr__


NOFLAG = RegexFlag(0, "NOFLAG")
TEMPLATE = T = RegexFlag(1, "TEMPLATE")
IGNORECASE = I = RegexFlag(2, "IGNORECASE")
LOCALE = L = RegexFlag(4, "LOCALE")
MULTILINE = M = RegexFlag(8, "MULTILINE")
DOTALL = S = RegexFlag(16, "DOTALL")
UNICODE = U = RegexFlag(32, "UNICODE")
VERBOSE = X = RegexFlag(64, "VERBOSE")
DEBUG = RegexFlag(128, "DEBUG")
ASCII = A = RegexFlag(256, "ASCII")
for _name in ("NOFLAG", "TEMPLATE", "T", "IGNORECASE", "I", "LOCALE", "L", "MULTILINE", "M",
              "DOTALL", "S", "UNICODE", "U", "VERBOSE", "X", "DEBUG", "ASCII", "A"):
    setattr(RegexFlag, _name, globals()[_name])
del _name


class PatternError(Exception):
    """Exception raised for invalid regular expressions."""

    def __init__(self, msg, pattern=None, pos=None):
        Exception.__init__(self, msg)
        self.msg = msg
        self.pattern = pattern
        self.pos = pos
        self.lineno = None
        self.colno = None


error = PatternError


def compile(pattern, flags=0):
    if isinstance(pattern, Pattern):
        if flags:
            raise ValueError("cannot process flags argument with a compiled pattern")
        return pattern
    try:
        return _re.compile(pattern, flags)
    except ValueError as exc:
        raise PatternError(str(exc), pattern) from None


Pattern = type(_re.compile("x", 0))
Match = type(_re.compile("x", 0).match("x"))


def purge():
    return None


def escape(pattern):
    return _re.escape(pattern)


def match(pattern, string, flags=0):
    return compile(pattern, flags).match(string)


def fullmatch(pattern, string, flags=0):
    return compile(pattern, flags).fullmatch(string)


def search(pattern, string, flags=0):
    return compile(pattern, flags).search(string)


def findall(pattern, string, flags=0):
    return compile(pattern, flags).findall(string)


def finditer(pattern, string, flags=0):
    return compile(pattern, flags).finditer(string)


def sub(pattern, repl, string, count=0, flags=0):
    return compile(pattern, flags).sub(repl, string, count)


def subn(pattern, repl, string, count=0, flags=0):
    return compile(pattern, flags).subn(repl, string, count)


def split(pattern, string, maxsplit=0, flags=0):
    return compile(pattern, flags).split(string, maxsplit)
