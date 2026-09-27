"""Python 3.14's keywords and soft keywords."""

__all__ = ["iskeyword", "issoftkeyword", "kwlist", "softkwlist"]

kwlist = [
    "False",
    "None",
    "True",
    "and",
    "as",
    "assert",
    "async",
    "await",
    "break",
    "class",
    "continue",
    "def",
    "del",
    "elif",
    "else",
    "except",
    "finally",
    "for",
    "from",
    "global",
    "if",
    "import",
    "in",
    "is",
    "lambda",
    "nonlocal",
    "not",
    "or",
    "pass",
    "raise",
    "return",
    "try",
    "while",
    "with",
    "yield",
]

softkwlist = ["_", "case", "match", "type"]

_keywords = frozenset(kwlist)
_soft_keywords = frozenset(softkwlist)


def iskeyword(s):
    return s in _keywords


def issoftkeyword(s):
    return s in _soft_keywords
