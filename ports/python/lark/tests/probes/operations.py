"""Exercise a bounded public API inside the actual guest."""

from lark import Lark, UnexpectedInput

parser = Lark('start: INT "+" INT\n%import common.INT', parser="lalr")
assert [str(x) for x in parser.parse("2+3").children] == ["2", "3"]
try:
    parser.parse("bad")
except UnexpectedInput:
    pass
else:
    raise AssertionError("invalid input accepted")
print("lark: public API probe passed")
