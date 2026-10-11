"""Exercise a bounded public API in the actual guest."""

from pycparser import CParser

tree = CParser().parse("int add(int a, int b) { return a + b; }")
assert tree.ext[0].decl.name == "add"
assert tree.ext[0].body.block_items[0].expr.op == "+"
print("pycparser: guest public API probe passed")
