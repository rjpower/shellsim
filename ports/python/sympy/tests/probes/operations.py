"""Check fixed symbolic simplification and exact solving in the guest."""

from sympy import Rational, diff, expand, simplify, solve, symbols

x = symbols("x")
assert expand((x + 1) ** 2) == x**2 + 2 * x + 1
assert simplify((x**2 - 1) / (x - 1)) == x + 1
assert diff(x**3, x) == 3 * x**2
assert solve(3 * x - 2, x) == [Rational(2, 3)]
print("sympy: expansion, simplification, differentiation and solving passed")
