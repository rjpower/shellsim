"""Parse fixed, small LaTeX expressions in the guest."""

from latex2sympy2_extended import latex2sympy
from sympy import Rational, simplify, symbols

assert latex2sympy(r"\frac{1}{2}") == Rational(1, 2)
x = symbols("x")
assert simplify(latex2sympy(r"x^2 + 2x + 1") - (x + 1) ** 2) == 0
print("latex2sympy2-extended: fixed rational and polynomial parsing passed")
