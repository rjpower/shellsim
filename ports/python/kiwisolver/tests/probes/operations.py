import kiwisolver as k

assert k.__version__ == "1.5.1"
x, y = k.Variable("x"), k.Variable("y")
solver = k.Solver()
constraint = x + y == 100
solver.addConstraint(constraint)
solver.addConstraint(x == 40)
solver.updateVariables()
assert x.value() == 40 and y.value() == 60
assert x.name() == "x" and x.context() is None
term = k.Term(y, 2)
assert term.variable() is y and term.coefficient() == 2 and term.value() == 120
expression = constraint.expression()
assert len(expression.terms()) == 2 and expression.constant() == -100 and expression.value() == 0
assert constraint.op() == "==" and constraint.strength() == k.strength.required and not constraint.violated()
assert 0 < k.strength.weak < k.strength.medium < k.strength.strong < k.strength.required


def raises(kind, action):
    try:
        action()
    except kind:
        return
    raise AssertionError(kind.__name__)


raises(k.UnsatisfiableConstraint, lambda: solver.addConstraint(x == 41))
raises(k.DuplicateConstraint, lambda: solver.addConstraint(constraint))
raises(k.UnknownConstraint, lambda: solver.removeConstraint(y == 999))
raises(TypeError, lambda: solver.addConstraint(None))
assert isinstance(solver.dumps(), str)
solver.dump()
solver.reset()
solver.addEditVariable(x, k.strength.strong)
solver.suggestValue(x, 17)
solver.updateVariables()
assert x.value() == 17
print("Kiwi constraints and C++ exception translation passed")
