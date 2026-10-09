import pycosat

assert pycosat.solve([[1], [-1]]) == "UNSAT"
assert pycosat.solve([[1]]) == [1]
try:
    pycosat.solve([[0]])
except ValueError:
    pass
else:
    raise AssertionError("zero literal must be rejected")
