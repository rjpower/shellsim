import pycosat

assert pycosat.solve([[1], [-1]]) == "UNSAT"
assert pycosat.solve([[1]]) == [1]
