import numpy as np

values, counts = np.unique([3, 1, 3, 2], return_counts=True)
assert values.tolist() == [1, 2, 3]
assert counts.tolist() == [1, 1, 2]
assert np.unique([]).tolist() == []
try:
    np.unique(np.array([[1, 2]]), axis=2)
except np.exceptions.AxisError:
    pass
else:
    raise AssertionError("invalid unique axis accepted")
print("unique sort fallback passed")
