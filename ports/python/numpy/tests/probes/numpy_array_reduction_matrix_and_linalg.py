import numpy as np

assert np.__version__ == "2.3.5"
a = np.array([[1, 2], [3, 4]], dtype=np.int64)
assert a.sum() == 10
assert (a @ a).tolist() == [[7, 10], [15, 22]]
assert np.allclose(np.linalg.solve(a, np.array([5.0, 11.0])), [1.0, 2.0])
assert np.dtype(np.intp).itemsize == 4
assert np.dtype(np.longdouble).itemsize == 16
assert np.finfo(np.longdouble).nmant == 112
try:
    a @ np.zeros((3, 2))
except ValueError:
    pass
else:
    raise AssertionError("incompatible matrix dimensions accepted")
print("numpy numerical operations passed")
