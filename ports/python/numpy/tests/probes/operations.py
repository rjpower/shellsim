import numpy as np

assert np.__version__ == "2.3.5"
assert np._core._multiarray_umath.__spec__.origin.endswith(".so")
assert "numpy._core._multiarray_umath" not in __import__("sys").builtin_module_names
a = np.array([[1, 2], [3, 4]], dtype=np.int64)
assert (a + 2).tolist() == [[3, 4], [5, 6]]
assert a.sum() == 10
assert (a @ a).tolist() == [[7, 10], [15, 22]]
assert np.allclose(np.linalg.solve(a, [5.0, 11.0]), [1.0, 2.0])
assert np.allclose(np.linalg.solve(a.astype(complex), [5.0 + 0j, 11.0 + 0j]), [1.0, 2.0])
assert np.dtype(np.intp).itemsize == 4
assert np.dtype(np.longdouble).itemsize == 16
assert np.finfo(np.longdouble).nmant == 112
values = np.array([1.0, 2.0, 3.0, 4.0])
assert np.allclose(np.fft.fft(values), [10, -2 + 2j, -2, -2 - 2j])
assert np.allclose(np.fft.ifft(np.fft.fft(values)).real, values)
assert np.allclose(np.fft.irfft(np.fft.rfft(values)), values)
assert np.unique([3, 1, 3, 2]).tolist() == [1, 2, 3]
modern = np.random.default_rng(42)
large = modern.poisson(2**33, 8)
assert large.dtype == np.dtype(np.int64)
assert (large > 2**32).all()
assert (large < 2**34).all()
assert modern.integers(2**40, 2**41, size=8).min() >= 2**40
legacy = np.random.RandomState(42)
assert legacy.poisson(20, 8).shape == (8,)
assert (legacy.binomial(100, 0.5, size=8) <= 100).all()
assert (legacy.randint(0, 100, size=8) < 100).all()
for operation in (lambda: a @ np.zeros((3, 2)), lambda: np.fft.fft(values, n=0)):
    try:
        operation()
    except ValueError:
        pass
    else:
        raise AssertionError("invalid numerical input accepted")
print("independent NumPy array/linalg/FFT/RNG passed")
