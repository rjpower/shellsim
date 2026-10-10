"""Exercise the installed upstream array, random, FFT and linear algebra modules."""

import threading

import numpy as np

assert np.__version__ == "2.3.5"
x = np.arange(6, dtype=np.float64).reshape(2, 3)
np.testing.assert_allclose(x @ x.T, [[5, 14], [14, 50]])
np.testing.assert_allclose(np.sin([0.0, np.pi / 2]), [0.0, 1.0], atol=1e-14)
np.testing.assert_allclose(np.linalg.solve([[3.0, 1.0], [1.0, 2.0]], [9.0, 8.0]), [2.0, 3.0])
np.testing.assert_allclose(np.fft.irfft(np.fft.rfft(np.arange(8.0))), np.arange(8.0), atol=1e-12)
rng = np.random.default_rng(19)
values = rng.normal(size=16)
assert values.shape == (16,) and np.isfinite(values).all()
assert np.dtype(np.intp).itemsize == 4
assert np.dtype(np.longdouble).itemsize == 16
assert np.finfo(np.longdouble).nmant == 112
try:
    np.arange(3).reshape(2, 2)
except ValueError:
    pass
else:
    raise AssertionError("invalid reshape accepted")
print("numpy arrays, linalg, FFT, random and target types passed")


barrier = threading.Barrier(3)
results, errors = {}, []


def calculate(index):
    try:
        barrier.wait()
        values = np.arange(1024, dtype=np.float64) + index
        results[index] = float(np.square(values).sum())
    except BaseException as error:
        errors.append(error)


workers = [threading.Thread(target=calculate, args=(index,)) for index in range(2)]
for worker in workers:
    worker.start()
barrier.wait()
for worker in workers:
    worker.join()
assert not errors and len(results) == 2
for index in range(2):
    assert results[index] == sum((value + index) ** 2 for value in range(1024))
print("numpy concurrent ufunc threads passed")
