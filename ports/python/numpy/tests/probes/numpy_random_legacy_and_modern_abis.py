import numpy as np

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
print("random integer ABIs passed")
