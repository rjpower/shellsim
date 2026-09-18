"""Deterministic NumPy random facade with explicit, capability-free state."""

import math
import numpy as np


class Generator:
    def __init__(self, seed=None):
        if seed is None:
            seed = 0
        if not isinstance(seed, int):
            raise TypeError("seed must be an integer or None")
        self._state = seed % 4294967296

    def _next(self):
        self._state = (1664525 * self._state + 1013904223) % 4294967296
        return self._state

    def _shape(self, size):
        if size is None:
            return None, 1
        if isinstance(size, int):
            shape = (size,)
        else:
            shape = tuple(size)
        count = 1
        for dimension in shape:
            if dimension < 0:
                raise ValueError("negative dimensions are not allowed")
            count *= dimension
        return shape, count

    def random(self, size=None, dtype=np.float64, out=None):
        if out is not None:
            raise ValueError("out is not supported by shellsim numpy.random")
        if dtype is not np.float32 and dtype is not np.float64:
            raise TypeError("random supports only float32 and float64")
        shape, count = self._shape(size)
        values = [self._next() / 4294967296 for _ in range(count)]
        if shape is None:
            return dtype(values[0])
        return np.array(values, dtype=dtype).reshape(shape)

    def standard_normal(self, size=None, dtype=np.float64, out=None):
        if out is not None:
            raise ValueError("out is not supported by shellsim numpy.random")
        if dtype is not np.float32 and dtype is not np.float64:
            raise TypeError("standard_normal supports only float32 and float64")
        shape, count = self._shape(size)
        values = []
        while len(values) < count:
            first = (self._next() + 1) / 4294967297
            second = self._next() / 4294967296
            radius = math.sqrt(-2.0 * math.log(first))
            values.append(radius * math.cos(2.0 * math.pi * second))
            if len(values) < count:
                values.append(radius * math.sin(2.0 * math.pi * second))
        if shape is None:
            return dtype(values[0])
        return np.array(values, dtype=dtype).reshape(shape)

    def integers(self, low, high=None, size=None, dtype=np.int64, endpoint=False):
        if high is None:
            high = low
            low = 0
        if endpoint:
            high += 1
        if high <= low:
            raise ValueError("high must be greater than low")
        shape, count = self._shape(size)
        width = high - low
        values = [low + self._next() % width for _ in range(count)]
        if shape is None:
            return dtype(values[0])
        return np.array(values, dtype=dtype).reshape(shape)


def default_rng(seed=None):
    return Generator(seed)


_legacy = Generator(0)


def seed(value=None):
    global _legacy
    _legacy = Generator(value)


def random(size=None):
    return _legacy.random(size)


def normal(loc=0.0, scale=1.0, size=None):
    return loc + scale * _legacy.standard_normal(size)


def randint(low, high=None, size=None, dtype=np.int64):
    return _legacy.integers(low, high, size, dtype)
