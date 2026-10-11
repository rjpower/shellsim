"""Evaluate a bounded task graph with the synchronous scheduler."""

from dask import compute, delayed


@delayed
def double(value):
    return value * 2


assert compute(double(3) + double(4), scheduler="synchronous") == (14,)


@delayed
def fail():
    raise ValueError("expected")


try:
    compute(fail(), scheduler="synchronous")
except ValueError:
    pass
else:
    raise AssertionError("failed task accepted")
print("dask: synchronous task graph and failed task passed")
