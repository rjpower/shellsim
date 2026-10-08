"""Exercise the actual static NumPy port when its verified bundle is supplied.

These tests run upstream NumPy inside shellsim's WASI guest. They cover numerical
behavior and deliberate build frontiers rather than comparing random streams.
"""

import os
from pathlib import Path

import pytest
import shellsim


@pytest.fixture
def numpy_runtime():
    bundle = os.environ.get("SHELLSIM_NUMPY_BUNDLE")
    if bundle is None:
        pytest.skip("set SHELLSIM_NUMPY_BUNDLE to a built NumPy WASI profile")
    runtime = shellsim.CPythonRuntime(Path(bundle))
    environment = shellsim.Environment(cpu=5_000_000_000, memory=512 * 1024 * 1024, disk=128 * 1024 * 1024)
    runtime.mount(environment)
    return runtime, environment


def test_numpy_array_reduction_matrix_and_linalg(numpy_runtime):
    runtime, environment = numpy_runtime
    result = runtime.run(
        environment,
        [
            "-c",
            """
import numpy as np
assert np.__version__ == '2.3.5'
a = np.array([[1, 2], [3, 4]], dtype=np.int64)
assert a.sum() == 10
assert (a @ a).tolist() == [[7, 10], [15, 22]]
assert np.allclose(np.linalg.solve(a, np.array([5., 11.])), [1., 2.])
assert np.dtype(np.intp).itemsize == 4
assert np.dtype(np.longdouble).itemsize == 16
assert np.finfo(np.longdouble).nmant == 112
try:
    a @ np.zeros((3, 2))
except ValueError:
    pass
else:
    raise AssertionError('incompatible matrix dimensions accepted')
print('numpy numerical operations passed')
""",
        ],
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == b"numpy numerical operations passed\n"


def test_numpy_unique(numpy_runtime):
    runtime, environment = numpy_runtime
    result = runtime.run(
        environment,
        [
            "-c",
            """
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
    raise AssertionError('invalid unique axis accepted')
print('unique sort fallback passed')
""",
        ],
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == b"unique sort fallback passed\n"


def test_numpy_random_legacy_and_modern_abis(numpy_runtime):
    runtime, environment = numpy_runtime
    result = runtime.run(
        environment,
        [
            "-c",
            """
import numpy as np
modern = np.random.default_rng(42)
large = modern.poisson(2**33, 8)
assert large.dtype == np.dtype(np.int64)
assert (large > 2**32).all()
assert (large < 2**34).all()
assert modern.integers(2**40, 2**41, size=8).min() >= 2**40
legacy = np.random.RandomState(42)
assert legacy.poisson(20, 8).shape == (8,)
assert (legacy.binomial(100, .5, size=8) <= 100).all()
assert (legacy.randint(0, 100, size=8) < 100).all()
print('random integer ABIs passed')
""",
        ],
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == b"random integer ABIs passed\n"


def test_numpy_fft_round_trip_and_invalid_length(numpy_runtime):
    runtime, environment = numpy_runtime
    result = runtime.run(
        environment,
        [
            "-c",
            """
import numpy as np
assert np.fft._pocketfft_umath.__spec__.origin == 'built-in'
values = np.array([1., 2., 3., 4.])
assert np.allclose(np.fft.fft(values), [10, -2+2j, -2, -2-2j])
assert np.allclose(np.fft.ifft(np.fft.fft(values)).real, values)
assert np.allclose(np.fft.irfft(np.fft.rfft(values)), values)
try:
    np.fft.fft(values, n=0)
except ValueError:
    pass
else:
    raise AssertionError('zero FFT length accepted')
print('FFT round trip and invalid length passed')
""",
        ],
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == b"FFT round trip and invalid length passed\n"


def test_numpy_floating_point_policy_frontier(numpy_runtime):
    runtime, environment = numpy_runtime
    port = next(port for port in runtime.manifest["native_ports"] if port["name"] == "numpy")
    assert port["features"]["floating_point_exceptions"] is False
    result = runtime.run(
        environment,
        [
            "-c",
            """
import numpy as np
np.seterr(all='raise')
# This pinned experimental profile cannot detect the hardware status flags.
assert np.isinf(np.divide(1., 0.))
print('floating-point policy gap reproduced')
""",
        ],
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == b"floating-point policy gap reproduced\n"
