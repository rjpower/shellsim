"""Run the independent NumPy wheel in an unchanged bare SDK 34 interpreter.

Trusted test setup stages the wheel directly; public package resolution has its
own tests. Every numerical operation executes inside the WASI guest.
"""

import hashlib
import json
import os
import zipfile
from pathlib import Path

import pytest
import shellsim


@pytest.fixture
def dynamic_numpy(tmp_path):
    bundle = os.environ.get("SHELLSIM_DYNAMIC_V2_ARTIFACTS")
    wheel_path = os.environ.get("SHELLSIM_NUMPY_DYNAMIC_WHEEL")
    if bundle is None or wheel_path is None:
        pytest.skip("set SDK 34 runtime and NumPy dynamic wheel artifact paths")
    runtime = shellsim.CPythonRuntime(Path(bundle))
    environment = shellsim.Environment(cpu=10_000_000_000, memory=1024 * 1024 * 1024, disk=128 * 1024 * 1024)
    runtime.mount(environment)
    with zipfile.ZipFile(wheel_path) as wheel:
        manifest = json.loads(wheel.read("numpy-2.3.5.dist-info/shellsim-native.json"))
        assert manifest["abi"] == "shellsim-wasi-sdk34-cpython3137-v2"
        assert len(manifest["artifacts"]) == 13
        for artifact in manifest["artifacts"]:
            assert hashlib.sha256(wheel.read(artifact["path"])).hexdigest() == artifact["sha256"]
        wheel.extractall(tmp_path)
    interpreter = environment.read_file("/usr/bin/python3.wasm")
    environment.mount(tmp_path, runtime.site_packages)
    return runtime, environment, interpreter


def test_numpy_dynamic_array_linalg_fft_and_rng(dynamic_numpy):
    runtime, environment, interpreter = dynamic_numpy
    result = runtime.run(
        environment,
        [
            "-c",
            """
import numpy as np
assert np.__version__ == '2.3.5'
assert np._core._multiarray_umath.__spec__.origin.endswith('.so')
assert 'numpy._core._multiarray_umath' not in __import__('sys').builtin_module_names
a = np.array([[1, 2], [3, 4]], dtype=np.int64)
assert (a + 2).tolist() == [[3, 4], [5, 6]]
assert a.sum() == 10
assert (a @ a).tolist() == [[7, 10], [15, 22]]
assert np.allclose(np.linalg.solve(a, [5., 11.]), [1., 2.])
assert np.allclose(np.linalg.solve(a.astype(complex), [5.+0j, 11.+0j]), [1., 2.])
assert np.dtype(np.intp).itemsize == 4
assert np.dtype(np.longdouble).itemsize == 16
assert np.finfo(np.longdouble).nmant == 112
values = np.array([1., 2., 3., 4.])
assert np.allclose(np.fft.fft(values), [10, -2+2j, -2, -2-2j])
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
assert (legacy.binomial(100, .5, size=8) <= 100).all()
assert (legacy.randint(0, 100, size=8) < 100).all()
for operation in (lambda: a @ np.zeros((3, 2)), lambda: np.fft.fft(values, n=0)):
    try:
        operation()
    except ValueError:
        pass
    else:
        raise AssertionError('invalid numerical input accepted')
print('independent NumPy array/linalg/FFT/RNG passed')
""",
        ],
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == b"independent NumPy array/linalg/FFT/RNG passed\n"
    assert environment.read_file("/usr/bin/python3.wasm") == interpreter
