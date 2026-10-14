"""Use the real guest to verify delivered bytecode, invalidation and reproducible packaging."""

import base64
import csv
import hashlib
import io
import json
import os
import zipfile
from pathlib import Path

import pytest

from ports._support.graph import plan
from ports._support.tests.test_runner import _wheel_port
from ports.buildomatic.bytecode import BytecodeCompiler
from ports.buildomatic.cache import delivery_key
from ports.buildomatic.delivery import DeliveryStore
from ports.buildomatic.delivery_products import publish_product, publish_runtime


@pytest.fixture
def runtime_bundle():
    path = os.environ.get("SHELLSIM_CPYTHON_BUNDLE")
    if path is None:
        pytest.skip("set SHELLSIM_CPYTHON_BUNDLE to a source-built CPython WASI bundle")
    return Path(path)


def test_delivered_wheel_loads_bytecode_and_invalidates_changed_source(tmp_path, runtime_bundle):
    from shellsim import Context
    from shellsim._pypi_catalog import unpack_wheel
    from shellsim.cpython import CPythonRuntime

    ports, root = tmp_path / "ports", tmp_path / "product"
    _wheel_port(ports, tmp_path / "sources", "example", [])
    port = plan(ports, ["python/example"]).ports[-1]
    (root / "wheels").mkdir(parents=True)
    source = ports / "python/example/example-1-py3-none-any.whl"
    (root / "wheels" / source.name).write_bytes(source.read_bytes())
    store = DeliveryStore(tmp_path / "store")
    compiler = BytecodeCompiler(runtime_bundle, "test-runtime")
    key = delivery_key("example-key", port=port, bytecode_runtime=compiler.key)
    record = publish_product(port, root, "example-key", store, key=key, bytecode=compiler)
    wheel = Path(store.root) / record["wheels"][0]["path"]
    prepared = wheel.read_bytes()
    with zipfile.ZipFile(wheel) as archive:
        rows = csv.reader(io.StringIO(archive.read("example-1.dist-info/RECORD").decode()))
        for name, digest, size in rows:
            if not digest:
                continue
            data = archive.read(name)
            assert int(size) == len(data)
            assert digest == "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")
    runtime = CPythonRuntime(runtime_bundle)
    environment = Context().Environment(cpu=10**10, memory=1024**3, disk=1024**3)
    runtime.mount(environment)
    site = tmp_path / "site"
    unpack_wheel(wheel, site)
    for path in site.rglob("*"):
        if path.is_file():
            environment.mkdir(runtime.site_packages + "/" + path.relative_to(site).parent.as_posix(), parents=True)
            environment.write_file(runtime.site_packages + "/" + path.relative_to(site).as_posix(), path.read_bytes())
    result = environment.run_python("""
import sys

def reject_source_compile(event, args):
    if event == 'compile' and args[1].endswith('/example.py'):
        raise AssertionError('import recompiled delivered source')
sys.addaudithook(reject_source_compile)
import example
assert example.VALUE == 42
""")
    assert result.returncode == 0, result.stderr
    environment.write_file(runtime.site_packages + "/example.py", "VALUE = 43\n")
    result = environment.run_python("import example; assert example.VALUE == 43")
    assert result.returncode == 0, result.stderr
    wheel.unlink()
    publish_product(port, root, "example-key", store, key=key, bytecode=compiler)
    assert wheel.read_bytes() == prepared


def test_runtime_archive_bytecode_is_covered_by_integrity_manifest(tmp_path, runtime_bundle):
    from shellsim import Context
    from shellsim.cpython import CPythonRuntime

    store = DeliveryStore(tmp_path / "store")
    record = publish_runtime(runtime_bundle, "test-runtime", store, key="test-delivery")
    with zipfile.ZipFile(Path(store.root) / record["path"]) as archive:
        archive.extractall(tmp_path / "installed")
    bundle = tmp_path / "installed/runtime"
    # ZIP extraction does not restore executable permissions; catalog installation does.
    (bundle / "rootfs/usr/bin/python3.wasm").chmod(0o755)
    runtime = CPythonRuntime(bundle)
    environment = Context().Environment(cpu=10**10, memory=1024**3, disk=1024**3)
    runtime.mount(environment)
    result = environment.run_python("""
import sys

def reject_source_compile(event, args):
    if event == 'compile' and args[1].endswith('/fractions.py'):
        raise AssertionError('stdlib bytecode was not loaded')
sys.addaudithook(reject_source_compile)
from fractions import Fraction
assert Fraction(1, 3) + Fraction(2, 3) == 1
""")
    assert result.returncode == 0, result.stderr
    manifest = json.loads((bundle / "manifest.json").read_text())
    bytecode = next(name for name in manifest["files"] if name.endswith("/fractions.cpython-313.pyc"))
    (bundle / "rootfs" / bytecode.lstrip("/")).write_bytes(b"changed")
    with pytest.raises(ValueError):
        CPythonRuntime(bundle)
