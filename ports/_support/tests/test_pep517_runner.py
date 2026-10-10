"""Run isolated backend hooks and reject unadmitted requirements before build."""

import importlib.metadata
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from ports._support import pep517_runner
from ports._support.python_pep517 import _backend_paths


@pytest.fixture
def backend_request(tmp_path):
    imports = tmp_path / "imports"
    imports.mkdir()
    distribution = importlib.metadata.distribution("packaging")
    for item in distribution.files:
        if item.parts[0] == "packaging" or item.parts[0].endswith(".dist-info"):
            source = distribution.locate_file(item)
            if source.is_file():
                destination = imports / item
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
    source = tmp_path / "source"
    source.mkdir()
    (source / "backend.py").write_text(
        "import os, pathlib, zipfile\n"
        "def build_wheel(output, config_settings):\n"
        "    import sys\n"
        "    assert not any('site-packages' in p for p in sys.path)\n"
        "    path = pathlib.Path(output) / 'sample-1-py3-none-any.whl'\n"
        "    with zipfile.ZipFile(path, 'w') as wheel:\n"
        "        wheel.writestr('sample.py', 'VALUE = 42\\n')\n"
        "    return path.name\n"
    )
    output = tmp_path / "output"
    output.mkdir()
    payload = {
        "backend_paths": [str(source)],
        "imports": str(imports),
        "configuration": str(tmp_path),
        "requires": ["packaging>=20"],
        "backend": "backend",
        "config_settings": {},
        "python_version": list(sys.version_info[:2]),
        "output": str(output),
        "response": str(tmp_path / "response.json"),
    }
    return tmp_path, source, payload


def invoke(request):
    root, _, payload = request
    path = root / "request.json"
    path.write_text(json.dumps(payload))
    return subprocess.run(
        [sys.executable, "-I", "-S", "-B", pep517_runner.__file__, str(path)],
        capture_output=True,
        check=False,
    )


def test_optional_requirement_hook_builds_with_isolated_imports(backend_request):
    assert invoke(backend_request).returncode == 0
    root, _, payload = backend_request
    response = json.loads((root / "response.json").read_text())
    assert response["additional_requires"] == []
    assert (Path(payload["output"]) / response["wheel"]).is_file()


@pytest.mark.parametrize("requirement", ["missing_backend==1", "packaging<1", "packaging @ https://example.org/p.whl"])
@pytest.mark.parametrize("additional", [False, True])
def test_missing_or_url_requirement_prevents_backend_output(backend_request, requirement, additional):
    root, source, payload = backend_request
    if additional:
        with (source / "backend.py").open("a") as stream:
            stream.write("def get_requires_for_build_wheel(config_settings):\n    return " + repr([requirement]) + "\n")
    else:
        payload["requires"] = [requirement]
    assert invoke(backend_request).returncode != 0
    assert not (root / "response.json").exists()
    assert list(Path(payload["output"]).iterdir()) == []


def test_backend_path_cannot_escape_source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "linked").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError):
        _backend_paths(source, {"backend-path": ["linked"]})


def test_backend_runtime_requirements_are_also_pinned(backend_request):
    root, _, payload = backend_request
    metadata = next(Path(payload["imports"]).glob("packaging-*.dist-info/METADATA"))
    metadata.write_text("Requires-Dist: unavailable_transitive_dependency==1\n" + metadata.read_text())
    assert invoke(backend_request).returncode != 0
    assert not (root / "response.json").exists()
    assert list(Path(payload["output"]).iterdir()) == []
