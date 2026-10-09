"""Check acceptance preflight prevents assets or compiler work on invalid recipes."""

from __future__ import annotations

import dataclasses
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from ports._support.acceptance import AcceptanceRequest, _native_command, accept_port
from ports._support.graph import Port


def _port(tmp_path: Path, tests: object) -> Port:
    return Port("python/example/recipe.json", tmp_path, "example", "1.0", "f" * 64, (), {"tests": tests})


def test_acceptance_requires_existing_declared_probe_before_release_fetch(tmp_path: Path) -> None:
    descriptor = tmp_path / "release.json"
    descriptor.write_text("{}")
    output = tmp_path / "proof"
    with pytest.raises(ValueError, match="declared acceptance"):
        accept_port(AcceptanceRequest(_port(tmp_path, []), descriptor, output, "pypi"))
    with pytest.raises(ValueError, match="regular port file"):
        accept_port(
            AcceptanceRequest(
                _port(tmp_path, [{"kind": "python", "script": "tests/missing.py"}]), descriptor, output, "pypi"
            )
        )
    assert not output.exists()


def test_acceptance_rejects_test_kind_mismatch_before_release_fetch(tmp_path: Path) -> None:
    descriptor = tmp_path / "release.json"
    descriptor.write_text("{}")
    source = tmp_path / "test.py"
    source.write_text("print('ok')\n")
    port = _port(tmp_path, [{"kind": "python", "script": "test.py"}])
    output = tmp_path / "proof"
    with pytest.raises(ValueError, match="Python package"):
        accept_port(AcceptanceRequest(port, descriptor, output, "native"))
    assert not output.exists()

    port = dataclasses.replace(port, recipe={"tests": [{"kind": "native", "source": "test.py"}]})
    with pytest.raises(ValueError, match="native port"):
        accept_port(AcceptanceRequest(port, descriptor, output, "pypi"))
    assert not output.exists()

    with pytest.raises(ValueError, match="verified build cohort"):
        accept_port(AcceptanceRequest(port, descriptor, output, "native"))
    assert not output.exists()


def test_native_acceptance_rejects_include_escape_before_build(tmp_path: Path) -> None:
    descriptor = tmp_path / "release.json"
    descriptor.write_text("{}")
    (tmp_path / "test.c").write_text("int main(void) { return 0; }\n")
    port = _port(
        tmp_path,
        [{"kind": "native", "source": "test.c", "include_directories": ["../host"], "link_inputs": []}],
    )
    cohort = SimpleNamespace(
        has_frontend=True, compiler=lambda: tmp_path / "cc", sysroot=SimpleNamespace(root=tmp_path / "cohort")
    )
    output = tmp_path / "proof"
    with pytest.raises(ValueError, match="escapes dependency sysroot"):
        accept_port(
            AcceptanceRequest(port, descriptor, output, "native", cohort=cohort, dependency_sysroot=tmp_path / "deps")
        )
    assert not output.exists()

    port = _port(tmp_path, [{"kind": "native", "source": "test.c", "link_inputs": ["lib/libz.a"] * 257}])
    with pytest.raises(ValueError, match="must be lists"):
        accept_port(
            AcceptanceRequest(port, descriptor, output, "native", cohort=cohort, dependency_sysroot=tmp_path / "deps")
        )
    assert not output.exists()

    port = _port(tmp_path, [{"kind": "native", "source": "test.c", "cohort_link_inputs": ["../host.a"]}])
    with pytest.raises(ValueError, match="escapes verified sysroot"):
        accept_port(
            AcceptanceRequest(port, descriptor, output, "native", cohort=cohort, dependency_sysroot=tmp_path / "deps")
        )
    assert not output.exists()


def test_successful_acceptance_does_not_publish_asset_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    descriptor = tmp_path / "release.json"
    descriptor.write_text("{}")
    (tmp_path / "probe.py").write_text("print('ok')\n")
    port = _port(tmp_path, [{"kind": "python", "script": "probe.py"}])
    cache = None

    @dataclasses.dataclass
    class Usage:
        cpu: int = 0

    class FakeEnvironment:
        @staticmethod
        def from_release(_descriptor: Path, *, cache_dir: Path, **_kwargs: object) -> FakeEnvironment:
            nonlocal cache
            cache = cache_dir
            (cache_dir / "large-asset").write_bytes(b"cached")
            return FakeEnvironment()

        def write_file(self, _path: str, _data: bytes, *, mode: int) -> None:
            assert mode == 0o644

        def run(self, _command: str) -> SimpleNamespace:
            return SimpleNamespace(stdout=b"ok\n", stderr=b"", returncode=0, stop_reason=None, usage=Usage())

    monkeypatch.setitem(sys.modules, "shellsim", SimpleNamespace(Environment=FakeEnvironment))
    output = tmp_path / "proof"
    accept_port(AcceptanceRequest(port, descriptor, output, "pypi"))
    assert cache is not None and not cache.exists()
    assert not (output / "cache").exists()


def test_native_probe_uses_only_hashed_cohort_archive(tmp_path: Path) -> None:
    root = tmp_path / "cohort"
    archive = root / "sysroot/lib/wasm32-wasip1-threads/libsetjmp.a"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"archive")
    relative = "lib/wasm32-wasip1-threads/libsetjmp.a"
    cohort = SimpleNamespace(
        has_frontend=True,
        compiler=lambda: tmp_path / "cc",
        compiler_flags=(),
        linker_flags=(),
        executable_flags=(),
        sysroot=SimpleNamespace(
            root=root, contents={"artifacts": {"sysroot/" + relative: hashlib.sha256(archive.read_bytes()).hexdigest()}}
        ),
    )
    command = _native_command(
        cohort, tmp_path / "dependencies", tmp_path / "probe.c", tmp_path / "probe.wasm", [], [relative], []
    )
    assert str(archive) in command
    archive.write_bytes(b"changed")
    with pytest.raises(ValueError, match="differs from verified sysroot"):
        _native_command(
            cohort, tmp_path / "dependencies", tmp_path / "probe.c", tmp_path / "probe.wasm", [], [relative], []
        )
