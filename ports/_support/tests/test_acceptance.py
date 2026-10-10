"""Check acceptance preflight prevents assets or compiler work on invalid recipes."""

from __future__ import annotations

import dataclasses
import hashlib
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from ports._support.acceptance import AcceptanceRequest, _fixtures, _native_command, accept_port
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


@pytest.mark.parametrize("limits", [{"cpu": -1}, {"memory": True}, {"disk": 2**32}, {"host": 1}, []])
def test_acceptance_rejects_invalid_resource_budgets_before_release_fetch(tmp_path, limits):
    port = _port(tmp_path, [{"kind": "python", "script": "probe.py"}])
    port = dataclasses.replace(port, recipe={**port.recipe, "test_limits": limits})
    output = tmp_path / "proof"
    with pytest.raises(ValueError, match="test_limits"):
        accept_port(AcceptanceRequest(port, tmp_path / "missing-release.json", output, "pypi"))
    assert not output.exists()


def test_acceptance_rejects_test_kind_mismatch_before_release_fetch(tmp_path: Path) -> None:
    descriptor = tmp_path / "release.json"
    descriptor.write_text("{}")
    source = tmp_path / "test.py"
    source.write_text("print('ok')\n")
    port = _port(tmp_path, [{"kind": "shell", "script": "test.py"}])
    output = tmp_path / "proof"
    with pytest.raises(ValueError, match="native port"):
        accept_port(AcceptanceRequest(port, descriptor, output, "pypi"))
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


@pytest.mark.parametrize("installation_kind", ["pypi", "pypi+native"])
def test_successful_acceptance_does_not_publish_asset_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, installation_kind: str
) -> None:
    descriptor = tmp_path / "release.json"
    descriptor.write_text("{}")
    (tmp_path / "probe.py").write_text("print('ok')\n")
    port = _port(tmp_path, [{"kind": "python", "script": "probe.py"}])
    (tmp_path / "input.txt").write_bytes(b"fixture input")
    port = dataclasses.replace(
        port,
        recipe={
            "tests": [
                {"kind": "python", "script": "probe.py", "files": [{"source": "input.txt", "destination": "input.txt"}]}
            ]
        },
    )
    written = {}
    cache = None

    @dataclasses.dataclass
    class Usage:
        cpu: int = 0

    class FakeEnvironment:
        @staticmethod
        def from_release(_descriptor: Path, *, cache_dir: Path, **_kwargs: object) -> FakeEnvironment:
            nonlocal cache
            cache = cache_dir
            assert _kwargs["pypi"] == ["example==1.0"]
            if installation_kind == "pypi+native":
                assert _kwargs["tools"] == ["example==1.0"]
            (cache_dir / "large-asset").write_bytes(b"cached")
            return FakeEnvironment()

        def write_file(self, _path: str, _data: bytes, *, mode: int) -> None:
            assert mode == 0o644
            written[_path] = _data

        def run(self, _command: str) -> SimpleNamespace:
            return SimpleNamespace(stdout=b"ok\n", stderr=b"", returncode=0, stop_reason=None, usage=Usage())

    monkeypatch.setitem(sys.modules, "shellsim", SimpleNamespace(Environment=FakeEnvironment))
    output = tmp_path / "proof"
    result = accept_port(AcceptanceRequest(port, descriptor, output, installation_kind))
    assert written["/work/input.txt"] == b"fixture input"
    assert result[0].fixture_sha256 == {"input.txt": hashlib.sha256(b"fixture input").hexdigest()}
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


@pytest.mark.parametrize("flag", ["-L/host", "-lhost", "-Wl,--wrap=ok,--script=/host", "-fplugin=/host"])
def test_native_probe_rejects_nonwrapper_link_flags(tmp_path, flag):
    cohort = SimpleNamespace(has_frontend=True)
    with pytest.raises(ValueError, match="linker wrapper"):
        _native_command(cohort, tmp_path, tmp_path / "test.c", tmp_path / "test.wasm", [], [], [], [flag])


def test_native_probe_passes_exact_wrapper_switches(tmp_path):
    cohort = SimpleNamespace(
        has_frontend=True,
        compiler=lambda: tmp_path / "cc",
        compiler_flags=(),
        linker_flags=(),
        executable_flags=(),
        sysroot=SimpleNamespace(root=tmp_path),
    )
    flag = "-Wl,--wrap=signal,--wrap=open,--wrap=openat"
    command = _native_command(cohort, tmp_path, tmp_path / "test.c", tmp_path / "test.wasm", [], [], [], [flag])
    assert flag in command


def test_native_probe_links_declared_shared_provider_with_real_lld(tmp_path):
    from ports._support.cohort import load_cohort
    from ports._support.wasm_metadata import function_signatures, needed_libraries

    descriptor = os.environ.get("SHELLSIM_BUILD_COHORT")
    if descriptor is None:
        pytest.skip("requires an explicitly admitted build cohort")
    cohort = load_cohort(Path(descriptor))
    dependency_sysroot = tmp_path / "dependencies"
    library = dependency_sysroot / "usr/local/lib/libactual.so"
    library.parent.mkdir(parents=True)
    provider = tmp_path / "provider.c"
    provider.write_text("int supplied(void) { return 37; }\n")
    subprocess.run(
        [
            str(cohort.compiler()),
            *cohort.compiler_flags,
            *cohort.linker_flags,
            *cohort.shared_library_flags,
            str(provider),
            "-Wl,--soname=libactual.so",
            "-o",
            str(library),
            str(cohort.compiler_runtime_archive),
        ],
        check=True,
    )
    source, output = tmp_path / "consumer.c", tmp_path / "consumer.wasm"
    source.write_text("extern int supplied(void); int main(void) { return supplied() != 37; }\n")
    command = _native_command(cohort, dependency_sysroot, source, output, ["lib/libactual.so"], [], [])
    subprocess.run(command, check=True)
    assert needed_libraries(output) == ["libactual.so"]
    imports, _ = function_signatures(output)
    assert imports["env", "supplied"] == ((), (0x7F,))
    rejected = subprocess.run(
        [argument for argument in command if argument != "-Wl,-Bdynamic"], capture_output=True, check=False
    )
    assert rejected.returncode != 0


def test_acceptance_fixtures_preserve_bytes_and_reject_destination_aliases(tmp_path):
    (tmp_path / "fixture.cpp").write_bytes(b"int supplied() { return 37; }\n")
    port = _port(tmp_path, [])
    declaration = {"source": "fixture.cpp", "destination": "fixture.cpp"}
    assert _fixtures(port, [declaration]) == {"fixture.cpp": (tmp_path / "fixture.cpp").read_bytes()}
    with pytest.raises(ValueError, match="duplicated"):
        _fixtures(port, [declaration, declaration])
    for destination in ("../escape", "/host", "a/../b", "shellsim-acceptance.sh"):
        with pytest.raises(ValueError):
            _fixtures(port, [{**declaration, "destination": destination}])
    with pytest.raises(ValueError, match="regular port file"):
        _fixtures(port, [{"source": "missing.cpp", "destination": "input.cpp"}])
