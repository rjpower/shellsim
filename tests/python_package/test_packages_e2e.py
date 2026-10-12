"""Build, distribute, load, and run a workspace package through the public host API."""

from __future__ import annotations

import base64
import io
import json
import shlex
import stat
import struct
import zipfile
from pathlib import Path

import pytest
import shellsim

GUEST_SOURCE = Path(__file__).resolve().parents[1] / "fixtures" / "host_tools"
SPEC = shellsim.PackageSpec(
    name="arithmetic-grader",
    version="0.1.0",
    entrypoint=("python3.14", "/work/app/grader.py"),
    required_tools=("workspace.read_file", "conversation.list", "grade.submit"),
)


@pytest.fixture
def source_tree(tmp_path: Path) -> Path:
    root = tmp_path / "grader"
    app = root / "app"
    app.mkdir(parents=True)
    for name in ("grader.py", "host_tools.py"):
        (app / name).write_bytes((GUEST_SOURCE / name).read_bytes())
    (app / "empty").mkdir()
    return root


def _zip_tree(root: Path) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root).as_posix()
            mode = stat.S_IMODE(path.stat().st_mode)
            name = "grader/" + relative + ("/" if path.is_dir() else "")
            member = zipfile.ZipInfo(name)
            member.create_system = 3
            member.external_attr = ((stat.S_IFDIR if path.is_dir() else stat.S_IFREG) | mode) << 16
            archive.writestr(member, b"" if path.is_dir() else path.read_bytes())
    return buffer.getvalue()


def _handlers(scores: list[float]) -> dict[str, object]:
    return {
        "workspace.read_file": lambda arguments: {
            "data_base64": base64.b64encode(b"42\n").decode(),
        },
        "conversation.list": lambda arguments: {"messages": ["What is 6 * 7?"]},
        "grade.submit": lambda arguments: scores.append(arguments["score"]) or {"accepted": True},
        "not.declared": lambda arguments: {"secret": True},
    }


def test_directory_zip_container_and_blob_url_round_trip(source_tree: Path) -> None:
    built = shellsim.Package.build_from_directory(source_tree, spec=SPEC)
    assert built.to_bytes() == shellsim.Package.build_from_directory(source_tree, spec=SPEC).to_bytes()
    from_zip = shellsim.Package.build_from_zip(_zip_tree(source_tree), spec=SPEC, strip_prefix="grader")
    assert from_zip.to_bytes() == built.to_bytes()

    staged = shellsim.Container(tools={})
    for path in sorted(source_tree.rglob("*")):
        destination = "/work/" + path.relative_to(source_tree).as_posix()
        mode = stat.S_IMODE(path.stat().st_mode)
        if path.is_dir():
            staged.mkdir(destination, mode=mode)
        else:
            staged.write_file(destination, path.read_bytes(), mode=mode)
    assert staged.run("printf ready").stdout == b"ready"
    from_container = shellsim.Package.build_from_container(staged, spec=SPEC)
    assert from_container.to_bytes() == built.to_bytes()
    assert staged.read_file("/work/app/grader.py") == (source_tree / "app/grader.py").read_bytes()

    blob = built.to_bytes()
    fetched = shellsim.Package.from_url(
        "memory://grader.shl",
        expected_sha256=built.sha256,
        fetcher=lambda url: (blob[:19], blob[19:]),
    )
    assert fetched.to_bytes() == blob
    first_scores: list[float] = []
    second_scores: list[float] = []
    first = fetched.instantiate(tools=_handlers(first_scores))
    second = fetched.instantiate(tools=_handlers(second_scores))
    assert first.run_entrypoint().stdout == b"1.0\n"
    assert second.run_entrypoint().stdout == b"1.0\n"
    assert first_scores == second_scores == [1.0]
    unlisted = (
        "from host_tools import ToolClient\n"
        "try:\n"
        "    ToolClient().call('not.declared', {})\n"
        "except RuntimeError as error:\n"
        "    print(str(error))\n"
    )
    assert first.run(f"cd /work/app && python3.14 -c {shlex.quote(unlisted)}").stdout == b"unknown tool\n"
    first.write_file("/work/app/new.txt", b"private")
    with pytest.raises(shellsim.SimulationError):
        second.read_file("/work/app/new.txt")
    assert first.run("test -d /work/app/empty").returncode == 0


def test_missing_tools_digest_and_live_export_are_rejected(source_tree: Path) -> None:
    package = shellsim.Package.build_from_directory(source_tree, spec=SPEC)
    with pytest.raises(ValueError, match="missing required tool"):
        package.instantiate(tools={})
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        shellsim.Package.from_url(
            "memory://grader.shl", expected_sha256="0" * 64, fetcher=lambda url: (package.to_bytes(),)
        )
    with pytest.raises(ValueError, match="HTTPS"):
        shellsim.Package.from_url("http://localhost/grader.shl", expected_sha256=package.sha256)

    observed: list[str] = []

    def list_conversation(arguments: dict[str, object]) -> dict[str, list[str]]:
        with pytest.raises(shellsim.SimulationError, match="quiescent"):
            shellsim.Package.build_from_container(container, spec=SPEC)
        observed.append("rejected")
        return {"messages": ["What is 6 * 7?"]}

    handlers = _handlers([])
    handlers["conversation.list"] = list_conversation
    container = package.instantiate(tools=handlers)
    assert container.run_entrypoint().returncode == 0
    assert observed == ["rejected"]


def _rewrite_package(blob: bytes, extra: dict[str, bytes] | None = None, manifest_change: object = None) -> bytes:
    with zipfile.ZipFile(io.BytesIO(blob)) as original:
        content = {member.filename: original.read(member) for member in original.infolist()}
    if manifest_change is not None:
        manifest = json.loads(content["shl.json"])
        manifest_change(manifest)
        content["shl.json"] = json.dumps(manifest).encode()
    content.update(extra or {})
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as rewritten:
        for name, data in content.items():
            rewritten.writestr(name, data)
    return buffer.getvalue()


def test_loader_rejects_tampered_or_undeclared_payload(source_tree: Path) -> None:
    blob = shellsim.Package.build_from_directory(source_tree, spec=SPEC).to_bytes()
    changed = b"x" * len((source_tree / "app/grader.py").read_bytes())
    with pytest.raises(ValueError, match="digest mismatch"):
        shellsim.Package.from_bytes(_rewrite_package(blob, extra={"rootfs/app/grader.py": changed}))
    with pytest.raises(ValueError, match="undeclared"):
        shellsim.Package.from_bytes(_rewrite_package(blob, extra={"rootfs/extra": b"hidden"}))
    with pytest.raises(ValueError, match="workspace path"):
        shellsim.Package.from_bytes(
            _rewrite_package(blob, manifest_change=lambda manifest: manifest["entries"][0].update(path="../escape"))
        )
    with pytest.raises(ValueError, match="working directory is missing"):
        shellsim.Package.from_bytes(
            _rewrite_package(blob, manifest_change=lambda manifest: manifest.update(working_directory="/work/missing"))
        )

    with pytest.raises(ValueError):
        shellsim.Package.from_bytes(
            _rewrite_package(blob, manifest_change=lambda manifest: manifest.update(requires_c_toolchain=True))
        )

    too_many_members = bytearray(blob)
    directory_end = too_many_members.rfind(b"PK\x05\x06")
    struct.pack_into("<HH", too_many_members, directory_end + 8, 10_002, 10_002)
    with pytest.raises(ValueError, match="too many members"):
        shellsim.Package.from_bytes(bytes(too_many_members))


def test_builder_rejects_source_links_duplicates_and_large_files(source_tree: Path) -> None:
    (source_tree / "link").symlink_to("app/grader.py")
    with pytest.raises(ValueError, match="unsupported source entry"):
        shellsim.Package.build_from_directory(source_tree, spec=SPEC)
    (source_tree / "link").unlink()
    # Admission uses the declared size before reading the source bytes.
    with (source_tree / "huge").open("wb") as output:
        output.truncate(64 * 1024 * 1024 + 1)
    with pytest.raises(ValueError, match="exceeds 64 MiB"):
        shellsim.Package.build_from_directory(source_tree, spec=SPEC)

    duplicate = io.BytesIO()
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(duplicate, "w") as archive:
            archive.writestr("same", b"one")
            archive.writestr("same", b"two")
    with pytest.raises(ValueError, match="duplicate"):
        shellsim.Package.build_from_zip(duplicate.getvalue(), spec=SPEC)

    linked = io.BytesIO()
    with zipfile.ZipFile(linked, "w") as archive:
        member = zipfile.ZipInfo("link")
        member.create_system = 3
        member.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(member, b"target")
    with pytest.raises(ValueError, match="links and special files"):
        shellsim.Package.build_from_zip(linked.getvalue(), spec=SPEC)

    compressed = io.BytesIO()
    with zipfile.ZipFile(compressed, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("bomb", b"x")
    oversized = bytearray(compressed.getvalue())
    # Oversized member metadata must be rejected before decompression.
    struct.pack_into("<I", oversized, 22, 64 * 1024 * 1024 + 1)
    directory = oversized.index(b"PK\x01\x02")
    struct.pack_into("<I", oversized, directory + 24, 64 * 1024 * 1024 + 1)
    with pytest.raises(ValueError, match="exceeds 64 MiB"):
        shellsim.Package.build_from_zip(bytes(oversized), spec=SPEC)


def test_package_entrypoint_runs_interactive_wasm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    fixture = Path(__file__).resolve().parents[1] / "fixtures/wasm/interactive_package.wasm"
    executable = tmp_path / "display.wasm"
    executable.write_bytes(fixture.read_bytes())
    executable.chmod(0o755)
    limits = shellsim.Limits(cpu=20_000_000, memory=64 * 1024**2, disk=64 * 1024**2)
    package = shellsim.Package.build_from_directory(
        tmp_path,
        spec=shellsim.PackageSpec(name="interactive", version="1", entrypoint=("/work/display.wasm",), limits=limits),
    )
    container = shellsim.Package.from_bytes(package.to_bytes()).instantiate(tools={}, limits=limits)
    with container.start_entrypoint() as action:
        first = None
        for _ in range(100):
            first = action.poll()
            if first.frame is not None or first.state == "complete":
                break
        assert first is not None and first.frame is not None
        assert (first.frame.width, first.frame.height, first.frame.pixels) == (2, 1, b"\0" * 8)
        action.inject_key(27, True)
        second = None
        for _ in range(100):
            second = action.poll()
            if second.frame is not None or second.state == "complete":
                break
        assert second is not None and second.frame is not None
        assert second.frame.generation == first.frame.generation + 1
        assert second.frame.pixels[0] == 255
        while second.state != "complete":
            second = action.poll()
        assert action.result.returncode == 0

    from shellsim._cli import main

    class FakeDisplayHost:
        def __init__(self, action: shellsim.Action) -> None:
            self.url = "http://127.0.0.1/display"
            self.frame: shellsim.DisplayFrame | None = None
            self.sent_key = False

        def __enter__(self) -> FakeDisplayHost:
            return self

        def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
            pass

        def publish(self, frame: shellsim.DisplayFrame) -> None:
            self.frame = frame

        def drain_keys(self) -> list[tuple[int, bool]]:
            if self.frame is not None and not self.sent_key:
                self.sent_key = True
                return [(27, True)]
            return []

        def stop_requested(self) -> bool:
            return False

        def mark_stopped(self) -> None:
            pass

    monkeypatch.setattr("shellsim.package._fetch_https", lambda url: (package.to_bytes(),))
    monkeypatch.setattr("shellsim._cli.DisplayHost", FakeDisplayHost)
    assert main(["run", "https://example.com/display.shl", "--sha256", package.sha256]) == 0
    assert "http://127.0.0.1/display" in capsys.readouterr().err


def test_package_accepts_large_file_and_requests_host_clock(tmp_path: Path) -> None:
    content = b"w" * (7 * 1024 * 1024)
    (tmp_path / "large.wad").write_bytes(content)
    package = shellsim.Package.build_from_directory(
        tmp_path,
        spec=shellsim.PackageSpec(name="large-game", version="1", entrypoint=("true",), requested_clock="real_time"),
    )
    loaded = shellsim.Package.from_bytes(package.to_bytes())
    assert loaded.spec.requested_clock == "real_time"
    assert loaded.instantiate(tools={}).clock == "virtual"
    real_time = loaded.instantiate(tools={}, clock="real_time")
    assert real_time.clock == "real_time"
    assert real_time.read_file("/work/large.wad") == content
