"""Build a pinned, unchanged pure sdist into the existing private Simple index.

Only the explicitly approved upstream setup script executes on the host, inside
an isolated pinned setuptools environment. Package imports and tests execute in
upstream guest CPython separately; a pure wheel does not establish dependency support.
"""

import argparse
import base64
import csv
import email
import hashlib
import io
import json
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

PORT = Path(__file__).resolve().parent
_MAX_FILES = 1024
_MAX_BYTES = 16 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(name: str) -> PurePosixPath:
    path = PurePosixPath(name.rstrip("/"))
    if (
        not name
        or not path.parts
        or "\\" in name
        or "\0" in name
        or path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != name.rstrip("/")
    ):
        raise ValueError("build archive contains an unsafe member path")
    return path


def _unpack_tool(wheel: Path, destination: Path):
    with zipfile.ZipFile(wheel) as archive:
        members = archive.infolist()
        if len(members) > _MAX_FILES or sum(member.file_size for member in members) > _MAX_BYTES:
            raise ValueError("build tool archive exceeds its bound")
        seen = set()
        for member in members:
            relative = _relative(member.filename)
            if relative in seen or (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("build tool archive has duplicate or linked members")
            seen.add(relative)
            path = destination / relative
            if member.is_dir():
                path.mkdir(parents=True, exist_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(archive.read(member))


def validate_wheel(wheel: Path, source: Path, recipe: dict) -> dict[str, str]:
    """Require pure tags, unchanged source/license bytes, dependencies and RECORD hashes."""
    if wheel.name != recipe["wheel"]:
        raise ValueError("built pure wheel name differs from the recipe")
    dist_info = f"{recipe['name']}-{recipe['version']}.dist-info"
    with zipfile.ZipFile(wheel) as archive:
        members = archive.infolist()
        if len(members) > _MAX_FILES or sum(member.file_size for member in members) > _MAX_BYTES:
            raise ValueError("built wheel exceeds its bound")
        files = {}
        for member in members:
            relative = _relative(member.filename)
            if member.is_dir() or relative.as_posix() in files:
                raise ValueError("pure wheel has duplicate or directory entries")
            if relative.parts[0] not in (recipe["name"], dist_info):
                raise ValueError("pure wheel has an unexpected payload root")
            if relative.suffix in (".so", ".wasm", ".a", ".dll", ".pyd"):
                raise ValueError("pure wheel contains native artifacts")
            files[relative.as_posix()] = archive.read(member)
        expected = {
            path.relative_to(source).as_posix(): path.read_bytes()
            for path in (source / recipe["name"]).rglob("*")
            if path.is_file()
        }
        actual = {name: data for name, data in files.items() if name.startswith(recipe["name"] + "/")}
        if actual != expected:
            raise ValueError("wheel package source differs from the upstream sdist")
        metadata = email.message_from_bytes(files[dist_info + "/METADATA"])
        if (
            metadata["Name"] != recipe["name"]
            or metadata["Version"] != recipe["version"]
            or metadata["Requires-Python"] != recipe["requires_python"]
            or sorted(value.replace(" ", "") for value in metadata.get_all("Requires-Dist", []))
            != sorted(recipe["requires_dist"])
        ):
            raise ValueError("wheel identity or requirements differ from upstream metadata")
        wheel_metadata = email.message_from_bytes(files[dist_info + "/WHEEL"])
        if wheel_metadata["Root-Is-Purelib"] != "true" or wheel_metadata.get_all("Tag") != ["py3-none-any"]:
            raise ValueError("wheel does not declare the expected pure Python tag")
        license_members = [name for name in files if name.endswith("/LICENSE.txt")]
        if len(license_members) != 1 or files[license_members[0]] != (source / "LICENSE.txt").read_bytes():
            raise ValueError("wheel does not preserve the upstream license")
        rows = list(csv.reader(io.StringIO(files[dist_info + "/RECORD"].decode())))
        if len(rows) != len(files) or len({row[0] for row in rows}) != len(rows):
            raise ValueError("wheel RECORD does not cover its files exactly")
        for name, recorded, size in rows:
            if name == dist_info + "/RECORD":
                if recorded or size:
                    raise ValueError("wheel RECORD must omit its own digest")
                continue
            payload = files[name]
            expected_hash = "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).decode().rstrip("=")
            if recorded != expected_hash or size != str(len(payload)):
                raise ValueError("wheel RECORD content differs from its payload")
        return {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}


def build(source_archive: Path, setuptools_wheel: Path, output: Path) -> Path:
    """Execute approved build code with pinned tools; emit wheel, provenance and Simple index."""
    recipe = json.loads((PORT / "recipe.json").read_text())
    if sha256(Path(__file__)) != recipe["build_script_sha256"]:
        raise ValueError("pure package builder differs from its recipe pin")
    if sha256(source_archive) != recipe["source"]["sha256"]:
        raise ValueError("CellPyLib source archive differs from its pin")
    tool = recipe["build_tools"]["setuptools"]
    if setuptools_wheel.name != tool["filename"] or sha256(setuptools_wheel) != tool["sha256"]:
        raise ValueError("setuptools build tool differs from its pin")
    output.mkdir(parents=True)
    tools = output / "build-tools"
    _unpack_tool(setuptools_wheel, tools)
    source_parent = output / "source"
    source_parent.mkdir()
    with tarfile.open(source_archive) as archive:
        members = archive.getmembers()
        if len(members) > _MAX_FILES or sum(member.size for member in members) > _MAX_BYTES:
            raise ValueError("source archive exceeds its bound")
        for member in members:
            if not (member.isfile() or member.isdir()) or _relative(member.name).parts[0] != "cellpylib-2.4.0":
                raise ValueError("source archive contains unsupported member entries")
        archive.extractall(source_parent, filter="data")
    source = source_parent / "cellpylib-2.4.0"
    wheels = output / "pure-wheels"
    wheels.mkdir()
    environment = {
        "PATH": "/usr/bin:/bin",
        "PYTHONPATH": str(tools.resolve()),
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "SOURCE_DATE_EPOCH": recipe["source_date_epoch"],
        "LC_ALL": "C.UTF-8",
        "TZ": "UTC",
    }
    with (output / "build.log").open("w") as log:
        subprocess.run(
            [
                sys.executable,
                "-S",
                "-c",
                "import sys; from setuptools.build_meta import build_wheel; build_wheel(sys.argv[1])",
                str(wheels.resolve()),
            ],
            cwd=source,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    wheel = wheels / recipe["wheel"]
    members = validate_wheel(wheel, source, recipe)
    index = output / "pure-simple/cellpylib"
    index.mkdir(parents=True)
    (index / "index.html").write_text(
        f'<a href="../../pure-wheels/{wheel.name}#sha256={sha256(wheel)}">{wheel.name}</a>\n'
    )
    provenance = {
        "recipe": recipe,
        "source_sha256": sha256(source_archive),
        "host_tools": {
            "python": {
                "path": str(Path(sys.executable).resolve()),
                "sha256": sha256(Path(sys.executable)),
                "version": sys.version,
            },
            "setuptools": tool,
        },
        "wheel": {"path": "pure-wheels/" + wheel.name, "sha256": sha256(wheel), "members": members},
        "private_index": "pure-simple",
        "guest_import": "unverified: Matplotlib provider required",
    }
    (output / "artifact.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return wheel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source_archive", "setuptools_wheel", "output"):
        parser.add_argument(name, type=Path)
    arguments = parser.parse_args()
    print(build(arguments.source_archive.resolve(), arguments.setuptools_wheel.resolve(), arguments.output.resolve()))


if __name__ == "__main__":
    main()
