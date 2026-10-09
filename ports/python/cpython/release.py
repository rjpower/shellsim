"""Seal an existing verified CPython runtime, catalog and host resolver for delivery."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import stat
import subprocess
import tempfile
import zipfile
from pathlib import Path

from shellsim._cpython_release import _descriptor, _digest, _validate_entry
from shellsim._native_release import _validate_entry as _validate_native_entry
from shellsim.cpython import _MAX_MANIFEST_BYTES, _PYTHON_PACKAGE_PLATFORM
from shellsim.native_packages import verify_release_catalog


def _host_requirements(uv: Path) -> tuple[str, list[str]]:
    """Record the actual glibc and shared-library floor of the Linux resolver."""
    with uv.open("rb") as source:
        header = source.read(20)
    if len(header) != 20 or header[:6] != b"\x7fELF\x02\x01" or int.from_bytes(header[18:20], "little") != 62:
        raise ValueError("the first release requires an x86-64 Linux ELF resolver")
    symbols = subprocess.run(["objdump", "-T", str(uv)], capture_output=True, text=True, check=True, timeout=60)
    versions = {
        tuple(map(int, version.split("."))) for version in re.findall(r"GLIBC_([0-9]+\.[0-9]+)", symbols.stdout)
    }
    if not versions:
        raise ValueError("resolver does not declare a measurable glibc floor")
    dynamic = subprocess.run(["readelf", "-d", str(uv)], capture_output=True, text=True, check=True, timeout=60)
    libraries = sorted(set(re.findall(r"Shared library: \[([^]]+)\]", dynamic.stdout)))
    return ".".join(map(str, max(versions))), libraries


def _archive(root: Path, destination: Path) -> None:
    """Write stable ZIP metadata so identical verified inputs have identical bytes."""
    with zipfile.ZipFile(destination, "w") as archive:
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.relative_to(root).as_posix() == "uv":
                continue
            name = path.relative_to(root).as_posix()
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (stat.S_IFREG | (0o755 if path.stat().st_mode & 0o111 else 0o644)) << 16
            archive.writestr(info, path.read_bytes())


def build_release(
    runtime: Path,
    universe: Path,
    uv: Path,
    output: Path,
    *,
    base_url: str | None = None,
    native_catalog: Path | None = None,
) -> Path:
    """Copy and verify one release; return its trusted local descriptor path."""
    if output.exists() or output.is_symlink():
        raise ValueError("release output already exists")
    if uv.is_symlink() or not uv.is_file():
        raise ValueError("patched uv resolver must be a regular file")
    floor, libraries = _host_requirements(uv)
    if base_url is not None and (not base_url.startswith("https://") or not base_url.endswith("/")):
        raise ValueError("published release base URL must be HTTPS and end in /")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".shellsim-release-", dir=output.parent) as temporary:
        work = Path(temporary)
        stage = work / "stage"
        stage.mkdir()
        shutil.copytree(runtime, stage / "runtime", symlinks=True)
        shutil.copytree(universe, stage / "universe", symlinks=True)
        shutil.copy2(uv, stage / "uv")
        (stage / "uv").chmod(0o755)
        resolver = {"sha256": _digest(stage / "uv", 256 * 1024 * 1024), "size": (stage / "uv").stat().st_size}
        manifest = json.loads((stage / "runtime/manifest.json").read_text())
        descriptor = {
            "schema_version": 1,
            "target": _PYTHON_PACKAGE_PLATFORM,
            "runtime_target": manifest["recipe"]["target"],
            "python_version": "3.13.7",
            "abi": manifest.get("dynamic_abi"),
            "runtime_manifest_sha256": _digest(stage / "runtime/manifest.json", _MAX_MANIFEST_BYTES),
            "catalog_sha256": _digest(stage / "universe/catalog.json", 1024 * 1024),
            "resolvers": {"linux-x86_64-glibc": {**resolver, "min_glibc": floor, "needed_libraries": libraries}},
        }
        _validate_entry(stage, descriptor, descriptor["resolvers"]["linux-x86_64-glibc"])
        archive = work / "cohort.zip"
        _archive(stage, archive)
        if archive.stat().st_size > 256 * 1024 * 1024:
            raise ValueError("release archive exceeds 256 MiB")
        descriptor["archive"] = {"sha256": _digest(archive, 256 * 1024 * 1024), "size": archive.stat().st_size}
        asset = work / "uv-linux-x86_64-glibc"
        shutil.copy2(stage / "uv", asset)
        descriptor["archive"]["url"] = (base_url or "") + "cohort.zip"
        descriptor["resolvers"]["linux-x86_64-glibc"]["url"] = (base_url or "") + asset.name
        if native_catalog is not None:
            verify_release_catalog(native_catalog)
            native_stage = work / "native-stage"
            native_stage.mkdir()
            shutil.copytree(native_catalog.parent, native_stage / "native", symlinks=True)
            native = {
                "catalog_sha256": _digest(native_stage / "native/catalog.json", 1024 * 1024),
                "archive": {"url": (base_url or "") + "native.zip"},
            }
            _validate_native_entry(native_stage, native)
            native_archive = work / "native.zip"
            _archive(native_stage, native_archive)
            if native_archive.stat().st_size > 256 * 1024 * 1024:
                raise ValueError("native release archive exceeds 256 MiB")
            native["archive"].update(
                {"sha256": _digest(native_archive, 256 * 1024 * 1024), "size": native_archive.stat().st_size}
            )
            descriptor["native"] = native
        (work / "release.json").write_text(json.dumps(descriptor, sort_keys=True, indent=2) + "\n")
        _descriptor(work / "release.json")
        deliver = work / "deliver"
        deliver.mkdir()
        assets = ["cohort.zip", "uv-linux-x86_64-glibc", "release.json"]
        if native_catalog is not None:
            assets.append("native.zip")
        for name in assets:
            (work / name).replace(deliver / name)
        deliver.replace(output)
    return output / "release.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime", "universe", "uv", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--base-url", help="immutable HTTPS release asset directory; local sibling files by default")
    parser.add_argument("--native-catalog", type=Path, help="verified native build-tool catalog")
    args = parser.parse_args()
    print(
        build_release(
            args.runtime,
            args.universe,
            args.uv,
            args.output,
            base_url=args.base_url,
            native_catalog=args.native_catalog,
        )
    )


if __name__ == "__main__":
    main()
