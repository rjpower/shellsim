"""Compose verified ABI-scoped package catalogs without changing package bytes.

This host-only command uses the public runtime's existing catalog and wheel
validators. It publishes a new directory only after every input and the
combined native dependency closure have been checked.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from collections.abc import Sequence
from pathlib import Path

from packaging.version import InvalidVersion, Version

_MAX_INPUTS = 32
_MAX_STAGED_BYTES = 2 * 1024**3
_MAX_INSPECTED_BYTES = 4 * 1024**3
_MAX_UNCOMPRESSED_BYTES = 2 * 1024**3
_MAX_WHEEL_MEMBERS = 50_000


def compose(runtime_bundle: Path, catalog_dirs: Sequence[Path], output: Path) -> Path:
    """Publish a merged catalog for one runtime ABI, keeping distinct versions.

    The output path must be absent. Package files and native providers are
    copied from verified bytes; conflicts and missing native dependencies leave
    no published output.
    """
    from shellsim import CPythonRuntime, PackageInstallError
    from shellsim._cpython_universe import (
        _MAX_NATIVE_BYTES,
        _MAX_PACKAGES,
        _MAX_PROVIDERS,
        _MAX_WHEEL_BYTES,
        Universe,
        _inspect_wheel,
        _verify_file,
        _verify_wasm,
    )

    if not 1 <= len(catalog_dirs) <= _MAX_INPUTS:
        raise ValueError("catalog composition needs between 1 and 32 inputs")
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    runtime = CPythonRuntime(runtime_bundle)
    abi = runtime.manifest.get("dynamic_abi")
    if not isinstance(abi, str) or not abi:
        raise ValueError("catalog composition needs a dynamic CPython runtime")
    runtime_files = runtime.manifest["files"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".shellsim-catalog-", dir=output.parent) as temporary:
        staging = Path(temporary) / "catalog"
        (staging / "wheels").mkdir(parents=True)
        (staging / "providers").mkdir()
        packages: dict[tuple[str, str], dict[str, str]] = {}
        identities: dict[tuple[str, Version], dict[str, str]] = {}
        providers: dict[str, dict[str, object]] = {}
        wheel_names: dict[str, tuple[str, str]] = {}
        required_providers: set[str] = set()
        examined = staged = unpacked = member_count = 0
        for directory in catalog_dirs:
            root = Path(directory)
            if root.is_symlink():
                raise PackageInstallError("catalog input cannot be a symbolic link")
            universe = Universe(root, abi=abi, python_version=runtime.version)
            if universe.default_index != "https://pypi.org/simple":
                raise PackageInstallError("catalog composition requires catalogued wheels, not a local pure index")
            for (name, version), package in sorted(universe.packages.items()):
                try:
                    identity = (name, Version(version))
                except InvalidVersion as error:
                    raise PackageInstallError(f"invalid curated package version: {name}=={version}") from error
                source = package["path"]
                digest = package["sha256"]
                examined += source.stat().st_size
                if examined > _MAX_INSPECTED_BYTES:
                    raise PackageInstallError("catalog inputs exceed the inspected byte limit")
                data = _verify_file(source, digest, _MAX_WHEEL_BYTES)
                key = (name, version)
                previous = identities.get(identity)
                if previous is not None:
                    if previous["sha256"] != digest or previous["version"] != version:
                        raise PackageInstallError(f"conflicting curated package version: {name}=={version}")
                    continue
                if len(packages) >= _MAX_PACKAGES or staged + len(data) > _MAX_STAGED_BYTES:
                    raise PackageInstallError("composed catalog exceeds package or byte limits")
                filename = source.name
                owner = wheel_names.get(filename)
                if owner is not None and owner != key:
                    raise PackageInstallError(f"wheel filename conflicts across packages: {filename}")
                destination = staging / "wheels" / filename
                destination.write_bytes(data)
                inspection = _inspect_wheel(destination, name=name, version=version, abi=abi, curated=True)
                unpacked += inspection.uncompressed_bytes
                member_count += inspection.file_count
                if unpacked > _MAX_UNCOMPRESSED_BYTES or member_count > _MAX_WHEEL_MEMBERS:
                    raise PackageInstallError("composed wheels exceed aggregate inspection limits")
                required_providers.update(inspection.dependencies)
                packages[key] = {
                    "name": name,
                    "version": version,
                    "wheel": "wheels/" + filename,
                    "sha256": digest,
                }
                identities[identity] = packages[key]
                wheel_names[filename] = key
                staged += len(data)
            for name, provider in sorted(universe.providers.items()):
                source = provider["path"]
                digest = provider["sha256"]
                examined += source.stat().st_size
                if examined > _MAX_INSPECTED_BYTES:
                    raise PackageInstallError("catalog inputs exceed the inspected byte limit")
                data = _verify_file(source, digest, _MAX_NATIVE_BYTES)
                _verify_wasm(data, abi)
                runtime_digest = runtime_files.get("/lib/" + name)
                if runtime_digest is not None and runtime_digest != digest:
                    raise PackageInstallError(f"native provider conflicts with the runtime: {name}")
                dependencies = sorted(provider["native_dependencies"])
                previous = providers.get(name)
                if previous is not None:
                    if previous["sha256"] != digest or previous["native_dependencies"] != dependencies:
                        raise PackageInstallError(f"conflicting native provider: {name}")
                    continue
                if len(providers) >= _MAX_PROVIDERS or staged + len(data) > _MAX_STAGED_BYTES:
                    raise PackageInstallError("composed catalog exceeds provider or byte limits")
                (staging / "providers" / name).write_bytes(data)
                providers[name] = {
                    "name": name,
                    "path": "providers/" + name,
                    "destination": "/lib/" + name,
                    "sha256": digest,
                    "native_dependencies": dependencies,
                }
                staged += len(data)
        catalog = {
            "schema_version": 1,
            "abi": abi,
            "target": "wasm32-wasip1",
            "python_version": runtime.version,
            "packages": [packages[key] for key in sorted(packages)],
            "native_providers": [providers[key] for key in sorted(providers)],
        }
        (staging / "catalog.json").write_text(json.dumps(catalog, sort_keys=True, indent=2) + "\n")
        merged = Universe(staging, abi=abi, python_version=runtime.version)
        for package in merged.packages.values():
            _verify_file(package["path"], package["sha256"], _MAX_WHEEL_BYTES)
        for provider in merged.providers.values():
            _verify_wasm(_verify_file(provider["path"], provider["sha256"], _MAX_NATIVE_BYTES), abi)
        merged.provider_closure(required_providers | set(merged.providers))
        if output.exists() or output.is_symlink():
            raise FileExistsError(output)
        staging.replace(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True, help="verified dynamic CPython bundle")
    parser.add_argument("--catalog", type=Path, action="append", required=True, help="input catalog directory")
    parser.add_argument("--output", type=Path, required=True, help="new merged catalog directory")
    args = parser.parse_args()
    print(compose(args.runtime, args.catalog, args.output))


if __name__ == "__main__":
    main()
