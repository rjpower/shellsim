"""Build upstream Python Meson projects against an admitted WASI cohort.

Meson's install plan supplies package paths and qualified extension names.
Host Python runs generators; target Python headers and native dependencies
remain explicit inputs. The runner seals wheels and development exports.
"""

from __future__ import annotations

import email
import fnmatch
import json
import re
import shutil
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import Mapping

from ports._support.native_adapters import (
    NativeAdapter,
    NativeBuildContext,
    NativeBuildRequest,
    build_native,
    write_build_file,
)
from ports._support.python_adapters import CPythonBuildContext, PythonBuildOutput, _source_file, _wheel
from ports._support.wasm import mark_abi
from ports._support.wasm_metadata import validate_provider_signatures
from ports.native.dependencies import file_hash

_MAX_FILES = 16384
_MAX_BYTES = 256 * 1024**2


@dataclass(frozen=True)
class PythonMesonBuildRequest:
    context: NativeBuildContext
    cpython: CPythonBuildContext
    recipe: Mapping
    host_packages: Mapping[str, Path] = field(default_factory=dict)


def wheel_destination(destination: str) -> str:
    """Admit a package path from Meson's target Python install placeholders."""
    for prefix in ("{py_platlib}/", "{py_purelib}/", "/usr/local/lib/python3.13/site-packages/"):
        if destination.startswith(prefix):
            value = destination[len(prefix) :]
            path = PurePosixPath(value)
            if not value or path.is_absolute() or ".." in path.parts or path.as_posix() != value:
                raise ValueError("Meson Python install destination escapes wheel")
            return value
    raise ValueError("Meson Python install destination is outside target site-packages")


def extension_destination(destination: str) -> str:
    """Remove a host SOABI suffix while preserving the qualified module name."""
    path = PurePosixPath(destination)
    match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)(?:\.cpython-[A-Za-z0-9_-]+)?\.so", path.name)
    if match is None or any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", part) for part in path.parent.parts):
        raise ValueError("invalid qualified Python extension destination")
    return str(path.with_name(match[1] + ".so"))


def _admitted_install_source(raw: str, source: Path, build: Path) -> Path:
    path = Path(raw)
    if (
        not path.is_absolute()
        or path.is_symlink()
        or not any(path.resolve().is_relative_to(root.resolve()) for root in (source, build))
    ):
        raise ValueError("Meson install source escapes admitted project")
    return path


def stage_install_plan(plan: Mapping, source: Path, build: Path, wheel: Path, tags: tuple[str, ...]) -> list[Path]:
    """Copy bounded upstream files, honoring its install exclusions and tags."""
    extensions = []
    count = total = 0

    def copy(path: Path, relative: str) -> None:
        nonlocal count, total
        path = _admitted_install_source(str(path), source, build)
        if not path.is_file():
            raise ValueError("Meson installed file is missing")
        count += 1
        total += path.stat().st_size
        if count > _MAX_FILES or total > _MAX_BYTES:
            raise ValueError("Python project install exceeds wheel bounds")
        destination = wheel / relative
        if destination.exists():
            raise ValueError("Meson install paths overlap")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)

    for section, entries in plan.items():
        for raw, spec in entries.items():
            if spec.get("tag") not in tags:
                continue
            # Upstream may install independent C development data outside Python.
            if not spec["destination"].startswith(
                ("{py_platlib}/", "{py_purelib}/", "/usr/local/lib/python3.13/site-packages/")
            ):
                continue
            relative = wheel_destination(spec["destination"])
            path = _admitted_install_source(raw, source, build)
            if section == "install_subdirs":
                excluded_dirs = spec.get("exclude_dirs", [])
                excluded_files = spec.get("exclude_files", [])
                for item in sorted(path.rglob("*")):
                    suffix = item.relative_to(path).as_posix()
                    if "__pycache__" in item.parts or any(
                        suffix == name or suffix.startswith(name + "/") for name in excluded_dirs
                    ):
                        continue
                    if item.is_dir():
                        continue
                    if any(fnmatch.fnmatchcase(suffix, pattern) for pattern in excluded_files):
                        continue
                    copy(item, relative + "/" + suffix)
            else:
                if path.suffix == ".so":
                    relative = extension_destination(relative)
                    extensions.append(wheel / relative)
                copy(path, relative)
    return extensions


def _target_pkg_config(request: PythonMesonBuildRequest, directory: Path) -> Path:
    context, python = request.context, request.cpython
    providers = {
        "python-3.13": ("3.13", [str(python.include_dir), str(python.generated_config_dir)]),
        "python-3.13-embed": ("3.13", [str(python.include_dir), str(python.generated_config_dir)]),
        "python3": ("3.13", [str(python.include_dir), str(python.generated_config_dir)]),
        "python3-embed": ("3.13", [str(python.include_dir), str(python.generated_config_dir)]),
    }
    for name, spec in request.recipe["build"].get("host_header_packages", {}).items():
        root = request.host_packages[spec["tool"]]
        include = root / spec["include"]
        if not include.resolve().is_relative_to(root.resolve()) or not include.is_dir():
            raise ValueError("host header package escapes admitted receipt")
        providers[name] = (spec["version"], [str(include)])
    wrapper = directory / "target-pkg-config"
    write_build_file(
        wrapper,
        "#!"
        + str(context.host_tools["python"])
        + "\nimport os, shlex, sys\n"
        + "providers = "
        + repr(providers)
        + "\n"
        + "arguments = sys.argv[1:]\nmatched = [name for name in arguments if name in providers]\n"
        + "if matched:\n"
        + " if len(matched) != 1: raise SystemExit('mixed target Python pkg-config query')\n"
        + " version, headers = providers[matched[0]]\n"
        + " if '--modversion' in arguments: print(version)\n"
        + " elif '--cflags' in arguments or '--cflags-only-I' in arguments: print(' '.join(shlex.quote('-I'+path) for path in headers))\n"
        + " elif '--libs' in arguments or '--exists' in arguments: pass\n"
        + " elif '--variable=prefix' in arguments: print('/usr/local')\n"
        + " elif '--variable=includes' in arguments: print(' '.join(headers))\n"
        + " elif any(arg.startswith('--variable=') for arg in arguments): raise SystemExit('unsupported target Python pkg-config variable')\n"
        + " raise SystemExit(0)\n"
        + "os.execv("
        + repr(str(context.host_tools["pkg-config"]))
        + ", ["
        + repr(str(context.host_tools["pkg-config"]))
        + "] + arguments)\n",
    )
    wrapper.chmod(0o755)
    return wrapper


def stage_development_exports(wheel: Path, staging: Path, exports: list[Mapping]) -> None:
    """Publish declared development files through native admission, not as modules.

    Curated wheel native entries are executable Wasm modules. Upstream static
    archives belong to the independently sealed development payload.
    """
    archives = set()
    for export in exports:
        relative = wheel_destination("{py_platlib}/" + export["source"])
        source = wheel / relative
        destination = staging / "usr/local" / export["destination"]
        if not destination.resolve().is_relative_to((staging / "usr/local").resolve()):
            raise ValueError("Python development export escapes staging prefix")
        if source.is_dir():
            shutil.copytree(source, destination, dirs_exist_ok=True)
            archives.update(source.rglob("*.a"))
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            if source.suffix == ".a":
                archives.add(source)
    if set(wheel.rglob("*.a")) != archives:
        raise ValueError("Python wheel contains an undeclared development archive")
    for archive in archives:
        archive.unlink()


def shared_providers(dependencies: Mapping[str, Path]) -> dict[str, Path]:
    """Index admitted providers once per actual prefix, rejecting name conflicts.

    A retained dependency closure exposes one verified merged prefix through
    several logical edges. Those edges share an identity. Distinct prefixes
    remain distinct providers, even when their file bytes happen to agree.
    """
    providers = {}
    for prefix in dict.fromkeys(path.resolve() for path in dependencies.values()):
        for path in prefix.rglob("*.so"):
            if path.name in providers:
                raise ValueError("declared native providers have duplicate library names")
            providers[path.name] = path
    return providers


def build_python_meson(request: PythonMesonBuildRequest) -> PythonBuildOutput:
    """Build shared extensions, retain upstream metadata, and stage devel exports."""
    context, python, recipe = request.context, request.cpython, request.recipe
    build = recipe["build"]
    if python.abi != recipe["abi"] or python.runtime_target != context.target or recipe["target"] != context.target:
        raise ValueError("Python Meson target differs from admitted interpreter")
    if not context.shared_library_flags:
        raise ValueError("Python Meson requires admitted side-module link flags")
    if "cython" not in context.host_tools:
        raise ValueError("Python Meson requires admitted Cython")
    metadata = _source_file(context.source, build.get("metadata", "PKG-INFO"))
    message = email.message_from_bytes(metadata.read_bytes())
    if message["Name"] != recipe["name"] or message["Version"] != recipe["version"]:
        raise ValueError("Python project metadata differs from recipe")
    if sorted(message.get_all("Requires-Dist", [])) != sorted(recipe.get("requires_dist", [])):
        raise ValueError("Python project dependency constraints differ from recipe")
    if context.staging_prefix.exists() or (context.build.exists() and not context.retained_workspace):
        raise ValueError("Python Meson build output already exists")
    context.build.mkdir(parents=True, exist_ok=True)
    package_config = _target_pkg_config(request, context.build)
    properties = dict(build.get("cross_properties", {}))
    for name, spec in build.get("dependency_properties", {}).items():
        root = context.dependencies[spec["port"]]
        path = root / spec["path"]
        if not path.resolve().is_relative_to(root.resolve()) or not path.is_dir():
            raise ValueError("Meson dependency property escapes admitted export")
        properties[name] = str(path)
    tools = {**context.host_tools, "pkg-config": package_config}
    tags = tuple(build.get("install_tags", ["runtime", "python-runtime", "devel"]))
    output = build_native(
        NativeBuildRequest(
            NativeAdapter.MESON,
            replace(context, host_tools=tools),
            configure_args=tuple(build.get("configure_args", []))
            + (
                "-Dpython.bytecompile=-1",
                "-Dpython.platlibdir=/usr/local/lib/python3.13/site-packages",
                "-Dpython.purelibdir=/usr/local/lib/python3.13/site-packages",
                "-Db_lundef=false",
            ),
            jobs=build.get("jobs", 2),
            meson_properties=properties,
            meson_install_tags=tags,
        )
    )
    started = time.perf_counter()
    meson_build = context.build / "meson-build"
    plan = json.loads((meson_build / "meson-info/intro-install_plan.json").read_text())
    wheel = context.staging_prefix.parent / "wheel-root"
    wheel.mkdir()
    extensions = stage_install_plan(plan, context.source, meson_build, wheel, tags)
    if not extensions:
        raise ValueError("Python Meson install contains no extension modules")
    providers = shared_providers(context.dependencies)
    required = build.get("required_shared_libraries", [])
    if not isinstance(required, list) or any(not isinstance(name, str) or name not in providers for name in required):
        raise ValueError("required shared libraries must name admitted target providers")
    referenced = set()
    artifacts = []
    for extension in sorted(extensions):
        mark_abi(extension, recipe["abi"].encode())
        dependencies = validate_provider_signatures(extension, providers)
        referenced.update(dependencies)
        artifacts.append(
            {
                "path": extension.relative_to(wheel).as_posix(),
                "sha256": file_hash(extension),
                "native_dependencies": dependencies,
            }
        )
    if not set(required) <= referenced:
        raise ValueError("Python extensions do not reference required shared providers")
    dist = recipe["name"].replace("-", "_") + "-" + recipe["version"] + ".dist-info"
    info = wheel / dist
    info.mkdir(exist_ok=True)
    (info / "METADATA").write_bytes(metadata.read_bytes())
    (info / "WHEEL").write_text(
        "Wheel-Version: 1.0\nGenerator: shellsim-python-meson\nRoot-Is-Purelib: false\nTag: cp313-cp313-wasm32_wasip1\n"
    )
    for raw in build.get("licenses", []):
        path = _source_file(context.source, raw)
        destination = info / "licenses" / raw
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    manifest = {
        "schema_version": 1,
        "name": recipe["name"],
        "version": recipe["version"],
        "abi": recipe["abi"],
        "recipe": dict(recipe),
        "cpython_receipt": dict(python.receipt),
        "artifacts": artifacts,
    }
    (info / "shellsim-native.json").write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n")
    stage_development_exports(wheel, context.staging_prefix, build.get("development_exports", []))
    wheels = context.staging_prefix / "wheels"
    wheels.mkdir()
    destination = wheels / (dist.removesuffix(".dist-info") + "-cp313-cp313-wasm32_wasip1.whl")
    _wheel(wheel, destination, dist + "/RECORD")
    print(f"ports: Python Meson package {time.perf_counter() - started:.2f}s", file=sys.stderr, flush=True)
    return PythonBuildOutput(context.staging_prefix, output.commands)
