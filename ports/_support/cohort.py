"""Admit an explicit host build cohort without ambient compiler discovery.

A setup descriptor binds SDK, compiler, sysroot and CPython receipts. Target
artifacts and headers are verified before paths become available to adapters.
The descriptor is a trusted host setup input, never a guest-controlled path.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from ports.toolchain.wasi_threads.dynamic import verify_sdk

TARGET = "wasm32-wasip1-threads"
ABI = "shellsim-wasi-sdk34-cpython3137-threads-v3"
HOST_TOOLS = frozenset({"cmake", "ninja", "make", "python", "meson", "pkg-config", "sh", "rm", "uv"})


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_json(path: Path) -> dict:
    if path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError("build receipt exceeds the metadata bound")
    return json.loads(path.read_text())


def child(root: Path, name: str) -> Path:
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("artifact receipt contains an escaping path")
    result = root / path
    if not result.resolve().is_relative_to(root.resolve()):
        raise ValueError("artifact path escapes its admitted root")
    return result


@dataclass(frozen=True)
class Tool:
    """A host executable admitted by bytes, rather than PATH lookup."""

    path: Path
    sha256: str
    receipt_path: Path | None = None
    receipt_sha256: str | None = None


@dataclass(frozen=True)
class Receipt:
    root: Path
    path: Path
    sha256: str
    contents: dict


@dataclass(frozen=True)
class CPythonReceipt:
    source_root: Path
    generated_config_dir: Path
    runtime_bundle: Path
    version: str
    dynamic_abi: str
    runtime_target: str
    wheel_platform: str
    source_sha256: str
    headers_sha256: str
    pyconfig_sha256: str
    runtime_manifest_sha256: str


@dataclass(frozen=True)
class BuildCohort:
    """Verified build inputs for a single native ABI cohort."""

    sdk: Receipt
    llvm: Receipt
    sysroot: Receipt
    cpython_manifest: Receipt | None
    runtime: Receipt | None
    python: CPythonReceipt | None
    host_tools: Mapping[str, Tool]
    target_tools: Mapping[str, Tool]
    identity: str
    has_frontend: bool

    @property
    def target(self) -> str:
        return TARGET

    @property
    def dynamic_abi(self) -> str:
        return ABI

    @property
    def toolchain_receipt(self) -> dict[str, str]:
        """Identify the actual compiler and platform products used by a node."""
        return {
            "cohort": self.identity,
            "target": self.target,
            "abi": self.dynamic_abi,
            "compiler": self.llvm.sha256,
            "platform": self.sysroot.sha256,
        }

    def compiler(self, *, cxx: bool = False) -> Path:
        if not self.has_frontend:
            raise ValueError("this cohort has no standard patched Clang frontend")
        return self.target_tools["cxx" if cxx else "cc"].path

    @property
    def flags(self) -> tuple[str, ...]:
        return (
            "--target=" + TARGET,
            "-pthread",
            "--sysroot=" + str(self.sysroot.root / "sysroot"),
            "-resource-dir=" + str(self.sdk.root / "lib/clang/23"),
        )

    @property
    def compiler_flags(self) -> tuple[str, ...]:
        if not self.has_frontend:
            raise ValueError("native builds require the standard patched Clang frontend")
        return (
            *self.flags,
            "-fPIC",
            "-fwasm-exceptions",
            "-mllvm",
            "-wasm-enable-wasi-dynamic-tls",
            "-mllvm",
            "-wasm-enable-sjlj",
            "-mllvm",
            "-wasm-use-legacy-eh=false",
        )

    @property
    def linker_flags(self) -> tuple[str, ...]:
        """Target link flags that also permit ordinary configure executables."""
        return (
            *self.flags,
            "-fuse-ld=" + str(self.llvm.root / "bin/wasm-ld"),
            "-Wl,--shared-memory,--serial-memory-init",
        )

    @property
    def executable_flags(self) -> tuple[str, ...]:
        """Process-owned memory and TLS exports for threaded guest executables."""
        return (
            "-Wl,--import-memory,--export-memory,--initial-memory=16777216,--max-memory=67108864",
            "-Wl,--export-all,--export-table,--growable-table,--export=__stack_pointer,--export=__tls_base",
            "-Wl,--emit-main-tls-info,--undefined=pthread_create",
        )

    @property
    def shared_library_flags(self) -> tuple[str, ...]:
        """Additional flags only for independently loaded native libraries."""
        return (
            "-shared",
            "-nostdlib",
            "-Wl,--shared-memory,--serial-memory-init,--defer-shared-init",
            "-Wl,--import-memory,--import-table,--export-all,--no-entry,--unresolved-symbols=import-dynamic",
        )

    def tool(self, name: str) -> Path:
        return self.host_tools[name].path


def receipt(base: Path, value: dict) -> Receipt:
    if set(value) != {"root", "manifest", "sha256"}:
        raise ValueError("cohort receipt fields differ")
    root = (base / value["root"]).resolve()
    path = (base / value["manifest"]).resolve()
    if file_hash(path) != value["sha256"]:
        raise ValueError("cohort receipt digest differs: " + str(path))
    return Receipt(root, path, value["sha256"], read_json(path))


def verify_files(root: Path, hashes: dict[str, str], *, prefix: str = "") -> None:
    for name, expected in hashes.items():
        if file_hash(child(root, prefix + name)) != expected:
            raise ValueError("cohort artifact differs: " + name)


def verify_product(product: Receipt) -> None:
    manifest = product.contents
    verify_files(product.root, manifest["artifacts"])
    aliases = dict(manifest.get("symlinks", {}))
    if manifest["identity"]["recipe"]["name"] == "llvm-wasi-threaded":
        aliases["bin/wasm-ld"] = "lld"
    for name, target in aliases.items():
        path = child(product.root, name)
        if not path.is_symlink() or os.readlink(path) != target:
            raise ValueError("cohort tool alias differs: " + name)
    actual_aliases = {str(path.relative_to(product.root)) for path in product.root.rglob("*") if path.is_symlink()}
    if actual_aliases != set(aliases):
        raise ValueError("cohort product contains undeclared tool aliases")
    actual = {
        str(path.relative_to(product.root))
        for path in product.root.rglob("*")
        if path.is_file() and not path.is_symlink() and path != product.path
    }
    if actual != set(manifest["artifacts"]):
        raise ValueError("cohort product contains unrecorded or missing files")


def local_recipe(name: str) -> dict:
    return json.loads((Path(__file__).resolve().parents[1] / name).read_text())


def verify_cpython_recipe(recipe: dict) -> None:
    """Admit historical build provenance while retaining the Python source policy.

    The trusted cohort descriptor pins the consumed manifest, including its
    historical driver hashes. Current Python driver and JSON metadata pins
    govern new builds. Compiled facade sources, headers and patches must still
    match, along with every other source and runtime ABI field.
    """
    current = local_recipe("python/cpython/threaded-recipe.json")
    scripts = recipe.get("build_scripts")
    if (
        set(recipe) != set(current)
        or not isinstance(scripts, list)
        or not 1 <= len(scripts) <= 256
        or any(
            not isinstance(item, dict)
            or set(item) != {"file", "sha256"}
            or not isinstance(item["file"], str)
            or not item["file"]
            or not isinstance(item["sha256"], str)
            or len(item["sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in item["sha256"])
            for item in scripts
        )
        or {key: value for key, value in recipe.items() if key != "build_scripts"}
        != {key: value for key, value in current.items() if key != "build_scripts"}
    ):
        raise ValueError("CPython producer source or ABI profile differs")
    historical = {item["file"]: item["sha256"] for item in scripts}
    expected = {item["file"]: item["sha256"] for item in current["build_scripts"]}
    if len(historical) != len(scripts) or historical.keys() != expected.keys():
        raise ValueError("CPython producer input declarations differ")
    if any(value != expected[name] and Path(name).suffix not in {".py", ".json"} for name, value in historical.items()):
        raise ValueError("CPython compiled facade source differs")


def verify_host_files(proof_path: Path, producer: dict, name: str, executable: Path) -> None:
    """Bind a package-backed host tool to its complete immutable code tree."""
    if set(producer) != {"schema_version", "kind", "name", "version", "root", "executable", "source", "files"}:
        raise ValueError("host package receipt fields differ")
    if producer["schema_version"] != 1 or producer["kind"] != "host-tool-files" or producer["name"] != name:
        raise ValueError("host package receipt profile differs")
    root = (proof_path.parent / producer["root"]).resolve()
    if child(root, producer["executable"]).resolve() != executable.resolve():
        raise ValueError("host package entrypoint differs")
    verify_files(root, producer["files"])
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    if actual != set(producer["files"]):
        raise ValueError("host package contains unrecorded or missing files")


def load_cohort(path: Path, *, expected_sha256: str | None = None) -> BuildCohort:
    """Verify a setup descriptor and every consumed target receipt and header.

    Relative roots resolve against the explicitly supplied setup descriptor.
    Local setup may reference external immutable build directories. Exported
    target cohorts use only contained roots; host tools remain explicit bindings.
    """
    path = path.resolve()
    if expected_sha256 is not None and file_hash(path) != expected_sha256:
        raise ValueError("build cohort descriptor digest differs")
    value = read_json(path)
    if set(value) != {
        "schema_version",
        "target",
        "dynamic_abi",
        "sdk",
        "llvm",
        "sysroot",
        "cpython",
        "runtime",
        "host_tools",
        "target_tools",
    }:
        raise ValueError("build cohort descriptor fields differ")
    if value["schema_version"] != 1 or value["target"] != TARGET or value["dynamic_abi"] != ABI:
        raise ValueError("unsupported build cohort profile")
    sdk, llvm, sysroot = (receipt(path.parent, value[name]) for name in ("sdk", "llvm", "sysroot"))
    cpython = receipt(path.parent, value["cpython"]) if value["cpython"] is not None else None
    runtime = receipt(path.parent, value["runtime"]) if value["runtime"] is not None else None
    overlay = sysroot.contents
    if overlay["identity"]["recipe"] != local_recipe("toolchain/wasi_threads/dynamic-recipe.json"):
        raise ValueError("sysroot producer profile differs")
    verify_sdk(sdk.root, overlay)
    if sdk.contents != overlay["identity"]["sdk_tooling"]:
        raise ValueError("SDK receipt differs from the sysroot input")
    compiler_recipe = llvm.contents["identity"]["recipe"]
    frontend = compiler_recipe.get("name") == "llvm-wasi-compiler"
    expected_recipe = "compiler-recipe.json" if frontend else "threaded-recipe.json"
    if compiler_recipe != local_recipe("toolchain/llvm/" + expected_recipe):
        raise ValueError("compiler producer profile differs")
    built_compiler = overlay["identity"]["compiler"]["identity"]["recipe"]
    if any(compiler_recipe[name] != built_compiler[name] for name in ("source", "patches")):
        raise ValueError("frontend and sysroot compiler source profiles differ")
    verify_product(llvm)
    if not frontend:
        alias = llvm.root / "bin/wasm-ld"
        if not alias.is_symlink() or os.readlink(alias) != "lld":
            raise ValueError("linker alias differs")
    verify_product(sysroot)
    if cpython is None and runtime is not None:
        raise ValueError("runtime admission requires its CPython build receipt")
    if cpython is not None:
        manifest = cpython.contents
        verify_cpython_recipe(manifest["recipe"])
        if manifest["dynamic_abi"] != ABI or manifest["recipe"]["target"] != TARGET:
            raise ValueError("CPython runtime ABI differs")
        profile = manifest["build_profile"]
        if profile["sysroot"] != overlay or json_hash(profile) != manifest["build_profile_sha256"]:
            raise ValueError("CPython build profile differs")
        rootfs = cpython.root / "rootfs"
        verify_files(rootfs, {name.lstrip("/"): digest for name, digest in manifest["files"].items()})
        source = cpython.root / ("Python-" + manifest["recipe"]["version"])
        generated = cpython.root / "wasi-build"
        headers = profile["headers"]
        actual_headers = {
            str(p.relative_to(cpython.root)): file_hash(p) for p in sorted((source / "Include").rglob("*.h"))
        }
        actual_headers["wasi-build/pyconfig.h"] = file_hash(generated / "pyconfig.h")
        if headers != actual_headers:
            raise ValueError("CPython header closure differs")
        if runtime is None:
            raise ValueError("Python cohorts require a separately admitted runtime bundle")
        runtime_manifest = runtime.contents
        if (
            runtime_manifest["dynamic_abi"] != ABI
            or runtime_manifest["recipe"] != manifest["recipe"]
            or runtime_manifest["build_profile_sha256"] != manifest["build_profile_sha256"]
            or runtime_manifest["build_profile"] != profile
            or runtime_manifest["files"]["/usr/bin/python3.wasm"] != manifest["files"]["/usr/bin/python3.wasm"]
        ):
            raise ValueError("assembled runtime differs from the admitted CPython build")
        verify_files(
            runtime.root / "rootfs", {name.lstrip("/"): digest for name, digest in runtime_manifest["files"].items()}
        )
    tools = {}
    if not set(value["host_tools"]) <= HOST_TOOLS:
        raise ValueError("unknown cohort host tool")
    for name, item in value["host_tools"].items():
        if set(item) != {"path", "sha256", "receipt"}:
            raise ValueError("host tool receipt fields differ")
        tool_path = Path(os.path.abspath(path.parent / item["path"]))
        if file_hash(tool_path) != item["sha256"]:
            raise ValueError("cohort host tool differs: " + name)
        proof = item["receipt"]
        proof_path = None
        proof_hash = None
        if proof is not None:
            if set(proof) != {"path", "sha256"}:
                raise ValueError("host producer receipt fields differ")
            proof_path = (path.parent / proof["path"]).resolve()
            proof_hash = proof["sha256"]
            if file_hash(proof_path) != proof_hash:
                raise ValueError("host producer receipt digest differs")
            producer = read_json(proof_path)
            if name == "uv":
                if producer["recipe"] != local_recipe("toolchain/uv/recipe.json"):
                    raise ValueError("patched uv producer differs")
                if producer["executable"]["sha256"] != item["sha256"] or not producer["build"]["locked"]:
                    raise ValueError("patched uv executable differs")
            else:
                verify_host_files(proof_path, producer, name, tool_path)
        elif name == "uv":
            raise ValueError("patched uv requires its production receipt")
        tools[name] = Tool(tool_path, item["sha256"], proof_path, proof_hash)
    target_tools = {}
    if set(value["target_tools"]) != {"cc", "cxx", "ar", "ranlib", "strip"}:
        raise ValueError("target tool entrypoints differ")
    providers = {"sdk": sdk, "llvm": llvm}
    expected_entries = {
        "cc": "bin/clang",
        "cxx": "bin/clang++",
        "ar": "bin/llvm-ar",
        "ranlib": "bin/llvm-ranlib",
        "strip": "bin/llvm-strip",
    }
    for name, item in value["target_tools"].items():
        if set(item) != {"provider", "path", "sha256"} or item["provider"] not in providers:
            raise ValueError("target tool receipt fields differ")
        if item["path"] != expected_entries[name]:
            raise ValueError("target executable role differs: " + name)
        provider = providers[item["provider"]]
        if frontend and name in {"cc", "cxx"} and item["provider"] != "llvm":
            raise ValueError("standard frontend must come from the admitted compiler")
        admitted = provider.contents if item["provider"] == "sdk" else provider.contents["artifacts"]
        tool_path = child(provider.root, item["path"])
        if item["provider"] == "llvm" and item["path"] in provider.contents.get("symlinks", {}):
            relative = str(tool_path.resolve().relative_to(provider.root))
        else:
            relative = item["path"]
        if admitted.get(relative) != item["sha256"] or file_hash(tool_path) != item["sha256"]:
            raise ValueError("target tool is not in its admitted product: " + name)
        target_tools[name] = Tool(tool_path, item["sha256"])
    python = None
    if cpython is not None:
        python = CPythonReceipt(
            source,
            generated,
            runtime.root,
            manifest["recipe"]["version"],
            ABI,
            TARGET,
            "wasm32_wasip1",
            manifest["recipe"]["source"]["sha256"],
            json_hash(headers),
            headers["wasi-build/pyconfig.h"],
            runtime.sha256,
        )
    return BuildCohort(
        sdk,
        llvm,
        sysroot,
        cpython,
        runtime,
        python,
        MappingProxyType(tools),
        MappingProxyType(target_tools),
        json_hash(value),
        frontend,
    )


def tool_reference(tool: Tool) -> dict:
    return {
        "path": str(Path(os.path.abspath(tool.path))),
        "sha256": tool.sha256,
        "receipt": {"path": str(tool.receipt_path.resolve()), "sha256": tool.receipt_sha256}
        if tool.receipt_path is not None
        else None,
    }


def write_setup(
    path: Path,
    *,
    sdk: Path,
    llvm: Path,
    sysroot: Path,
    cpython: Path | None,
    runtime: Path | None,
    host_tools: Mapping[str, Tool],
) -> BuildCohort:
    """Write and admit an explicit local setup from immutable producer outputs.

    The setup references external build directories. Use ``export_cohort`` for
    contained target inputs. Host tools are always explicit byte-pinned bindings.
    """
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    overlay = read_json(sysroot / "manifest.json")
    sdk_receipt = path.with_name(path.stem + "-sdk.json")
    sdk_receipt.write_text(json.dumps(overlay["identity"]["sdk_tooling"], indent=2, sort_keys=True) + "\n")

    def reference(root: Path, manifest: Path) -> dict:
        return {"root": str(root.resolve()), "manifest": str(manifest.resolve()), "sha256": file_hash(manifest)}

    compiler_manifest = read_json(llvm / "manifest.json")
    frontend = compiler_manifest["identity"]["recipe"].get("name") == "llvm-wasi-compiler"
    entries = {}
    for name, executable in {
        "cc": "clang",
        "cxx": "clang++",
        "ar": "llvm-ar",
        "ranlib": "llvm-ranlib",
        "strip": "llvm-strip",
    }.items():
        provider = "llvm" if frontend else "sdk"
        root = llvm if frontend else sdk
        entries[name] = {
            "provider": provider,
            "path": "bin/" + executable,
            "sha256": file_hash(root / "bin" / executable),
        }
    value = {
        "schema_version": 1,
        "target": TARGET,
        "dynamic_abi": ABI,
        "sdk": reference(sdk, sdk_receipt),
        "llvm": reference(llvm, llvm / "manifest.json"),
        "sysroot": reference(sysroot, sysroot / "manifest.json"),
        "cpython": reference(cpython, cpython / "manifest.json") if cpython is not None else None,
        "runtime": reference(runtime, runtime / "manifest.json") if runtime is not None else None,
        "host_tools": {name: tool_reference(tool) for name, tool in host_tools.items()},
        "target_tools": entries,
    }
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return load_cohort(path)


def export_cohort(cohort: BuildCohort, destination: Path) -> Path:
    """Copy admitted target inputs into a relocatable setup directory.

    Host executables remain explicit external bindings: copying a binary alone
    would not package its host interpreter, shared libraries or Python environment.
    The target roots and receipts are contained and can move as one directory.
    """
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    items = {
        "sdk": cohort.sdk,
        "llvm": cohort.llvm,
        "sysroot": cohort.sysroot,
        "cpython": cohort.cpython_manifest,
        "runtime": cohort.runtime,
    }
    references = {}
    for name, product in items.items():
        if product is None:
            references[name] = None
            continue
        output = destination / name
        if name == "cpython":
            output.mkdir()
            shutil.copytree(cohort.python.source_root / "Include", output / cohort.python.source_root.name / "Include")
            (output / "wasi-build").mkdir()
            shutil.copyfile(cohort.python.generated_config_dir / "pyconfig.h", output / "wasi-build/pyconfig.h")
            shutil.copytree(product.root / "rootfs", output / "rootfs")
        else:
            shutil.copytree(product.root, output, symlinks=True)
        manifest_name = "sdk-receipt.json" if name == "sdk" else "manifest.json"
        shutil.copyfile(product.path, output / manifest_name)
        references[name] = {"root": name, "manifest": name + "/" + manifest_name, "sha256": product.sha256}
    value = {
        "schema_version": 1,
        "target": TARGET,
        "dynamic_abi": ABI,
        **references,
        "host_tools": {name: tool_reference(tool) for name, tool in cohort.host_tools.items()},
        "target_tools": {
            name: {
                "provider": "llvm" if tool.path.is_relative_to(cohort.llvm.root) else "sdk",
                "path": str(
                    tool.path.relative_to(
                        cohort.llvm.root if tool.path.is_relative_to(cohort.llvm.root) else cohort.sdk.root
                    )
                ),
                "sha256": tool.sha256,
            }
            for name, tool in cohort.target_tools.items()
        },
    }
    path = destination / "cohort.json"
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    load_cohort(path)
    return path


@dataclass(frozen=True)
class CompilerBootstrap:
    """Explicit seed inputs for the host LLVM producer, outside the guest graph."""

    archive: Path
    archive_sha256: str
    work: Path
    tools: Mapping[str, Tool]


def load_bootstrap(path: Path) -> CompilerBootstrap:
    """Admit pinned host compilers and build tools without searching PATH."""
    value = read_json(path)
    if set(value) != {"schema_version", "archive", "work", "tools"} or value["schema_version"] != 1:
        raise ValueError("compiler bootstrap descriptor fields differ")
    if set(value["archive"]) != {"path", "sha256"} or set(value["tools"]) != {"cc", "cxx", "cmake", "ninja"}:
        raise ValueError("compiler bootstrap inputs differ")
    archive = (path.parent / value["archive"]["path"]).resolve()
    if file_hash(archive) != value["archive"]["sha256"]:
        raise ValueError("compiler bootstrap archive differs")
    tools = {}
    for name, item in value["tools"].items():
        if set(item) != {"path", "sha256"}:
            raise ValueError("compiler bootstrap tool fields differ")
        executable = (path.parent / item["path"]).resolve()
        if file_hash(executable) != item["sha256"]:
            raise ValueError("compiler bootstrap tool differs: " + name)
        tools[name] = Tool(executable, item["sha256"])
    return CompilerBootstrap(
        archive, value["archive"]["sha256"], (path.parent / value["work"]).resolve(), MappingProxyType(tools)
    )


def resolve_compiler(cohort: BuildCohort, bootstrap: CompilerBootstrap | None) -> Receipt:
    """Run the host producer when seeded, or reuse its admitted immutable product."""
    from ports.toolchain.llvm.compiler import build
    from ports.toolchain.llvm.compiler import verify_product as verify_compiler

    if bootstrap is None:
        cohort.compiler()
        verify_compiler(cohort.llvm.root, cohort.llvm.contents["identity"])
        return cohort.llvm
    if file_hash(bootstrap.archive) != bootstrap.archive_sha256:
        raise ValueError("compiler bootstrap archive changed after admission")
    for name, tool in bootstrap.tools.items():
        if file_hash(tool.path) != tool.sha256:
            raise ValueError("compiler bootstrap tool changed after admission: " + name)
    root = build(
        bootstrap.archive, *(bootstrap.tools[name].path for name in ("cc", "cxx", "cmake", "ninja")), bootstrap.work
    )
    path = root / "manifest.json"
    return Receipt(root, path, file_hash(path), read_json(path))


def resolved_toolchain(cohort: BuildCohort, compiler: Receipt, sysroot: Receipt) -> BuildCohort:
    """Bind adapter flags and entrypoints to the graph's selected producer results."""
    from dataclasses import replace

    if compiler.contents["identity"]["recipe"] != local_recipe("toolchain/llvm/compiler-recipe.json"):
        raise ValueError("resolved compiler producer differs")
    if sysroot.contents["identity"]["recipe"] != local_recipe("toolchain/wasi_threads/dynamic-recipe.json"):
        raise ValueError("resolved platform producer differs")
    target_tools = {
        name: Tool(compiler.root / "bin" / executable, file_hash(compiler.root / "bin" / executable))
        for name, executable in {
            "cc": "clang",
            "cxx": "clang++",
            "ar": "llvm-ar",
            "ranlib": "llvm-ranlib",
            "strip": "llvm-strip",
        }.items()
    }
    identity = json_hash({"cohort": cohort.identity, "compiler": compiler.sha256, "platform": sysroot.sha256})
    return replace(
        cohort,
        llvm=compiler,
        sysroot=sysroot,
        target_tools=MappingProxyType(target_tools),
        identity=identity,
        has_frontend=True,
    )


@dataclass(frozen=True)
class PlatformBootstrap:
    """Pinned archives and output slot for the WASI libc platform producer."""

    sdk_archive: Path
    sdk_sha256: str
    libc_archive: Path
    libc_sha256: str
    work: Path


def load_platform_bootstrap(path: Path) -> PlatformBootstrap:
    value = read_json(path)
    if set(value) != {"schema_version", "sdk_archive", "libc_archive", "work"} or value["schema_version"] != 1:
        raise ValueError("platform bootstrap descriptor fields differ")
    archives = {}
    for name in ("sdk_archive", "libc_archive"):
        item = value[name]
        if set(item) != {"path", "sha256"}:
            raise ValueError("platform archive fields differ")
        archive = (path.parent / item["path"]).resolve()
        if file_hash(archive) != item["sha256"]:
            raise ValueError("platform bootstrap archive differs")
        archives[name] = (archive, item["sha256"])
    return PlatformBootstrap(
        *archives["sdk_archive"], *archives["libc_archive"], (path.parent / value["work"]).resolve()
    )


def resolve_platform(cohort: BuildCohort, compiler: Receipt, bootstrap: PlatformBootstrap | None) -> Receipt:
    """Run the pinned libc producer or reuse a complete byte-verified product."""
    from ports.toolchain.wasi_threads.dynamic import build, compiler_identity

    if bootstrap is None:
        verify_product(cohort.sysroot)
        return cohort.sysroot
    recipe = local_recipe("toolchain/wasi_threads/dynamic-recipe.json")
    for path, expected, declared in (
        (bootstrap.sdk_archive, bootstrap.sdk_sha256, recipe["sdk"]["sha256"]),
        (bootstrap.libc_archive, bootstrap.libc_sha256, recipe["wasi_libc"]["sha256"]),
    ):
        if expected != declared or file_hash(path) != expected:
            raise ValueError("platform producer archive differs")
    tools = [cohort.host_tools[name] for name in ("cmake", "ninja")]
    for tool in tools:
        if file_hash(tool.path) != tool.sha256:
            raise ValueError("platform build tool changed")
    root = bootstrap.work / "prefix"
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        product = Receipt(root, manifest_path, file_hash(manifest_path), read_json(manifest_path))
        expected_compiler = compiler_identity(
            compiler.root, Path(__file__).parents[1] / "toolchain/llvm/threaded-recipe.json"
        )
        if (
            product.contents["identity"]["recipe"] != recipe
            or product.contents["identity"]["compiler"] != expected_compiler
        ):
            raise ValueError("platform workspace producer inputs differ")
        if product.contents["identity"]["tools"] != {str(tool.path): tool.sha256 for tool in tools}:
            raise ValueError("platform workspace build tools differ")
        verify_sdk(cohort.sdk.root, product.contents)
        verify_product(product)
        return product
    root = build(
        bootstrap.sdk_archive, bootstrap.libc_archive, compiler.root, *(tool.path for tool in tools), bootstrap.work
    )
    manifest_path = root / "manifest.json"
    product = Receipt(root, manifest_path, file_hash(manifest_path), read_json(manifest_path))
    verify_product(product)
    return product
