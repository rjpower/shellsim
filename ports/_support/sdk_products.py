"""Typed SDK products and inventory admission at the host build boundary.

The graph materializes products before constructing this context. Product paths
come from pinned producers or a verified import; guest input cannot select them.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from ports.toolchain.runtime_profile import COMMON_COMPILER_FLAGS, host_executable_flags
from ports.toolchain.wasi_threads.dynamic import verify_sdk

HOST_TOOLS = frozenset(
    {
        "cmake",
        "ninja",
        "make",
        "python",
        "meson",
        "pkg-config",
        "sh",
        "rm",
        "uv",
        "cc",
        "cython",
        "pybind11-config",
        "f2py",
    }
)


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
class MaterializedSDK:
    """Concrete verified compiler, platform, Python and host tool products."""

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

    target: str
    dynamic_abi: str

    @property
    def toolchain_receipt(self) -> dict[str, object]:
        """Identify the actual compiler and platform products used by a node."""
        # The format-1 native artifact field is retained so immutable released
        # catalogs remain installable. It is an SDK product identity, not setup.
        return {
            "cohort": self.identity,
            "target": self.target,
            "abi": self.dynamic_abi,
            "compiler": self.llvm.sha256,
            "platform": self.sysroot.sha256,
            "sdk": self.sysroot.contents["identity"]["recipe"]["sdk"],
        }

    def compiler(self, *, cxx: bool = False) -> Path:
        if not self.has_frontend:
            raise ValueError("this SDK has no standard patched Clang frontend")
        return self.target_tools["cxx" if cxx else "cc"].path

    @property
    def compiler_resource_directory(self) -> Path:
        """Resource headers and compiler runtime are supplied by SDK tooling."""
        return self.sdk.root / "lib/clang/23"

    @property
    def linker(self) -> Path:
        """Patched host linker from the selected immutable compiler product."""
        return self.llvm.root / "bin/wasm-ld"

    @property
    def flags(self) -> tuple[str, ...]:
        return (
            "--target=" + self.target,
            "-pthread",
            "--sysroot=" + str(self.sysroot.root / "sysroot"),
            "-resource-dir=" + str(self.compiler_resource_directory),
        )

    @property
    def compiler_flags(self) -> tuple[str, ...]:
        if not self.has_frontend:
            raise ValueError("native builds require the standard patched Clang frontend")
        return (*self.flags, *COMMON_COMPILER_FLAGS)

    @property
    def linker_flags(self) -> tuple[str, ...]:
        """Target link flags that also permit ordinary configure executables."""
        return (
            *self.flags,
            "-fuse-ld=" + str(self.linker),
            "-Wl,--shared-memory,--serial-memory-init",
        )

    @property
    def executable_flags(self) -> tuple[str, ...]:
        """Process-owned memory and TLS exports for threaded guest executables."""
        return host_executable_flags(self.sysroot.root / "sysroot", self.target)

    @property
    def shared_library_flags(self) -> tuple[str, ...]:
        """Additional flags only for independently loaded native libraries."""
        return (
            "-shared",
            "-nostdlib",
            "-Wl,--shared-memory,--serial-memory-init,--defer-shared-init,--fatal-warnings",
            "-Wl,--import-memory,--import-table,--export-all,--no-entry,--unresolved-symbols=import-dynamic",
        )

    @property
    def compiler_runtime_archive(self) -> Path:
        """Use the exact SDK builtin archive, independently of guest libc state."""
        relative = (
            "lib/clang/23/lib/" + self.target.replace("wasm32-", "wasm32-unknown-", 1) + "/libclang_rt.builtins.a"
        )
        if relative not in self.sdk.contents:
            raise ValueError("SDK receipt omits target compiler runtime archive")
        return self.sdk.root / relative

    def tool(self, name: str) -> Path:
        return self.host_tools[name].path


def receipt(base: Path, value: dict) -> Receipt:
    if set(value) != {"root", "manifest", "sha256"}:
        raise ValueError("SDK receipt fields differ")
    root = (base / value["root"]).resolve()
    path = (base / value["manifest"]).resolve()
    if file_hash(path) != value["sha256"]:
        raise ValueError("SDK receipt digest differs: " + str(path))
    return Receipt(root, path, value["sha256"], read_json(path))


def verify_files(root: Path, hashes: dict[str, str], *, prefix: str = "") -> None:
    for name, expected in hashes.items():
        if file_hash(child(root, prefix + name)) != expected:
            raise ValueError("SDK artifact differs: " + name)


def verify_product(product: Receipt) -> None:
    manifest = product.contents
    verify_files(product.root, manifest["artifacts"])
    aliases = dict(manifest.get("symlinks", {}))
    if manifest["identity"]["recipe"]["name"] == "llvm-wasi-threaded":
        aliases["bin/wasm-ld"] = "lld"
    for name, target in aliases.items():
        path = child(product.root, name)
        if not path.is_symlink() or os.readlink(path) != target:
            raise ValueError("SDK tool alias differs: " + name)
    actual_aliases = {str(path.relative_to(product.root)) for path in product.root.rglob("*") if path.is_symlink()}
    if actual_aliases != set(aliases):
        raise ValueError("SDK product contains undeclared tool aliases")
    actual = {
        str(path.relative_to(product.root))
        for path in product.root.rglob("*")
        if path.is_file() and not path.is_symlink() and path != product.path
    }
    if actual != set(manifest["artifacts"]):
        raise ValueError("SDK product contains unrecorded or missing files")


def local_recipe(name: str) -> dict:
    return json.loads((Path(__file__).resolve().parents[1] / name).read_text())


def verify_cpython_recipe(recipe: dict) -> None:
    """Admit historical build provenance while retaining the Python source policy.

    The trusted SDK descriptor pins the consumed manifest, including its
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
    fields = {"schema_version", "kind", "name", "version", "root", "executable", "source", "files"}
    if producer.get("schema_version") == 2:
        fields.add("symlinks")
    if set(producer) != fields:
        raise ValueError("host package receipt fields differ")
    if producer["schema_version"] not in {1, 2} or producer["kind"] != "host-tool-files" or producer["name"] != name:
        raise ValueError("host package receipt profile differs")
    root = (proof_path.parent / producer["root"]).resolve()
    if child(root, producer["executable"]).resolve() != executable.resolve():
        raise ValueError("host package entrypoint differs")
    if name in {"meson", "python", "cython", "f2py", "pybind11-config"}:
        if producer["schema_version"] != 2:
            raise ValueError("Python-backed host tool requires source and base interpreter proof")
        from ports._support.host_tools import verify_source

        verify_source(root, producer, name)
    verify_files(root, producer["files"])
    if producer["schema_version"] == 2:
        aliases = {str(p.relative_to(root)): os.readlink(p) for p in root.rglob("*") if p.is_symlink()}
        if aliases != producer["symlinks"]:
            raise ValueError("host package aliases differ")
        for name in aliases:
            if not child(root, name).exists():
                raise ValueError("host package alias is missing")
    actual = {
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file() and (producer["schema_version"] == 1 or not p.is_symlink())
    }
    if actual != set(producer["files"]):
        raise ValueError("host package contains unrecorded or missing files")


def admit_host_tools(base: Path, bindings: dict) -> Mapping[str, Tool]:
    """Verify host bindings without admitting or requiring any target product."""
    tools = {}
    if not set(bindings) <= HOST_TOOLS:
        raise ValueError("unknown SDK host tool")
    for name, item in bindings.items():
        if set(item) != {"path", "sha256", "receipt"}:
            raise ValueError("host tool receipt fields differ")
        tool_path = Path(os.path.abspath(base / item["path"]))
        if file_hash(tool_path) != item["sha256"]:
            raise ValueError("SDK host tool differs: " + name)
        proof = item["receipt"]
        proof_path = None
        proof_hash = None
        if proof is not None:
            if set(proof) != {"path", "sha256"}:
                raise ValueError("host producer receipt fields differ")
            proof_path = (base / proof["path"]).resolve()
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
        elif name in {"uv", "meson", "python", "cython", "f2py", "pybind11-config"}:
            raise ValueError("host tool requires its production receipt: " + name)
        tools[name] = Tool(tool_path, item["sha256"], proof_path, proof_hash)
    return MappingProxyType(tools)


def admit_sdk(
    sdk: Receipt,
    llvm: Receipt,
    sysroot: Receipt,
    cpython: Receipt | None,
    runtime: Receipt | None,
    host_tools: Mapping[str, Tool],
    target_bindings: dict,
    *,
    target: str,
    abi: str,
) -> MaterializedSDK:
    """Check cross-product provenance and inventories before adapters use paths."""
    overlay = sysroot.contents
    if overlay["identity"]["recipe"]["dynamic_abi"] != abi:
        raise ValueError("SDK ABI differs from platform producer")
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
        if manifest["dynamic_abi"] != abi or manifest["recipe"]["target"] != target:
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
            raise ValueError("Python SDKs require a separately admitted runtime bundle")
        runtime_manifest = runtime.contents
        if (
            runtime_manifest["dynamic_abi"] != abi
            or runtime_manifest["recipe"] != manifest["recipe"]
            or runtime_manifest["build_profile_sha256"] != manifest["build_profile_sha256"]
            or runtime_manifest["build_profile"] != profile
            or runtime_manifest["files"]["/usr/bin/python3.wasm"] != manifest["files"]["/usr/bin/python3.wasm"]
        ):
            raise ValueError("assembled runtime differs from the admitted CPython build")
        verify_files(
            runtime.root / "rootfs", {name.lstrip("/"): digest for name, digest in runtime_manifest["files"].items()}
        )
    target_tools = {}
    value = {"target_tools": target_bindings}
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
            abi,
            target,
            "wasm32_wasip1",
            manifest["recipe"]["source"]["sha256"],
            json_hash(headers),
            headers["wasi-build/pyconfig.h"],
            runtime.sha256,
        )
    return MaterializedSDK(
        sdk,
        llvm,
        sysroot,
        cpython,
        runtime,
        python,
        MappingProxyType(host_tools),
        MappingProxyType(target_tools),
        json_hash({"compiler": llvm.sha256, "platform": sysroot.sha256, "tooling": sdk.sha256}),
        frontend,
        target,
        abi,
    )


def tool_reference(tool: Tool) -> dict:
    return {
        "path": str(Path(os.path.abspath(tool.path))),
        "sha256": tool.sha256,
        "receipt": {"path": str(tool.receipt_path.resolve()), "sha256": tool.receipt_sha256}
        if tool.receipt_path is not None
        else None,
    }


def resolved_toolchain(sdk_context: MaterializedSDK, compiler: Receipt, sysroot: Receipt) -> MaterializedSDK:
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
    identity = json_hash({"tooling": sdk_context.sdk.sha256, "compiler": compiler.sha256, "platform": sysroot.sha256})
    return replace(
        sdk_context,
        llvm=compiler,
        sysroot=sysroot,
        target_tools=MappingProxyType(target_tools),
        identity=identity,
        has_frontend=True,
    )
