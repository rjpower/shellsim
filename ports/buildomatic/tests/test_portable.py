"""Exercise real inventory/admission with tiny products and optional release evidence.

Only producer-policy pins are replaced in synthetic fixtures; receipt hashing,
complete inventories, aliases, Python closure proofs and SDK admission run as
production code. No compiler or interpreter is executed from these fixtures.
"""

import json
import os
import shutil
import stat
from dataclasses import replace
from pathlib import Path

import pytest

from ports._support.host_tools import definition, make_read_only, python_closure
from ports._support.import_sdk import load_legacy_cohort
from ports._support.sdk_products import admit_sdk, file_hash, json_hash, receipt
from ports.buildomatic.portable import PortableLimits, descriptor_digest, export_sdk, import_sdk, original_root_bindings
from ports.toolchain.wasi_threads.dynamic import sdk_tooling


def write(root, name, data=b"fixture", mode=0o644):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    path.chmod(mode)
    return path


def proof(root, value, name="manifest.json"):
    path = write(root, name, (json.dumps(value, indent=2) + "\n\n").encode())
    return receipt(Path("/"), {"root": str(root), "manifest": str(path), "sha256": file_hash(path)})


def hashes(root):
    return {p.relative_to(root).as_posix(): file_hash(p) for p in root.rglob("*") if p.is_file() and not p.is_symlink()}


@pytest.fixture
def sdk_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr("ports._support.producer_policy.verify_policy", lambda *_args: None)
    monkeypatch.setattr("ports._support.sdk_products.verify_cpython_recipe", lambda *_args: None)
    tooling_root = tmp_path / "original-sdk"
    for name in ("clang", "clang.cfg", "clang++.cfg", "llvm-ar", "llvm-nm", "llvm-strip"):
        write(tooling_root, "bin/" + name, name.encode(), 0o755)
    (tooling_root / "bin/clang++").symlink_to("clang")
    (tooling_root / "bin/llvm-ranlib").symlink_to("llvm-ar")
    write(tooling_root, "lib/clang/23/include/stddef.h")
    write(tooling_root, "lib/clang/23/lib/wasm32-unknown-wasip1-threads/libclang_rt.builtins.a")
    tooling = sdk_tooling(tooling_root)
    sdk = proof(tooling_root, tooling)
    # SDK admission binds its consumed subset, never unrelated SDK content.
    write(tooling_root, "unrelated-sdk-file", b"excluded")
    compiler_root = tmp_path / "original-compiler"
    for name in ("clang", "llvm-ar", "llvm-strip", "lld"):
        write(compiler_root, "bin/" + name, name.encode(), 0o755)
    aliases = {"bin/clang++": "clang", "bin/llvm-ranlib": "llvm-ar", "bin/wasm-ld": "lld"}
    for name, target in aliases.items():
        (compiler_root / name).symlink_to(target)
    compiler = proof(
        compiler_root,
        {
            "identity": {"recipe": {"name": "llvm-wasi-compiler"}},
            "artifacts": hashes(compiler_root),
            "symlinks": aliases,
        },
    )
    platform_root = tmp_path / "original-platform"
    write(platform_root, "sysroot/include/stdio.h")
    write(platform_root, "sysroot/lib/libc.a")
    platform = proof(
        platform_root,
        {
            "identity": {
                "recipe": {
                    "name": "fixture-platform",
                    "dynamic_abi": "fixture-abi",
                    "sdk_tooling_digest": json_hash(tooling),
                    "sdk": {},
                },
                "sdk_tooling": tooling,
                "compiler": compiler.contents,
            },
            "artifacts": hashes(platform_root),
        },
    )
    targets = {
        name: {"provider": "llvm", "path": "bin/" + executable, "sha256": file_hash(compiler_root / "bin" / executable)}
        for name, executable in {
            "cc": "clang",
            "cxx": "clang++",
            "ar": "llvm-ar",
            "ranlib": "llvm-ranlib",
            "strip": "llvm-strip",
        }.items()
    }
    native = write(tmp_path / "host-native", "sh", b"native-tool", 0o755)
    hosts = {"sh": {"path": str(native), "sha256": file_hash(native), "receipt": None}}

    def create(python=False, host_python=False, host_aliases=None):
        cpython = runtime = None
        if python:
            root = tmp_path / "original-cpython"
            write(root, "Python-3.13.7/Include/Python.h", b"Python header")
            write(root, "Python-3.13.7/Include/internal/pycore.h", b"internal header")
            write(root, "wasi-build/pyconfig.h", b"Python config")
            write(root, "rootfs/usr/bin/python3.wasm", b"guest Python", 0o755)
            write(root, "rootfs/usr/lib/python3.13/_sysconfigdata.py", b"config")
            headers = {name: digest for name, digest in hashes(root).items() if name.endswith(".h")}
            profile = {"sysroot": platform.contents, "headers": headers}
            manifest = {
                "recipe": {"version": "3.13.7", "target": "wasm32-wasip1-threads", "source": {"sha256": "a" * 64}},
                "dynamic_abi": "fixture-abi",
                "build_profile": profile,
                "build_profile_sha256": json_hash(profile),
                "files": {
                    "/" + name.removeprefix("rootfs/"): digest
                    for name, digest in hashes(root).items()
                    if name.startswith("rootfs/")
                },
            }
            cpython = proof(root, manifest)
            runtime_root = tmp_path / "original-runtime"
            shutil.copytree(root / "rootfs", runtime_root / "rootfs")
            runtime = proof(runtime_root, manifest)
            write(root, "Python-3.13.7/Modules/unrelated.c", b"excluded source")
            write(root, "wasi-build/unrelated.o", b"excluded build work")
        if host_python:
            base = tmp_path / "host-base"
            write(base, "bin/python3.13", b"native-python", 0o755)
            write(base, "include/python3.13/patchlevel.h", b'#define PY_VERSION "3.13.15"\n')
            write(base, "lib/python3.13/os.py", b"stdlib")
            write(base, "lib/python3.13/lib-dynload/_struct.so", b"extension")
            (base / "bin/python").symlink_to("python3.13")
            make_read_only(base)
            environment = tmp_path / "host-environment"
            write(environment, "bin/python", b"native-python", 0o755)
            write(environment, "bin/cython", ("#!" + str(environment / "bin/python") + "\n").encode(), 0o755)
            write(
                environment,
                "pyvenv.cfg",
                ("home = " + str(base / "bin") + "\ninclude-system-site-packages = false\n").encode(),
            )
            write(environment, "lib/python3.13/site-packages/package/__init__.py", b"package closure")
            for alias, target in (host_aliases or {}).items():
                (environment / alias).symlink_to(target, target_is_directory=True)
            value = {
                "schema_version": 2,
                "kind": "host-tool-files",
                "name": "python",
                "version": "3.13.15",
                "root": str(environment),
                "executable": "bin/python",
                "files": hashes(environment),
                "symlinks": host_aliases or {},
                "source": {"definition": definition()["python"], "base_python": python_closure(environment, base)},
            }
            host_receipt = proof(tmp_path / "host-receipts", value, "python.json")
            hosts["python"] = {
                "path": str(environment / "bin/python"),
                "sha256": file_hash(environment / "bin/python"),
                "receipt": {"path": str(host_receipt.path), "sha256": host_receipt.sha256},
            }
            make_read_only(environment)
        from ports._support.sdk_products import admit_host_tools

        return admit_sdk(
            sdk,
            compiler,
            platform,
            cpython,
            runtime,
            admit_host_tools(Path("/"), hosts),
            targets,
            target="wasm32-wasip1-threads",
            abi="fixture-abi",
        )

    yield create
    # Our exported products are deliberately read-only, including failed imports.
    for path in (tmp_path, *tmp_path.rglob("*")):
        if path.is_dir() and not path.is_symlink():
            path.chmod(path.stat().st_mode | 0o700)


def load(path):
    return json.loads(path.read_text())


def rewrite(path, value):
    path.chmod(0o644)
    path.write_text(json.dumps(value))


@pytest.mark.parametrize("python", [False, True])
def test_fresh_directory_round_trip_preserves_receipts_and_complete_inputs(tmp_path, sdk_fixture, python):
    context = sdk_fixture(python=python)
    descriptor = export_sdk(context, tmp_path / "export")
    assert descriptor_digest(descriptor) == file_hash(descriptor)
    imported = import_sdk(descriptor, tmp_path / "fresh")
    assert imported.identity == context.identity
    assert imported.host_tools == context.host_tools
    assert imported.flags != context.flags
    for original, restored in (
        (context.sdk, imported.sdk),
        (context.llvm, imported.llvm),
        (context.sysroot, imported.sysroot),
        (context.cpython_manifest, imported.cpython_manifest),
        (context.runtime, imported.runtime),
    ):
        if original is None:
            assert restored is None
            continue
        assert restored.root != original.root
        assert restored.path.read_bytes() == original.path.read_bytes()
        assert restored.sha256 == original.sha256
        assert not restored.root.stat().st_mode & 0o222
        assert all(not p.stat().st_mode & 0o222 for p in restored.root.rglob("*") if not p.is_symlink())
    assert os.readlink(imported.sdk.root / "bin/llvm-ranlib") == "llvm-ar"
    assert os.readlink(imported.llvm.root / "bin/wasm-ld") == "lld"
    assert imported.compiler().stat().st_mode & stat.S_IXUSR
    assert not (imported.sdk.root / "unrelated-sdk-file").exists()
    if python:
        assert (imported.python.source_root / "Include/internal/pycore.h").is_file()
        assert (imported.python.generated_config_dir / "pyconfig.h").is_file()
        assert (imported.runtime.root / "rootfs/usr/lib/python3.13/_sysconfigdata.py").is_file()
        assert not (imported.cpython_manifest.root / "wasi-build/unrelated.o").exists()
        assert not (imported.python.source_root / "Modules/unrelated.c").exists()


def test_host_python_closure_and_receipts_keep_absolute_roots(tmp_path, sdk_fixture):
    context = sdk_fixture(host_python=True)
    originals = {
        path: path.read_bytes()
        for path in (
            context.host_tools["python"].receipt_path,
            tmp_path / "host-environment/pyvenv.cfg",
            tmp_path / "host-environment/bin/cython",
        )
    }
    descriptor = export_sdk(context, tmp_path / "export")
    value = load(descriptor)
    stable = {root["path"]: root for root in value["roots"].values() if root["stable"]}
    assert str(tmp_path / "host-base") in stable
    assert str(tmp_path / "host-environment") in stable
    assert "lib/python3.13/lib-dynload/_struct.so" in stable[str(tmp_path / "host-base")]["files"]
    assert "lib/python3.13/site-packages/package/__init__.py" in stable[str(tmp_path / "host-environment")]["files"]
    imported = import_sdk(descriptor, tmp_path / "fresh")
    assert imported.host_tools["python"].path == context.host_tools["python"].path
    assert all(path.read_bytes() == data for path, data in originals.items())
    root_name = next(
        name for name, root in value["roots"].items() if root["path"] == str(tmp_path / "host-environment")
    )
    with pytest.raises(ValueError):
        import_sdk(descriptor, tmp_path / "other", bindings={root_name: tmp_path / "relocated-host"})
    assert not (tmp_path / "other").exists()


@pytest.mark.parametrize("kind", ["relative", "absolute", "chain"])
def test_host_directory_alias_round_trip_preserves_declared_closure(tmp_path, sdk_fixture, kind):
    environment = tmp_path / "host-environment"
    aliases = {"lib64": str(environment / "lib") if kind == "absolute" else "lib"}
    if kind == "chain":
        aliases["lib-alias"] = "lib64"
    context = sdk_fixture(host_python=True, host_aliases=aliases)
    receipt_bytes = context.host_tools["python"].receipt_path.read_bytes()
    config_bytes = (environment / "pyvenv.cfg").read_bytes()
    environment.chmod(0o755)
    (environment / "lib").chmod(0o755)
    (environment / "lib/unrelated").mkdir()
    descriptor = export_sdk(context, tmp_path / "export")
    root = next(value for value in load(descriptor)["roots"].values() if value["path"] == str(environment))
    assert root["symlinks"] == aliases
    assert "lib" in root["directories"]
    assert "lib/unrelated" not in root["directories"]
    assert all(not name.startswith("lib64/") for name in root["files"])
    for path in (environment, *environment.rglob("*")):
        if path.is_dir() and not path.is_symlink():
            path.chmod(0o755)
    shutil.rmtree(environment)
    imported = import_sdk(descriptor, tmp_path / "fresh", bindings=original_root_bindings(descriptor))
    assert imported.host_tools == context.host_tools
    assert imported.host_tools["python"].receipt_path.read_bytes() == receipt_bytes
    assert (environment / "pyvenv.cfg").read_bytes() == config_bytes
    assert all(os.readlink(environment / name) == target for name, target in aliases.items())
    assert (environment / "lib64/python3.13/site-packages/package/__init__.py").read_bytes() == b"package closure"
    assert not (environment / "lib/unrelated").exists()


@pytest.mark.parametrize("kind", ["tamper", "missing", "missing-alias", "escape", "cycle", "parent", "product"])
def test_host_directory_alias_descriptor_frontiers(tmp_path, sdk_fixture, kind):
    descriptor = export_sdk(sdk_fixture(host_python=True, host_aliases={"lib64": "lib"}), tmp_path / "export")
    value = load(descriptor)
    root = next(root for root in value["roots"].values() if root["path"] == str(tmp_path / "host-environment"))
    if kind == "tamper":
        root["symlinks"]["lib64"] = "bin"
    elif kind == "missing":
        root["directories"].pop("lib")
    elif kind == "missing-alias":
        root["symlinks"].pop("lib64")
    elif kind == "escape":
        root["symlinks"]["lib64"] = "../host-base/lib"
    elif kind == "cycle":
        root["symlinks"].update({"lib64": "lib-alias", "lib-alias": "lib64"})
    elif kind == "parent":
        root["files"]["lib64/injected"] = next(iter(root["files"].values()))
    else:
        value["roots"]["llvm"]["symlinks"]["lib64"] = "bin"
    rewrite(descriptor, value)
    with pytest.raises(ValueError):
        import_sdk(descriptor, tmp_path / "fresh")
    assert not (tmp_path / "fresh").exists()


@pytest.mark.parametrize("target", ["bin", "../host-base/lib", None])
def test_host_directory_alias_mount_tamper_is_not_overwritten(tmp_path, sdk_fixture, target):
    descriptor = export_sdk(sdk_fixture(host_python=True, host_aliases={"lib64": "lib"}), tmp_path / "export")
    environment = tmp_path / "host-environment"
    environment.chmod(0o755)
    alias = environment / "lib64"
    alias.unlink()
    if target is not None:
        alias.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError):
        import_sdk(descriptor, tmp_path / "fresh")
    if target is None:
        assert not alias.exists()
    else:
        assert os.readlink(alias) == target
    assert not (tmp_path / "fresh").exists()


def test_directory_alias_export_cannot_select_an_undeclared_directory(tmp_path):
    from ports.buildomatic.portable import _inventory

    root = tmp_path / "host"
    write(root, "bin/python")
    (root / "unselected").mkdir()
    (root / "lib64").symlink_to("unselected", target_is_directory=True)
    with pytest.raises(ValueError):
        _inventory(root, {"bin/python": file_hash(root / "bin/python")}, {"lib64": "unselected"}, stable=True)


def test_host_directory_alias_inventory_does_not_traverse_unselected_files(tmp_path):
    from ports.buildomatic.portable import _inventory

    root = tmp_path / "host"
    selected = write(root, "lib/package/__init__.py", b"declared package")
    write(root, "lib/unselected.py", b"exclude undeclared file")
    (root / "lib64").symlink_to("lib", target_is_directory=True)
    inventory = _inventory(root, {"lib/package/__init__.py": file_hash(selected)}, {"lib64": "lib"}, stable=True)
    assert set(inventory.files) == {"lib/package/__init__.py"}
    assert inventory.symlinks == {"lib64": "lib"}
    assert set(inventory.directories) == {".", "lib", "lib/package"}


def test_explicit_original_bindings_preserve_flags_and_verify_mounts(tmp_path, sdk_fixture):
    context = sdk_fixture(python=True)
    descriptor = export_sdk(context, tmp_path / "export")
    restored = import_sdk(descriptor, tmp_path / "unused", original_bindings=True)
    assert restored.flags == context.flags
    assert restored.compiler() == context.compiler()
    assert (context.sdk.root / "unrelated-sdk-file").read_bytes() == b"excluded"
    assert (context.cpython_manifest.root / "wasi-build/unrelated.o").read_bytes() == b"excluded build work"
    assert not (tmp_path / "unused").exists()
    (context.llvm.root / "bin/clang").write_bytes(b"changed mount")
    with pytest.raises(ValueError):
        import_sdk(descriptor, tmp_path / "unused", original_bindings=True)
    assert (context.llvm.root / "bin/clang").read_bytes() == b"changed mount"


def test_product_binding_at_explicit_fresh_root(tmp_path, sdk_fixture):
    descriptor = export_sdk(sdk_fixture(), tmp_path / "export")
    imported = import_sdk(descriptor, tmp_path / "fresh", bindings={"llvm": tmp_path / "fixed-compiler"})
    assert imported.llvm.root == tmp_path / "fixed-compiler"
    assert not (tmp_path / "fresh/llvm").exists()


def test_original_product_bindings_restore_missing_roots_without_build_work(tmp_path, sdk_fixture):
    context = sdk_fixture(python=True)
    descriptor = export_sdk(context, tmp_path / "export")
    for product in (context.sdk, context.llvm, context.sysroot, context.cpython_manifest, context.runtime):
        shutil.rmtree(product.root)
    imported = import_sdk(descriptor, tmp_path / "unused", original_bindings=True)
    assert imported.flags == context.flags
    assert imported.llvm.root == context.llvm.root
    assert imported.python.generated_config_dir == context.python.generated_config_dir
    assert not (context.cpython_manifest.root / "wasi-build/unrelated.o").exists()
    assert not (tmp_path / "unused").exists()


def test_absolute_internal_host_alias_remains_at_its_stable_root(tmp_path, sdk_fixture):
    context = sdk_fixture()
    tool = context.host_tools["sh"].path
    target = tool.with_name("native-target")
    tool.rename(target)
    tool.symlink_to(target)
    descriptor = export_sdk(context, tmp_path / "export")
    value = load(descriptor)
    name = next(name for name, root in value["roots"].items() if root["path"] == str(tool.parent))
    assert value["roots"][name]["symlinks"]["sh"] == str(target)
    shutil.rmtree(tool.parent)
    imported = import_sdk(descriptor, tmp_path / "fresh", bindings={name: tool.parent})
    assert os.readlink(imported.host_tools["sh"].path) == str(target)


def test_original_product_with_external_receipt_does_not_mutate_mount(tmp_path, sdk_fixture):
    context = sdk_fixture()
    external = tmp_path / "external-sdk-receipt.json"
    context.sdk.path.rename(external)
    context = replace(context, sdk=replace(context.sdk, path=external))
    descriptor = export_sdk(context, tmp_path / "export")
    imported = import_sdk(descriptor, tmp_path / "unused", original_bindings=True)
    assert imported.flags == context.flags
    assert imported.sdk.path.read_bytes() == external.read_bytes()
    assert not (context.sdk.root / "manifest.json").exists()
    assert not (tmp_path / "unused").exists()


def test_stable_mount_requires_explicit_restore_and_preserves_bytes(tmp_path, sdk_fixture):
    descriptor = export_sdk(sdk_fixture(host_python=True), tmp_path / "export")
    value = load(descriptor)
    stable = {name: Path(root["path"]) for name, root in value["roots"].items() if root["stable"]}
    expected = (tmp_path / "host-environment/pyvenv.cfg").read_bytes()
    for root in stable.values():
        for path in (root, *root.rglob("*")):
            if path.is_dir() and not path.is_symlink():
                path.chmod(0o755)
        shutil.rmtree(root)
    with pytest.raises(ValueError):
        import_sdk(descriptor, tmp_path / "missing")
    imported = import_sdk(descriptor, tmp_path / "fresh", bindings=stable)
    assert (tmp_path / "host-environment/pyvenv.cfg").read_bytes() == expected
    assert imported.host_tools["python"].path == tmp_path / "host-environment/bin/python"
    assert not (tmp_path / "host-base").stat().st_mode & 0o222


def test_original_root_bindings_restore_all_roots_on_a_fresh_worker(tmp_path, sdk_fixture):
    context = sdk_fixture(python=True, host_python=True)
    descriptor = export_sdk(context, tmp_path / "export")
    bindings = original_root_bindings(descriptor)
    assert bindings == {name: Path(root["path"]) for name, root in load(descriptor)["roots"].items()}
    config = (tmp_path / "host-environment/pyvenv.cfg").read_bytes()
    for root in bindings.values():
        for path in (root, *root.rglob("*")):
            if path.is_dir() and not path.is_symlink():
                path.chmod(0o755)
        shutil.rmtree(root)
    with pytest.raises(ValueError):
        import_sdk(descriptor, tmp_path / "unused", original_bindings=True)
    imported = import_sdk(descriptor, tmp_path / "unused", bindings=bindings, original_bindings=True)
    assert imported.flags == context.flags
    assert imported.host_tools == context.host_tools
    assert (tmp_path / "host-environment/pyvenv.cfg").read_bytes() == config
    assert not (tmp_path / "unused").exists()
    assert original_root_bindings(descriptor) == bindings
    value = load(descriptor)
    value["roots"]["sdk"]["path"] = "../unsafe"
    rewrite(descriptor, value)
    with pytest.raises(ValueError):
        original_root_bindings(descriptor)


@pytest.mark.parametrize("kind", ["blob", "receipt", "alias", "escape", "parent", "mode", "identity", "unknown"])
def test_tampered_bundle_or_descriptor_is_rejected(tmp_path, sdk_fixture, kind):
    descriptor = export_sdk(sdk_fixture(), tmp_path / "export")
    value = load(descriptor)
    root = value["roots"]["llvm"]
    if kind in {"blob", "receipt"}:
        digest = root["files"]["bin/clang"]["sha256"] if kind == "blob" else value["products"]["llvm"]["sha256"]
        blob = descriptor.parent / "blobs" / digest
        blob.chmod(0o644)
        blob.write_bytes(b"changed bytes")
    elif kind == "alias":
        root["symlinks"]["bin/clang++"] = "/usr/bin/clang"
    elif kind == "escape":
        root["files"]["../escape"] = root["files"].pop("bin/clang")
    elif kind == "parent":
        root["directories"].pop("bin")
    elif kind == "mode":
        root["files"]["bin/clang"]["mode"] = 0o4755
    elif kind == "identity":
        value["identity"] = "0" * 64
    else:
        value["unknown"] = True
    rewrite(descriptor, value)
    with pytest.raises(ValueError):
        import_sdk(descriptor, tmp_path / "fresh")
    assert not (tmp_path / "escape").exists()


@pytest.mark.parametrize(
    "limits",
    [
        PortableLimits(max_files=1),
        PortableLimits(max_bytes=1),
        PortableLimits(max_file_bytes=1),
        PortableLimits(metadata_bytes=1),
    ],
)
def test_export_and_import_respect_resource_bounds(tmp_path, sdk_fixture, limits):
    context = sdk_fixture()
    descriptor = export_sdk(context, tmp_path / "export")
    with pytest.raises(ValueError):
        export_sdk(context, tmp_path / "too-small", limits=limits)
    with pytest.raises(ValueError):
        import_sdk(descriptor, tmp_path / "fresh", limits=limits)


def test_no_overwrite_and_no_unrelated_mount_files(tmp_path, sdk_fixture):
    context = sdk_fixture()
    descriptor = export_sdk(context, tmp_path / "export")
    with pytest.raises(FileExistsError):
        export_sdk(context, descriptor.parent)
    import_sdk(descriptor, tmp_path / "fresh")
    root = tmp_path / "fresh/llvm"
    root.chmod(0o755)
    write(root, "unrelated", b"preserve")
    with pytest.raises(ValueError):
        import_sdk(descriptor, tmp_path / "fresh")
    assert (root / "unrelated").read_bytes() == b"preserve"
    with pytest.raises(ValueError):
        import_sdk(descriptor, tmp_path / "fresh", bindings={"typo": tmp_path / "unknown"})


def test_export_rechecks_receipt_provenance_and_header_inventory(tmp_path, sdk_fixture):
    context = sdk_fixture(python=True)
    (context.python.source_root / "Include/Python.h").write_bytes(b"changed header")
    with pytest.raises(ValueError):
        export_sdk(context, tmp_path / "export")
    assert not (tmp_path / "export").exists()
    context.sdk.path.write_bytes(context.sdk.path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        export_sdk(context, tmp_path / "export")


def test_missing_exported_host_closure_cannot_use_ambient_original(tmp_path, sdk_fixture):
    descriptor = export_sdk(sdk_fixture(host_python=True), tmp_path / "export")
    value = load(descriptor)
    name = next(name for name, root in value["roots"].items() if root["path"] == str(tmp_path / "host-base"))
    del value["roots"][name]
    rewrite(descriptor, value)
    with pytest.raises(ValueError):
        import_sdk(descriptor, tmp_path / "fresh")
    assert not (tmp_path / "fresh").exists()


def test_descriptor_identity_is_independent_of_host_mapping_order(tmp_path, sdk_fixture):
    context = sdk_fixture(host_python=True)
    first = export_sdk(context, tmp_path / "first")
    reordered = replace(context, host_tools=dict(reversed(list(context.host_tools.items()))))
    second = export_sdk(reordered, tmp_path / "second")
    assert descriptor_digest(first) == descriptor_digest(second)


def test_streamed_blob_transport_avoids_read_bytes(tmp_path, sdk_fixture, monkeypatch):
    context = sdk_fixture(python=True)
    read_bytes = Path.read_bytes

    def reject_blob_read_bytes(path):
        if path.parent.name == "blobs":
            pytest.fail("binary blob transport must stream")
        # Existing SDK admission has its own hashing implementation.
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", reject_blob_read_bytes)
    descriptor = export_sdk(context, tmp_path / "export")
    imported = import_sdk(descriptor, tmp_path / "fresh")
    assert imported.identity == context.identity


def test_alias_chain_preserves_bytes_and_rejects_cycle(tmp_path, sdk_fixture):
    context = sdk_fixture()
    alias = context.sdk.root / "bin/clang++"
    alias.unlink()
    (context.sdk.root / "bin/clang-alias").symlink_to("clang")
    alias.symlink_to("clang-alias")
    descriptor = export_sdk(context, tmp_path / "export")
    imported = import_sdk(descriptor, tmp_path / "fresh")
    assert os.readlink(imported.sdk.root / "bin/clang++") == "clang-alias"
    assert os.readlink(imported.sdk.root / "bin/clang-alias") == "clang"
    value = load(descriptor)
    value["roots"]["sdk"]["symlinks"]["bin/clang-alias"] = "clang++"
    rewrite(descriptor, value)
    with pytest.raises(ValueError):
        import_sdk(descriptor, tmp_path / "cycle")


def test_nested_or_symlink_mount_bindings_are_rejected(tmp_path, sdk_fixture):
    descriptor = export_sdk(sdk_fixture(), tmp_path / "export")
    with pytest.raises(ValueError):
        import_sdk(descriptor, tmp_path / "fresh", bindings={"llvm": tmp_path / "fresh/sdk/nested"})
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "symlink-mount").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        import_sdk(descriptor, tmp_path / "fresh", bindings={"llvm": tmp_path / "symlink-mount"})
    assert not list(outside.iterdir())
    with pytest.raises(ValueError):
        import_sdk(descriptor, tmp_path / "fresh", bindings={"llvm": tmp_path / "symlink-mount/nested"})
    assert not list(outside.iterdir())


def test_existing_release_round_trip(tmp_path):
    original = os.environ.get("SHELLSIM_BUILD_COHORT")
    if original is None:
        pytest.skip("set SHELLSIM_BUILD_COHORT to an existing complete admitted SDK")
    context = load_legacy_cohort(Path(original))
    descriptor = export_sdk(context, tmp_path / "export")
    imported = import_sdk(descriptor, tmp_path / "fresh")
    assert imported.identity == context.identity
    assert imported.llvm.path.read_bytes() == context.llvm.path.read_bytes()
    assert imported.host_tools == context.host_tools
