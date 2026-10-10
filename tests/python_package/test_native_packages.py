"""Synthetic sealed catalogs exercise resolution and atomic virtual installation."""

import hashlib
import json
import zipfile

import pytest
import shellsim
from shellsim.native_packages import _digest, _NativePackageUniverse


@pytest.fixture
def catalog(tmp_path):
    records = []

    def add(name, version="1", dependencies=(), pinned=(), payload=None, recipe_name=None, tool=True):
        prefix = tmp_path / (name + "-" + version)
        prefix.mkdir()
        data = payload or (name + version).encode()
        (prefix / "tool").write_bytes(data)
        recipe = {
            "name": recipe_name or name,
            "version": version,
            "target": "wasm32-wasip1",
            "target_profile": "fixture-v1",
            "exports": {"tools" if tool else "headers": ["tool"]},
            "target_dependencies": list(pinned),
        }
        inputs = {
            "recipe": recipe,
            "toolchain": {"profile": {"name": "fixture-v1", "target": "wasm32-wasip1"}},
            "dependency_artifacts": {dep["port"]: dep["hash"] for dep in pinned},
        }
        # Hash is artifact input metadata; the recipe itself describes version/port.
        for dep in recipe["target_dependencies"]:
            dep.pop("hash")
        inputs["recipe_sha256"] = _digest(recipe)
        manifest = {"inputs": inputs, "files": {"tool": hashlib.sha256(data).hexdigest()}}
        manifest["artifact_sha256"] = _digest(manifest)
        (prefix / "artifact.json").write_text(json.dumps(manifest))
        record = {
            "name": name,
            "version": version,
            "kind": "build-tool" if tool else "devel",
            "artifact": prefix.name,
            "artifact_sha256": manifest["artifact_sha256"],
            "recipe_name": recipe["name"],
            "target_profile": "fixture-v1",
            "toolchain_sha256": _digest(inputs["toolchain"]),
            "destinations": {"tool": "/usr/local/bin/" + name},
            "dependencies": list(dependencies),
        }
        records.append(record)
        return record

    def universe():
        path = tmp_path / "catalog.json"
        path.write_text(json.dumps({"format": 1, "target": "wasm32-wasip1", "packages": records}))
        return _NativePackageUniverse(path)

    return add, universe, tmp_path


@pytest.fixture
def wheel_runtime(tmp_path):
    bundle = tmp_path / "bundle"
    binary = bundle / "rootfs/usr/bin/python3.wasm"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"\0asm\x01\0\0\0")
    binary.chmod(0o755)
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "recipe": {"version": "3.13.7", "target": "wasm32-wasip1", "prefix": "/usr"},
                "site_packages": "/usr/lib/python3.13/site-packages",
                "files": {"/usr/bin/python3.wasm": hashlib.sha256(binary.read_bytes()).hexdigest()},
            }
        )
    )
    wheel = tmp_path / "example-1-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("example.py", b"python-owned")
        archive.writestr("example-1.dist-info/WHEEL", "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
    return shellsim.CPythonRuntime(bundle), wheel


@pytest.mark.parametrize("python_first", [False, True])
@pytest.mark.parametrize("identical", [False, True])
def test_python_and_native_files_share_atomic_conflict_policy(catalog, wheel_runtime, python_first, identical):
    add, universe, _ = catalog
    runtime, wheel = wheel_runtime
    payload = b"python-owned" if identical else b"native-owned"
    record = add("collision", payload=payload, tool=False)
    destination = runtime.site_packages + "/example.py"
    record["destinations"] = {"tool": destination}
    other = add("new-file", tool=False)
    other["destinations"] = {"tool": "/usr/local/share/new-file"}
    env = shellsim.Environment()
    runtime.mount(env)
    native = universe()
    if python_first:
        runtime.install_wheel(env, wheel)

        def install():
            native.install(env, ["collision", "new-file"])

        exception = shellsim.SimulationError
        before = b"python-owned"
    else:
        native.install(env, "collision")

        def install():
            runtime.install_wheel(env, wheel)

        exception = shellsim.PackageInstallError
        before = payload
    if identical:
        install()
    else:
        with pytest.raises(exception):
            install()
        assert env.run("test ! -e /usr/local/share/new-file").returncode == 0
        if not python_first:
            assert env.run("test ! -e /usr/lib/python3.13/site-packages/example-1.dist-info/WHEEL").returncode == 0
    assert env.read_file(destination) == before


def test_python_package_disk_failure_preserves_simulation_error(wheel_runtime):
    runtime, wheel = wheel_runtime
    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr("oversized.py", b"x" * 200_000)
    env = shellsim.Environment(disk=100_000)
    runtime.mount(env)
    with pytest.raises(shellsim.SimulationError):
        runtime.install_wheel(env, wheel)
    assert env.run("test ! -e /usr/lib/python3.13/site-packages/example.py").returncode == 0


def test_single_string_range_and_recipe_alias(catalog):
    add, universe, _ = catalog
    add("alias", "1", recipe_name="upstream")
    add("alias", "2", recipe_name="upstream")
    env = shellsim.Environment()
    assert universe().install(env, "alias>=1,<2") == {"alias": "1"}
    assert env.read_file("/usr/local/bin/alias") == b"alias1"
    assert env.run("test -x /usr/local/bin/alias").returncode == 0


@pytest.mark.parametrize(
    "destination",
    ["/usr/lib/python3.13/lib-dynload/zlib.so", "/usr/lib/../../../work/zlib.so", "/usr/library/zlib.so"],
)
def test_system_library_exports_require_canonical_install_paths(catalog, destination):
    add, universe, _ = catalog
    record = add("module")
    record["destinations"] = {"tool": destination}
    env = shellsim.Environment()
    if destination == "/usr/lib/python3.13/lib-dynload/zlib.so":
        universe().install(env, "module")
        assert env.read_file(destination) == b"module1"
        return
    with pytest.raises(ValueError):
        universe().install(env, "module")
    assert env.run("test ! -e /work/zlib.so").returncode == 0


def test_global_constraints_backtrack(catalog):
    add, universe, _ = catalog
    add("provider", "1")
    add("provider", "2")
    add(
        "consumer",
        "1",
        [{"requirement": "provider>=1,<3", "kind": "build-tool"}],
        [{"port": "provider", "version": "1", "hash": "placeholder"}],
    )
    # A real consumer must pin the selected provider artifact exactly.
    records = universe()._packages
    provider = records["provider"][1]
    root = catalog[2]
    manifest_path = root / "consumer-1/artifact.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["inputs"]["dependency_artifacts"]["provider"] = provider.digest
    manifest.pop("artifact_sha256")
    manifest["artifact_sha256"] = _digest(manifest)
    manifest_path.write_text(json.dumps(manifest))
    path = root / "catalog.json"
    value = json.loads(path.read_text())
    value["packages"][-1]["artifact_sha256"] = manifest["artifact_sha256"]
    path.write_text(json.dumps(value))
    assert _NativePackageUniverse(path).install(shellsim.Environment(), "consumer") == {
        "consumer": "1",
        "provider": "1",
    }


@pytest.mark.parametrize(
    "field,value", [("target_profile", "wrong"), ("recipe_name", "wrong"), ("toolchain_sha256", "0" * 64)]
)
def test_wrong_identity_has_no_vfs_effect(catalog, field, value):
    add, universe, _ = catalog
    record = add("tool")
    record[field] = value
    env = shellsim.Environment()
    env.write_file("/sentinel", b"unchanged")
    with pytest.raises(ValueError):
        universe().install(env, "tool")
    assert env.read_file("/sentinel") == b"unchanged"
    assert env.run("test ! -e /usr/local/bin/tool").returncode == 0


def test_corrupt_file_rejected(catalog):
    add, universe, root = catalog
    add("tool")
    (root / "tool-1/tool").write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        universe().install(shellsim.Environment(), "tool")


def test_omitted_native_dependency_rejected(catalog):
    add, universe, _ = catalog
    add("tool", pinned=[{"port": "provider", "version": "1", "hash": "0" * 64}])
    with pytest.raises(ValueError):
        universe().install(shellsim.Environment(), "tool")


@pytest.mark.parametrize("spec", ["tool @ https://example.com/tool", "tool[extra]", 'tool; python_version>"3"'])
def test_unsupported_requirement_rejected(catalog, spec):
    add, universe, _ = catalog
    add("tool")
    with pytest.raises(ValueError):
        universe().install(shellsim.Environment(), spec)


def test_directory_chain_is_bounded(catalog, monkeypatch):
    import shellsim.native_packages as native

    add, universe, root = catalog
    add("tool")
    path = root / "tool-1"
    for _ in range(8):
        path /= "empty"
        path.mkdir()
    monkeypatch.setattr(native, "_MAX_FILES", 5)
    with pytest.raises(ValueError):
        universe().install(shellsim.Environment(), "tool")


def test_conflicting_exports_rejected_before_mount(catalog):
    add, universe, _ = catalog
    add("one")
    two = add("two")
    two["destinations"] = {"tool": "/usr/local/bin/one"}
    env = shellsim.Environment()
    with pytest.raises(ValueError):
        universe().install(env, ["one", "two"])
    assert env.run("test ! -e /usr/local/bin/one").returncode == 0


def test_dependency_hash_mismatch_rejected(catalog):
    add, universe, _ = catalog
    add("provider")
    add(
        "consumer",
        dependencies=[{"requirement": "provider", "kind": "build-tool"}],
        pinned=[{"port": "provider", "version": "1", "hash": "0" * 64}],
    )
    with pytest.raises(ValueError):
        universe().install(shellsim.Environment(), "consumer")


def test_disk_failure_is_atomic(catalog):
    add, universe, _ = catalog
    add("tool", payload=b"x" * (1024 * 1024))
    env = shellsim.Environment(disk=64 * 1024)
    env.write_file("/sentinel", b"unchanged")
    with pytest.raises(shellsim.SimulationError):
        universe().install(env, "tool")
    assert env.read_file("/sentinel") == b"unchanged"
    assert env.run("test ! -e /usr/local/bin/tool").returncode == 0


def test_sequential_provider_replacement_rejected_across_universes(catalog):
    add, universe, root = catalog
    provider = add("provider", "1")
    add("provider", "2")
    add(
        "consumer",
        dependencies=[{"requirement": "provider==1", "kind": "build-tool"}],
        pinned=[{"port": "provider", "version": "1", "hash": provider["artifact_sha256"]}],
    )
    env = shellsim.Environment()
    assert universe().install(env, "consumer") == {"consumer": "1", "provider": "1"}
    with pytest.raises(ValueError):
        _NativePackageUniverse(root / "catalog.json").install(env, "provider==2")
    assert env.read_file("/usr/local/bin/provider") == b"provider1"
    assert env.read_file("/usr/local/bin/consumer") == b"consumer1"
    before = env.run("stat -c %i /usr/local/bin/provider").stdout
    assert universe().install(env, "provider==1") == {"provider": "1"}
    assert universe().install(env, "provider>=1,<3") == {"provider": "1"}
    assert universe().install(env, "provider") == {"provider": "1"}
    assert env.run("stat -c %i /usr/local/bin/provider").stdout == before
    add("unrelated")
    assert universe().install(env, "unrelated") == {"unrelated": "1"}
    assert env.read_file("/usr/local/bin/provider") == b"provider1"


def test_sequential_destination_conflict_rejected(catalog):
    add, universe, _ = catalog
    add("one")
    two = add("two")
    two["destinations"] = {"tool": "/usr/local/bin/one"}
    env = shellsim.Environment()
    universe().install(env, "one")
    with pytest.raises(ValueError):
        universe().install(env, "two")
    assert env.read_file("/usr/local/bin/one") == b"one1"


def test_failed_mount_does_not_record_installed_version(catalog):
    add, universe, _ = catalog
    add("tool", "1")
    add("tool", "2", payload=b"x" * (1024 * 1024))
    env = shellsim.Environment(disk=64 * 1024)
    with pytest.raises(shellsim.SimulationError):
        universe().install(env, "tool==2")
    assert universe().install(env, "tool==1") == {"tool": "1"}


def test_modified_installed_export_is_not_silently_skipped(catalog):
    add, universe, _ = catalog
    add("tool")
    env = shellsim.Environment()
    universe().install(env, "tool")
    env.write_file("/usr/local/bin/tool", b"modified")
    with pytest.raises(ValueError):
        universe().install(env, "tool")
    assert env.read_file("/usr/local/bin/tool") == b"modified"


def test_missing_installed_export_is_not_silently_skipped(catalog):
    add, universe, _ = catalog
    add("tool")
    env = shellsim.Environment()
    universe().install(env, "tool")
    env.run("rm /usr/local/bin/tool").check_returncode()
    with pytest.raises(shellsim.SimulationError):
        universe().install(env, "tool")
