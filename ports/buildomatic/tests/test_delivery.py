"""Exercise real offline builds, metadata-only exports and incremental promotion."""

import json
import zipfile
from pathlib import Path

import pytest

from ports.__main__ import main
from ports._support import runner
from ports._support.graph import plan
from ports._support.tests.test_runner import _wheel_port
from ports.buildomatic.delivery import DeliveryStore, export
from ports.buildomatic.delivery_products import publish_product


@pytest.fixture
def built(tmp_path):
    ports, store = tmp_path / "ports", tmp_path / "store"
    _wheel_port(ports, store / "execution/cache/ports", "base", [])
    _wheel_port(ports, store / "execution/cache/ports", "app", ["base"])
    build = runner.build_graph(ports, ["python/app"], None, store, offline=True)
    return ports, store, build


def test_offline_build_cli_writes_default_state(tmp_path, monkeypatch):
    ports = tmp_path / "ports"
    _wheel_port(ports, tmp_path / "build/ports/execution/cache/ports", "six", [])
    monkeypatch.chdir(tmp_path)
    main(["build", "python/six", "--ports", str(ports), "--offline"])
    graph = json.loads(Path("build/ports/graph.json").read_text())
    assert graph["packages"][0]["name"] == "six"
    assert (Path("build/ports/store") / graph["packages"][0]["path"]).is_file()
    main(["export", "--ports", str(ports), "--url-prefix", "http://127.0.0.1:8080"])
    catalog = json.loads(Path("build/ports/export/catalog.json").read_text())
    assert catalog["packages"][0]["url"].startswith("http://127.0.0.1:8080/assets/")


def test_export_reads_only_saved_graph(built, monkeypatch, tmp_path):
    _, state, _ = built
    (state / "results.json").unlink()
    original = Path.open

    def guard(path, *args, **kwargs):
        if path.suffix == ".whl":
            raise AssertionError("export read a wheel")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guard)
    for _ in range(2):
        result = export(state, tmp_path / "export", "https://packages.example/ports", simple=True)
        assert result == {"packages": 2, "native": 0, "assets_copied": 0}
    document = json.loads((tmp_path / "export/catalog.json").read_text())
    assert {package["name"] for package in document["packages"]} == {"base", "app"}
    assert "base-1-py3-none-any.whl" in (tmp_path / "export/simple/base/index.html").read_text()


def test_product_records_avoid_reopening_wheels(built, monkeypatch):
    from ports.buildomatic import delivery_products as delivery

    ports, state, build = built

    def reject(*args):
        raise AssertionError("warm result reopened wheel metadata")

    monkeypatch.setattr(delivery, "wheel_metadata", reject)
    runner.build_graph(ports, ["python/app"], None, state, offline=True)


def test_promotion_copies_only_new_packages(built, tmp_path):
    ports, state, _ = built
    output = tmp_path / "published"
    first = export(state, output, "https://packages.example", copy_assets=True)
    assert first["assets_copied"] == 2
    repeat = export(state, output, "https://packages.example", copy_assets=True)
    assert repeat["assets_copied"] == 0
    _wheel_port(ports, state / "execution/cache/ports", "extra", [])
    runner.build_graph(ports, ["python/app", "python/extra"], None, state, offline=True)
    added = export(state, output, "https://packages.example", copy_assets=True)
    assert added["assets_copied"] == 1
    catalog = json.loads((output / "catalog.json").read_text())
    for package in catalog["packages"]:
        assert (output / package["path"]).is_file()


def test_final_wheel_requirements_and_native_metadata(tmp_path):
    ports, store = tmp_path / "ports", tmp_path / "store"
    _wheel_port(ports, store / "execution/cache/ports", "native", [])
    port = plan(ports, ["python/native"]).ports[-1]
    root = tmp_path / "product"
    (root / "wheels").mkdir(parents=True)
    wheel = root / "wheels/native-1-cp313-cp313-wasm32_wasip1.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            "native-1.dist-info/METADATA",
            'Name: Native\nVersion: 1\nRequires-Python: >=3.13\nRequires-Dist: numpy>=2; extra == "fast"\nProvides-Extra: fast\n',
        )
        archive.writestr("native-1.dist-info/WHEEL", "Tag: cp313-cp313-wasm32_wasip1\n")
        archive.writestr(
            "native-1.dist-info/shellsim-native.json",
            json.dumps({"abi": "test-abi", "artifacts": [{"native_dependencies": ["libopenblas.so"]}]}),
        )
    record = publish_product(port, root, "native-key", DeliveryStore(store))
    package = record["wheels"][0]
    assert package["name"] == "native" and package["abi"] == "test-abi"
    assert package["requires_dist"] == ['numpy>=2; extra == "fast"']
    assert package["native_dependencies"] == ["libopenblas.so"] and package["extras"] == ["fast"]


@pytest.mark.parametrize("destination", ["/lib/libtest.so", "/usr/lib/python3.13/lib-dynload/libtest.so"])
def test_native_providers_are_emitted_with_destinations(tmp_path, destination):
    from dataclasses import replace

    ports, store = tmp_path / "ports", tmp_path / "store"
    _wheel_port(ports, store / "execution/cache/ports", "provider", [])
    port = plan(ports, ["python/provider"]).ports[-1]
    port = replace(port, recipe={**port.recipe, "install": {"destinations": {"lib/libtest.so": destination}}})
    root = tmp_path / "product"
    (root / "native/lib").mkdir(parents=True)
    (root / "native/lib/libtest.so").write_bytes(b"\x00asm\x01\x00\x00\x00")
    (root / "native/artifact.json").write_text(
        json.dumps({"inputs": {"recipe": {"abi": "test", "exports": {"shared_libraries": ["lib/libtest.so"]}}}})
    )
    record = publish_product(port, root, "provider-key", DeliveryStore(store))
    assert record["native"][0]["destination"] == destination
    assert record["native"][0]["native_dependencies"] == []


def test_guest_tools_are_emitted_as_executables(tmp_path):
    port = plan(Path(__file__).resolve().parents[2], ["native/make"]).ports[-1]
    root = tmp_path / "product"
    (root / "native/bin").mkdir(parents=True)
    (root / "native/bin/make").write_bytes(b"\x00asm\x01\x00\x00\x00")
    notice = root / "native/share/licenses/make/COPYING"
    notice.parent.mkdir(parents=True)
    notice.write_text("fixture license")
    (root / "native/artifact.json").write_text(json.dumps({"inputs": {"recipe": port.recipe}}))
    store = DeliveryStore(tmp_path / "canonical")
    record = publish_product(port, root, "make-key", store)
    tool = record["native"][0]
    assert tool["destination"] == "/usr/bin/make" and tool["kind"] == "tools"
    assert (store.local / tool["path"]).read_bytes() == b"\x00asm\x01\x00\x00\x00"


@pytest.mark.parametrize("path", ["../secret", "/secret", "a/../secret"])
def test_export_rejects_escaping_asset_paths(tmp_path, path):
    (tmp_path / "graph.json").write_text(json.dumps({"packages": [{"path": path}], "native": []}))
    with pytest.raises(ValueError):
        export(tmp_path, tmp_path / "export", "https://packages.example")
    assert not (tmp_path / "export").exists()


def test_interrupted_local_promotion_can_retry(tmp_path):
    from types import SimpleNamespace

    source = DeliveryStore(tmp_path / "unused")
    source.local = None
    source.base = "source"

    def interrupted(key, target):
        Path(target).write_bytes(b"partial")
        raise ConnectionError("interrupted")

    source.fs = SimpleNamespace(get_file=interrupted)
    output = DeliveryStore(tmp_path / "output")
    with pytest.raises(ConnectionError):
        source.promote("assets/package.whl", output)
    assert not output.exists("assets/package.whl")
    assert list((tmp_path / "output/assets").iterdir()) == []
    source.fs.get_file = lambda key, target: Path(target).write_bytes(b"complete")
    assert source.promote("assets/package.whl", output)
    assert (tmp_path / "output/assets/package.whl").read_bytes() == b"complete"


def test_guest_catalog_omits_host_and_platform_roots(built):
    from dataclasses import replace

    from ports._support.graph import Graph
    from ports.buildomatic.delivery import delivery_graph

    _, state, build = built
    base = build.graph.ports[0]
    host = replace(
        base, reference="host/tool/recipe.json", recipe={**base.recipe, "role": "host-tool"}, dependencies=()
    )
    platform = replace(
        base,
        reference="native/platform/recipe.json",
        recipe={**base.recipe, "role": "target-platform"},
        dependencies=(),
    )
    graph = Graph((host.reference, platform.reference, *build.graph.roots), (host, platform, *build.graph.ports))
    products = json.loads((state / "results.json").read_text())["products"]
    catalog = delivery_graph(graph, products)
    assert catalog["roots"] == list(build.graph.roots)
    assert {node["reference"] for node in catalog["ports"]} == set(build.results)
