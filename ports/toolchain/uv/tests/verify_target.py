"""Exercise WASI marker and wheel selection with an offline Simple API index."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import tempfile
from html import escape
from pathlib import Path
from zipfile import ZipFile

import tomllib


def wheel(index: Path, name: str, tag: str, requirements: list[str] | None = None) -> str:
    normalized = name.replace("-", "_")
    filename = f"{normalized}-1.0-{tag}.whl"
    package_dir = index / name
    package_dir.mkdir(parents=True, exist_ok=True)
    dist_info = f"{normalized}-1.0.dist-info"
    metadata = f"Metadata-Version: 2.3\nName: {name}\nVersion: 1.0\nRequires-Python: >=3.13\n"
    for requirement in requirements or []:
        metadata += f"Requires-Dist: {requirement}\n"
    with ZipFile(package_dir / filename, "w") as archive:
        archive.writestr(f"{dist_info}/METADATA", metadata)
        archive.writestr(
            f"{dist_info}/WHEEL",
            f"Wheel-Version: 1.0\nGenerator: shellsim-test\nRoot-Is-Purelib: {str(tag == 'py3-none-any').lower()}\nTag: {tag}\n",
        )
        if tag == "cp313-cp313-wasm32_wasip1":
            archive.writestr(f"{normalized}.cpython-313-wasm32-wasi.so", b"\x00asm\x01\x00\x00\x00")
        else:
            archive.writestr(f"{normalized}/__init__.py", "VALUE = 1\n")
        archive.writestr(f"{dist_info}/RECORD", "")
    return filename


def write_index(package_dir: Path) -> None:
    links = "".join(
        f'<a href="{escape(path.name)}#sha256={hashlib.sha256(path.read_bytes()).hexdigest()}">'
        f"{escape(path.name)}</a>\n"
        for path in sorted(package_dir.glob("*.whl"))
    )
    (package_dir / "index.html").write_text(links)


def run(uv: Path, *arguments: str, cwd: Path) -> str:
    result = subprocess.run(
        [str(uv), *arguments],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise AssertionError(f"uv {' '.join(arguments)} failed:\n{result.stderr}")
    return result.stdout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("uv", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="shellsim-uv-wasi-") as directory:
        root = Path(directory)
        index = root / "index"
        wheel(index, "native-fixture", "cp313-cp313-wasm32_wasip1")
        wheel(index, "native-fixture", "cp313-cp313-manylinux_2_28_x86_64")
        wheel(index, "native-fixture", "cp313-cp313-pyodide_2025_0_wasm32")
        wheel(index, "native-fixture", "cp312-cp312-wasm32_wasip1")
        wheel(index, "wasi-only", "py3-none-any")
        wheel(index, "patch-only", "py3-none-any")
        wheel(index, "linux-only", "py3-none-any")
        wheel(index, "pyodide-only", "py3-none-any")
        wheel(
            index,
            "selector",
            "py3-none-any",
            [
                "native-fixture==1.0; sys_platform == 'wasi' and platform_machine == 'wasm32'",
                "wasi-only==1.0; platform_system == 'wasi' and platform_release == '0.0.0' and platform_version == '0.0.0'",
                "patch-only==1.0; python_full_version == '3.13.7' and implementation_version == '3.13.7' and platform_python_implementation == 'CPython' and os_name == 'posix'",
                "linux-only==1.0; sys_platform == 'linux'",
                "pyodide-only==1.0; sys_platform == 'emscripten'",
            ],
        )
        for package_dir in index.iterdir():
            write_index(package_dir)
        (root / "requirements.txt").write_text("selector==1.0\n")
        compiled = run(
            args.uv,
            "pip",
            "compile",
            "requirements.txt",
            "--python-platform",
            "wasm32-wasip1",
            "--python",
            sys.executable,
            "--no-python-downloads",
            "--python-version",
            "3.13.7",
            "--only-binary",
            ":all:",
            "--index-url",
            index.as_uri(),
            "--offline",
            "--no-cache",
            cwd=root,
        )
        names = {line.split("==")[0] for line in compiled.splitlines() if "==" in line and not line.startswith("#")}
        assert names == {"native-fixture", "selector", "wasi-only", "patch-only"}, compiled
        (root / "pyproject.toml").write_text(
            "[project]\nname = 'wasi-lock-probe'\nversion = '0.0.0'\n"
            "requires-python = '==3.13.*'\ndependencies = ['selector==1.0']\n"
            "[tool.uv]\nenvironments = [\"sys_platform == 'wasi' and platform_machine == 'wasm32'\"]\n"
            "[[tool.uv.index]]\nname = 'native'\ndefault = true\n"
            f"url = '{index.as_uri()}'\n"
        )
        run(args.uv, "lock", "--offline", "--no-cache", cwd=root)
        lock = tomllib.loads((root / "uv.lock").read_text())
        packages = {package["name"]: package for package in lock["package"]}
        assert "native-fixture" in packages, packages.keys()
        assert "wasi-only" in packages, packages.keys()
        assert "linux-only" not in packages, packages.keys()
        assert "pyodide-only" not in packages, packages.keys()
        urls = [wheel.get("url", wheel.get("path")) for wheel in packages["native-fixture"]["wheels"]]
        assert len(urls) == 1 and urls[0].endswith("cp313-cp313-wasm32_wasip1.whl"), urls
        expected = hashlib.sha256((index / "native-fixture" / Path(urls[0]).name).read_bytes()).hexdigest()
        assert packages["native-fixture"]["wheels"][0]["hash"] == f"sha256:{expected}"
    print("WASI target selected CPython 3.13.7 native and pure wheels with WASI markers")


if __name__ == "__main__":
    main()
