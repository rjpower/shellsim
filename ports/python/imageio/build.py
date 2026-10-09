"""Build the pinned imageio pure wheel with the reviewed WASI import adaptation."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from ports._support.build import check_build_scripts
from ports._support.pure_wheel import verified_files

DIRECTORY = Path(__file__).parent
RECIPE = json.loads((DIRECTORY / "recipe.json").read_text())


def adapted_wheel(wheel: Path, work: Path, verified: dict[str, bytes]) -> tuple[Path, dict]:
    """Disable unsupported dynamic loaders while preserving the pure PNG provider."""
    member = "imageio/core/findlib.py"
    patch = DIRECTORY / "imageio-wasi.patch"
    if hashlib.sha256(patch.read_bytes()).hexdigest() != RECIPE["adaptation"]["sha256"]:
        raise ValueError("imageio adaptation patch digest mismatch")
    destination = work / "adapted" / wheel.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    files = dict(verified)
    source = files[member].decode()
    if source.count("import ctypes\n") != 1 or source.count("    # Checks\n") != 1:
        raise ValueError("imageio dynamic loader differs from reviewed patch")
    source = source.replace("import ctypes\n", "").replace(
        "    # Checks\n",
        '    if sys.platform == "wasi":\n'
        '        raise NotImplementedError("imageio dynamic libraries are unsupported on WASI")\n\n'
        "    import ctypes\n\n    # Checks\n",
    )
    files[member] = source.encode()
    plugin = "imageio/plugins/pillow.py"
    source = files[plugin].decode()
    eager = "from PIL import ExifTags, GifImagePlugin, Image, ImageSequence, UnidentifiedImageError"
    branch = '        if self._image.format == "GIF":\n            # Converting GIF'
    if source.count(eager) != 1 or source.count(branch) != 1:
        raise ValueError("imageio Pillow plugin differs from reviewed patch")
    source = source.replace(eager, eager.replace("GifImagePlugin, ", "")).replace(
        branch,
        '        if self._image.format == "GIF":\n            from PIL import GifImagePlugin\n\n            # Converting GIF',
    )
    files[plugin] = source.encode()
    record = "imageio-2.37.0.dist-info/RECORD"
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    for name, data in sorted(files.items()):
        if name != record:
            sha = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
            writer.writerow([name, "sha256=" + sha, len(data)])
    writer.writerow([record, "", ""])
    files[record] = output.getvalue().encode()
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as derived:
        for name, data in sorted(files.items()):
            entry = zipfile.ZipInfo(name, date_time=(2025, 1, 20, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            derived.writestr(entry, data)
    return destination, {
        "patch_sha256": hashlib.sha256(patch.read_bytes()).hexdigest(),
        "wheel_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "disabled": ["dynamic-library-loading"],
    }


def build(wheel: Path, output: Path) -> Path:
    """Verify upstream bytes and preserve metadata while adapting only two modules."""
    check_build_scripts(RECIPE, DIRECTORY)
    files = verified_files(wheel, RECIPE)
    destination, evidence = adapted_wheel(wheel, output, files)
    evidence.update(recipe=RECIPE, source=RECIPE["source"], name=RECIPE["name"], version=RECIPE["version"])
    (output / "manifest.json").write_text(json.dumps(evidence, indent=2) + "\n")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(build(args.wheel, args.output))


if __name__ == "__main__":
    main()
