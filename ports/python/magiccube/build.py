"""Copy the pinned upstream pure wheel unchanged after metadata validation."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from ports._support.build import check_build_scripts
from ports._support.pure_wheel import verified_files

PORT = Path(__file__).parent


def build(wheel: Path, output: Path) -> Path:
    recipe = json.loads((PORT / "recipe.json").read_text())
    check_build_scripts(recipe, PORT)
    if wheel.stat().st_size > 64 * 1024**2:
        raise ValueError("pure wheel exceeds build bounds")
    source = wheel.read_bytes()
    verified_files(wheel, recipe, source=source)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / wheel.name
    destination.write_bytes(source)
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "recipe": recipe,
                "wheel": destination.name,
                "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(build(args.wheel, args.output))


if __name__ == "__main__":
    main()
