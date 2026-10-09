"""Build SDK24 or SDK34 dynamic-link test fixtures, separate from runtime ports."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sdk", choices=("24", "34"), default="34")
    parser.add_argument("--runtime", type=Path)
    args = parser.parse_args()
    if args.sdk == "34":
        from sdk34 import build

        if args.runtime is None:
            parser.error("--runtime is required for SDK34 fixtures")
        build(args.bundle.resolve(), args.runtime.resolve(), args.output.resolve())
    else:
        from sdk24 import build

        build(args.bundle.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
