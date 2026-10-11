"""Build port graphs or export their delivery metadata."""

import argparse
import sys
from typing import Sequence


def main(arguments: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ports", description=__doc__)
    parser.add_argument("command", choices=("build", "export"))
    arguments = list(sys.argv[1:] if arguments is None else arguments)
    command = parser.parse_args(arguments[:1]).command
    if command == "build":
        from ports._support.runner import main as build

        build(arguments[1:])
    else:
        from ports.buildomatic.delivery import main as export

        export(arguments[1:])


if __name__ == "__main__":
    main()
