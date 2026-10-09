"""Verify the pinned Magiccube package using the public guest package installer."""

import argparse
from pathlib import Path

from shellsim import CPythonRuntime, Environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--universe", type=Path, required=True)
    parser.add_argument("--uv", type=Path, required=True)
    args = parser.parse_args()
    runtime = CPythonRuntime(args.bundle, universe=args.universe, uv=args.uv)
    environment = Environment(cpu=10_000_000_000, memory=1024**3, disk=128 * 1024**2)
    runtime.mount(environment)
    runtime.install_pypi(environment, "magiccube==0.3.0")
    environment.write_file(
        "/tmp/magiccube_probe.py", (Path(__file__).parent / "tests/probes/operations.py").read_bytes()
    )
    result = runtime.run(environment, ["/tmp/magiccube_probe.py"])
    print(result.stdout_text, end="")
    if result.returncode:
        raise RuntimeError(result.stderr_text)


if __name__ == "__main__":
    main()
