"""Play the precompiled Doom port through shellsim's existing browser viewer."""

from __future__ import annotations

import argparse
import sys
import time

import shellsim


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, help="ports catalog")
    args = parser.parse_args()
    environment = shellsim.Context(pypi_catalog=args.catalog).Environment(tools=["doom"], clock="real_time", limits=shellsim.Limits(cpu=400_000_000_000, memory=1024**3, disk=128 * 1024**2, output=4 * 1024**2))
    container = environment.to_container()
    with container.start("doom -iwad /usr/share/games/doom/freedoom1.wad") as action:
        with shellsim.DisplayHost(action) as viewer:
            print(f"shellsim display: {viewer.url}", flush=True)
            while True:
                event = action.poll()
                sys.stdout.buffer.write(event.stdout)
                sys.stderr.buffer.write(event.stderr)
                if event.frame is not None:
                    viewer.publish(event.frame)
                for code, pressed in viewer.drain_keys():
                    action.inject_key(code, pressed)
                if viewer.stop_requested():
                    action.stop()
                if event.state == "complete":
                    viewer.mark_stopped()
                    return event.returncode if event.returncode is not None else 1
                if event.state == "blocked":
                    time.sleep(0.002)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
