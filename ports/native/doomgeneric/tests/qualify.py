"""Qualify compiled artifacts through the existing interactive Container API.

Compare equal virtual frame generations with and without input, so ordinary
animation cannot satisfy the input assertion. No source compiler is installed.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Callable

import shellsim

LIMITS = shellsim.Limits(cpu=20_000_000_000, memory=1024 * 1024 * 1024, disk=128 * 1024 * 1024, output=1024 * 1024)
COMMAND = "doom -iwad /usr/share/games/doom/freedoom1.wad -warp 1 1 -skill 1 -nomonsters -nosound -nomusic"


def container(doom: bytes, wad: bytes, limits: shellsim.Limits = LIMITS) -> shellsim.Container:
    guest = shellsim.Container(tools={}, limits=limits, clock="virtual")
    guest.write_file("/usr/bin/doom", doom, mode=0o755)
    guest.write_file("/usr/share/games/doom/freedoom1.wad", wad)
    return guest


def play(guest: shellsim.Container, *, turn: bool) -> bytes:
    with guest.start(COMMAND) as action:
        frames = 0
        output = bytearray()
        for _ in range(20_000):
            event = action.poll()
            output.extend(event.stdout + event.stderr)
            if event.state == "complete":
                raise AssertionError(
                    f"Doom exited before gameplay: {event.returncode}, {output.decode(errors='replace')}"
                )
            if event.frame is None:
                continue
            frame = event.frame
            assert (frame.width, frame.height) == (640, 400)
            frames += 1
            if frames == 1:
                assert len(set(frame.pixels[0::4])) > 8, "empty or uniform frame"
            if frames == 80:
                if turn:
                    action.inject_key(61444, True)
            if frames == 160:
                action.stop()
                assert action.poll().state == "complete"
                assert action.result.returncode != 0, "cancellation returned success"
                assert guest.run("printf alive").stdout == b"alive", "cancel left the container unusable"
                return frame.pixels
        raise AssertionError("Doom did not produce enough frames within the poll bound")


def qualify(make_guest: Callable[[shellsim.Limits], shellsim.Container]) -> None:
    baseline = play(make_guest(LIMITS), turn=False)
    turned = play(make_guest(LIMITS), turn=True)
    assert baseline != turned, "key input did not change gameplay"
    assert baseline == play(make_guest(LIMITS), turn=False), "virtual gameplay is nondeterministic"
    constrained = make_guest(shellsim.Limits(cpu=100_000, memory=LIMITS.memory, disk=LIMITS.disk, output=LIMITS.output))
    result = constrained.run(COMMAND)
    assert result.returncode != 0 and result.stop_reason == "cpu_exhausted", "Doom escaped the CPU budget"
    constrained = make_guest(
        shellsim.Limits(cpu=LIMITS.cpu, memory=64 * 1024**2, disk=LIMITS.disk, output=LIMITS.output)
    )
    result = constrained.run(COMMAND)
    assert result.returncode != 0 and result.stop_reason == "memory_exhausted", "Doom escaped the memory budget"
    print("Doom boot, frames, input, cancellation, repeatability and CPU/memory budgets passed")
    print("baseline frame SHA-256:", hashlib.sha256(baseline).hexdigest())
    print("turned frame SHA-256:", hashlib.sha256(turned).hexdigest())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", help="qualify the installed tool/data closure using the parent delivery bridge")
    parser.add_argument("--doom", type=Path, help="host-SDK compiled bin/doom artifact")
    parser.add_argument("--wad", type=Path, help="pinned Freedoom Phase 1 IWAD")
    args = parser.parse_args()
    if args.catalog:
        if args.doom or args.wad:
            parser.error("choose --catalog or --doom with --wad")
        qualify(
            lambda limits: shellsim.Context(pypi_catalog=args.catalog)
            .Environment(tools=["doom"], limits=limits)
            .to_container()
        )
        return
    if args.doom is None or args.wad is None:
        parser.error("choose --catalog or --doom with --wad")
    doom, wad = args.doom.read_bytes(), args.wad.read_bytes()
    qualify(lambda limits: container(doom, wad, limits))


if __name__ == "__main__":
    main()
