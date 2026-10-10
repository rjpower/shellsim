#!/usr/bin/env python3
"""Measure installed guest Clang without rebuilding LLVM or its package artifacts.

Run in separate processes against the same release and cache to distinguish process-local
reuse from persistent JIT reuse. Each repetition rebuilds and verifies the same small C
project. Timings are host observations, never guest clock values.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import time
from pathlib import Path

import shellsim

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', required=True, type=Path)
    parser.add_argument('--cache', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--repeat', type=int, default=2)
    parser.add_argument('--make', action='store_true')
    parser.add_argument('--linker-only', action='store_true')
    args = parser.parse_args()
    records = []
    started = time.perf_counter()
    env = shellsim.Environment.from_release(
        args.release.resolve(),
        tools=['clang==23.1.0rc3', 'zlib==1.3.1', 'make==4.4.1'],
        cache_dir=args.cache.resolve(),
        limits=shellsim.Limits(cpu=500_000_000_000, memory=8 * 1024**3, disk=805_306_368),
    )
    setup_seconds = time.perf_counter() - started
    clang = env.read_file('/usr/bin/clang')
    identity = {'bytes': len(clang), 'sha256': hashlib.sha256(clang).hexdigest()}
    for name in ('main.c', 'codec.c'):
        env.write_file('/work/' + name, (ROOT / 'tests/fixtures/native_make' / name).read_bytes())
    makefile = (ROOT / 'tests/fixtures/native_make/Makefile').read_text().replace('/opt/zlib', '/usr/local').replace('cc ', 'clang ')
    env.write_file('/work/Makefile', makefile.encode())
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def measure(name: str, command: str, repetition: int) -> None:
        started = time.perf_counter()
        result = env.run(command)
        records.append({
            'phase': name,
            'repetition': repetition,
            'seconds': time.perf_counter() - started,
            'command': command,
            'returncode': result.returncode,
            'stop_reason': result.stop_reason,
            'usage': dataclasses.asdict(result.usage),
            'stdout': result.stdout.decode('utf-8'),
            'stderr': result.stderr.decode('utf-8'),
        })
        args.output.write_text(json.dumps({
            'setup_seconds': setup_seconds, 'clang': identity, 'records': records,
        }, indent=2) + '\n')
        print(f'{name} [{repetition}]: {records[-1]["seconds"]:.3f}s', flush=True)
        if result.returncode != 0 or result.stop_reason is not None:
            raise RuntimeError(f'{name} failed: {result.stderr!r}')
        if name == 'run' and not result.stdout.endswith(
            b'zlib 1.3.1: roundtrip, CRC32, invalid input passed\n'
        ):
            raise AssertionError(result.stdout)

    for repetition in range(args.repeat):
        if args.linker_only:
            measure('load-linker-and-version', 'wasm-ld --version', repetition)
            continue
        if args.make:
            result = env.run('cd /work && rm -f main.o codec.o codec.wasm')
            if result.returncode != 0:
                raise RuntimeError('cannot clear prior project outputs')
            measure('make', 'cd /work && make -j2', repetition)
            measure('run', 'cd /work && ./codec.wasm', repetition)
            continue
        measure('load-and-version', 'clang --version', repetition)
        measure('load-linker-and-version', 'wasm-ld --version', repetition)
        for name in ('main', 'codec'):
            measure('compile-' + name,
                    f'cd /work && clang -I/usr/local/include -c {name}.c -o {name}.o', repetition)
        measure('link', 'cd /work && clang -o codec.wasm main.o codec.o -L/usr/local/lib -lz', repetition)
        measure('run', 'cd /work && chmod +x codec.wasm && ./codec.wasm', repetition)


if __name__ == '__main__':
    main()
