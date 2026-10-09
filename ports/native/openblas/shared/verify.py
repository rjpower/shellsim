"""Check the shared provider through an independent consumer in unchanged CPython."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from ports.dynamic.build import mark_abi
from ports.native.dependencies import file_hash, target_environment, target_profile, toolchain_identity, verify_artifact


def number(data, offset):
    value = 0
    for shift in range(0, 35, 7):
        byte = data[offset]
        offset += 1
        if shift == 28 and byte > 15:
            raise ValueError("overflowing Wasm metadata integer")
        value |= (byte & 127) << shift
        if byte < 128:
            return value, offset
    raise ValueError("overflowing Wasm metadata integer")


def string(data, offset):
    size, offset = number(data, offset)
    end = offset + size
    if end > len(data):
        raise ValueError("truncated Wasm metadata string")
    return data[offset:end].decode(), end


def needed_libraries(path):
    """Read LLVM's declared dependency list from the verified target artifact."""
    data = path.read_bytes()
    if data[:8] != b"\0asm\x01\0\0\0":
        raise ValueError("expected a Wasm core module")
    offset = 8
    needed = []
    while offset < len(data):
        kind = data[offset]
        size, payload = number(data, offset + 1)
        offset = payload + size
        if offset > len(data):
            raise ValueError("truncated Wasm section")
        if kind != 0:
            continue
        name, payload = string(data, payload)
        if name != "dylink.0":
            continue
        section = data[payload:offset]
        cursor = 0
        while cursor < len(section):
            kind = section[cursor]
            size, payload = number(section, cursor + 1)
            cursor = payload + size
            if cursor > len(section):
                raise ValueError("truncated dylink subsection")
            if kind != 2:
                continue
            count, payload = number(section, payload)
            for _ in range(count):
                name, payload = string(section, payload)
                needed.append(name)
            if payload != cursor:
                raise ValueError("invalid dylink dependency length")
    return needed


def build_consumer(artifact, sdk, runtime, work):
    """Link only the consumer's references, recording a real libopenblas.so dependency."""
    manifest = verify_artifact(artifact)
    recipe = manifest["inputs"]["recipe"]
    if manifest["inputs"]["toolchain"] != toolchain_identity(recipe, sdk):
        raise ValueError("consumer SDK differs from the shared provider SDK")
    base = json.loads((runtime / "manifest.json").read_text())
    if base["dynamic_abi"] != recipe["abi"] or base["recipe"]["version"] != "3.13.7":
        raise ValueError("consumer requires the fixed SDK34 CPython 3.13.7 runtime")
    source = Path(base["source_bundle"])
    profile = target_profile(recipe)
    work.mkdir(parents=True, exist_ok=True)
    consumer = work / "openblas_probe.so"
    subprocess.run(
        [
            str(sdk / "bin/clang"),
            *profile["compiler_flags"],
            *profile["cpp_flags"],
            "-fPIC",
            "-nostdlib",
            "-shared",
            "-Wl,--no-entry,--unresolved-symbols=import-dynamic,--export=PyInit_openblas_probe,--fatal-warnings",
            "-I" + str(source / "Python-3.13.7/Include"),
            "-I" + str(source / "wasi-build"),
            "-I" + str(artifact / "include"),
            str(Path(__file__).with_name("probe_extension.c")),
            "-L" + str(artifact / "lib"),
            "-lopenblas",
            "-o",
            str(consumer),
        ],
        env=target_environment(sdk),
        check=True,
    )
    mark_abi(consumer, recipe["abi"].encode())
    if needed_libraries(consumer) != [recipe["soname"]]:
        raise ValueError("consumer does not declare the shared OpenBLAS dependency")
    if needed_libraries(artifact / "lib/libopenblas.so") != recipe["needed_libraries"]:
        raise ValueError("provider has an undeclared shared-library dependency")
    return consumer


def run_probe(runtime, artifact, consumer, memory):
    import shellsim

    fixed = shellsim.CPythonRuntime(runtime)
    environment = shellsim.Environment(cpu=2_000_000_000, memory=memory, disk=128 * 1024 * 1024)
    fixed.mount(environment)
    before = environment.read_file("/usr/bin/python3.wasm")
    environment.mkdir("/lib", parents=True)
    environment.write_file("/lib/libopenblas.so", (artifact / "lib/libopenblas.so").read_bytes())
    environment.write_file(fixed.site_packages + "/openblas_probe.so", consumer.read_bytes())
    result = environment.run(
        "SHELLSIM_OPENBLAS_ABI=123 PYTHONHOME=/usr PYTHONDONTWRITEBYTECODE=1 "
        "/usr/bin/python3.wasm -c 'import openblas_probe; openblas_probe.probe()'"
    )
    if environment.read_file("/usr/bin/python3.wasm") != before:
        raise AssertionError("the fixed interpreter bytes changed")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--sdk", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--memory-mib", type=int, default=1024)
    args = parser.parse_args()
    consumer = build_consumer(args.artifact, args.sdk, args.runtime, args.work_dir)
    result = run_probe(args.runtime, args.artifact, consumer, args.memory_mib * 1024 * 1024)
    evidence = {
        "artifact_sha256": verify_artifact(args.artifact)["artifact_sha256"],
        "library_sha256": file_hash(args.artifact / "lib/libopenblas.so"),
        "consumer_sha256": file_hash(consumer),
        "needed": needed_libraries(consumer),
        "runtime_manifest_sha256": file_hash(args.runtime / "manifest.json"),
        "interpreter_sha256": file_hash(args.runtime / "rootfs/usr/bin/python3.wasm"),
        "probe_source_sha256": file_hash(Path(__file__).parent.parent / "probe.c"),
        "consumer_source_sha256": file_hash(Path(__file__).with_name("probe_extension.c")),
        "memory_limit": args.memory_mib * 1024 * 1024,
        "returncode": result.returncode,
        "stdout": result.stdout.decode(),
        "stderr": result.stderr.decode(),
        "usage": {
            "cpu_used": result.usage.cpu_used,
            "memory_peak": result.usage.memory_peak,
            "memory_current": result.usage.memory_current,
        },
    }
    (args.work_dir / "verification.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))
    if result.returncode != 0 or result.stderr:
        raise RuntimeError("shared numerical ABI guest probe failed")
    if not result.stdout.endswith(b"OpenBLAS: dgemm, dgesv, invalid input, complex and REAL ABI passed\n"):
        raise RuntimeError("shared numerical ABI probe did not complete")


if __name__ == "__main__":
    main()
