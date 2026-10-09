"""Build independent LLVM dylink proof artifacts against a trusted static CPython build.

This opt-in spike relinks one loader-enabled interpreter once, then builds two C extensions
independently. It never changes the source bundle, native dependency profiles or their hashes.
"""

from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
from pathlib import Path

from ports._support.wasm import mark_abi, run

ABI = b"shellsim-wasi-sdk24-cpython3137-v1"


def build(bundle: Path, output: Path) -> None:
    """Keep both extensions separate from the interpreter's one-time loader relink."""
    recipe = json.loads((bundle / "manifest.json").read_text())["recipe"]
    if recipe["target_profile"] != "wasi-cpython-v1" or recipe["sdk"]["version"] != "24.0":
        raise ValueError("Dynamic ABI v1 requires the static SDK 24 v1 bundle; exception profiles are unsupported")
    source = bundle / "Python-3.13.7"
    guest = bundle / "wasi-build"
    sdk = bundle / "wasi-sdk-24.0-x86_64-linux"
    clang = str(sdk / "bin/clang")
    fixtures = Path(__file__).parent
    output.mkdir(parents=True, exist_ok=True)
    bridge = output / "bridge.o"
    run([clang, "-O2", "-c", str(Path(__file__).parents[4] / "ports/toolchain/wasi_sdk/dynamic.c"), "-o", str(bridge)])
    common = [
        "-Wl,--export-all,--export-table,--growable-table,--export=__stack_pointer",
        "-Wl,--wrap=dlopen,--wrap=dlsym,--wrap=dlerror,--wrap=dlclose",
    ]
    main = output / "main.wasm"
    run([clang, "-O2", str(fixtures / "main.c"), str(bridge), *common, "-o", str(main)])
    mark_abi(main, ABI)
    library = output / "library.so"
    run(
        [
            clang,
            "-O2",
            "-fPIC",
            "-nostdlib",
            "-shared",
            "-Wl,--no-entry,--export-all",
            str(fixtures / "library.c"),
            "-o",
            str(library),
        ]
    )
    mark_abi(library, ABI)
    broken = output / "wrong-abi.so"
    # Avoid a duplicate marker: produce a separate unmarked module first.
    run(
        [
            clang,
            "-O2",
            "-fPIC",
            "-nostdlib",
            "-shared",
            "-Wl,--no-entry,--export-all",
            str(fixtures / "library.c"),
            "-o",
            str(broken),
        ]
    )
    mark_abi(broken, b"incompatible-profile")
    # Ask the existing Makefile for its exact link inputs, without rebuilding or modifying it.
    rule = "shellsim_dynamic_link:; @echo $(LINKCC) $(PY_CORE_LDFLAGS) $(LINKFORSHARED) Programs/python.o $(LINK_PYTHON_OBJS) $(LIBS) $(MODLIBS) $(SYSLIBS)"
    link = subprocess.check_output(
        ["make", "--no-print-directory", "--eval", rule, "shellsim_dynamic_link"], cwd=guest, text=True
    )
    python = output / "python3.wasm"
    run([*shlex.split(link), str(bridge), *common, "-o", str(python)], cwd=guest)
    run([str(sdk / "bin/llvm-strip"), str(python)])
    mark_abi(python, ABI)
    extensions = {}
    for name, value in [("tiny_one", 17), ("tiny_two", 40)]:
        target = output / f"{name}.so"
        run(
            [
                clang,
                "-O2",
                "-fPIC",
                "-nostdlib",
                "-shared",
                "-Wl,--no-entry",
                f"-DEXTENSION_NAME={name}",
                f"-DEXTENSION_VALUE={value}",
                f"-I{source / 'Include'}",
                f"-I{guest}",
                str(fixtures / "extension.c"),
                "-o",
                str(target),
            ]
        )
        mark_abi(target, ABI)
        extensions[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    result = {
        "abi": ABI.decode(),
        "source_bundle": str(bundle),
        "interpreter_sha256": hashlib.sha256(python.read_bytes()).hexdigest(),
        "extensions": extensions,
    }
    (output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
