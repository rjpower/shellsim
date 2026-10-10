"""Stage a newly compiled threaded OpenBLAS archive and canonical-runtime side."""

import shutil
import subprocess
from pathlib import Path

from ports._support.native_adapters import NativeBuildContext
from ports.native.dependencies import target_environment


def install(context: NativeBuildContext, version: str) -> None:
    directory = Path(__file__).parent
    source = context.source
    prefix = context.staging_prefix / "usr/local"
    for child in ("lib/pkgconfig", "include", "licenses"):
        (prefix / child).mkdir(parents=True, exist_ok=True)
    archive = prefix / "lib/libopenblas.a"
    shutil.copyfile(source / "libopenblas.a", archive)
    (prefix / "include/cblas.h").write_text(
        (source / "cblas.h").read_text().replace('"common.h"', '"openblas_config.h"')
    )
    config = "\n".join(
        " ".join([parts[0], "OPENBLAS_" + parts[1], *parts[2:]])
        for line in (source / "config.h").read_text().splitlines()
        if (parts := line.split())
    )
    (prefix / "include/openblas_config.h").write_text(
        "#ifndef OPENBLAS_CONFIG_H\n#define OPENBLAS_CONFIG_H\n"
        + config
        + "\n"
        + f'#define OPENBLAS_VERSION " OpenBLAS {version} "\n'
        + (source / "openblas_config_template.h").read_text()
        + "\n#endif\n"
    )
    subprocess.run(
        [
            context.target_tools["cc"],
            *context.compiler_flags,
            *context.linker_flags,
            *context.shared_library_flags,
            "-Wl,--export-all,-soname,libopenblas.so,--fatal-warnings",
            "-Wl,--whole-archive",
            str(archive),
            "-Wl,--no-whole-archive",
            *context.shared_library_inputs,
            "-o",
            str(prefix / "lib/libopenblas.so"),
        ],
        env=target_environment(Path(context.sdk)),
        check=True,
    )
    shutil.copyfile(
        directory.parents[1] / "toolchain/wasi_sdk/notices/llvm-LICENSE.TXT",
        prefix / "licenses/compiler-rt.txt",
    )
    (prefix / "lib/pkgconfig/openblas.pc").write_text(
        "prefix=/usr/local\nlibdir=${prefix}/lib\nincludedir=${prefix}/include\n"
        "Name: OpenBLAS\nDescription: Scalar BLAS and translated LAPACK\nVersion: 0.3.31\n"
        "Libs: -L${libdir} -lopenblas\nCflags: -I${includedir}\n"
    )
