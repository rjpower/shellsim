"""Generate upstream array/image bindings over the admitted OpenCV archives."""

import ast
import email
import json
import re
import subprocess
from dataclasses import replace

from ports.api import BuildContext, python_extension


def build(ctx: BuildContext):
    native = ctx.require_native()
    source = native.source / "opencv"
    bindings = source / "modules/python/src2"
    headers = []
    custom = []
    for module in ("core", "flann", "geometry", "imgproc"):
        root = source / "modules" / module
        headers.extend(
            sorted(
                (root / "include").rglob("*.hpp"),
                key=lambda path: (path.name == "bindings.hpp", len(path.parts), path.as_posix()),
            )
        )
        headers.extend(sorted((root / "misc/python").glob("python_*.hpp")))
        headers.extend(sorted((root / "misc/python").glob("shadow*.hpp")))
        custom.extend(sorted((root / "misc/python").glob("python_*.hpp")))
        custom.extend(sorted((root / "misc/python").glob("pyopencv*.hpp")))
    # Match upstream's binding-generator exclusions for private/header-only APIs.
    headers = [
        path
        for path in headers
        if not any(
            value in path.as_posix()
            for value in (
                "/cuda/",
                "/hal/",
                "/opencl/",
                "/private/",
                "/legacy/",
                "fast_math.hpp",
                "utils/trace.hpp",
                "_inl.hpp",
                ".inl.hpp",
                ".details.hpp",
                ".private.hpp",
            )
        )
    ]
    modules = native.dependencies["native/opencv"] / "include/opencv5/opencv2/opencv_modules.hpp"
    definitions = dict.fromkeys(re.findall(r"^#define\s+(HAVE_OPENCV_\w+)", modules.read_text(), re.MULTILINE), 1)
    # These are the upstream binding parser's settings, independent of codegen.
    definitions.update(
        {
            "CV_VERSION_MAJOR": 5,
            "CV_VERSION_MINOR": 0,
            "CV_VERSION_PATCH": 0,
            "OPENCV_ABI_COMPATIBILITY": 500,
            "OPENCV_BINDINGS_PARSER": 1,
            "OPENCV_HAVE_FILESYSTEM_SUPPORT": 1,
            **dict.fromkeys(
                (
                    "EIGEN_WORLD_VERSION",
                    "EIGEN_MAJOR_VERSION",
                    "EIGEN_MINOR_VERSION",
                    "CV_ENABLE_UNROLLED",
                    "CV__EXCEPTION_PTR",
                    "CV_NEON",
                    "TBB_INTERFACE_VERSION",
                    "CV_SSE2",
                    "CV_VSX",
                    "OPENCV_SUPPORTS_FP_DENORMALS_HINT",
                    "CV_LOG_STRIP_LEVEL",
                    "CV_LOG_LEVEL_SILENT",
                    "CERES_FOUND",
                ),
                0,
            ),
            **{
                "CV_LOG_LEVEL_" + name: i
                for i, name in enumerate(("FATAL", "ERROR", "WARN", "INFO", "DEBUG", "VERBOSE"), 1)
            },
        }
    )
    config = bindings / "shellsim-generator.json"
    config.write_text(json.dumps({"headers": [str(path) for path in headers], "preprocessor_definitions": definitions}))
    subprocess.run(
        [
            str(native.host_tools["python"]),
            str(bindings / "gen2.py"),
            "--config",
            str(config),
            "--output_dir",
            str(bindings),
        ],
        check=True,
    )
    generated_include = bindings / "pyopencv_generated_include.h"
    generated_include.write_text("#include <iterator>\n" + generated_include.read_text())
    (bindings / "pyopencv_custom_headers.h").write_text("".join('#include "' + str(path) + '"\n' for path in custom))
    # The sdist's old PKG-INFO omits dependencies that setup.py declares.
    metadata = email.message_from_bytes((native.source / "PKG-INFO").read_bytes())
    setup = ast.parse((native.source / "setup.py").read_text())
    requirements = next(
        ast.literal_eval(node.value)
        for node in ast.walk(setup)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "install_requires" for target in node.targets)
    )
    for requirement in requirements:
        metadata["Requires-Dist"] = requirement
    (native.source / "shellsim-PKG-INFO").write_bytes(metadata.as_bytes())
    # Upstream binding utilities include core private headers from the source tree.
    ctx = replace(
        ctx,
        native=replace(
            native,
            compiler_flags=(
                *native.compiler_flags,
                "-I" + str(source / "modules/core/include"),
            ),
        ),
    )
    return python_extension(
        ctx,
        module="cv2",
        metadata="shellsim-PKG-INFO",
        sources=tuple(
            "opencv/modules/python/src2/" + name
            for name in (
                "cv2.cpp",
                "cv2_util.cpp",
                "cv2_numpy.cpp",
                "cv2_convert.cpp",
                "cv2_highgui.cpp",
            )
        ),
        include_directories=("include", "include/opencv5", "python/numpy/_core/include"),
        link_inputs=(
            "lib/libopencv_imgproc.a",
            "lib/libopencv_core.a",
            "lib/libopencv_geometry.a",
            "lib/libopencv_flann.a",
            "lib/opencv5/3rdparty/libzlib.a",
        ),
        licenses=("LICENSE.txt", "LICENSE-3RD-PARTY.txt"),
    )
