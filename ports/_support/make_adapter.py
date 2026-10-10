"""Build configure/Make projects with admitted host generators and target tools."""

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from ports._support.native_adapters import NativeAdapter, NativeBuildRequest


def build_make(
    request: NativeBuildRequest, wrappers: Mapping[str, Path], run: Callable[[Sequence[object], Path], None]
) -> None:
    context = request.context
    if request.adapter is NativeAdapter.PLAIN_MAKE:
        make = context.host_tools["make"]
        bindings = [
            "CC=" + str(wrappers["cc"]),
            "CXX=" + str(wrappers["cxx"]),
            "HOSTCC=" + str(context.host_tools["cc"]),
            "AR=" + str(context.target_tools["ar"]),
            "RANLIB=" + str(context.target_tools["ranlib"]),
        ]
        # Targets may append to one archive; preserve recipe order between them.
        for target in request.build_targets or (None,):
            run(
                [make, "-j", request.jobs, *bindings, *request.build_args, *([target] if target else [])],
                context.source,
            )
        if request.install_targets:
            run(
                [
                    make,
                    *bindings,
                    *request.build_args,
                    *request.install_targets,
                    "PREFIX=" + str(request.install_prefix),
                    "DESTDIR=" + str(context.staging_prefix),
                ],
                context.source,
            )
    else:
        build = context.build / "configure-build"
        build.mkdir()
        run(
            [
                context.host_tools["sh"],
                context.source / "configure",
                *request.configure_args,
                "--prefix=" + str(request.install_prefix),
            ],
            build,
        )
        make = context.host_tools["make"]
        run([make, "-j", request.jobs, *request.build_args, *request.build_targets], build)
        run([make, *request.build_args, *request.install_targets, "DESTDIR=" + str(context.staging_prefix)], build)
