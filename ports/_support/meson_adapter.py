"""Configure Meson cross projects and retain verified compilation state."""

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from ports._support.native_adapters import NativeBuildRequest, write_build_file


def _literal(value: object) -> str:
    return repr(str(value))


def build_meson(
    request: NativeBuildRequest,
    wrappers: Mapping[str, Path],
    tools: Path,
    run: Callable[[Sequence[object], Path], None],
) -> None:
    context = request.context
    python = context.host_tools["python"]
    cross = tools / "cross.ini"
    write_build_file(
        cross,
        "[binaries]\nc = "
        + _literal(wrappers["cc"])
        + "\ncpp = "
        + _literal(wrappers["cxx"])
        + "\nar = "
        + _literal(context.target_tools["ar"])
        + "\nstrip = "
        + _literal(context.target_tools["strip"])
        + "\npkg-config = "
        + _literal(context.host_tools["pkg-config"])
        + "".join(
            "\n" + name + " = " + _literal(context.host_tools[name])
            for name in ("cython", "f2py")
            if name in context.host_tools
        )
        + "\n[host_machine]\nsystem = 'wasi'\ncpu_family = 'wasm32'\ncpu = 'wasm32'\nendian = 'little'\n[properties]\nneeds_exe_wrapper = true\nsys_root = "
        + _literal(context.dependency_sysroot)
        + "\n"
        + "".join(
            name + " = " + (str(value).lower() if isinstance(value, (bool, int)) else _literal(value)) + "\n"
            for name, value in sorted(request.meson_properties.items())
        ),
    )
    native_file = tools / "native.ini"
    write_build_file(
        native_file,
        "[binaries]\npython = "
        + _literal(python)
        + "\n"
        + "".join(
            name + " = " + _literal(context.host_tools[name]) + "\n"
            for name in ("cython", "f2py")
            if name in context.host_tools
        ),
    )
    meson = context.host_tools["meson"]
    build = context.build / "meson-build"
    if not context.retained_workspace or not (build / "meson-private/coredata.dat").exists():
        run(
            [
                meson,
                "setup",
                build,
                context.source,
                *request.configure_args,
                "--cross-file",
                cross,
                "--native-file",
                native_file,
                "--prefix",
                request.install_prefix,
                "--wrap-mode=nodownload",
            ],
            context.build,
        )
    if request.meson_install_tags:
        plan = json.loads((build / "meson-info/intro-install_plan.json").read_text())
        selected = []
        for source, spec in plan.get("targets", {}).items():
            if spec["tag"] in request.meson_install_tags:
                path = Path(source)
                if not path.is_absolute() or not path.is_relative_to(build):
                    raise ValueError("Meson install target escapes build directory")
                selected.append(str(path.relative_to(build)))
        if not selected:
            raise ValueError("Meson install tags select no build targets")
        run([context.host_tools["ninja"], "-C", build, "-j", request.jobs, *sorted(selected)], context.build)
    else:
        run([meson, "compile", "-C", build, "-j", request.jobs, *request.build_targets], context.build)
    run(
        [
            meson,
            "install",
            "-C",
            build,
            "--no-rebuild",
            *(["--tags=" + ",".join(request.meson_install_tags)] if request.meson_install_tags else []),
        ],
        context.build,
    )
