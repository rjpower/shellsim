"""Build make against the admitted SDK and dependencies."""

from ports.api import BuildContext, configure_make


def build(ctx: BuildContext):
    return configure_make(
        ctx,
        jobs=2,
        configure_args=[
            "--host=wasm32-wasi",
            "--build=x86_64-pc-linux-gnu",
            "--disable-nls",
            "--without-guile",
            "make_cv_synchronous_posix_spawn=yes",
        ],
        configure_environment={
            "CFLAGS": "-O2 -g0 -D_WASI_EMULATED_SIGNAL=1",
            "LDFLAGS": "-Wl,--wrap=signal,--wrap=open,--wrap=openat",
            "LIBS": "-lshellsim-posix -lwasi-emulated-signal",
            "ac_cv_func_posix_spawnattr_setsigmask": "no",
            "ac_cv_func_pselect": "no",
            "ac_cv_func_sigaction": "no",
            "ac_cv_func_sigprocmask": "no",
            "ac_cv_func_sigsetmask": "no",
            "ac_cv_have_decl_bsd_signal": "no",
        },
        build_args=[
            "CFLAGS=-O2 -g0 -D_WASI_EMULATED_SIGNAL=1 -include process_abi.h -include process_port.h -include exec_port.h -include tempfile_port.h -include posix.h"
        ],
    )
