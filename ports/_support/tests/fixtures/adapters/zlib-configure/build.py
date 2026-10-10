"""Exercise the alternate upstream adapter in tests."""

from ports.api import BuildContext, configure_make


def build(ctx: BuildContext):
    return configure_make(ctx, **{"configure_args": ["--static"], "install_prefix": "/usr/local", "jobs": 2})
