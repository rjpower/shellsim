"""Build upstream expandvars with its declared dependency closure."""

from ports.api import BuildContext, host_wheel


def build(ctx: BuildContext):
    return host_wheel(ctx)
