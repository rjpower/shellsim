"""Build shellsim-posix against the admitted SDK and dependencies."""

from ports.api import BuildContext, cmake


def build(ctx: BuildContext):
    return cmake(ctx, jobs=2)
