"""Build upstream Tqdm with foreground progress enabled on WASI."""

from ports.api import BuildContext, python_pep517


def build(ctx: BuildContext):
    return python_pep517(ctx)
