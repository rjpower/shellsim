"""Build Threadpoolctl with virtual provider discovery and real native controls."""

from ports.api import BuildContext, python_pep517


def build(ctx: BuildContext):
    return python_pep517(ctx, preserve_license="LICENSE")
