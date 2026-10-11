"""Build upstream aiohappyeyeballs with its declared dependency closure."""

from ports.api import BuildContext, pure_wheel


def build(ctx: BuildContext):
    return pure_wheel(ctx)
