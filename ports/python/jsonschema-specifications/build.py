"""Stage the official jsonschema-specifications wheel unchanged."""

from ports.api import BuildContext, pure_wheel


def build(ctx: BuildContext):
    return pure_wheel(ctx)
