"""Build the upstream pure YAML parser without its optional LibYAML extension."""

from ports.api import BuildContext, python_pep517


def build(ctx: BuildContext):
    return python_pep517(ctx, preserve_license="LICENSE")
