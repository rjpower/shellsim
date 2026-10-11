"""Build upstream aiohttp with its declared dependency closure."""

from ports.api import BuildContext, python_pep517


def build(ctx: BuildContext):
    return python_pep517(ctx)
