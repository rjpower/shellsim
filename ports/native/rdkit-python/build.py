"""Build one shared RDKit chemistry provider for the Python wrapper cohort."""

from ports.native.rdkit_bindings import build_support


def build(ctx):
    return build_support(ctx, bindings=False)
