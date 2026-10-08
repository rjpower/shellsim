"""Reduction registry shared by ordinary copying and package serializers."""

__all__ = ["pickle", "constructor", "dispatch_table"]

dispatch_table = {}


def pickle(ob_type, pickle_function, constructor_ob=None):
    """Register a reducer for ``ob_type`` without installing a pickle loader."""
    if not callable(pickle_function):
        raise TypeError("reduction functions must be callable")
    dispatch_table[ob_type] = pickle_function
    if constructor_ob is not None:
        constructor(constructor_ob)


def constructor(object):
    """Validate the optional legacy constructor argument to ``pickle``."""
    if not callable(object):
        raise TypeError("constructors must be callable")
