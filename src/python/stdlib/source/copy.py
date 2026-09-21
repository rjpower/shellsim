"""Bounded copies for the built-in container types."""


def copy(value):
    if isinstance(value, list):
        return list(value)
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, set):
        return set(value)
    if isinstance(value, tuple):
        return tuple(value)
    return value


def deepcopy(value, memo=None):
    """Recursively copy acyclic built-in containers.

    Object hooks and cyclic containers are outside this small compatibility slice.
    """
    if isinstance(value, list):
        return [deepcopy(item, memo) for item in value]
    if isinstance(value, dict):
        return {deepcopy(key, memo): deepcopy(item, memo) for key, item in value.items()}
    if isinstance(value, set):
        return {deepcopy(item, memo) for item in value}
    if isinstance(value, tuple):
        return tuple(deepcopy(item, memo) for item in value)
    return value
