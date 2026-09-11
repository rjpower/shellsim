"""Capability-free subset of abc used by ordinary class declarations."""


class ABC:
    pass


ABCMeta = type


def abstractmethod(function):
    return function
