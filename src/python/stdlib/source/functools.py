"""Small capability-free functools layer over the native bounded reduce loop."""

from _functools import reduce


class partial:
    def __init__(self, func, *args):
        if not callable(func):
            raise TypeError("the first argument must be callable")
        self.func = func
        self.args = args
        self.keywords = {}

    def __call__(self, *args):
        return self.func(*(self.args + args))
