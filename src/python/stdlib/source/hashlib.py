"""Incremental hash facade backed by shellsim's exact one-shot primitives."""

import _hashlib


class _Hash:
    def __init__(self, name, data=""):
        self.name = name
        self._data = data

    def update(self, data):
        self._data += data

    def hexdigest(self):
        return _hashlib.hexdigest(self.name, self._data)

    def copy(self):
        return _Hash(self.name, self._data)


def md5(data=""):
    return _Hash("md5", data)


def sha1(data=""):
    return _Hash("sha1", data)


def sha256(data=""):
    return _Hash("sha256", data)


def sha512(data=""):
    return _Hash("sha512", data)
