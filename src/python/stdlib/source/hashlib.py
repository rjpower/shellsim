"""Hash objects over shellsim's one-shot native digests.

Each object buffers the bytes fed to it and digests them on demand, so the hash of a stream of
updates equals the hash of the concatenated data. The memory held is the modeled size of that
buffer.
"""

import _hashlib

__all__ = [
    "md5", "sha1", "sha256", "sha512", "new", "algorithms_guaranteed", "algorithms_available",
]

_SIZES = {"md5": (16, 64), "sha1": (20, 64), "sha256": (32, 64), "sha512": (64, 128)}

algorithms_guaranteed = frozenset(_SIZES)
algorithms_available = frozenset(_SIZES)


class _Hash:
    def __init__(self, name, data=b""):
        self.name = name
        self._data = bytes(data)

    @property
    def digest_size(self):
        return _SIZES[self.name][0]

    @property
    def block_size(self):
        return _SIZES[self.name][1]

    def update(self, data):
        self._data += bytes(data)

    def hexdigest(self):
        return _hashlib.hexdigest(self.name, self._data)

    def digest(self):
        return bytes.fromhex(self.hexdigest())

    def copy(self):
        return _Hash(self.name, self._data)

    def __repr__(self):
        return "<%s _hashlib.HASH object @ 0x%x>" % (self.name, id(self))


def new(name, data=b"", *, usedforsecurity=True):
    name = name.lower()
    if name not in _SIZES:
        raise ValueError("unsupported hash type " + name)
    return _Hash(name, data)


def md5(data=b"", *, usedforsecurity=True):
    return _Hash("md5", data)


def sha1(data=b"", *, usedforsecurity=True):
    return _Hash("sha1", data)


def sha256(data=b"", *, usedforsecurity=True):
    return _Hash("sha256", data)


def sha512(data=b"", *, usedforsecurity=True):
    return _Hash("sha512", data)


def file_digest(fileobj, digest, /, *, _bufsize=2**18):
    digestobj = new(digest) if isinstance(digest, str) else digest()
    while True:
        chunk = fileobj.read(_bufsize)
        if not chunk:
            break
        digestobj.update(chunk)
    return digestobj
