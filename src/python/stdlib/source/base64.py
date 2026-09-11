"""RFC 4648 helpers over shellsim's bounded byte codec core."""

from _base64 import b64decode as _b64decode
from _base64 import b64encode
from _base64 import urlsafe_b64decode as _urlsafe_b64decode
from _base64 import urlsafe_b64encode


def b64decode(value):
    if isinstance(value, str):
        value = value.encode()
    return _b64decode(value)


def urlsafe_b64decode(value):
    if isinstance(value, str):
        value = value.encode()
    return _urlsafe_b64decode(value)


standard_b64encode = b64encode
standard_b64decode = b64decode
