import base64
import codecs
import hashlib
import struct
import zlib

value = b"\x00\xffhello"
print(repr(value), list(value), value.hex())
encoded = base64.b64encode(value)
print(encoded, base64.b64decode(encoded))
packed = struct.pack(">Hi", 513, -7)
print(packed, struct.unpack(">Hi", packed))
print(zlib.decompress(zlib.compress(value)), zlib.crc32(b"123456789"))
print(hashlib.sha256(b"abc").hexdigest())
print(codecs.decode("Uryyb", "rot_13"))
