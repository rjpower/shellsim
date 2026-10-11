"""Exercise real bzip2 streaming, virtual file I/O and malformed input."""

import bz2

payload = b"upstream bzip2 on the virtual filesystem\n" * 256
assert bz2.decompress(bz2.compress(payload)) == payload
compressor = bz2.BZ2Compressor()
encoded = compressor.compress(payload[:311]) + compressor.compress(payload[311:]) + compressor.flush()
decompressor = bz2.BZ2Decompressor()
assert decompressor.decompress(encoded) == payload
assert decompressor.eof
try:
    bz2.decompress(b"corrupt compressed input")
except OSError:
    pass
else:
    raise AssertionError("corrupt bzip2 stream accepted")
with bz2.open("/work/payload.bz2", "wb") as stream:
    stream.write(payload)
with bz2.open("/work/payload.bz2", "rb") as stream:
    assert stream.read() == payload
print("bz2: guest compression, streaming, files and errors passed")
