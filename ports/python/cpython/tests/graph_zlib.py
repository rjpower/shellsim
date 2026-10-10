"""Exercise upstream zlib, streaming/gzip, errors and independent guest threads."""

import gzip
import threading
import zlib

payload = b"upstream threaded stdlib zlib" * 4096
assert zlib.decompress(zlib.compress(payload)) == payload
stream = zlib.compressobj()
compressed = stream.compress(payload[:12345]) + stream.compress(payload[12345:]) + stream.flush()
assert zlib.decompress(compressed) == payload
assert gzip.decompress(gzip.compress(payload, mtime=0)) == payload
assert zlib.crc32(b"123456789") == 0xCBF43926
try:
    zlib.decompress(b"invalid stream")
except zlib.error:
    pass
else:
    raise AssertionError("corrupt zlib stream accepted")
results = []


def worker():
    results.append(zlib.decompress(zlib.compress(payload)) == payload)


threads = [threading.Thread(target=worker) for _ in range(2)]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()
assert results == [True, True]
print("upstream zlib roundtrip, streaming, gzip, CRC, errors and threads passed")
