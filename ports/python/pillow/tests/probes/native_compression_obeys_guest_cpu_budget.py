import zlib

print("compression loop entered", flush=True)
payload = bytes(range(256)) * 256
while True:
    zlib.compress(payload)
