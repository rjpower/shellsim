"""Exercise a bounded public API inside the actual guest."""

from charset_normalizer import from_bytes

assert str(from_bytes("héllo".encode()).best()) == "héllo"
print("charset-normalizer: public API probe passed")
