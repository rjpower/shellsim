"""Exercise a bounded public API inside the actual guest."""

import idna

assert idna.encode("bücher.example") == b"xn--bcher-kva.example"
assert idna.decode(b"xn--bcher-kva.example") == "bücher.example"
try:
    idna.encode("bad domain")
except idna.IDNAError:
    pass
else:
    raise AssertionError("invalid domain accepted")
print("idna: public API probe passed")
