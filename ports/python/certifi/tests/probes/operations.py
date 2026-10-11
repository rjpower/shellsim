"""Exercise a bounded public API inside the actual guest."""

from pathlib import Path

import certifi

assert "BEGIN CERTIFICATE" in Path(certifi.where()).read_text()
print("certifi: public API probe passed")
