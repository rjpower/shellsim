"""Exercise a bounded public API in the actual guest."""

from pathlib2 import Path

path = Path("/tmp/port-pathlib2.txt")
assert path.write_text("virtual") == 7
assert path.read_text() == "virtual"
assert path.parent == Path("/tmp")
print("pathlib2: guest public API probe passed")
