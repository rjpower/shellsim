"""Exercise a bounded public API inside the actual guest."""

from io import StringIO

from tqdm import tqdm

output = StringIO()
assert list(tqdm(range(3), file=output, disable=True)) == [0, 1, 2]
assert output.getvalue() == ""
assert list(tqdm(range(3), file=output, mininterval=0, miniters=1, ascii=True)) == [0, 1, 2]
assert "100%" in output.getvalue()
print("tqdm: foreground iteration and real progress output passed")
