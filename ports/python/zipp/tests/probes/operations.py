"""Exercise a bounded public API in the actual guest."""

from io import BytesIO
from zipfile import ZipFile

import zipp

buffer = BytesIO()
with ZipFile(buffer, "w") as archive:
    archive.writestr("folder/value.txt", "content")
buffer.seek(0)
path = zipp.Path(ZipFile(buffer)) / "folder" / "value.txt"
assert path.read_text() == "content"
print("zipp: guest public API probe passed")
