from collections import Counter, deque
from datetime import datetime, timedelta
import json
import os
import zlib

counts = Counter("abaca")
print(counts["a"], counts["z"], counts.total())
values = deque([1, 2])
values.appendleft(0)
values.rotate(1)
print(list(values))
print(json.dumps({"a": [1, 2]}, indent=2))
print(os.path.join("/tmp", "a", "b.txt"), os.path.splitext("x.tar"))
print((datetime.strptime("2024-02-28", "%Y-%m-%d") + timedelta(days=2)).strftime("%Y-%m-%d"))
print(zlib.crc32(b"123456789"))
