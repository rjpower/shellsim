"""Exercise indexed tables, grouping, joins, CSV and virtual time zones."""

from io import StringIO

import pandas as pd

frame = pd.DataFrame({"group": ["a", "b", "a"], "value": [1, 3, 5]})
assert frame.groupby("group")["value"].sum().to_dict() == {"a": 6, "b": 3}
assert pd.read_csv(StringIO(frame.to_csv(index=False))).equals(frame)
joined = frame.merge(pd.DataFrame({"group": ["a", "b"], "label": [8, 9]}), on="group")
assert joined["label"].tolist() == [8, 9, 8]
assert pd.Timestamp("2024-01-02", tz="Europe/Berlin").hour == 0
frame.to_csv("/tmp/port-table.csv", index=False)
assert pd.read_csv("/tmp/port-table.csv").equals(frame)
try:
    pd.read_csv("/tmp/port-table.csv", memory_map=True)
except NotImplementedError:
    pass
else:
    raise AssertionError("unsupported memory mapping accepted")
try:
    frame["missing"]
except KeyError:
    pass
else:
    raise AssertionError("missing column accepted")
print("pandas: grouping, join, CSV, timezone and rejection passed")
