"""Exercise a bounded public API in the actual guest."""

import narwhals as nw
import pandas as pd

frame = nw.from_native(pd.DataFrame({"values": [1, 2, 3]}))
assert frame.select(nw.col("values").sum()).to_native()["values"].tolist() == [6]
print("narwhals: guest public API probe passed")
