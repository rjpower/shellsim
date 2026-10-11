"""Exercise labeled dimensions and alignment without external storage."""

import xarray as xr

array = xr.DataArray([[1, 2], [3, 4]], dims=("row", "col"), coords={"row": ["a", "b"], "col": [0, 1]})
assert array.sel(row="b").values.tolist() == [3, 4]
assert array.sum(dim="col").values.tolist() == [3, 7]
assert (array + 2).values.tolist() == [[3, 4], [5, 6]]
try:
    array.sel(row="missing")
except KeyError:
    pass
else:
    raise AssertionError("missing coordinate accepted")
print("xarray: selection, reduction, arithmetic and rejection passed")
