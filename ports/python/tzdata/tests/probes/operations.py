"""Load the wheel's timezone data explicitly rather than host zone files."""

from datetime import datetime
from importlib.resources import files
from zoneinfo import ZoneInfo

with files("tzdata.zoneinfo.Europe").joinpath("Berlin").open("rb") as stream:
    zone = ZoneInfo.from_file(stream, key="Europe/Berlin")
assert datetime(2024, 1, 2, tzinfo=zone).utcoffset().total_seconds() == 3600
print("tzdata: bundled Berlin timezone passed")
