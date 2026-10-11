"""Exercise fixed date parsing and calendar arithmetic without host time."""

from datetime import datetime

from dateutil.parser import isoparse
from dateutil.relativedelta import relativedelta

assert isoparse("2024-02-29T12:34:56+00:00").isoformat() == "2024-02-29T12:34:56+00:00"
assert datetime(2024, 1, 31) + relativedelta(months=1) == datetime(2024, 2, 29)
try:
    isoparse("2024-02-30")
except ValueError:
    pass
else:
    raise AssertionError("invalid calendar date accepted")
print("python-dateutil: fixed dates and month arithmetic passed")
