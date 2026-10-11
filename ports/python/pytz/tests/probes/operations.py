"""Use bundled time zones with a fixed date, independent of the host clock."""

from datetime import datetime, timedelta

import pytz

berlin = pytz.timezone("Europe/Berlin")
value = berlin.localize(datetime(2020, 1, 1, 12))
assert value.utcoffset() == timedelta(hours=1)
assert value.astimezone(pytz.UTC).hour == 11
try:
    pytz.timezone("Missing/Zone")
except pytz.UnknownTimeZoneError:
    pass
else:
    raise AssertionError("absent time zone accepted")
print("pytz: bundled zone conversion and absent zone passed")
