"""Exercise local public behavior in the actual guest."""

from datetime import datetime

from freezegun import freeze_time

with freeze_time("2020-01-02 03:04:05"):
    assert datetime.now() == datetime(2020, 1, 2, 3, 4, 5)
print("freezegun: guest probe passed")
