"""Deterministic Gregorian date and time types over shellsim's virtual clock."""

import time


def _pad(value, width=2):
    text = str(value)
    return "0" * (width - len(text)) + text


def _days_from_civil(year, month, day):
    if month <= 2:
        year -= 1
    era = year // 400
    yoe = year - era * 400
    shifted = month - 3 if month > 2 else month + 9
    doy = (153 * shifted + 2) // 5 + day - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468


def _civil_from_days(days):
    value = days + 719468
    era = value // 146097
    doe = value - era * 146097
    yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
    year = yoe + era * 400
    doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
    mp = (5 * doy + 2) // 153
    day = doy - (153 * mp + 2) // 5 + 1
    month = mp + 3 if mp < 10 else mp - 9
    if month <= 2:
        year += 1
    return year, month, day


def _month_number(name):
    names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    return names.index(name) + 1


class _UTCZone:
    pass


UTC = _UTCZone()


class timezone:
    utc = UTC

    def __init__(self, offset):
        self.offset = offset


class timedelta:
    def __init__(self, days=0, seconds=0, microseconds=0, milliseconds=0, minutes=0, hours=0, weeks=0):
        self._microseconds = (
            ((weeks * 7 + days) * 86400 + hours * 3600 + minutes * 60 + seconds) * 1000000
            + milliseconds * 1000 + microseconds
        )

    @property
    def days(self):
        return self._microseconds // 86400000000

    @property
    def seconds(self):
        return (self._microseconds // 1000000) % 86400

    @property
    def microseconds(self):
        return self._microseconds % 1000000

    def total_seconds(self):
        return self._microseconds / 1000000

    def __add__(self, other):
        if isinstance(other, timedelta):
            return timedelta(microseconds=self._microseconds + other._microseconds)
        return other + self

    def __radd__(self, other):
        return other + self

    def __sub__(self, other):
        return timedelta(microseconds=self._microseconds - other._microseconds)

    def __neg__(self):
        return timedelta(microseconds=-self._microseconds)

    def __eq__(self, other):
        return isinstance(other, timedelta) and self._microseconds == other._microseconds

    def __lt__(self, other):
        return self._microseconds < other._microseconds


class datetime:
    def __init__(self, year, month, day, hour=0, minute=0, second=0, microsecond=0, tzinfo=None):
        self.year = year
        self.month = month
        self.day = day
        self.hour = hour
        self.minute = minute
        self.second = second
        self.microsecond = microsecond
        self.tzinfo = tzinfo

    def _epoch_microseconds(self):
        return (
            (_days_from_civil(self.year, self.month, self.day) * 86400
             + self.hour * 3600 + self.minute * 60 + self.second) * 1000000
            + self.microsecond
        )

    @classmethod
    def fromtimestamp(cls, value, tz=None):
        whole = int(value)
        days = whole // 86400
        rest = whole % 86400
        year, month, day = _civil_from_days(days)
        return cls(year, month, day, rest // 3600, (rest % 3600) // 60, rest % 60, 0, tz)

    @classmethod
    def now(cls, tz=None):
        return cls.fromtimestamp(time.time(), tz)

    @classmethod
    def utcnow(cls):
        return cls.fromtimestamp(time.time())

    @classmethod
    def fromisoformat(cls, value):
        text = value.rstrip("Z")
        pieces = text.split("T") if "T" in text else text.split(" ")
        date_parts = pieces[0].split("-")
        year = int(date_parts[0])
        month = int(date_parts[1])
        day = int(date_parts[2])
        if len(pieces) == 1:
            return cls(year, month, day)
        clock = pieces[1].split("+")[0]
        clock_parts = clock.split(":")
        second_parts = clock_parts[2].split(".") if len(clock_parts) > 2 else ["0"]
        microsecond = int((second_parts[1] + "000000")[:6]) if len(second_parts) > 1 else 0
        return cls(year, month, day, int(clock_parts[0]), int(clock_parts[1]), int(second_parts[0]), microsecond, UTC if value.endswith("Z") else None)

    @classmethod
    def strptime(cls, value, format):
        if format == "%Y-%m-%d":
            parts = value.split("-")
            return cls(int(parts[0]), int(parts[1]), int(parts[2]))
        if format == "%Y-%m-%d %H:%M:%S":
            parts = value.split(" ")
            date_parts = parts[0].split("-")
            clock = parts[1].split(":")
            return cls(int(date_parts[0]), int(date_parts[1]), int(date_parts[2]), int(clock[0]), int(clock[1]), int(clock[2]))
        if format == "%d/%b/%Y:%H:%M:%S +0000":
            parts = value.split("/")
            tail = parts[2].split(":")
            return cls(int(tail[0]), _month_number(parts[1]), int(parts[0]), int(tail[1]), int(tail[2]), int(tail[3].split(" ")[0]), 0, UTC)
        raise ValueError("unsupported datetime format")

    def strftime(self, format):
        result = format
        replacements = [
            ("%Y", _pad(self.year, 4)), ("%m", _pad(self.month)), ("%d", _pad(self.day)),
            ("%H", _pad(self.hour)), ("%M", _pad(self.minute)), ("%S", _pad(self.second)),
        ]
        for marker, value in replacements:
            result = result.replace(marker, value)
        return result

    def isoformat(self, sep="T"):
        result = self.strftime("%Y-%m-%d") + sep + self.strftime("%H:%M:%S")
        if self.microsecond != 0:
            result += "." + _pad(self.microsecond, 6)
        if self.tzinfo is UTC:
            result += "+00:00"
        return result

    def timestamp(self):
        return self._epoch_microseconds() / 1000000

    def astimezone(self, tz=None):
        return datetime(self.year, self.month, self.day, self.hour, self.minute, self.second, self.microsecond, tz)

    def __add__(self, other):
        if not isinstance(other, timedelta):
            raise TypeError("datetime addition requires timedelta")
        return datetime.fromtimestamp((self._epoch_microseconds() + other._microseconds) / 1000000, self.tzinfo)

    def __sub__(self, other):
        if isinstance(other, timedelta):
            return datetime.fromtimestamp((self._epoch_microseconds() - other._microseconds) / 1000000, self.tzinfo)
        if isinstance(other, datetime):
            return timedelta(microseconds=self._epoch_microseconds() - other._epoch_microseconds())
        raise TypeError("unsupported datetime subtraction")

    def __eq__(self, other):
        return isinstance(other, datetime) and self._epoch_microseconds() == other._epoch_microseconds()

    def __lt__(self, other):
        return self._epoch_microseconds() < other._epoch_microseconds()

    def __str__(self):
        return self.isoformat(" ")


class date(datetime):
    @classmethod
    def today(cls):
        value = datetime.now()
        return cls(value.year, value.month, value.day)

    def isoformat(self):
        return self.strftime("%Y-%m-%d")

    def __str__(self):
        return self.isoformat()
