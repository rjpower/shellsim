"""Gregorian dates, times, time deltas and fixed-offset time zones.

The implementation follows the documented semantics of CPython's ``datetime``: proleptic
Gregorian ordinals, microsecond resolution, naive and aware comparisons, and ``fold``. The
simulated local time zone is UTC, so ``now()`` and ``utcnow()`` agree and ``fromtimestamp``
never consults a host zone database. ``strftime`` and ``strptime`` are implemented here with the
C89 directives ``datetime`` documents.
"""

import time as _time

__all__ = ["date", "datetime", "time", "timedelta", "timezone", "tzinfo", "MINYEAR", "MAXYEAR", "UTC"]

MINYEAR = 1
MAXYEAR = 9999
_MAXORDINAL = 3652059

_DAYS_IN_MONTH = [-1, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
_DAYS_BEFORE_MONTH = [-1]
_days_before = 0
for _days in _DAYS_IN_MONTH[1:]:
    _DAYS_BEFORE_MONTH.append(_days_before)
    _days_before += _days
del _days_before, _days

_MONTHNAMES = [None, "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
_FULL_MONTHNAMES = [None, "January", "February", "March", "April", "May", "June", "July", "August",
                    "September", "October", "November", "December"]
_DAYNAMES = [None, "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
_FULL_DAYNAMES = [None, "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _is_leap(year):
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def _days_before_year(year):
    y = year - 1
    return y * 365 + y // 4 - y // 100 + y // 400


def _days_in_month(year, month):
    if month == 2 and _is_leap(year):
        return 29
    return _DAYS_IN_MONTH[month]


def _days_before_month(year, month):
    return _DAYS_BEFORE_MONTH[month] + (month > 2 and _is_leap(year))


def _ymd2ord(year, month, day):
    return _days_before_year(year) + _days_before_month(year, month) + day


_DI400Y = _days_before_year(401)
_DI100Y = _days_before_year(101)
_DI4Y = _days_before_year(5)


def _ord2ymd(n):
    n -= 1
    n400, n = divmod(n, _DI400Y)
    year = n400 * 400 + 1
    n100, n = divmod(n, _DI100Y)
    n4, n = divmod(n, _DI4Y)
    n1, n = divmod(n, 365)
    year += n100 * 100 + n4 * 4 + n1
    if n1 == 4 or n100 == 4:
        return year - 1, 12, 31
    leapyear = n1 == 3 and (n4 != 24 or n100 == 3)
    month = (n + 50) >> 5
    preceding = _DAYS_BEFORE_MONTH[month] + (month > 2 and leapyear)
    if preceding > n:
        month -= 1
        preceding -= _DAYS_IN_MONTH[month] + (month == 2 and leapyear)
    n -= preceding
    return year, month, n + 1


def _isoweek1monday(year):
    first_day = _ymd2ord(year, 1, 1)
    first_weekday = (first_day + 6) % 7
    week1monday = first_day - first_weekday
    if first_weekday > 3:
        week1monday += 7
    return week1monday


def _isoweek_to_gregorian(year, week, day):
    if not MINYEAR <= year <= MAXYEAR:
        raise ValueError("Year is out of range: %d" % year)
    if not 0 < week < 53:
        out_of_range = True
        if week == 53:
            first_weekday = _ymd2ord(year, 1, 1) % 7
            if first_weekday == 4 or (first_weekday == 3 and _is_leap(year)):
                out_of_range = False
        if out_of_range:
            raise ValueError("Invalid week: %d" % week)
    if not 0 < day < 8:
        raise ValueError("Invalid weekday: %d (range is [1, 7])" % day)
    day_offset = (week - 1) * 7 + (day - 1)
    return _ord2ymd(_isoweek1monday(year) + day_offset)


def _check_int(value, what):
    if isinstance(value, bool) or not isinstance(value, int):
        index = getattr(type(value), "__index__", None)
        if index is None or isinstance(value, bool):
            raise TypeError("an integer is required (got type %s)" % type(value).__name__)
        value = index(value)
    return value


def _check_date_fields(year, month, day):
    year = _check_int(year, "year")
    month = _check_int(month, "month")
    day = _check_int(day, "day")
    if not MINYEAR <= year <= MAXYEAR:
        raise ValueError("year %d is out of range" % year)
    if not 1 <= month <= 12:
        raise ValueError("month must be in 1..12")
    if not 1 <= day <= _days_in_month(year, month):
        raise ValueError("day is out of range for month")
    return year, month, day


def _check_time_fields(hour, minute, second, microsecond, fold):
    hour = _check_int(hour, "hour")
    minute = _check_int(minute, "minute")
    second = _check_int(second, "second")
    microsecond = _check_int(microsecond, "microsecond")
    if not 0 <= hour <= 23:
        raise ValueError("hour must be in 0..23")
    if not 0 <= minute <= 59:
        raise ValueError("minute must be in 0..59")
    if not 0 <= second <= 59:
        raise ValueError("second must be in 0..59")
    if not 0 <= microsecond <= 999999:
        raise ValueError("microsecond must be in 0..999999")
    if fold not in (0, 1):
        raise ValueError("fold must be either 0 or 1")
    return hour, minute, second, microsecond, fold


def _check_tzinfo_arg(tz):
    if tz is not None and not isinstance(tz, tzinfo):
        raise TypeError("tzinfo argument must be None or of a tzinfo subclass")


def _check_utc_offset(name, offset):
    if offset is None:
        return
    if not isinstance(offset, timedelta):
        raise TypeError("tzinfo.%s() must return None or timedelta, not '%s'" % (name, type(offset).__name__))
    if not -timedelta(1) < offset < timedelta(1):
        raise ValueError("%s()=%s, must be strictly between -timedelta(hours=24) and timedelta(hours=24)" % (name, offset))


def _cmp(x, y):
    return 0 if x == y else 1 if x > y else -1


def _divide_and_round(a, b):
    q, r = divmod(a, b)
    r *= 2
    greater_than_half = r > b if b > 0 else r < b
    if greater_than_half or r == b and q % 2 == 1:
        q += 1
    return q


def _format_offset(off, sep=":"):
    if off is None:
        return ""
    sign = "-" if off < timedelta(0) else "+"
    if off < timedelta(0):
        off = -off
    hh, mm = divmod(off, timedelta(hours=1))
    mm, ss = divmod(mm, timedelta(minutes=1))
    text = "%s%02d%s%02d" % (sign, hh, sep, mm)
    if ss or ss.microseconds:
        text += "%s%02d" % (sep, ss.seconds)
        if ss.microseconds:
            text += ".%06d" % ss.microseconds
    return text


def _format_time(hh, mm, ss, us, timespec="auto"):
    if timespec == "auto":
        timespec = "microseconds" if us else "seconds"
    if timespec == "hours":
        return "%02d" % hh
    if timespec == "minutes":
        return "%02d:%02d" % (hh, mm)
    if timespec == "seconds":
        return "%02d:%02d:%02d" % (hh, mm, ss)
    if timespec == "milliseconds":
        return "%02d:%02d:%02d.%03d" % (hh, mm, ss, us // 1000)
    if timespec == "microseconds":
        return "%02d:%02d:%02d.%06d" % (hh, mm, ss, us)
    raise ValueError("Unknown timespec value")


def _strftime(fmt, year, month, day, hour, minute, second, microsecond, weekday, yearday, offset, zone):
    """Render ``fmt`` with the C89 directives plus ``%f``, ``%z``, ``%Z``, ``%G``, ``%u`` and ``%V``."""
    if not isinstance(fmt, str):
        raise TypeError("strftime() argument 1 must be str, not %s" % type(fmt).__name__)
    out = []
    index = 0
    length = len(fmt)
    while index < length:
        char = fmt[index]
        index += 1
        if char != "%":
            out.append(char)
            continue
        if index >= length:
            out.append("%")
            break
        code = fmt[index]
        index += 1
        if code == "Y":
            out.append("%04d" % year)
        elif code == "m":
            out.append("%02d" % month)
        elif code == "d":
            out.append("%02d" % day)
        elif code == "H":
            out.append("%02d" % hour)
        elif code == "M":
            out.append("%02d" % minute)
        elif code == "S":
            out.append("%02d" % second)
        elif code == "f":
            out.append("%06d" % microsecond)
        elif code == "y":
            out.append("%02d" % (year % 100))
        elif code == "I":
            out.append("%02d" % (hour % 12 or 12))
        elif code == "p":
            out.append("AM" if hour < 12 else "PM")
        elif code == "j":
            out.append("%03d" % yearday)
        elif code == "a":
            out.append(_DAYNAMES[weekday + 1])
        elif code == "A":
            out.append(_FULL_DAYNAMES[weekday + 1])
        elif code == "b" or code == "h":
            out.append(_MONTHNAMES[month])
        elif code == "B":
            out.append(_FULL_MONTHNAMES[month])
        elif code == "w":
            out.append(str((weekday + 1) % 7))
        elif code == "u":
            out.append(str(weekday + 1))
        elif code == "e":
            out.append("%2d" % day)
        elif code == "C":
            out.append("%02d" % (year // 100))
        elif code == "n":
            out.append("\n")
        elif code == "t":
            out.append("\t")
        elif code == "%":
            out.append("%")
        elif code == "z":
            out.append(_format_offset(offset, ""))
        elif code == "Z":
            out.append(zone or "")
        elif code == "U":
            out.append("%02d" % ((yearday + 6 - (weekday + 1) % 7) // 7))
        elif code == "W":
            out.append("%02d" % ((yearday + 6 - weekday) // 7))
        elif code in "GV":
            iso_year, iso_week, _ = date(year, month, day).isocalendar()
            out.append("%04d" % iso_year if code == "G" else "%02d" % iso_week)
        elif code == "c":
            out.append("%s %s %2d %02d:%02d:%02d %04d" % (
                _DAYNAMES[weekday + 1], _MONTHNAMES[month], day, hour, minute, second, year))
        elif code == "x":
            out.append("%02d/%02d/%02d" % (month, day, year % 100))
        elif code == "X":
            out.append("%02d:%02d:%02d" % (hour, minute, second))
        elif code == "D":
            out.append("%02d/%02d/%02d" % (month, day, year % 100))
        elif code == "F":
            out.append("%04d-%02d-%02d" % (year, month, day))
        elif code == "T":
            out.append("%02d:%02d:%02d" % (hour, minute, second))
        elif code == "R":
            out.append("%02d:%02d" % (hour, minute))
        elif code == "r":
            out.append("%02d:%02d:%02d %s" % (hour % 12 or 12, minute, second, "AM" if hour < 12 else "PM"))
        else:
            raise ValueError("Invalid format string")
    return "".join(out)


# ---- strptime ----

_STRPTIME_PATTERNS = {
    "Y": r"(?P<Y>\d\d\d\d)", "m": r"(?P<m>1[0-2]|0[1-9]|[1-9])", "d": r"(?P<d>3[01]|[12]\d|0[1-9]|[1-9]| [1-9])",
    "H": r"(?P<H>2[0-3]|[0-1]\d|\d)", "I": r"(?P<I>1[0-2]|0[1-9]|[1-9])", "M": r"(?P<M>[0-5]\d|\d)",
    "S": r"(?P<S>6[0-1]|[0-5]\d|\d)", "f": r"(?P<f>[0-9]{1,6})", "y": r"(?P<y>\d\d)",
    "j": r"(?P<j>36[0-6]|3[0-5]\d|[12]\d\d|0[1-9]\d|00[1-9]|[1-9]\d|0[1-9]|[1-9])",
    "p": r"(?P<p>AM|PM|am|pm)", "z": r"(?P<z>[+-]\d\d:?[0-5]\d(:?[0-5]\d(\.\d{1,6})?)?|Z)",
    "Z": r"(?P<Z>UTC|GMT|[A-Z]{3,5})", "w": r"(?P<w>[0-6])", "u": r"(?P<u>[1-7])",
    "U": r"(?P<U>5[0-3]|[0-4]\d|\d)", "W": r"(?P<W>5[0-3]|[0-4]\d|\d)",
    "G": r"(?P<G>\d\d\d\d)", "V": r"(?P<V>5[0-3]|0[1-9]|[1-4]\d|\d)",
    "%": "%",
}
_STRPTIME_NAMES = {
    "a": ("a", _DAYNAMES[1:]), "A": ("A", _FULL_DAYNAMES[1:]),
    "b": ("b", _MONTHNAMES[1:]), "B": ("B", _FULL_MONTHNAMES[1:]),
}
_STRPTIME_EXPANSIONS = {"c": "%a %b %d %H:%M:%S %Y", "x": "%m/%d/%y", "X": "%H:%M:%S",
                        "D": "%m/%d/%y", "F": "%Y-%m-%d", "T": "%H:%M:%S", "R": "%H:%M", "h": "%b"}


def _strptime_regex(fmt):
    import re

    pieces = []
    index = 0
    while index < len(fmt):
        char = fmt[index]
        index += 1
        if char != "%":
            if char.isspace():
                pieces.append(r"\s+")
            else:
                pieces.append(re.escape(char))
            continue
        if index >= len(fmt):
            raise ValueError("stray %% in format '%s'" % fmt)
        code = fmt[index]
        index += 1
        if code in _STRPTIME_EXPANSIONS:
            pieces.append(_strptime_regex(_STRPTIME_EXPANSIONS[code]))
        elif code in _STRPTIME_PATTERNS:
            pieces.append(_STRPTIME_PATTERNS[code])
        elif code in _STRPTIME_NAMES:
            group, names = _STRPTIME_NAMES[code]
            alternatives = sorted((name.lower() for name in names), key=len, reverse=True)
            pieces.append("(?P<%s>%s)" % (group, "|".join(re.escape(name) for name in alternatives)))
        else:
            raise ValueError("'%s' is a bad directive in format '%s'" % (code, fmt))
    return "".join(pieces)


def _strptime(text, fmt):
    """Parse ``text`` with ``fmt`` into ``(year, month, day, hour, minute, second, microsecond,
    tzinfo)``; missing date fields default to 1900-01-01 as in CPython."""
    import re

    if not isinstance(text, str):
        raise TypeError("strptime() argument 0 must be str, not %s" % type(text).__name__)
    pattern = re.compile(_strptime_regex(fmt), re.IGNORECASE)
    match = pattern.match(text)
    if match is None:
        raise ValueError("time data %r does not match format %r" % (text, fmt))
    if match.end() != len(text):
        raise ValueError("unconverted data remains: %s" % text[match.end():])
    found = match.groupdict()
    year = None
    month = day = 1
    hour = minute = second = microsecond = 0
    weekday = julian = None
    week_of_year = None
    week_starts_monday = None
    iso_year = iso_week = None
    tz = None
    for group, value in found.items():
        if value is None:
            continue
        if group == "Y":
            year = int(value)
        elif group == "y":
            year = int(value)
            year += 2000 if year <= 68 else 1900
        elif group == "G":
            iso_year = int(value)
        elif group == "m":
            month = int(value)
        elif group == "B":
            month = [name.lower() for name in _FULL_MONTHNAMES[1:]].index(value.lower()) + 1
        elif group == "b":
            month = [name.lower() for name in _MONTHNAMES[1:]].index(value.lower()) + 1
        elif group == "d":
            day = int(value)
        elif group == "H":
            hour = int(value)
        elif group == "I":
            hour = int(value)
            ampm = (found.get("p") or "").lower()
            if ampm in ("", "am"):
                if hour == 12:
                    hour = 0
            elif ampm == "pm":
                if hour != 12:
                    hour += 12
        elif group == "M":
            minute = int(value)
        elif group == "S":
            second = int(value)
        elif group == "f":
            microsecond = int(value + "0" * (6 - len(value)))
        elif group == "A":
            weekday = [name.lower() for name in _FULL_DAYNAMES[1:]].index(value.lower())
        elif group == "a":
            weekday = [name.lower() for name in _DAYNAMES[1:]].index(value.lower())
        elif group == "w":
            weekday = (int(value) - 1) % 7
        elif group == "u":
            weekday = int(value) - 1
        elif group == "j":
            julian = int(value)
        elif group in ("U", "W"):
            week_of_year = int(value)
            week_starts_monday = group == "W"
        elif group == "V":
            iso_week = int(value)
        elif group == "z":
            tz = _parse_offset(value)
    if iso_year is not None or iso_week is not None:
        if iso_year is None or iso_week is None or weekday is None:
            raise ValueError("ISO year directive '%G' must be used with the ISO week directive '%V' and a weekday directive ('%A', '%a', '%w', or '%u').")
        year, month, day = _isoweek_to_gregorian(iso_year, iso_week, weekday + 1)
    elif year is None:
        year = 1900
    if julian is not None:
        year, month, day = _ord2ymd(_ymd2ord(year, 1, 1) + julian - 1)
    elif week_of_year is not None and weekday is not None:
        first = _ymd2ord(year, 1, 1)
        first_weekday = (first + 6) % 7
        if week_starts_monday:
            week_start = first - first_weekday + (7 if first_weekday != 0 else 0)
            offset = weekday
        else:
            sunday_based_first = (first_weekday + 1) % 7
            week_start = first - sunday_based_first + (7 if sunday_based_first != 0 else 0)
            offset = (weekday + 1) % 7
        if week_of_year == 0:
            week_start -= 7
        ordinal = week_start + (week_of_year - 1) * 7 + offset
        year, month, day = _ord2ymd(ordinal)
    if tz is not None:
        tz_name = found.get("Z")
        if tz_name and tz != timezone.utc:
            tz = timezone(tz.utcoffset(None), tz_name)
    return year, month, day, hour, minute, second, microsecond, tz


def _parse_offset(text):
    if text == "Z":
        return timezone.utc
    sign = -1 if text[0] == "-" else 1
    body = text[1:].replace(":", "")
    hours = int(body[0:2])
    minutes = int(body[2:4])
    seconds = int(body[4:6]) if len(body) >= 6 else 0
    micro = 0
    if "." in body:
        fraction = body.split(".", 1)[1]
        micro = int(fraction + "0" * (6 - len(fraction)))
    offset = timedelta(hours=hours, minutes=minutes, seconds=seconds, microseconds=micro)
    return timezone(sign * offset)


# ---- timedelta ----


class timedelta:
    """A duration: the difference between two dates, times or datetimes."""

    __slots__ = "_days", "_seconds", "_microseconds", "_hashcode"

    def __new__(cls, days=0, seconds=0, microseconds=0, milliseconds=0, minutes=0, hours=0, weeks=0):
        for value in (days, seconds, minutes, hours, weeks, milliseconds, microseconds):
            if not isinstance(value, (int, float)):
                raise TypeError("unsupported type for timedelta %s component: %s" % ("days", type(value).__name__))
        total_us = 0.0
        exact_us = 0
        exact = True
        days += weeks * 7
        seconds += minutes * 60 + hours * 3600
        microseconds += milliseconds * 1000
        for amount, scale in ((days, 86400 * 1000000), (seconds, 1000000), (microseconds, 1)):
            if isinstance(amount, float):
                exact = False
                total_us += amount * scale
            else:
                exact_us += amount * scale
        if exact:
            total = exact_us
        else:
            total = exact_us + _round_half_even(total_us)
        d, rem = divmod(total, 86400 * 1000000)
        s, us = divmod(rem, 1000000)
        if abs(d) > 999999999:
            raise OverflowError("days=%d; must have magnitude <= 999999999" % d)
        self = object.__new__(cls)
        self._days = d
        self._seconds = s
        self._microseconds = us
        self._hashcode = -1
        return self

    def __repr__(self):
        args = []
        if self._days:
            args.append("days=%d" % self._days)
        if self._seconds:
            args.append("seconds=%d" % self._seconds)
        if self._microseconds:
            args.append("microseconds=%d" % self._microseconds)
        if not args:
            args.append("0")
        return "datetime.timedelta(%s)" % ", ".join(args)

    def __str__(self):
        mm, ss = divmod(self._seconds, 60)
        hh, mm = divmod(mm, 60)
        text = "%d:%02d:%02d" % (hh, mm, ss)
        if self._days:
            text = "%d day%s, %s" % (self._days, "" if abs(self._days) == 1 else "s", text)
        if self._microseconds:
            text += ".%06d" % self._microseconds
        return text

    def total_seconds(self):
        return ((self.days * 86400 + self.seconds) * 10**6 + self.microseconds) / 10**6

    @property
    def days(self):
        return self._days

    @property
    def seconds(self):
        return self._seconds

    @property
    def microseconds(self):
        return self._microseconds

    def _to_microseconds(self):
        return (self._days * 86400 + self._seconds) * 1000000 + self._microseconds

    def __add__(self, other):
        if isinstance(other, timedelta):
            return timedelta(self._days + other._days, self._seconds + other._seconds,
                             microseconds=self._microseconds + other._microseconds)
        return NotImplemented

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, timedelta):
            return timedelta(self._days - other._days, self._seconds - other._seconds,
                             microseconds=self._microseconds - other._microseconds)
        return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, timedelta):
            return -self + other
        return NotImplemented

    def __neg__(self):
        return timedelta(-self._days, -self._seconds, microseconds=-self._microseconds)

    def __pos__(self):
        return self

    def __abs__(self):
        if self._days < 0:
            return -self
        return self

    def __mul__(self, other):
        if isinstance(other, int):
            return timedelta(microseconds=self._to_microseconds() * other)
        if isinstance(other, float):
            import fractions

            a, b = fractions.Fraction(other).as_integer_ratio()
            return timedelta(microseconds=_divide_and_round(self._to_microseconds() * a, b))
        return NotImplemented

    __rmul__ = __mul__

    def __floordiv__(self, other):
        if not isinstance(other, (int, timedelta)):
            return NotImplemented
        usec = self._to_microseconds()
        if isinstance(other, timedelta):
            return usec // other._to_microseconds()
        return timedelta(microseconds=usec // other)

    def __truediv__(self, other):
        if not isinstance(other, (int, float, timedelta)):
            return NotImplemented
        usec = self._to_microseconds()
        if isinstance(other, timedelta):
            return usec / other._to_microseconds()
        if isinstance(other, int):
            return timedelta(microseconds=_divide_and_round(usec, other))
        import fractions

        a, b = fractions.Fraction(other).as_integer_ratio()
        return timedelta(microseconds=_divide_and_round(b * usec, a))

    def __mod__(self, other):
        if isinstance(other, timedelta):
            return timedelta(microseconds=self._to_microseconds() % other._to_microseconds())
        return NotImplemented

    def __divmod__(self, other):
        if isinstance(other, timedelta):
            q, r = divmod(self._to_microseconds(), other._to_microseconds())
            return q, timedelta(microseconds=r)
        return NotImplemented

    def __eq__(self, other):
        if isinstance(other, timedelta):
            return self._cmp(other) == 0
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, timedelta):
            return self._cmp(other) <= 0
        return NotImplemented

    def __lt__(self, other):
        if isinstance(other, timedelta):
            return self._cmp(other) < 0
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, timedelta):
            return self._cmp(other) >= 0
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, timedelta):
            return self._cmp(other) > 0
        return NotImplemented

    def _cmp(self, other):
        return _cmp(self._getstate(), other._getstate())

    def __hash__(self):
        if self._hashcode == -1:
            self._hashcode = hash(self._getstate())
        return self._hashcode

    def __bool__(self):
        return self._days != 0 or self._seconds != 0 or self._microseconds != 0

    def _getstate(self):
        return (self._days, self._seconds, self._microseconds)

    def __reduce__(self):
        return (self.__class__, self._getstate())


def _round_half_even(value):
    rounded = round(value)
    return int(rounded)


timedelta.min = timedelta(-999999999)
timedelta.max = timedelta(days=999999999, hours=23, minutes=59, seconds=59, microseconds=999999)
timedelta.resolution = timedelta(microseconds=1)


# ---- tzinfo and timezone ----


class tzinfo:
    """Abstract base for time zone information; subclasses define the three query methods."""

    __slots__ = ()

    def tzname(self, dt):
        raise NotImplementedError("tzinfo subclass must override tzname()")

    def utcoffset(self, dt):
        raise NotImplementedError("tzinfo subclass must override utcoffset()")

    def dst(self, dt):
        raise NotImplementedError("tzinfo subclass must override dst()")

    def fromutc(self, dt):
        if not isinstance(dt, datetime):
            raise TypeError("fromutc() requires a datetime argument")
        if dt.tzinfo is not self:
            raise ValueError("dt.tzinfo is not self")
        dtoff = dt.utcoffset()
        if dtoff is None:
            raise ValueError("fromutc() requires a non-None utcoffset() result")
        dtdst = dt.dst()
        if dtdst is None:
            raise ValueError("fromutc() requires a non-None dst() result")
        delta = dtoff - dtdst
        if delta:
            dt += delta
            dtdst = dt.dst()
            if dtdst is None:
                raise ValueError("fromutc(): dt.dst gave inconsistent results; cannot convert")
        return dt + dtdst

    def __reduce__(self):
        getinitargs = getattr(self, "__getinitargs__", None)
        args = getinitargs() if getinitargs else ()
        state = getattr(self, "__dict__", None) or None
        if state is None:
            return (self.__class__, args)
        return (self.__class__, args, state)


class timezone(tzinfo):
    """A fixed offset from UTC with an optional name."""

    __slots__ = "_offset", "_name"

    _Omitted = object()

    def __new__(cls, offset, name=_Omitted):
        if not isinstance(offset, timedelta):
            raise TypeError("offset must be a timedelta")
        if name is cls._Omitted:
            if not offset:
                return cls.utc
            name = None
        elif not isinstance(name, str):
            raise TypeError("name must be a string")
        if not cls._minoffset <= offset <= cls._maxoffset:
            raise ValueError("offset must be a timedelta strictly between -timedelta(hours=24) and timedelta(hours=24).")
        return cls._create(offset, name)

    @classmethod
    def _create(cls, offset, name=None):
        self = tzinfo.__new__(cls)
        self._offset = offset
        self._name = name
        return self

    def __getinitargs__(self):
        if self._name is None:
            return (self._offset,)
        return (self._offset, self._name)

    def __eq__(self, other):
        if isinstance(other, timezone):
            return self._offset == other._offset
        return NotImplemented

    def __hash__(self):
        return hash(self._offset)

    def __repr__(self):
        if self is self.utc:
            return "datetime.timezone.utc"
        if self._name is None:
            return "datetime.timezone(%r)" % self._offset
        return "datetime.timezone(%r, %r)" % (self._offset, self._name)

    def __str__(self):
        return self.tzname(None)

    def utcoffset(self, dt):
        if isinstance(dt, datetime) or dt is None:
            return self._offset
        raise TypeError("utcoffset() argument must be a datetime instance or None")

    def tzname(self, dt):
        if isinstance(dt, datetime) or dt is None:
            if self._name is None:
                return self._name_from_offset(self._offset)
            return self._name
        raise TypeError("tzname() argument must be a datetime instance or None")

    def dst(self, dt):
        if isinstance(dt, datetime) or dt is None:
            return None
        raise TypeError("dst() argument must be a datetime instance or None")

    def fromutc(self, dt):
        if isinstance(dt, datetime):
            if dt.tzinfo is not self:
                raise ValueError("fromutc: dt.tzinfo is not self")
            return dt + self._offset
        raise TypeError("fromutc() argument must be a datetime instance or None")

    _maxoffset = timedelta(hours=24, microseconds=-1)
    _minoffset = -_maxoffset

    @staticmethod
    def _name_from_offset(delta):
        if not delta:
            return "UTC"
        if delta < timedelta(0):
            sign = "-"
            delta = -delta
        else:
            sign = "+"
        hours, rest = divmod(delta, timedelta(hours=1))
        minutes, rest = divmod(rest, timedelta(minutes=1))
        seconds = rest.seconds
        microseconds = rest.microseconds
        if microseconds:
            return "UTC%s%02d:%02d:%02d.%06d" % (sign, hours, minutes, seconds, microseconds)
        if seconds:
            return "UTC%s%02d:%02d:%02d" % (sign, hours, minutes, seconds)
        return "UTC%s%02d:%02d" % (sign, hours, minutes)


UTC = timezone.utc = timezone._create(timedelta(0))
timezone.min = timezone._create(-timedelta(hours=23, minutes=59))
timezone.max = timezone._create(timedelta(hours=23, minutes=59))


# ---- date ----


class _IsoCalendarDate(tuple):
    _fields = ("year", "week", "weekday")

    def __new__(cls, year, week, weekday):
        return tuple.__new__(cls, (year, week, weekday))

    year = property(lambda self: self[0])
    week = property(lambda self: self[1])
    weekday = property(lambda self: self[2])

    def __reduce__(self):
        return (tuple, (tuple(self),))

    def __repr__(self):
        return "datetime.IsoCalendarDate(year=%d, week=%d, weekday=%d)" % tuple(self)


def _parse_isoformat_date(dtstr):
    length = len(dtstr)
    if length < 7:
        raise ValueError("Invalid isoformat string: %r" % dtstr)
    year = int(dtstr[0:4])
    has_sep = dtstr[4] == "-"
    pos = 4 + has_sep
    if dtstr[pos:pos + 1] == "W":
        pos += 1
        weekno = int(dtstr[pos:pos + 2])
        pos += 2
        dayno = 1
        if length > pos:
            if (dtstr[pos:pos + 1] == "-") != has_sep:
                raise ValueError("Inconsistent use of dash separator")
            pos += has_sep
            dayno = int(dtstr[pos:pos + 1])
            pos += 1
        if pos != length:
            raise ValueError("Invalid isoformat string: %r" % dtstr)
        return list(_isoweek_to_gregorian(year, weekno, dayno))
    month = int(dtstr[pos:pos + 2])
    pos += 2
    if has_sep:
        if dtstr[pos:pos + 1] != "-":
            raise ValueError("Invalid date separator: %s" % dtstr[pos:pos + 1])
        pos += 1
    day = int(dtstr[pos:pos + 2])
    pos += 2
    if pos != length:
        raise ValueError("Invalid isoformat string: %r" % dtstr)
    return [year, month, day]


class date:
    """A calendar date in the proleptic Gregorian calendar."""

    __slots__ = "_year", "_month", "_day", "_hashcode"

    def __new__(cls, year, month=None, day=None):
        if month is None and isinstance(year, (bytes, str)) and len(year) == 4 and 1 <= ord(year[2:3]) <= 12:
            if isinstance(year, str):
                year = year.encode("latin1")
            self = object.__new__(cls)
            self._setstate(year)
            self._hashcode = -1
            return self
        year, month, day = _check_date_fields(year, month, day)
        self = object.__new__(cls)
        self._year = year
        self._month = month
        self._day = day
        self._hashcode = -1
        return self

    @classmethod
    def fromtimestamp(cls, t):
        y, m, d, hh, mm, ss, weekday, jday, dst = _time.gmtime(t)
        return cls(y, m, d)

    @classmethod
    def today(cls):
        return cls.fromtimestamp(_time.time())

    @classmethod
    def fromordinal(cls, n):
        if not 1 <= n <= _MAXORDINAL:
            raise ValueError("ordinal must be in 1..%d" % _MAXORDINAL)
        y, m, d = _ord2ymd(n)
        return cls(y, m, d)

    @classmethod
    def fromisoformat(cls, date_string):
        if not isinstance(date_string, str):
            raise TypeError("fromisoformat: argument must be str")
        if len(date_string) not in (7, 8, 10):
            raise ValueError("Invalid isoformat string: %r" % date_string)
        try:
            return cls(*_parse_isoformat_date(date_string))
        except Exception:
            raise ValueError("Invalid isoformat string: %r" % date_string) from None

    @classmethod
    def fromisocalendar(cls, year, week, day):
        return cls(*_isoweek_to_gregorian(year, week, day))

    @classmethod
    def strptime(cls, date_string, format):
        year, month, day, hour, minute, second, microsecond, tz = _strptime(date_string, format)
        return cls(year, month, day)

    def __repr__(self):
        return "%s.%s(%d, %d, %d)" % (self.__class__.__module__, self.__class__.__qualname__,
                                      self._year, self._month, self._day)

    def ctime(self):
        weekday = self.toordinal() % 7 or 7
        return "%s %s %2d 00:00:00 %04d" % (_DAYNAMES[weekday], _MONTHNAMES[self._month], self._day, self._year)

    def strftime(self, format):
        return _strftime(format, self._year, self._month, self._day, 0, 0, 0, 0,
                         self.weekday(), self.toordinal() - _days_before_year(self._year), None, None)

    def __format__(self, fmt):
        if not isinstance(fmt, str):
            raise TypeError("must be str, not %s" % type(fmt).__name__)
        if len(fmt) != 0:
            return self.strftime(fmt)
        return str(self)

    def isoformat(self):
        return "%04d-%02d-%02d" % (self._year, self._month, self._day)

    __str__ = isoformat

    @property
    def year(self):
        return self._year

    @property
    def month(self):
        return self._month

    @property
    def day(self):
        return self._day

    def timetuple(self):
        return _build_struct_time(self._year, self._month, self._day, 0, 0, 0, -1)

    def toordinal(self):
        return _ymd2ord(self._year, self._month, self._day)

    def replace(self, year=None, month=None, day=None):
        if year is None:
            year = self._year
        if month is None:
            month = self._month
        if day is None:
            day = self._day
        return type(self)(year, month, day)

    __replace__ = replace

    def __hash__(self):
        if self._hashcode == -1:
            self._hashcode = hash(self._getstate())
        return self._hashcode

    def __eq__(self, other):
        if isinstance(other, date) and not isinstance(other, datetime):
            return self._cmp(other) == 0
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, date) and not isinstance(other, datetime):
            return self._cmp(other) <= 0
        return NotImplemented

    def __lt__(self, other):
        if isinstance(other, date) and not isinstance(other, datetime):
            return self._cmp(other) < 0
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, date) and not isinstance(other, datetime):
            return self._cmp(other) >= 0
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, date) and not isinstance(other, datetime):
            return self._cmp(other) > 0
        return NotImplemented

    def _cmp(self, other):
        return _cmp((self._year, self._month, self._day), (other._year, other._month, other._day))

    def __add__(self, other):
        if isinstance(other, timedelta):
            o = self.toordinal() + other.days
            if 0 < o <= _MAXORDINAL:
                return type(self).fromordinal(o)
            raise OverflowError("result out of range")
        return NotImplemented

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, timedelta):
            return self + timedelta(-other.days)
        if isinstance(other, date) and not isinstance(other, datetime):
            return timedelta(self.toordinal() - other.toordinal())
        return NotImplemented

    def weekday(self):
        return (self.toordinal() + 6) % 7

    def isoweekday(self):
        return self.toordinal() % 7 or 7

    def isocalendar(self):
        year = self._year
        week1monday = _isoweek1monday(year)
        today = _ymd2ord(self._year, self._month, self._day)
        week, day = divmod(today - week1monday, 7)
        if week < 0:
            year -= 1
            week1monday = _isoweek1monday(year)
            week, day = divmod(today - week1monday, 7)
        elif week >= 52:
            if today >= _isoweek1monday(year + 1):
                year += 1
                week = 0
        return _IsoCalendarDate(year, week + 1, day + 1)

    def _getstate(self):
        yhi, ylo = divmod(self._year, 256)
        return (bytes([yhi, ylo, self._month, self._day]),)

    def _setstate(self, string):
        yhi, ylo, self._month, self._day = string
        self._year = yhi * 256 + ylo

    def __reduce__(self):
        return (self.__class__, self._getstate())


date.min = date(1, 1, 1)
date.max = date(9999, 12, 31)
date.resolution = timedelta(days=1)


def _build_struct_time(y, m, d, hh, mm, ss, dstflag):
    wday = (_ymd2ord(y, m, d) + 6) % 7
    dnum = _days_before_month(y, m) + d
    return _time.struct_time((y, m, d, hh, mm, ss, wday, dnum, dstflag))


# ---- time ----


def _parse_hh_mm_ss_ff(tstr):
    len_str = len(tstr)
    time_comps = [0, 0, 0, 0]
    pos = 0
    for comp in range(3):
        if (len_str - pos) < 2:
            raise ValueError("Incomplete time component")
        time_comps[comp] = int(tstr[pos:pos + 2])
        pos += 2
        next_char = tstr[pos:pos + 1]
        if comp == 0:
            has_sep = next_char == ":"
        if not next_char or comp >= 2:
            break
        if has_sep and next_char != ":":
            raise ValueError("Invalid time separator: %c" % next_char)
        pos += has_sep
    if pos < len_str:
        if tstr[pos] not in ".,":
            raise ValueError("Invalid microsecond component")
        pos += 1
        len_remainder = len_str - pos
        if len_remainder >= 6:
            to_parse = 6
        else:
            to_parse = len_remainder
        time_comps[3] = int(tstr[pos:(pos + to_parse)])
        if to_parse < 6:
            time_comps[3] *= 10 ** (6 - to_parse)
        if len_remainder > to_parse and not all(c.isdigit() for c in tstr[pos + to_parse:]):
            raise ValueError("Non-digit values in unparsed fraction")
    return time_comps


def _parse_isoformat_time(tstr):
    len_str = len(tstr)
    if len_str < 2:
        raise ValueError("Isoformat time too short")
    tz_pos = tstr.find("-") + 1 or tstr.find("+") + 1 or (tstr.find("Z") + 1 if tstr.endswith("Z") else 0)
    timestr = tstr[:tz_pos - 1] if tz_pos > 0 else tstr
    time_comps = _parse_hh_mm_ss_ff(timestr)
    hour, minute, second, microsecond = time_comps
    became_next_day = False
    error_from_components = False
    if hour == 24:
        if all(time_comp == 0 for time_comp in time_comps[1:]):
            hour = 0
            time_comps[0] = hour
            became_next_day = True
        else:
            error_from_components = True
    tzi = None
    if tz_pos == len_str and tstr[-1] == "Z":
        tzi = timezone.utc
    elif tz_pos > 0:
        tzstr = tstr[tz_pos:]
        if len(tzstr) in (0, 1, 3):
            raise ValueError("Malformed time zone string")
        tz_comps = _parse_hh_mm_ss_ff(tzstr)
        if all(x == 0 for x in tz_comps):
            tzi = timezone.utc
        else:
            tzsign = -1 if tstr[tz_pos - 1] == "-" else 1
            td = timedelta(hours=tz_comps[0], minutes=tz_comps[1], seconds=tz_comps[2], microseconds=tz_comps[3])
            tzi = timezone(tzsign * td)
    time_comps.append(tzi)
    return time_comps, became_next_day, error_from_components


class time:
    """A time of day, independent of any particular date, with optional time zone."""

    __slots__ = "_hour", "_minute", "_second", "_microsecond", "_tzinfo", "_hashcode", "_fold"

    def __new__(cls, hour=0, minute=0, second=0, microsecond=0, tzinfo=None, *, fold=0):
        if isinstance(hour, (bytes, str)) and len(hour) == 6 and ord(hour[0:1]) & 0x7F < 24:
            if isinstance(hour, str):
                hour = hour.encode("latin1")
            self = object.__new__(cls)
            self.__setstate(hour, minute or None)
            self._hashcode = -1
            return self
        hour, minute, second, microsecond, fold = _check_time_fields(hour, minute, second, microsecond, fold)
        _check_tzinfo_arg(tzinfo)
        self = object.__new__(cls)
        self._hour = hour
        self._minute = minute
        self._second = second
        self._microsecond = microsecond
        self._tzinfo = tzinfo
        self._hashcode = -1
        self._fold = fold
        return self

    @property
    def hour(self):
        return self._hour

    @property
    def minute(self):
        return self._minute

    @property
    def second(self):
        return self._second

    @property
    def microsecond(self):
        return self._microsecond

    @property
    def tzinfo(self):
        return self._tzinfo

    @property
    def fold(self):
        return self._fold

    def __eq__(self, other):
        if isinstance(other, time):
            return self._cmp(other, allow_mixed=True) == 0
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, time):
            return self._cmp(other) <= 0
        return NotImplemented

    def __lt__(self, other):
        if isinstance(other, time):
            return self._cmp(other) < 0
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, time):
            return self._cmp(other) >= 0
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, time):
            return self._cmp(other) > 0
        return NotImplemented

    def _cmp(self, other, allow_mixed=False):
        mytz = self._tzinfo
        ottz = other._tzinfo
        myoff = otoff = None
        if mytz is ottz:
            base_compare = True
        else:
            myoff = self.utcoffset()
            otoff = other.utcoffset()
            base_compare = myoff == otoff
        if base_compare:
            return _cmp((self._hour, self._minute, self._second, self._microsecond),
                        (other._hour, other._minute, other._second, other._microsecond))
        if myoff is None or otoff is None:
            if allow_mixed:
                return 2
            raise TypeError("can't compare offset-naive and offset-aware times")
        myhhmm = self._hour * 60 + self._minute - myoff // timedelta(minutes=1)
        othhmm = other._hour * 60 + other._minute - otoff // timedelta(minutes=1)
        return _cmp((myhhmm, self._second, self._microsecond), (othhmm, other._second, other._microsecond))

    def __hash__(self):
        if self._hashcode == -1:
            if self.fold:
                t = self.replace(fold=0)
            else:
                t = self
            tzoff = t.utcoffset()
            if not tzoff:
                self._hashcode = hash(t._getstate()[0])
            else:
                h, m = divmod(timedelta(hours=self.hour, minutes=self.minute) - tzoff, timedelta(hours=1))
                if 0 <= h < 24:
                    self._hashcode = hash(time(h, m, self.second, self.microsecond))
                else:
                    self._hashcode = hash((h, m, self.second, self.microsecond))
        return self._hashcode

    def _tzstr(self):
        return _format_offset(self.utcoffset())

    def __repr__(self):
        if self._microsecond != 0:
            s = ", %d, %d" % (self._second, self._microsecond)
        elif self._second != 0:
            s = ", %d" % self._second
        else:
            s = ""
        s = "%s.%s(%d, %d%s)" % (self.__class__.__module__, self.__class__.__qualname__, self._hour, self._minute, s)
        if self._tzinfo is not None:
            s = s[:-1] + ", tzinfo=%r" % self._tzinfo + ")"
        if self._fold:
            s = s[:-1] + ", fold=1)"
        return s

    def isoformat(self, timespec="auto"):
        s = _format_time(self._hour, self._minute, self._second, self._microsecond, timespec)
        tz = self._tzstr()
        if tz:
            s += tz
        return s

    __str__ = isoformat

    @classmethod
    def fromisoformat(cls, time_string):
        if not isinstance(time_string, str):
            raise TypeError("fromisoformat: argument must be str")
        time_string = time_string.removeprefix("T")
        try:
            components, _, error_from_components = _parse_isoformat_time(time_string)
        except ValueError:
            raise ValueError("Invalid isoformat string: %r" % time_string) from None
        if error_from_components:
            raise ValueError("minute, second, and microsecond must be 0 when hour is 24")
        return cls(*components)

    @classmethod
    def strptime(cls, date_string, format):
        year, month, day, hour, minute, second, microsecond, tz = _strptime(date_string, format)
        return cls(hour, minute, second, microsecond, tz)

    def strftime(self, format):
        return _strftime(format, 1900, 1, 1, self._hour, self._minute, self._second, self._microsecond,
                         0, 1, self.utcoffset(), self.tzname())

    def __format__(self, fmt):
        if not isinstance(fmt, str):
            raise TypeError("must be str, not %s" % type(fmt).__name__)
        if len(fmt) != 0:
            return self.strftime(fmt)
        return str(self)

    def utcoffset(self):
        if self._tzinfo is None:
            return None
        offset = self._tzinfo.utcoffset(None)
        _check_utc_offset("utcoffset", offset)
        return offset

    def tzname(self):
        if self._tzinfo is None:
            return None
        name = self._tzinfo.tzname(None)
        if name is not None and not isinstance(name, str):
            raise TypeError("tzinfo.tzname() must return None or string, not '%s'" % type(name).__name__)
        return name

    def dst(self):
        if self._tzinfo is None:
            return None
        offset = self._tzinfo.dst(None)
        _check_utc_offset("dst", offset)
        return offset

    def replace(self, hour=None, minute=None, second=None, microsecond=None, tzinfo=True, *, fold=None):
        if hour is None:
            hour = self.hour
        if minute is None:
            minute = self.minute
        if second is None:
            second = self.second
        if microsecond is None:
            microsecond = self.microsecond
        if tzinfo is True:
            tzinfo = self.tzinfo
        if fold is None:
            fold = self._fold
        return type(self)(hour, minute, second, microsecond, tzinfo, fold=fold)

    __replace__ = replace

    def _getstate(self, protocol=3):
        us2, us3 = divmod(self._microsecond, 256)
        us1, us2 = divmod(us2, 256)
        h = self._hour
        if self._fold and protocol > 3:
            h += 128
        basestate = bytes([h, self._minute, self._second, us1, us2, us3])
        if self._tzinfo is None:
            return (basestate,)
        return (basestate, self._tzinfo)

    def __setstate(self, string, tzinfo):
        if tzinfo is not None and not isinstance(tzinfo, _tzinfo_class):
            raise TypeError("bad tzinfo state arg")
        h, self._minute, self._second, us1, us2, us3 = string
        if h > 127:
            self._fold = 1
            self._hour = h - 128
        else:
            self._fold = 0
            self._hour = h
        self._microsecond = (((us1 << 8) | us2) << 8) | us3
        self._tzinfo = tzinfo

    def __reduce_ex__(self, protocol):
        return (self.__class__, self._getstate(protocol))

    def __reduce__(self):
        return self.__reduce_ex__(2)


_tzinfo_class = tzinfo
time.min = time(0, 0, 0)
time.max = time(23, 59, 59, 999999)
time.resolution = timedelta(microseconds=1)


# ---- datetime ----


class datetime(date):
    """A date and time of day, naive or aware."""

    __slots__ = date.__slots__ + time.__slots__

    def __new__(cls, year, month=None, day=None, hour=0, minute=0, second=0, microsecond=0, tzinfo=None, *, fold=0):
        if isinstance(year, (bytes, str)) and len(year) == 10 and 1 <= ord(year[2:3]) & 0x7F <= 12:
            if isinstance(year, str):
                year = year.encode("latin1")
            self = object.__new__(cls)
            self.__setstate(year, month)
            self._hashcode = -1
            return self
        year, month, day = _check_date_fields(year, month, day)
        hour, minute, second, microsecond, fold = _check_time_fields(hour, minute, second, microsecond, fold)
        _check_tzinfo_arg(tzinfo)
        self = object.__new__(cls)
        self._year = year
        self._month = month
        self._day = day
        self._hour = hour
        self._minute = minute
        self._second = second
        self._microsecond = microsecond
        self._tzinfo = tzinfo
        self._hashcode = -1
        self._fold = fold
        return self

    @property
    def hour(self):
        return self._hour

    @property
    def minute(self):
        return self._minute

    @property
    def second(self):
        return self._second

    @property
    def microsecond(self):
        return self._microsecond

    @property
    def tzinfo(self):
        return self._tzinfo

    @property
    def fold(self):
        return self._fold

    @classmethod
    def _fromtimestamp(cls, t, utc, tz):
        frac, t = _math_modf(t)
        us = round(frac * 1e6)
        if us >= 1000000:
            t += 1
            us -= 1000000
        elif us < 0:
            t -= 1
            us += 1000000
        y, m, d, hh, mm, ss, weekday, jday, dst = _time.gmtime(t)
        ss = min(ss, 59)
        result = cls(y, m, d, hh, mm, ss, us, tz)
        if tz is not None and not utc:
            result = tz.fromutc(result)
        return result

    @classmethod
    def fromtimestamp(cls, timestamp, tz=None):
        _check_tzinfo_arg(tz)
        return cls._fromtimestamp(timestamp, tz is not None and False, tz)

    @classmethod
    def utcfromtimestamp(cls, t):
        return cls._fromtimestamp(t, True, None)

    @classmethod
    def now(cls, tz=None):
        return cls.fromtimestamp(_time.time(), tz)

    @classmethod
    def utcnow(cls):
        return cls.utcfromtimestamp(_time.time())

    @classmethod
    def combine(cls, date, time, tzinfo=True):
        if not isinstance(date, _date_class):
            raise TypeError("date argument must be a date instance")
        if not isinstance(time, _time_class):
            raise TypeError("time argument must be a time instance")
        if tzinfo is True:
            tzinfo = time.tzinfo
        return cls(date.year, date.month, date.day, time.hour, time.minute, time.second,
                   time.microsecond, tzinfo, fold=time.fold)

    @classmethod
    def fromisoformat(cls, date_string):
        if not isinstance(date_string, str):
            raise TypeError("fromisoformat: argument must be str")
        if len(date_string) < 7:
            raise ValueError("Invalid isoformat string: %r" % date_string)
        separator_location = _find_isoformat_datetime_separator(date_string)
        dstr = date_string[0:separator_location]
        tstr = date_string[(separator_location + 1):]
        try:
            date_components = _parse_isoformat_date(dstr)
        except ValueError:
            raise ValueError("Invalid isoformat string: %r" % date_string) from None
        if tstr:
            try:
                time_components, became_next_day, error_from_components = _parse_isoformat_time(tstr)
            except ValueError:
                raise ValueError("Invalid isoformat string: %r" % date_string) from None
            if error_from_components:
                raise ValueError("minute, second, and microsecond must be 0 when hour is 24")
            if became_next_day:
                year, month, day = date_components
                if month == 12 and day == 31:
                    year += 1
                    month = 1
                    day = 1
                elif day == _days_in_month(year, month):
                    month += 1
                    day = 1
                else:
                    day += 1
                date_components = [year, month, day]
        else:
            time_components = [0, 0, 0, 0, None]
        return cls(*(date_components + time_components))

    def timetuple(self):
        dst = self.dst()
        if dst is None:
            dst = -1
        elif dst:
            dst = 1
        else:
            dst = 0
        return _build_struct_time(self.year, self.month, self.day, self.hour, self.minute, self.second, dst)

    def _mktime(self):
        epoch = datetime(1970, 1, 1)
        return (self - epoch) // timedelta(0, 1)

    def timestamp(self):
        if self._tzinfo is None:
            s = self._mktime()
            return s + self.microsecond / 1e6
        return (self - _EPOCH).total_seconds()

    def utctimetuple(self):
        offset = self.utcoffset()
        if offset:
            self -= offset
        y, m, d = self.year, self.month, self.day
        hh, mm, ss = self.hour, self.minute, self.second
        return _build_struct_time(y, m, d, hh, mm, ss, 0)

    def date(self):
        return date(self._year, self._month, self._day)

    def time(self):
        return time(self.hour, self.minute, self.second, self.microsecond, fold=self.fold)

    def timetz(self):
        return time(self.hour, self.minute, self.second, self.microsecond, self._tzinfo, fold=self.fold)

    def replace(self, year=None, month=None, day=None, hour=None, minute=None, second=None,
                microsecond=None, tzinfo=True, *, fold=None):
        if year is None:
            year = self.year
        if month is None:
            month = self.month
        if day is None:
            day = self.day
        if hour is None:
            hour = self.hour
        if minute is None:
            minute = self.minute
        if second is None:
            second = self.second
        if microsecond is None:
            microsecond = self.microsecond
        if tzinfo is True:
            tzinfo = self.tzinfo
        if fold is None:
            fold = self.fold
        return type(self)(year, month, day, hour, minute, second, microsecond, tzinfo, fold=fold)

    __replace__ = replace

    def _local_timezone(self):
        return timezone(timedelta(0), "UTC")

    def astimezone(self, tz=None):
        if tz is None:
            tz = self._local_timezone()
        elif not isinstance(tz, tzinfo):
            raise TypeError("tz argument must be an instance of tzinfo")
        mytz = self.tzinfo
        if mytz is None:
            mytz = self._local_timezone()
            myoffset = mytz.utcoffset(self)
        else:
            myoffset = mytz.utcoffset(self)
            if myoffset is None:
                mytz = self.replace(tzinfo=None)._local_timezone()
                myoffset = mytz.utcoffset(self)
        if tz is mytz:
            return self
        utc = (self - myoffset).replace(tzinfo=tz)
        return tz.fromutc(utc)

    def ctime(self):
        weekday = self.toordinal() % 7 or 7
        return "%s %s %2d %02d:%02d:%02d %04d" % (
            _DAYNAMES[weekday], _MONTHNAMES[self._month], self._day,
            self._hour, self._minute, self._second, self._year)

    def isoformat(self, sep="T", timespec="auto"):
        s = "%04d-%02d-%02d%c" % (self._year, self._month, self._day, sep) + _format_time(
            self._hour, self._minute, self._second, self._microsecond, timespec)
        off = self.utcoffset()
        tz = _format_offset(off)
        if tz:
            s += tz
        return s

    def __repr__(self):
        L = [self._year, self._month, self._day, self._hour, self._minute, self._second, self._microsecond]
        if L[-1] == 0:
            del L[-1]
        if L[-1] == 0:
            del L[-1]
        s = "%s.%s(%s)" % (self.__class__.__module__, self.__class__.__qualname__, ", ".join(map(str, L)))
        if self._tzinfo is not None:
            s = s[:-1] + ", tzinfo=%r" % self._tzinfo + ")"
        if self._fold:
            s = s[:-1] + ", fold=1)"
        return s

    def __str__(self):
        return self.isoformat(sep=" ")

    @classmethod
    def strptime(cls, date_string, format):
        year, month, day, hour, minute, second, microsecond, tz = _strptime(date_string, format)
        return cls(year, month, day, hour, minute, second, microsecond, tz)

    def strftime(self, format):
        return _strftime(format, self._year, self._month, self._day, self._hour, self._minute,
                         self._second, self._microsecond, self.weekday(),
                         self.toordinal() - _days_before_year(self._year), self.utcoffset(), self.tzname())

    def utcoffset(self):
        if self._tzinfo is None:
            return None
        offset = self._tzinfo.utcoffset(self)
        _check_utc_offset("utcoffset", offset)
        return offset

    def tzname(self):
        if self._tzinfo is None:
            return None
        name = self._tzinfo.tzname(self)
        if name is not None and not isinstance(name, str):
            raise TypeError("tzinfo.tzname() must return None or string, not '%s'" % type(name).__name__)
        return name

    def dst(self):
        if self._tzinfo is None:
            return None
        offset = self._tzinfo.dst(self)
        _check_utc_offset("dst", offset)
        return offset

    def __eq__(self, other):
        if isinstance(other, datetime):
            return self._cmp(other, allow_mixed=True) == 0
        if not isinstance(other, date):
            return NotImplemented
        return False

    def __le__(self, other):
        if isinstance(other, datetime):
            return self._cmp(other) <= 0
        return NotImplemented

    def __lt__(self, other):
        if isinstance(other, datetime):
            return self._cmp(other) < 0
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, datetime):
            return self._cmp(other) >= 0
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, datetime):
            return self._cmp(other) > 0
        return NotImplemented

    def _cmp(self, other, allow_mixed=False):
        mytz = self._tzinfo
        ottz = other._tzinfo
        myoff = otoff = None
        if mytz is ottz:
            base_compare = True
        else:
            myoff = self.utcoffset()
            otoff = other.utcoffset()
            if allow_mixed:
                if myoff != self.replace(fold=not self.fold).utcoffset():
                    return 2
                if otoff != other.replace(fold=not other.fold).utcoffset():
                    return 2
            base_compare = myoff == otoff
        if base_compare:
            return _cmp((self._year, self._month, self._day, self._hour, self._minute, self._second, self._microsecond),
                        (other._year, other._month, other._day, other._hour, other._minute, other._second, other._microsecond))
        if myoff is None or otoff is None:
            if allow_mixed:
                return 2
            raise TypeError("can't compare offset-naive and offset-aware datetimes")
        diff = self - other
        if diff.days < 0:
            return -1
        return diff and 1 or 0

    def __add__(self, other):
        if not isinstance(other, timedelta):
            return NotImplemented
        delta = timedelta(self.toordinal(), hours=self._hour, minutes=self._minute, seconds=self._second,
                          microseconds=self._microsecond)
        delta += other
        hour, rem = divmod(delta.seconds, 3600)
        minute, second = divmod(rem, 60)
        if 0 < delta.days <= _MAXORDINAL:
            return type(self).combine(date.fromordinal(delta.days),
                                      time(hour, minute, second, delta.microseconds, tzinfo=self._tzinfo))
        raise OverflowError("result out of range")

    __radd__ = __add__

    def __sub__(self, other):
        if not isinstance(other, datetime):
            if isinstance(other, timedelta):
                return self + -other
            return NotImplemented
        days1 = self.toordinal()
        days2 = other.toordinal()
        secs1 = self._second + self._minute * 60 + self._hour * 3600
        secs2 = other._second + other._minute * 60 + other._hour * 3600
        base = timedelta(days1 - days2, secs1 - secs2, self._microsecond - other._microsecond)
        if self._tzinfo is other._tzinfo:
            return base
        myoff = self.utcoffset()
        otoff = other.utcoffset()
        if myoff == otoff:
            return base
        if myoff is None or otoff is None:
            raise TypeError("cannot mix naive and timezone-aware time")
        return base + otoff - myoff

    def __hash__(self):
        if self._hashcode == -1:
            if self.fold:
                t = self.replace(fold=0)
            else:
                t = self
            tzoff = t.utcoffset()
            if tzoff is None:
                self._hashcode = hash(t._getstate()[0])
            else:
                days = _ymd2ord(self.year, self.month, self.day)
                seconds = self.hour * 3600 + self.minute * 60 + self.second
                self._hashcode = hash(timedelta(days, seconds, self.microsecond) - tzoff)
        return self._hashcode

    def _getstate(self, protocol=3):
        yhi, ylo = divmod(self._year, 256)
        us2, us3 = divmod(self._microsecond, 256)
        us1, us2 = divmod(us2, 256)
        m = self._month
        if self._fold and protocol > 3:
            m += 128
        basestate = bytes([yhi, ylo, m, self._day, self._hour, self._minute, self._second, us1, us2, us3])
        if self._tzinfo is None:
            return (basestate,)
        return (basestate, self._tzinfo)

    def __setstate(self, string, tzinfo):
        if tzinfo is not None and not isinstance(tzinfo, _tzinfo_class):
            raise TypeError("bad tzinfo state arg")
        yhi, ylo, m, self._day, self._hour, self._minute, self._second, us1, us2, us3 = string
        if m > 127:
            self._fold = 1
            self._month = m - 128
        else:
            self._fold = 0
            self._month = m
        self._year = yhi * 256 + ylo
        self._microsecond = (((us1 << 8) | us2) << 8) | us3
        self._tzinfo = tzinfo

    def __reduce_ex__(self, protocol):
        return (self.__class__, self._getstate(protocol))

    def __reduce__(self):
        return self.__reduce_ex__(2)


def _math_modf(value):
    whole = int(value)
    return value - whole, whole


def _find_isoformat_datetime_separator(dtstr):
    len_dtstr = len(dtstr)
    if len_dtstr == 7:
        return 7
    assert len_dtstr > 7
    date_separator = "-"
    week_indicator = "W"
    if dtstr[4] == date_separator:
        if dtstr[5] == week_indicator:
            if len_dtstr < 8:
                raise ValueError("Invalid ISO string")
            if len_dtstr > 8 and dtstr[8] == date_separator:
                if len_dtstr == 9:
                    raise ValueError("Invalid ISO string")
                if len_dtstr > 10 and dtstr[10].isdigit():
                    return 8
                return 10
            if len_dtstr > 8 and not dtstr[8].isdigit():
                return 8
            return 7 if len_dtstr == 7 else 8
        return 10
    if dtstr[4] == week_indicator:
        if len_dtstr > 7 and dtstr[7].isdigit():
            if len_dtstr > 8:
                return 8
            return len_dtstr
        return 7
    return 8


_date_class = date
_time_class = time
datetime.min = datetime(1, 1, 1)
datetime.max = datetime(9999, 12, 31, 23, 59, 59, 999999)
datetime.resolution = timedelta(microseconds=1)
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
