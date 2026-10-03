"""Clocks and calendar conversions on shellsim's virtual clock.

The clocks come from the ``_time`` native core and advance only with modeled work and
``sleep``. The modeled process runs in UTC, so ``localtime`` and ``gmtime`` agree and the
timezone values are zero.
"""

from _time import (
    monotonic,
    monotonic_ns,
    perf_counter,
    perf_counter_ns,
    process_time,
    process_time_ns,
    sleep,
    time,
    time_ns,
)

CLOCK_REALTIME = 0
CLOCK_MONOTONIC = 1
CLOCK_PROCESS_CPUTIME_ID = 2
CLOCK_THREAD_CPUTIME_ID = 3
CLOCK_MONOTONIC_RAW = 4
CLOCK_BOOTTIME = 7
CLOCK_TAI = 11

timezone = 0
altzone = 0
daylight = 0
tzname = ("UTC", "UTC")

_DAY_NAMES = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
_DAY_FULL = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
_MONTH_NAMES = (
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
)
_MONTH_FULL = (
    "January", "February", "March", "April", "May", "June", "July", "August", "September",
    "October", "November", "December",
)
_DAYS_IN_MONTH = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


class struct_time(tuple):
    _fields = (
        "tm_year", "tm_mon", "tm_mday", "tm_hour", "tm_min", "tm_sec", "tm_wday", "tm_yday",
        "tm_isdst",
    )
    n_fields = 11
    n_sequence_fields = 9
    n_unnamed_fields = 0

    def __new__(cls, values):
        values = tuple(values)
        if len(values) < 9:
            raise TypeError("time.struct_time() takes an at least 9-sequence")
        if len(values) > 11:
            raise TypeError("time.struct_time() takes an at most 11-sequence")
        self = tuple.__new__(cls, values[:9])
        self._zone = values[9] if len(values) > 9 else "UTC"
        self._gmtoff = values[10] if len(values) > 10 else 0
        return self

    tm_year = property(lambda self: self[0])
    tm_mon = property(lambda self: self[1])
    tm_mday = property(lambda self: self[2])
    tm_hour = property(lambda self: self[3])
    tm_min = property(lambda self: self[4])
    tm_sec = property(lambda self: self[5])
    tm_wday = property(lambda self: self[6])
    tm_yday = property(lambda self: self[7])
    tm_isdst = property(lambda self: self[8])
    tm_zone = property(lambda self: self._zone)
    tm_gmtoff = property(lambda self: self._gmtoff)

    def __repr__(self):
        return "time.struct_time(" + ", ".join(
            field + "=" + repr(value) for field, value in zip(self._fields, self)
        ) + ")"


def _is_leap(year):
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def _days_before_year(year):
    y = year - 1
    return y * 365 + y // 4 - y // 100 + y // 400


def _days_before_month(year, month):
    days = sum(_DAYS_IN_MONTH[: month - 1])
    if month > 2 and _is_leap(year):
        days += 1
    return days


def _civil_from_days(days):
    """Calendar date for a day count since 1970-01-01, valid for any year."""
    days += 719468
    era = (days if days >= 0 else days - 146096) // 146097
    doe = days - era * 146097
    yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
    year = yoe + era * 400
    doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
    mp = (5 * doy + 2) // 153
    day = doy - (153 * mp + 2) // 5 + 1
    month = mp + 3 if mp < 10 else mp - 9
    if month <= 2:
        year += 1
    return year, month, day


def gmtime(seconds=None):
    if seconds is None:
        seconds = time()
    whole = int(seconds // 1)
    days, remainder = divmod(whole, 86400)
    year, month, day = _civil_from_days(days)
    hour, remainder = divmod(remainder, 3600)
    minute, second = divmod(remainder, 60)
    weekday = (days + 3) % 7
    yearday = _days_before_month(year, month) + day
    return struct_time((year, month, day, hour, minute, second, weekday, yearday, 0, "UTC", 0))


localtime = gmtime


def _check_tuple(t):
    if t is None:
        return gmtime()
    if not isinstance(t, tuple):
        raise TypeError("Tuple or struct_time argument required")
    if len(t) < 9:
        raise TypeError("time tuple must have at least 9 elements")
    return t


def mktime(t):
    t = _check_tuple(t)
    year, month, day, hour, minute, second = t[0], t[1], t[2], t[3], t[4], t[5]
    if month < 1 or month > 12:
        raise OverflowError("mktime argument out of range")
    days = _days_before_year(year) - _days_before_year(1970) + _days_before_month(year, month) + day - 1
    return float(days * 86400 + hour * 3600 + minute * 60 + second)


def asctime(t=None):
    t = _check_tuple(t)
    return "%s %s %2d %02d:%02d:%02d %d" % (
        _DAY_NAMES[t[6]], _MONTH_NAMES[t[1] - 1], t[2], t[3], t[4], t[5], t[0]
    )


def ctime(seconds=None):
    return asctime(gmtime(seconds))


def strftime(format, t=None):
    t = _check_tuple(t)
    year, month, day, hour, minute, second, weekday, yearday = (
        t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7],
    )
    result = []
    index = 0
    while index < len(format):
        char = format[index]
        if char != "%" or index + 1 >= len(format):
            result.append(char)
            index += 1
            continue
        directive = format[index + 1]
        index += 2
        if directive == "Y":
            result.append("%d" % year)
        elif directive == "y":
            result.append("%02d" % (year % 100))
        elif directive == "C":
            result.append("%02d" % (year // 100))
        elif directive == "m":
            result.append("%02d" % month)
        elif directive == "d":
            result.append("%02d" % day)
        elif directive == "e":
            result.append("%2d" % day)
        elif directive == "H":
            result.append("%02d" % hour)
        elif directive == "I":
            result.append("%02d" % (hour % 12 or 12))
        elif directive == "M":
            result.append("%02d" % minute)
        elif directive == "S":
            result.append("%02d" % second)
        elif directive == "p":
            result.append("AM" if hour < 12 else "PM")
        elif directive == "j":
            result.append("%03d" % yearday)
        elif directive == "a":
            result.append(_DAY_NAMES[weekday])
        elif directive == "A":
            result.append(_DAY_FULL[weekday])
        elif directive == "b" or directive == "h":
            result.append(_MONTH_NAMES[month - 1])
        elif directive == "B":
            result.append(_MONTH_FULL[month - 1])
        elif directive == "w":
            result.append(str((weekday + 1) % 7))
        elif directive == "u":
            result.append(str(weekday + 1))
        elif directive == "z":
            result.append("+0000")
        elif directive == "Z":
            result.append("UTC")
        elif directive == "%":
            result.append("%")
        elif directive == "n":
            result.append("\n")
        elif directive == "t":
            result.append("\t")
        elif directive == "F":
            result.append("%d-%02d-%02d" % (year, month, day))
        elif directive == "T":
            result.append("%02d:%02d:%02d" % (hour, minute, second))
        elif directive == "R":
            result.append("%02d:%02d" % (hour, minute))
        elif directive == "D":
            result.append("%02d/%02d/%02d" % (month, day, year % 100))
        elif directive == "c":
            result.append(asctime(t))
        elif directive == "x":
            result.append("%02d/%02d/%02d" % (month, day, year % 100))
        elif directive == "X":
            result.append("%02d:%02d:%02d" % (hour, minute, second))
        elif directive == "s":
            result.append("%d" % int(mktime(t)))
        elif directive == "U":
            result.append("%02d" % ((yearday + 6 - (weekday + 1) % 7) // 7))
        elif directive == "W":
            result.append("%02d" % ((yearday + 6 - weekday) // 7))
        elif directive == "G":
            result.append("%d" % year)
        elif directive == "V":
            result.append("%02d" % max(1, (yearday + 6 - weekday) // 7))
        elif directive == "f":
            result.append("000000")
        else:
            result.append("%" + directive)
    return "".join(result)


def _strptime_number(string, position, width_max, name):
    end = position
    while end < len(string) and end - position < width_max and string[end].isdigit():
        end += 1
    if end == position:
        raise ValueError("time data %r does not match format (%s)" % (string, name))
    return int(string[position:end]), end


def _strptime_name(string, position, names):
    lowered = string[position:].lower()
    for index, name in enumerate(names):
        if lowered.startswith(name.lower()):
            return index, position + len(name)
    raise ValueError("time data %r does not match format" % string)


def strptime(string, format="%a %b %d %H:%M:%S %Y"):
    year, month, day, hour, minute, second = 1900, 1, 1, 0, 0, 0
    yearday = None
    weekday = None
    pm = None
    position = 0
    index = 0
    while index < len(format):
        char = format[index]
        if char != "%":
            if char.isspace():
                while position < len(string) and string[position].isspace():
                    position += 1
            elif position < len(string) and string[position] == char:
                position += 1
            else:
                raise ValueError("time data %r does not match format %r" % (string, format))
            index += 1
            continue
        directive = format[index + 1] if index + 1 < len(format) else ""
        index += 2
        if directive == "Y":
            year, position = _strptime_number(string, position, 4, "%Y")
        elif directive == "y":
            year, position = _strptime_number(string, position, 2, "%y")
            year += 2000 if year < 69 else 1900
        elif directive == "m":
            month, position = _strptime_number(string, position, 2, "%m")
        elif directive == "d" or directive == "e":
            day, position = _strptime_number(string, position, 2, "%d")
        elif directive == "H":
            hour, position = _strptime_number(string, position, 2, "%H")
        elif directive == "I":
            hour, position = _strptime_number(string, position, 2, "%I")
        elif directive == "M":
            minute, position = _strptime_number(string, position, 2, "%M")
        elif directive == "S":
            second, position = _strptime_number(string, position, 2, "%S")
        elif directive == "j":
            yearday, position = _strptime_number(string, position, 3, "%j")
        elif directive == "f":
            _, position = _strptime_number(string, position, 6, "%f")
        elif directive == "p":
            marker, position = _strptime_name(string, position, ("AM", "PM"))
            pm = marker == 1
        elif directive == "b" or directive == "h":
            month_index, position = _strptime_name(string, position, _MONTH_NAMES)
            month = month_index + 1
        elif directive == "B":
            month_index, position = _strptime_name(string, position, _MONTH_FULL)
            month = month_index + 1
        elif directive == "a":
            weekday, position = _strptime_name(string, position, _DAY_NAMES)
        elif directive == "A":
            weekday, position = _strptime_name(string, position, _DAY_FULL)
        elif directive == "z":
            if position < len(string) and string[position] in "+-":
                position += 1
                _, position = _strptime_number(string, position, 4, "%z")
            elif string.startswith("Z", position):
                position += 1
        elif directive == "Z":
            _, position = _strptime_name(string, position, ("UTC", "GMT"))
        elif directive == "%":
            if string.startswith("%", position):
                position += 1
            else:
                raise ValueError("time data %r does not match format %r" % (string, format))
        else:
            raise ValueError("'%s' is a bad directive in format '%s'" % (directive, format))
    if position != len(string):
        raise ValueError("unconverted data remains: " + string[position:])
    if pm is not None:
        if pm and hour < 12:
            hour += 12
        elif not pm and hour == 12:
            hour = 0
    if month < 1 or month > 12:
        raise ValueError("time data %r does not match format %r" % (string, format))
    if yearday is None:
        yearday = _days_before_month(year, month) + day
    if weekday is None:
        days = _days_before_year(year) - _days_before_year(1970) + yearday - 1
        weekday = (days + 3) % 7
    return struct_time((year, month, day, hour, minute, second, weekday, yearday, -1))


def tzset():
    return None


def thread_time():
    return process_time()


def thread_time_ns():
    return process_time_ns()


def clock_gettime(clk_id):
    if clk_id == CLOCK_REALTIME or clk_id == CLOCK_TAI:
        return time()
    if clk_id in (CLOCK_MONOTONIC, CLOCK_MONOTONIC_RAW, CLOCK_BOOTTIME):
        return monotonic()
    if clk_id in (CLOCK_PROCESS_CPUTIME_ID, CLOCK_THREAD_CPUTIME_ID):
        return process_time()
    raise OSError(22, "Invalid argument")


def clock_gettime_ns(clk_id):
    if clk_id == CLOCK_REALTIME or clk_id == CLOCK_TAI:
        return time_ns()
    if clk_id in (CLOCK_MONOTONIC, CLOCK_MONOTONIC_RAW, CLOCK_BOOTTIME):
        return monotonic_ns()
    if clk_id in (CLOCK_PROCESS_CPUTIME_ID, CLOCK_THREAD_CPUTIME_ID):
        return process_time_ns()
    raise OSError(22, "Invalid argument")


def clock_getres(clk_id):
    clock_gettime(clk_id)
    return 1e-09


def clock_settime(clk_id, seconds):
    raise PermissionError(1, "Operation not permitted")


clock_settime_ns = clock_settime


class _ClockInfo:
    def __init__(self, implementation, monotonic, adjustable, resolution):
        self.implementation = implementation
        self.monotonic = monotonic
        self.adjustable = adjustable
        self.resolution = resolution

    def __repr__(self):
        return "namespace(adjustable=%r, implementation=%r, monotonic=%r, resolution=%r)" % (
            self.adjustable, self.implementation, self.monotonic, self.resolution,
        )


def get_clock_info(name):
    if name == "time":
        return _ClockInfo("clock_gettime(CLOCK_REALTIME)", False, True, 1e-09)
    if name in ("monotonic", "perf_counter"):
        return _ClockInfo("clock_gettime(CLOCK_MONOTONIC)", True, False, 1e-09)
    if name in ("process_time", "thread_time"):
        return _ClockInfo("clock_gettime(CLOCK_PROCESS_CPUTIME_ID)", True, False, 1e-09)
    raise ValueError("unknown clock")


def pthread_getcpuclockid(thread_id):
    return CLOCK_THREAD_CPUTIME_ID
