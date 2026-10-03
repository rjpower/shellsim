# Portable checks of the pure-Python standard-library surface: each test exercises the public API
# an ordinary program relies on and asserts values and exception types, never message text.

import bisect
import cmath
import codecs
import copy
import datetime
import fnmatch
import functools
import hashlib
import json
import math
import random
import re
import struct
import textwrap
import zlib


def test_functools_wrappers_caches_and_dispatch():
    def base(a, b=2, *, c=3):
        """base doc"""
        return a + b + c

    @functools.wraps(base)
    def wrapper(*args, **kwargs):
        return base(*args, **kwargs)

    assert wrapper.__name__ == "base"
    assert wrapper.__doc__ == "base doc"
    assert wrapper.__wrapped__ is base

    add_one = functools.partial(base, 1, c=0)
    assert add_one() == 3
    assert add_one(5) == 6
    assert add_one.func is base and add_one.args == (1,) and add_one.keywords == {"c": 0}

    calls = []

    @functools.lru_cache(maxsize=2)
    def square(n):
        calls.append(n)
        return n * n

    assert [square(2), square(2), square(3), square(4), square(2)] == [4, 4, 9, 16, 4]
    assert calls == [2, 3, 4, 2]
    info = square.cache_info()
    assert (info.hits, info.misses, info.maxsize, info.currsize) == (1, 4, 2, 2)
    square.cache_clear()
    assert square.cache_info().currsize == 0

    @functools.singledispatch
    def describe(value):
        return "object"

    @describe.register(int)
    def _(value):
        return "int"

    describe.register(list, lambda value: "list")
    assert [describe(1), describe([]), describe("s")] == ["int", "list", "object"]
    assert describe.dispatch(bool) is describe.registry[int]

    @functools.total_ordering
    class Version:
        def __init__(self, n):
            self.n = n

        def __eq__(self, other):
            return self.n == other.n

        def __lt__(self, other):
            return self.n < other.n

    assert Version(1) <= Version(2) and Version(3) > Version(2) and Version(2) >= Version(2)
    assert functools.reduce(lambda x, y: x * y, [1, 2, 3, 4]) == 24
    assert sorted([3, 1, 2], key=functools.cmp_to_key(lambda a, b: b - a)) == [3, 2, 1]

    class Holder:
        def __init__(self):
            self.computed = 0

        @functools.cached_property
        def value(self):
            self.computed += 1
            return 42

    holder = Holder()
    assert holder.value == 42 and holder.value == 42 and holder.computed == 1


def test_copy_handles_cycles_memo_and_custom_hooks():
    shared = [1, 2]
    original = {"a": shared, "b": shared, "t": (shared, 3)}
    shallow = copy.copy(original)
    deep = copy.deepcopy(original)
    assert shallow["a"] is shared
    assert deep["a"] is not shared and deep["a"] is deep["b"] and deep["t"][0] is deep["a"]

    cyclic = []
    cyclic.append(cyclic)
    clone = copy.deepcopy(cyclic)
    assert clone[0] is clone and clone is not cyclic

    class Point:
        def __init__(self, x, y):
            self.x = x
            self.y = y

        def __copy__(self):
            return Point(self.x, "copied")

        def __deepcopy__(self, memo):
            return Point(copy.deepcopy(self.x, memo), "deep")

    assert copy.copy(Point(1, 2)).y == "copied"
    assert copy.deepcopy(Point([1], 2)).y == "deep"
    assert isinstance(copy.Error(), Exception) and copy.error is copy.Error


def test_struct_packs_and_reports_errors_as_struct_error():
    assert struct.calcsize("<ihb") == 7
    assert struct.calcsize(">iI") == 8
    packed = struct.pack("<i2sf?", -2, b"ab", 1.5, True)
    assert struct.unpack("<i2sf?", packed) == (-2, b"ab", 1.5, True)
    assert struct.unpack_from("<h", b"\x00\x01\x00", 1) == (1,)
    buffer = bytearray(4)
    struct.pack_into("<H", buffer, 2, 258)
    assert bytes(buffer) == b"\x00\x00\x02\x01"
    assert list(struct.iter_unpack("<h", b"\x01\x00\x02\x00")) == [(1,), (2,)]
    compiled = struct.Struct(">I")
    assert compiled.size == 4 and compiled.unpack(compiled.pack(7)) == (7,)
    for failing in (
        lambda: struct.pack("<b", 300),
        lambda: struct.pack("<i", 1, 2),
        lambda: struct.unpack("<i", b"\x00"),
        lambda: struct.pack("<i", "text"),
    ):
        try:
            failing()
        except struct.error:
            pass
        else:
            raise AssertionError("struct.error expected")
    assert issubclass(struct.error, Exception)


def test_json_round_trips_with_options_and_positions_errors():
    document = {"b": [1, 2.5, None, True], "a": {"nested": "ü"}, "c": "line\nbreak"}
    text = json.dumps(document, sort_keys=True, indent=2, ensure_ascii=False)
    assert text.startswith('{\n  "a": {')
    assert json.loads(text) == document
    assert json.dumps("ü") == '"\\u00fc"'
    assert json.dumps([1, 2], separators=(",", ":")) == "[1,2]"
    assert json.dumps({1: "one", (1, 2): "skipped"}, skipkeys=True) == '{"1": "one"}'
    assert json.loads('{"x": 1e3, "y": -0.5}') == {"x": 1000.0, "y": -0.5}
    assert json.loads("[1, 2]", object_hook=None) == [1, 2]
    assert json.loads('{"__v": 1}', object_hook=lambda d: d.get("__v", d)) == 1
    assert json.loads("1.5", parse_float=str) == "1.5"

    class Encoder(json.JSONEncoder):
        def default(self, value):
            if isinstance(value, set):
                return sorted(value)
            return super().default(value)

    assert json.dumps({"s": {3, 1}}, cls=Encoder) == '{"s": [1, 3]}'
    try:
        json.dumps({"s": {1}})
    except TypeError:
        pass
    else:
        raise AssertionError("TypeError expected")
    try:
        json.loads('{"a": 1,}')
    except json.JSONDecodeError as error:
        assert isinstance(error, ValueError)
        assert error.lineno == 1 and error.colno == 8 and error.pos == 7
    else:
        raise AssertionError("JSONDecodeError expected")
    try:
        json.loads("NaN", parse_constant=lambda name: (_ for _ in ()).throw(ValueError(name)))
    except ValueError:
        pass
    decoded = json.JSONDecoder().raw_decode("[1] trailing")
    assert decoded == ([1], 3)


def test_codecs_lookup_incremental_and_streams():
    info = codecs.lookup("UTF8")
    assert info.name == "utf-8"
    assert codecs.encode("é", "utf-8") == b"\xc3\xa9"
    assert codecs.decode(b"\xc3\xa9", "utf-8") == "é"
    assert codecs.encode("abc", "rot13") == "nop"
    decoder = codecs.getincrementaldecoder("utf-8")()
    assert decoder.decode(b"\xc3") == "" and decoder.decode(b"\xa9", final=True) == "é"
    encoder = codecs.getincrementalencoder("latin-1")()
    assert encoder.encode("é") == b"\xe9"
    assert "".join(codecs.iterdecode([b"a", b"b"], "ascii")) == "ab"
    assert codecs.BOM_UTF8 == b"\xef\xbb\xbf"
    assert codecs.lookup_error("strict") is codecs.strict_errors
    try:
        codecs.lookup("no-such-codec")
    except LookupError:
        pass
    else:
        raise AssertionError("LookupError expected")
    try:
        "é".encode("ascii")
    except UnicodeEncodeError:
        pass
    assert "é".encode("ascii", "backslashreplace") == b"\\xe9"


def test_textwrap_fnmatch_and_bisect():
    text = "The quick brown fox jumps over the lazy dog and keeps running far away"
    assert textwrap.wrap(text, width=20) == [
        "The quick brown fox",
        "jumps over the lazy",
        "dog and keeps",
        "running far away",
    ]
    assert textwrap.fill("a b c", width=3) == "a b\nc"
    assert textwrap.shorten("hello world again", width=12) == "hello [...]"
    assert textwrap.dedent("  a\n    b\n  c") == "a\n  b\nc"
    assert textwrap.indent("a\n\nb", "> ") == "> a\n\n> b"
    assert textwrap.wrap("long-hyphenated-word here", width=10) == ["long-hyphe", "nated-word", "here"]
    assert textwrap.wrap("long-hyphenated-word here", width=10, break_long_words=False) == [
        "long-",
        "hyphenated-",
        "word here",
    ]
    wrapper = textwrap.TextWrapper(width=10, max_lines=1, placeholder="…")
    assert wrapper.wrap("one two three four") == ["one two…"]

    assert fnmatch.fnmatch("data.txt", "*.txt") and fnmatch.fnmatch("data.txt", "d?ta.*")
    assert not fnmatch.fnmatchcase("Data.TXT", "*.txt")
    assert fnmatch.filter(["a.py", "b.txt", "c.py"], "*.py") == ["a.py", "c.py"]
    assert fnmatch.fnmatch("x[1]", "x[[]1]")
    assert re.match(fnmatch.translate("*.py"), "a.py") and not re.match(fnmatch.translate("*.py"), "a.pyc")

    values = [1, 3, 3, 5]
    assert bisect.bisect_left(values, 3) == 1 and bisect.bisect_right(values, 3) == 3
    assert bisect.bisect(values, 4) == 3
    bisect.insort(values, 4)
    assert values == [1, 3, 3, 4, 5]
    records = [("a", 1), ("b", 3)]
    bisect.insort_left(records, ("c", 2), key=lambda record: record[1])
    assert records == [("a", 1), ("c", 2), ("b", 3)]
    assert bisect.bisect_left(values, 3, lo=2) == 2


def test_zlib_and_cmath():
    payload = b"shellsim " * 50
    compressed = zlib.compress(payload, 6)
    assert len(compressed) < len(payload)
    assert zlib.decompress(compressed) == payload
    assert zlib.crc32(b"abc") == 891568578 and zlib.adler32(b"abc") == 38600999
    compressor = zlib.compressobj()
    streamed = compressor.compress(payload[:100]) + compressor.compress(payload[100:]) + compressor.flush()
    assert zlib.decompress(streamed) == payload
    decompressor = zlib.decompressobj()
    assert decompressor.decompress(streamed[:10]) + decompressor.decompress(streamed[10:]) == payload
    assert decompressor.eof
    try:
        zlib.decompress(b"not deflate data")
    except zlib.error as error:
        assert isinstance(error, Exception)
    else:
        raise AssertionError("zlib.error expected")

    assert cmath.sqrt(-4) == 2j
    assert abs(cmath.exp(1j * cmath.pi) + 1) < 1e-12
    assert cmath.phase(-1) == math.pi
    magnitude, angle = cmath.polar(1j)
    assert magnitude == 1.0 and abs(angle - math.pi / 2) < 1e-12
    assert cmath.isclose(cmath.rect(1, 0), 1 + 0j)
    assert cmath.isnan(cmath.nan) and cmath.isinf(cmath.inf) and cmath.infj == complex(0, math.inf)
    assert abs(cmath.log(cmath.e) - 1) < 1e-12 and abs(cmath.log(8, 2) - 3) < 1e-12


def test_random_api_is_deterministic_per_seed():
    first = random.Random(1234)
    second = random.Random(1234)
    draws = [first.random() for _ in range(5)]
    assert draws == [second.random() for _ in range(5)]
    assert all(0.0 <= draw < 1.0 for draw in draws)
    assert first.randint(1, 6) in range(1, 7)
    assert first.randrange(0, 10, 2) % 2 == 0
    assert first.choice("abc") in "abc"
    population = list(range(20))
    sample = first.sample(population, 5)
    assert len(sample) == 5 and len(set(sample)) == 5 and set(sample) <= set(population)
    assert first.sample(["x", "y"], 3, counts=[2, 2]).count("x") <= 2
    assert len(first.choices([1, 2, 3], weights=[1, 0, 0], k=4)) == 4
    assert first.choices([1, 2, 3], weights=[1, 0, 0], k=4) == [1, 1, 1, 1]
    deck = [1, 2, 3, 4]
    first.shuffle(deck)
    assert sorted(deck) == [1, 2, 3, 4]
    assert 0 <= first.getrandbits(10) < 1024
    assert len(first.randbytes(3)) == 3
    assert 0.0 <= first.uniform(0, 1) <= 1.0
    state = first.getstate()
    before = first.random()
    first.setstate(state)
    assert first.random() == before
    assert random.Random("seed text").random() == random.Random("seed text").random()
    for failing in (lambda: first.randrange(5, 1), lambda: first.sample(population, 21), lambda: first.choice([])):
        try:
            failing()
        except (ValueError, IndexError):
            pass
        else:
            raise AssertionError("ValueError or IndexError expected")
    assert isinstance(random.SystemRandom().random(), float)


def test_hashlib_digests_and_copies():
    sha = hashlib.sha256(b"abc")
    assert sha.hexdigest() == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert sha.digest_size == 32 and sha.block_size == 64 and sha.name == "sha256"
    assert len(sha.digest()) == 32
    forked = sha.copy()
    forked.update(b"d")
    assert forked.hexdigest() != sha.hexdigest()
    assert hashlib.md5(b"").hexdigest() == "d41d8cd98f00b204e9800998ecf8427e"
    assert hashlib.sha1(b"abc").hexdigest() == "a9993e364706816aba3e25717850c26c9cd0d89d"
    assert hashlib.new("sha512", b"abc").hexdigest()[:16] == "ddaf35a193617aba"
    assert {"md5", "sha1", "sha256", "sha512"} <= set(hashlib.algorithms_available)
    try:
        hashlib.new("no-such-hash")
    except ValueError:
        pass
    else:
        raise AssertionError("ValueError expected")
    try:
        hashlib.sha256("text")
    except TypeError:
        pass
    else:
        raise AssertionError("TypeError expected")


def test_datetime_arithmetic_formatting_and_parsing():
    moment = datetime.datetime(2024, 2, 28, 23, 30, 15, 500000)
    later = moment + datetime.timedelta(hours=1, minutes=45)
    assert later == datetime.datetime(2024, 2, 29, 1, 15, 15, 500000)
    assert (later - moment) == datetime.timedelta(seconds=6300)
    assert datetime.timedelta(1, 2, 3, 4, 5, 6, 1) == datetime.timedelta(days=8, seconds=21902, microseconds=4003)
    assert datetime.timedelta(days=1, hours=-25).days == -1
    assert str(datetime.timedelta(hours=26, microseconds=1)) == "1 day, 2:00:00.000001"
    assert moment.isoformat() == "2024-02-28T23:30:15.500000"
    assert moment.strftime("%Y-%m-%d %H:%M:%S %j %A %b %%") == "2024-02-28 23:30:15 059 Wednesday Feb %"
    assert datetime.datetime.strptime("2024-02-29 01:15", "%Y-%m-%d %H:%M") == datetime.datetime(2024, 2, 29, 1, 15)
    assert datetime.datetime.fromisoformat("2024-02-29T01:15:00+02:00").utcoffset() == datetime.timedelta(hours=2)
    assert datetime.date(2024, 2, 29).weekday() == 3 and datetime.date(2024, 2, 29).isoweekday() == 4
    assert datetime.date(2024, 12, 30).isocalendar() == (2025, 1, 1)
    assert datetime.date.fromordinal(datetime.date(2000, 1, 1).toordinal() + 366) == datetime.date(2001, 1, 1)
    assert datetime.date(2024, 3, 1) - datetime.date(2024, 2, 1) == datetime.timedelta(days=29)
    utc = datetime.timezone.utc
    aware = datetime.datetime(1970, 1, 2, tzinfo=utc)
    assert aware.timestamp() == 86400.0
    assert datetime.datetime.fromtimestamp(86400, utc) == aware
    assert aware.astimezone(datetime.timezone(datetime.timedelta(hours=-5))).hour == 19
    assert str(datetime.timezone(datetime.timedelta(hours=5, minutes=30))) == "UTC+05:30"
    assert datetime.time(13, 5, tzinfo=utc).isoformat() == "13:05:00+00:00"
    assert datetime.time.fromisoformat("01:02:03.000004").microsecond == 4
    assert moment.replace(year=2025).year == 2025
    assert moment.date() == datetime.date(2024, 2, 28) and moment.time() == datetime.time(23, 30, 15, 500000)
    assert datetime.MINYEAR == 1 and datetime.MAXYEAR == 9999
    for failing in (
        lambda: datetime.date(2023, 2, 29),
        lambda: datetime.datetime.strptime("x", "%Y"),
        lambda: aware - moment,
        lambda: datetime.datetime.fromisoformat("not a date"),
    ):
        try:
            failing()
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError("ValueError or TypeError expected")
