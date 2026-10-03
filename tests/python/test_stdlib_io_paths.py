# Portable checks of io, pathlib, shutil, contextlib, inspect and the builtins they rely on.
# Filesystem cases work inside a temporary directory so the CPython reference run leaves no files.
# ExceptionGroup is a 3.11 builtin; ruff's Python 3.9 target does not know it.
# ruff: noqa: F821

import contextlib
import inspect
import io
import os
import pathlib
import shutil
import sys
import tempfile


def test_open_modes_positions_and_newlines():
    with tempfile.TemporaryDirectory() as root:
        path = os.path.join(root, "f.txt")
        with open(path, "w", encoding="utf-8", newline="") as handle:
            assert isinstance(handle, io.TextIOWrapper) and handle.mode == "w" and handle.encoding == "utf-8"
            assert handle.write("héllo\nworld\r\nlast") == 17
            assert handle.tell() == 18
        with open(path, "rb") as handle:
            assert isinstance(handle, io.BufferedReader)
            assert handle.read(3) == b"h\xc3\xa9" and handle.peek()[:2] == b"ll"
            assert handle.readline() == b"llo\n" and handle.readlines() == [b"world\r\n", b"last"]
        with open(path, encoding="utf-8") as handle:
            assert handle.readline() == "héllo\n"
            assert handle.read() == "world\nlast" and handle.newlines == ("\n", "\r\n")
        with open(path, encoding="utf-8", newline="") as handle:
            assert handle.read() == "héllo\nworld\r\nlast"
        with open(path, "rb", buffering=0) as handle:
            assert isinstance(handle, io.FileIO) and handle.readall() == b"h\xc3\xa9llo\nworld\r\nlast"
        with open(path, "r+b") as handle:
            handle.seek(1)
            handle.write(b"E")
            handle.seek(0)
            assert handle.read(3) == b"hE\xa9"
            handle.truncate(3)
        with open(path, "ab") as handle:
            handle.write(b"!")
        with open(path, "rb") as handle:
            assert handle.read() == b"hE\xa9!"
        # Two open append handles interleave: each write lands at the live end of the file
        # rather than overwriting the other handle's writes with a stale snapshot.
        first = open(path, "a", encoding="latin-1")
        second = open(path, "a", encoding="latin-1")
        first.write("1")
        first.flush()
        second.write("2")
        second.flush()
        first.write("3")
        first.close()
        second.close()
        with open(path, "rb") as handle:
            assert handle.read() == b"hE\xa9!123"
        for mode, exception in (("rw", ValueError), ("x", FileExistsError)):
            try:
                open(path, mode)
            except exception:
                pass
            else:
                raise AssertionError(f"{exception.__name__} expected")
        try:
            open(path, "rb", encoding="utf-8")
        except ValueError:
            pass
        else:
            raise AssertionError("ValueError expected")
        try:
            open(os.path.join(root, "missing.txt"))
        except FileNotFoundError as error:
            assert error.errno == 2
        else:
            raise AssertionError("FileNotFoundError expected")
        with open(path, "rb") as handle:
            try:
                handle.write(b"x")
            except io.UnsupportedOperation:
                pass
            else:
                raise AssertionError("UnsupportedOperation expected")
        handle = open(path)
        handle.close()
        assert handle.closed
        try:
            handle.read()
        except ValueError:
            pass
        else:
            raise AssertionError("ValueError expected")


def test_in_memory_streams_and_wrappers():
    binary = io.BytesIO(b"abc\ndef")
    assert binary.read(2) == b"ab" and binary.readline() == b"c\n" and binary.getvalue() == b"abc\ndef"
    assert binary.seek(-2, io.SEEK_END) == 5 and binary.read() == b"ef"
    binary.write(b"XY")
    assert binary.getvalue() == b"abc\ndefXY" and binary.getbuffer().nbytes == 9
    binary.truncate(2)
    assert binary.getvalue() == b"ab"
    binary.close()
    try:
        binary.getvalue()
    except ValueError:
        pass
    else:
        raise AssertionError("ValueError expected")

    text = io.StringIO("a\r\nb\rc\n", newline=None)
    assert text.read() == "a\nb\nc\n" and text.newlines == ("\r", "\n", "\r\n")
    text = io.StringIO(newline="\r\n")
    text.write("x\ny")
    assert text.getvalue() == "x\r\ny"
    text = io.StringIO("line1\nline2\n")
    assert list(text) == ["line1\n", "line2\n"] and text.tell() == 12
    text.seek(0)
    assert text.readline(2) == "li" and text.readline() == "ne1\n"

    wrapper = io.TextIOWrapper(io.BytesIO("ä\nb".encode("utf-8")), encoding="utf-8")
    assert wrapper.read() == "ä\nb"
    writer = io.TextIOWrapper(io.BytesIO(), encoding="latin-1", write_through=True)
    writer.write("é")
    assert writer.buffer.getvalue() == b"\xe9"
    decoder = io.IncrementalNewlineDecoder(None, True)
    assert decoder.decode("a\r") == "a" and decoder.decode("\nb\r") == "\nb"
    assert decoder.decode("", final=True) == "\n" and decoder.newlines == ("\r", "\r\n")

    reader = io.BufferedReader(io.BytesIO(b"0123456789"), buffer_size=4)
    assert reader.read(3) == b"012" and reader.read1(2) == b"3" and reader.peek()[:1] == b"4"
    assert reader.read() == b"456789"

    class Source(io.RawIOBase):
        def __init__(self):
            self.data = bytearray(b"hello world")
            self.offset = 0

        def readable(self):
            return True

        def readinto(self, buffer):
            count = min(len(buffer), len(self.data) - self.offset)
            buffer[:count] = self.data[self.offset : self.offset + count]
            self.offset += count
            return count

    assert io.BufferedReader(Source()).read() == b"hello world"
    assert Source().read(5) == b"hello" and Source().readall() == b"hello world"
    assert isinstance(sys.stdout, io.TextIOBase) and isinstance(io.StringIO(), io.TextIOBase)
    assert issubclass(io.FileIO, io.RawIOBase) and not isinstance(io.BytesIO(), io.TextIOBase)
    assert sys.stdout.fileno() == 1 and sys.stderr.fileno() == 2 and not sys.stdout.isatty()
    assert io.DEFAULT_BUFFER_SIZE > 0 and io.SEEK_END == 2


def test_print_writes_to_any_file_object_and_honours_redirected_stdout():
    buffer = io.StringIO()
    print("a", 1, sep="-", end="!", file=buffer, flush=True)
    assert buffer.getvalue() == "a-1!"
    with contextlib.redirect_stdout(buffer):
        print("captured")
    assert buffer.getvalue() == "a-1!captured\n"
    with contextlib.redirect_stderr(buffer):
        print("err", file=sys.stderr)
    assert buffer.getvalue().endswith("err\n")
    sys.stdout.flush()


def test_pure_paths_split_join_match_and_compare():
    path = pathlib.PurePosixPath("/usr/local/lib/python3.12/site-packages/foo.tar.gz")
    assert path.name == "foo.tar.gz" and path.stem == "foo.tar" and path.suffix == ".gz"
    assert path.suffixes == [".tar", ".gz"] and path.parent.name == "site-packages"
    assert path.parents[1].name == "python3.12" and len(path.parents) == 6
    assert path.parts[:3] == ("/", "usr", "local") and path.root == "/" and path.anchor == "/"
    assert str(path.with_name("bar.txt")) == "/usr/local/lib/python3.12/site-packages/bar.txt"
    assert path.with_suffix(".zip").name == "foo.tar.zip" and path.with_stem("baz").name == "baz.gz"
    assert path.with_suffix("").name == "foo.tar" and path.is_absolute()
    assert path.as_uri() == "file:///usr/local/lib/python3.12/site-packages/foo.tar.gz"
    PurePath = pathlib.PurePath
    assert str(PurePath("a//b/./c/")) == "a/b/c" and str(PurePath("")) == "." and str(PurePath(".")) == "."
    assert str(PurePath("a", "/b", "c")) == "/b/c" and str(PurePath("a") / "b" / PurePath("c")) == "a/b/c"
    assert str("x" / PurePath("y")) == "x/y"
    assert PurePath("a/b") == PurePath("a//b") and PurePath("a") < PurePath("b")
    assert PurePath("a/b") in {PurePath("a/b")} and hash(PurePath("a")) == hash(PurePath("a/"))
    assert type(PurePath("a")) is pathlib.PurePosixPath
    assert str(PurePath("/a/b/c").relative_to("/a")) == "b/c"
    assert PurePath("/a/b/c").is_relative_to("/a/b") and not PurePath("/a/b/c").is_relative_to("/x")
    assert str(PurePath("/a/b").relative_to("/a/c/d", walk_up=True)) == "../../b"
    for failing in (
        lambda: PurePath("/a/b").relative_to("/x"),
        lambda: PurePath("a/b").relative_to("/a"),
        lambda: PurePath("/").with_name("x"),
        lambda: PurePath("a").with_suffix("x"),
        lambda: PurePath("a").with_name("x/y"),
    ):
        try:
            failing()
        except ValueError:
            pass
        else:
            raise AssertionError("ValueError expected")
    assert PurePath("a/b.py").match("*.py") and PurePath("a/b.py").match("a/*.py")
    assert PurePath("/a/b.py").match("/*.py") is False and not PurePath("a/b.py").match("c/*.py")
    assert PurePath("a/b/c.py").full_match("a/**/*.py") and not PurePath("a/b/c.py").full_match("a/*.py")
    assert PurePath(".hidden").suffix == "" and PurePath(".hidden").stem == ".hidden"
    assert PurePath("..").name == ".." and PurePath("a/..").parts == ("a", "..")
    assert PurePath("/").name == "" and PurePath("/").parent == PurePath("/") and PurePath("a").parent == PurePath(".")
    assert os.fspath(PurePath("q")) == "q" and bytes(PurePath("q")) == b"q"
    assert isinstance(pathlib.Path("x"), PurePath) and type(pathlib.Path("x")) is pathlib.PosixPath
    assert pathlib.Path("x") == PurePath("x")
    try:
        PurePath(1)
    except TypeError:
        pass
    else:
        raise AssertionError("TypeError expected")


def test_concrete_paths_read_write_glob_and_walk():
    with tempfile.TemporaryDirectory() as root_text:
        root = pathlib.Path(root_text) / "tree"
        root.mkdir()
        (root / "a" / "b").mkdir(parents=True)
        (root / "a" / "b").mkdir(parents=True, exist_ok=True)
        for failing, exception in (
            (lambda: (root / "a").mkdir(), FileExistsError),
            (lambda: (root / "zz" / "y").mkdir(), FileNotFoundError),
        ):
            try:
                failing()
            except exception:
                pass
            else:
                raise AssertionError(f"{exception.__name__} expected")
        assert (root / "a" / "f1.py").write_text("print(1)\n") == 9
        assert (root / "a" / "f2.txt").write_text("hé", encoding="utf-8") == 2
        assert (root / "a" / "b" / "f3.py").write_bytes(b"\x00\x01") == 2
        (root / ".hid.py").write_text("")
        (root / "a" / "f1.py").touch()
        (root / "a" / "new").touch(exist_ok=False)
        try:
            (root / "a" / "new").touch(exist_ok=False)
        except FileExistsError:
            pass
        else:
            raise AssertionError("FileExistsError expected")
        assert (root / "a" / "f2.txt").read_text(encoding="utf-8") == "hé"
        assert (root / "a" / "f2.txt").read_bytes() == b"h\xc3\xa9"
        assert (root / "a" / "f1.py").stat().st_size == 9
        assert (root / "a").is_dir() and not (root / "a").is_file() and (root / "a" / "f1.py").is_file()
        assert not (root / "nope").exists() and not (root / "nope").is_dir()

        def names(paths):
            return sorted(str(path.relative_to(root)) for path in paths)

        assert names(root.iterdir()) == [".hid.py", "a"]
        assert names(root.glob("*.py")) == [".hid.py"]
        assert names(root.glob("**/*.py")) == [".hid.py", "a/b/f3.py", "a/f1.py"]
        assert names(root.rglob("*.py")) == [".hid.py", "a/b/f3.py", "a/f1.py"]
        assert names(root.glob("a/*")) == ["a/b", "a/f1.py", "a/f2.txt", "a/new"]
        assert names(root.glob("*/")) == ["a"] and names(root.glob("**/b")) == ["a/b"]
        assert names(root.glob("a/[fn]*")) == ["a/f1.py", "a/f2.txt", "a/new"]
        walked = [(str(top.relative_to(root)), sorted(dirs), sorted(files)) for top, dirs, files in root.walk()]
        assert walked == [(".", ["a"], [".hid.py"]), ("a", ["b"], ["f1.py", "f2.txt", "new"]), ("a/b", [], ["f3.py"])]
        assert [str(top.relative_to(root)) for top, _, _ in root.walk(top_down=False)] == ["a/b", "a", "."]
        with (root / "a" / "f1.py").open("ab") as handle:
            handle.write(b"#\n")
        assert (root / "a" / "f1.py").read_text() == "print(1)\n#\n"
        assert (root / "a" / "f1.py").samefile(root / "a" / "f1.py")
        moved = (root / "a" / "f1.py").rename(root / "a" / "g1.py")
        assert moved.exists() and not (root / "a" / "f1.py").exists()
        replaced = moved.replace(root / "g1.py")
        assert replaced == root / "g1.py" and replaced.exists()
        (root / "lnk").symlink_to("a")
        assert (root / "lnk").is_symlink() and (root / "lnk").is_dir()
        assert str((root / "lnk").readlink()) == "a" and (root / "lnk").resolve().name == "a"
        assert (root / "lnk" / "f2.txt").exists()
        (root / "lnk").unlink()
        (root / "g1.py").unlink()
        (root / "g1.py").unlink(missing_ok=True)
        try:
            (root / "g1.py").unlink()
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("FileNotFoundError expected")
        try:
            (root / "a").rmdir()
        except OSError:
            pass
        else:
            raise AssertionError("OSError expected")
        (root / "a" / "b").chmod(0o700)
        assert (root / "a" / "b").stat().st_mode & 0o777 == 0o700
        assert pathlib.Path.cwd() == pathlib.Path(os.getcwd())
        assert pathlib.Path("rel").absolute().is_absolute()
        assert pathlib.Path("x").resolve() == pathlib.Path.cwd() / "x"
        assert pathlib.Path("/").is_mount() and not root.is_mount()
        assert str(pathlib.Path.from_uri("file:///a/b%20c")) == "/a/b c"
        copy_root = root.parent / "copy"
        assert shutil.copytree(root, copy_root) == copy_root
        assert [str(path.relative_to(copy_root)) for path in copy_root.rglob("*.txt")] == ["a/f2.txt"]
        shutil.rmtree(root)
        assert not root.exists()


def test_shutil_copies_moves_and_archives():
    with tempfile.TemporaryDirectory() as root_text:
        root = pathlib.Path(root_text)
        source = root / "src"
        source.mkdir()
        (source / "a.txt").write_text("alpha")
        (source / "skip.log").write_text("log")
        (source / "sub").mkdir()
        (source / "sub" / "b.txt").write_text("beta")
        copied = shutil.copy(source / "a.txt", root / "a-copy.txt")
        assert pathlib.Path(copied).read_text() == "alpha"
        assert str(shutil.copy2(source / "a.txt", root / "second.txt")).endswith("second.txt")
        tree = shutil.copytree(source, root / "dst", ignore=shutil.ignore_patterns("*.log"))
        assert sorted(path.name for path in pathlib.Path(tree).iterdir()) == ["a.txt", "sub"]
        moved = shutil.move(str(root / "dst"), str(root / "moved"))
        assert pathlib.Path(moved).is_dir() and not (root / "dst").exists()
        archive = shutil.make_archive(str(root / "bundle"), "zip", root_dir=str(source))
        assert archive.endswith("bundle.zip")
        shutil.unpack_archive(archive, str(root / "unpacked"))
        assert (root / "unpacked" / "sub" / "b.txt").read_text() == "beta"
        assert "zip" in [name for name, _ in shutil.get_archive_formats()]
        usage = shutil.disk_usage(str(root))
        assert usage.total >= usage.used >= 0
        assert shutil.which("definitely-not-a-command-name") is None
        with open(source / "a.txt", "rb") as src, io.BytesIO() as dst:
            shutil.copyfileobj(src, dst)
            assert dst.getvalue() == b"alpha"
        try:
            shutil.copyfile(source / "a.txt", source / "a.txt")
        except shutil.SameFileError as error:
            assert isinstance(error, shutil.Error) and isinstance(error, OSError)
        else:
            raise AssertionError("SameFileError expected")
        try:
            shutil.rmtree(root / "missing")
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("FileNotFoundError expected")
        shutil.rmtree(root / "missing", ignore_errors=True)


def test_contextlib_stacks_decorators_and_exception_groups():
    events = []

    class Resource:
        def close(self):
            events.append("closed")

    with contextlib.closing(Resource()):
        pass
    with contextlib.nullcontext(5) as value:
        assert value == 5
    with contextlib.ExitStack() as stack:
        stack.callback(events.append, "cb1")
        stack.callback(events.append, "cb2")
        stack.push(lambda *exc: events.append(("push", exc[0])))
        detached = stack.pop_all()
    assert events == ["closed"]
    detached.close()
    assert events == ["closed", ("push", None), "cb2", "cb1"]

    def swallow(*exc):
        return True

    with contextlib.ExitStack() as stack:
        stack.push(swallow)
        raise KeyError("swallowed")

    class Managed(contextlib.AbstractContextManager):
        def __exit__(self, *exc):
            return None

    with Managed() as managed:
        assert isinstance(managed, Managed)

    class Duck:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    assert issubclass(Duck, contextlib.AbstractContextManager)
    assert not issubclass(int, contextlib.AbstractContextManager)

    class Decorated(contextlib.ContextDecorator):
        def __enter__(self):
            events.append("in")
            return self

        def __exit__(self, *exc):
            events.append("out")
            return False

    @Decorated()
    def body():
        events.append("body")

    body()
    assert events[-3:] == ["in", "body", "out"]

    with contextlib.suppress(ValueError):
        raise ExceptionGroup("g", [ValueError("a")])
    try:
        with contextlib.suppress(ValueError):
            raise ExceptionGroup("g", [ValueError("a"), KeyError("b")])
    except ExceptionGroup as group:
        assert [type(exc) for exc in group.exceptions] == [KeyError]
    else:
        raise AssertionError("ExceptionGroup expected")

    with tempfile.TemporaryDirectory() as root:
        with contextlib.chdir(root):
            assert os.path.samefile(os.getcwd(), root)
        assert not os.path.samefile(os.getcwd(), root)


def test_exception_groups_split_and_validate():
    group = ExceptionGroup("g", [ValueError("a"), KeyError("b"), ExceptionGroup("inner", [TypeError("t")])])
    assert group.message == "g" and len(group.exceptions) == 3 and isinstance(group, Exception)
    assert isinstance(group, BaseExceptionGroup) and str(group) == "g (3 sub-exceptions)"
    matched, rest = group.split(ValueError)
    assert [type(exc) for exc in matched.exceptions] == [ValueError]
    assert [type(exc) for exc in rest.exceptions] == [KeyError, ExceptionGroup]
    matched, rest = group.split((TypeError, KeyError))
    assert [type(exc) for exc in matched.exceptions] == [KeyError, ExceptionGroup]
    assert [type(exc) for exc in rest.exceptions] == [ValueError]
    assert group.subgroup(OSError) is None
    assert group.subgroup(lambda exc: isinstance(exc, KeyError)).exceptions[0].args == ("b",)
    for failing, exception in (
        (lambda: ExceptionGroup("x", []), ValueError),
        (lambda: ExceptionGroup("x", [KeyboardInterrupt()]), TypeError),
        (lambda: ExceptionGroup("x", [1]), ValueError),
    ):
        try:
            failing()
        except exception:
            pass
        else:
            raise AssertionError(f"{exception.__name__} expected")
    try:
        raise ExceptionGroup("raised", [ValueError(1)])
    except ExceptionGroup as caught:
        assert caught.exceptions[0].args == (1,)


def test_inspect_predicates_members_and_binding():
    def plain(a, b=2, *args, c, d=4, **kw):
        """Plain doc."""

    class Thing:
        """Class doc.

            indented more
        end"""

        x = 1

        def method(self, y):
            pass

        @classmethod
        def build(cls):
            pass

        @staticmethod
        def helper():
            pass

        @property
        def prop(self):
            return 1

    async def coroutine_function():
        pass

    def generator_function():
        yield 1

    assert inspect.isfunction(plain) and inspect.isfunction(Thing.method)
    assert inspect.ismethod(Thing().method) and not inspect.ismethod(plain) and not inspect.isfunction(Thing().method)
    assert inspect.isbuiltin(len) and inspect.isroutine(len) and inspect.isroutine(Thing().method)
    assert inspect.isclass(Thing) and not inspect.isclass(Thing()) and inspect.ismodule(sys)
    assert inspect.isgeneratorfunction(generator_function) and not inspect.isgeneratorfunction(plain)
    assert inspect.iscoroutinefunction(coroutine_function) and not inspect.iscoroutinefunction(plain)
    generator = generator_function()
    coroutine = coroutine_function()
    assert inspect.isgenerator(generator) and not inspect.isgenerator(coroutine)
    assert inspect.iscoroutine(coroutine) and inspect.isawaitable(coroutine) and not inspect.isawaitable(1)
    coroutine.close()
    assert inspect.isdatadescriptor(Thing.__dict__["prop"]) and inspect.ismethoddescriptor(Thing.__dict__["helper"])
    assert inspect.getdoc(Thing).splitlines() == ["Class doc.", "", "    indented more", "end"]
    assert inspect.getdoc(plain) == "Plain doc."
    assert inspect.cleandoc("  a\n    b\n\n     c\n  ") == "a\nb\n\n c"
    assert [name for name, _ in inspect.getmembers(Thing, inspect.isfunction)] == ["helper", "method"]
    assert [name for name, _ in inspect.getmembers(Thing(), inspect.ismethod)] == ["build", "method"]
    assert ("x", 1) in inspect.getmembers(Thing, lambda value: isinstance(value, int))
    assert inspect.getmro(Thing) == (Thing, object)
    assert type(inspect.getattr_static(Thing(), "prop")) is property
    assert type(inspect.getattr_static(Thing, "build")) is classmethod
    assert inspect.getattr_static(Thing(), "missing", 7) == 7
    assert inspect.getmodule(inspect.signature) is inspect
    assert inspect.currentframe() is None or inspect.currentframe().f_code is not None
    try:
        inspect.getattr_static(Thing(), "missing")
    except AttributeError:
        pass
    else:
        raise AssertionError("AttributeError expected")

    signature = inspect.signature(plain)
    bound = signature.bind(1, c=3)
    assert bound.args == (1,) and bound.kwargs == {"c": 3}
    bound.apply_defaults()
    assert bound.arguments == {"a": 1, "b": 2, "args": (), "c": 3, "d": 4, "kw": {}}
    assert signature.bind(1, 2, 3, 4, c=5, e=6).arguments["kw"] == {"e": 6}
    assert signature.bind_partial(1).arguments == {"a": 1}
    assert inspect.getcallargs(plain, 1, c=9)["d"] == 4
    for failing in (lambda: signature.bind(), lambda: signature.bind(1), lambda: signature.bind(1, c=2, a=3)):
        try:
            failing()
        except TypeError:
            pass
        else:
            raise AssertionError("TypeError expected")
    parameter = inspect.Parameter("z", inspect.Parameter.KEYWORD_ONLY, default=None)
    assert (
        str(parameter) == "z=None"
        and parameter.replace(default=inspect.Parameter.empty).default is inspect.Parameter.empty
    )
    assert str(inspect.Signature([parameter])) == "(*, z=None)"
    assert str(signature.replace(parameters=list(signature.parameters.values())[:2])) == "(a, b=2)"
    spec = inspect.getfullargspec(plain)
    assert spec.args == ["a", "b"] and spec.varargs == "args" and spec.varkw == "kw"
    assert spec.defaults == (2,) and spec.kwonlyargs == ["c", "d"] and spec.kwonlydefaults == {"d": 4}
    try:
        inspect.getsource(plain)
    except OSError:
        pass
    try:
        inspect.signature(1)
    except TypeError:
        pass
    else:
        raise AssertionError("TypeError expected")


def test_object_and_builtin_surface_used_by_the_stdlib():
    class Thing:
        """Documented."""

        @classmethod
        def build(cls):
            return cls()

        @staticmethod
        def helper():
            return 1

    assert Thing.__doc__ == "Documented." and Thing.__module__ == __name__
    assert isinstance(Thing.__dict__["build"], classmethod) and isinstance(Thing.__dict__["helper"], staticmethod)
    assert type(Thing.__dict__["build"]).__name__ == "classmethod" and Thing.__dict__["build"].__func__ is not None
    assert type(property()) is property and property(lambda self: 1).fget is not None
    assert list.__hash__ is None and dict.__hash__ is None and set.__hash__ is None
    assert int("101", base=2) == 5 and bytes.fromhex("61 62") == b"ab" and bytearray.fromhex("63") == bytearray(b"c")
    assert "a\nb\r\nc".splitlines(keepends=True) == ["a\n", "b\r\n", "c"]
    assert b"a\nb\r\nc".splitlines() == [b"a", b"b", b"c"]
    assert type.__qualname__ == "type" and Thing.__qualname__.endswith("Thing")

    class Blob:
        def __bytes__(self):
            return b"blob"

    assert bytes(Blob()) == b"blob"

    def pairs():
        yield 1, [2], [3]

    assert list(pairs()) == [(1, [2], [3])]

    inspect.injected_name = "value"
    assert inspect.injected_name == "value" and "injected_name" in dir(inspect)
    del inspect.injected_name
    assert not hasattr(inspect, "injected_name")
    assert sys.modules["__main__"].__name__ == "__main__"
    try:
        os.environ["SHELLSIM_SURFACE_MISSING_KEY"]
    except KeyError:
        pass
    else:
        raise AssertionError("KeyError expected")
    assert os.uname().sysname and os.getpid() > 0 and os.cpu_count() >= 1
    assert os.path.join("a", "b") == "a/b" and os.sep == "/"
