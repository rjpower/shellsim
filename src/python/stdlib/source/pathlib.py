"""Object-oriented filesystem paths over shellsim's virtual filesystem.

``PurePath`` and ``PurePosixPath`` do string work only. ``Path`` and ``PosixPath`` perform I/O
through the ``os`` module, so every access stays inside the simulated filesystem. Windows paths
are not modelled: ``PureWindowsPath`` and ``WindowsPath`` raise ``NotImplementedError``.
"""

import fnmatch
import os
import posixpath
import stat as _stat

__all__ = [
    "UnsupportedOperation", "PurePath", "PurePosixPath", "PureWindowsPath", "Path", "PosixPath",
    "WindowsPath",
]


class UnsupportedOperation(NotImplementedError):
    """Raised for path operations the simulated filesystem does not provide."""


def _parse(segments):
    """Split joined segments into (root, parts); an absolute segment discards what precedes it."""
    text = ""
    for segment in segments:
        if segment.startswith("/"):
            text = segment
        elif text and not text.endswith("/"):
            text = text + "/" + segment
        else:
            text = text + segment
    root = "/" if text.startswith("/") else ""
    parts = [part for part in text.split("/") if part and part != "."]
    return root, parts


def _segment_text(segment):
    if isinstance(segment, PurePath):
        return segment._text
    if isinstance(segment, str):
        return segment
    if isinstance(segment, bytes):
        raise TypeError("argument should be a str or an os.PathLike object where __fspath__ returns a str, not 'bytes'")
    if hasattr(segment, "__fspath__"):
        result = segment.__fspath__()
        if isinstance(result, str):
            return result
        raise TypeError("expected __fspath__() to return str, not " + type(result).__name__)
    raise TypeError(
        "argument should be a str or an os.PathLike object where __fspath__ returns a str, not "
        + repr(type(segment).__name__)
    )


class PurePath:
    """A path without any I/O: joining, splitting, suffixes, matching and ordering."""

    def __new__(cls, *args, **kwargs):
        if cls is PurePath:
            cls = PurePosixPath
        return object.__new__(cls)

    def __init__(self, *args):
        segments = [_segment_text(segment) for segment in args]
        self._root, self._parts = _parse(segments)
        self._text = self._root + "/".join(self._parts) if (self._root or self._parts) else "."

    def __reduce__(self):
        return (type(self), tuple(self.parts))

    def with_segments(self, *segments):
        """Build a new path of the same class; subclasses override to pass state along."""
        return type(self)(*segments)

    def __str__(self):
        return self._text

    def __fspath__(self):
        return self._text

    def __bytes__(self):
        return self._text.encode("utf-8")

    def __repr__(self):
        return f"{type(self).__name__}({self._text!r})"

    def as_posix(self):
        return self._text

    def as_uri(self):
        if not self.is_absolute():
            raise ValueError("relative path can't be expressed as a file URI")
        from urllib.parse import quote
        return "file://" + quote(self._text)

    def __eq__(self, other):
        if not isinstance(other, PurePath):
            return NotImplemented
        return self._text == other._text

    def __hash__(self):
        return hash(self._text)

    def _compare_key(self):
        return (self._root, tuple(self._parts))

    def __lt__(self, other):
        if not isinstance(other, PurePath):
            return NotImplemented
        return self._compare_key() < other._compare_key()

    def __le__(self, other):
        if not isinstance(other, PurePath):
            return NotImplemented
        return self._compare_key() <= other._compare_key()

    def __gt__(self, other):
        if not isinstance(other, PurePath):
            return NotImplemented
        return self._compare_key() > other._compare_key()

    def __ge__(self, other):
        if not isinstance(other, PurePath):
            return NotImplemented
        return self._compare_key() >= other._compare_key()

    @property
    def drive(self):
        return ""

    @property
    def root(self):
        return self._root

    @property
    def anchor(self):
        return self._root

    @property
    def parts(self):
        if self._root:
            return (self._root,) + tuple(self._parts)
        return tuple(self._parts)

    @property
    def name(self):
        return self._parts[-1] if self._parts else ""

    @property
    def suffix(self):
        """The final extension, including a lone trailing dot, as in Python 3.14."""
        name = self.name
        index = name.rfind(".")
        if index <= 0:
            return ""
        return name[index:]

    @property
    def suffixes(self):
        name = self.name.lstrip(".")
        pieces = name.split(".")
        return ["." + piece for piece in pieces[1:]]

    @property
    def stem(self):
        name = self.name
        suffix = self.suffix
        return name[: -len(suffix)] if suffix else name

    @property
    def parent(self):
        if not self._parts:
            return self
        return self.with_segments(self._root + "/".join(self._parts[:-1]))

    @property
    def parents(self):
        return _Parents(self)

    def with_name(self, name):
        if not self.name:
            raise ValueError(f"{self!r} has an empty name")
        if not name or "/" in name or name == ".":
            raise ValueError(f"Invalid name {name!r}")
        return self.with_segments(self._root + "/".join(self._parts[:-1] + [name]))

    def with_stem(self, stem):
        suffix = self.suffix
        if not suffix:
            return self.with_name(stem)
        if not stem:
            raise ValueError(f"{self!r} has a non-empty suffix")
        return self.with_name(stem + suffix)

    def with_suffix(self, suffix):
        if suffix and not suffix.startswith("."):
            raise ValueError(f"Invalid suffix {suffix!r}")
        if "/" in suffix:
            raise ValueError(f"Invalid suffix {suffix!r}")
        name = self.name
        if not name:
            raise ValueError(f"{self!r} has an empty name")
        return self.with_name(self.stem + suffix)

    def joinpath(self, *segments):
        return self.with_segments(self._text, *segments)

    def __truediv__(self, other):
        try:
            return self.joinpath(other)
        except TypeError:
            return NotImplemented

    def __rtruediv__(self, other):
        try:
            return self.with_segments(other, self._text)
        except TypeError:
            return NotImplemented

    def is_absolute(self):
        return bool(self._root)

    def is_reserved(self):
        return False

    def is_relative_to(self, other, *_deprecated):
        other = self.with_segments(other)
        return other == self or other in self.parents

    def relative_to(self, other, *_deprecated, walk_up=False):
        other = self.with_segments(other)
        for step, candidate in enumerate([other] + list(other.parents)):
            if candidate == self or candidate in self.parents:
                break
            if not walk_up:
                raise ValueError(f"{self._text!r} is not in the subpath of {other._text!r}")
            if candidate.name == "..":
                raise ValueError(f"'..' segment in {other._text!r} cannot be walked")
        else:
            raise ValueError(f"{self._text!r} and {other._text!r} have different anchors")
        remainder = self._parts[len(candidate._parts):]
        return self.with_segments(*([".."] * step + remainder))

    def match(self, path_pattern, *, case_sensitive=None):
        """Right-anchored glob match: a relative pattern matches a tail of the path."""
        pattern = self.with_segments(path_pattern)
        if not pattern.parts:
            raise ValueError("empty pattern")
        pattern_parts = list(pattern.parts)
        own = list(self.parts)
        if pattern.is_absolute():
            if len(own) != len(pattern_parts):
                return False
        elif len(pattern_parts) > len(own):
            return False
        own = own[len(own) - len(pattern_parts):]
        return all(fnmatch.fnmatchcase(part, pat) for part, pat in zip(own, pattern_parts))

    def full_match(self, pattern, *, case_sensitive=None):
        """Whole-path glob match where ``**`` spans any number of segments."""
        pattern = self.with_segments(pattern)
        if pattern.is_absolute() != self.is_absolute():
            return False
        return _match_segments(list(self._parts), list(pattern._parts))


def _match_segments(parts, pattern):
    if not pattern:
        return not parts
    head = pattern[0]
    if head == "**":
        return any(_match_segments(parts[index:], pattern[1:]) for index in range(len(parts) + 1))
    if not parts:
        return False
    return fnmatch.fnmatchcase(parts[0], head) and _match_segments(parts[1:], pattern[1:])


class _Parents:
    """Sequence of a path's logical ancestors, nearest first."""

    def __init__(self, path):
        self._path = path
        self._count = len(path._parts)

    def __len__(self):
        return self._count

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(self[i] for i in range(*index.indices(self._count)))
        if index < 0:
            index += self._count
        if not 0 <= index < self._count:
            raise IndexError(index)
        path = self._path
        return path.with_segments(path._root + "/".join(path._parts[: self._count - index - 1]))

    def __iter__(self):
        for index in range(self._count):
            yield self[index]

    def __contains__(self, item):
        return any(parent == item for parent in self)

    def __repr__(self):
        return f"<{type(self._path).__name__}.parents>"


class PurePosixPath(PurePath):
    """A pure path with POSIX separators."""


class PureWindowsPath(PurePath):
    """Windows paths are not modelled by the simulated filesystem."""

    def __new__(cls, *args, **kwargs):
        raise NotImplementedError("shellsim does not model Windows paths")


class Path(PurePath):
    """A concrete path whose methods read and change the simulated filesystem."""

    def __new__(cls, *args, **kwargs):
        if cls is Path:
            cls = PosixPath
        return object.__new__(cls)

    @classmethod
    def cwd(cls):
        return cls(os.getcwd())

    @classmethod
    def home(cls):
        home = os.environ.get("HOME")
        if not home:
            raise RuntimeError("Could not determine home directory.")
        return cls(home)

    @classmethod
    def from_uri(cls, uri):
        if not uri.startswith("file:"):
            raise ValueError(f"URI does not start with 'file:': {uri!r}")
        from urllib.parse import unquote
        text = uri[5:]
        if text.startswith("///"):
            text = text[2:]
        elif text.startswith("//"):
            raise ValueError(f"URI is not absolute: {uri!r}")
        path = cls(unquote(text))
        if not path.is_absolute():
            raise ValueError(f"URI is not absolute: {uri!r}")
        return path

    # ---- querying ----

    def stat(self, *, follow_symlinks=True):
        return os.stat(self._text, follow_symlinks=follow_symlinks)

    def lstat(self):
        return os.lstat(self._text)

    def _mode(self, follow_symlinks=True):
        try:
            return self.stat(follow_symlinks=follow_symlinks).st_mode
        except (OSError, ValueError):
            return None

    def exists(self, *, follow_symlinks=True):
        return self._mode(follow_symlinks) is not None

    def is_dir(self, *, follow_symlinks=True):
        mode = self._mode(follow_symlinks)
        return mode is not None and _stat.S_ISDIR(mode)

    def is_file(self, *, follow_symlinks=True):
        mode = self._mode(follow_symlinks)
        return mode is not None and _stat.S_ISREG(mode)

    def is_symlink(self):
        mode = self._mode(False)
        return mode is not None and _stat.S_ISLNK(mode)

    def is_junction(self):
        return False

    def is_mount(self):
        return self.is_absolute() and not self._parts

    def is_block_device(self):
        mode = self._mode()
        return mode is not None and _stat.S_ISBLK(mode)

    def is_char_device(self):
        mode = self._mode()
        return mode is not None and _stat.S_ISCHR(mode)

    def is_fifo(self):
        mode = self._mode()
        return mode is not None and _stat.S_ISFIFO(mode)

    def is_socket(self):
        mode = self._mode()
        return mode is not None and _stat.S_ISSOCK(mode)

    def samefile(self, other_path):
        other = other_path if isinstance(other_path, Path) else self.with_segments(other_path)
        return os.path.samefile(self._text, other._text)

    def owner(self, *, follow_symlinks=True):
        return os.getlogin()

    def group(self, *, follow_symlinks=True):
        return os.getlogin()

    # ---- directory listing ----

    def iterdir(self):
        for name in os.listdir(self._text):
            yield self._child(name)

    def _child(self, name):
        return self.with_segments(self._text, name)

    def _scandir_entries(self):
        """``(child, is_dir)`` pairs for a directory, or nothing if it cannot be listed."""
        try:
            names = os.listdir(self._text)
        except OSError:
            return
        for name in names:
            child = self._child(name)
            yield child, child.is_dir()

    def glob(self, pattern, *, case_sensitive=None, recurse_symlinks=False):
        """Children matching ``pattern``; ``**`` matches any number of directories.

        ```python
        sorted(Path("src").glob("**/*.py"))
        ```
        """
        pattern_text = _segment_text(pattern)
        if not pattern_text:
            raise ValueError("Unacceptable pattern: ''")
        if pattern_text.startswith("/"):
            raise NotImplementedError("Non-relative patterns are unsupported")
        segments = [part for part in pattern_text.split("/") if part and part != "."]
        want_dir = pattern_text.endswith("/")
        for match in _glob_segments(self, segments):
            if want_dir and not match.is_dir():
                continue
            yield match

    def rglob(self, pattern, *, case_sensitive=None, recurse_symlinks=False):
        pattern_text = _segment_text(pattern)
        if pattern_text.startswith("/"):
            raise NotImplementedError("Non-relative patterns are unsupported")
        return self.glob("**/" + pattern_text if pattern_text else "**", case_sensitive=case_sensitive)

    def walk(self, top_down=True, on_error=None, follow_symlinks=False):
        """``(dirpath, dirnames, filenames)`` triples like ``os.walk`` with Path dirpaths."""
        stack = [self]
        while stack:
            top = stack.pop()
            if isinstance(top, tuple):
                yield top
                continue
            try:
                names = os.listdir(top._text)
            except OSError as error:
                if on_error is not None:
                    on_error(error)
                continue
            dirnames = []
            filenames = []
            for name in names:
                child = top._child(name)
                if child.is_dir(follow_symlinks=follow_symlinks):
                    dirnames.append(name)
                else:
                    filenames.append(name)
            if top_down:
                yield top, dirnames, filenames
            else:
                stack.append((top, dirnames, filenames))
            for name in reversed(dirnames):
                stack.append(top._child(name))

    # ---- path resolution ----

    def absolute(self):
        if self.is_absolute():
            return self
        return self.with_segments(os.getcwd(), self._text)

    def resolve(self, strict=False):
        return self.with_segments(os.path.realpath(self._text, strict=strict))

    def readlink(self):
        return self.with_segments(os.readlink(self._text))

    def expanduser(self):
        if self._parts and self._parts[0].startswith("~") and not self._root:
            expanded = os.path.expanduser(self._parts[0])
            if expanded.startswith("~"):
                raise RuntimeError("Could not determine home directory.")
            return self.with_segments(expanded, *self._parts[1:])
        return self

    # ---- reading and writing ----

    def open(self, mode="r", buffering=-1, encoding=None, errors=None, newline=None):
        import io
        return io.open(self._text, mode, buffering, encoding, errors, newline)

    def read_bytes(self):
        with self.open("rb") as handle:
            return handle.read()

    def read_text(self, encoding=None, errors=None, newline=None):
        with self.open("r", encoding=encoding, errors=errors, newline=newline) as handle:
            return handle.read()

    def write_bytes(self, data):
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError("a bytes-like object is required, not %r" % type(data).__name__)
        with self.open("wb") as handle:
            return handle.write(bytes(data))

    def write_text(self, data, encoding=None, errors=None, newline=None):
        if not isinstance(data, str):
            raise TypeError("data must be str, not %s" % type(data).__name__)
        with self.open("w", encoding=encoding, errors=errors, newline=newline) as handle:
            return handle.write(data)

    # ---- creating, changing and removing ----

    def touch(self, mode=0o666, exist_ok=True):
        if self.exists():
            if not exist_ok:
                raise FileExistsError(17, "File exists", self._text)
            os.utime(self._text)
            return
        with self.open("wb"):
            pass
        os.chmod(self._text, mode & ~0o022)

    def mkdir(self, mode=0o777, parents=False, exist_ok=False):
        try:
            if parents:
                os.makedirs(self._text, mode, exist_ok=False)
            else:
                os.mkdir(self._text, mode)
        except FileExistsError:
            if not exist_ok or not self.is_dir():
                raise

    def chmod(self, mode, *, follow_symlinks=True):
        os.chmod(self._text, mode, follow_symlinks=follow_symlinks)

    def lchmod(self, mode):
        self.chmod(mode, follow_symlinks=False)

    def unlink(self, missing_ok=False):
        try:
            os.remove(self._text)
        except FileNotFoundError:
            if not missing_ok:
                raise

    def rmdir(self):
        os.rmdir(self._text)

    def rename(self, target):
        os.rename(self._text, target)
        return self.with_segments(target)

    def replace(self, target):
        os.replace(self._text, target)
        return self.with_segments(target)

    def symlink_to(self, target, target_is_directory=False):
        os.symlink(target, self._text, target_is_directory)

    def hardlink_to(self, target):
        os.link(target, self._text)


def _glob_segments(base, segments):
    """Paths under ``base`` matching the remaining pattern segments, depth first."""
    if not segments:
        yield base
        return
    head = segments[0]
    rest = segments[1:]
    if head == "**":
        # Match zero or more directory levels, then the rest of the pattern.
        seen = {base._text}
        pending = [base]
        while pending:
            directory = pending.pop()
            yield from _glob_segments(directory, rest)
            for child, is_dir in directory._scandir_entries():
                if is_dir and not child.is_symlink() and child._text not in seen:
                    seen.add(child._text)
                    pending.append(child)
        return
    if any(char in head for char in "*?["):
        for child, is_dir in base._scandir_entries():
            if fnmatch.fnmatchcase(child.name, head):
                if rest and not is_dir:
                    continue
                yield from _glob_segments(child, rest)
        return
    child = base._child(head)
    if rest:
        if child.is_dir():
            yield from _glob_segments(child, rest)
    elif child.exists(follow_symlinks=False):
        yield child


class PosixPath(Path, PurePosixPath):
    """A concrete POSIX path."""


class WindowsPath(Path, PureWindowsPath):
    """Windows paths are not modelled by the simulated filesystem."""

    def __new__(cls, *args, **kwargs):
        raise NotImplementedError("shellsim does not model Windows paths")
