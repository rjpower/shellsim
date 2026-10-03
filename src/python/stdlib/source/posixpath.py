"""POSIX path operations over shellsim's virtual filesystem (``os.path``)."""

import _shellsim_vfs
from _os import getcwd, getenv

curdir = "."
pardir = ".."
extsep = "."
sep = "/"
pathsep = ":"
defpath = "/bin:/usr/bin"
altsep = None
devnull = "/dev/null"
supports_unicode_filenames = False


def _fspath(value):
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        raise TypeError("shellsim paths must be str, not bytes")
    if hasattr(value, "__fspath__"):
        result = value.__fspath__()
        if isinstance(result, str):
            return result
        raise TypeError("__fspath__() must return str")
    raise TypeError(
        "expected str, bytes or os.PathLike object, not " + type(value).__name__
    )


def normcase(path):
    return _fspath(path)


def isabs(path):
    return _fspath(path).startswith("/")


def join(first, *parts):
    result = _fspath(first)
    for part in parts:
        part = _fspath(part)
        if part.startswith("/"):
            result = part
        elif result == "" or result.endswith("/"):
            result += part
        else:
            result += "/" + part
    return result


def split(path):
    path = _fspath(path)
    index = path.rfind("/") + 1
    head, tail = path[:index], path[index:]
    if head and head != "/" * len(head):
        head = head.rstrip("/")
    return head, tail


def splitext(path):
    path = _fspath(path)
    slash = path.rfind("/")
    dot = path.rfind(".")
    if dot > slash:
        name_start = slash + 1
        while name_start < dot:
            if path[name_start] != ".":
                return path[:dot], path[dot:]
            name_start += 1
    return path, ""


def splitdrive(path):
    return "", _fspath(path)


def splitroot(path):
    path = _fspath(path)
    if not path.startswith("/"):
        return "", "", path
    if path.startswith("//") and not path.startswith("///"):
        return "", "//", path[2:]
    return "", "/", path[1:]


def basename(path):
    path = _fspath(path)
    return path[path.rfind("/") + 1 :]


def dirname(path):
    path = _fspath(path)
    index = path.rfind("/") + 1
    head = path[:index]
    if head and head != "/" * len(head):
        head = head.rstrip("/")
    return head


def islink(path):
    return _shellsim_vfs.is_symlink(_fspath(path))


def lexists(path):
    path = _fspath(path)
    return _shellsim_vfs.exists(path) or _shellsim_vfs.is_symlink(path)


def exists(path):
    return _shellsim_vfs.exists(_fspath(path))


def isdir(path):
    return _shellsim_vfs.is_dir(_fspath(path))


def isfile(path):
    return _shellsim_vfs.is_file(_fspath(path))


def ismount(path):
    return abspath(path) == "/"


def isjunction(path):
    _fspath(path)
    return False


def isdevdrive(path):
    _fspath(path)
    return False


def getsize(path):
    return _shellsim_vfs.stat(_fspath(path), True)[1]


def getmtime(path):
    return _shellsim_vfs.stat(_fspath(path), True)[2] / 1000.0


getatime = getmtime
getctime = getmtime


def expanduser(path):
    path = _fspath(path)
    if not path.startswith("~"):
        return path
    index = path.find("/")
    if index < 0:
        index = len(path)
    if index == 1:
        home = getenv("HOME")
        if home is None:
            home = "/root"
    else:
        return path
    home = home.rstrip("/")
    result = home + path[index:]
    return result if result else "/"


def expandvars(path):
    path = _fspath(path)
    if "$" not in path:
        return path
    result = []
    index = 0
    length = len(path)
    while index < length:
        char = path[index]
        if char != "$":
            result.append(char)
            index += 1
            continue
        if index + 1 < length and path[index + 1] == "{":
            end = path.find("}", index + 2)
            if end < 0:
                result.append(path[index:])
                break
            name = path[index + 2 : end]
            value = getenv(name)
            result.append(path[index : end + 1] if value is None else value)
            index = end + 1
            continue
        end = index + 1
        while end < length and (path[end].isalnum() or path[end] == "_"):
            end += 1
        name = path[index + 1 : end]
        if name == "":
            result.append("$")
        else:
            value = getenv(name)
            result.append(path[index:end] if value is None else value)
        index = end
    return "".join(result)


def normpath(path):
    path = _fspath(path)
    if path == "":
        return "."
    initial_slashes = 1 if path.startswith("/") else 0
    if initial_slashes and path.startswith("//") and not path.startswith("///"):
        initial_slashes = 2
    parts = []
    for part in path.split("/"):
        if part == "" or part == ".":
            continue
        if part != ".." or (not initial_slashes and not parts) or (parts and parts[-1] == ".."):
            parts.append(part)
        elif parts:
            parts.pop()
    result = "/".join(parts)
    if initial_slashes:
        result = "/" * initial_slashes + result
    return result or "."


def abspath(path):
    path = _fspath(path)
    if not path.startswith("/"):
        path = join(getcwd(), path)
    return normpath(path)


def realpath(path, *, strict=False):
    path = abspath(path)
    resolved = "/"
    for part in path.split("/"):
        if part == "":
            continue
        candidate = join(resolved, part)
        depth = 0
        while _shellsim_vfs.is_symlink(candidate):
            depth += 1
            if depth > 40:
                if strict:
                    raise OSError(40, "Too many levels of symbolic links", path)
                break
            target = _shellsim_vfs.readlink(candidate)
            candidate = normpath(target if target.startswith("/") else join(resolved, target))
        if strict and not _shellsim_vfs.exists(candidate):
            raise FileNotFoundError(2, "No such file or directory", path)
        resolved = candidate
    return resolved


def relpath(path, start=None):
    path = _fspath(path)
    if path == "":
        raise ValueError("no path specified")
    start = curdir if start is None else _fspath(start)
    start_parts = [part for part in abspath(start).split("/") if part]
    path_parts = [part for part in abspath(path).split("/") if part]
    common = 0
    while (
        common < len(start_parts)
        and common < len(path_parts)
        and start_parts[common] == path_parts[common]
    ):
        common += 1
    parts = [pardir] * (len(start_parts) - common) + path_parts[common:]
    return "/".join(parts) if parts else curdir


def commonprefix(paths):
    if not paths:
        return ""
    paths = [_fspath(path) if not isinstance(path, (list, tuple)) else path for path in paths]
    first = min(paths)
    last = max(paths)
    for index, char in enumerate(first):
        if char != last[index]:
            return first[:index]
    return first


def commonpath(paths):
    paths = [_fspath(path) for path in paths]
    if not paths:
        raise ValueError("commonpath() arg is an empty sequence")
    absolute = paths[0].startswith("/")
    for path in paths:
        if path.startswith("/") != absolute:
            raise ValueError("Can't mix absolute and relative paths")
    split_paths = [[part for part in path.split("/") if part and part != "."] for path in paths]
    shortest = min(split_paths)
    longest = max(split_paths)
    common = shortest
    for index, part in enumerate(shortest):
        if part != longest[index]:
            common = shortest[:index]
            break
    prefix = "/" if absolute else ""
    return prefix + "/".join(common)


def samestat(first, second):
    return tuple(first) == tuple(second)


def samefile(first, second):
    return realpath(first) == realpath(second)


def sameopenfile(first, second):
    return first == second
