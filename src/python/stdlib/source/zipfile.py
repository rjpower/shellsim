"""ZIP32 files over the modeled VFS and shellsim's bounded archive command."""

import _shellsim_vfs
import io
import struct
import subprocess
import tempfile
import zlib


ZIP_STORED = 0
ZIP_DEFLATED = 8
BadZipFile = ValueError
LargeZipFile = ValueError


def _safe_name(name):
    name = str(name).replace("\\", "/")
    if name.startswith("/") or any(part in ("", ".", "..") for part in name.split("/")):
        raise ValueError("unsafe ZIP member name: " + repr(name))
    return name


def _mkdir_parent(path):
    parent = "/".join(path.split("/")[:-1])
    if parent:
        _shellsim_vfs.mkdir(parent, True, True)


class ZipInfo:
    def __init__(self, filename="NoName", date_time=(1980, 1, 1, 0, 0, 0)):
        self.filename = filename
        self.orig_filename = filename
        self.date_time = date_time
        self.compress_type = ZIP_STORED
        self.comment = b""
        self.extra = b""
        self.create_system = 3
        self.external_attr = 0
        self.CRC = 0
        self.compress_size = 0
        self.file_size = 0

    def is_dir(self):
        return self.filename.endswith("/")


class ZipFile:
    def __init__(self, file, mode="r", compression=ZIP_STORED, allowZip64=True,
                 compresslevel=None, strict_timestamps=True, metadata_encoding=None):
        if mode not in ("r", "w", "x", "a"):
            raise ValueError("ZipFile mode must be 'r', 'w', 'x', or 'a'")
        if mode != "r" and compression != ZIP_STORED:
            raise ValueError("shellsim ZipFile writing supports ZIP_STORED only")
        self.mode = mode
        self.compression = compression
        self._closed = False
        self._entries = []
        self._sink = file if hasattr(file, "write") else None
        self.filename = None if self._sink is not None else str(file)
        self._temporary_archive = None
        self._extracted = None
        if mode == "x" and self.filename is not None and _shellsim_vfs.exists(self.filename):
            raise FileExistsError(self.filename)
        if mode in ("r", "a"):
            self._archive = self._read_archive_source(file)
            self._names = self._list_names()
            if mode == "a":
                for name in self._names:
                    self._entries.append((name, self.read(name)))
        else:
            self._archive = None
            self._names = []

    def _read_archive_source(self, file):
        if hasattr(file, "read"):
            data = file.read()
            temporary = tempfile.NamedTemporaryFile(mode="wb", suffix=".zip", delete=False)
            temporary.write(data)
            temporary.close()
            self._temporary_archive = temporary.name
            return temporary.name
        return str(file)

    def _list_names(self):
        try:
            output = subprocess.check_output(["unzip", "-Z1", self._archive])
        except Exception as error:
            raise BadZipFile("invalid ZIP archive: " + str(error))
        return [line for line in output.decode("utf-8").splitlines() if line]

    def _extract_once(self):
        if self._extracted is None:
            self._extracted = tempfile.mkdtemp(prefix="zipfile-")
            try:
                subprocess.check_call(["unzip", "-qo", self._archive, "-d", self._extracted])
            except Exception as error:
                raise BadZipFile("cannot extract ZIP archive: " + str(error))

    def namelist(self):
        return list(self._names)

    def infolist(self):
        return [self.getinfo(name) for name in self._names]

    def getinfo(self, name):
        if name not in self._names:
            raise KeyError("There is no item named " + repr(name) + " in the archive")
        info = ZipInfo(name)
        if not name.endswith("/"):
            data = self.read(name)
            info.file_size = len(data)
            info.compress_size = len(data)
            info.CRC = zlib.crc32(data)
        return info

    def read(self, name, pwd=None):
        if pwd is not None:
            raise ValueError("encrypted ZIP members are not supported")
        name = name.filename if isinstance(name, ZipInfo) else str(name)
        if name not in self._names:
            raise KeyError("There is no item named " + repr(name) + " in the archive")
        if name.endswith("/"):
            return b""
        self._extract_once()
        return _shellsim_vfs.read_bytes(self._extracted + "/" + name)

    def open(self, name, mode="r", pwd=None, force_zip64=False):
        if mode != "r":
            raise ValueError("ZipFile.open writing is not supported; use writestr")
        return io.BytesIO(self.read(name, pwd))

    def extract(self, member, path=None, pwd=None):
        name = member.filename if isinstance(member, ZipInfo) else str(member)
        name = _safe_name(name.rstrip("/")) + ("/" if name.endswith("/") else "")
        base = "." if path is None else str(path)
        target = base.rstrip("/") + "/" + name
        if name.endswith("/"):
            _shellsim_vfs.mkdir(target, True, True)
        else:
            _mkdir_parent(target)
            _shellsim_vfs.write_bytes(target, self.read(name, pwd))
        return target

    def extractall(self, path=None, members=None, pwd=None):
        selected = self._names if members is None else members
        for member in selected:
            self.extract(member, path, pwd)

    def testzip(self):
        for name in self._names:
            try:
                self.read(name)
            except Exception:
                return name
        return None

    def write(self, filename, arcname=None, compress_type=None, compresslevel=None):
        self._check_writable(compress_type)
        filename = str(filename)
        if arcname is None:
            arcname = filename.rstrip("/").split("/")[-1]
        arcname = _safe_name(arcname)
        self._entries.append((arcname, _shellsim_vfs.read_bytes(filename)))

    def writestr(self, zinfo_or_arcname, data, compress_type=None, compresslevel=None):
        self._check_writable(compress_type)
        name = (zinfo_or_arcname.filename
                if isinstance(zinfo_or_arcname, ZipInfo) else str(zinfo_or_arcname))
        name = _safe_name(name.rstrip("/")) + ("/" if name.endswith("/") else "")
        if isinstance(data, str):
            data = data.encode("utf-8")
        self._entries.append((name, data))

    def _check_writable(self, compress_type):
        if self.mode == "r":
            raise ValueError("write() requires mode 'w', 'x', or 'a'")
        if compress_type is not None and compress_type != ZIP_STORED:
            raise ValueError("shellsim ZipFile writing supports ZIP_STORED only")

    def close(self):
        if self._closed:
            return
        if self.mode != "r":
            data = _encode_stored(self._entries)
            if self._sink is not None:
                self._sink.seek(0)
                self._sink.write(data)
            else:
                _mkdir_parent(self.filename)
                _shellsim_vfs.write_bytes(self.filename, data)
        if self._temporary_archive is not None and _shellsim_vfs.exists(self._temporary_archive):
            _shellsim_vfs.remove_file(self._temporary_archive)
        if self._extracted is not None and _shellsim_vfs.exists(self._extracted):
            _shellsim_vfs.remove_tree(self._extracted)
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        self.close()
        return False


def _encode_stored(entries):
    output = b""
    central = b""
    for name, data in entries:
        encoded = name.encode("utf-8")
        crc = zlib.crc32(data)
        offset = len(output)
        output += struct.pack("<IHHHHHIIIHH", 0x04034B50, 20, 0, ZIP_STORED,
                              0, 0, crc, len(data), len(data), len(encoded), 0)
        output += encoded + data
        directory = name.endswith("/")
        mode = (0o040755 if directory else 0o100644) << 16
        central += struct.pack("<IHHHHHHIIIHHHHHII", 0x02014B50, 0x031E, 20, 0,
                               ZIP_STORED, 0, 0, crc, len(data), len(data), len(encoded),
                               0, 0, 0, 0, mode, offset)
        central += encoded
    result = output + central
    result += struct.pack("<IHHHHIIH", 0x06054B50, 0, 0, len(entries), len(entries),
                          len(central), len(output), 0)
    return result


def is_zipfile(filename):
    try:
        with ZipFile(filename, "r") as archive:
            archive.namelist()
        return True
    except Exception:
        return False
