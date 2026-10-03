"""File mode constants and predicates, as in CPython's ``stat`` module."""

ST_MODE = 0
ST_INO = 1
ST_DEV = 2
ST_NLINK = 3
ST_UID = 4
ST_GID = 5
ST_SIZE = 6
ST_ATIME = 7
ST_MTIME = 8
ST_CTIME = 9

S_IFMT_MASK = 0o170000
S_IFSOCK = 0o140000
S_IFLNK = 0o120000
S_IFREG = 0o100000
S_IFBLK = 0o060000
S_IFDIR = 0o040000
S_IFCHR = 0o020000
S_IFIFO = 0o010000
S_IFDOOR = 0
S_IFPORT = 0
S_IFWHT = 0

S_ISUID = 0o4000
S_ISGID = 0o2000
S_ENFMT = S_ISGID
S_ISVTX = 0o1000
S_IREAD = 0o0400
S_IWRITE = 0o0200
S_IEXEC = 0o0100
S_IRWXU = 0o0700
S_IRUSR = 0o0400
S_IWUSR = 0o0200
S_IXUSR = 0o0100
S_IRWXG = 0o0070
S_IRGRP = 0o0040
S_IWGRP = 0o0020
S_IXGRP = 0o0010
S_IRWXO = 0o0007
S_IROTH = 0o0004
S_IWOTH = 0o0002
S_IXOTH = 0o0001

UF_SETTABLE = 0x0000FFFF
UF_NODUMP = 0x00000001
UF_IMMUTABLE = 0x00000002
UF_APPEND = 0x00000004
UF_OPAQUE = 0x00000008
UF_NOUNLINK = 0x00000010
UF_COMPRESSED = 0x00000020
UF_TRACKED = 0x00000040
UF_DATAVAULT = 0x00000080
UF_HIDDEN = 0x00008000
SF_SETTABLE = 0xFFFF0000
SF_ARCHIVED = 0x00010000
SF_IMMUTABLE = 0x00020000
SF_APPEND = 0x00040000
SF_RESTRICTED = 0x00080000
SF_NOUNLINK = 0x00100000
SF_SNAPSHOT = 0x00200000
SF_FIRMLINK = 0x00800000
SF_DATALESS = 0x40000000

FILE_ATTRIBUTE_ARCHIVE = 32
FILE_ATTRIBUTE_COMPRESSED = 2048
FILE_ATTRIBUTE_DEVICE = 64
FILE_ATTRIBUTE_DIRECTORY = 16
FILE_ATTRIBUTE_ENCRYPTED = 16384
FILE_ATTRIBUTE_HIDDEN = 2
FILE_ATTRIBUTE_INTEGRITY_STREAM = 32768
FILE_ATTRIBUTE_NORMAL = 128
FILE_ATTRIBUTE_NOT_CONTENT_INDEXED = 8192
FILE_ATTRIBUTE_NO_SCRUB_DATA = 131072
FILE_ATTRIBUTE_OFFLINE = 4096
FILE_ATTRIBUTE_READONLY = 1
FILE_ATTRIBUTE_REPARSE_POINT = 1024
FILE_ATTRIBUTE_SPARSE_FILE = 512
FILE_ATTRIBUTE_SYSTEM = 4
FILE_ATTRIBUTE_TEMPORARY = 256
FILE_ATTRIBUTE_VIRTUAL = 65536


def S_IMODE(mode):
    return mode & 0o7777


def S_IFMT(mode):
    return mode & S_IFMT_MASK


def S_ISDIR(mode):
    return S_IFMT(mode) == S_IFDIR


def S_ISCHR(mode):
    return S_IFMT(mode) == S_IFCHR


def S_ISBLK(mode):
    return S_IFMT(mode) == S_IFBLK


def S_ISREG(mode):
    return S_IFMT(mode) == S_IFREG


def S_ISFIFO(mode):
    return S_IFMT(mode) == S_IFIFO


def S_ISLNK(mode):
    return S_IFMT(mode) == S_IFLNK


def S_ISSOCK(mode):
    return S_IFMT(mode) == S_IFSOCK


def S_ISDOOR(mode):
    return False


def S_ISPORT(mode):
    return False


def S_ISWHT(mode):
    return False


_filemode_table = (
    (
        (S_IFLNK, "l"),
        (S_IFSOCK, "s"),
        (S_IFREG, "-"),
        (S_IFBLK, "b"),
        (S_IFDIR, "d"),
        (S_IFCHR, "c"),
        (S_IFIFO, "p"),
    ),
    ((S_IRUSR, "r"),),
    ((S_IWUSR, "w"),),
    ((S_IXUSR | S_ISUID, "s"), (S_ISUID, "S"), (S_IXUSR, "x")),
    ((S_IRGRP, "r"),),
    ((S_IWGRP, "w"),),
    ((S_IXGRP | S_ISGID, "s"), (S_ISGID, "S"), (S_IXGRP, "x")),
    ((S_IROTH, "r"),),
    ((S_IWOTH, "w"),),
    ((S_IXOTH | S_ISVTX, "t"), (S_ISVTX, "T"), (S_IXOTH, "x")),
)


def filemode(mode):
    """Convert a file's mode to a string of the form '-rwxrwxrwx'."""
    perm = []
    for index, table in enumerate(_filemode_table):
        for bit, char in table:
            if index == 0:
                if S_IFMT(mode) == bit:
                    perm.append(char)
                    break
            elif mode & bit == bit:
                perm.append(char)
                break
        else:
            perm.append("?" if index == 0 else "-")
    return "".join(perm)
