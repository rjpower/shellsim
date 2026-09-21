"""Common file-tree operations confined to shellsim's modeled VFS."""

import _shellsim_vfs
import os


def _path(value):
    return os.fspath(value)


def _copy_file(source, destination):
    source = _path(source)
    destination = _path(destination)
    if _shellsim_vfs.is_dir(destination):
        destination = os.path.join(destination, os.path.basename(source))
    _shellsim_vfs.write_bytes(destination, _shellsim_vfs.read_bytes(source))
    return destination


def copyfile(source, destination, follow_symlinks=True):
    return _copy_file(source, destination)


def copy(source, destination, follow_symlinks=True):
    return _copy_file(source, destination)


def copy2(source, destination, follow_symlinks=True):
    return _copy_file(source, destination)


def copytree(source, destination, dirs_exist_ok=False):
    source = _path(source)
    destination = _path(destination)
    if _shellsim_vfs.is_symlink(source):
        raise ValueError("copytree does not follow symbolic links")
    _shellsim_vfs.mkdir(destination, True, dirs_exist_ok)
    for name in _shellsim_vfs.list_dir(source):
        child_source = os.path.join(source, name)
        child_destination = os.path.join(destination, name)
        if _shellsim_vfs.is_symlink(child_source):
            raise ValueError("copytree does not follow symbolic links")
        if _shellsim_vfs.is_dir(child_source):
            copytree(child_source, child_destination, dirs_exist_ok)
        else:
            _copy_file(child_source, child_destination)
    return destination


def rmtree(path, ignore_errors=False, onerror=None):
    try:
        _shellsim_vfs.remove_tree(_path(path))
    except OSError as error:
        if onerror is not None:
            onerror(rmtree, path, (OSError, error, None))
        elif not ignore_errors:
            raise
