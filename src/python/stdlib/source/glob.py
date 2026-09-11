"""Pattern matching over shellsim's modeled VFS."""

import _shellsim_vfs


def glob(pathname, recursive=False, include_hidden=False):
    return _shellsim_vfs.glob(pathname)
