# Virtual file locks

Build CPython 3.13.7's upstream fcntl extension against the cached interpreter
headers. A small C bridge calls the virtual descriptor ABI for whole-file flock
and descriptor duplication/CLOEXEC controls. It replaces neither the interpreter
nor the SDK.

Whole-file shared and exclusive locks follow the virtual inode and open
description. Duplication and fork share ownership; the last close releases it.
Nonblocking contention raises BlockingIOError; blocking acquisition suspends the
guest on the cooperative scheduler. Lock upgrades release the prior mode before
acquisition, matching flock's non-atomic conversion.

Closing or reusing a descriptor from another thread during its pending lock
acquisition returns ENOTSUP; the request never locks the replacement file.
Record locks, ioctl and other fcntl commands return ENOTSUP. No operation uses
host descriptors or host locks. The recipe installs this module as a stdlib
overlay for the Locket, Partd and Dask dependency graph.
