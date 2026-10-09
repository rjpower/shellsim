# Bounded virtual account lookup

The libc facade implements getpwnam/getpwuid from guest `/etc/passwd`, with static
result storage valid until the next lookup. It accepts at most 256 records,
4096-byte lines and 255-byte account names. UID/GID fields are checked unsigned
32-bit values. Malformed records and embedded NUL bytes fail with EINVAL; size
limits fail with ERANGE; absent records return NULL with errno 0. File errors retain
an errno. No account or identity comes from the host.

There is no authenticated virtual login session, so getlogin returns NULL/ENXIO.
GNU make can use ordinary HOME fallback and real `~account` lookup. This static
storage facade is intended for the single-threaded native-tool profile; it does
not claim reentrant account APIs or thread-safe lookup storage.
