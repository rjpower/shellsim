/* POSIX account records read exclusively from the guest's /etc/passwd. */
#ifndef SHELLSIM_WASI_PWD_H
#define SHELLSIM_WASI_PWD_H
#include <sys/types.h>
struct passwd {
    char *pw_name;
    char *pw_passwd;
    uid_t pw_uid;
    gid_t pw_gid;
    char *pw_gecos;
    char *pw_dir;
    char *pw_shell;
};
struct passwd *getpwnam(const char *name);
struct passwd *getpwuid(uid_t uid);
#endif
