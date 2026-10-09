/* Bounded account lookup over the simulated filesystem, with POSIX static-result
 * lifetime. The virtual kernel has no authenticated login-session identity. */
#include "pwd.h"
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define MAX_RECORDS 256
#define LINE_BYTES 4096
#define NAME_BYTES 256

static char line[LINE_BYTES];
static struct passwd result;

static int read_line(FILE *file) {
    size_t used = 0;
    for (;;) {
        int ch = fgetc(file);
        if (ch == EOF) {
            if (ferror(file)) return -(errno ? errno : EIO);
            line[used] = 0;
            return used != 0;
        }
        if (!ch) return -EINVAL;
        if (ch == '\n') { line[used] = 0; return 1; }
        if (used == sizeof line - 1) return -ERANGE;
        line[used++] = (char)ch;
    }
}

static int number(const char *text, unsigned *value) {
    if (!*text) return EINVAL;
    for (const char *p = text; *p; ++p)
        if (*p < '0' || *p > '9') return EINVAL;
    char *end;
    errno = 0;
    unsigned long parsed = strtoul(text, &end, 10);
    if (errno || *end || parsed > UINT32_MAX) return EINVAL;
    *value = (unsigned)parsed;
    return 0;
}

static struct passwd *lookup(const char *name, uid_t uid) {
    FILE *file = fopen("/etc/passwd", "r");
    if (!file) return NULL;
    int failure = 0;
    for (unsigned count = 0; count < MAX_RECORDS; ++count) {
        errno = 0;
        int state = read_line(file);
        if (state <= 0) {
            failure = -state;
            goto done;
        }
        size_t length = strlen(line);
        if (length && line[length - 1] == '\r') line[--length] = 0;
        if (!length || line[0] == '#') continue;
        char *fields[7];
        fields[0] = line;
        for (unsigned i = 1; i < 7; ++i) {
            char *separator = strchr(fields[i - 1], ':');
            if (!separator) { failure = EINVAL; goto done; }
            *separator = 0;
            fields[i] = separator + 1;
        }
        unsigned user, group;
        if (!*fields[0] || strlen(fields[0]) >= NAME_BYTES || strchr(fields[6], ':')
            || number(fields[2], &user) || number(fields[3], &group)) {
            failure = EINVAL;
            goto done;
        }
        if (name ? strcmp(name, fields[0]) != 0 : uid != user) continue;
        result = (struct passwd){fields[0], fields[1], user, group, fields[4], fields[5], fields[6]};
        fclose(file);
        errno = 0;
        return &result;
    }
    failure = fgetc(file) == EOF && !ferror(file) ? 0 : ERANGE;
done:
    fclose(file);
    errno = failure;
    return NULL;
}

struct passwd *getpwnam(const char *name) {
    if (!name) { errno = EINVAL; return NULL; }
    if (strnlen(name, NAME_BYTES) == NAME_BYTES) { errno = ERANGE; return NULL; }
    return lookup(name, 0);
}

struct passwd *getpwuid(uid_t uid) { return lookup(NULL, uid); }

char *getlogin(void) { errno = ENXIO; return NULL; }
