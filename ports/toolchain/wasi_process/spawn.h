/* POSIX spawn declarations backed by shellsim's versioned virtual-process ABI.
 * Unsupported attributes fail before a child is created. */
#ifndef SHELLSIM_WASI_SPAWN_H
#define SHELLSIM_WASI_SPAWN_H

#include <sys/types.h>

#include "process_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

#define POSIX_SPAWN_RESETIDS 0x01
#define POSIX_SPAWN_SETPGROUP 0x02
#define POSIX_SPAWN_SETSIGDEF 0x04
#define POSIX_SPAWN_SETSIGMASK 0x08

typedef struct {
    unsigned short flags;
    uint32_t signal_defaults;
} posix_spawnattr_t;

typedef struct {
    uint32_t count;
    size_t owned_path_bytes;
    char *owned_paths[SHELLSIM_PROCESS_MAX_ACTIONS];
    struct shellsim_process_action actions[SHELLSIM_PROCESS_MAX_ACTIONS];
} posix_spawn_file_actions_t;

int posix_spawnattr_init(posix_spawnattr_t *attributes);
int posix_spawnattr_destroy(posix_spawnattr_t *attributes);
int posix_spawnattr_setflags(posix_spawnattr_t *attributes, short flags);
int posix_spawnattr_setpgroup(posix_spawnattr_t *attributes, pid_t group);
int posix_spawnattr_setsigdefault(posix_spawnattr_t *attributes, const void *signals);
int posix_spawnattr_setsigdefault_mask_np(posix_spawnattr_t *attributes, uint32_t signals);
int posix_spawnattr_setsigmask(posix_spawnattr_t *attributes, const void *signals);

int posix_spawn_file_actions_init(posix_spawn_file_actions_t *actions);
int posix_spawn_file_actions_destroy(posix_spawn_file_actions_t *actions);
int posix_spawn_file_actions_addclose(posix_spawn_file_actions_t *actions, int fd);
int posix_spawn_file_actions_adddup2(posix_spawn_file_actions_t *actions, int source, int destination);
int posix_spawn_file_actions_addclosefrom_np(posix_spawn_file_actions_t *actions, int minimum);
int posix_spawn_file_actions_addopen(
    posix_spawn_file_actions_t *actions, int fd, const char *path, int flags, mode_t mode
);
int posix_spawn_file_actions_addchdir_np(posix_spawn_file_actions_t *actions, const char *path);

int posix_spawn(
    pid_t *pid, const char *path, const posix_spawn_file_actions_t *actions,
    const posix_spawnattr_t *attributes, char *const argv[], char *const environment[]
);
int posix_spawnp(
    pid_t *pid, const char *path, const posix_spawn_file_actions_t *actions,
    const posix_spawnattr_t *attributes, char *const argv[], char *const environment[]
);

#ifdef __cplusplus
}
#endif
#endif
