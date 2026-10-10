// Real libc/C++ runtime state and both imported data/function relocations are
// consumed before any application constructor or main command executes.
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <pthread.h>

extern "C" {
extern int main_data[2];
extern char **environ;
extern unsigned long __stack_chk_guard;
int main_callback(int);
int side_data[2] = {31, 43};
}

static int *volatile data_with_addend = &main_data[1];
static int (*volatile callback)(int) = main_callback;
static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
static int initialized;

__attribute__((constructor)) static void side_ctor(void) {
  if (!environ || !environ[0] || !__stack_chk_guard)
    std::abort();
  auto *p = static_cast<char *>(std::malloc(32));
  const char *v = std::getenv("BOOTSTRAP_TEST");
  if (!p || !v || std::strcmp(v, "present") || callback(7) != 12 ||
      *data_with_addend != 29)
    std::abort();
  if (pthread_mutex_lock(&mutex) || pthread_mutex_unlock(&mutex))
    std::abort();
  std::strcpy(p, "side runtime ctor passed");
  std::cout << p << std::endl;
  std::free(p);
  initialized = 1;
}

extern "C" int side_ready(void) { return initialized; }
