// This instrumented implementation-priority ctor proves process-once startup
// across _start and worker instantiation; user app constructors remain later.
#include <atomic>
#include <cstdlib>
#include <iostream>
#include <pthread.h>

extern "C" {
int side_ready(void);
extern int side_data[2];
void __wasm_call_runtime_ctors(void);
int main_data[2] = {17, 29};
int main_callback(int value) { return value + 5; }
}

static std::atomic<int> bootstrap_count{0};
static int *side_data_with_addend = &side_data[1];
static int (*ready)(void) = side_ready;
static bool app_initialized;

__attribute__((constructor(85))) static void count_bootstrap(void) {
  ++bootstrap_count;
}

__attribute__((constructor)) static void application_ctor(void) {
  if (!ready() || *side_data_with_addend != 43 || bootstrap_count != 1)
    std::abort();
  app_initialized = true;
}

static void *worker(void *) {
  __wasm_call_runtime_ctors();
  return bootstrap_count == 1 ? nullptr : reinterpret_cast<void *>(1);
}

int main(void) {
  if (!app_initialized || !ready())
    return 1;
  __wasm_call_runtime_ctors();
  pthread_t thread;
  void *result = nullptr;
  if (pthread_create(&thread, nullptr, worker, nullptr) ||
      pthread_join(thread, &result) || result || bootstrap_count != 1)
    return 2;
  std::cout << "runtime bootstrap once; application ctor after side passed"
            << std::endl;
  return 0;
}
