#!/bin/sh
set -eu
cd /work
cat > value.c <<'C'
int value(void) { return 42; }
C
cat > main.c <<'C'
#include <stdio.h>
int value(void);
int main(void) { printf("guest C %d\n", value()); return value() != 42; }
C
clang -c value.c -o value.o
ar rcs libvalue.a value.o
ranlib libvalue.a
llvm-nm libvalue.a > symbols.txt
grep -q 'value' symbols.txt
cat > c.rsp <<'ARGS'
main.c libvalue.a -o c-program.wasm
ARGS
cc @c.rsp
chmod +x c-program.wasm
./c-program.wasm
cat > cpp.cpp <<'CPP'
#include <iostream>
#include <stdexcept>
#include <vector>
template<class T> T sum(const std::vector<T>& values) {
  T total = 0;
  for (T value : values) total += value;
  return total;
}
int main() {
  try { throw std::runtime_error("guest exception"); }
  catch (const std::runtime_error&) {
    std::cout << "guest C++ " << sum(std::vector<int>{20, 22}) << '\n';
    return sum(std::vector<int>{20, 22}) != 42;
  }
  return 1;
}
CPP
c++ cpp.cpp -o cpp-program.wasm
chmod +x cpp-program.wasm
./cpp-program.wasm
cat > thread.c <<'C'
#include <errno.h>
#include <pthread.h>
#include <stdio.h>
static void *worker(void *unused) { errno = 29; return (void *)0; }
int main(void) {
  pthread_t thread;
  errno = 17;
  if (pthread_create(&thread, 0, worker, 0)) return 1;
  if (pthread_join(thread, 0)) return 2;
  if (errno != 17) return 3;
  puts("guest thread");
  return 0;
}
C
clang thread.c -o thread-program.wasm
chmod +x thread-program.wasm
./thread-program.wasm
printf '#include <stddef.h>\n#define ANSWER 42\nANSWER\n' > preprocess.c
clang -E -P preprocess.c > preprocess.txt
grep -q '^42$' preprocess.txt
printf 'int main( {\n' > invalid.c
if clang invalid.c -o invalid.wasm > invalid.stdout 2> invalid.stderr; then
  exit 1
fi
if clang missing.c -o missing.wasm > missing.stdout 2> missing.stderr; then
  exit 1
fi
printf 'guest compiler acceptance passed\n'
