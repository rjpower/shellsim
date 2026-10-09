#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main_value = 100;
int main_callback(int value) { return value * 2; }

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "/lib/libfixture.so";
    void *library = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!library) { printf("load error: %s\n", dlerror()); return 1; }
    int (*add)(int) = (int (*)(int))dlsym(library, "library_add");
    int *value = dlsym(library, "shared_value");
    if (!add || !value) { printf("symbol error: %s\n", dlerror()); return 2; }
    printf("%d %d\n", add(4), *value);
    main_value = 200;
    printf("%d %d\n", add(1), *value);
    if (dlopen(path, RTLD_NOW | RTLD_LOCAL) != library) return 3;
    if (dlsym(library, "missing_symbol") || !dlerror()) return 4;
    printf("shared data, callback, constructor, repeat load, missing symbol: ok\n");
    return 0;
}
