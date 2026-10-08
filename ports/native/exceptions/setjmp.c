/* Cover nested handlers, an indirect call, zero normalization and repeated recovery. */
#include <setjmp.h>
#include <stdio.h>
#include <string.h>
void library_jump(jmp_buf destination, int value);
static jmp_buf outer, inner;
static void nested(void) {
    if (setjmp(inner) == 0) library_jump(outer, 42);
}
int main(int argc, char **argv) {
    volatile int recovered = 0;
    int value = setjmp(outer);
    if (!value) {
        recovered++;
        void (*call)(void) = nested;
        call();
        return 1;
    }
    if (value != 42 || recovered != 1) return 2;
    value = setjmp(inner);
    if (!value) library_jump(inner, 0);
    if (value != 1) return 3;
    if (argc > 1 && strcmp(argv[1], "loop") == 0) {
        for (;;) { if (setjmp(inner) == 0) library_jump(inner, 1); }
    }
    if (argc > 1 && strcmp(argv[1], "stress") == 0) {
        volatile unsigned int count = 0;
        while (count < 1000000) {
            if (setjmp(inner) == 0) library_jump(inner, 1);
            ++count;
        }
        puts("setjmp: one million recoveries passed");
        return 0;
    }
    puts("setjmp: nested cross-library recovery and zero normalization passed");
    return 0;
}
