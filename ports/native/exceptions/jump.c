/* Compile this separately so the nonlocal jump crosses a static library. */
#include <setjmp.h>
void library_jump(jmp_buf destination, int value) { longjmp(destination, value); }
