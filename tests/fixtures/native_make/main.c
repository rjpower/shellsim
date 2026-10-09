
#include <zlib.h>
#include <stdio.h>
int check_codec(void);
int main(void) {
    int result = check_codec();
    if (result) { fprintf(stderr, "codec check failed: %d\n", result); return result; }
    printf("zlib %s: roundtrip, CRC32, invalid input passed\n", zlibVersion());
    return 0;
}
