/* Exercise actual FreeType initialization, version and malformed-memory APIs. */
#include <ft2build.h>
#include FT_FREETYPE_H
#include <stdio.h>
int main(void) {
    FT_Library library;
    FT_Face face;
    const unsigned char invalid[] = {0, 1, 2, 3};
    FT_Int major, minor, patch;
    if (FT_Init_FreeType(&library)) return 1;
    FT_Library_Version(library, &major, &minor, &patch);
    if (major != 2 || minor != 13 || patch != 3) return 2;
    if (!FT_New_Memory_Face(library, invalid, sizeof(invalid), 0, &face)) return 3;
    if (FT_Done_FreeType(library)) return 4;
    puts("FreeType: initialization, version and malformed-input rejection passed");
    return 0;
}
