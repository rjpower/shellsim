/* Exercise the sealed static provider inside the virtual filesystem. */
#include <stdio.h>
#include <ft2build.h>
#include FT_FREETYPE_H

int main(void) {
    FT_Library library;
    FT_Face face;
    int major, minor, patch;
    const unsigned char invalid[] = {0, 1, 2, 3};
    if (FT_Init_FreeType(&library)) return 1;
    FT_Library_Version(library, &major, &minor, &patch);
    if (major != 2 || minor != 13 || patch != 3) return 2;
    if (!FT_New_Memory_Face(library, invalid, sizeof(invalid), 0, &face)) return 3;
    if (FT_New_Face(library, "/font.ttf", 0, &face)) return 4;
    if (FT_Set_Pixel_Sizes(face, 0, 24)) return 5;
    if (FT_Load_Char(face, 'A', FT_LOAD_RENDER)) return 6;
    FT_Bitmap *bitmap = &face->glyph->bitmap;
    unsigned sum = 0;
    if (!bitmap->width || !bitmap->rows || bitmap->pitch <= 0) return 7;
    for (unsigned row = 0; row < bitmap->rows; ++row)
        for (unsigned col = 0; col < bitmap->width; ++col)
            sum += bitmap->buffer[row * bitmap->pitch + col];
    if (!sum) return 8;
    FT_Done_Face(face);
    FT_Done_FreeType(library);
    puts("FreeType 2.13.3 rasterization and invalid font rejection passed");
    return 0;
}
