/* Exercise VFS font loading, scalable glyph rasterization and malformed input. */
#include <ft2build.h>
#include FT_FREETYPE_H
#include <stdio.h>
#include <string.h>
int main(int argc, char **argv) {
    FT_Library library;
    FT_Face face;
    if (FT_Init_FreeType(&library)) return 1;
    const unsigned char invalid[] = {0, 1, 2, 3};
    if (!FT_New_Memory_Face(library, invalid, sizeof(invalid), 0, &face)) return 2;
    if (FT_New_Face(library, "/font.ttf", 0, &face)) return 3;
    if (!FT_IS_SCALABLE(face)) return 4;
    if (FT_Set_Pixel_Sizes(face, 0, 24)) return 5;
    for (;;) {
        if (FT_Load_Char(face, 'A', FT_LOAD_RENDER)) return 6;
        FT_Bitmap *bitmap = &face->glyph->bitmap;
        if (!bitmap->width || !bitmap->rows || bitmap->pixel_mode != FT_PIXEL_MODE_GRAY) return 7;
        unsigned long sum = 0;
        for (unsigned int y = 0; y < bitmap->rows; ++y)
            for (unsigned int x = 0; x < bitmap->width; ++x) sum += bitmap->buffer[y * bitmap->pitch + x];
        if (!sum) return 8;
        if (argc < 2 || strcmp(argv[1], "loop")) break;
    }
    FT_Done_Face(face);
    FT_Done_FreeType(library);
    puts("FreeType: scalable glyph and malformed font passed");
    return 0;
}
