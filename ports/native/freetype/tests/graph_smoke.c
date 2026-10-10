/* Exercise shared font parsing/rasterization and FreeType's shared zlib edge. */
#include <ft2build.h>
#include FT_FREETYPE_H
#include FT_GZIP_H
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

/* An authored bitmap font avoids dependence on ambient host fonts. */
static const unsigned char font[] =
    "STARTFONT 2.1\nFONT -shellsim-probe-medium-r-normal--8-80-75-75-c-80-iso10646-1\n"
    "SIZE 8 75 75\nFONTBOUNDINGBOX 8 8 0 0\nSTARTPROPERTIES 2\nFONT_ASCENT 8\nFONT_DESCENT 0\n"
    "ENDPROPERTIES\nCHARS 1\nSTARTCHAR A\nENCODING 65\nSWIDTH 1000 0\nDWIDTH 8 0\n"
    "BBX 8 8 0 0\nBITMAP\n18\n24\n42\n7E\n42\n42\n42\n00\nENDCHAR\nENDFONT\n";
/* A standard gzip stream containing "font". */
static const unsigned char gzip_font[] = {31,139,8,0,0,0,0,0,2,255,75,203,207,43,1,0,210,8,148,208,4,0,0,0};
static void *allocate(FT_Memory memory, long size) {
    (void)memory;
    return malloc((size_t)size);
}
static void release(FT_Memory memory, void *block) {
    (void)memory;
    free(block);
}
static void *resize(FT_Memory memory, long current, long requested, void *block) {
    (void)memory;
    (void)current;
    return realloc(block, (size_t)requested);
}
int main(void) {
    struct FT_MemoryRec_ memory = {NULL, allocate, release, resize};
    FT_Library library;
    FT_Face face;
    const unsigned char invalid[] = {0,1,2,3};
    unsigned char expanded[8];
    FT_ULong length = sizeof expanded;
    if (FT_Init_FreeType(&library)) return 1;
    if (!FT_New_Memory_Face(library, invalid, sizeof invalid, 0, &face)) return 2;
    if (FT_New_Memory_Face(library, font, sizeof font - 1, 0, &face)) return 3;
    if (FT_Select_Size(face, 0)) return 4;
    if (FT_Get_Char_Index(face, 'A') == 0) return 5;
    if (FT_Load_Char(face, 'A', FT_LOAD_RENDER)) return 6;
    if (face->glyph->bitmap.width != 8 || face->glyph->bitmap.rows != 8) return 7;
    if (face->glyph->bitmap.buffer[0] != 0x18 || face->glyph->bitmap.buffer[3] != 0x7e) return 8;
    if (FT_Done_Face(face)) return 9;
    if (FT_Gzip_Uncompress(&memory, expanded, &length, gzip_font, sizeof gzip_font)) return 10;
    if (length != 4 || memcmp(expanded, "font", 4)) return 11;
    length = sizeof expanded;
    if (!FT_Gzip_Uncompress(&memory, expanded, &length, invalid, sizeof invalid)) return 12;
    length = sizeof expanded;
    if (FT_Gzip_Uncompress(&memory, expanded, &length, gzip_font, sizeof gzip_font)) return 13;
    if (FT_Done_FreeType(library)) return 14;
    puts("Shared FreeType: glyph raster, malformed recovery and shared zlib passed");
    return 0;
}
