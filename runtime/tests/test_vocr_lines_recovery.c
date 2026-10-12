/* Procedural recovery safety: no screenshots, fonts, models or heap in runtime. */
#include "../vocr_lines.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GUARD 32u
#define REQUIRE(x) do { if (!(x)) { fprintf(stderr, "recovery safety line %d\n", __LINE__); return 1; } } while (0)

static uint32_t rng = 0x74329b1u;
static uint32_t random_word(void)
{
    rng ^= rng << 13;
    rng ^= rng >> 17;
    rng ^= rng << 5;
    return rng;
}

static int checked(const VocrLinesImage *im, unsigned offset)
{
    size_t need = vocr_lines_arena_bytes(im->width, im->height), i;
    unsigned char *block = (unsigned char *)malloc(need + 2u * GUARD + 4u);
    unsigned char *arena;
    VocrLineSet old, got;
    unsigned n, j;
    REQUIRE(block != NULL);
    memset(block, 0xa5, need + 2u * GUARD + 4u);
    arena = block + GUARD + offset;
    REQUIRE(vocr_find_lines(im, VOCR_LINES_TEXTBOX, arena, need, &old) == VOCR_LINES_OK);
    REQUIRE(vocr_find_lines(im, VOCR_LINES_TEXTBOX_RECOVERY, arena, need, &got) == VOCR_LINES_OK);
    REQUIRE(got.count <= VOCR_LINES_MAX && got.count >= old.count);
    REQUIRE(got.truncated == old.truncated);
    for (i = 0; i < GUARD + offset; i++) REQUIRE(block[i] == 0xa5);
    for (i = GUARD + offset + need; i < need + 2u * GUARD + 4u; i++) REQUIRE(block[i] == 0xa5);
    for (n = 0; n < got.count; n++) {
        const VocrLine *line = &got.lines[n];
        REQUIRE(line->w && line->h && line->x < im->width && line->y < im->height);
        REQUIRE(line->w <= im->width - line->x && line->h <= im->height - line->y);
        REQUIRE(line->dark <= 1u);
        if (n) REQUIRE(got.lines[n - 1u].y < line->y ||
                       (got.lines[n - 1u].y == line->y && got.lines[n - 1u].x <= line->x));
        for (j = 0; j < n; j++) REQUIRE(memcmp(line, &got.lines[j], sizeof(*line)) != 0);
    }
    for (n = 0; n < old.count; n++) {
        for (j = 0; j < got.count; j++)
            if (!memcmp(&old.lines[n], &got.lines[j], sizeof(VocrLine))) break;
        REQUIRE(j < got.count); /* original C3 crop/polarity must remain exact */
    }
    REQUIRE(vocr_find_lines(im, VOCR_LINES_TEXTBOX_RECOVERY, arena, need - 1u, &got) == VOCR_LINES_E_ARENA);
    free(block);
    return 0;
}

int main(void)
{
    unsigned k, n = 0;
    for (k = 0; k < 2400; k++) {
        unsigned width = 1u + random_word() % 960u, height = 1u + random_word() % 544u;
        unsigned bpp = k % 2u ? VOCR_LINES_RGBA : VOCR_LINES_GREY;
        unsigned offset = k % 4u;
        size_t stride, bytes, i;
        unsigned char *pixels;
        VocrLinesImage image;
        if (k < 64) { width = 1u + k % 8u; height = 1u + k / 8u; }
        if (k % 101u == 0) { width = 960; height = 544; }
        if (k % 103u == 0) { width = 480; height = 272; }
        stride = (size_t)width * bpp + random_word() % 9u;
        bytes = stride * height;
        pixels = (unsigned char *)malloc(bytes);
        REQUIRE(pixels != NULL);
        for (i = 0; i < bytes; i++) {
            unsigned value = random_word();
            pixels[i] = k % 5u == 0 ? 128 : k % 5u == 1 ? (value & 1u ? 235 : 20) : (unsigned char)value;
        }
        image.pixels = pixels; image.width = width; image.height = height;
        image.stride = stride; image.bpp = bpp;
        REQUIRE(checked(&image, offset) == 0);
        free(pixels);
        n++;
    }
    printf("RECOVERY_SAFETY_OK %u generated images; exact arenas, alignments0-3, bounds, preserved C3\n", n);
    return 0;
}
