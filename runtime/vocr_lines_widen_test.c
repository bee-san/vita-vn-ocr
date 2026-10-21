/* vocr_lines_widen_test - self-test and parity driver of vocr_lines_widen.c (not part of the console build).
 *
 *   vocr_lines_widen_test               the self-test
 *   vocr_lines_widen_test --parity FILE one "x w sides" line per case (tests/lines_widen_parity.py writes them)
 *
 * FILE: u32 count, then per case u32 width, height, bpp, rx, rw, x, w and height x width x bpp pixel bytes.
 */
#include "vocr_lines_widen.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;
#define CHECK(c)                                                              \
    do {                                                                      \
        if (!(c)) {                                                           \
            fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #c); \
            failures++;                                                       \
        }                                                                     \
    } while (0)

#define W 60
#define H 24
static unsigned char px[H * W * 4];

/* background 40, ink 230 in columns [a, b) of rows 4..19 */
static void strip(unsigned bpp)
{
    unsigned y, x, k;
    for (y = 0; y < H; y++)
        for (x = 0; x < W; x++)
            for (k = 0; k < bpp; k++) px[(y * W + x) * bpp + k] = 40;
}

static void ink(unsigned bpp, unsigned a, unsigned b)
{
    unsigned y, x, k;
    for (y = 4; y < 20; y++)
        for (x = a; x < b; x++)
            for (k = 0; k < bpp && k < 3; k++) px[(y * W + x) * bpp + k] = 230;
}

static int run(unsigned bpp, unsigned rx, unsigned rw, unsigned *x, unsigned *w)
{
    VocrLinesImage im;
    im.pixels = px;
    im.width = W;
    im.height = H;
    im.stride = (size_t)W * bpp;
    im.bpp = bpp;
    return vocr_line_widen(&im, rx, rw, x, w);
}

static void selftest(void)
{
    unsigned bpp, x, w;
    for (bpp = 1; bpp <= 4; bpp += 3) {
        /* a glyph [8, 16) cut by the region starting at 12: continued to 8, minus the padding */
        strip(bpp);
        ink(bpp, 8, 16);
        ink(bpp, 20, 30);
        x = 0, w = 24; /* region [12, 48): the finder's box from its first column */
        CHECK(run(bpp, 12, 36, &x, &w) == VOCR_WIDEN_LEFT && x == 5 && w == 31);
        /* the same box, but the region starts in the gap before the glyph at 20: no ink at the edge */
        x = 0, w = 20;
        CHECK(run(bpp, 17, 23, &x, &w) == 0 && x == 17 && w == 20);
        /* a box that does not start at the region's first column is never widened */
        x = 1, w = 20;
        CHECK(run(bpp, 12, 36, &x, &w) == 0 && x == 13 && w == 20);
        /* right: a glyph [40, 52) cut by the region ending at 44 */
        strip(bpp);
        ink(bpp, 30, 36);
        ink(bpp, 40, 52);
        x = 10, w = 24; /* region [10, 44) */
        CHECK(run(bpp, 10, 34, &x, &w) == VOCR_WIDEN_RIGHT && x == 20 && w == 35);
        /* across a gap shorter than max(2, h/8) = 3 columns, not across a longer one */
        strip(bpp);
        ink(bpp, 5, 8);
        ink(bpp, 10, 13);
        ink(bpp, 15, 23);
        x = 0, w = 30; /* region [17, 47): the glyph at 15 is cut; gaps of 2 columns: continued to 5 */
        CHECK(run(bpp, 17, 30, &x, &w) == VOCR_WIDEN_LEFT && x == 2 && w == 45);
        /* ink that runs on to the margin's end (a frame, a scene) is no cut character */
        strip(bpp);
        ink(bpp, 0, 16);
        ink(bpp, 20, 30);
        x = 0, w = 24;
        CHECK(run(bpp, 12, 36, &x, &w) == 0 && x == 12 && w == 24);
        strip(bpp);
        ink(bpp, 40, 60);
        x = 10, w = 24; /* region [10, 44): its right edge cuts ink that reaches the strip's end */
        CHECK(run(bpp, 10, 34, &x, &w) == 0 && x == 20 && w == 24);
        strip(bpp);
        ink(bpp, 2, 5);
        ink(bpp, 9, 20);
        x = 0, w = 30; /* a gap of 4 columns [5, 9): stops at 9 */
        CHECK(run(bpp, 14, 30, &x, &w) == VOCR_WIDEN_LEFT && x == 6 && w == 38);
        /* a region at the image's own edge has no margin: nothing to continue into */
        strip(bpp);
        ink(bpp, 0, 6);
        x = 0, w = 20;
        CHECK(run(bpp, 0, 30, &x, &w) == 0 && x == 0 && w == 20);
        /* faint texture (range 10) is not ink: MIN_INK */
        strip(bpp);
        ink(bpp, 20, 40);
        {
            unsigned y, c, k;
            for (y = 0; y < H; y++)
                for (c = 0; c < 20; c++)
                    for (k = 0; k < bpp && k < 3; k++) px[(y * W + c) * bpp + k] = (unsigned char)(40 + (y % 2) * 10);
        }
        x = 0, w = 20;
        CHECK(run(bpp, 20, 30, &x, &w) == 0 && x == 20 && w == 20);
    }
    /* errors */
    x = 0, w = 10;
    CHECK(run(1, 50, 20, &x, &w) == VOCR_LINES_E_ARG);
    x = 5, w = 10;
    CHECK(run(1, 10, 12, &x, &w) == VOCR_LINES_E_ARG);
    x = 0, w = 0;
    CHECK(run(1, 10, 12, &x, &w) == VOCR_LINES_E_ARG);
    CHECK(vocr_line_widen(NULL, 0, 1, &x, &w) == VOCR_LINES_E_ARG);
}

static uint32_t rd(const unsigned char **p)
{
    const unsigned char *q = *p;
    *p += 4;
    return (uint32_t)q[0] | (uint32_t)q[1] << 8 | (uint32_t)q[2] << 16 | (uint32_t)q[3] << 24;
}

static int parity(const char *path)
{
    FILE *f = fopen(path, "rb");
    unsigned char *buf;
    const unsigned char *p;
    long len;
    uint32_t n, i;
    if (!f || fseek(f, 0, SEEK_END) || (len = ftell(f)) < 4 || fseek(f, 0, SEEK_SET)) return 2;
    buf = (unsigned char *)malloc((size_t)len);
    if (!buf || fread(buf, 1, (size_t)len, f) != (size_t)len) return 2;
    fclose(f);
    p = buf;
    n = rd(&p);
    for (i = 0; i < n; i++) {
        VocrLinesImage im;
        unsigned rx, rw, x, w;
        int rc;
        im.width = rd(&p);
        im.height = rd(&p);
        im.bpp = rd(&p);
        rx = rd(&p);
        rw = rd(&p);
        x = rd(&p);
        w = rd(&p);
        im.stride = (size_t)im.width * im.bpp;
        im.pixels = p;
        p += (size_t)im.height * im.stride;
        rc = vocr_line_widen(&im, rx, rw, &x, &w);
        printf("%d %u %u\n", rc, x, w);
    }
    free(buf);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc == 3 && !strcmp(argv[1], "--parity")) return parity(argv[2]);
    if (argc != 1) {
        fprintf(stderr, "usage: vocr_lines_widen_test [--parity FILE]\n");
        return 2;
    }
    selftest();
    if (failures) {
        fprintf(stderr, "vocr_lines_widen_test: %d failures\n", failures);
        return 1;
    }
    printf("vocr_lines_widen_test: all checks passed\n");
    return 0;
}
