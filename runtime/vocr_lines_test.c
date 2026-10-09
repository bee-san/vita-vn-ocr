/* Host test driver for vocr_lines.c (not part of the console runtime).
 *
 *   vocr_lines_test BATCH     run every image in BATCH, print one result line each
 *   vocr_lines_test --arena W H
 *   vocr_lines_test --selftest
 *
 * BATCH is a sequence of records, all integers little-endian uint32:
 *   width, height, bpp (1 or 4), finder (0 grad, 1 legacy), stride, then
 *   height * stride bytes of pixels.
 * Output per record: "rc count truncated" then "x y w h dark" per line.
 * Every image and the arena get their own exactly-sized heap block, and the arena
 * starts at an odd address, so ASan sees any overrun or misaligned access.
 */
#include "vocr_lines.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int read_u32(FILE *f, unsigned *v)
{
    unsigned char b[4];
    if (fread(b, 1, 4, f) != 4) return 0;
    *v = (unsigned)b[0] | (unsigned)b[1] << 8 | (unsigned)b[2] << 16 | (unsigned)b[3] << 24;
    return 1;
}

static int run_one(const VocrLinesImage *im, int finder, size_t extra, VocrLineSet *out)
{
    size_t need = vocr_lines_arena_bytes(im->width, im->height);
    unsigned char *mem = (unsigned char *)malloc(need + 1 + extra);
    int rc;
    if (!mem) return -100;
    rc = vocr_find_lines(im, finder, mem + 1, need + extra, out);
    free(mem);
    return rc;
}

static int batch(const char *path)
{
    FILE *f = fopen(path, "rb");
    unsigned w, h, bpp, finder, stride;
    if (!f) {
        perror(path);
        return 2;
    }
    while (read_u32(f, &w)) {
        VocrLinesImage im;
        VocrLineSet out;
        unsigned char *px;
        size_t bytes;
        unsigned i;
        int rc;
        if (!read_u32(f, &h) || !read_u32(f, &bpp) || !read_u32(f, &finder) || !read_u32(f, &stride)) {
            fprintf(stderr, "truncated header\n");
            fclose(f);
            return 2;
        }
        bytes = (size_t)stride * h;
        px = (unsigned char *)malloc(bytes ? bytes : 1);
        if (!px || fread(px, 1, bytes, f) != bytes) {
            fprintf(stderr, "truncated pixels\n");
            free(px);
            fclose(f);
            return 2;
        }
        im.pixels = px;
        im.width = w;
        im.height = h;
        im.stride = stride;
        im.bpp = bpp;
        rc = run_one(&im, (int)finder, 0, &out);
        printf("%d %u %u", rc, out.count, out.truncated);
        for (i = 0; i < out.count; i++)
            printf(" %u %u %u %u %u", out.lines[i].x, out.lines[i].y, out.lines[i].w, out.lines[i].h,
                   out.lines[i].dark);
        printf("\n");
        free(px);
    }
    fclose(f);
    return 0;
}

#define CHECK(c)                                                       \
    do {                                                               \
        if (!(c)) {                                                    \
            fprintf(stderr, "selftest failed line %d: %s\n", __LINE__, #c); \
            return 1;                                                  \
        }                                                              \
    } while (0)

/* Two dark text-like lines on a light box, drawn as rows of small blocks. */
static void draw_lines(unsigned char *g, unsigned w, unsigned h, unsigned stride, unsigned bpp,
                       unsigned char ink, unsigned char paper)
{
    unsigned x, y, k;
    for (y = 0; y < h; y++)
        for (x = 0; x < w; x++)
            for (k = 0; k < bpp; k++) g[(size_t)y * stride + (size_t)x * bpp + k] = paper;
    for (y = 0; y < h; y++) {
        int line = (y >= 10 && y < 30) || (y >= 44 && y < 64);
        if (!line) continue;
        for (x = 12; x < w - 40; x++) {
            int glyph = (x % 24) < 18 && ((x % 24) % 6 < 3 || (y % 8) < 3);
            if (!glyph) continue;
            for (k = 0; k < bpp; k++) g[(size_t)y * stride + (size_t)x * bpp + k] = ink;
        }
    }
}

static int selftest(void)
{
    enum { W = 300, H = 80 };
    static unsigned char buf[H * (W * 4 + 8)];
    VocrLinesImage im;
    VocrLineSet out;
    unsigned char small[16];
    size_t need = vocr_lines_arena_bytes(W, H);
    unsigned char *arena = (unsigned char *)malloc(need + 8);
    int rc, pol;
    CHECK(arena != NULL);
    CHECK(vocr_lines_arena_bytes(0, 5) == 0 && vocr_lines_arena_bytes(5, 0) == 0);
    CHECK(vocr_lines_arena_bytes(960, 544) >= vocr_lines_arena_bytes(480, 272));

    /* dark-on-light RGBA with padded stride: both lines found, flagged dark */
    for (pol = 0; pol < 2; pol++) {
        im.pixels = buf;
        im.width = W;
        im.height = H;
        im.bpp = VOCR_LINES_RGBA;
        im.stride = W * 4 + 8;
        draw_lines(buf, W, H, (unsigned)im.stride, 4, pol ? 250 : 20, pol ? 20 : 235);
        rc = vocr_find_lines(&im, VOCR_LINES_GRAD, arena + 1, need, &out);
        CHECK(rc == VOCR_LINES_OK);
        CHECK(out.count == 2 && !out.truncated);
        CHECK(out.lines[0].y <= 10 && out.lines[0].y + out.lines[0].h >= 30);
        CHECK(out.lines[1].y <= 44 && out.lines[1].y + out.lines[1].h >= 64);
        CHECK(out.lines[0].y + out.lines[0].h <= out.lines[1].y);
        CHECK(out.lines[0].x <= 12 && out.lines[0].x + out.lines[0].w >= W - 46);
        CHECK(out.lines[0].dark == (unsigned)!pol && out.lines[1].dark == (unsigned)!pol);
        rc = vocr_find_lines(&im, VOCR_LINES_LEGACY, arena + 1, need, &out);
        if (pol) CHECK(rc == VOCR_LINES_OK && out.count == 2);   /* bright text: legacy finds it */
        else CHECK(rc == VOCR_LINES_E_REGION);                  /* light box: legacy rejects */
    }
    /* the same image as grey gives the same boxes */
    {
        static unsigned char grey[H * W];
        VocrLineSet out2;
        unsigned i;
        draw_lines(grey, W, H, W, 1, 20, 235);
        im.pixels = grey;
        im.bpp = VOCR_LINES_GREY;
        im.stride = W;
        CHECK(vocr_find_lines(&im, VOCR_LINES_GRAD, arena, need, &out2) == VOCR_LINES_OK);
        im.pixels = buf;
        im.bpp = VOCR_LINES_RGBA;
        im.stride = W * 4 + 8;
        draw_lines(buf, W, H, (unsigned)im.stride, 4, 20, 235);
        CHECK(vocr_find_lines(&im, VOCR_LINES_GRAD, arena + 3, need, &out) == VOCR_LINES_OK);
        CHECK(out.count == out2.count);
        for (i = 0; i < out.count; i++)
            CHECK(!memcmp(&out.lines[i], &out2.lines[i], sizeof(VocrLine)));
    }
    /* errors */
    CHECK(vocr_find_lines(&im, VOCR_LINES_GRAD, arena, need - 1, &out) == VOCR_LINES_E_ARENA);
    CHECK(vocr_find_lines(&im, 7, arena, need, &out) == VOCR_LINES_E_ARG);
    CHECK(vocr_find_lines(NULL, VOCR_LINES_GRAD, arena, need, &out) == VOCR_LINES_E_ARG);
    CHECK(vocr_find_lines(&im, VOCR_LINES_GRAD, NULL, need, &out) == VOCR_LINES_E_ARG);
    CHECK(vocr_find_lines(&im, VOCR_LINES_GRAD, arena, need, NULL) == VOCR_LINES_E_ARG);
    im.stride = W * 4 - 1;
    CHECK(vocr_find_lines(&im, VOCR_LINES_GRAD, arena, need, &out) == VOCR_LINES_E_ARG);
    im.stride = W * 4 + 8;
    im.bpp = 3;
    CHECK(vocr_find_lines(&im, VOCR_LINES_GRAD, arena, need, &out) == VOCR_LINES_E_ARG);
    im.bpp = VOCR_LINES_RGBA;
    CHECK(vocr_line_dark(&im, W - 1, 0, 2, 1) == VOCR_LINES_E_ARG);
    CHECK(vocr_line_dark(&im, 0, 0, 0, 1) == VOCR_LINES_E_ARG);
    CHECK(vocr_line_dark(&im, 0, 0, W, H) == 1);
    /* tiny and flat images: no lines, no crash */
    memset(small, 128, sizeof(small));
    im.pixels = small;
    im.bpp = VOCR_LINES_GREY;
    for (im.width = 1; im.width <= 4; im.width++) {
        im.height = 16 / im.width;
        im.stride = im.width;
        CHECK(vocr_find_lines(&im, VOCR_LINES_GRAD, arena, need, &out) == VOCR_LINES_OK && !out.count);
        CHECK(vocr_find_lines(&im, VOCR_LINES_LEGACY, arena, need, &out) == VOCR_LINES_OK && !out.count);
    }
    free(arena);
    printf("selftest ok, arena %lu bytes at 960x544, %lu at 480x272\n",
           (unsigned long)vocr_lines_arena_bytes(960, 544), (unsigned long)vocr_lines_arena_bytes(480, 272));
    return 0;
}

int main(int argc, char **argv)
{
    if (argc == 2 && !strcmp(argv[1], "--selftest")) return selftest();
    if (argc == 4 && !strcmp(argv[1], "--arena")) {
        printf("%lu\n", (unsigned long)vocr_lines_arena_bytes((unsigned)strtoul(argv[2], NULL, 10),
                                                              (unsigned)strtoul(argv[3], NULL, 10)));
        return 0;
    }
    if (argc == 2) return batch(argv[1]);
    fprintf(stderr, "usage: %s BATCH | --arena W H | --selftest\n", argv[0]);
    return 2;
}
