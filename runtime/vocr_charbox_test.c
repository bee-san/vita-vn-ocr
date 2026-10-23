/* vocr_charbox_test - self-test and parity driver of vocr_charbox.c (not part of the console build).
 *
 *   vocr_charbox_test                       the self-test (contracts, error paths, edge cases)
 *   vocr_charbox_test --parity VOCAB CASES  one result line per case of CASES (tests/charbox_parity.py writes them)
 *
 * VOCAB: one class per line, UTF-8; class 0 is the blank and is not in the file, so line i is class i + 1. The
 * driver's model is that vocabulary at height 32, stride 8, max width 1024 (no weights: vocr_charbox.c reads only
 * the classes' texts and the geometry).
 * CASES (little-endian): u32 count, then per case u32 kind.
 *   kind 0 (find): u32 frames, u32 stride, frames x stride u16 classes, u32 w, u32 h, u32 flags, u32 has_cal,
 *                  9 i16 (shift, groups 0..7), u32 cap, u32 ink (0 none, 1 profile, 2 GREY8, 3 RGBA8), then the
 *                  profile (w u32) or the pixels (w x h, or w x h x 4 bytes)
 *   kind 1 (align): u32 n, u32 m, n u32 code points, m u32 code points
 * Results: "F <rc> <n>" then per character " <cls> <f0> <f1> <group> <c> <x0> <x1> <i0> <i1>"; "A <rc>" then
 * " <map[j]>" per character of dst.
 */
#include "vocr_charbox.h"

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

static const unsigned char dummy_blob[16];

static void model_init(vocr_model *m, const uint16_t *off, const char *voc, unsigned n)
{
    memset(m, 0, sizeof *m);
    m->blob = dummy_blob;
    m->height = 32;
    m->max_width = 1024;
    m->stride = 8;
    m->n_classes = n;
    m->vocab_off = off;
    m->vocab = voc;
}

/* ------------------------------------------------------------- the self-test */

static const char *const tiny[] = {"", "あ", "い", "、", "。", "「", "」", "っ", "ー", "…", "！", "A", "漢", "ab"};
enum { NTINY = 14 };
static uint16_t tiny_off[NTINY + 1];
static char tiny_voc[128];

static void tiny_model(vocr_model *m)
{
    size_t used = 0;
    unsigned i;
    for (i = 0; i < NTINY; i++) {
        size_t n = strlen(tiny[i]);
        tiny_off[i] = (uint16_t)used;
        memcpy(tiny_voc + used, tiny[i], n);
        used += n;
    }
    tiny_off[NTINY] = (uint16_t)used;
    model_init(m, tiny_off, tiny_voc, NTINY);
}

static void selftest(void)
{
    vocr_model m;
    vocr_charbox out[VOCR_CHARBOX_MAX];
    vocr_charbox_cal cal;
    unsigned n = 99, i;
    /* あ blank あ: two characters (a blank separates repeats); あ あ: one */
    static const uint16_t c1[] = {1, 0, 1, 1, 0, 0, 2, 2, 2, 0};
    static const uint16_t c2[] = {0, 0, 0, 0};
    unsigned char grey[32 * 40];
    uint32_t prof[40];
    vocr_image im;
    tiny_model(&m);

    CHECK(vocr_charbox_find(&m, c1, 1, 10, 80, 32, 0, NULL, NULL, NULL, out, VOCR_CHARBOX_MAX, &n) == 0);
    CHECK(n == 3);
    CHECK(out[0].cls == 1 && out[0].f0 == 0 && out[0].f1 == 0);
    CHECK(out[1].cls == 1 && out[1].f0 == 2 && out[1].f1 == 3);
    CHECK(out[2].cls == 2 && out[2].f0 == 6 && out[2].f1 == 8);
    /* 80 x 32 reads at 80 resized px: frame f's centre is 8 f + 4 px, x 16 */
    CHECK(out[0].c == 4 * 16 && out[1].c == 20 * 16 && out[2].c == 52 * 16);
    CHECK(out[0].x0 == 0 && out[0].x1 == 12 * 16 && out[1].x0 == 12 * 16 && out[2].x1 == 60 * 16);
    for (i = 0; i < n; i++) CHECK(out[i].i0 == out[i].x0 && out[i].i1 == out[i].x1);
    /* RUN: the run's centre */
    CHECK(vocr_charbox_find(&m, c1, 1, 10, 80, 32, VOCR_CHARBOX_RUN, NULL, NULL, NULL, out, VOCR_CHARBOX_MAX, &n) == 0);
    CHECK(out[1].c == 24 * 16 && out[2].c == 60 * 16);
    /* a crop twice as wide maps every centre twice as far (resized width is the same) */
    CHECK(vocr_charbox_find(&m, c1, 1, 10, 160, 64, 0, NULL, NULL, NULL, out, VOCR_CHARBOX_MAX, &n) == 0);
    CHECK(out[0].c == 8 * 16 && out[2].c == 104 * 16);
    /* the calibration: a shift for every character, an offset for its group */
    memset(&cal, 0, sizeof cal);
    cal.shift = -16;
    cal.group[VOCR_CHARBOX_G_OTHER] = 8;
    CHECK(vocr_charbox_find(&m, c1, 1, 10, 80, 32, 0, &cal, NULL, NULL, out, VOCR_CHARBOX_MAX, &n) == 0);
    CHECK(out[0].c == 4 * 16 - 8 && out[0].group == VOCR_CHARBOX_G_OTHER);
    cal.shift = -32767; /* clipped to the crop */
    CHECK(vocr_charbox_find(&m, c1, 1, 10, 80, 32, 0, &cal, NULL, NULL, out, VOCR_CHARBOX_MAX, &n) == 0);
    CHECK(out[0].c == 0);
    /* cap: the line's first characters */
    CHECK(vocr_charbox_find(&m, c1, 1, 10, 80, 32, 0, NULL, NULL, NULL, out, 2, &n) == 0 && n == 2);
    CHECK(out[1].f1 == 3);
    /* no character */
    CHECK(vocr_charbox_find(&m, c2, 1, 4, 80, 32, VOCR_CHARBOX_INK, NULL, NULL, prof, out, 8, &n) == 0 && n == 0);
    /* stride: the top-1 of a top-K array */
    {
        static const uint16_t k2[] = {1, 5, 0, 1, 2, 9, 2, 3};
        CHECK(vocr_charbox_find(&m, k2, 2, 4, 80, 32, 0, NULL, NULL, NULL, out, 8, &n) == 0 && n == 2);
        CHECK(out[0].cls == 1 && out[0].f0 == 0 && out[1].cls == 2 && out[1].f0 == 2 && out[1].f1 == 3);
    }
    /* groups */
    CHECK(vocr_charbox_group(0x3001) == VOCR_CHARBOX_G_CLOSE && vocr_charbox_group(0x300C) == VOCR_CHARBOX_G_OPEN);
    CHECK(vocr_charbox_group(0x2026) == VOCR_CHARBOX_G_DOTS && vocr_charbox_group(0x3063) == VOCR_CHARBOX_G_SMALL);
    CHECK(vocr_charbox_group(0x30FC) == VOCR_CHARBOX_G_LONG && vocr_charbox_group(0xFF01) == VOCR_CHARBOX_G_MARK);
    CHECK(vocr_charbox_group('A') == VOCR_CHARBOX_G_HALF && vocr_charbox_group(0x6F22) == VOCR_CHARBOX_G_OTHER);
    {   /* a class of two code points is group 0 */
        static const uint16_t c3[] = {13, 0};
        CHECK(vocr_charbox_find(&m, c3, 1, 2, 40, 32, 0, NULL, NULL, NULL, out, 8, &n) == 0 && n == 1);
        CHECK(out[0].group == VOCR_CHARBOX_G_OTHER);
    }
    /* INK: two glyphs with a gap on a 40 x 32 crop (read at 40 resized px); profile and image agree */
    memset(grey, 0, sizeof grey);
    for (i = 0; i < 40; i++) {
        const int ink = (i >= 3 && i < 15) || (i >= 22 && i < 36);
        grey[10 * 40 + i] = ink ? 200 : 0;
        grey[11 * 40 + i] = ink ? 100 : 0;
    }
    im.pixels = grey;
    im.width = 40;
    im.height = 32;
    im.stride = 40;
    im.format = VOCR_GREY8;
    CHECK(vocr_charbox_profile(&im, prof) == 0);
    CHECK(prof[0] == 0 && prof[5] == 200 + 100 + 100);
    {
        static const uint16_t c4[] = {0, 1, 0, 2, 0};
        vocr_charbox a[4], b[4];
        unsigned na, nb;
        CHECK(vocr_charbox_find(&m, c4, 1, 5, 40, 32, VOCR_CHARBOX_INK, NULL, &im, NULL, a, 4, &na) == 0);
        CHECK(vocr_charbox_find(&m, c4, 1, 5, 40, 32, VOCR_CHARBOX_INK, NULL, NULL, prof, b, 4, &nb) == 0);
        CHECK(na == 2 && nb == 2 && !memcmp(a, b, sizeof a[0] * 2));
        /* centres 12 and 28 px; the cut in the middle of the gap 15..21; each ink box its cell's ink */
        CHECK(a[0].x1 == 296 && a[1].x0 == 296);
        CHECK(a[0].x0 == 4 * 16 && a[0].i0 == 4 * 16 && a[0].i1 == 15 * 16 && a[0].c == 152);
        CHECK(a[1].i0 == 22 * 16 && a[1].i1 == 36 * 16 && a[1].x1 == 36 * 16 && a[1].c == 464);
        im.format = VOCR_GREY8 | VOCR_INVERT; /* polarity-free */
        CHECK(vocr_charbox_find(&m, c4, 1, 5, 40, 32, VOCR_CHARBOX_INK, NULL, &im, NULL, b, 4, &nb) == 0);
        CHECK(!memcmp(a, b, sizeof a[0] * 2));
        im.format = VOCR_GREY8;
    }
    /* errors */
    CHECK(vocr_charbox_find(NULL, c1, 1, 10, 80, 32, 0, NULL, NULL, NULL, out, 8, &n) == VOCR_E_ARG);
    CHECK(vocr_charbox_find(&m, NULL, 1, 10, 80, 32, 0, NULL, NULL, NULL, out, 8, &n) == VOCR_E_ARG);
    CHECK(vocr_charbox_find(&m, c1, 0, 10, 80, 32, 0, NULL, NULL, NULL, out, 8, &n) == VOCR_E_ARG);
    CHECK(vocr_charbox_find(&m, c1, 1, 0, 80, 32, 0, NULL, NULL, NULL, out, 8, &n) == VOCR_E_ARG);
    CHECK(vocr_charbox_find(&m, c1, 1, 10, 0, 32, 0, NULL, NULL, NULL, out, 8, &n) == VOCR_E_ARG);
    CHECK(vocr_charbox_find(&m, c1, 1, 10, 80, 0, 0, NULL, NULL, NULL, out, 8, &n) == VOCR_E_ARG);
    CHECK(vocr_charbox_find(&m, c1, 1, 10, 80, 32, 0, NULL, NULL, NULL, out, 0, &n) == VOCR_E_ARG);
    CHECK(vocr_charbox_find(&m, c1, 1, 10, 80, 32, 0, NULL, NULL, NULL, out, VOCR_CHARBOX_MAX + 1, &n) == VOCR_E_ARG);
    CHECK(vocr_charbox_find(&m, c1, 1, 10, 80, 32, 0, NULL, NULL, NULL, out, 8, NULL) == VOCR_E_ARG);
    CHECK(vocr_charbox_find(&m, c1, 1, 10, 80, 32, 8, NULL, NULL, NULL, out, 8, &n) == VOCR_E_ARG);
    CHECK(vocr_charbox_find(&m, c1, 1, 10, 80, 32, VOCR_CHARBOX_INK, NULL, NULL, NULL, out, 8, &n) == VOCR_E_ARG);
    CHECK(vocr_charbox_find(&m, c1, 1, 10, 80, 32, VOCR_CHARBOX_INK, NULL, &im, NULL, out, 8, &n) == VOCR_E_ARG);
    {
        static const uint16_t bad[] = {NTINY};
        CHECK(vocr_charbox_find(&m, bad, 1, 1, 80, 32, 0, NULL, NULL, NULL, out, 8, &n) == VOCR_E_ARG && n == 0);
    }
    CHECK(vocr_charbox_profile(NULL, prof) == VOCR_E_ARG && vocr_charbox_profile(&im, NULL) == VOCR_E_ARG);
    /* align */
    {
        static const uint32_t s[] = {'a', 'b', 'c', 'd', 'e'};
        static const uint32_t d1[] = {'a', 'b', 'x', 'd', 'e'};     /* a substitution */
        static const uint32_t d2[] = {'a', 'b', 'x', 'c', 'd', 'e'}; /* an insertion */
        static const uint32_t d3[] = {'a', 'b', 'd', 'e'};           /* a deletion */
        int16_t map[8];
        CHECK(vocr_charbox_align(s, 5, d1, 5, map) == 0);
        CHECK(map[0] == 0 && map[1] == 1 && map[2] == 2 && map[3] == 3 && map[4] == 4);
        CHECK(vocr_charbox_align(s, 5, d2, 6, map) == 0);
        CHECK(map[0] == 0 && map[1] == 1 && map[2] == -1 && map[3] == 2 && map[4] == 3 && map[5] == 4);
        CHECK(vocr_charbox_align(s, 5, d3, 4, map) == 0);
        CHECK(map[0] == 0 && map[1] == 1 && map[2] == 3 && map[3] == 4);
        CHECK(vocr_charbox_align(s, 0, d1, 5, map) == 0 && map[0] == -1 && map[4] == -1);
        CHECK(vocr_charbox_align(NULL, 5, d1, 5, map) == VOCR_E_ARG);
        CHECK(vocr_charbox_align(s, 5, d1, 5, NULL) == VOCR_E_ARG);
    }
}

/* ------------------------------------------------------------- the parity driver */

static int rd(FILE *f, void *p, size_t n) { return fread(p, 1, n, f) == n; }

static int u32(FILE *f, uint32_t *v) { return rd(f, v, 4); }

static int parity(const char *vocab_path, const char *cases_path)
{
    static char voc[1 << 20];
    static uint16_t off[1 << 16];
    vocr_model m;
    FILE *fv = fopen(vocab_path, "rb"), *fc = fopen(cases_path, "rb");
    size_t vb, i, used = 0;
    unsigned ncls = 1;
    uint32_t count, k;
    if (!fv || !fc) return 2;
    vb = fread(voc, 1, sizeof voc - 1, fv);
    fclose(fv);
    off[0] = 0;
    off[1] = 0; /* class 0: "" */
    for (i = 0; i < vb && ncls < 65535; i++) {
        if (voc[i] == '\n') {
            off[ncls + 1] = (uint16_t)used;
            ncls++;
            continue;
        }
        voc[used++] = voc[i];
        off[ncls + 1] = (uint16_t)used;
    }
    if (vb && voc[vb - 1] != '\n') ncls++;
    model_init(&m, off, voc, ncls);
    if (!u32(fc, &count)) return 2;
    for (k = 0; k < count; k++) {
        uint32_t kind;
        if (!u32(fc, &kind)) return 2;
        if (kind == 0) {
            uint32_t frames, stride, w, h, flags, has_cal, cap, ink, j;
            int16_t calv[9];
            uint16_t *cls;
            uint32_t *prof = NULL;
            unsigned char *pix = NULL;
            vocr_charbox_cal cal;
            vocr_image im;
            static vocr_charbox out[VOCR_CHARBOX_MAX];
            unsigned n = 0;
            int rc;
            if (!u32(fc, &frames) || !u32(fc, &stride)) return 2;
            cls = (uint16_t *)malloc((size_t)frames * stride * 2 + 2);
            if (!cls || !rd(fc, cls, (size_t)frames * stride * 2)) return 2;
            if (!u32(fc, &w) || !u32(fc, &h) || !u32(fc, &flags) || !u32(fc, &has_cal) || !rd(fc, calv, 18) ||
                !u32(fc, &cap) || !u32(fc, &ink))
                return 2;
            cal.shift = calv[0];
            for (j = 0; j < 8; j++) cal.group[j] = calv[1 + j];
            if (ink == 1) {
                prof = (uint32_t *)malloc((size_t)w * 4 + 4);
                if (!prof || !rd(fc, prof, (size_t)w * 4)) return 2;
            } else if (ink == 2 || ink == 3) {
                const size_t bpp = ink == 2 ? 1 : 4;
                pix = (unsigned char *)malloc((size_t)w * h * bpp + 1);
                if (!pix || !rd(fc, pix, (size_t)w * h * bpp)) return 2;
                im.pixels = pix;
                im.width = w;
                im.height = h;
                im.stride = (size_t)w * bpp;
                im.format = ink == 2 ? VOCR_GREY8 : VOCR_RGBA8;
            }
            rc = vocr_charbox_find(&m, cls, stride, frames, w, h, flags, has_cal ? &cal : NULL, pix ? &im : NULL,
                                   prof, out, cap, &n);
            printf("F %d %u", rc, n);
            for (j = 0; j < n; j++)
                printf(" %u %u %u %u %ld %ld %ld %ld %ld", out[j].cls, out[j].f0, out[j].f1, out[j].group,
                       (long)out[j].c, (long)out[j].x0, (long)out[j].x1, (long)out[j].i0, (long)out[j].i1);
            printf("\n");
            free(cls);
            free(prof);
            free(pix);
        } else if (kind == 1) {
            uint32_t n, mm, j;
            uint32_t *src, *dst;
            int16_t *map;
            int rc;
            if (!u32(fc, &n) || !u32(fc, &mm)) return 2;
            src = (uint32_t *)malloc((size_t)n * 4 + 4);
            dst = (uint32_t *)malloc((size_t)mm * 4 + 4);
            map = (int16_t *)malloc((size_t)mm * 2 + 2);
            if (!src || !dst || !map || !rd(fc, src, (size_t)n * 4) || !rd(fc, dst, (size_t)mm * 4)) return 2;
            rc = vocr_charbox_align(src, n, dst, mm, map);
            printf("A %d", rc);
            for (j = 0; j < mm; j++) printf(" %d", map[j]);
            printf("\n");
            free(src);
            free(dst);
            free(map);
        } else {
            return 2;
        }
    }
    fclose(fc);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc == 4 && !strcmp(argv[1], "--parity")) return parity(argv[2], argv[3]);
    if (argc != 1) {
        fprintf(stderr, "usage: vocr_charbox_test [--parity VOCAB CASES]\n");
        return 2;
    }
    selftest();
    if (failures) {
        fprintf(stderr, "%d checks failed\n", failures);
        return 1;
    }
    printf("CHARBOX_SELFTEST_OK\n");
    return 0;
}
