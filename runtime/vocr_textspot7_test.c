/* Batch driver of vocr_textspot7.c for the C == twin checks (eval/textspot7.py parity, tests/test_textspot7.py).
 *
 *   vocr_textspot7_test BATCH
 *
 * BATCH: a sequence of little-endian int32 records:
 *   kind 1 (choose): mode tau lam kappa ns ng, then ns x (x y w h conf), then ng x (x y w h)
 *       -> prints "C <0|1>" (0 the spotter's, 1 GRAD's), or "C E<rc>"
 *   kind 2 (post): steps x0 y0 x1 y1 n, then n x (x y w h conf dark), then (only with EXTEND or TIGHTEN) the
 *       region's (x1 - x0) x (y1 - y0) grey bytes, padded to a multiple of 4
 *       -> prints "P <m> x y w h conf dark ..." or "P E<rc>"
 *   kind 3 (post_rows): kind 2's record, run through vocr_textspot7_post_rows with a row callback over the bytes
 *       (it also checks that every pass asks for rows in increasing order) -> the same line as kind 2, or "P E<rc>"
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vocr_textspot7.h"

typedef struct {
    const unsigned char *px;
    unsigned w;
    int last, back;     /* back: requests for a row below the previous one (a new pass) */
} RowsSrc;

static const unsigned char *row_cb(void *user, unsigned y)
{
    RowsSrc *r = (RowsSrc *)user;
    if ((int)y <= r->last)
        r->back++;
    r->last = (int)y;
    return r->px + (size_t)y * r->w;
}

static int rd(FILE *f, int *v, unsigned n)
{
    unsigned i;
    unsigned char b[4];
    for (i = 0; i < n; i++) {
        if (fread(b, 1, 4, f) != 4)
            return -1;
        v[i] = (int)((unsigned)b[0] | ((unsigned)b[1] << 8) | ((unsigned)b[2] << 16) | ((unsigned)b[3] << 24));
    }
    return 0;
}

int main(int argc, char **argv)
{
    FILE *f;
    static unsigned char pix[960 * 544 * 2], arena[1 << 21];
    int kind;
    if (argc != 2 || !(f = fopen(argv[1], "rb"))) {
        fprintf(stderr, "usage: vocr_textspot7_test BATCH\n");
        return 2;
    }
    while (rd(f, &kind, 1) == 0) {
        if (kind == 1) {
            int hd[6], v[5];
            VocrT7Box s[VOCR_T7_MAX_BOXES], g[VOCR_T7_MAX_BOXES];
            VocrT7Params p;
            int i, rc;
            if (rd(f, hd, 6) || hd[4] < 0 || hd[5] < 0 || hd[4] > (int)VOCR_T7_MAX_BOXES ||
                hd[5] > (int)VOCR_T7_MAX_BOXES)
                return 3;
            p.mode = (unsigned)hd[0];
            p.tau = hd[1];
            p.lam = hd[2];
            p.kappa = hd[3];
            for (i = 0; i < hd[4]; i++) {
                if (rd(f, v, 5))
                    return 3;
                s[i].x = v[0]; s[i].y = v[1]; s[i].w = v[2]; s[i].h = v[3]; s[i].conf = v[4]; s[i].dark = 0;
            }
            for (i = 0; i < hd[5]; i++) {
                if (rd(f, v, 4))
                    return 3;
                g[i].x = v[0]; g[i].y = v[1]; g[i].w = v[2]; g[i].h = v[3]; g[i].conf = 0; g[i].dark = 0;
            }
            rc = vocr_textspot7_choose(s, (unsigned)hd[4], g, (unsigned)hd[5], &p);
            if (rc < 0)
                printf("C E%d\n", rc);
            else
                printf("C %d\n", rc);
        } else if (kind == 2 || kind == 3) {
            int hd[6], v[6], i, rc, W, H;
            VocrT7Box fb[VOCR_T7_MAX_BOXES], out[VOCR_T7_MAX_BOXES];
            unsigned m = 0, steps;
            VocrLinesImage im;
            if (rd(f, hd, 6) || hd[5] < 0 || hd[5] > (int)VOCR_T7_MAX_BOXES)
                return 3;
            steps = (unsigned)hd[0];
            W = hd[3] - hd[1];
            H = hd[4] - hd[2];
            for (i = 0; i < hd[5]; i++) {
                if (rd(f, v, 6))
                    return 3;
                fb[i].x = v[0]; fb[i].y = v[1]; fb[i].w = v[2]; fb[i].h = v[3]; fb[i].conf = v[4];
                fb[i].dark = (unsigned)v[5];
            }
            memset(&im, 0, sizeof im);
            if (steps & (VOCR_T7_EXTEND | VOCR_T7_TIGHTEN)) {
                size_t nb = (size_t)W * (size_t)H, padded = (nb + 3u) & ~(size_t)3u;
                if (W <= 0 || H <= 0 || padded > sizeof pix || fread(pix, 1, padded, f) != padded)
                    return 3;
                im.pixels = pix;
                im.width = (unsigned)W;
                im.height = (unsigned)H;
                im.stride = (size_t)W;
                im.bpp = VOCR_LINES_GREY;
            }
            if (kind == 2) {
                rc = vocr_textspot7_post(fb, (unsigned)hd[5], hd[1], hd[2], hd[3], hd[4],
                                         (steps & (VOCR_T7_EXTEND | VOCR_T7_TIGHTEN)) ? &im : NULL, steps,
                                         arena, sizeof arena, out, &m);
            } else {
                RowsSrc rs = {pix, (unsigned)(W > 0 ? W : 1), -1, 0};
                VocrLinesRows src = {(unsigned)(W > 0 ? W : 0), (unsigned)(H > 0 ? H : 0), VOCR_LINES_GREY, row_cb,
                                     &rs};
                rc = vocr_textspot7_post_rows(fb, (unsigned)hd[5], hd[1], hd[2], hd[3], hd[4],
                                              (steps & (VOCR_T7_EXTEND | VOCR_T7_TIGHTEN)) ? &src : NULL, steps,
                                              arena, sizeof arena, out, &m);
            }
            if (rc < 0) {
                printf("P E%d\n", rc);
                continue;
            }
            printf("P %u", m);
            for (i = 0; i < (int)m; i++)
                printf(" %d %d %d %d %d %u", out[i].x, out[i].y, out[i].w, out[i].h, out[i].conf, out[i].dark);
            printf("\n");
        } else if (kind == 4) {   /* arena sizes: w h steps -> "A <image form> <rows form>" */
            int v[3];
            if (rd(f, v, 3))
                return 3;
            printf("A %zu %zu\n", vocr_textspot7_arena_bytes_steps((unsigned)v[0], (unsigned)v[1], (unsigned)v[2]),
                   vocr_textspot7_rows_arena_bytes_steps((unsigned)v[0], (unsigned)v[1], (unsigned)v[2]));
        } else {
            return 4;
        }
    }
    fclose(f);
    return 0;
}
