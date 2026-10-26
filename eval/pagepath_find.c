/* eval/pagepath_find.c - the line boxes Rubi reads from a region, for eval/pagepath.py (issue #129).
 *
 * Rubi's game worker (bee-san/Rubi_for_vita main bd130a8, core/vocr_ocr.c vjo_vocr_ocr_policy) cuts a region's
 * lines in three steps, reproduced here on a grey region held in memory:
 *   1. the finder: vocr_find_lines(finder) (Rubi: VOCR_LINES_TEXTBOX_RECOVERY over the row form, which equals the
 *      in-memory form bit for bit: runtime/tests/test_vocr_lines_rows.c);
 *   2. probe_short_lines (#194, policy.short_lines = 1): a short last line under the bottom line that no finder
 *      mode keeps. A port of Rubi's function: the same arithmetic, with the strip windows of Rubi's capture
 *      (VJO_VOCR_STRIP_BYTES 56 KiB of the region's width, at most 64 rows: vjo_vocr_strip_rows, stream_fetch);
 *   3. the refinement (ocr_refine_lines = psp: VOCR_REFINE_ALL on PSP titles), each box on its own, with Rubi's
 *      cap: a box's lines may take the job to 16 lines, one kept for each box still to come; past that the box is
 *      refined without SPLIT.
 * Built with -DPAGEPATH_FIX, step 1 is vocr_find_lines_fix(finder, fix) (runtime/vocr_lines_fix.h) instead.
 *
 * Batch input: records of eight little-endian u32 (width, height, finder, fix, probe, refine flags, 0, 0) and then
 * width x height grey bytes. Output: one line per record, "rc found probed n" then n times "x y w h dark".
 * Provenance and licence: probe_short_lines below is adapted from bee-san/Rubi_for_vita core/vocr_ocr.c at bd130a8,
 * where the function entered in #194 (49310ed, 2026-10-17, while Rubi's LICENSE was the GPL-3.0 text). At bd130a8
 * ("Update LICENSE", 2026-10-26) Rubi's LICENSE file is the EUPL v1.2 (its README still says GPL-3.0). Both
 * repositories have the same owner (bee-san), who publishes this one under GPL-3.0-or-later; the EUPL v1.2's
 * compatibility clause (Article 5 and its appendix) also lists the GPL v3. The owner may want to confirm.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vocr_lines.h"
#include "vocr_lines_refine.h"
#ifdef PAGEPATH_FIX
#include "vocr_lines_fix.h"
#endif

#define MAX_LINES 16u          /* VJO_VOCR_MAX_LINES */
#define STRIP_BYTES 57344u     /* VJO_VOCR_STRIP_BYTES (56 KiB) */
#define STRIP_MAX_ROWS 64u     /* VJO_VOCR_STRIP_MAX_ROWS */
#define MAX_WIDTH 960u         /* VJO_OCR_MAX_WIDTH */
#define PROBE_PAD 3u

typedef struct {
    const unsigned char *px; /* the whole region, w x h grey */
    unsigned w, h, strip_rows, y0, rows;
} Stream;

static unsigned strip_rows_of(unsigned w, unsigned h)
{
    unsigned rows = STRIP_BYTES / w;
    if (rows > STRIP_MAX_ROWS) rows = STRIP_MAX_ROWS;
    if (rows > h) rows = h;
    return rows ? rows : 1u;
}

/* stream_fetch after the finder's passes (first_next == h): a strip of strip_rows rows from y, moved up at the
 * region's bottom. */
static void stream_fetch(Stream *s, unsigned y)
{
    unsigned n = s->strip_rows;
    if (y + n > s->h) y = s->h - n;
    s->y0 = y;
    s->rows = n;
}

static unsigned probe_contrast(const unsigned char *r, size_t stride, unsigned x)
{
    unsigned lo = 255, hi = 0;
    for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++) {
            unsigned v = r[(ptrdiff_t)dy * (ptrdiff_t)stride + (ptrdiff_t)x + dx];
            if (v < lo) lo = v;
            if (v > hi) hi = v;
        }
    return hi - lo;
}

static unsigned probe_short_lines(Stream *s, VocrLineSet *set)
{
    uint16_t hist[256];
    uint8_t cols[MAX_WIDTH];
    uint16_t rowc[64];
    unsigned added = 0;
    while (set->count && set->count < MAX_LINES) {
        const VocrLine *prev = &set->lines[0];
        for (unsigned i = 1; i < set->count; i++)
            if (set->lines[i].y + set->lines[i].h > prev->y + prev->h) prev = &set->lines[i];
        unsigned lh = prev->h, ys = prev->y + prev->h, ye, threshold, total = 0, seen = 0;
        if (lh < 4u * PROBE_PAD || ys + lh * 45u / 100u >= s->h || s->w < 3) break;
        ye = ys + lh + lh / 2u;
        if (ye > s->h - 1u) ye = s->h - 1u;
        stream_fetch(s, prev->y);
        memset(hist, 0, sizeof(hist));
        for (unsigned y = prev->y + 1u; y + 1u < prev->y + prev->h && y + 1u < s->y0 + s->rows; y++) {
            if (y < s->y0 + 1u) continue;
            const unsigned char *r = s->px + (size_t)y * s->w;
            for (unsigned x = prev->x + 1u; x + 1u < prev->x + prev->w && x + 1u < s->w; x++) {
                unsigned c = probe_contrast(r, s->w, x);
                if (hist[c] < 0xFFFFu) { hist[c]++; total++; }
            }
        }
        if (!total) break;
        threshold = 255;
        for (unsigned v = 0; v < 256; v++) {
            seen += hist[v];
            if (seen * 100u >= total * 99u) { threshold = v; break; }
        }
        threshold = threshold * 2u / 5u;
        if (threshold < 40u) threshold = 40u;
        stream_fetch(s, ys - 1u);
        if (ye > s->y0 + s->rows - 1u) ye = s->y0 + s->rows - 1u;
        if (ye <= ys || ys < s->y0 + 1u || ye - ys > sizeof(rowc) / sizeof(rowc[0])) break;
        unsigned xa = prev->x > lh ? prev->x - lh : 1u, top = 0, bot = 0, inked = 0, x1 = 0, x2 = 0;
        if (xa < 1u) xa = 1u;
        memset(cols, 0, sizeof(cols));
        for (unsigned y = ys; y < ye; y++) {
            const unsigned char *r = s->px + (size_t)y * s->w;
            unsigned n = 0;
            for (unsigned x = xa; x + 1u < s->w; x++)
                if (probe_contrast(r, s->w, x) >= threshold) {
                    n++;
                    if (cols[x] < 255u) cols[x]++;
                }
            rowc[y - ys] = (uint16_t)(n > 0xFFFFu ? 0xFFFFu : n);
            if (n >= 2u) {
                if (!inked) top = y;
                bot = y;
                inked = 1;
            }
        }
        if (!inked) break;
        unsigned gh = bot - top + 1u, glyph = lh - 2u * PROBE_PAD;
        if (gh * 100u < glyph * 45u || gh * 10u > lh * 13u) break; /* not a line's height */
        for (x1 = xa; x1 + 1u < s->w && cols[x1] < 2u; x1++) {}
        if (x1 + 1u >= s->w || x1 > prev->x + lh * 3u / 2u) break; /* not where the line above starts */
        x2 = x1;
        for (unsigned x = x1, gap = 0; x + 1u < s->w; x++) {
            if (cols[x] >= 2u) { x2 = x; gap = 0; }
            else if (++gap > lh) break; /* past a glyph's width of nothing: the line ends */
        }
        if ((x2 - x1 + 1u) * 2u < glyph) break; /* not a glyph's width */
        VocrLine *ln = &set->lines[set->count];
        ln->x = x1 > PROBE_PAD ? x1 - PROBE_PAD : 0;
        ln->y = top > ys + PROBE_PAD ? top - PROBE_PAD : ys;
        ln->w = x2 + 1u + PROBE_PAD - ln->x;
        if (ln->x + ln->w > s->w) ln->w = s->w - ln->x;
        ln->h = bot + 1u + PROBE_PAD - ln->y;
        if (ln->y + ln->h > s->h) ln->h = s->h - ln->y;
        ln->dark = prev->dark;
        set->count++;
        added++;
    }
    (void)rowc;
    return added;
}

static unsigned rd32(const unsigned char *p)
{
    return (unsigned)p[0] | (unsigned)p[1] << 8 | (unsigned)p[2] << 16 | (unsigned)p[3] << 24;
}

static unsigned char *load(const char *path, size_t *n)
{
    FILE *f = fopen(path, "rb");
    long len;
    unsigned char *buf;
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    len = ftell(f);
    fseek(f, 0, SEEK_SET);
    buf = (unsigned char *)malloc((size_t)len + 1u);
    if (!buf || fread(buf, 1, (size_t)len, f) != (size_t)len) { fclose(f); free(buf); return NULL; }
    fclose(f);
    *n = (size_t)len;
    return buf;
}

int main(int argc, char **argv)
{
    size_t n = 0, off = 0;
    unsigned char *buf;
    if (argc != 2) {
        fprintf(stderr, "usage: %s BATCH\n", argv[0]);
        return 2;
    }
    buf = load(argv[1], &n);
    if (!buf) { fprintf(stderr, "cannot read %s\n", argv[1]); return 2; }
    while (off + 32u <= n) {
        unsigned w = rd32(buf + off), h = rd32(buf + off + 4), finder = rd32(buf + off + 8);
        unsigned fix = rd32(buf + off + 12), probe = rd32(buf + off + 16), rflags = rd32(buf + off + 20);
        size_t need, rneed;
        unsigned char *arena;
        VocrLinesImage im;
        VocrLineSet set, out;
        unsigned found = 0, probed = 0, li = 0, i;
        int rc;
        off += 32u;
        if (!w || !h || off + (size_t)w * h > n) { fprintf(stderr, "truncated batch\n"); free(buf); return 2; }
        im.pixels = buf + off; im.width = w; im.height = h; im.stride = w; im.bpp = VOCR_LINES_GREY;
        need = vocr_lines_arena_bytes(w, h);
        rneed = vocr_refine_arena_bytes(w, h);
        if (rneed > need) need = rneed;
        arena = (unsigned char *)malloc(need + 16u);
        if (!arena) { free(buf); return 2; }
#ifdef PAGEPATH_FIX
        rc = vocr_find_lines_fix(&im, (int)finder, fix, arena, need, &set);
#else
        (void)fix;
        rc = vocr_find_lines(&im, (int)finder, arena, need, &set);
#endif
        found = rc ? 0u : set.count;
        if (!rc && probe && !set.truncated && w <= MAX_WIDTH) {
            Stream s = {buf + off, w, h, strip_rows_of(w, h), 0, 0};
            probed = probe_short_lines(&s, &set);
        }
        out.count = 0;
        out.truncated = set.truncated;
        for (i = 0; !rc && i < set.count; i++) {
            const VocrLine *box = &set.lines[i];
            VocrLineSet one;
            unsigned flags = rflags, room, k;
            int rrc = 0;
            one.count = 1; one.truncated = 0; one.lines[0] = *box;
            if (rflags) {
                unsigned long rows = 0;
                room = MAX_LINES - (li < MAX_LINES ? li : MAX_LINES);
                room = room > set.count - i - 1u ? room - (set.count - i - 1u) : 1u;
                for (;;) {
                    one.count = 1; one.truncated = 0; one.lines[0] = *box;
                    rrc = vocr_refine_lines(&im, flags, arena, need, &one, &rows);
                    if (rrc || one.count <= room || !(flags & VOCR_REFINE_SPLIT)) break;
                    flags &= ~(unsigned)VOCR_REFINE_SPLIT;
                }
                if (rrc) { one.count = 1; one.lines[0] = *box; }
            }
            /* Rubi's room rule keeps the job within MAX_LINES lines in all */
            for (k = 0; k < one.count && out.count < MAX_LINES; k++, li++) out.lines[out.count++] = one.lines[k];
        }
        printf("%d %u %u %u", rc, found, probed, rc ? 0u : out.count);
        for (i = 0; !rc && i < out.count; i++)
            printf(" %u %u %u %u %u", out.lines[i].x, out.lines[i].y, out.lines[i].w, out.lines[i].h,
                   out.lines[i].dark);
        printf("\n");
        free(arena);
        off += (size_t)w * h;
    }
    free(buf);
    return 0;
}
