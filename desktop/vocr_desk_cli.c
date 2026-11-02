/* vocr_desk_cli.c - the desktop page pipeline from the command line (vocr_desk.h; docs/desktop.md).
 *
 *   vocr_desk_cli DET.vdsk REC.vocr FRAMES.bin [--threads N] [--size fit:960|long:N|native] [--reps R]
 *                 [--out OUT.jsonl] [--quiet]
 *
 * FRAMES.bin: u32 count, then per frame u32 width, u32 height and width x height grey bytes (row-major). The frames
 * are read one at a time, so the process holds one frame. Per frame: one untimed run, then R timed runs; prints
 * "frame W H boxes lines detect_us page_us" (the medians). --out writes one JSON line per frame: its lines in reading
 * order with box, tag, dark, vertical flag and text. At the end: "peak_rss_kib N" (getrusage) and the totals.
 */
#define _POSIX_C_SOURCE 200112L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>

#include "vocr_desk.h"

static unsigned char *load(const char *path, size_t *n)
{
    FILE *f = fopen(path, "rb");
    unsigned char *b = NULL;
    long sz;
    void *p = NULL;
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) == 0 && (sz = ftell(f)) > 0 && fseek(f, 0, SEEK_SET) == 0 &&
        !posix_memalign(&p, 64, (size_t)sz)) {
        b = (unsigned char *)p;
        if (fread(b, 1, (size_t)sz, f) != (size_t)sz) {
            free(b);
            b = NULL;
        }
        *n = (size_t)sz;
    }
    fclose(f);
    return b;
}

/* a policy's n: a decimal number of 1-8192, or 0 (refused by vocr_desk_page_new) for anything else */
static unsigned size_n(const char *s)
{
    char *end;
    unsigned long v = strtoul(s, &end, 10);
    return (*s && !*end && v >= 1 && v <= VOCR_DESK_MAX_SIDE) ? (unsigned)v : 0u;
}

static unsigned rd32(const unsigned char *p)
{
    return (unsigned)p[0] | (unsigned)p[1] << 8 | (unsigned)p[2] << 16 | (unsigned)p[3] << 24;
}

static int cmpd(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

static double median(double *v, unsigned n)
{
    qsort(v, n, sizeof *v, cmpd);
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

static void json_str(FILE *f, const char *s)
{
    fputc('"', f);
    for (; *s; s++) {
        unsigned char c = (unsigned char)*s;
        if (c == '"' || c == '\\') fprintf(f, "\\%c", c);
        else if (c < 0x20) fprintf(f, "\\u%04x", c);
        else fputc(c, f);
    }
    fputc('"', f);
}

int main(int argc, char **argv)
{
    size_t dn = 0, rn = 0;
    unsigned char *det, *rec, hdr[8], *frame = NULL;
    size_t cap = 0;
    unsigned threads = 1, reps = 3, count, i, r, n = 0, quiet = 0;
    int kind = -1, err = 0, a;
    const char *out_path = NULL;
    FILE *fr, *out = NULL;
    vocr_desk_page *p;
    double *td, *tp, sum_det = 0, sum_page = 0;
    struct rusage ru;
    if (argc < 4) {
        fprintf(stderr, "usage: %s DET.vdsk REC.vocr FRAMES.bin [--threads N] [--size fit:960|long:N|native] "
                        "[--reps R] [--out OUT.jsonl] [--quiet]\n", argv[0]);
        return 2;
    }
    for (a = 4; a < argc; a++) {
        if (!strcmp(argv[a], "--threads") && a + 1 < argc) threads = (unsigned)atoi(argv[++a]);
        else if (!strcmp(argv[a], "--reps") && a + 1 < argc) reps = (unsigned)atoi(argv[++a]);
        else if (!strcmp(argv[a], "--out") && a + 1 < argc) out_path = argv[++a];
        else if (!strcmp(argv[a], "--quiet")) quiet = 1;
        else if (!strcmp(argv[a], "--size") && a + 1 < argc) {
            const char *s = argv[++a];
            if (!strcmp(s, "native")) kind = VOCR_DESK_SIZE_NATIVE, n = 0;
            else if (!strncmp(s, "fit:", 4)) kind = VOCR_DESK_SIZE_FIT, n = size_n(s + 4);
            else if (!strncmp(s, "long:", 5)) kind = VOCR_DESK_SIZE_LONG, n = size_n(s + 5);
            else {
                fprintf(stderr, "--size: fit:N, long:N or native\n");
                return 2;
            }
        } else {
            fprintf(stderr, "unknown argument %s\n", argv[a]);
            return 2;
        }
    }
    if (reps < 1) reps = 1;
    det = load(argv[1], &dn);
    rec = load(argv[2], &rn);
    fr = fopen(argv[3], "rb");
    if (!det || !rec || !fr || fread(hdr, 1, 4, fr) != 4) {
        fprintf(stderr, "cannot read the inputs\n");
        return 2;
    }
    p = vocr_desk_page_new(det, dn, rec, rn, threads, kind, n, &err);
    if (!p) {
        fprintf(stderr, "vocr_desk_page_new: %d\n", err);
        return 1;
    }
    if (out_path) out = fopen(out_path, "w");
    td = (double *)malloc(reps * sizeof *td);
    tp = (double *)malloc(reps * sizeof *tp);
    count = rd32(hdr);
    for (i = 0; i < count; i++) {
        unsigned w, h, k;
        const vocr_desk_set *bs;
        if (fread(hdr, 1, 8, fr) != 8) return 1;
        w = rd32(hdr);
        h = rd32(hdr + 4);
        if ((size_t)w * h > cap) {
            free(frame);
            cap = (size_t)w * h;
            frame = (unsigned char *)malloc(cap);
            if (!frame) return 1;
        }
        if (fread(frame, 1, (size_t)w * h, fr) != (size_t)w * h) return 1;
        if ((err = vocr_desk_page_run(p, frame, w, h, w))) {
            fprintf(stderr, "frame %u: %d\n", i, err);
            return 1;
        }
        for (r = 0; r < reps; r++) {
            vocr_desk_page_run(p, frame, w, h, w);
            td[r] = vocr_desk_page_detect_us(p);
            tp[r] = vocr_desk_page_total_us(p);
        }
        bs = vocr_desk_page_boxes(p);
        {
            double md = median(td, reps), mp = median(tp, reps);
            sum_det += md;
            sum_page += mp;
            if (!quiet)
                printf("%u %u %u %u %u %.1f %.1f\n", i, w, h, bs->count, vocr_desk_page_count(p), md, mp);
        }
        if (out) {
            fprintf(out, "{\"frame\": %u, \"w\": %u, \"h\": %u, \"truncated\": %u, \"lines\": [", i, w, h,
                    bs->truncated);
            for (k = 0; k < vocr_desk_page_count(p); k++) {
                const vocr_desk_line *ln = vocr_desk_page_line(p, k);
                fprintf(out, "%s{\"box\": [%d, %d, %d, %d], \"tag\": %u, \"dark\": %u, \"vertical\": %u, \"text\": ",
                        k ? ", " : "", ln->x, ln->y, ln->w, ln->h, ln->tag, ln->dark, ln->vertical);
                json_str(out, ln->text);
                fputc('}', out);
            }
            fprintf(out, "]}\n");
        }
    }
    getrusage(RUSAGE_SELF, &ru);
    printf("frames %u mean_detect_us %.1f mean_page_us %.1f threads %u\n", count, count ? sum_det / count : 0.0,
           count ? sum_page / count : 0.0, threads);
    printf("peak_rss_kib %ld\n", (long)ru.ru_maxrss);
    if (out) fclose(out);
    vocr_desk_page_free(p);
    fclose(fr);
    free(frame);
    free(det);
    free(rec);
    free(td);
    free(tp);
    return 0;
}
