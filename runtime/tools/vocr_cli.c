/* vocr_cli - host / qemu driver for the vocr runtime (not part of the console build).
 *
 *   vocr_cli MODEL.vocr [opts] IMAGE.pgm|ppm...  one text line per image on stdout
 *   vocr_cli MODEL.vocr [opts] --stdin           image paths on stdin, one per line
 *   vocr_cli MODEL.vocr [opts] --pack LINES.bin  many crops in one file
 *
 * Images are binary PGM (P5, grey) or PPM (P6, RGB, run through the RGBA path), maxval 255.
 * --stdin fits eval/run.py: --model 'cmd:runtime/build/vocr_cli MODEL.vocr --stdin'.
 *
 * opts: --mode auto|f32|w8f|w8i  --tile N  --head N  --max-width N
 *       --invert           read every crop as 255 - grey (VOCR_INVERT: dark text)
 *       --logits OUT.bin   per line: u32 frames, u32 classes, float32 logits
 *       --input-out OUT.bin  per line: u32 height, u32 padded width, u32 width, then the
 *                          preprocessed grey input (height x padded width bytes)
 *       --repeat N         run every line N times (timing)
 *       --arena W[,W...]   print arena bytes for these max widths and exit
 *       --crc              verify the blob CRC first
 * A pack file is u32 count, then per crop u32 height, u32 width, u32 channels (1 grey,
 * 4 RGBA), then height*width*channels bytes.
 * Timing (wall clock, after the first run) goes to stderr.
 */
#define _POSIX_C_SOURCE 200112L
#include "../vocr.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int invert_flag; /* VOCR_INVERT or 0, from --invert, for every crop */

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static void *read_file(const char *path, size_t *n, size_t align)
{
    FILE *f = fopen(path, "rb");
    void *buf = NULL;
    long len;
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) || (len = ftell(f)) < 0 || fseek(f, 0, SEEK_SET)) {
        fclose(f);
        return NULL;
    }
    if (align > 1) {
        size_t sz = ((size_t)len + align - 1) / align * align;
        if (posix_memalign(&buf, align, sz ? sz : align)) buf = NULL;
    } else {
        buf = malloc((size_t)len ? (size_t)len : 1u);
    }
    if (buf && fread(buf, 1, (size_t)len, f) != (size_t)len) {
        free(buf);
        buf = NULL;
    }
    fclose(f);
    *n = (size_t)len;
    return buf;
}

static unsigned rd32(const unsigned char *p)
{
    return (unsigned)p[0] | (unsigned)p[1] << 8 | (unsigned)p[2] << 16 | (unsigned)p[3] << 24;
}

/* Binary PGM (P5) or PPM (P6), maxval 255. Returns pixels inside buf; *ch = 1 or 3. */
static unsigned char *parse_pnm(unsigned char *buf, size_t n, unsigned *w, unsigned *h, unsigned *ch)
{
    unsigned v[3];
    size_t i = 2;
    int k;
    if (n < 3 || buf[0] != 'P' || (buf[1] != '5' && buf[1] != '6')) return NULL;
    *ch = buf[1] == '5' ? 1u : 3u;
    for (k = 0; k < 3; k++) {
        while (i < n && (buf[i] == ' ' || buf[i] == '\n' || buf[i] == '\r' || buf[i] == '\t' || buf[i] == '#')) {
            if (buf[i] == '#')
                while (i < n && buf[i] != '\n') i++;
            else
                i++;
        }
        v[k] = 0;
        while (i < n && buf[i] >= '0' && buf[i] <= '9' && v[k] < 1000000u)
            v[k] = v[k] * 10u + (unsigned)(buf[i++] - '0');
    }
    if (i >= n || v[2] != 255 || v[0] > 65535u || v[1] > 65535u) return NULL;
    i++;
    *w = v[0];
    *h = v[1];
    if (!*w || !*h || (n - i) / *ch < (size_t)*w * *h) return NULL;
    return buf + i;
}

static int run_line(vocr_ctx *ctx, const vocr_model *m, const unsigned char *px, unsigned w,
                    unsigned h, unsigned ch, int repeat, FILE *logits_out, FILE *input_out,
                    double *t_total, unsigned long *frames)
{
    vocr_image img;
    vocr_extra ex;
    char text[4096];
    float *logits = NULL;
    int rc = 0, r;
    unsigned cap = (unsigned)vocr_resized_width(m, w, h) / m->stride + 1u;
    img.pixels = px;
    img.width = w;
    img.height = h;
    img.stride = (size_t)w * ch;
    img.format = (ch == 4 ? VOCR_RGBA8 : VOCR_GREY8) | invert_flag;
    memset(&ex, 0, sizeof ex);
    if (input_out) {
        unsigned rw = 0, pw = 0, hdr3[3];
        unsigned char *buf;
        if (vocr_preprocess(m, &img, NULL, 0, &rw, &pw)) return VOCR_E_ARG;
        buf = (unsigned char *)malloc((size_t)pw * m->height);
        if (!buf || vocr_preprocess(m, &img, buf, (size_t)pw * m->height, &rw, &pw)) {
            free(buf);
            return VOCR_E_ARG;
        }
        hdr3[0] = m->height;
        hdr3[1] = pw;
        hdr3[2] = rw;
        fwrite(hdr3, sizeof hdr3, 1, input_out);
        fwrite(buf, 1, (size_t)pw * m->height, input_out);
        free(buf);
    }
    if (logits_out) {
        logits = (float *)malloc((size_t)cap * m->n_classes * sizeof(float));
        if (!logits) return VOCR_E_ARG;
        ex.logits = logits;
        ex.cap_frames = cap;
    }
    for (r = 0; r < (repeat > 0 ? repeat : 1) + (repeat > 0 ? 1 : 0); r++) {
        double t0 = now();
        rc = vocr_run(ctx, &img, text, sizeof text, r == 0 ? &ex : NULL);
        if (r > 0 || repeat <= 0) *t_total += now() - t0;
        if (rc && rc != VOCR_E_TEXT) break;
    }
    *frames += ex.frames;
    if (rc && rc != VOCR_E_TEXT) {
        fprintf(stderr, "vocr_run: %s\n", vocr_strerror(rc));
        free(logits);
        return rc;
    }
    printf("%s\n", text);
    if (logits_out) {
        unsigned hdr[2];
        hdr[0] = ex.frames;
        hdr[1] = m->n_classes;
        fwrite(hdr, sizeof hdr, 1, logits_out);
        fwrite(logits, sizeof(float), (size_t)ex.frames * m->n_classes, logits_out);
    }
    free(logits);
    return 0;
}

static int run_pack(vocr_ctx *ctx, const vocr_model *m, const char *path, int repeat, FILE *lf,
                    FILE *inf, double *t_total, unsigned long *frames, unsigned long *lines)
{
    size_t n, off = 4;
    unsigned char *buf = (unsigned char *)read_file(path, &n, 1);
    unsigned count, j;
    int rc = 0;
    if (!buf || n < 4) {
        fprintf(stderr, "cannot read %s\n", path);
        free(buf);
        return 1;
    }
    count = rd32(buf);
    for (j = 0; j < count && !rc; j++) {
        unsigned h, w, ch;
        if (n - off < 12) break;
        h = rd32(buf + off);
        w = rd32(buf + off + 4);
        ch = rd32(buf + off + 8);
        off += 12;
        if ((ch != 1 && ch != 4) || (size_t)h * w * ch > n - off) break;
        rc = run_line(ctx, m, buf + off, w, h, ch, repeat, lf, inf, t_total, frames);
        off += (size_t)h * w * ch;
        ++*lines;
    }
    if (!rc && j != count) {
        fprintf(stderr, "truncated pack file\n");
        rc = 1;
    }
    free(buf);
    return rc ? 1 : 0;
}

static int run_image(vocr_ctx *ctx, const vocr_model *m, const char *path, int repeat, FILE *lf,
                     FILE *inf, double *t_total, unsigned long *frames, unsigned long *lines)
{
    size_t n, j;
    unsigned w = 0, h = 0, ch = 1;
    unsigned char *buf = (unsigned char *)read_file(path, &n, 1);
    unsigned char *px = buf ? parse_pnm(buf, n, &w, &h, &ch) : NULL, *rgba = NULL;
    int rc = 1;
    if (!px) {
        fprintf(stderr, "cannot read PGM/PPM %s\n", path);
    } else if (ch == 3 && !(rgba = (unsigned char *)malloc((size_t)w * h * 4u))) {
        fprintf(stderr, "out of memory\n");
    } else {
        if (rgba)
            for (j = 0; j < (size_t)w * h; j++) {
                rgba[4 * j] = px[3 * j];
                rgba[4 * j + 1] = px[3 * j + 1];
                rgba[4 * j + 2] = px[3 * j + 2];
                rgba[4 * j + 3] = 255;
            }
        if (!run_line(ctx, m, rgba ? rgba : px, w, h, rgba ? 4u : 1u, repeat, lf, inf, t_total, frames))
            rc = 0;
    }
    free(rgba);
    free(buf);
    ++*lines;
    return rc;
}

/* One image path per stdin line (trailing CR/LF stripped; empty lines skipped). */
static int run_stdin(vocr_ctx *ctx, const vocr_model *m, int repeat, FILE *lf, FILE *inf,
                     double *t_total, unsigned long *frames, unsigned long *lines)
{
    char path[4096];
    while (fgets(path, sizeof path, stdin)) {
        size_t len = strlen(path);
        while (len && (path[len - 1] == '\n' || path[len - 1] == '\r')) path[--len] = 0;
        if (!len) continue;
        if (run_image(ctx, m, path, repeat, lf, inf, t_total, frames, lines)) return 1;
        fflush(stdout);
    }
    return ferror(stdin) ? 1 : 0;
}

static int mode_of(const char *v)
{
    return !strcmp(v, "auto") ? VOCR_MODE_AUTO : !strcmp(v, "f32") ? VOCR_MODE_F32
         : !strcmp(v, "w8f") ? VOCR_MODE_W8F : !strcmp(v, "w8i") ? VOCR_MODE_W8I : -1;
}

int main(int argc, char **argv)
{
    vocr_model m;
    vocr_ctx ctx;
    vocr_opts o;
    size_t blob_n = 0, need;
    void *blob = NULL, *arena = NULL;
    const char *pack = NULL, *logits_path = NULL, *arena_list = NULL, *input_path = NULL;
    const char *mode_name = "auto";
    FILE *lf = NULL, *inf = NULL;
    int i, rc, status = 1, repeat = 0, crc = 0, use_stdin = 0;
    unsigned long lines = 0, frames = 0;
    double t_total = 0.0;
    if (argc < 2) {
        fprintf(stderr, "usage: %s MODEL.vocr [opts] IMAGE.pgm/ppm... | --stdin | --pack FILE\n", argv[0]);
        return 2;
    }
    memset(&o, 0, sizeof o);
    for (i = 2; i < argc; i++) { /* options first; image paths are read in a second pass */
        const char *a = argv[i];
        int has = i + 1 < argc;
        if (!strcmp(a, "--mode") && has) mode_name = argv[++i];
        else if (!strcmp(a, "--tile") && has) o.tile_frames = (unsigned)atoi(argv[++i]);
        else if (!strcmp(a, "--head") && has) o.head_frames = (unsigned)atoi(argv[++i]);
        else if (!strcmp(a, "--max-width") && has) o.max_width = (unsigned)atoi(argv[++i]);
        else if (!strcmp(a, "--logits") && has) logits_path = argv[++i];
        else if (!strcmp(a, "--input-out") && has) input_path = argv[++i];
        else if (!strcmp(a, "--repeat") && has) repeat = atoi(argv[++i]);
        else if (!strcmp(a, "--pack") && has) pack = argv[++i];
        else if (!strcmp(a, "--arena") && has) arena_list = argv[++i];
        else if (!strcmp(a, "--crc")) crc = 1;
        else if (!strcmp(a, "--stdin")) use_stdin = 1;
        else if (!strcmp(a, "--invert")) invert_flag = VOCR_INVERT;
        else if (a[0] == '-' && a[1] == '-') {
            fprintf(stderr, "unknown option or missing value: %s\n", a);
            return 2;
        }
    }
    if ((o.mode = mode_of(mode_name)) < 0) {
        fprintf(stderr, "unknown --mode %s (auto, f32, w8f or w8i)\n", mode_name);
        return 2;
    }
    blob = read_file(argv[1], &blob_n, 16);
    if (!blob) {
        fprintf(stderr, "cannot read %s\n", argv[1]);
        goto done;
    }
    if (crc && vocr_check_crc(blob, blob_n)) {
        fprintf(stderr, "CRC mismatch\n");
        goto done;
    }
    rc = vocr_load(&m, blob, blob_n);
    if (rc) {
        fprintf(stderr, "vocr_load: %s\n", vocr_strerror(rc));
        goto done;
    }
    if (arena_list) {
        const char *p = arena_list;
        while (*p) {
            char *end;
            vocr_opts oo = o;
            oo.max_width = (unsigned)strtoul(p, &end, 10);
            printf("max_width %u arena %lu\n", oo.max_width, (unsigned long)vocr_arena_bytes(&m, &oo));
            p = end;
            if (*p != ',') break;
            p++;
        }
        printf("blob %lu\n", (unsigned long)m.bytes);
        status = 0;
        goto done;
    }
    need = vocr_arena_bytes(&m, &o);
    if (!need) {
        fprintf(stderr, "invalid options for this model\n");
        goto done;
    }
    if (posix_memalign(&arena, 16, need)) arena = NULL;
    rc = arena ? vocr_init(&ctx, &m, &o, arena, need) : VOCR_E_ARG;
    if (rc) {
        fprintf(stderr, "vocr_init: %s\n", vocr_strerror(rc));
        goto done;
    }
    fprintf(stderr, "blob %lu bytes, arena %lu bytes, mode %d, tile %u, head %u\n",
            (unsigned long)m.bytes, (unsigned long)need, ctx.mode, ctx.tile, ctx.head);
    if (logits_path && !(lf = fopen(logits_path, "wb"))) {
        fprintf(stderr, "cannot write %s\n", logits_path);
        goto done;
    }
    if (input_path && !(inf = fopen(input_path, "wb"))) {
        fprintf(stderr, "cannot write %s\n", input_path);
        goto done;
    }
    if (pack && run_pack(&ctx, &m, pack, repeat, lf, inf, &t_total, &frames, &lines)) goto done;
    for (i = 2; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "--crc") || !strcmp(a, "--stdin") || !strcmp(a, "--invert")) continue;
        if (a[0] == '-' && a[1] == '-') {
            i++; /* option value */
            continue;
        }
        if (run_image(&ctx, &m, a, repeat, lf, inf, &t_total, &frames, &lines)) goto done;
    }
    if (use_stdin && run_stdin(&ctx, &m, repeat, lf, inf, &t_total, &frames, &lines)) goto done;
    if (lines)
        fprintf(stderr, "%lu lines, %lu frames, %.3f ms/line, %.1f us/frame%s\n", lines, frames,
                1e3 * t_total / (double)lines / (repeat > 0 ? repeat : 1),
                1e6 * t_total / (double)(frames ? frames : 1) / (repeat > 0 ? repeat : 1),
                repeat > 0 ? " (after warm-up)" : "");
    status = 0;
done:
    if (lf && fclose(lf)) status = 1;
    if (inf && fclose(inf)) status = 1;
    free(arena);
    free(blob);
    return status;
}
