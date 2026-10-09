/* vocr_test - PS Vita test app for the vita-vn-ocr runtime (runtime/vocr.h, vocr_lines.h).
 *
 * It reads ux0:data/vocr/, a bundle that vita/bundle.py writes on a PC:
 *   cases.bin    line crops and dialogue regions, plus what the host build of the same C
 *                runtime produced for each one (text, CRC-32 of the logits, line boxes)
 *   *.vocr       the weights blobs named in cases.bin (train/export.py)
 *   font.bin     optional: a 16-px bitmap font for the screen (rasterised from OFL M+ 1p)
 *   config.txt   optional: key=value lines, see Config below
 * For every case and model it runs the recognizer once with logits (to check them bit for
 * bit against the host) and then `repeat` timed runs (sceKernelGetProcessTimeWide), and
 * for regions the whole console pipeline: vocr_find_lines, then vocr_run per line with
 * VOCR_INVERT on dark lines. Line crops of real screenshots take their polarity from
 * vocr_line_dark, as the line finder would give it. Rows go to ux0:data/vocr/results.tsv
 * (and stdout); vita/compare.py reads that file back on the PC.
 *
 * Status per case and model: MATCH = same text, same logits CRC (and same line boxes) as
 * the host; TEXT = same text, different logits; DIFF = different text; ERR = an error code.
 * The work runs on a worker thread with a 64 KiB stack (the overlay's job stack without
 * ncnn); a separate probe thread measures the runtime's own stack use. Not part of the
 * runtime: only this harness uses SCE APIs and newlib.
 */
#include <psp2/ctrl.h>
#include <psp2/io/fcntl.h>
#include <psp2/io/stat.h>
#include <psp2/kernel/processmgr.h>
#include <psp2/kernel/sysmem.h>
#include <psp2/kernel/threadmgr.h>
#include <psp2/perf.h>
#include <psp2/power.h>

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cases.h"
#include "screen.h"
#include "vocr.h"
#include "vocr_lines.h"

#ifndef VOCR_GIT
#define VOCR_GIT "unknown"
#endif
#define APP_VERSION "1"
#define DIR "ux0:data/vocr/"
#define MAX_MODELS VC_MAX_MODELS
#define MAX_REPEAT 15
#define TEXT_CAP VC_TEXT_CAP
#define N_PMU 6
#define PROBE_STACK (32 * 1024)

typedef char lines_max_matches[VC_MAX_LINES == VOCR_LINES_MAX ? 1 : -1];

/* ---------------------------------------------------------------- config */

typedef struct {
    int clock;          /* ARM clock in MHz for the run (0: leave it) */
    int repeat;         /* timed runs per case and model */
    int autoexit;       /* exit when done instead of waiting for START */
    int pmu;            /* try the ARM performance counters (ScePerf) */
    int limit;          /* at most this many cases (0: all) */
    int stack_kb;       /* worker stack */
    int cpu;            /* worker CPU (0-2), -1: any */
    char only[64];      /* only cases whose id contains this */
    char models[64];    /* only these models (comma list of names), "" = all */
} Config;

static Config cfg = {444, 3, 0, 1, 0, 64, 0, "", ""};

static void set_key(const char *k, const char *v)
{
    if (!strcmp(k, "clock")) cfg.clock = atoi(v);
    else if (!strcmp(k, "repeat")) cfg.repeat = atoi(v);
    else if (!strcmp(k, "autoexit")) cfg.autoexit = atoi(v);
    else if (!strcmp(k, "pmu")) cfg.pmu = atoi(v);
    else if (!strcmp(k, "limit")) cfg.limit = atoi(v);
    else if (!strcmp(k, "stack_kb")) cfg.stack_kb = atoi(v);
    else if (!strcmp(k, "cpu")) cfg.cpu = atoi(v);
    else if (!strcmp(k, "only")) snprintf(cfg.only, sizeof cfg.only, "%s", v);
    else if (!strcmp(k, "models")) snprintf(cfg.models, sizeof cfg.models, "%s", v);
}

static void load_config(void)
{
    char buf[2048], *line, *save = NULL;
    SceUID fd = sceIoOpen(DIR "config.txt", SCE_O_RDONLY, 0);
    int n;
    if (fd < 0) return;
    n = sceIoRead(fd, buf, sizeof buf - 1);
    sceIoClose(fd);
    if (n <= 0) return;
    buf[n] = 0;
    for (line = strtok_r(buf, "\r\n", &save); line; line = strtok_r(NULL, "\r\n", &save)) {
        char *eq = strchr(line, '=');
        if (line[0] == '#' || !eq) continue;
        *eq = 0;
        set_key(line, eq + 1);
    }
    if (cfg.repeat < 1) cfg.repeat = 1;
    if (cfg.repeat > MAX_REPEAT) cfg.repeat = MAX_REPEAT;
    if (cfg.stack_kb < 16) cfg.stack_kb = 16;
    if (cfg.cpu > 2) cfg.cpu = -1;
}

/* ---------------------------------------------------------------- output */

static SceUID out_fd = -1;
static char out_buf[16384];
static size_t out_len;

static void out_flush(void)
{
    if (out_fd >= 0 && out_len) sceIoWrite(out_fd, out_buf, out_len);
    out_len = 0;
}

/* One line to results.tsv and stdout (Vita3K logs it; on a console nothing reads it).
 * Only the worker thread emits, so the line buffer is static: it stays off the worker's
 * stack, which config.txt's stack_kb may make as small as 16 KiB. */
static void emit(const char *fmt, ...)
{
    static char line[TEXT_CAP * 3];
    va_list ap;
    int n;
    va_start(ap, fmt);
    n = vsnprintf(line, sizeof line - 1, fmt, ap);
    va_end(ap);
    if (n < 0) return;
    if (n > (int)sizeof line - 2) n = (int)sizeof line - 2;
    line[n++] = '\n';
    line[n] = 0;
    fputs(line, stdout);
    fflush(stdout);
    if (out_len + (size_t)n > sizeof out_buf) out_flush();
    if ((size_t)n > sizeof out_buf) {
        if (out_fd >= 0) sceIoWrite(out_fd, line, (SceSize)n);
    } else {
        memcpy(out_buf + out_len, line, (size_t)n);
        out_len += (size_t)n;
    }
}

/* Text with tab, newline and backslash escaped, for a TSV field. */
static const char *esc(const char *s, char *buf, size_t cap)
{
    size_t o = 0;
    for (; *s && o + 3 < cap; s++) {
        if (*s == '\t' || *s == '\n' || *s == '\\') {
            buf[o++] = '\\';
            buf[o++] = *s == '\t' ? 't' : *s == '\n' ? 'n' : '\\';
        } else {
            buf[o++] = *s;
        }
    }
    buf[o] = 0;
    return buf;
}

static void say(uint32_t color, const char *fmt, ...)
{
    char line[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(line, sizeof line, fmt, ap);
    va_end(ap);
    scr_log(color, line);
}

/* ---------------------------------------------------------------- memory and files */

static void *mb_alloc(const char *name, size_t n, SceUID *uid_out)
{
    void *base = NULL;
    SceUID uid = sceKernelAllocMemBlock(name, SCE_KERNEL_MEMBLOCK_TYPE_USER_RW,
                                        (SceSize)((n + 4095u) & ~(size_t)4095u), NULL);
    if (uid < 0 || sceKernelGetMemBlockBase(uid, &base) < 0) return NULL;
    if (uid_out) *uid_out = uid;
    return base;
}

static int read_full(SceUID fd, void *dst, size_t n)
{
    unsigned char *p = (unsigned char *)dst;
    while (n) {
        SceSSize got = sceIoRead(fd, p, n > (1u << 20) ? (1u << 20) : (SceSize)n);
        if (got <= 0) return -1;
        p += got;
        n -= (size_t)got;
    }
    return 0;
}

/* The whole file in a memblock (4 KiB aligned, so 16-byte aligned as vocr_load needs). */
static void *read_file(const char *path, size_t *n)
{
    SceUID fd = sceIoOpen(path, SCE_O_RDONLY, 0);
    SceOff size;
    void *buf;
    if (fd < 0) return NULL;
    size = sceIoLseek(fd, 0, SCE_SEEK_END);
    sceIoLseek(fd, 0, SCE_SEEK_SET);
    buf = size > 0 ? mb_alloc("vocr_file", (size_t)size, NULL) : NULL;
    if (!buf || read_full(fd, buf, (size_t)size)) buf = NULL;
    sceIoClose(fd);
    *n = buf ? (size_t)size : 0;
    return buf;
}

static uint32_t crc_table[256];

static void crc_init(void)
{
    uint32_t i, c;
    int k;
    for (i = 0; i < 256; i++) {
        for (c = i, k = 0; k < 8; k++) c = (c >> 1) ^ (0xEDB88320u & (0u - (c & 1u)));
        crc_table[i] = c;
    }
}

/* zlib's crc32: crc = crc32_update(crc, ...) chains, starting from 0. */
static uint32_t crc32_update(uint32_t crc, const void *data, size_t n)
{
    const unsigned char *p = (const unsigned char *)data;
    crc = ~crc;
    while (n--) crc = crc_table[(crc ^ *p++) & 0xFFu] ^ (crc >> 8);
    return ~crc;
}

static unsigned long long now_us(void) { return sceKernelGetProcessTimeWide(); }

/* ---------------------------------------------------------------- performance counters */

static const SceUInt8 pmu_events[N_PMU] = {
    SCE_PERF_ARM_PMON_CYCLE_COUNT, SCE_PERF_ARM_PMON_INST_RENAME, SCE_PERF_ARM_PMON_NEON_RENAME,
    SCE_PERF_ARM_PMON_DCACHE_MISS, SCE_PERF_ARM_PMON_DCACHE_STALL, SCE_PERF_ARM_PMON_ISSUE_NO_DISPATCH};
static const char *pmu_names[N_PMU] = {"cycles", "insns", "neon", "dmiss", "dstall", "nodispatch"};
static int pmu_state; /* 0 untried, 1 works, -1 unavailable */
static int pmu_err;

static int pmu_start(void)
{
    int i, rc;
    if (!cfg.pmu || pmu_state < 0) return 0;
    rc = scePerfArmPmonReset(SCE_PERF_ARM_PMON_THREAD_ID_SELF);
    for (i = 0; rc >= 0 && i < N_PMU; i++)
        rc = scePerfArmPmonSelectEvent(SCE_PERF_ARM_PMON_THREAD_ID_SELF, (SceUInt32)i, pmu_events[i]);
    if (rc >= 0) rc = scePerfArmPmonStart(SCE_PERF_ARM_PMON_THREAD_ID_SELF);
    if (rc < 0) {
        pmu_state = -1;
        pmu_err = rc;
        return 0;
    }
    return 1;
}

static void pmu_stop(int started, unsigned long long v[N_PMU])
{
    int i;
    for (i = 0; i < N_PMU; i++) v[i] = 0;
    if (!started) return;
    scePerfArmPmonStop(SCE_PERF_ARM_PMON_THREAD_ID_SELF);
    for (i = 0; i < N_PMU; i++) {
        SceUInt32 x = 0;
        if (scePerfArmPmonGetCounterValue(SCE_PERF_ARM_PMON_THREAD_ID_SELF, (SceUInt32)i, &x) < 0) {
            pmu_state = -1;
            return;
        }
        v[i] = x;
    }
    if (!v[0]) pmu_state = -1; /* counters that never count (an emulator) */
    else pmu_state = 1;
}

/* ---------------------------------------------------------------- bundle */

typedef struct {
    unsigned time_ms_n;
    float *ms;          /* per line case: median ms of the timed runs */
    float *us_frame;    /* per line case: median us per CTC frame */
    float *region_ms;   /* per region case: median ms of the whole pipeline */
    unsigned n_line, n_region;
    unsigned n, match, text_only, diff, err, box_diff;
    unsigned long long frames, pmu_frames, pmu[N_PMU];
} Stats;

typedef struct {
    char name[17], file[65];
    int mode;
    uint32_t blob_bytes, blob_crc, host_arena;
    const unsigned char *blob;
    size_t blob_n;
    vocr_model m;
    vocr_ctx c;
    void *arena;
    size_t arena_bytes;
    int active;
    Stats st;
} Model;

typedef VcCase Case;

static Model models[MAX_MODELS];
static unsigned n_models, n_cases, n_classes, max_frames, max_lines, max_case_bytes, max_rw, max_rh;
static unsigned char *case_buf;
static char *case_strings;
static float *logits;
static void *lines_arena;
static size_t lines_arena_bytes;
static volatile int stop_requested;

static int model_wanted(const char *name)
{
    char list[80], *t, *save = NULL;
    if (!cfg.models[0]) return 1;
    snprintf(list, sizeof list, "%s", cfg.models);
    for (t = strtok_r(list, ",", &save); t; t = strtok_r(NULL, ",", &save))
        if (!strcmp(t, name)) return 1;
    return 0;
}

static int load_models(SceUID fd)
{
    unsigned char hdr[VC_HEADER_BYTES], rec[VC_MODEL_BYTES];
    VcHeader h;
    unsigned i, j;
    if (read_full(fd, hdr, sizeof hdr) || vc_header(hdr, &h)) {
        say(SCR_RED, "cases.bin: not a version-1 bundle");
        return -1;
    }
    n_models = h.n_models;
    n_cases = h.n_cases;
    max_case_bytes = h.max_case_bytes;
    max_lines = h.max_lines;
    max_frames = h.max_frames;
    n_classes = h.n_classes;
    max_rw = h.max_rw;
    max_rh = h.max_rh;
    for (i = 0; i < n_models; i++) {
        Model *md = &models[i];
        VcModel vm;
        char path[128];
        if (read_full(fd, rec, sizeof rec)) return -1;
        vc_model(rec, &vm);
        memcpy(md->name, vm.name, sizeof md->name);
        memcpy(md->file, vm.file, sizeof md->file);
        md->mode = (int)vm.mode;
        md->blob_bytes = vm.blob_bytes;
        md->blob_crc = vm.blob_crc;
        md->host_arena = vm.host_arena;
        md->active = model_wanted(md->name);
        for (j = 0; j < i; j++) /* models that share a blob file load it once */
            if (!strcmp(models[j].file, md->file) && models[j].blob) {
                md->blob = models[j].blob;
                md->blob_n = models[j].blob_n;
            }
        if (!md->blob) {
            size_t n;
            unsigned long long t0 = now_us();
            snprintf(path, sizeof path, DIR "%s", md->file);
            md->blob = (const unsigned char *)read_file(path, &n);
            md->blob_n = n;
            if (!md->blob) {
                say(SCR_RED, "cannot read %s", path);
                emit("# error cannot-read %s", path);
                md->active = 0;
                continue;
            }
            emit("# blob %s bytes=%u read_us=%llu file_crc=%s vocr_check_crc=%d", md->file, (unsigned)n,
                 now_us() - t0, crc32_update(0, md->blob, n) == md->blob_crc ? "ok" : "BAD",
                 vocr_check_crc(md->blob, n));
        }
    }
    return 0;
}

static int plan_models(void)
{
    unsigned i;
    int any = 0;
    for (i = 0; i < n_models; i++) {
        Model *md = &models[i];
        vocr_opts o;
        int rc;
        if (!md->active || !md->blob) continue;
        rc = vocr_load(&md->m, md->blob, md->blob_n);
        memset(&o, 0, sizeof o);
        o.mode = md->mode;
        md->arena_bytes = rc ? 0 : vocr_arena_bytes(&md->m, &o);
        md->arena = md->arena_bytes ? mb_alloc("vocr_arena", md->arena_bytes, NULL) : NULL;
        if (!rc) rc = md->arena ? vocr_init(&md->c, &md->m, &o, md->arena, md->arena_bytes) : VOCR_E_ARENA;
        emit("# model %s file=%s mode=%d blob=%u arena=%u host_arena=%u params=%u classes=%u rc=%d %s",
             md->name, md->file, md->mode, (unsigned)md->blob_n, (unsigned)md->arena_bytes, md->host_arena,
             md->m.params, md->m.n_classes, rc, vocr_strerror(rc));
        if (rc) {
            md->active = 0;
            say(SCR_RED, "%s: %s", md->name, vocr_strerror(rc));
            continue;
        }
        if (md->m.n_classes != n_classes) {
            md->active = 0;
            say(SCR_RED, "%s: %u classes, bundle says %u", md->name, md->m.n_classes, n_classes);
            continue;
        }
        md->st.ms = (float *)malloc(sizeof(float) * n_cases);
        md->st.us_frame = (float *)malloc(sizeof(float) * n_cases);
        md->st.region_ms = (float *)malloc(sizeof(float) * n_cases);
        if (!md->st.ms || !md->st.us_frame || !md->st.region_ms) return -1;
        any = 1;
    }
    return any ? 0 : -1;
}

/* ---------------------------------------------------------------- running */

typedef struct {
    char text[TEXT_CAP];
    uint32_t crc, frames, dark;
    VocrLineSet set;
    int rc;
} Result;

static int run_one(Model *md, const vocr_image *im, char *text, size_t cap, Result *r, int check)
{
    vocr_extra ex;
    int rc;
    if (!check) return vocr_run(&md->c, im, text, cap, NULL);
    memset(&ex, 0, sizeof ex);
    ex.logits = logits;
    ex.cap_frames = max_frames;
    rc = vocr_run(&md->c, im, text, cap, &ex);
    if (!rc) {
        if (ex.frames > max_frames) return VOCR_E_WIDTH;
        r->crc = crc32_update(r->crc, logits, (size_t)ex.frames * n_classes * sizeof(float));
        r->frames += ex.frames;
    }
    return rc;
}

/* One pass over a case. check: also the logits CRC (untimed). */
static int run_case(Model *md, const Case *cs, Result *r, int check)
{
    r->crc = 0;
    r->frames = 0;
    r->text[0] = 0;
    if (cs->kind == VC_KIND_LINE) {
        vocr_image im;
        im.pixels = cs->px;
        im.width = cs->w;
        im.height = cs->h;
        im.stride = (size_t)cs->w * cs->ch;
        im.format = (cs->ch == 4 ? VOCR_RGBA8 : VOCR_GREY8) | (r->dark ? VOCR_INVERT : 0);
        return run_one(md, &im, r->text, sizeof r->text, r, check);
    } else {
        VocrLinesImage li;
        size_t used = 0;
        unsigned i;
        int rc;
        li.pixels = cs->px;
        li.width = cs->w;
        li.height = cs->h;
        li.stride = (size_t)cs->w * 4u;
        li.bpp = VOCR_LINES_RGBA;
        rc = vocr_find_lines(&li, VOCR_LINES_GRAD, lines_arena, lines_arena_bytes, &r->set);
        if (rc) return rc - 1000; /* finder errors, kept apart from the recognizer's */
        for (i = 0; i < r->set.count; i++) {
            const VocrLine *ln = &r->set.lines[i];
            vocr_image im;
            im.pixels = cs->px + (size_t)ln->y * li.stride + (size_t)ln->x * 4u;
            im.width = ln->w;
            im.height = ln->h;
            im.stride = li.stride;
            im.format = VOCR_RGBA8 | (ln->dark ? VOCR_INVERT : 0);
            if (i) {
                if (used + 1 >= sizeof r->text) return VOCR_E_TEXT;
                r->text[used++] = '\n';
                r->text[used] = 0;
            }
            rc = run_one(md, &im, r->text + used, sizeof r->text - used, r, check);
            if (rc) return rc;
            used += strlen(r->text + used);
        }
        return 0;
    }
}

static int cmp_float(const void *a, const void *b)
{
    float x = *(const float *)a, y = *(const float *)b;
    return x < y ? -1 : x > y;
}

static float median(float *v, unsigned n)
{
    if (!n) return 0.0f;
    qsort(v, n, sizeof v[0], cmp_float);
    return n & 1 ? v[n / 2] : 0.5f * (v[n / 2 - 1] + v[n / 2]);
}

static int boxes_equal(const Case *cs, unsigned mi, const Result *r)
{
    unsigned i;
    if (cs->kind != VC_KIND_REGION) return 1;
    if (r->set.count != cs->exp[mi].n_lines) return 0;
    for (i = 0; i < r->set.count; i++) {
        const VocrLine *l = &r->set.lines[i];
        const uint32_t *b = cs->exp[mi].box[i];
        if (l->x != b[0] || l->y != b[1] || l->w != b[2] || l->h != b[3] || l->dark != b[4]) return 0;
    }
    return 1;
}

static Result res;

/* Runs every active model on one case; returns the colour of its screen line. */
static uint32_t do_case(const Case *cs, unsigned index)
{
    unsigned mi, k;
    /* static (worker thread only): 16 KiB that would otherwise sit on the worker's stack */
    static char e1[TEXT_CAP * 2], e2[TEXT_CAP * 2];
    char summary[200];
    size_t sl = 0;
    uint32_t color = SCR_GREEN;
    unsigned dark = 0;
    if (cs->kind == VC_KIND_LINE && (cs->flags & VC_FLAG_AUTO_POLARITY)) {
        VocrLinesImage li;
        int d;
        li.pixels = cs->px;
        li.width = cs->w;
        li.height = cs->h;
        li.stride = (size_t)cs->w * cs->ch;
        li.bpp = cs->ch;
        d = vocr_line_dark(&li, 0, 0, cs->w, cs->h);
        dark = d > 0;
    } else if (cs->kind == VC_KIND_LINE) {
        dark = (cs->flags & VC_FLAG_DARK) != 0;
    }
    summary[0] = 0;
    for (mi = 0; mi < n_models; mi++) {
        Model *md = &models[mi];
        Stats *st = &md->st;
        float runs[MAX_REPEAT];
        unsigned long long pmu[N_PMU], t0;
        const char *status;
        int rc, started, boxes_ok = 1;
        unsigned frames;
        if (!md->active) continue;
        res.dark = dark;
        rc = run_case(md, cs, &res, 1);
        frames = res.frames;
        if (!rc) {
            uint32_t crc = res.crc;
            started = pmu_start();
            for (k = 0; k < (unsigned)cfg.repeat && !rc; k++) {
                t0 = now_us();
                rc = run_case(md, cs, &res, 0);
                runs[k] = (float)(now_us() - t0) / 1000.0f;
            }
            pmu_stop(started, pmu);
            res.crc = crc;
        } else {
            memset(pmu, 0, sizeof pmu);
        }
        st->n++;
        boxes_ok = boxes_equal(cs, mi, &res);
        if (rc) {
            status = "ERR";
            st->err++;
        } else if (strcmp(res.text, cs->exp[mi].text)) {
            status = "DIFF";
            st->diff++;
        } else if (res.crc != cs->exp[mi].crc || frames != cs->exp[mi].frames || !boxes_ok) {
            status = "TEXT";
            st->text_only++;
        } else {
            status = "MATCH";
            st->match++;
        }
        if (!boxes_ok) st->box_diff++;
        if (status[0] == 'D' || status[0] == 'E') color = SCR_RED;
        else if (status[0] == 'T' && color == SCR_GREEN) color = SCR_YELLOW;
        {
            float t_min = 0.0f, t_med = 0.0f, usf = 0.0f;
            if (!rc) {
                float sorted[MAX_REPEAT];
                memcpy(sorted, runs, sizeof(float) * (unsigned)cfg.repeat);
                t_med = median(sorted, (unsigned)cfg.repeat);
                t_min = sorted[0];
                usf = frames ? 1000.0f * t_med / (float)frames : 0.0f;
                if (cs->kind == VC_KIND_LINE) {
                    st->ms[st->n_line] = t_med;
                    st->us_frame[st->n_line++] = usf;
                } else {
                    st->region_ms[st->n_region++] = t_med;
                }
                st->frames += frames;
                if (pmu[0]) {
                    for (k = 0; k < N_PMU; k++) st->pmu[k] += pmu[k];
                    st->pmu_frames += (unsigned long long)frames * (unsigned)cfg.repeat;
                }
            }
            emit("%s\t%s\t%s\t%s\t%d\t%u\t%u\t%u\t%u\t%u\t%.3f\t%.3f\t%.2f\t%llu\t%llu\t%llu\t%llu\t%llu\t%llu"
                 "\t%08lx\t%08lx\t%s\t%s",
                 cs->id, cs->kind == VC_KIND_LINE ? "line" : "region", md->name, status, rc, dark, cs->w, cs->h,
                 frames, cs->kind == VC_KIND_REGION ? res.set.count : 1u, t_min, t_med, usf,
                 pmu[0] / (unsigned)cfg.repeat, pmu[1] / (unsigned)cfg.repeat, pmu[2] / (unsigned)cfg.repeat,
                 pmu[3] / (unsigned)cfg.repeat, pmu[4] / (unsigned)cfg.repeat, pmu[5] / (unsigned)cfg.repeat,
                 (unsigned long)cs->exp[mi].crc, (unsigned long)res.crc, esc(cs->exp[mi].text, e1, sizeof e1),
                 esc(res.text, e2, sizeof e2));
            sl += (size_t)snprintf(summary + sl, sl < sizeof summary ? sizeof summary - sl : 0, "%-5s %6.1f ",
                                   status, t_med);
        }
    }
    {
        char line[400];
        const char *t = res.text;
        char flat[160];
        size_t i;
        for (i = 0; t[i] && i < sizeof flat - 1; i++) flat[i] = t[i] == '\n' ? ' ' : t[i];
        flat[i] = 0;
        snprintf(line, sizeof line, "%4u %s %s %s", index + 1, cs->id, summary, flat);
        scr_log(color, line);
    }
    return color;
}

/* ---------------------------------------------------------------- stack probe */

static volatile int probe_stack_used = -1, probe_stack_size, probe_free_kernel = -1;

static unsigned char probe_line[32 * 1024];
static unsigned char probe_region[960 * 64 * 4];

/* The deepest runtime calls on a synthetic 1024-px line and a region, every model. */
static __attribute__((noinline)) void probe_work(void)
{
    static char text[TEXT_CAP];
    static vocr_ctx c;
    VocrLinesImage li;
    VocrLineSet set;
    vocr_image im;
    unsigned i, x, y;
    for (y = 0; y < 32; y++)
        for (x = 0; x < 1024; x++) probe_line[y * 1024 + x] = (unsigned char)(((x / 7 + y / 5) % 3) ? 30 : 220);
    for (y = 0; y < 64; y++)
        for (x = 0; x < 960 * 4; x++)
            probe_region[y * 960 * 4 + x] = (unsigned char)((y > 8 && y < 26 && (x / 4) % 9 < 5) ? 240 : 20);
    for (i = 0; i < n_models; i++) {
        Model *md = &models[i];
        vocr_opts o;
        if (!md->active) continue;
        memset(&o, 0, sizeof o);
        o.mode = md->mode;
        if (vocr_arena_bytes(&md->m, &o) && !vocr_init(&c, &md->m, &o, md->arena, md->arena_bytes)) {
            im.pixels = probe_line;
            im.width = 1024;
            im.height = 32;
            im.stride = 1024;
            im.format = VOCR_GREY8;
            vocr_run(&c, &im, text, sizeof text, NULL);
            im.format = VOCR_GREY8 | VOCR_INVERT;
            vocr_run(&c, &im, text, sizeof text, NULL);
        }
    }
    li.pixels = probe_region;
    li.width = 960;
    li.height = 64;
    li.stride = 960 * 4;
    li.bpp = VOCR_LINES_RGBA;
    vocr_find_lines(&li, VOCR_LINES_GRAD, lines_arena, lines_arena_bytes, &set);
    vocr_find_lines(&li, VOCR_LINES_LEGACY, lines_arena, lines_arena_bytes, &set);
}

static int probe_main(SceSize args, void *argp)
{
    SceKernelThreadInfo info;
    volatile unsigned char here = 0;
    unsigned char *base, *sp = (unsigned char *)&here, *p;
    (void)args;
    (void)argp;
    memset(&info, 0, sizeof info);
    info.size = sizeof info;
    if (sceKernelGetThreadInfo(sceKernelGetThreadId(), &info) < 0) return 0;
    base = (unsigned char *)info.stack;
    probe_stack_size = info.stackSize;
    if (!base || sp <= base + 1024 || sp > base + info.stackSize) return 0;
    memset(base, 0xA5, (size_t)(sp - base) - 512);
    probe_work();
    for (p = base; p < sp && *p == 0xA5; p++) {
    }
    probe_stack_used = (int)(base + info.stackSize - p);
    probe_free_kernel = sceKernelGetThreadStackFreeSize(0);
    return 0;
}

static void run_probe(void)
{
    SceUID th = sceKernelCreateThread("vocr_probe", probe_main, 0x10000100, PROBE_STACK, 0, 0, NULL);
    if (th < 0) return;
    if (sceKernelStartThread(th, 0, NULL) >= 0) sceKernelWaitThreadEnd(th, NULL, NULL);
    sceKernelDeleteThread(th);
    emit("# stack probe_used=%d probe_stack=%d kernel_free=%d (vocr_arena_bytes, vocr_init, vocr_run "
         "on a 1024-px line, vocr_find_lines grad and legacy)",
         probe_stack_used, probe_stack_size, probe_free_kernel);
}

/* ---------------------------------------------------------------- worker */

static int stop_pressed(void)
{
    SceCtrlData pad;
    memset(&pad, 0, sizeof pad);
    sceCtrlPeekBufferPositive(0, &pad, 1);
    return (pad.buttons & SCE_CTRL_CIRCLE) != 0;
}

static void summary(unsigned done, unsigned long long t_all)
{
    unsigned i, k;
    for (i = 0; i < n_models; i++) {
        Model *md = &models[i];
        Stats *st = &md->st;
        char pmu_txt[256] = "";
        float ms, usf, rms;
        if (!md->active) continue;
        ms = median(st->ms, st->n_line);
        usf = median(st->us_frame, st->n_line);
        rms = median(st->region_ms, st->n_region);
        if (st->pmu_frames) {
            size_t o = 0;
            for (k = 0; k < N_PMU; k++)
                o += (size_t)snprintf(pmu_txt + o, sizeof pmu_txt - o, " %s_per_frame=%.0f", pmu_names[k],
                                      (double)st->pmu[k] / (double)st->pmu_frames);
        }
        emit("# summary model=%s cases=%u match=%u text_only=%u diff=%u err=%u box_diff=%u line_cases=%u "
             "line_ms_median=%.2f us_per_frame_median=%.1f ms_640px_est=%.1f region_cases=%u region_ms_median=%.2f "
             "frames=%llu%s",
             md->name, st->n, st->match, st->text_only, st->diff, st->err, st->box_diff, st->n_line, ms, usf,
             usf * 80.0f / 1000.0f, st->n_region, rms, st->frames, pmu_txt);
        say(st->diff || st->err ? SCR_RED : st->text_only ? SCR_YELLOW : SCR_GREEN,
            "%s: %u/%u match, %u text-only, %u diff, %u err; median %.1f ms/line, %.0f us/frame (%.0f ms per 640 px)",
            md->name, st->match, st->n, st->text_only, st->diff, st->err, ms, usf, usf * 80.0f / 1000.0f);
    }
    emit("# pmu state=%d err=0x%08x", pmu_state, (unsigned)pmu_err);
    emit("# done cases=%u of %u total_s=%.1f%s", done, n_cases, (double)t_all / 1e6,
         stop_requested ? " (stopped with CIRCLE)" : "");
}

static int worker_main(SceSize args, void *argp)
{
    SceUID fd;
    Case cs;
    unsigned i, done = 0;
    unsigned long long t_all = now_us();
    size_t mb = 0, cb = 0;
    char line[160];
    (void)args;
    (void)argp;
    crc_init();
    sceIoMkdir(DIR, 0777);
    out_fd = sceIoOpen(DIR "results.tsv", SCE_O_WRONLY | SCE_O_CREAT | SCE_O_TRUNC, 0666);
    vocr_sizes(&mb, &cb);
    emit("# vocr_test version=%s git=%s built=%s", APP_VERSION, VOCR_GIT, __DATE__ " " __TIME__);
    emit("# config clock=%d repeat=%d pmu=%d limit=%d stack_kb=%d cpu=%d only=%s models=%s", cfg.clock,
         cfg.repeat, cfg.pmu, cfg.limit, cfg.stack_kb, cfg.cpu, cfg.only, cfg.models);
    emit("# clock arm_mhz=%d bus_mhz=%d", scePowerGetArmClockFrequency(), scePowerGetBusClockFrequency());
    emit("# sizes vocr_model=%u vocr_ctx=%u", (unsigned)mb, (unsigned)cb);
    out_flush();
    fd = sceIoOpen(DIR "cases.bin", SCE_O_RDONLY, 0);
    if (fd < 0) {
        say(SCR_RED, "no " DIR "cases.bin: copy the bundle from vita/bundle.py (vita/README.md)");
        emit("# error no-cases.bin 0x%08x", (unsigned)fd);
        goto out;
    }
    if (load_models(fd) || plan_models()) {
        say(SCR_RED, "no model could be loaded (see results.tsv)");
        goto out;
    }
    lines_arena_bytes = vocr_lines_arena_bytes(max_rw ? max_rw : 960, max_rh ? max_rh : 544);
    lines_arena = mb_alloc("vocr_lines", lines_arena_bytes, NULL);
    logits = (float *)mb_alloc("vocr_logits", (size_t)max_frames * n_classes * sizeof(float), NULL);
    case_buf = (unsigned char *)mb_alloc("vocr_case", max_case_bytes, NULL);
    case_strings = (char *)malloc(VC_STRINGS_BYTES);
    if (!lines_arena || !logits || !case_buf || !case_strings) {
        say(SCR_RED, "out of memory");
        emit("# error out-of-memory");
        goto out;
    }
    emit("# lines arena=%u max_region=%ux%u", (unsigned)lines_arena_bytes, max_rw, max_rh);
    out_flush();
    for (i = 0; i < n_models; i++)
        if (models[i].active) {
            snprintf(line, sizeof line, "%.16s: blob %u B, arena %u B", models[i].name, (unsigned)models[i].blob_n,
                     (unsigned)models[i].arena_bytes);
            scr_head(1 + (i > 1), SCR_WHITE, line);
        }
    run_probe();
    snprintf(line, sizeof line, "line finder arena %u B, runtime stack %d B, ARM %d MHz", (unsigned)lines_arena_bytes,
             probe_stack_used, scePowerGetArmClockFrequency());
    scr_head(3, SCR_GREY, line);
    emit("#cols id\tkind\tmodel\tstatus\trc\tdark\twidth\theight\tframes\tlines\tms_min\tms_median\tus_per_frame"
         "\tcycles\tinsns\tneon\tdmiss\tdstall\tnodispatch\tcrc_host\tcrc_vita\ttext_host\ttext_vita");
    out_flush();
    for (i = 0; i < n_cases && !stop_requested; i++) {
        uint32_t len;
        unsigned char lb[4];
        if (read_full(fd, lb, 4) || (len = vc_rd32(lb)) < 4 || len > max_case_bytes || read_full(fd, case_buf, len - 4) ||
            vc_case(case_buf, len - 4, n_models, case_strings, &cs)) {
            say(SCR_RED, "cases.bin: bad record %u", i);
            emit("# error bad-record %u", i);
            break;
        }
        if (cfg.only[0] && !strstr(cs.id, cfg.only)) continue;
        if (cfg.limit && done >= (unsigned)cfg.limit) break;
        do_case(&cs, i);
        done++;
        /* each case's rows reach ux0: before the next case starts, so if the app dies on the
         * console, results.tsv keeps every finished case and the first case missing from it
         * is the one that was running */
        out_flush();
        /* a full bundle runs for minutes without input: keep the console from suspending or
         * turning the screen off (resets those timers, like a button press would) */
        sceKernelPowerTick(SCE_KERNEL_POWER_TICK_DEFAULT);
        if (stop_pressed()) stop_requested = 1;
        snprintf(line, sizeof line, "%u / %u cases   (CIRCLE: stop)", i + 1, n_cases);
        scr_status(SCR_CYAN, line, i + 1, n_cases);
    }
    sceIoClose(fd);
    summary(done, now_us() - t_all);
    scr_status(SCR_GREEN, cfg.autoexit ? "done" : "done - press START to exit", 1, 1);
out:
    out_flush();
    if (out_fd >= 0) sceIoClose(out_fd);
    return 0;
}

int main(void)
{
    SceUID th;
    size_t fn = 0;
    void *fontdata;
    char line[128];
    scr_init();
    fontdata = read_file(DIR "font.bin", &fn);
    if (fontdata) scr_load_font(fontdata, fn);
    load_config();
    if (cfg.clock > 0) scePowerSetArmClockFrequency(cfg.clock);
    sceCtrlSetSamplingMode(SCE_CTRL_MODE_DIGITAL);
    snprintf(line, sizeof line, "vita-vn-ocr runtime test %s (%s)   ARM %d MHz", APP_VERSION, VOCR_GIT,
             scePowerGetArmClockFrequency());
    scr_head(0, SCR_CYAN, line);
    scr_status(SCR_CYAN, "starting", 0, 1);
    th = sceKernelCreateThread("vocr_worker", worker_main, 0x10000100, cfg.stack_kb * 1024, 0,
                               cfg.cpu >= 0 ? (SCE_KERNEL_CPU_MASK_USER_0 << cfg.cpu) : 0, NULL);
    if (th >= 0 && sceKernelStartThread(th, 0, NULL) >= 0) sceKernelWaitThreadEnd(th, NULL, NULL);
    if (!cfg.autoexit) {
        SceCtrlData pad;
        do {
            sceKernelDelayThread(50 * 1000);
            memset(&pad, 0, sizeof pad);
            sceCtrlPeekBufferPositive(0, &pad, 1);
        } while (!(pad.buttons & SCE_CTRL_START));
    }
    sceKernelExitProcess(0);
    return 0;
}
