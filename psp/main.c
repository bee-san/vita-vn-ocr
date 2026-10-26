/* vocr_test - PSP EBOOT that runs the vocr recognizer (runtime/vocr.c, scalar path) on line crops
 * from its own directory on the memory stick, and reports the text, the time per line and the
 * memory it used. The same EBOOT runs on a PSP (ms0:/PSP/GAME/VOCR/) and in PPSSPPHeadless,
 * which psp/ppsspp_cli.py drives as a drop-in for runtime/build/vocr_cli.
 *
 * Inputs, in the EBOOT's directory (paths in vocr.cfg are relative to it unless they have a
 * device prefix such as ms0: or host0:):
 *   model.vocr    the weights blob (train/export.py)
 *   lines.bin     crops in vocr_cli's pack format: u32 count, then per crop u32 height,
 *                 u32 width, u32 channels (1 grey, 4 RGBA), height*width*channels bytes
 *   expect.txt    optional, one line per crop: "<crc32 of the logits, 8 hex digits><TAB><text>",
 *                 written by psp/pack.py from the host runtime; every line is then checked
 *   vocr.cfg      optional key=value lines (all optional):
 *                   model pack expect texts results stats logits input_out arena_out  file names
 *                   mode=auto|f32|w8f|w8i  tile=N head=N max_width=N   (vocr_opts; 0 = default)
 *                   invert=0|1   VOCR_INVERT on every crop (dark text)
 *                   crc=0|1      check the blob's CRC-32 first
 *                   repeat=N     time N runs per line after an untimed first run (default 1;
 *                                0 = time the first run, which also writes logits)
 *                   arena=W,W..  only report vocr_arena_bytes() for these max widths
 *                   clock=MHz    CPU clock to set (default 333, bus at half; 0 = leave it)
 *                   stack_kb=N   stack of the thread that runs vocr (default 16)
 *                   wait=0|1     wait for START before exiting (default 1)
 *                   gate=off|on|score  vocr_opts.head_gate (docs/headgate.md; default off)
 *                   two_level=0|1      vocr_opts.two_level (docs/head2level.md; default 0)
 *                   finder=-1|0|2|3    page path: each crop is a dialogue region; run the line finder
 *                                (vocr_lines.h: 0 GRAD, 2 TEXTBOX, 3 TEXTBOX_RECOVERY; -1 = off, the
 *                                default: each crop is one line) and the refinement, then read every
 *                                line it returns, inverted when the finder calls it dark
 *                   refine=FLAGS       vocr_refine_lines flags in finder mode (default 31, every step;
 *                                0 = the finder's boxes as they are)
 *                   probe=0|1    report the largest free block of every memory partition 1-12 at
 *                                start (partitions a user EBOOT cannot use report 0)
 * Outputs (same directory): texts.txt (one decoded line per crop, like vocr_cli's stdout; in finder
 * mode one line per region, its lines' texts joined by tabs), results.txt (per-line report),
 * stats.json (memory and timing), and on request logits/input dumps in vocr_cli's --logits /
 * --input-out formats (fixed-line mode). Progress goes to the screen and stdout.
 *
 * Host check (expect.txt): the CRC-32 is of the first (untimed) run's logits, every class of every
 * frame, which the runtime computes without skipping; the text is the last run's, so with gate=on
 * or two_level=1 and repeat >= 1 it is the gated or two-level text. In finder mode the CRC-32 runs
 * over every line's logits in order (zlib.crc32 of their concatenation) and the text is the
 * region's line texts joined by tabs.
 *
 * Timing is sceKernelGetSystemTimeWide() around vocr_run() (and, in finder mode, around the
 * finder plus the refinement) in the worker thread. On a PSP that is wall-clock time; under
 * PPSSPP it is emulated time, which is not the console's speed.
 */
#include <pspctrl.h>
#include <pspdebug.h>
#include <pspkernel.h>
#include <psppower.h>
#include <pspsysmem.h>
#include <pspthreadman.h>

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include "vocr.h"
#include "vocr_lines.h"
#include "vocr_lines_refine.h"

PSP_MODULE_INFO("vocr_test", PSP_MODULE_USER, 1, 0);
PSP_MAIN_THREAD_ATTR(PSP_THREAD_ATTR_USER | PSP_THREAD_ATTR_VFPU);
/* newlib's heap: small, so the memory the kernel reports free is what vocr can still get */
PSP_HEAP_SIZE_KB(256);

#define TEXT_CAP 4096
#define REGION_TEXT_CAP 16384
#define PART_USER 2
#define N_PARTITIONS 12

/* ---------------------------------------------------------------- small utilities */

#ifdef VOCR_PSP_VFPU
#define VARIANT "VFPU matrix products"
#else
#define VARIANT "scalar FPU path"
#endif
#ifdef VOCR_PSP_LARGE_MEMORY
#define MEMVARIANT "large"   /* PSP_LARGE_MEMORY = 1: the 64 MB PSPs' larger user partition */
#else
#define MEMVARIANT "24mb"
#endif

static char g_dir[256];

static void join(char *out, size_t cap, const char *name)
{
    if (strchr(name, ':'))
        snprintf(out, cap, "%s", name);
    else
        snprintf(out, cap, "%s%s", g_dir, name);
}

static char *u64s(unsigned long long v, char *buf) /* decimal, buf >= 21 bytes */
{
    char tmp[21];
    int n = 0, i = 0;
    do {
        tmp[n++] = (char)('0' + (int)(v % 10u));
        v /= 10u;
    } while (v);
    while (n) buf[i++] = tmp[--n];
    buf[i] = 0;
    return buf;
}

typedef struct { /* a partition-2 block: kernel bookkeeping for exact accounting */
    SceUID id;
    void *p;
    unsigned size;
} block_t;

static int block_alloc(block_t *b, const char *name, unsigned size)
{
    b->id = sceKernelAllocPartitionMemory(PART_USER, name, PSP_SMEM_Low, size ? size : 1u, NULL);
    b->p = b->id >= 0 ? sceKernelGetBlockHeadAddr(b->id) : NULL;
    b->size = b->p ? size : 0;
    return b->p ? 0 : -1;
}

static void block_free(block_t *b)
{
    if (b->id >= 0 && b->p) sceKernelFreePartitionMemory(b->id);
    b->id = -1;
    b->p = NULL;
    b->size = 0;
}

static int read_all(SceUID fd, void *dst, unsigned n)
{
    unsigned char *p = (unsigned char *)dst;
    while (n) {
        int r = sceIoRead(fd, p, n > 0x40000u ? 0x40000u : n);
        if (r <= 0) return -1;
        p += r;
        n -= (unsigned)r;
    }
    return 0;
}

static int write_all(SceUID fd, const void *src, unsigned n)
{
    const unsigned char *p = (const unsigned char *)src;
    while (n) {
        int r = sceIoWrite(fd, p, n > 0x40000u ? 0x40000u : n);
        if (r <= 0) return -1;
        p += r;
        n -= (unsigned)r;
    }
    return 0;
}

/* Free memory of a partition, measured by allocating it: the largest block it can give (binary
 * search to 256 B, the kernel's granularity), then the next largest, until what is left is under
 * 4 KB. Allocation is the only measure that means the same on a PSP and in PPSSPP (pspsdk's
 * pspSdkTotalFreeUserMemSize() is an endless loop in this pspsdk, and sceKernelTotalFreeMemSize()
 * is documented to report the kernel partition on hardware). A partition this user-mode app may
 * not allocate from gives 0, and its first error code in *err. */
static unsigned largest_block_in(int partition, int *err)
{
    unsigned lo = 0, hi = 64u << 20;
    if (err) *err = 0;
    while (hi - lo > 256u) {
        const unsigned mid = (lo + (hi - lo) / 2u) & ~255u;
        const SceUID id = sceKernelAllocPartitionMemory(partition, "vocr_probe", PSP_SMEM_Low, mid, NULL);
        if (id >= 0) {
            sceKernelFreePartitionMemory(id);
            lo = mid;
        } else {
            if (err && !*err) *err = (int)id;
            hi = mid;
        }
    }
    return lo;
}

static unsigned largest_block(void) { return largest_block_in(PART_USER, NULL); }

static unsigned user_free(unsigned *largest)
{
    SceUID ids[32];
    unsigned n = 0, total = 0, first = 0;
    while (n < 32u) {
        const unsigned sz = largest_block();
        SceUID id;
        if (sz < 4096u) break;
        id = sceKernelAllocPartitionMemory(PART_USER, "vocr_probe", PSP_SMEM_Low, sz, NULL);
        if (id < 0) break;
        ids[n++] = id;
        total += sz;
        if (!first) first = sz;
    }
    while (n) sceKernelFreePartitionMemory(ids[--n]);
    if (largest) *largest = first;
    return total;
}

static SceUID open_w(const char *name)
{
    char path[320];
    join(path, sizeof path, name);
    return sceIoOpen(path, PSP_O_WRONLY | PSP_O_CREAT | PSP_O_TRUNC, 0777);
}

static void fd_printf(SceUID fd, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
static void fd_printf(SceUID fd, const char *fmt, ...)
{
    static char buf[REGION_TEXT_CAP + 1024]; /* main thread only */
    va_list ap;
    int n;
    if (fd < 0) return;
    va_start(ap, fmt);
    n = vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    if (n > 0) write_all(fd, buf, (unsigned)(n < (int)sizeof buf ? n : (int)sizeof buf - 1));
}

/* screen + stdout (PPSSPPHeadless prints stdout; psplink's pspsh shows it too) */
static void say(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
static void say(const char *fmt, ...)
{
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    pspDebugScreenPrintf("%s", buf);
    printf("%s", buf);
}

static uint32_t crc_table[256];

static void crc_init(void)
{
    uint32_t i, c;
    int k;
    for (i = 0; i < 256; i++) {
        c = i;
        for (k = 0; k < 8; k++) c = (c >> 1) ^ (0xEDB88320u & (0u - (c & 1u)));
        crc_table[i] = c;
    }
}

/* zlib.crc32(data, crc): start with crc = 0, feed the pieces in order */
static uint32_t crc32_update(uint32_t crc, const void *data, size_t n)
{
    const unsigned char *p = (const unsigned char *)data;
    uint32_t c = ~crc;
    while (n--) c = crc_table[(c ^ *p++) & 0xFFu] ^ (c >> 8);
    return ~c;
}

/* ---------------------------------------------------------------- configuration */

typedef struct {
    char model[128], pack[128], expect[128], texts[128], results[128], stats[128];
    char logits[128], input_out[128], arena_out[128], arena[128];
    int mode, invert, crc, repeat, clock, stack_kb, wait, finder, probe;
    unsigned tile, head, max_width, gate, two_level, refine;
} cfg_t;

static void cfg_defaults(cfg_t *c)
{
    memset(c, 0, sizeof *c);
    strcpy(c->model, "model.vocr");
    strcpy(c->pack, "lines.bin");
    strcpy(c->expect, "expect.txt");
    strcpy(c->texts, "texts.txt");
    strcpy(c->results, "results.txt");
    strcpy(c->stats, "stats.json");
    strcpy(c->arena_out, "arena.txt");
    c->repeat = 1;
    c->clock = 333;
    c->stack_kb = 16;
    c->wait = 1;
    c->finder = -1;
    c->refine = VOCR_REFINE_DEFAULT;
}

static int mode_of(const char *v)
{
    return !strcmp(v, "auto") ? VOCR_MODE_AUTO : !strcmp(v, "f32") ? VOCR_MODE_F32
         : !strcmp(v, "w8f") ? VOCR_MODE_W8F : !strcmp(v, "w8i") ? VOCR_MODE_W8I : -1;
}

static const char *mode_name(int m)
{
    return m == VOCR_MODE_F32 ? "f32" : m == VOCR_MODE_W8F ? "w8f" : m == VOCR_MODE_W8I ? "w8i" : "auto";
}

static const char *gate_name(unsigned g)
{
    return g == VOCR_HEAD_GATE_ON ? "on" : g == VOCR_HEAD_GATE_SCORE ? "score" : "off";
}

static unsigned to_u(const char *v)
{
    unsigned n = 0;
    while (*v >= '0' && *v <= '9') n = n * 10u + (unsigned)(*v++ - '0');
    return n;
}

static int to_i(const char *v)
{
    return *v == '-' ? -(int)to_u(v + 1) : (int)to_u(v);
}

static int cfg_set(cfg_t *c, const char *k, const char *v)
{
    struct { const char *key; char *dst; } s[] = {
        {"model", c->model}, {"pack", c->pack}, {"expect", c->expect}, {"texts", c->texts},
        {"results", c->results}, {"stats", c->stats}, {"logits", c->logits},
        {"input_out", c->input_out}, {"arena_out", c->arena_out}, {"arena", c->arena}};
    unsigned i;
    for (i = 0; i < sizeof s / sizeof s[0]; i++)
        if (!strcmp(k, s[i].key)) {
            snprintf(s[i].dst, 128, "%s", v);
            return 0;
        }
    if (!strcmp(k, "mode")) return (c->mode = mode_of(v)) < 0 ? -1 : 0;
    if (!strcmp(k, "gate")) {
        if (!strcmp(v, "off") || !strcmp(v, "0")) c->gate = VOCR_HEAD_GATE_OFF;
        else if (!strcmp(v, "on") || !strcmp(v, "1")) c->gate = VOCR_HEAD_GATE_ON;
        else if (!strcmp(v, "score") || !strcmp(v, "2")) c->gate = VOCR_HEAD_GATE_SCORE;
        else return -1;
        return 0;
    }
    if (!strcmp(k, "finder")) {
        c->finder = to_i(v);
        return c->finder == -1 || c->finder == VOCR_LINES_GRAD || c->finder == VOCR_LINES_TEXTBOX ||
               c->finder == VOCR_LINES_TEXTBOX_RECOVERY ? 0 : -1;
    }
    if (!strcmp(k, "tile")) c->tile = to_u(v);
    else if (!strcmp(k, "head")) c->head = to_u(v);
    else if (!strcmp(k, "max_width")) c->max_width = to_u(v);
    else if (!strcmp(k, "invert")) c->invert = (int)to_u(v);
    else if (!strcmp(k, "crc")) c->crc = (int)to_u(v);
    else if (!strcmp(k, "repeat")) c->repeat = (int)to_u(v);
    else if (!strcmp(k, "clock")) c->clock = (int)to_u(v);
    else if (!strcmp(k, "stack_kb")) c->stack_kb = (int)to_u(v);
    else if (!strcmp(k, "wait")) c->wait = (int)to_u(v);
    else if (!strcmp(k, "two_level")) c->two_level = to_u(v);
    else if (!strcmp(k, "refine")) c->refine = to_u(v);
    else if (!strcmp(k, "probe")) c->probe = (int)to_u(v);
    else return -1;
    return 0;
}

static int cfg_read(cfg_t *c, const char *name, char *err, size_t err_cap)
{
    char path[320], buf[2048], *line, *next;
    SceUID fd;
    int n;
    join(path, sizeof path, name);
    fd = sceIoOpen(path, PSP_O_RDONLY, 0);
    if (fd < 0) return 0; /* no config: defaults */
    n = sceIoRead(fd, buf, sizeof buf - 1);
    sceIoClose(fd);
    if (n < 0) n = 0;
    buf[n] = 0;
    for (line = buf; line && *line; line = next) {
        char *eq, *end;
        next = strchr(line, '\n');
        if (next) *next++ = 0;
        end = line + strlen(line);
        while (end > line && (end[-1] == '\r' || end[-1] == ' ')) *--end = 0;
        if (!*line || *line == '#') continue;
        eq = strchr(line, '=');
        if (!eq) continue;
        *eq = 0;
        if (cfg_set(c, line, eq + 1)) {
            snprintf(err, err_cap, "vocr.cfg: bad entry %s=%s", line, eq + 1);
            return -1;
        }
    }
    return 1;
}

/* ---------------------------------------------------------------- the worker thread
 * vocr (and the line finder) run in their own thread with a fixed stack, so the stack they need
 * can be read back with sceKernelGetThreadStackFreeSize() (the kernel fills new stacks with 0xFF). */

enum { JOB_RUN = 1, JOB_ARENA = 2, JOB_QUIT = 3, JOB_FIND = 4 };

typedef struct {
    int kind, rc;
    vocr_ctx *ctx;
    const vocr_model *m;
    vocr_opts o;
    vocr_image img;
    char *text;
    vocr_extra *ex;
    size_t bytes;
    SceInt64 us;
    /* JOB_FIND */
    VocrLinesImage lim;
    int finder;
    unsigned refine;
    void *farena;
    size_t farena_bytes;
    VocrLineSet set;
} job_t;

static job_t g_job;
static SceUID g_sem_job, g_sem_done;

static int worker(SceSize args, void *argp)
{
    (void)args;
    (void)argp;
    for (;;) {
        sceKernelWaitSema(g_sem_job, 1, NULL);
        if (g_job.kind == JOB_QUIT) break;
        if (g_job.kind == JOB_RUN) {
            SceInt64 t0 = sceKernelGetSystemTimeWide();
            g_job.rc = vocr_run(g_job.ctx, &g_job.img, g_job.text, TEXT_CAP, g_job.ex);
            g_job.us = sceKernelGetSystemTimeWide() - t0;
        } else if (g_job.kind == JOB_FIND) {
            SceInt64 t0 = sceKernelGetSystemTimeWide();
            g_job.rc = vocr_find_lines(&g_job.lim, g_job.finder, g_job.farena, g_job.farena_bytes, &g_job.set);
            if (!g_job.rc && g_job.refine)
                g_job.rc = vocr_refine_lines(&g_job.lim, g_job.refine, g_job.farena, g_job.farena_bytes,
                                             &g_job.set, NULL);
            g_job.us = sceKernelGetSystemTimeWide() - t0;
        } else {
            g_job.bytes = vocr_arena_bytes(g_job.m, &g_job.o);
        }
        sceKernelSignalSema(g_sem_done, 1);
    }
    sceKernelSignalSema(g_sem_done, 1);
    return 0;
}

static void job_do(void)
{
    sceKernelSignalSema(g_sem_job, 1);
    sceKernelWaitSema(g_sem_done, 1, NULL);
}

/* ---------------------------------------------------------------- expected outputs */

typedef struct {
    char *buf;          /* expect.txt in memory, lines split in place */
    char **text;        /* per line: expected text */
    uint32_t *crc;      /* per line: expected logits CRC-32 */
    unsigned n;
    block_t blk;
} expect_t;

static unsigned hexv(const char *s)
{
    unsigned v = 0, i;
    for (i = 0; i < 8; i++) {
        char ch = s[i];
        v = v * 16u + (unsigned)(ch >= 'a' ? ch - 'a' + 10 : ch >= 'A' ? ch - 'A' + 10 : ch - '0');
    }
    return v;
}

static int expect_load(expect_t *e, const char *name, unsigned max_lines)
{
    char path[320];
    SceUID fd;
    SceOff len;
    unsigned i, text_bytes;
    char *p;
    memset(e, 0, sizeof *e);
    e->blk.id = -1;
    join(path, sizeof path, name);
    fd = sceIoOpen(path, PSP_O_RDONLY, 0);
    if (fd < 0) return 0;
    len = sceIoLseek(fd, 0, PSP_SEEK_END);
    sceIoLseek(fd, 0, PSP_SEEK_SET);
    /* the text, NUL-terminated and rounded to 4 bytes, then max_lines pointers and CRCs */
    text_bytes = ((unsigned)len + 1u + 3u) & ~3u;
    if (len < 0 || block_alloc(&e->blk, "vocr_expect",
                               text_bytes + max_lines * (unsigned)(sizeof(char *) + sizeof(uint32_t))) ||
        read_all(fd, e->blk.p, (unsigned)len)) {
        sceIoClose(fd);
        block_free(&e->blk);
        return -1;
    }
    sceIoClose(fd);
    e->buf = (char *)e->blk.p;
    e->buf[len] = 0;
    e->text = (char **)(void *)(e->buf + text_bytes);
    e->crc = (uint32_t *)(void *)(e->text + max_lines);
    for (p = e->buf, i = 0; *p && i < max_lines; i++) {
        char *nl = strchr(p, '\n');
        if (nl) *nl = 0;
        if (strlen(p) < 9 || p[8] != '\t') return -1;
        e->crc[i] = hexv(p);
        e->text[i] = p + 9;
        if (!nl) {
            i++;
            break;
        }
        p = nl + 1;
    }
    e->n = i;
    return 1;
}

/* ---------------------------------------------------------------- one line through the recognizer */

typedef struct {        /* what one line's runs report */
    unsigned frames, chars, gated, escaped;
    SceInt64 us;        /* the timed runs' mean (or the one run's time) */
    uint32_t crc;       /* the first run's logits CRC, fed in (zlib.crc32 continuation) */
    int rc;
} line_out_t;

/* Reads one line (img) runs times: the first run fills ex_logits (if any) and is untimed unless it
 * is the only run; the later, timed runs pass a zeroed vocr_extra, so the head gate and the
 * two-level head act on them. text holds the last run's text. */
static void read_line(vocr_ctx *ctx, const vocr_image *img, char *text, unsigned runs, float *logits,
                      unsigned cap_frames, unsigned n_classes, SceUID fd_logits, line_out_t *lo)
{
    vocr_extra ex, ex_t;
    unsigned r;
    SceInt64 us = 0;
    memset(&ex, 0, sizeof ex);
    if (logits) {
        ex.logits = logits;
        ex.cap_frames = cap_frames;
    }
    lo->rc = 0;
    for (r = 0; r < runs; r++) {
        memset(&ex_t, 0, sizeof ex_t);
        g_job.kind = JOB_RUN;
        g_job.ctx = ctx;
        g_job.img = *img;
        g_job.text = text;
        g_job.ex = r == 0 ? &ex : &ex_t;
        job_do();
        if (g_job.rc && g_job.rc != VOCR_E_TEXT) break;
        if (r == 0) {
            lo->frames = ex.frames;
            lo->chars = ex.chars;
            lo->gated = ex.gated;
            lo->escaped = ex.escaped;
            if (logits) {
                lo->crc = crc32_update(lo->crc, logits, (size_t)ex.frames * n_classes * 4u);
                if (fd_logits >= 0) {
                    unsigned lh[2];
                    lh[0] = ex.frames;
                    lh[1] = n_classes;
                    write_all(fd_logits, lh, 8);
                    write_all(fd_logits, logits, ex.frames * n_classes * 4u);
                }
            }
        } else {
            lo->chars = ex_t.chars;
            lo->gated = ex_t.gated;
            lo->escaped = ex_t.escaped;
        }
        if (r > 0 || runs == 1) us += g_job.us;
    }
    lo->rc = g_job.rc && g_job.rc != VOCR_E_TEXT ? g_job.rc : 0;
    if (runs > 1) us /= (SceInt64)(runs - 1);
    lo->us = us;
}

/* ---------------------------------------------------------------- main */

static void finish(const cfg_t *c)
{
    if (c->wait) {
        SceCtrlData pad;
        int i;
        say("\nPress START to exit.\n");
        sceCtrlSetSamplingMode(0);
        for (i = 0; i < 60 * 120; i++) { /* two minutes at most */
            sceCtrlReadBufferPositive(&pad, 1);
            if (pad.Buttons & PSP_CTRL_START) break;
            sceKernelDelayThread(1000000 / 60);
        }
    }
    sceKernelExitGame();
}

int main(int argc, char **argv)
{
    static cfg_t c;
    static char text[TEXT_CAP], rtext[REGION_TEXT_CAP], boxes[16 * 48], err[512], path[320];
    vocr_model m;
    vocr_ctx ctx;
    vocr_opts o;
    block_t b_blob = {-1, NULL, 0}, b_arena = {-1, NULL, 0}, b_crop = {-1, NULL, 0};
    block_t b_logits = {-1, NULL, 0}, b_input = {-1, NULL, 0}, b_farena = {-1, NULL, 0};
    expect_t ex_lines;
    SceUID fd_pack = -1, fd_texts = -1, fd_res = -1, fd_logits = -1, fd_input = -1, thid = -1;
    unsigned free_start, max_start, free_loaded, max_loaded, free_min, count = 0, j, need, k_total, k_max;
    unsigned long long us_total = 0, frames_total = 0, chars_total = 0, us_max = 0;
    unsigned long long finder_us_total = 0, gated_total = 0, escaped_total = 0, lines_found = 0;
    unsigned text_ok = 0, crc_ok = 0, failed = 0, stack_free = 0, stack_bytes, farena_need = 0;
    unsigned part_free[N_PARTITIONS + 1];
    int part_err[N_PARTITIONS + 1];
    unsigned arena_w[3] = {256, 640, 1024};
    size_t arena_at[3] = {0, 0, 0}, model_sz, ctx_sz;
    int rc, have_cfg, have_expect = 0, cpu_mhz, bus_mhz, p;
    char n1[24], n2[24], n3[24], n4[24]; /* one per u64s() in a call: arguments are all
                                                   * evaluated before the call */
    SceOff blob_len;
    SceUID fd;

    pspDebugScreenInit();
    crc_init();
    memset(&ex_lines, 0, sizeof ex_lines);
    memset(part_free, 0, sizeof part_free);
    memset(part_err, 0, sizeof part_err);
    ex_lines.blk.id = -1;
    /* the EBOOT's directory: argv[0] is e.g. ms0:/PSP/GAME/VOCR/EBOOT.PBP */
    snprintf(g_dir, sizeof g_dir, "%s", argc > 0 && argv[0] ? argv[0] : "ms0:/PSP/GAME/VOCR/EBOOT.PBP");
    {
        char *slash = strrchr(g_dir, '/');
        if (slash) slash[1] = 0;
        else snprintf(g_dir, sizeof g_dir, "ms0:/PSP/GAME/VOCR/");
    }
    cfg_defaults(&c);
    have_cfg = cfg_read(&c, "vocr.cfg", err, sizeof err);
    say("vocr PSP test (runtime/vocr.c, " VARIANT ", memory " MEMVARIANT ")\ndir %s, config %s\n", g_dir,
        have_cfg > 0 ? "vocr.cfg" : "defaults");
    fd_res = open_w(c.results);
    if (have_cfg < 0) goto fail;
    if (c.clock > 0) scePowerSetClockFrequency(c.clock, c.clock, c.clock / 2);
    cpu_mhz = scePowerGetCpuClockFrequencyInt();
    bus_mhz = scePowerGetBusClockFrequencyInt();
    free_start = user_free(&max_start);
    k_total = (unsigned)sceKernelTotalFreeMemSize();
    k_max = (unsigned)sceKernelMaxFreeMemSize();
    vocr_sizes(&model_sz, &ctx_sz);
    say("CPU %d MHz, bus %d MHz; user memory free %s B (largest block %s B)\n", cpu_mhz, bus_mhz,
        u64s(free_start, n1), u64s(max_start, n2));
    if (c.probe) {
        for (p = 1; p <= N_PARTITIONS; p++) {
            part_free[p] = largest_block_in(p, &part_err[p]);
            say("partition %2d: largest free block %s B%s\n", p, u64s(part_free[p], n1),
                part_free[p] ? "" : " (not allocatable here)");
        }
    }

    /* ---- the blob */
    join(path, sizeof path, c.model);
    fd = sceIoOpen(path, PSP_O_RDONLY, 0);
    if (fd < 0) {
        snprintf(err, sizeof err, "cannot open %s", path);
        goto fail;
    }
    blob_len = sceIoLseek(fd, 0, PSP_SEEK_END);
    sceIoLseek(fd, 0, PSP_SEEK_SET);
    if (blob_len <= 0 || block_alloc(&b_blob, "vocr_blob", (unsigned)blob_len) ||
        read_all(fd, b_blob.p, (unsigned)blob_len)) {
        sceIoClose(fd);
        snprintf(err, sizeof err, "cannot load %s (%u bytes)", path, (unsigned)blob_len);
        goto fail;
    }
    sceIoClose(fd);
    if (c.crc && vocr_check_crc(b_blob.p, b_blob.size)) {
        snprintf(err, sizeof err, "CRC mismatch in %s", path);
        goto fail;
    }
    rc = vocr_load(&m, b_blob.p, b_blob.size);
    if (rc) {
        snprintf(err, sizeof err, "vocr_load: %s", vocr_strerror(rc));
        goto fail;
    }
    memset(&o, 0, sizeof o);
    o.tile_frames = c.tile;
    o.head_frames = c.head;
    o.max_width = c.max_width;
    o.mode = c.mode;
    o.head_gate = c.gate;
    o.two_level = c.two_level;

    /* ---- the worker thread */
    stack_bytes = (unsigned)c.stack_kb * 1024u;
    g_sem_job = sceKernelCreateSema("vocr_job", 0, 0, 1, NULL);
    g_sem_done = sceKernelCreateSema("vocr_done", 0, 0, 1, NULL);
    thid = sceKernelCreateThread("vocr_worker", worker, 0x20, (int)stack_bytes,
                                 PSP_THREAD_ATTR_USER | PSP_THREAD_ATTR_VFPU, NULL);
    if (g_sem_job < 0 || g_sem_done < 0 || thid < 0 || sceKernelStartThread(thid, 0, NULL) < 0) {
        snprintf(err, sizeof err, "cannot start the worker thread (%d)", (int)thid);
        goto fail;
    }

    /* ---- arena bytes: for this config's widths only (vocr_cli --arena), or the report's 3 */
    g_job.kind = JOB_ARENA;
    g_job.m = &m;
    if (c.arena[0]) {
        SceUID fd_a = open_w(c.arena_out);
        const char *q = c.arena;
        while (*q) {
            g_job.o = o;
            g_job.o.max_width = to_u(q);
            job_do();
            fd_printf(fd_a, "max_width %u arena %s\n", g_job.o.max_width, u64s(g_job.bytes, n1));
            while (*q && *q != ',') q++;
            if (*q == ',') q++;
        }
        fd_printf(fd_a, "blob %u\n", (unsigned)m.bytes);
        if (fd_a >= 0) sceIoClose(fd_a);
        say("arena sizes written to %s\n", c.arena_out);
        goto done;
    }
    for (j = 0; j < 3; j++) {
        g_job.o = o;
        g_job.o.max_width = arena_w[j];
        job_do();
        arena_at[j] = g_job.bytes;
    }
    g_job.o = o;
    job_do();
    need = (unsigned)g_job.bytes;
    if (!need || block_alloc(&b_arena, "vocr_arena", need)) {
        snprintf(err, sizeof err, "arena: %u bytes not available", need);
        goto fail;
    }
    rc = vocr_init(&ctx, &m, &o, b_arena.p, b_arena.size);
    if (rc) {
        snprintf(err, sizeof err, "vocr_init: %s", vocr_strerror(rc));
        goto fail;
    }
    say("blob %u B (%s, %u classes), arena %u B, mode %s, tile %u, head %u, max width %u, gate %s, two-level %u\n",
        (unsigned)m.bytes, m.int8 ? "int8" : "fp32", m.n_classes, need, mode_name(ctx.mode), ctx.tile,
        ctx.head, ctx.max_width, gate_name(c.gate), c.two_level);
    say("arena at max width 256 / 640 / 1024: %s / %s / %s B\n", u64s(arena_at[0], n1),
        u64s(arena_at[1], n2), u64s(arena_at[2], n3));
    if (c.finder >= 0) { /* the line finder's arena: any region up to a whole PSP screen */
        farena_need = (unsigned)vocr_lines_arena_bytes(480u, 272u);
        if (block_alloc(&b_farena, "vocr_lines", farena_need)) {
            snprintf(err, sizeof err, "finder arena: %u bytes not available", farena_need);
            goto fail;
        }
        say("finder %d, refine flags %u, finder arena %u B (480 x 272)\n", c.finder, c.refine, farena_need);
    }

    /* ---- crops */
    join(path, sizeof path, c.pack);
    fd_pack = sceIoOpen(path, PSP_O_RDONLY, 0);
    if (fd_pack < 0 || read_all(fd_pack, &count, 4)) {
        snprintf(err, sizeof err, "cannot read %s", path);
        goto fail;
    }
    if (c.expect[0]) {
        have_expect = expect_load(&ex_lines, c.expect, count);
        if (have_expect < 0 || (have_expect > 0 && ex_lines.n != count)) {
            snprintf(err, sizeof err, "%s: bad or %u lines for %u crops", c.expect, ex_lines.n, count);
            goto fail;
        }
    }
    if (have_expect > 0 || c.logits[0]) {
        unsigned frames_cap = ctx.max_width / m.stride + 1u;
        if (block_alloc(&b_logits, "vocr_logits", frames_cap * m.n_classes * 4u)) {
            snprintf(err, sizeof err, "no memory for the logits buffer");
            goto fail;
        }
    }
    if (c.finder < 0 && c.input_out[0] && block_alloc(&b_input, "vocr_input", (ctx.max_width + 64u) * m.height)) {
        snprintf(err, sizeof err, "no memory for the input buffer");
        goto fail;
    }
    if (c.logits[0]) fd_logits = open_w(c.logits);
    if (c.finder < 0 && c.input_out[0]) fd_input = open_w(c.input_out);
    fd_texts = open_w(c.texts);
    free_loaded = user_free(&max_loaded);
    free_min = free_loaded;
    if (c.finder >= 0)
        fd_printf(fd_res, "# region\tframes\tchars\tus\ttext_ok\tlogits_ok\tlines\tfinder_us\tgated\tescaped\tboxes\ttext\n");
    else
        fd_printf(fd_res, "# line\tframes\tchars\tus\ttext_ok\tlogits_ok\ttext\tgated\tescaped\n");
    say("%u %s from %s%s\n", count, c.finder >= 0 ? "regions" : "crops", c.pack,
        have_expect > 0 ? ", checked against expect.txt" : "");

    for (j = 0; j < count; j++) {
        unsigned hdr[3], h, w, ch, bytes, runs = (unsigned)(c.repeat > 0 ? c.repeat + 1 : 1);
        unsigned frames = 0, chars = 0, gated = 0, escaped = 0, nl = 0, k;
        SceInt64 us = 0, fus = 0;
        uint32_t crc = 0;
        int t_ok = -1, l_ok = -1;
        float *logits = (float *)b_logits.p;
        const unsigned cap_frames = b_logits.p ? b_logits.size / 4u / m.n_classes : 0u;
        vocr_image img;
        line_out_t lo;
        if (read_all(fd_pack, hdr, 12)) {
            snprintf(err, sizeof err, "truncated pack at line %u", j);
            goto fail;
        }
        h = hdr[0];
        w = hdr[1];
        ch = hdr[2];
        /* checked before h * w * ch is formed, which must not wrap: at most 16 MB per crop
         * (a whole 480 x 272 RGBA screen is 510 KB) */
        if ((ch != 1 && ch != 4) || !h || !w || h > 65535u || w > 65535u ||
            (unsigned long long)h * w * ch > (16u << 20)) {
            snprintf(err, sizeof err, "bad crop header at line %u", j);
            goto fail;
        }
        if (c.finder >= 0 && (w > 480u || h > 272u)) {
            snprintf(err, sizeof err, "region %u is %u x %u, over the PSP screen", j, w, h);
            goto fail;
        }
        bytes = h * w * ch;
        if (bytes > b_crop.size) {
            unsigned f;
            block_free(&b_crop);
            if (block_alloc(&b_crop, "vocr_crop", bytes)) {
                snprintf(err, sizeof err, "no memory for a %u-byte crop", bytes);
                goto fail;
            }
            f = user_free(NULL);
            if (f < free_min) free_min = f;
        }
        if (read_all(fd_pack, b_crop.p, bytes)) {
            snprintf(err, sizeof err, "truncated pack at line %u", j);
            goto fail;
        }
        rtext[0] = 0;
        boxes[0] = 0;
        if (c.finder < 0) {
            img.pixels = (const unsigned char *)b_crop.p;
            img.width = w;
            img.height = h;
            img.stride = (size_t)w * ch;
            img.format = (ch == 4 ? VOCR_RGBA8 : VOCR_GREY8) | (c.invert ? VOCR_INVERT : 0);
            memset(&lo, 0, sizeof lo);
            read_line(&ctx, &img, text, runs, logits, cap_frames, m.n_classes, fd_logits, &lo);
            if (lo.rc) {
                snprintf(err, sizeof err, "vocr_run, line %u: %s", j, vocr_strerror(lo.rc));
                goto fail;
            }
            frames = lo.frames;
            chars = lo.chars;
            gated = lo.gated;
            escaped = lo.escaped;
            us = lo.us;
            crc = lo.crc;
            snprintf(rtext, sizeof rtext, "%s", text);
            if (fd_input >= 0) {
                unsigned rw = 0, pw = 0, ih[3];
                vocr_preprocess(&m, &img, (unsigned char *)b_input.p, b_input.size, &rw, &pw);
                ih[0] = m.height;
                ih[1] = pw;
                ih[2] = rw;
                write_all(fd_input, ih, 12);
                write_all(fd_input, b_input.p, pw * m.height);
            }
        } else {
            /* the page path: finder + refinement on the region, then every line it returns */
            size_t used = 0, bl = 0;
            g_job.kind = JOB_FIND;
            g_job.lim.pixels = (const unsigned char *)b_crop.p;
            g_job.lim.width = w;
            g_job.lim.height = h;
            g_job.lim.stride = (size_t)w * ch;
            g_job.lim.bpp = ch;
            g_job.finder = c.finder;
            g_job.refine = c.refine;
            g_job.farena = b_farena.p;
            g_job.farena_bytes = b_farena.size;
            job_do();
            if (g_job.rc) {
                snprintf(err, sizeof err, "finder, region %u: rc %d", j, g_job.rc);
                goto fail;
            }
            fus = g_job.us;
            nl = g_job.set.count;
            lines_found += nl;
            for (k = 0; k < nl; k++) {
                const VocrLine *ln = &g_job.set.lines[k];
                size_t tl;
                img.pixels = (const unsigned char *)b_crop.p + (size_t)ln->y * w * ch + (size_t)ln->x * ch;
                img.width = ln->w;
                img.height = ln->h;
                img.stride = (size_t)w * ch;
                img.format = (ch == 4 ? VOCR_RGBA8 : VOCR_GREY8) | ((ln->dark != 0) != (c.invert != 0) ? VOCR_INVERT : 0);
                memset(&lo, 0, sizeof lo);
                lo.crc = crc;
                read_line(&ctx, &img, text, runs, logits, cap_frames, m.n_classes, fd_logits, &lo);
                if (lo.rc) {
                    snprintf(err, sizeof err, "vocr_run, region %u line %u: %s", j, k, vocr_strerror(lo.rc));
                    goto fail;
                }
                crc = lo.crc;
                frames += lo.frames;
                chars += lo.chars;
                gated += lo.gated;
                escaped += lo.escaped;
                us += lo.us;
                tl = strlen(text);
                if (used + tl + 2 < sizeof rtext) {
                    if (k) rtext[used++] = '\t';
                    memcpy(rtext + used, text, tl);
                    used += tl;
                    rtext[used] = 0;
                }
                bl += (size_t)snprintf(boxes + bl, sizeof boxes - bl, "%s%u,%u,%u,%u,%u", k ? ";" : "", ln->x,
                                       ln->y, ln->w, ln->h, ln->dark);
                if (bl >= sizeof boxes) bl = sizeof boxes - 1;
            }
            finder_us_total += (unsigned long long)fus;
        }
        if (have_expect > 0) {
            t_ok = !strcmp(rtext, ex_lines.text[j]);
            if (b_logits.p) l_ok = crc == ex_lines.crc[j];
        }
        frames_total += frames;
        chars_total += chars;
        gated_total += gated;
        escaped_total += escaped;
        us_total += (unsigned long long)us;
        if ((unsigned long long)(us + fus) > us_max) us_max = (unsigned long long)(us + fus);
        text_ok += t_ok == 1;
        crc_ok += l_ok == 1;
        failed += t_ok == 0 || l_ok == 0;
        fd_printf(fd_texts, "%s\n", rtext);
        if (c.finder >= 0)
            fd_printf(fd_res, "%u\t%u\t%u\t%u\t%d\t%d\t%u\t%u\t%u\t%u\t%s\t%s\n", j, frames, chars, (unsigned)us, t_ok,
                      l_ok, nl, (unsigned)fus, gated, escaped, boxes, rtext);
        else
            fd_printf(fd_res, "%u\t%u\t%u\t%u\t%d\t%d\t%s\t%u\t%u\n", j, frames, chars, (unsigned)us, t_ok, l_ok,
                      rtext, gated, escaped);
        if (j < 12 || j + 1 == count || t_ok == 0 || l_ok == 0)
            say("#%u %3u frames %6u.%03u ms %s\n", j, frames, (unsigned)((us + fus) / 1000),
                (unsigned)((us + fus) % 1000), t_ok < 0 ? "" : t_ok && l_ok ? "same as host" : "DIFFERS FROM HOST");
        else if (j == 12)
            say("...\n");
    }
    sceIoClose(fd_pack);
    fd_pack = -1;
    stack_free = (unsigned)sceKernelGetThreadStackFreeSize(thid);

    say("\n%u %s, %s frames: %s us total, %s us/frame (max %s us per %s)\n", count,
        c.finder >= 0 ? "regions" : "lines", u64s(frames_total, n1), u64s(us_total, n2),
        u64s(frames_total ? us_total / frames_total : 0, n3), u64s(us_max, n4), c.finder >= 0 ? "region" : "line");
    if (c.finder >= 0)
        say("finder + refinement %s us total, %s lines found\n", u64s(finder_us_total, n1), u64s(lines_found, n2));
    if (c.gate == VOCR_HEAD_GATE_ON || c.two_level)
        say("head gate skipped %s frames, two-level escape fired on %s\n", u64s(gated_total, n1),
            u64s(escaped_total, n2));
    if (have_expect > 0)
        say("host check: %u/%u texts identical, %u/%u logits bit-identical%s\n", text_ok, count, crc_ok,
            count, failed ? "  <-- MISMATCH" : "");
    say("vocr stack used %u of %u B; user memory free %s B before loading, %s B with model + "
        "arena + buffers\n", stack_bytes - stack_free, stack_bytes, u64s(free_start, n1), u64s(free_loaded, n2));
    {
        SceUID fd_s = open_w(c.stats);
        fd_printf(fd_s, "{\"app\": \"vocr_test\", \"variant\": \"" VARIANT "\", \"memory\": \"" MEMVARIANT "\", "
                 "\"model\": \"%s\", \"blob_bytes\": %u, \"int8\": %u, "
                 "\"classes\": %u, \"params\": %u, \"mode\": \"%s\", \"tile\": %u, \"head\": %u, "
                 "\"max_width\": %u, \"invert\": %d, \"repeat\": %d, \"gate\": \"%s\", \"two_level\": %u,\n",
                 c.model, (unsigned)m.bytes, m.int8, m.n_classes, m.params, mode_name(ctx.mode), ctx.tile,
                 ctx.head, ctx.max_width, c.invert, c.repeat, gate_name(c.gate), c.two_level);
        fd_printf(fd_s, " \"arena_bytes\": %u, \"arena_at\": {\"256\": %u, \"640\": %u, \"1024\": %u}, "
                 "\"sizeof_model\": %u, \"sizeof_ctx\": %u, \"stack_bytes\": %u, \"stack_used\": %u,\n",
                 need, (unsigned)arena_at[0], (unsigned)arena_at[1], (unsigned)arena_at[2],
                 (unsigned)model_sz, (unsigned)ctx_sz, stack_bytes, stack_bytes - stack_free);
        fd_printf(fd_s, " \"cpu_mhz\": %d, \"bus_mhz\": %d, \"user_free_start\": %u, \"user_max_block_start\": %u, "
                 "\"user_free_loaded\": %u, \"user_max_block_loaded\": %u, \"user_free_min\": %u, "
                 "\"kernel_total_free_start\": %u, \"kernel_max_free_start\": %u, "
                 "\"crop_buffer_bytes\": %u, \"logits_buffer_bytes\": %u, \"input_buffer_bytes\": %u,\n",
                 cpu_mhz, bus_mhz, free_start, max_start, free_loaded, max_loaded, free_min, k_total, k_max,
                 b_crop.size, b_logits.size, b_input.size);
        fd_printf(fd_s, " \"finder\": %d, \"refine\": %u, \"finder_arena_bytes\": %u, \"finder_us_total\": %s, "
                 "\"lines_found\": %s, \"gated\": %s, \"escaped\": %s, \"partitions\": [",
                 c.finder, c.finder >= 0 ? c.refine : 0u, farena_need, u64s(finder_us_total, n1),
                 u64s(lines_found, n2), u64s(gated_total, n3), u64s(escaped_total, n4));
        for (p = 1; c.probe && p <= N_PARTITIONS; p++)
            fd_printf(fd_s, "%s{\"id\": %d, \"largest_free\": %u, \"first_error\": %d}", p > 1 ? ", " : "", p,
                      part_free[p], part_err[p]);
        fd_printf(fd_s, "],\n");
        fd_printf(fd_s, " \"lines\": %u, \"frames\": %s, \"chars\": %s, \"us_total\": %s, \"us_max_line\": %s, "
                 "\"expect\": %s, \"text_ok\": %u, \"logits_ok\": %u}\n",
                 count, u64s(frames_total, n1), u64s(chars_total, n2), u64s(us_total, n3), u64s(us_max, n4),
                 have_expect > 0 ? "true" : "false", text_ok, crc_ok);
        if (fd_s >= 0) sceIoClose(fd_s);
    }
    fd_printf(fd_res, "# done: %u %s\n", count, c.finder >= 0 ? "regions" : "lines");
    goto done;

fail:
    say("\nERROR: %s\n", err);
    fd_printf(fd_res, "# ERROR: %s\n", err);
done:
    if (thid >= 0) {
        g_job.kind = JOB_QUIT;
        job_do();
        sceKernelWaitThreadEnd(thid, NULL);
        sceKernelDeleteThread(thid);
    }
    if (fd_pack >= 0) sceIoClose(fd_pack);
    if (fd_texts >= 0) sceIoClose(fd_texts);
    if (fd_logits >= 0) sceIoClose(fd_logits);
    if (fd_input >= 0) sceIoClose(fd_input);
    if (fd_res >= 0) sceIoClose(fd_res);
    block_free(&ex_lines.blk);
    block_free(&b_farena);
    block_free(&b_input);
    block_free(&b_logits);
    block_free(&b_crop);
    block_free(&b_arena);
    block_free(&b_blob);
    finish(&c);
    return 0;
}
