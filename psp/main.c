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
 * Outputs (same directory): texts.txt (one decoded line per crop, like vocr_cli's stdout),
 * results.txt (per-line report), stats.json (memory and timing), and on request logits/input
 * dumps in vocr_cli's --logits / --input-out formats. Progress goes to the screen and stdout.
 *
 * Timing is sceKernelGetSystemTimeWide() around vocr_run() in the worker thread. On a PSP that
 * is wall-clock time; under PPSSPP it is emulated time, which is not the console's speed.
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

PSP_MODULE_INFO("vocr_test", PSP_MODULE_USER, 1, 0);
PSP_MAIN_THREAD_ATTR(PSP_THREAD_ATTR_USER | PSP_THREAD_ATTR_VFPU);
/* newlib's heap: small, so the memory the kernel reports free is what vocr can still get */
PSP_HEAP_SIZE_KB(256);

#define TEXT_CAP 4096
#define PART_USER 2

/* ---------------------------------------------------------------- small utilities */

#ifdef VOCR_PSP_VFPU
#define VARIANT "VFPU matrix products"
#else
#define VARIANT "scalar FPU path"
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

/* Free user memory, measured by allocating it: the largest block partition 2 can give (binary
 * search to 256 B, the kernel's granularity), then the next largest, until what is left is
 * under 4 KB. Allocation is the only measure that means the same on a PSP and in PPSSPP
 * (pspsdk's pspSdkTotalFreeUserMemSize() is an endless loop in this pspsdk, and
 * sceKernelTotalFreeMemSize() is documented to report the kernel partition on hardware). */
static unsigned largest_block(void)
{
    unsigned lo = 0, hi = 64u << 20;
    while (hi - lo > 256u) {
        const unsigned mid = (lo + (hi - lo) / 2u) & ~255u;
        const SceUID id = sceKernelAllocPartitionMemory(PART_USER, "vocr_probe", PSP_SMEM_Low, mid, NULL);
        if (id >= 0) {
            sceKernelFreePartitionMemory(id);
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return lo;
}

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
    static char buf[TEXT_CAP + 512]; /* main thread only */
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

static uint32_t crc32_buf(const void *data, size_t n) /* zlib.crc32 */
{
    const unsigned char *p = (const unsigned char *)data;
    uint32_t c = 0xFFFFFFFFu;
    while (n--) c = crc_table[(c ^ *p++) & 0xFFu] ^ (c >> 8);
    return ~c;
}

/* ---------------------------------------------------------------- configuration */

typedef struct {
    char model[128], pack[128], expect[128], texts[128], results[128], stats[128];
    char logits[128], input_out[128], arena_out[128], arena[128];
    int mode, invert, crc, repeat, clock, stack_kb, wait;
    unsigned tile, head, max_width;
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

static unsigned to_u(const char *v)
{
    unsigned n = 0;
    while (*v >= '0' && *v <= '9') n = n * 10u + (unsigned)(*v++ - '0');
    return n;
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
    if (!strcmp(k, "tile")) c->tile = to_u(v);
    else if (!strcmp(k, "head")) c->head = to_u(v);
    else if (!strcmp(k, "max_width")) c->max_width = to_u(v);
    else if (!strcmp(k, "invert")) c->invert = (int)to_u(v);
    else if (!strcmp(k, "crc")) c->crc = (int)to_u(v);
    else if (!strcmp(k, "repeat")) c->repeat = (int)to_u(v);
    else if (!strcmp(k, "clock")) c->clock = (int)to_u(v);
    else if (!strcmp(k, "stack_kb")) c->stack_kb = (int)to_u(v);
    else if (!strcmp(k, "wait")) c->wait = (int)to_u(v);
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
 * vocr runs in its own thread with a fixed stack, so the stack it needs can be read back with
 * sceKernelGetThreadStackFreeSize() (the kernel fills new stacks with 0xFF). */

enum { JOB_RUN = 1, JOB_ARENA = 2, JOB_QUIT = 3 };

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
    static char text[TEXT_CAP], err[512], path[320];
    vocr_model m;
    vocr_ctx ctx;
    vocr_opts o;
    block_t b_blob = {-1, NULL, 0}, b_arena = {-1, NULL, 0}, b_crop = {-1, NULL, 0};
    block_t b_logits = {-1, NULL, 0}, b_input = {-1, NULL, 0};
    expect_t ex_lines;
    SceUID fd_pack = -1, fd_texts = -1, fd_res = -1, fd_logits = -1, fd_input = -1, thid = -1;
    unsigned free_start, max_start, free_loaded, max_loaded, free_min, count = 0, j, need, k_total, k_max;
    unsigned long long us_total = 0, frames_total = 0, chars_total = 0, us_max = 0;
    unsigned text_ok = 0, crc_ok = 0, failed = 0, stack_free = 0, stack_bytes;
    unsigned arena_w[3] = {256, 640, 1024};
    size_t arena_at[3] = {0, 0, 0}, model_sz, ctx_sz;
    int rc, have_cfg, have_expect = 0, cpu_mhz, bus_mhz;
    char n1[24], n2[24], n3[24], n4[24];
    SceOff blob_len;
    SceUID fd;

    pspDebugScreenInit();
    crc_init();
    memset(&ex_lines, 0, sizeof ex_lines);
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
    say("vocr PSP test (runtime/vocr.c, " VARIANT ")\ndir %s, config %s\n", g_dir,
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
        const char *p = c.arena;
        while (*p) {
            g_job.o = o;
            g_job.o.max_width = to_u(p);
            job_do();
            fd_printf(fd_a, "max_width %u arena %s\n", g_job.o.max_width, u64s(g_job.bytes, n1));
            while (*p && *p != ',') p++;
            if (*p == ',') p++;
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
    say("blob %u B (%s, %u classes), arena %u B, mode %s, tile %u, head %u, max width %u\n",
        (unsigned)m.bytes, m.int8 ? "int8" : "fp32", m.n_classes, need, mode_name(ctx.mode), ctx.tile,
        ctx.head, ctx.max_width);
    say("arena at max width 256 / 640 / 1024: %s / %s / %s B\n", u64s(arena_at[0], n1),
        u64s(arena_at[1], n2), u64s(arena_at[2], n3));

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
    if (c.input_out[0] && block_alloc(&b_input, "vocr_input", (ctx.max_width + 64u) * m.height)) {
        snprintf(err, sizeof err, "no memory for the input buffer");
        goto fail;
    }
    if (c.logits[0]) fd_logits = open_w(c.logits);
    if (c.input_out[0]) fd_input = open_w(c.input_out);
    fd_texts = open_w(c.texts);
    free_loaded = user_free(&max_loaded);
    free_min = free_loaded;
    fd_printf(fd_res, "# line\tframes\tchars\tus\ttext_ok\tlogits_ok\ttext\n");
    say("%u crops from %s%s\n", count, c.pack, have_expect > 0 ? ", checked against expect.txt" : "");

    for (j = 0; j < count; j++) {
        unsigned hdr[3], h, w, ch, bytes, r, runs = (unsigned)(c.repeat > 0 ? c.repeat + 1 : 1);
        vocr_extra ex;
        SceInt64 us = 0;
        int t_ok = -1, l_ok = -1;
        if (read_all(fd_pack, hdr, 12)) {
            snprintf(err, sizeof err, "truncated pack at line %u", j);
            goto fail;
        }
        h = hdr[0];
        w = hdr[1];
        ch = hdr[2];
        bytes = h * w * ch;
        if ((ch != 1 && ch != 4) || !h || !w || h > 65535u || w > 65535u) {
            snprintf(err, sizeof err, "bad crop header at line %u", j);
            goto fail;
        }
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
        g_job.kind = JOB_RUN;
        g_job.ctx = &ctx;
        g_job.img.pixels = (const unsigned char *)b_crop.p;
        g_job.img.width = w;
        g_job.img.height = h;
        g_job.img.stride = (size_t)w * ch;
        g_job.img.format = (ch == 4 ? VOCR_RGBA8 : VOCR_GREY8) | (c.invert ? VOCR_INVERT : 0);
        g_job.text = text;
        for (r = 0; r < runs; r++) {
            memset(&ex, 0, sizeof ex);
            if (b_logits.p) {
                ex.logits = (float *)b_logits.p;
                ex.cap_frames = b_logits.size / 4u / m.n_classes;
            }
            g_job.ex = r == 0 ? &ex : NULL; /* logits only from the first, untimed, run */
            job_do();
            if (g_job.rc && g_job.rc != VOCR_E_TEXT) break;
            if (r == 0) {
                frames_total += ex.frames;
                chars_total += ex.chars;
                if (b_logits.p) {
                    if (have_expect > 0) {
                        const uint32_t crc = crc32_buf(b_logits.p, (size_t)ex.frames * m.n_classes * 4u);
                        l_ok = crc == ex_lines.crc[j];
                        t_ok = !strcmp(text, ex_lines.text[j]);
                    }
                    if (fd_logits >= 0) {
                        unsigned lh[2];
                        lh[0] = ex.frames;
                        lh[1] = m.n_classes;
                        write_all(fd_logits, lh, 8);
                        write_all(fd_logits, b_logits.p, ex.frames * m.n_classes * 4u);
                    }
                }
                if (fd_input >= 0) {
                    unsigned rw = 0, pw = 0, ih[3];
                    vocr_preprocess(&m, &g_job.img, (unsigned char *)b_input.p, b_input.size, &rw, &pw);
                    ih[0] = m.height;
                    ih[1] = pw;
                    ih[2] = rw;
                    write_all(fd_input, ih, 12);
                    write_all(fd_input, b_input.p, pw * m.height);
                }
            }
            if (r > 0 || runs == 1) us += g_job.us;
        }
        if (g_job.rc && g_job.rc != VOCR_E_TEXT) {
            snprintf(err, sizeof err, "vocr_run, line %u: %s", j, vocr_strerror(g_job.rc));
            goto fail;
        }
        if (runs > 1) us /= (SceInt64)(runs - 1);
        us_total += (unsigned long long)us;
        if ((unsigned long long)us > us_max) us_max = (unsigned long long)us;
        text_ok += t_ok == 1;
        crc_ok += l_ok == 1;
        failed += t_ok == 0 || l_ok == 0;
        fd_printf(fd_texts, "%s\n", text);
        fd_printf(fd_res, "%u\t%u\t%u\t%u\t%d\t%d\t%s\n", j, ex.frames, ex.chars, (unsigned)us, t_ok, l_ok, text);
        if (j < 12 || j + 1 == count || t_ok == 0 || l_ok == 0)
            say("#%u %3u frames %6u.%03u ms %s\n", j, ex.frames, (unsigned)(us / 1000),
                (unsigned)(us % 1000), t_ok < 0 ? "" : t_ok && l_ok ? "same as host" : "DIFFERS FROM HOST");
        else if (j == 12)
            say("...\n");
    }
    sceIoClose(fd_pack);
    fd_pack = -1;
    stack_free = (unsigned)sceKernelGetThreadStackFreeSize(thid);

    say("\n%u lines, %s frames: %s us total, %s us/line, %s us/frame (max %s us/line)\n", count,
        u64s(frames_total, n1), u64s(us_total, n2), u64s(count ? us_total / count : 0, n3),
        u64s(frames_total ? us_total / frames_total : 0, n1), u64s(us_max, n2));
    if (have_expect > 0)
        say("host check: %u/%u texts identical, %u/%u logits bit-identical%s\n", text_ok, count, crc_ok,
            count, failed ? "  <-- MISMATCH" : "");
    say("vocr stack used %u of %u B; user memory free %s B before loading, %s B with model + "
        "arena + buffers\n", stack_bytes - stack_free, stack_bytes, u64s(free_start, n1), u64s(free_loaded, n2));
    {
        SceUID fd_s = open_w(c.stats);
        fd_printf(fd_s, "{\"app\": \"vocr_test\", \"variant\": \"" VARIANT "\", \"model\": \"%s\", \"blob_bytes\": %u, \"int8\": %u, "
                 "\"classes\": %u, \"params\": %u, \"mode\": \"%s\", \"tile\": %u, \"head\": %u, "
                 "\"max_width\": %u, \"invert\": %d, \"repeat\": %d,\n",
                 c.model, (unsigned)m.bytes, m.int8, m.n_classes, m.params, mode_name(ctx.mode), ctx.tile,
                 ctx.head, ctx.max_width, c.invert, c.repeat);
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
        fd_printf(fd_s, " \"lines\": %u, \"frames\": %s, \"chars\": %s, \"us_total\": %s, \"us_max_line\": %s, "
                 "\"expect\": %s, \"text_ok\": %u, \"logits_ok\": %u}\n",
                 count, u64s(frames_total, n1), u64s(chars_total, n2), u64s(us_total, n3), u64s(us_max, n4),
                 have_expect > 0 ? "true" : "false", text_ok, crc_ok);
        if (fd_s >= 0) sceIoClose(fd_s);
    }
    fd_printf(fd_res, "# done: %u lines\n", count);
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
    block_free(&b_input);
    block_free(&b_logits);
    block_free(&b_crop);
    block_free(&b_arena);
    block_free(&b_blob);
    finish(&c);
    return 0;
}
