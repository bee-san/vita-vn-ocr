/* vocr_tsd_page.c - textspot-desktop's page pipeline and thread pool (vocr_tsd.h): the detector (vocr_tsd.c), then
 * every horizontal box grown by the reading pad and read by VOCR's recognizer (runtime/vocr.c, W8I, greedy, the
 * blob's head sections on, as desktop/vocr_desk_page.c reads), the lines on the pool (one recognizer arena per
 * thread). POSIX threads unless VOCR_DESK_NO_THREADS. */
#define _POSIX_C_SOURCE 200112L
#include "vocr_tsd.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "vocr.h"
#include "vocr_tsd_resize.h"
#include "vocr_lines.h"

#ifndef VOCR_DESK_NO_THREADS
#include <pthread.h>
#endif

#define MAX_THREADS 64
/* tasks per detector step for a pool of n threads: one row band per thread (the split never changes the bits) */
#define TASKS_OF(n) (n)
#define SPIN 4000u   /* pause iterations a pool thread spins on before it sleeps (about 0.1 ms) */

static double now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return 1e6 * (double)ts.tv_sec + 1e-3 * (double)ts.tv_nsec;
}

static void *amalloc(size_t n)
{
    void *p = NULL;
    if (posix_memalign(&p, 64, n ? n : 64)) return NULL;
    return p;
}

/* ----------------------------------------------------------------------------- the pool */

typedef void (*worker_fn)(void *arg, unsigned task, unsigned worker);

typedef struct {
    vocr_tsd_pool *p;
    unsigned index;
} pool_arg;

struct vocr_tsd_pool {
    unsigned threads;
    worker_fn fn;              /* the current job: fn(arg, task, worker) for task < ntasks */
    void *arg;
    unsigned ntasks, fixed;    /* fixed: worker w runs tasks w, w + threads, ... (else they are handed out) */
    unsigned next, checked, gen, quit;   /* atomics (GCC builtins) */
#ifndef VOCR_DESK_NO_THREADS
    pthread_mutex_t mu;
    pthread_cond_t go;
    unsigned sleepers, started;
    pthread_t th[MAX_THREADS];
    pool_arg who[MAX_THREADS];
#endif
};

#define A_LOAD(x) __atomic_load_n(&(x), __ATOMIC_ACQUIRE)
#define A_STORE(x, v) __atomic_store_n(&(x), (v), __ATOMIC_RELEASE)
#define A_ADD(x, v) __atomic_fetch_add(&(x), (v), __ATOMIC_ACQ_REL)

#ifndef VOCR_DESK_NO_THREADS
static void relax(void)
{
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#endif
}

static void pool_work(vocr_tsd_pool *p, unsigned worker)
{
    if (p->fixed) {   /* the same rows on the same thread step after step: its caches hold them */
        unsigned t;
        for (t = worker; t < p->ntasks; t += p->threads) p->fn(p->arg, t, worker);
        return;
    }
    for (;;) {
        unsigned t = A_ADD(p->next, 1u);
        if (t >= p->ntasks) break;
        p->fn(p->arg, t, worker);
    }
}

static void *pool_main(void *arg)
{
    pool_arg *a = (pool_arg *)arg;
    vocr_tsd_pool *p = a->p;
    unsigned seen = 0;
    for (;;) {
        unsigned g, spins = 0;
        while ((g = A_LOAD(p->gen)) == seen) {
            if (A_LOAD(p->quit)) return NULL;
            if (++spins < SPIN) {
                relax();
                continue;
            }
            pthread_mutex_lock(&p->mu);
            p->sleepers++;
            while (A_LOAD(p->gen) == seen && !A_LOAD(p->quit)) pthread_cond_wait(&p->go, &p->mu);
            p->sleepers--;
            pthread_mutex_unlock(&p->mu);
            spins = 0;
        }
        seen = g;
        pool_work(p, a->index);
        A_ADD(p->checked, 1u);
    }
}
#endif

vocr_tsd_pool *vocr_tsd_pool_new(unsigned threads)
{
    vocr_tsd_pool *p;
    if (threads < 1 || threads > MAX_THREADS) return NULL;
    p = (vocr_tsd_pool *)calloc(1, sizeof *p);
    if (!p) return NULL;
    p->threads = 1;
#ifndef VOCR_DESK_NO_THREADS
    if (threads > 1) {
        unsigned i;
        pthread_mutex_init(&p->mu, NULL);
        pthread_cond_init(&p->go, NULL);
        p->started = 1;
        for (i = 1; i < threads; i++) {
            p->who[i].p = p;
            p->who[i].index = i;
            if (pthread_create(&p->th[i], NULL, pool_main, &p->who[i])) break;
            p->threads = i + 1;
        }
        if (p->threads != threads) {
            vocr_tsd_pool_free(p);
            return NULL;
        }
    }
#endif
    return p;
}

unsigned vocr_tsd_pool_threads(const vocr_tsd_pool *p)
{
    return p ? p->threads : 0;
}

/* run fn(arg, task, worker) for every task on the pool (the caller is worker 0); returns when all are done. fixed:
 * worker w runs tasks w, w + threads, ... (the detector's row bands: each thread keeps its rows across steps), else
 * the tasks are handed out as threads come free (the lines) */
static void pool_run_w(vocr_tsd_pool *p, worker_fn fn, void *arg, unsigned ntasks, unsigned fixed)
{
    unsigned t;
#ifdef VOCR_DESK_NO_THREADS
    (void)p;
    (void)fixed;
#else
    if (p->threads > 1 && ntasks > 1) {
        p->fn = fn;
        p->arg = arg;
        p->ntasks = ntasks;
        p->fixed = fixed;
        A_STORE(p->next, 0u);
        A_STORE(p->checked, 0u);
        A_STORE(p->gen, p->gen + 1u);
        pthread_mutex_lock(&p->mu);
        if (p->sleepers) pthread_cond_broadcast(&p->go);
        pthread_mutex_unlock(&p->mu);
        pool_work(p, 0);
        while (A_LOAD(p->checked) < p->threads - 1) relax();
        return;
    }
#endif
    for (t = 0; t < ntasks; t++) fn(arg, t, 0);
}

typedef struct {               /* a vocr_tsd_runner job on the pool */
    vocr_tsd_task_fn fn;
    void *arg;
    unsigned ntasks;
} runner_job;

static void runner_task(void *arg, unsigned task, unsigned worker)
{
    const runner_job *j = (const runner_job *)arg;
    (void)worker;
    j->fn(j->arg, task, j->ntasks);
}

void vocr_tsd_pool_run(void *pool, vocr_tsd_task_fn fn, void *arg, unsigned ntasks)
{
    runner_job j;
    j.fn = fn;
    j.arg = arg;
    j.ntasks = ntasks;
    pool_run_w((vocr_tsd_pool *)pool, runner_task, &j, ntasks, 1);
}

void vocr_tsd_pool_free(vocr_tsd_pool *p)
{
    if (!p) return;
#ifndef VOCR_DESK_NO_THREADS
    if (p->started) {
        unsigned i;
        pthread_mutex_lock(&p->mu);
        A_STORE(p->quit, 1u);
        pthread_cond_broadcast(&p->go);
        pthread_mutex_unlock(&p->mu);
        for (i = 1; i < p->threads; i++) pthread_join(p->th[i], NULL);
        pthread_cond_destroy(&p->go);
        pthread_mutex_destroy(&p->mu);
    }
#endif
    free(p);
}

/* ----------------------------------------------------------------------------- the page */

typedef struct {
    vocr_ctx ctx;
    void *arena;
    int rc;
} rec_worker;

struct vocr_tsd_page {
    vocr_tsd_model det;
    vocr_model rec;
    vocr_opts opts;
    vocr_tsd_pool *pool;
    unsigned threads, kind, n, tags, refine_on;
    unsigned rule;             /* the reading rule (VOCR_TSD_RULE_*) and its values a, b, c */
    double ra, rb, rc;
    vocr_tsdr_opts refine;     /* the box refinement's options (refine_on) */
    rec_worker work[MAX_THREADS];
    void *arena;               /* the detector's arena (grown as frames need) */
    size_t arena_bytes;
    unsigned char *grey;       /* an RGB(A) frame's luma */
    size_t grey_cap;
    vocr_tsd_set set;
    unsigned count;
    vocr_tsd_line lines[VOCR_TSD_MAX_BOXES];
    const unsigned char *g;    /* the grey frame being read */
    unsigned w, h, stride;
    double ph, pv;             /* the frame's pad of the last run (a frame or px rule) */
    double detect_us, total_us;
};

vocr_tsd_page *vocr_tsd_page_new(const void *det, size_t det_bytes, const void *rec, size_t rec_bytes,
                                 unsigned threads, int kind, unsigned n, unsigned pad_h, unsigned pad_v,
                                 unsigned pad_px, int *err)
{
    vocr_tsd_page *p;
    unsigned i;
    size_t need;
    int rc = VOCR_TSD_OK;
    if (err) *err = VOCR_TSD_OK;
    if (!det || !rec || threads < 1 || threads > MAX_THREADS ||
        (kind >= 0 && !vocr_tsd_policy_ok((unsigned)kind, n))) {
        if (err) *err = VOCR_TSD_E_ARG;
        return NULL;
    }
    p = (vocr_tsd_page *)calloc(1, sizeof *p);
    if (!p) {
        if (err) *err = VOCR_TSD_E_MEM;
        return NULL;
    }
    rc = vocr_tsd_load(&p->det, det, det_bytes);
    if (!rc && vocr_load(&p->rec, rec, rec_bytes)) rc = VOCR_TSD_E_REC;
    if (rc) {
        free(p);
        if (err) *err = rc;
        return NULL;
    }
    p->threads = threads;
    p->kind = kind < 0 ? p->det.size_kind : (unsigned)kind;
    p->n = kind < 0 ? p->det.size_n : n;
    p->rule = pad_px ? VOCR_TSD_RULE_PX : VOCR_TSD_RULE_FRAME;
    p->ra = (double)pad_h;
    p->rb = (double)pad_v;
    p->rc = 2.0;
    p->tags = ~0u;
    p->opts.mode = VOCR_MODE_W8I;
    p->opts.head_gate = p->rec.gate.hidden ? VOCR_HEAD_GATE_ON : VOCR_HEAD_GATE_OFF;
    p->opts.two_level = p->rec.two.l1_blocks ? 1u : 0u;
    need = vocr_arena_bytes(&p->rec, &p->opts);
    for (i = 0; i < threads && !rc; i++) {
        rec_worker *wk = &p->work[i];
        wk->arena = need ? amalloc(need) : NULL;
        if (!wk->arena || vocr_init(&wk->ctx, &p->rec, &p->opts, wk->arena, need))
            rc = wk->arena ? VOCR_TSD_E_REC : VOCR_TSD_E_MEM;
    }
    if (!rc) {
        p->pool = vocr_tsd_pool_new(threads);
        if (!p->pool) rc = VOCR_TSD_E_MEM;
    }
    if (rc) {
        vocr_tsd_page_free(p);
        if (err) *err = rc;
        return NULL;
    }
    return p;
}

void vocr_tsd_page_set_tags(vocr_tsd_page *p, unsigned mask)
{
    if (p) p->tags = mask;
}

static int rule_value_ok(double v)
{
    return v >= 0.0 && v <= 4096.0;   /* false for NaN */
}

int vocr_tsd_page_set_rule(vocr_tsd_page *p, unsigned kind, double a, double b, double c)
{
    if (!p || kind > VOCR_TSD_RULE_CLIP || !rule_value_ok(a) || !rule_value_ok(b) ||
        (kind == VOCR_TSD_RULE_FRAME && !rule_value_ok(c)))
        return VOCR_TSD_E_ARG;
    p->rule = kind;
    p->ra = a;
    p->rb = b;
    p->rc = kind == VOCR_TSD_RULE_FRAME ? c : 0.0;
    return VOCR_TSD_OK;
}

void vocr_tsd_page_set_refine(vocr_tsd_page *p, const vocr_tsdr_opts *opts)
{
    if (!p) return;
    p->refine_on = opts != NULL;
    if (opts) p->refine = *opts;
}

void vocr_tsd_page_free(vocr_tsd_page *p)
{
    unsigned i;
    if (!p) return;
    vocr_tsd_pool_free(p->pool);
    for (i = 0; i < MAX_THREADS; i++) free(p->work[i].arena);
    free(p->arena);
    free(p->grey);
    free(p);
}

/* the luma of an RGB(A) frame (vocr_tsd_grey) into the page's buffer; a grey frame as it is */
static int grey_of(vocr_tsd_page *p, const unsigned char *img, unsigned w, unsigned h, unsigned stride,
                   unsigned channels)
{
    if (channels == 1) {
        p->g = img;
        p->stride = stride;
    } else {
        if ((size_t)w * h > p->grey_cap) {
            free(p->grey);
            p->grey = (unsigned char *)amalloc((size_t)w * h);
            p->grey_cap = p->grey ? (size_t)w * h : 0;
            if (!p->grey) return VOCR_TSD_E_MEM;
        }
        vocr_tsd_grey(img, w, h, stride, channels, p->grey);
        p->g = p->grey;
        p->stride = w;
    }
    p->w = w;
    p->h = h;
    return VOCR_TSD_OK;
}

static int detect(vocr_tsd_page *p, const unsigned char *img, unsigned w, unsigned h, unsigned stride,
                  unsigned channels)
{
    size_t need;
    int rc;
    if (!p || !img || !w || !h || w > VOCR_TSD_MAX_SIDE || h > VOCR_TSD_MAX_SIDE ||
        (channels != 1 && channels != 3 && channels != 4) || stride < (size_t)w * channels)
        return VOCR_TSD_E_ARG;
    rc = grey_of(p, img, w, h, stride, channels);
    if (rc) return rc;
    need = vocr_tsd_arena_bytes_ex(&p->det, p->kind, p->n, w, h, 1, TASKS_OF(p->threads), (int)p->refine_on);
    if (!need) return VOCR_TSD_E_ARG;
    if (need > p->arena_bytes) {
        free(p->arena);
        p->arena = amalloc(need);
        p->arena_bytes = p->arena ? need : 0;
        if (!p->arena) return VOCR_TSD_E_MEM;
    }
    p->count = 0;
    return vocr_tsd_detect_ex(&p->det, p->g, w, h, p->stride, 1, p->kind, p->n, p->refine_on ? &p->refine : NULL,
                              p->arena, p->arena_bytes, &p->set, vocr_tsd_pool_run, p->pool, TASKS_OF(p->threads));
}

int vocr_tsd_page_detect(vocr_tsd_page *p, const unsigned char *img, unsigned w, unsigned h, unsigned stride,
                         unsigned channels)
{
    double t0 = now_us();
    int rc = detect(p, img, w, h, stride, channels);
    if (p) p->detect_us = p->total_us = now_us() - t0;
    return rc;
}

/* reading order (tools/tsdesk.reading_order) of the boxes ids[0 .. n): by (centre y, x), stable; rows by vertical
 * overlap with the row's first box of at least half the smaller height; left to right (stable) within a row */
static void reading_order(const vocr_tsd_box *bx, const unsigned *ids, unsigned n, unsigned *order)
{
    unsigned idx[VOCR_TSD_MAX_BOXES], row_of[VOCR_TSD_MAX_BOXES], first[VOCR_TSD_MAX_BOXES];
    unsigned i, j, rows = 0, k = 0, r;
    for (i = 0; i < n; i++) idx[i] = ids[i];
    for (i = 1; i < n; i++) {   /* stable insertion sort by (y + h / 2, x) */
        const unsigned v = idx[i];
        const double ky = bx[v].y + bx[v].h / 2.0, kx = bx[v].x;
        j = i;
        while (j > 0) {
            const vocr_tsd_box *b = &bx[idx[j - 1]];
            const double by = b->y + b->h / 2.0;
            if (by > ky || (by == ky && b->x > kx)) {
                idx[j] = idx[j - 1];
                j--;
            } else {
                break;
            }
        }
        idx[j] = v;
    }
    for (i = 0; i < n; i++) {
        const vocr_tsd_box *b = &bx[idx[i]];
        if (rows) {
            const vocr_tsd_box *f = &bx[first[rows - 1]];
            const double fb = f->y + f->h, bb = b->y + b->h;
            const double top = b->y > f->y ? b->y : f->y, bot = bb < fb ? bb : fb;
            const double mh = b->h < f->h ? b->h : f->h;
            if (bot - top >= 0.5 * mh) {
                row_of[i] = rows - 1;
                continue;
            }
        }
        first[rows] = idx[i];
        row_of[i] = rows++;
    }
    for (r = 0; r < rows; r++) {
        const unsigned start = k;
        for (i = 0; i < n; i++)
            if (row_of[i] == r) order[k++] = idx[i];
        for (i = start + 1; i < k; i++) {   /* stable by x */
            const unsigned v = order[i];
            j = i;
            while (j > start && bx[order[j - 1]].x > bx[v].x) {
                order[j] = order[j - 1];
                j--;
            }
            order[j] = v;
        }
    }
}

/* the frame rule's pad of an H px tall frame: max(lo, nearbyint(k H / 544)) (eval/tsdesk_eval.pad_px / floor_hv) */
static double frame_pad(double k, double lo, unsigned H)
{
    const double v = nearbyint(k * (double)H / 544.0);
    return v < lo ? lo : v;
}

/* the clip rule's sides of box b (eval/tsdesk_eval.clip_sides on [x, y, x + w, y + h] against every box of the set):
 * s[0..3] = left, top, right, bottom, from ph / pv */
static void clip_sides(const vocr_tsd_set *set, const vocr_tsd_box *b, double ph, double pv, double *s)
{
    const double x0 = b->x, y0 = b->y, x1 = b->x + b->w, y1 = b->y + b->h;
    double left = ph, top = pv, right = ph, bottom = pv;
    unsigned k;
    for (k = 0; k < set->count; k++) {
        const vocr_tsd_box *o = &set->boxes[k];
        const double ox0 = o->x, oy0 = o->y, ox1 = o->x + o->w, oy1 = o->y + o->h;
        const double across = (y1 < oy1 ? y1 : oy1) - (y0 > oy0 ? y0 : oy0);
        const double along = (x1 < ox1 ? x1 : ox1) - (x0 > ox0 ? x0 : ox0);
        const double ha = y1 - y0, hb = oy1 - oy0;
        double g;
        if (ox0 == x0 && oy0 == y0 && ox1 == x1 && oy1 == y1) continue;
        if (across >= 0.5 * (hb < ha ? hb : ha) && along <= 0.0) {
            if (ox0 >= x1) {
                g = (ox0 - x1) / 2.0;
                if (g < right) right = g;
            } else if (ox1 <= x0) {
                g = (x0 - ox1) / 2.0;
                if (g < left) left = g;
            }
        } else if (along > 0.0 && across <= 0.0) {
            if (oy0 >= y1) {
                g = (oy0 - y1) / 2.0;
                if (g < bottom) bottom = g;
            } else if (oy1 <= y0) {
                g = (y0 - oy1) / 2.0;
                if (g < top) top = g;
            }
        }
    }
    s[0] = left;
    s[1] = top;
    s[2] = right;
    s[3] = bottom;
}

/* the line rule's pad of a box (eval/tsdesk_eval.rel_hv on [x, y, x + w, y + h]) */
static void line_pad(const vocr_tsd_page *p, const vocr_tsd_box *b, double *ph, double *pv)
{
    double bw = (b->x + b->w) - b->x, bh = (b->y + b->h) - b->y, s, u, v;
    bw = bw > 1.0 ? bw : 1.0;
    bh = bh > 1.0 ? bh : 1.0;
    s = bw >= bh ? bh : bw;
    u = (bw >= bh ? p->ra : p->rb) * s;
    v = (bw >= bh ? p->rb : p->ra) * s;
    *ph = u > 1.0 ? u : 1.0;
    *pv = v > 1.0 ? v : 1.0;
}

static int read_line(vocr_tsd_page *p, rec_worker *wk, unsigned i)
{
    vocr_tsd_line *ln = &p->lines[i];
    vocr_image vi;
    vocr_extra ex;
    VocrLinesImage im;
    float topv[8 * 129];
    uint16_t topc[8 * 129];
    int dark, rc;
    unsigned cap;
    ln->text[0] = '\0';
    ln->conf_min = 1.0f;
    ln->dark = 0;
    if (ln->vertical) return 0;
    im.pixels = p->g;
    im.width = p->w;
    im.height = p->h;
    im.stride = p->stride;
    im.bpp = VOCR_LINES_GREY;
    dark = vocr_line_dark(&im, (unsigned)ln->x, (unsigned)ln->y, (unsigned)ln->w, (unsigned)ln->h);
    ln->dark = dark > 0;
    vi.pixels = p->g + (size_t)ln->y * p->stride + (size_t)ln->x;
    vi.width = (unsigned)ln->w;
    vi.height = (unsigned)ln->h;
    vi.stride = p->stride;
    vi.format = VOCR_GREY8 | (ln->dark ? VOCR_INVERT : 0);
    memset(&ex, 0, sizeof ex);
    cap = vocr_resized_width(&p->rec, vi.width, vi.height) / p->rec.stride + 1;
    if (cap <= 129) {
        ex.topk = 8;
        ex.topk_logit = topv;
        ex.topk_class = topc;
        ex.cap_frames = cap;
    }
    rc = vocr_run(&wk->ctx, &vi, ln->text, sizeof ln->text, &ex);
    if (rc && rc != VOCR_E_TEXT) return VOCR_TSD_E_REC;
    if (ex.topk) ln->conf_min = ex.conf_min;
    return 0;
}

static void line_task(void *arg, unsigned task, unsigned worker)
{
    vocr_tsd_page *p = (vocr_tsd_page *)arg;
    rec_worker *wk = &p->work[worker];
    int rc = read_line(p, wk, task);
    if (rc) wk->rc = rc;
}

int vocr_tsd_page_run(vocr_tsd_page *p, const unsigned char *img, unsigned w, unsigned h, unsigned stride,
                      unsigned channels)
{
    const double t0 = now_us();
    unsigned i, k = 0, n = 0, ids[VOCR_TSD_MAX_BOXES], order[VOCR_TSD_MAX_BOXES];
    int rc = detect(p, img, w, h, stride, channels);
    if (!p) return VOCR_TSD_E_ARG;
    p->detect_us = now_us() - t0;
    if (rc) return rc;
    if (p->rule == VOCR_TSD_RULE_PX) {
        p->ph = p->ra;
        p->pv = p->rb;
    } else if (p->rule == VOCR_TSD_RULE_FRAME) {
        p->ph = frame_pad(p->ra, p->rc, h);
        p->pv = frame_pad(p->rb, 2.0, h);
    } else {
        p->ph = p->pv = -1.0;
    }
    for (i = 0; i < p->set.count; i++)
        if (p->set.boxes[i].tag < 32 && (p->tags >> p->set.boxes[i].tag & 1u)) ids[n++] = i;
    reading_order(p->set.boxes, ids, n, order);
    for (i = 0; i < n; i++) {
        const vocr_tsd_box *b = &p->set.boxes[order[i]];
        double ph = p->ph, pv = p->pv, sd[4], x0, y0, x1, y1;
        vocr_tsd_line *ln = &p->lines[k];
        if (p->rule == VOCR_TSD_RULE_LINE || p->rule == VOCR_TSD_RULE_CLIP) line_pad(p, b, &ph, &pv);
        sd[0] = sd[2] = ph;
        sd[1] = sd[3] = pv;
        if (p->rule == VOCR_TSD_RULE_CLIP) clip_sides(&p->set, b, ph, pv, sd);
        x0 = floor(b->x - sd[0]);
        y0 = floor(b->y - sd[1]);
        x1 = ceil(b->x + b->w + sd[2]);
        y1 = ceil(b->y + b->h + sd[3]);
        if (x0 < 0) x0 = 0;
        if (y0 < 0) y0 = 0;
        if (x1 > (double)w) x1 = (double)w;
        if (y1 > (double)h) y1 = (double)h;
        if (!(x1 > x0 && y1 > y0)) continue;
        ln->x = (int)x0;
        ln->y = (int)y0;
        ln->w = (int)(x1 - x0);
        ln->h = (int)(y1 - y0);
        ln->bx = b->x;
        ln->by = b->y;
        ln->bw = b->w;
        ln->bh = b->h;
        ln->logit = b->logit;
        ln->tag = b->tag;
        ln->vertical = b->vertical;
        k++;
    }
    p->count = k;
    for (i = 0; i < p->threads; i++) p->work[i].rc = 0;
    pool_run_w(p->pool, line_task, p, k, 0);
    for (i = 0; i < p->threads; i++)
        if (p->work[i].rc) return p->work[i].rc;
    p->total_us = now_us() - t0;
    return VOCR_TSD_OK;
}

unsigned vocr_tsd_page_count(const vocr_tsd_page *p)
{
    return p ? p->count : 0;
}

const vocr_tsd_line *vocr_tsd_page_line(const vocr_tsd_page *p, unsigned i)
{
    return p && i < p->count ? &p->lines[i] : NULL;
}

size_t vocr_tsd_line_bytes(void)
{
    return sizeof(vocr_tsd_line);
}

const vocr_tsd_set *vocr_tsd_page_boxes(const vocr_tsd_page *p)
{
    return p ? &p->set : NULL;
}

void vocr_tsd_page_pad(const vocr_tsd_page *p, int *ph, int *pv)
{
    if (ph) *ph = p ? (int)p->ph : 0;
    if (pv) *pv = p ? (int)p->pv : 0;
}

double vocr_tsd_page_detect_us(const vocr_tsd_page *p)
{
    return p ? p->detect_us : 0.0;
}

double vocr_tsd_page_total_us(const vocr_tsd_page *p)
{
    return p ? p->total_us : 0.0;
}
