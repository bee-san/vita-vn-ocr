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
#include "vocr_charbox.h"
#include "vocr_tsd_resize.h"
#include "vocr_lines.h"

#ifndef VOCR_DESK_NO_THREADS
#include <pthread.h>
#endif

#define MAX_THREADS 64
#define TSD_CHARS 130  /* greedy characters kept per line (a line has at most 129 CTC frames) */

/* the greedy characters of a read line (the reading stage's cut: textspot-desktop-boxes): their classes and their
 * centres in the crop's px x VOCR_CHARBOX_Q (runtime/vocr_charbox.c, flags 0, the page's calibration) */
typedef struct {
    unsigned n;
    uint16_t cls[TSD_CHARS];
    int32_t c[TSD_CHARS];
} tsd_chars;

/* an output line's characters: those of read line src in [first, end), centres from that line's crop x */
typedef struct {
    unsigned src, first, end;
    int cx;
} tsd_lchars;
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
    vocr_tsd_model det;        /* the detector (an ensemble's first member) */
    vocr_tsd_model more[VOCR_TSD_MAX_ENS - 1];   /* an ensemble's other members (vocr_tsd_page_new_ens) */
    const vocr_tsd_model *dets[VOCR_TSD_MAX_ENS];
    unsigned ndet;
    vocr_model rec;
    vocr_opts opts;
    vocr_tsd_pool *pool;
    unsigned threads, kind, n, tags, refine_on;
    unsigned rule;             /* the reading rule (VOCR_TSD_RULE_*) and its values a, b, c */
    double ra, rb, rc;
    vocr_tsdr_opts refine;     /* the box refinement's options (refine_on) */
    unsigned decode_on, post_on, drop_on;
    vocr_tsd_decode_opts decode;   /* the decode's options (decode_on; else the blob's) */
    vocr_tsdp_opts post;       /* the post-pass's (post_on) */
    vocr_tsdd_opts drop;       /* the reading filter's (drop_on) */
    rec_worker work[MAX_THREADS];
    void *arena;               /* the detector's arena (grown as frames need) */
    size_t arena_bytes;
    unsigned char *grey;       /* an RGB(A) frame's luma */
    size_t grey_cap;
    vocr_tsd_set set;
    unsigned count;
    vocr_tsd_line lines[VOCR_TSD_MAX_BOXES];
    int16_t cal[1 + VOCR_CHARBOX_GROUPS];   /* the recognizer's charbox calibration (vocr_tsd_page_set_charbox_cal) */
    unsigned cal_set;
    tsd_chars *chars;          /* per read line: its greedy characters (the cut on or the calibration set), else NULL */
    vocr_tsd_line *cut_out;    /* the cut's output lines (VOCR_TSD_MAX_BOXES) */
    tsd_lchars lch[VOCR_TSD_MAX_BOXES], lch_tmp[VOCR_TSD_MAX_BOXES];
    void *cut_scr;             /* the cut's scratch (vocr_tsdp_scratch_bytes(w, h, 1)), grown as frames need */
    size_t cut_scr_bytes;
    const unsigned char *g;    /* the grey frame being read */
    unsigned w, h, stride;
    double ph, pv;             /* the frame's pad of the last run (a frame or px rule) */
    double detect_us, total_us;
};

vocr_tsd_page *vocr_tsd_page_new(const void *det, size_t det_bytes, const void *rec, size_t rec_bytes,
                                 unsigned threads, int kind, unsigned n, unsigned pad_h, unsigned pad_v,
                                 unsigned pad_px, int *err)
{
    return vocr_tsd_page_new_ens(&det, &det_bytes, 1, rec, rec_bytes, threads, kind, n, pad_h, pad_v, pad_px, err);
}

vocr_tsd_page *vocr_tsd_page_new_ens(const void *const *dets, const size_t *det_bytes, unsigned ndet,
                                     const void *rec, size_t rec_bytes, unsigned threads, int kind, unsigned n,
                                     unsigned pad_h, unsigned pad_v, unsigned pad_px, int *err)
{
    vocr_tsd_page *p;
    unsigned i;
    size_t need;
    int rc = VOCR_TSD_OK;
    if (err) *err = VOCR_TSD_OK;
    if (!dets || !det_bytes || ndet < 1 || ndet > VOCR_TSD_MAX_ENS || !rec || threads < 1 || threads > MAX_THREADS ||
        (kind >= 0 && !vocr_tsd_policy_ok((unsigned)kind, n))) {
        if (err) *err = VOCR_TSD_E_ARG;
        return NULL;
    }
    for (i = 0; i < ndet; i++)
        if (!dets[i]) {
            if (err) *err = VOCR_TSD_E_ARG;
            return NULL;
        }
    p = (vocr_tsd_page *)calloc(1, sizeof *p);
    if (!p) {
        if (err) *err = VOCR_TSD_E_MEM;
        return NULL;
    }
    rc = vocr_tsd_load(&p->det, dets[0], det_bytes[0]);
    p->dets[0] = &p->det;
    for (i = 1; i < ndet && !rc; i++) {
        rc = vocr_tsd_load(&p->more[i - 1], dets[i], det_bytes[i]);
        p->dets[i] = &p->more[i - 1];
        /* an ensemble's members share the first's output format (vocr_tsd_ens_detect's rule) */
        if (!rc && (p->more[i - 1].hs != p->det.hs || p->more[i - 1].out != p->det.out ||
                    p->more[i - 1].tags != p->det.tags))
            rc = VOCR_TSD_E_ARG;
    }
    p->ndet = ndet;
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

void vocr_tsd_page_set_decode(vocr_tsd_page *p, const vocr_tsd_decode_opts *opts)
{
    if (!p) return;
    p->decode_on = opts != NULL;
    if (opts) p->decode = *opts;
}

void vocr_tsd_page_set_post(vocr_tsd_page *p, const vocr_tsdp_opts *opts)
{
    if (!p) return;
    p->post_on = opts != NULL;
    if (opts) p->post = *opts;
}

static int chars_alloc(vocr_tsd_page *p);

void vocr_tsd_page_set_drop(vocr_tsd_page *p, const vocr_tsdd_opts *opts)
{
    if (!p) return;
    p->drop_on = opts != NULL;
    if (opts) p->drop = *opts;
    if (opts && opts->cut_pct) chars_alloc(p);    /* vocr_tsd_page_run refuses the cut if this failed */
}

/* the kept characters' buffers, when the cut is on or the calibration set (allocated once) */
static int chars_alloc(vocr_tsd_page *p)
{
    if (p->chars) return 0;
    p->chars = (tsd_chars *)amalloc(VOCR_TSD_MAX_BOXES * sizeof *p->chars);
    p->cut_out = (vocr_tsd_line *)amalloc(VOCR_TSD_MAX_BOXES * sizeof *p->cut_out);
    if (!p->chars || !p->cut_out) {
        free(p->chars);
        free(p->cut_out);
        p->chars = NULL;
        p->cut_out = NULL;
        return VOCR_TSD_E_ARG;
    }
    memset(p->chars, 0, VOCR_TSD_MAX_BOXES * sizeof *p->chars);
    return 0;
}

int vocr_tsd_page_set_charbox_cal(vocr_tsd_page *p, const int *cal)
{
    unsigned k;
    if (!p) return VOCR_TSD_E_ARG;
    if (!cal) {
        memset(p->cal, 0, sizeof p->cal);
        p->cal_set = 0;
        return 0;
    }
    for (k = 0; k <= VOCR_CHARBOX_GROUPS; k++)
        if (cal[k] < -32768 || cal[k] > 32767) return VOCR_TSD_E_ARG;
    for (k = 0; k <= VOCR_CHARBOX_GROUPS; k++) p->cal[k] = (int16_t)cal[k];
    p->cal_set = 1;
    return chars_alloc(p);
}

void vocr_tsd_page_free(vocr_tsd_page *p)
{
    unsigned i;
    if (!p) return;
    vocr_tsd_pool_free(p->pool);
    for (i = 0; i < MAX_THREADS; i++) free(p->work[i].arena);
    free(p->arena);
    free(p->grey);
    free(p->chars);
    free(p->cut_out);
    free(p->cut_scr);
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
    vocr_tsd_opts opt;
    if (!p || !img || !w || !h || w > VOCR_TSD_MAX_SIDE || h > VOCR_TSD_MAX_SIDE ||
        (channels != 1 && channels != 3 && channels != 4) || stride < (size_t)w * channels)
        return VOCR_TSD_E_ARG;
    rc = grey_of(p, img, w, h, stride, channels);
    if (rc) return rc;
    opt.decode = p->decode_on ? &p->decode : NULL;
    opt.refine = p->refine_on ? &p->refine : NULL;
    opt.post = p->post_on ? &p->post : NULL;
    need = p->ndet > 1 ? vocr_tsd_ens_arena_bytes(p->dets, p->ndet, p->kind, p->n, w, h, 1, TASKS_OF(p->threads), &opt)
                       : vocr_tsd_arena_bytes_opt(&p->det, p->kind, p->n, w, h, 1, TASKS_OF(p->threads), &opt);
    if (!need) return VOCR_TSD_E_ARG;
    if (need > p->arena_bytes) {
        free(p->arena);
        p->arena = amalloc(need);
        p->arena_bytes = p->arena ? need : 0;
        if (!p->arena) return VOCR_TSD_E_MEM;
    }
    p->count = 0;
    if (p->ndet > 1)
        return vocr_tsd_ens_detect(p->dets, p->ndet, p->g, w, h, p->stride, 1, p->kind, p->n, &opt, p->arena,
                                   p->arena_bytes, &p->set, vocr_tsd_pool_run, p->pool, TASKS_OF(p->threads));
    return vocr_tsd_detect_opt(&p->det, p->g, w, h, p->stride, 1, p->kind, p->n, &opt, p->arena, p->arena_bytes,
                               &p->set, vocr_tsd_pool_run, p->pool, TASKS_OF(p->threads));
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

/* the reading stage's cut (textspot-desktop-boxes): the greedy characters of read line i from its frames' top-1
 * classes (vocr_charbox_find, flags 0, the page's calibration), in the crop's px x 16 */
static void keep_chars(vocr_tsd_page *p, unsigned i, const vocr_extra *ex, const uint16_t *topc,
                       const vocr_image *vi)
{
    tsd_chars *ch = &p->chars[i];
    vocr_charbox cb[TSD_CHARS];
    vocr_charbox_cal cal;
    unsigned n = 0, k;
    ch->n = 0;
    if (!ex->topk || !ex->frames) return;
    cal.shift = p->cal[0];
    for (k = 0; k < VOCR_CHARBOX_GROUPS; k++) cal.group[k] = p->cal[1 + k];
    if (vocr_charbox_find(&p->rec, topc, ex->topk, ex->frames, vi->width, vi->height, 0, &cal, NULL, NULL, cb,
                          TSD_CHARS, &n))
        return;
    for (k = 0; k < n; k++) {
        ch->cls[k] = cb[k].cls;
        ch->c[k] = cb[k].c;
    }
    ch->n = n;
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
    ln->conf_mean = 1.0f;
    ln->dark = 0;
    if (p->chars) p->chars[i].n = 0;
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
    if (ex.topk) {
        ln->conf_min = ex.conf_min;
        ln->conf_mean = ex.conf_mean;
    }
    if (p->chars) keep_chars(p, i, &ex, topc, &vi);   /* the cut's characters (only when it is on) */
    return 0;
}

/* the reading filter (vocr_tsd_page_set_drop: tools/tsdesk.drop_set on the read horizontal lines, read_frame's
 * drop): the lines it rejects leave the page, the others keep their order */
static int drop_lines(vocr_tsd_page *p)
{
    double xywh[4 * VOCR_TSD_MAX_BOXES];
    const char *text[VOCR_TSD_MAX_BOXES];
    float conf[VOCR_TSD_MAX_BOXES];
    unsigned char gone[VOCR_TSD_MAX_BOXES];
    unsigned at[VOCR_TSD_MAX_BOXES], i, nh = 0, k = 0;
    int rc;
    for (i = 0; i < p->count; i++) {
        const vocr_tsd_line *ln = &p->lines[i];
        if (ln->vertical) continue;
        xywh[4 * nh] = ln->bx;
        xywh[4 * nh + 1] = ln->by;
        xywh[4 * nh + 2] = ln->bw;
        xywh[4 * nh + 3] = ln->bh;
        text[nh] = ln->text;
        conf[nh] = ln->conf_mean;
        at[nh++] = i;
    }
    rc = vocr_tsdd_drop(xywh, text, conf, nh, &p->drop, gone);
    if (rc < 0) return VOCR_TSD_E_ARG;
    if (!rc) return VOCR_TSD_OK;
    for (i = 0; i < nh; i++)
        if (gone[i]) p->lines[at[i]].h = -1;   /* marked */
    for (i = 0; i < p->count; i++)
        if (p->lines[i].h >= 0) {
            if (k != i) p->lines[k] = p->lines[i];
            if (k != i) p->lch[k] = p->lch[i];
            k++;
        }
    p->count = k;
    return VOCR_TSD_OK;
}

/* the reading rule's crop of an output piece's tight box: vocr_tsd_page_run's arithmetic for a box (the frame or px
 * pad of the run, the line rule's own pad, the clip rule's sides against the frame's boxes) */
static int piece_crop(const vocr_tsd_page *p, vocr_tsd_line *o)
{
    vocr_tsd_box b;
    double ph = p->ph, pv = p->pv, sd[4], x0, y0, x1, y1;
    memset(&b, 0, sizeof b);
    b.x = o->bx;
    b.y = o->by;
    b.w = o->bw;
    b.h = o->bh;
    if (p->rule == VOCR_TSD_RULE_LINE || p->rule == VOCR_TSD_RULE_CLIP) line_pad(p, &b, &ph, &pv);
    sd[0] = sd[2] = ph;
    sd[1] = sd[3] = pv;
    if (p->rule == VOCR_TSD_RULE_CLIP) clip_sides(&p->set, &b, ph, pv, sd);
    x0 = floor(b.x - sd[0]);
    y0 = floor(b.y - sd[1]);
    x1 = ceil(b.x + b.w + sd[2]);
    y1 = ceil(b.y + b.h + sd[3]);
    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;
    if (x1 > (double)p->w) x1 = (double)p->w;
    if (y1 > (double)p->h) y1 = (double)p->h;
    if (!(x1 > x0 && y1 > y0)) return 0;
    o->x = (int)x0;
    o->y = (int)y0;
    o->w = (int)(x1 - x0);
    o->h = (int)(y1 - y0);
    return 1;
}

/* the UTF-8 code points of a string (tools/tsdesk's len of a piece's text) */
static unsigned cps(const char *t)
{
    unsigned n = 0;
    for (; *t; t++) n += ((unsigned char)*t & 0xC0u) != 0x80u;
    return n;
}

/* the reading stage's cut (tools/tsdesk.cut_and_filter's cut_read_lines; vocr_tsdc_cut): each read horizontal line
 * cut into items in its place, a piece keeping the line's fields but its tight box (x0, y, x1 - x0, h), the rule's
 * crop of it and its characters' text; a line whose pieces would take the page past VOCR_TSD_MAX_BOXES lines stays
 * whole. With the cut off (only the calibration set) the lines stay as they are and keep their characters. */
static int cut_lines(vocr_tsd_page *p)
{
    unsigned i, k = 0;
    const int on = p->drop_on && p->drop.cut_pct;
    size_t need;
    if (on) {
        if (p->drop.cut_pct < 0 || p->drop.cut_rel_pct < 0) return VOCR_TSD_E_ARG;
        need = vocr_tsdp_scratch_bytes(p->w, p->h, 1);
        if (need > p->cut_scr_bytes) {
            free(p->cut_scr);
            p->cut_scr = amalloc(need);
            p->cut_scr_bytes = p->cut_scr ? need : 0;
            if (!p->cut_scr) return VOCR_TSD_E_ARG;
        }
    }
    for (i = 0; i < p->count; i++) {
        const vocr_tsd_line *ln = &p->lines[i];
        const tsd_chars *ch = &p->chars[i];
        const unsigned left = p->count - i - 1, n = ln->vertical ? 0 : ch->n;
        double cx[TSD_CHARS], x0[TSD_CHARS], x1[TSD_CHARS], box[4];
        unsigned char mark[TSD_CHARS];
        unsigned first[TSD_CHARS], end[TSD_CHARS], np = 0, j, q;
        int rc = 0;
        if (on && n) {
            for (j = 0; j < n; j++) {
                unsigned len = 0;
                const char *t = vocr_class_text(&p->rec, ch->cls[j], &len);
                cx[j] = (double)ln->x + (double)ch->c[j] / (double)VOCR_CHARBOX_Q;
                mark[j] = (unsigned char)(t ? vocr_tsdc_is_mark(t, len) : 0);
            }
            box[0] = ln->bx;
            box[1] = ln->by;
            box[2] = ln->bx + ln->bw;
            box[3] = ln->by + ln->bh;
            rc = vocr_tsdc_cut(p->g, p->w, p->h, p->stride, box, cx, mark, n, p->drop.cut_pct, p->drop.cut_rel_pct,
                               p->cut_scr, p->cut_scr_bytes, x0, x1, first, end, &np);
            if (rc < 0) return VOCR_TSD_E_ARG;
        }
        /* the pieces' texts (their characters' classes, whole characters within VOCR_TSD_TEXT) */
        if (rc == 1) {
            for (q = 0; q < np; q++) {
                vocr_tsd_line *o = &p->cut_out[k + q < VOCR_TSD_MAX_BOXES ? k + q : VOCR_TSD_MAX_BOXES - 1];
                size_t used = 0;
                if (k + np + left > VOCR_TSD_MAX_BOXES) break;
                for (j = first[q]; j < end[q]; j++) {
                    unsigned len = 0;
                    const char *t = vocr_class_text(&p->rec, ch->cls[j], &len);
                    if (!t || used + len >= sizeof o->text) break;
                    memcpy(o->text + used, t, len);
                    used += len;
                }
                o->text[used] = '\0';
            }
            if (k + np + left > VOCR_TSD_MAX_BOXES) rc = 0;
        }
        if (rc == 1 && ln->tag == 0 && p->drop.cut_dlg != 1) {   /* a dialogue-tagged line (cut_dlg) */
            if (p->drop.cut_dlg == 0) rc = 0;
            else
                for (q = 0; q < np; q++)
                    if (cps(p->cut_out[k + q].text) > (unsigned)p->drop.cut_dlg) {
                        rc = 0;
                        break;
                    }
        }
        if (rc != 1) {
            p->cut_out[k] = *ln;
            p->lch_tmp[k].src = i;
            p->lch_tmp[k].first = 0;
            p->lch_tmp[k].end = n;
            p->lch_tmp[k].cx = ln->x;
            k++;
            continue;
        }
        for (q = 0; q < np; q++) {
            vocr_tsd_line *o = &p->cut_out[k];
            char text[VOCR_TSD_TEXT];
            memcpy(text, o->text, sizeof text);
            *o = *ln;
            memcpy(o->text, text, sizeof text);
            o->bx = x0[q];
            o->bw = x1[q] - x0[q];
            if (!piece_crop(p, o)) {
                o->x = ln->x;
                o->w = 0;
            }
            p->lch_tmp[k].src = i;
            p->lch_tmp[k].first = first[q];
            p->lch_tmp[k].end = end[q];
            p->lch_tmp[k].cx = ln->x;
            k++;
        }
    }
    memcpy(p->lines, p->cut_out, k * sizeof *p->lines);
    memcpy(p->lch, p->lch_tmp, k * sizeof *p->lch);
    p->count = k;
    return 0;
}

unsigned vocr_tsd_page_line_chars(const vocr_tsd_page *p, unsigned i, int32_t *c16, unsigned *off, unsigned cap)
{
    unsigned j, n = 0, used = 0;
    const tsd_chars *ch;
    const tsd_lchars *lc;
    if (!p || !p->chars || i >= p->count || p->lines[i].vertical) return 0;
    lc = &p->lch[i];
    ch = &p->chars[lc->src];
    for (j = lc->first; j < lc->end && n < cap; j++, n++) {
        unsigned len = 0;
        if (c16) c16[n] = (int32_t)lc->cx * VOCR_CHARBOX_Q + ch->c[j];
        if (off) off[n] = used;
        (void)vocr_class_text(&p->rec, ch->cls[j], &len);
        used += len;
    }
    return n;
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
    if (p->drop_on && p->drop.cut_pct && !p->chars) return VOCR_TSD_E_ARG;
    if (p->chars && (rc = cut_lines(p)) != 0) return rc;
    if (p->drop_on && (rc = drop_lines(p)) != 0) return rc;
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
