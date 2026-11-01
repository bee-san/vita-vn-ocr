/* vocr_desk_page.c - the desktop page pipeline (vocr_desk.h): the desktop detector, then every horizontal box read by
 * VOCR's recognizer (runtime/vocr.c, W8I, greedy, with the blob's head sections as Rubi runs them), the lines on a
 * pool of threads (one recognizer arena each). POSIX threads unless VOCR_DESK_NO_THREADS. */
#define _POSIX_C_SOURCE 200112L
#include "vocr_desk.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "vocr.h"
#include "vocr_lines.h"

#ifndef VOCR_DESK_NO_THREADS
#include <pthread.h>
#endif

#define MAX_THREADS 64

typedef struct {
    int x, y, w, h;
} crop_box;

struct vocr_desk_page;

typedef struct {
    struct vocr_desk_page *page;
    vocr_ctx ctx;
    void *arena;
    unsigned index;
#ifndef VOCR_DESK_NO_THREADS
    pthread_t thread;
#endif
    int rc;
} worker;

struct vocr_desk_page {
    vocr_desk_model det;
    unsigned char *grey_buf;     /* vocr_desk_page_run_rgb's luma */
    size_t grey_cap;
    vocr_model rec;
    vocr_opts opts;
    unsigned threads, kind, n;
    void *arena;
    size_t arena_bytes;
    vocr_desk_set set;
    unsigned count;
    vocr_desk_line lines[VOCR_DESK_MAX_BOXES];
    crop_box crops[VOCR_DESK_MAX_BOXES];
    unsigned order[VOCR_DESK_MAX_BOXES];
    worker work[MAX_THREADS];
    /* the frame of the current run */
    const unsigned char *grey;
    unsigned w, h, stride;
    double detect_us, total_us;
#ifndef VOCR_DESK_NO_THREADS
    pthread_mutex_t mu;
    pthread_cond_t go, done;
    unsigned generation, next, finished, quit, started;
#endif
};

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

/* read line i of the current run with worker wk */
static int read_line(vocr_desk_page *p, worker *wk, unsigned i)
{
    const crop_box *c = &p->crops[i];
    vocr_desk_line *ln = &p->lines[i];
    vocr_image vi;
    vocr_extra ex;
    VocrLinesImage im;
    float topv[8 * 129];
    uint16_t topc[8 * 129];
    int dark, rc;
    unsigned cap;
    ln->text[0] = '\0';
    ln->conf_min = 1.0f;
    if (ln->vertical) return 0;
    im.pixels = p->grey;
    im.width = p->w;
    im.height = p->h;
    im.stride = p->stride;
    im.bpp = VOCR_LINES_GREY;
    dark = vocr_line_dark(&im, (unsigned)c->x, (unsigned)c->y, (unsigned)c->w, (unsigned)c->h);
    ln->dark = dark > 0;
    vi.pixels = p->grey + (size_t)c->y * p->stride + (size_t)c->x;
    vi.width = (unsigned)c->w;
    vi.height = (unsigned)c->h;
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
    if (rc && rc != VOCR_E_TEXT) return VOCR_DESK_E_REC;
    if (ex.topk) ln->conf_min = ex.conf_min;
    return 0;
}

#ifndef VOCR_DESK_NO_THREADS
static void work_lines(vocr_desk_page *p, worker *wk)
{
    for (;;) {
        unsigned i;
        int rc;
        pthread_mutex_lock(&p->mu);
        i = p->next < p->count ? p->next++ : p->count;
        pthread_mutex_unlock(&p->mu);
        if (i >= p->count) break;
        rc = read_line(p, wk, p->order[i]);
        if (rc) wk->rc = rc;
    }
}

static void *thread_main(void *arg)
{
    worker *wk = (worker *)arg;
    vocr_desk_page *p = wk->page;
    unsigned seen = 0;
    for (;;) {
        pthread_mutex_lock(&p->mu);
        while (p->generation == seen && !p->quit) pthread_cond_wait(&p->go, &p->mu);
        if (p->quit) {
            pthread_mutex_unlock(&p->mu);
            return NULL;
        }
        seen = p->generation;
        pthread_mutex_unlock(&p->mu);
        work_lines(p, wk);
        pthread_mutex_lock(&p->mu);
        p->finished++;
        pthread_cond_signal(&p->done);
        pthread_mutex_unlock(&p->mu);
    }
}
#endif

vocr_desk_page *vocr_desk_page_new(const void *det, size_t det_bytes, const void *rec, size_t rec_bytes,
                                   unsigned threads, int kind, unsigned n, int *err)
{
    vocr_desk_page *p;
    unsigned i;
    size_t need;
    int rc = VOCR_DESK_OK;
    if (err) *err = VOCR_DESK_OK;
    if (!det || !rec || threads < 1 || threads > MAX_THREADS) {
        if (err) *err = VOCR_DESK_E_ARG;
        return NULL;
    }
    p = (vocr_desk_page *)calloc(1, sizeof *p);
    if (!p) {
        if (err) *err = VOCR_DESK_E_MEM;
        return NULL;
    }
    rc = vocr_desk_load(&p->det, det, det_bytes);
    if (!rc && vocr_load(&p->rec, rec, rec_bytes)) rc = VOCR_DESK_E_REC;
    if (rc) {
        free(p);
        if (err) *err = rc;
        return NULL;
    }
    p->threads = threads;
    p->kind = kind < 0 ? p->det.size_kind : (unsigned)kind;
    p->n = kind < 0 ? p->det.size_n : n;
    p->opts.mode = VOCR_MODE_W8I;
    p->opts.head_gate = p->rec.gate.hidden ? VOCR_HEAD_GATE_ON : VOCR_HEAD_GATE_OFF;
    p->opts.two_level = p->rec.two.l1_blocks ? 1u : 0u;
    need = vocr_arena_bytes(&p->rec, &p->opts);
    for (i = 0; i < threads; i++) {
        worker *wk = &p->work[i];
        wk->page = p;
        wk->index = i;
        wk->arena = need ? amalloc(need) : NULL;
        if (!wk->arena || vocr_init(&wk->ctx, &p->rec, &p->opts, wk->arena, need)) {
            rc = wk->arena ? VOCR_DESK_E_REC : VOCR_DESK_E_MEM;
            break;
        }
    }
#ifndef VOCR_DESK_NO_THREADS
    if (!rc && threads > 1) {
        pthread_mutex_init(&p->mu, NULL);
        pthread_cond_init(&p->go, NULL);
        pthread_cond_init(&p->done, NULL);
        p->started = 1;
        for (i = 1; i < threads; i++)
            if (pthread_create(&p->work[i].thread, NULL, thread_main, &p->work[i])) {
                rc = VOCR_DESK_E_MEM;
                p->threads = i;
                break;
            }
    }
#endif
    if (rc) {
        vocr_desk_page_free(p);
        if (err) *err = rc;
        return NULL;
    }
    return p;
}

void vocr_desk_page_free(vocr_desk_page *p)
{
    unsigned i;
    if (!p) return;
#ifndef VOCR_DESK_NO_THREADS
    if (p->started) {
        pthread_mutex_lock(&p->mu);
        p->quit = 1;
        pthread_cond_broadcast(&p->go);
        pthread_mutex_unlock(&p->mu);
        for (i = 1; i < p->threads; i++) pthread_join(p->work[i].thread, NULL);
        pthread_cond_destroy(&p->go);
        pthread_cond_destroy(&p->done);
        pthread_mutex_destroy(&p->mu);
    }
#endif
    for (i = 0; i < MAX_THREADS; i++) free(p->work[i].arena);
    free(p->arena);
    free(p->grey_buf);
    free(p);
}

/* reading order (tools/vocr_desktop.reading_order): by (centre y, x), stable; rows by vertical overlap with the row's
 * first box of at least half the smaller height; left to right (stable) within a row */
static void reading_order(const vocr_desk_set *s, unsigned *order)
{
    unsigned n = s->count, i, j, rows = 0;
    unsigned idx[VOCR_DESK_MAX_BOXES], row_of[VOCR_DESK_MAX_BOXES], first[VOCR_DESK_MAX_BOXES];
    for (i = 0; i < n; i++) idx[i] = i;
    for (i = 1; i < n; i++) {   /* stable insertion sort by (y + h / 2, x) */
        unsigned v = idx[i];
        double ky = s->boxes[v].y + s->boxes[v].h / 2.0, kx = s->boxes[v].x;
        j = i;
        while (j > 0) {
            const vocr_desk_box *b = &s->boxes[idx[j - 1]];
            double by = b->y + b->h / 2.0;
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
        const vocr_desk_box *b = &s->boxes[idx[i]];
        if (rows) {
            const vocr_desk_box *f = &s->boxes[first[rows - 1]];
            double top = f->y > b->y ? f->y : b->y, bot = (f->y + f->h) < (b->y + b->h) ? f->y + f->h : b->y + b->h;
            double mh = f->h < b->h ? f->h : b->h;
            if (bot - top >= 0.5 * mh) {
                row_of[i] = rows - 1;
                continue;
            }
        }
        first[rows] = idx[i];
        row_of[i] = rows++;
    }
    {
        unsigned k = 0, r;
        for (r = 0; r < rows; r++) {
            unsigned start = k;
            for (i = 0; i < n; i++)
                if (row_of[i] == r) order[k++] = idx[i];
            for (i = start + 1; i < k; i++) {   /* stable by x */
                unsigned v = order[i];
                j = i;
                while (j > start && s->boxes[order[j - 1]].x > s->boxes[v].x) {
                    order[j] = order[j - 1];
                    j--;
                }
                order[j] = v;
            }
        }
    }
}

static int detect(vocr_desk_page *p, const unsigned char *grey, unsigned w, unsigned h, unsigned stride)
{
    size_t need;
    if (!p || !grey || !w || !h || stride < w) return VOCR_DESK_E_ARG;
    need = vocr_desk_arena_bytes(&p->det, p->kind, p->n, w, h);
    if (!need) return VOCR_DESK_E_ARG;
    if (need > p->arena_bytes) {
        free(p->arena);
        p->arena = amalloc(need);
        p->arena_bytes = p->arena ? need : 0;
        if (!p->arena) return VOCR_DESK_E_MEM;
    }
    p->count = 0;
    return vocr_desk_detect(&p->det, grey, w, h, stride, p->kind, p->n, p->arena, p->arena_bytes, &p->set);
}

int vocr_desk_page_detect(vocr_desk_page *p, const unsigned char *grey, unsigned w, unsigned h, unsigned stride)
{
    double t0 = now_us();
    int rc = detect(p, grey, w, h, stride);
    if (p) p->detect_us = p->total_us = now_us() - t0;
    return rc;
}

static int run(vocr_desk_page *p, const unsigned char *grey, unsigned w, unsigned h, unsigned stride, double t0);

int vocr_desk_page_run(vocr_desk_page *p, const unsigned char *grey, unsigned w, unsigned h, unsigned stride)
{
    return run(p, grey, w, h, stride, now_us());
}

int vocr_desk_page_run_rgb(vocr_desk_page *p, const unsigned char *rgb, unsigned w, unsigned h, unsigned stride,
                           unsigned channels)
{
    double t0 = now_us();
    if (!p || !rgb || !w || !h || (channels != 3 && channels != 4) || stride < (size_t)w * channels)
        return VOCR_DESK_E_ARG;
    if ((size_t)w * h > p->grey_cap) {
        free(p->grey_buf);
        p->grey_buf = (unsigned char *)amalloc((size_t)w * h);
        p->grey_cap = p->grey_buf ? (size_t)w * h : 0;
        if (!p->grey_buf) return VOCR_DESK_E_MEM;
    }
    vocr_desk_grey(rgb, w, h, stride, channels, p->grey_buf);
    return run(p, p->grey_buf, w, h, w, t0);
}

static int run(vocr_desk_page *p, const unsigned char *grey, unsigned w, unsigned h, unsigned stride, double t0)
{
    unsigned i, k = 0, order[VOCR_DESK_MAX_BOXES];
    int rc = detect(p, grey, w, h, stride);
    double t1 = now_us();
    if (!p) return VOCR_DESK_E_ARG;
    p->detect_us = t1 - t0;
    if (rc) return rc;
    reading_order(&p->set, order);
    for (i = 0; i < p->set.count; i++) {
        const vocr_desk_box *b = &p->set.boxes[order[i]];
        double x0 = floor(b->x), y0 = floor(b->y), x1 = ceil(b->x + b->w), y1 = ceil(b->y + b->h);
        vocr_desk_line *ln = &p->lines[k];
        if (x0 < 0) x0 = 0;
        if (y0 < 0) y0 = 0;
        if (x1 > (double)w) x1 = (double)w;
        if (y1 > (double)h) y1 = (double)h;
        if (!(x1 > x0 && y1 > y0)) continue;
        p->crops[k].x = (int)x0;
        p->crops[k].y = (int)y0;
        p->crops[k].w = (int)(x1 - x0);
        p->crops[k].h = (int)(y1 - y0);
        ln->x = p->crops[k].x;
        ln->y = p->crops[k].y;
        ln->w = p->crops[k].w;
        ln->h = p->crops[k].h;
        ln->tag = b->tag;
        ln->vertical = b->vertical;
        ln->dark = 0;
        p->order[k] = k;
        k++;
    }
    p->count = k;
    p->grey = grey;
    p->w = w;
    p->h = h;
    p->stride = stride;
    for (i = 0; i < p->threads; i++) p->work[i].rc = 0;
#ifndef VOCR_DESK_NO_THREADS
    if (p->threads > 1 && k > 1) {
        pthread_mutex_lock(&p->mu);
        p->next = 0;
        p->finished = 0;
        p->generation++;
        pthread_cond_broadcast(&p->go);
        pthread_mutex_unlock(&p->mu);
        work_lines(p, &p->work[0]);
        pthread_mutex_lock(&p->mu);
        while (p->finished < p->threads - 1) pthread_cond_wait(&p->done, &p->mu);
        pthread_mutex_unlock(&p->mu);
    } else
#endif
    {
        for (i = 0; i < k; i++) {
            rc = read_line(p, &p->work[0], i);
            if (rc) p->work[0].rc = rc;
        }
    }
    for (i = 0; i < p->threads; i++)
        if (p->work[i].rc) return p->work[i].rc;
    p->total_us = now_us() - t0;
    return VOCR_DESK_OK;
}

unsigned vocr_desk_page_count(const vocr_desk_page *p)
{
    return p ? p->count : 0;
}

const vocr_desk_line *vocr_desk_page_line(const vocr_desk_page *p, unsigned i)
{
    return p && i < p->count ? &p->lines[i] : NULL;
}

const vocr_desk_set *vocr_desk_page_boxes(const vocr_desk_page *p)
{
    return p ? &p->set : NULL;
}

double vocr_desk_page_detect_us(const vocr_desk_page *p)
{
    return p ? p->detect_us : 0.0;
}

double vocr_desk_page_total_us(const vocr_desk_page *p)
{
    return p ? p->total_us : 0.0;
}
