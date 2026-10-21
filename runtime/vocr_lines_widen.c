/* vocr_lines_widen.c - a line the region's edge cuts, continued past the edge (vocr_lines_widen.h, issue #109). */
#include "vocr_lines_widen.h"

#define PAD 3u          /* the finder's padding (vocr_lines.c PAD) */
#define MIN_INK 16u     /* a column's grey range below this is never ink */

static unsigned grey_at(const VocrLinesImage *im, unsigned x, unsigned y)
{
    const unsigned char *p = im->pixels + (size_t)y * im->stride + (size_t)x * im->bpp;
    return im->bpp == VOCR_LINES_GREY ? p[0] : (77u * p[0] + 150u * p[1] + 29u * p[2]) >> 8;
}

/* max - min of column x's grey values over the image's rows */
static unsigned range_at(const VocrLinesImage *im, unsigned x)
{
    unsigned y, lo = 255, hi = 0;
    for (y = 0; y < im->height; y++) {
        const unsigned g = grey_at(im, x, y);
        if (g < lo) lo = g;
        if (g > hi) hi = g;
    }
    return hi - lo;
}

int vocr_line_widen(const VocrLinesImage *im, unsigned rx, unsigned rw, unsigned *x, unsigned *w)
{
    unsigned c, cmax = 0, t, gap_max, left, right, sides = 0;
    if (!im || !im->pixels || !x || !w || !im->width || !im->height ||
        (im->bpp != VOCR_LINES_GREY && im->bpp != VOCR_LINES_RGBA) || im->stride < (size_t)im->width * im->bpp ||
        !rw || rx > im->width || rw > im->width - rx || !*w || *x > rw || *w > rw - *x)
        return VOCR_LINES_E_ARG;
    left = rx + *x;
    right = left + *w; /* exclusive */
    for (c = left; c < right; c++) {
        const unsigned r = range_at(im, c);
        if (r > cmax) cmax = r;
    }
    t = cmax / 3u > MIN_INK ? cmax / 3u : MIN_INK;
    gap_max = im->height / 8u > 2u ? im->height / 8u : 2u;
    /* left: the box starts at the region's first column, and the edge's two columns hold ink */
    if (*x == 0 && rx > 0 && (range_at(im, rx) >= t || (rw > 1 && range_at(im, rx + 1) >= t))) {
        unsigned last = rx, gap = 0, ended = 0;
        for (c = rx; c-- > 0;) {
            if (range_at(im, c) >= t) {
                last = c;
                gap = 0;
            } else if (++gap >= gap_max) {
                ended = 1;
                break;
            }
        }
        if (ended && last < rx) { /* a cut glyph ends inside the margin; ink up to its edge is no glyph */
            left = last > PAD ? last - PAD : 0u;
            sides |= VOCR_WIDEN_LEFT;
        }
    }
    /* right: the box ends at the region's last column, and the edge's two columns hold ink */
    if (*x + *w == rw && rx + rw < im->width &&
        (range_at(im, rx + rw - 1u) >= t || (rw > 1 && range_at(im, rx + rw - 2u) >= t))) {
        unsigned last = rx + rw - 1u, gap = 0, ended = 0;
        for (c = rx + rw; c < im->width; c++) {
            if (range_at(im, c) >= t) {
                last = c;
                gap = 0;
            } else if (++gap >= gap_max) {
                ended = 1;
                break;
            }
        }
        if (ended && last >= rx + rw) {
            right = last + 1u + PAD < im->width ? last + 1u + PAD : im->width;
            sides |= VOCR_WIDEN_RIGHT;
        }
    }
    *x = left;
    *w = right - left;
    return (int)sides;
}
