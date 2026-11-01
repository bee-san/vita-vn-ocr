/* vocr_desk.h - VOCR's desktop line detector and page pipeline in C (issue #169, docs/desktop.md).
 *
 * The desktop detector is a small float32 fully convolutional network (train/desktop_det.py): a 3x3 stem at stride 2,
 * four stages of separable blocks at strides 4, 8, 16 and 32, a top-down feature pyramid back to stride 4, one
 * separable smoothing block and a 1x1 head with 4 + T maps (K and O logits, A1 and A2 in input px, T tag logits). This
 * file runs it from a VDSK blob (train/desktop_det.py export) with no dependency but libm: channel-innermost float32
 * activations, AVX2 + FMA kernels when the compiler targets them (-mavx2 -mfma, or -march=x86-64-v3) and plain C
 * otherwise. It is the C path beside the ONNX file of the same weights (tools/vocr_desktop.py); the two compute the
 * same network in a different order of float operations, so their maps differ by float rounding (docs/desktop.md
 * measures how often a box changes). Given the same maps, vocr_desk_decode gives tools/vocr_desktop.decode's boxes
 * bit for bit (the same sums in the same order, in double).
 *
 * The page pipeline (vocr_desk_page_*) adds VOCR's recognizer: every horizontal box the detector finds is cut from the
 * grey frame, its polarity taken by vocr_line_dark (runtime/vocr_lines.c) and read by runtime/vocr.c in W8I, greedy,
 * as Rubi reads a line; lines are read on `threads` threads (one recognizer arena each). Unlike runtime/, this
 * desktop code allocates its own memory (malloc) and may use threads (POSIX threads; build with
 * -DVOCR_DESK_NO_THREADS to leave them out). Nothing here is console code: Rubi links none of it.
 */
#ifndef VOCR_DESK_H
#define VOCR_DESK_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define VOCR_DESK_VERSION 1u
#define VOCR_DESK_HEADER 144u           /* header bytes of a VDSK blob */
#define VOCR_DESK_STRIDE 4u             /* the maps' stride in network-input px */
#define VOCR_DESK_ALIGN 32u             /* the network's input sides are multiples of this */
#define VOCR_DESK_MAX_TAGS 4u
#define VOCR_DESK_MAX_BLOCKS 8u         /* separable blocks per stage after its down-sampling block */
#define VOCR_DESK_MAX_CHANNELS 512u
#define VOCR_DESK_MAX_BOXES 256u        /* boxes kept per frame (the highest mean kernel logits) */
#define VOCR_DESK_MAX_SIDE 8192u        /* a frame side the detector accepts */
#define VOCR_DESK_TEXT 1024u            /* bytes of a line's text, NUL included */

enum {
    VOCR_DESK_OK = 0,
    VOCR_DESK_E_ARG = -1,     /* a bad argument: a NULL pointer, a frame of 0 or more than 8192 px a side */
    VOCR_DESK_E_BLOB = -2,    /* not a VDSK version-1 blob, or one this build cannot read */
    VOCR_DESK_E_CRC = -3,     /* the blob's CRC-32 does not match */
    VOCR_DESK_E_ARENA = -4,   /* the arena is too small */
    VOCR_DESK_E_MEM = -5,     /* an allocation failed (the page pipeline) */
    VOCR_DESK_E_REC = -6      /* the recognizer refused its blob or a line */
};

enum { VOCR_DESK_SIZE_NATIVE = 0, VOCR_DESK_SIZE_FIT = 1, VOCR_DESK_SIZE_LONG = 2 };

typedef struct {
    const float *w, *b;       /* weights in the kernel's layout, bias */
    unsigned cin, cout;       /* a depthwise layer: cin == cout */
    unsigned stride, dil;
} vocr_desk_layer;

typedef struct {               /* a parsed blob: points into it, so keep the blob alive */
    unsigned c[5], n[4], F, tags, min_cells;
    unsigned size_kind, size_n;            /* the default input policy: fit:960 */
    unsigned dil_len[4], dil[4][4];        /* each stage's dilation pattern */
    double conf_min_logit, pad_pct, pad_min, min_across;
    unsigned pad_frame, pad_frame_min;     /* pad_frame > 0: boxes grown by max(pad_frame_min, nearbyint(pad_frame x
                                              frame height / 544)) frame px after the decode (pad_pct, pad_min 0) */
    vocr_desk_layer stem;                  /* 3x3 stride 2, 1 -> c0 */
    vocr_desk_layer dw[4][1 + VOCR_DESK_MAX_BLOCKS], pw[4][1 + VOCR_DESK_MAX_BLOCKS];   /* [s][0]: down block */
    vocr_desk_layer lat[4];                /* 1x1 c[s + 1] -> F */
    vocr_desk_layer sdw, spw;              /* the smoothing block */
    vocr_desk_layer head;                  /* 1x1 F -> 4 + tags (A1, A2 already in input px) */
    size_t params;
} vocr_desk_model;

typedef struct {
    double x, y, w, h;          /* frame px */
    double logit;               /* the mean K logit over the box's kernel cells */
    double tconf;               /* (best tag score - second) / cells */
    unsigned vertical, tag, cells;
} vocr_desk_box;

typedef struct {
    unsigned count, truncated;  /* truncated: more boxes than VOCR_DESK_MAX_BOXES passed the decode */
    unsigned in_w, in_h;        /* the network input's image part (before padding) */
    vocr_desk_box boxes[VOCR_DESK_MAX_BOXES];   /* sorted: logit descending, then y, then x */
} vocr_desk_set;

/* Parse and check a blob (magic, version, sizes, CRC-32). */
int vocr_desk_load(vocr_desk_model *m, const void *blob, size_t bytes);

/* The network input for a w x h frame under a size policy (kind, n): fit:n scales the longer side down to n px when it
 * is longer (never up), long:n scales it to n px, native keeps the frame; the image part iw x ih (each side
 * nearbyint(side * f), at least 1) and the padded input pw x ph (multiples of 32). */
void vocr_desk_input_size(unsigned kind, unsigned n, unsigned w, unsigned h, unsigned *iw, unsigned *ih,
                          unsigned *pw, unsigned *ph);

/* Arena bytes for vocr_desk_detect on a w x h frame under a policy (vocr_desk_maps needs the same for its input). */
size_t vocr_desk_arena_bytes(const vocr_desk_model *m, unsigned kind, unsigned n, unsigned w, unsigned h);

/* The network on a padded grey input (pw x ph, multiples of 32, row-major, stride pw): *maps points into the arena,
 * channel-innermost float32 [ph / 4][pw / 4][4 + tags]. */
int vocr_desk_maps(const vocr_desk_model *m, const unsigned char *in, unsigned pw, unsigned ph, void *arena,
                   size_t bytes, const float **maps);

/* The decode on maps (as vocr_desk_maps gives them, mw x mh cells of 4 + tags channels): the boxes of the iw x ih
 * image part, in input px (tools/vocr_desktop.decode's rule). scratch: (mw * mh + 2) ints and the per-component
 * sums, vocr_desk_decode_bytes(mw, mh, tags). */
size_t vocr_desk_decode_bytes(unsigned mw, unsigned mh, unsigned tags);
int vocr_desk_decode(const vocr_desk_model *m, const float *maps, unsigned mw, unsigned mh, unsigned iw, unsigned ih,
                     void *scratch, size_t bytes, vocr_desk_set *out);

/* The whole detector on a grey frame (w x h, row stride `stride`): the policy's resize (Q14 area weights down,
 * bilinear up), edge padding to multiples of 32, the network and the decode; boxes in frame px. */
int vocr_desk_detect(const vocr_desk_model *m, const unsigned char *grey, unsigned w, unsigned h, unsigned stride,
                     unsigned kind, unsigned n, void *arena, size_t bytes, vocr_desk_set *out);

/* The resize the detector uses (exposed for tests): Q14 area weights when a side shrinks, bilinear (pixel centres)
 * when it grows, separable, integer arithmetic. */
void vocr_desk_resize(const unsigned char *src, unsigned w, unsigned h, unsigned stride, unsigned char *dst,
                      unsigned dw, unsigned dh, int *scratch);
size_t vocr_desk_resize_scratch(unsigned w, unsigned h, unsigned dw, unsigned dh);

/* sizeof(vocr_desk_model) and sizeof(vocr_desk_set), for callers that allocate them (ctypes). */
size_t vocr_desk_model_bytes(void);
size_t vocr_desk_set_bytes(void);

/* Integer luma of an RGB(A) frame: (77 R + 150 G + 29 B) >> 8, as the recognizer's. */
void vocr_desk_grey(const unsigned char *rgb, unsigned w, unsigned h, unsigned stride, unsigned channels,
                    unsigned char *out);

/* ---- the page pipeline: detector + VOCR recognizer (runtime/vocr.c), lines on `threads` threads ---- */

typedef struct vocr_desk_page vocr_desk_page;

typedef struct {
    int x, y, w, h;             /* the crop read, frame px */
    unsigned tag, dark, vertical;
    float conf_min;             /* the recognizer's lowest character confidence (0..1) */
    char text[VOCR_DESK_TEXT];  /* greedy UTF-8; empty for a vertical box (not read) */
} vocr_desk_line;

/* det/rec: the blobs (kept alive by the caller); threads >= 1; kind/n: the input policy (kind < 0: the blob's). */
vocr_desk_page *vocr_desk_page_new(const void *det, size_t det_bytes, const void *rec, size_t rec_bytes,
                                   unsigned threads, int kind, unsigned n, int *err);
/* Read a grey frame: lines in reading order (rows top to bottom by vertical overlap, left to right in a row). */
int vocr_desk_page_run(vocr_desk_page *p, const unsigned char *grey, unsigned w, unsigned h, unsigned stride);
/* The same on an RGB (channels 3) or RGBA (4) frame: its integer luma first (vocr_desk_grey, timed as the page). */
int vocr_desk_page_run_rgb(vocr_desk_page *p, const unsigned char *rgb, unsigned w, unsigned h, unsigned stride,
                           unsigned channels);
/* The detector alone on a grey frame (vocr_desk_page_boxes gives its boxes; no line is read). */
int vocr_desk_page_detect(vocr_desk_page *p, const unsigned char *grey, unsigned w, unsigned h, unsigned stride);
unsigned vocr_desk_page_count(const vocr_desk_page *p);
const vocr_desk_line *vocr_desk_page_line(const vocr_desk_page *p, unsigned i);
const vocr_desk_set *vocr_desk_page_boxes(const vocr_desk_page *p);
/* The detector's last run alone, in us, and the whole page (detector + lines), in us. */
double vocr_desk_page_detect_us(const vocr_desk_page *p);
double vocr_desk_page_total_us(const vocr_desk_page *p);
void vocr_desk_page_free(vocr_desk_page *p);

#ifdef __cplusplus
}
#endif

#endif
