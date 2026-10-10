/* cases.h - parser of cases.bin, the test bundle vita/bundle.py writes (format in its
 * docstring). Plain C99 without any SCE API, so the host test (tests/test_vita.py, through
 * vita/tests/cases_dump.c) checks the same code the Vita app runs. */
#ifndef VOCR_CASES_H
#define VOCR_CASES_H

#include <stddef.h>
#include <stdint.h>

#define VC_MAX_MODELS 4
#define VC_MAX_LINES 16 /* = VOCR_LINES_MAX */
#define VC_TEXT_CAP 4096
#define VC_HEADER_BYTES 64u
#define VC_MODEL_BYTES 96u
/* room for one case's NUL-terminated id, ground truth and expected texts */
#define VC_STRINGS_BYTES (256 + (VC_TEXT_CAP + 1) * (VC_MAX_MODELS + 1))

enum { VC_KIND_LINE = 0, VC_KIND_REGION = 1 };
enum { VC_FLAG_DARK = 1, VC_FLAG_AUTO_POLARITY = 2 };

typedef struct {
    uint32_t n_models, n_cases, max_case_bytes, max_lines, max_frames, n_classes, max_rw, max_rh;
} VcHeader;

typedef struct {
    char name[17], file[65];
    uint32_t mode, blob_bytes, blob_crc, host_arena;
} VcModel;

typedef struct {
    const char *text;
    uint32_t crc, frames, n_lines;
    uint32_t box[VC_MAX_LINES][5]; /* x, y, w, h, dark */
} VcExpect;

typedef struct {
    uint32_t kind, flags, w, h, ch;
    const char *id, *gt;
    const unsigned char *px; /* w * h * ch bytes inside the record */
    VcExpect exp[VC_MAX_MODELS];
} VcCase;

uint32_t vc_rd32(const unsigned char *p);
/* The 64-byte header; 0, or -1 if it is not a valid version-1 bundle header. */
int vc_header(const unsigned char *p, VcHeader *h);
/* One 96-byte model record. */
void vc_model(const unsigned char *p, VcModel *m);
/* One case record, without its leading u32 length (n bytes). strings must hold
 * VC_STRINGS_BYTES; the case's strings point into it. 0, or -1 for a malformed record. */
int vc_case(const unsigned char *p, size_t n, unsigned n_models, char *strings, VcCase *c);

#endif
