/* cases.c - see cases.h. */
#include "cases.h"

#include <string.h>

#define MAGIC "VOCRCASE"

uint32_t vc_rd32(const unsigned char *p)
{
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}

int vc_header(const unsigned char *p, VcHeader *h)
{
    if (memcmp(p, MAGIC, 8) || vc_rd32(p + 8) != 1) return -1;
    h->n_models = vc_rd32(p + 12);
    h->n_cases = vc_rd32(p + 16);
    h->max_case_bytes = vc_rd32(p + 20);
    h->max_lines = vc_rd32(p + 24);
    h->max_frames = vc_rd32(p + 28);
    h->n_classes = vc_rd32(p + 32);
    h->max_rw = vc_rd32(p + 36);
    h->max_rh = vc_rd32(p + 40);
    if (!h->n_models || h->n_models > VC_MAX_MODELS || !h->n_cases || h->max_case_bytes < 32 ||
        h->max_lines > VC_MAX_LINES || !h->max_frames || !h->n_classes)
        return -1;
    return 0;
}

void vc_model(const unsigned char *p, VcModel *m)
{
    memset(m, 0, sizeof *m);
    memcpy(m->name, p, 16);
    memcpy(m->file, p + 16, 64);
    m->mode = vc_rd32(p + 80);
    m->blob_bytes = vc_rd32(p + 84);
    m->blob_crc = vc_rd32(p + 88);
    m->host_arena = vc_rd32(p + 92);
}

static uint32_t pad4(uint32_t n) { return (n + 3u) & ~3u; }

int vc_case(const unsigned char *p, size_t n, unsigned n_models, char *strings, VcCase *c)
{
    const unsigned char *end = p + n;
    uint32_t id_len, gt_len, i, j;
    char *s = strings;
    if (n < 28 || n_models > VC_MAX_MODELS) return -1;
    memset(c, 0, sizeof *c);
    c->kind = vc_rd32(p);
    c->flags = vc_rd32(p + 4);
    c->w = vc_rd32(p + 8);
    c->h = vc_rd32(p + 12);
    c->ch = vc_rd32(p + 16);
    id_len = vc_rd32(p + 20);
    gt_len = vc_rd32(p + 24);
    p += 28;
    if (c->kind > VC_KIND_REGION || (c->ch != 1 && c->ch != 4) || (c->kind == VC_KIND_REGION && c->ch != 4) ||
        !c->w || !c->h || c->w > 65535u || c->h > 65535u || id_len > 255 || gt_len > VC_TEXT_CAP ||
        (size_t)(end - p) < pad4(id_len + gt_len))
        return -1;
    memcpy(s, p, id_len);
    s[id_len] = 0;
    c->id = s;
    s += id_len + 1;
    memcpy(s, p + id_len, gt_len);
    s[gt_len] = 0;
    c->gt = s;
    s += gt_len + 1;
    p += pad4(id_len + gt_len);
    for (i = 0; i < n_models; i++) {
        VcExpect *e = &c->exp[i];
        uint32_t tl;
        if (end - p < 16) return -1;
        tl = vc_rd32(p);
        e->crc = vc_rd32(p + 4);
        e->frames = vc_rd32(p + 8);
        e->n_lines = vc_rd32(p + 12);
        p += 16;
        if (tl > VC_TEXT_CAP || e->n_lines > VC_MAX_LINES || (size_t)(end - p) < 20u * e->n_lines + pad4(tl))
            return -1;
        for (j = 0; j < e->n_lines; j++, p += 20) {
            e->box[j][0] = vc_rd32(p);
            e->box[j][1] = vc_rd32(p + 4);
            e->box[j][2] = vc_rd32(p + 8);
            e->box[j][3] = vc_rd32(p + 12);
            e->box[j][4] = vc_rd32(p + 16);
        }
        memcpy(s, p, tl);
        s[tl] = 0;
        e->text = s;
        s += tl + 1;
        p += pad4(tl);
    }
    /* 64-bit: w * h * ch reaches 2^34 and would wrap in the Vita's 32-bit size_t */
    if ((uint64_t)(end - p) < (uint64_t)c->w * c->h * c->ch) return -1;
    c->px = p;
    return 0;
}
