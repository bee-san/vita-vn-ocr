/* SPDX-License-Identifier: Apache-2.0
 * Adapted from rtr46/meikiocr 0.3.5 (Copyright rtr46), ocr.py:
 * confidence filtering, stable horizontal NMS, and swapped-pair correction.
 */
#include "decode.h"
#include <math.h>

typedef struct {
    int32_t code;
    int x1, x2;
    double score;
} Candidate;

int meiki_decode(const MeikiOutput *raw, unsigned crop_width,
                 unsigned effective_width, char *text, size_t cap) {
    Candidate candidates[48], accepted[48];
    unsigned count = 0, used = 0;
    size_t bytes = 0;
    if (!raw || !text || !cap || !crop_width || crop_width > 960 ||
        !effective_width || effective_width > 960)
        return -1;
    text[0] = 0;
    for (unsigned i = 0; i < 48; ++i) {
        float score = raw->scores[i], x1 = raw->boxes[4 * i],
              x2 = raw->boxes[4 * i + 2];
        int32_t cp = raw->codes[i];
        if (!isfinite(score) || !isfinite(x1) || !isfinite(x2) || score < .1f ||
            x1 >= (float)effective_width || cp <= 0 || cp > 0x10ffff ||
            (cp >= 0xd800 && cp <= 0xdfff))
            continue;
        if (x1 > (float)effective_width)
            x1 = (float)effective_width;
        if (x2 > (float)effective_width)
            x2 = (float)effective_width;
        /* NumPy's float32 box arithmetic, then Python's truncating int(). */
        float left = (x1 / (float)effective_width) * (float)crop_width;
        float right = (x2 / (float)effective_width) * (float)crop_width;
        if (left < -1000000 || right < -1000000 || left > 1000000 ||
            right > 1000000)
            return -2;
        Candidate c = {cp, (int)left, (int)right, (double)score};
        unsigned at = count++;
        while (at && candidates[at - 1].score < c.score) {
            candidates[at] = candidates[at - 1];
            --at;
        }
        candidates[at] = c;
    }
    for (unsigned i = 0; i < count; ++i) {
        Candidate c = candidates[i];
        int overlap = 0;
        for (unsigned j = 0; j < used; ++j) {
            Candidate a = accepted[j];
            if (c.x1 >= a.x2 || a.x1 >= c.x2)
                continue;
            int start = c.x1 > a.x1 ? c.x1 : a.x1;
            int end = c.x2 < a.x2 ? c.x2 : a.x2;
            double ca = c.x2 - c.x1 + 1e-6, aa = a.x2 - a.x1 + 1e-6;
            if ((end - start) / (ca < aa ? ca : aa) > .3) {
                overlap = 1;
                break;
            }
        }
        if (overlap)
            continue;
        unsigned at = used++;
        while (at && accepted[at - 1].x1 > c.x1) {
            accepted[at] = accepted[at - 1];
            --at;
        }
        accepted[at] = c;
    }
    static const int32_t pairs[][2] = {
        {0x5121, 0x5080}, {0x8ac7, 0x5197}, {0x6c70, 0x6dd8}, {0x6cb1, 0x6ec2},
        {0x6523, 0x75d9}, {0x914a, 0x9169}, {0x9ead, 0x9eba}, {0x54ed, 0x615f}};
    for (unsigned p = 0; p < sizeof(pairs) / sizeof(pairs[0]); ++p) {
        for (unsigned i = 0; i + 1 < used; ++i) {
            if (accepted[i].code == pairs[p][0] &&
                accepted[i + 1].code == pairs[p][1]) {
                accepted[i].code = pairs[p][1];
                accepted[i + 1].code = pairs[p][0];
                break;
            }
        }
    }
    for (unsigned i = 0; i < used; ++i) {
        uint32_t cp = (uint32_t)accepted[i].code;
        unsigned n = cp < 0x80 ? 1 : cp < 0x800 ? 2 : cp < 0x10000 ? 3 : 4;
        if (n >= cap - bytes) {
            text[bytes] = 0;
            return -3;
        }
        if (n == 1)
            text[bytes++] = (char)cp;
        else {
            text[bytes++] = (char)((n == 2   ? 0xc0
                                    : n == 3 ? 0xe0
                                             : 0xf0) |
                                   (cp >> (6 * (n - 1))));
            for (unsigned j = n - 1; j; --j)
                text[bytes++] = (char)(0x80 | ((cp >> (6 * (j - 1))) & 0x3f));
        }
    }
    text[bytes] = 0;
    return (int)used;
}
