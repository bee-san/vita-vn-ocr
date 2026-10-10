/* Meiki detector preprocessing binding: geometry, RGBA->BGR and NCHW.
 * Geometry follows rtr46/meikiocr 0.3.5, commit
 * 52aa607ecdc1038c7bf390f37cd9a8b306e8a08a, meikiocr/ocr.py:213-231.
 * The uint8 bilinear arithmetic is a scalar adaptation of OpenCV 4.12.0
 * modules/imgproc/src/resize.cpp (HResizeLinear and VResizeLinear<uchar>).
 * https://github.com/opencv/opencv/blob/4.12.0/modules/imgproc/src/resize.cpp
 * Source SHA256: 89db2e4de92e1d03890f3e7fe517f895a36514f355da097b2c8e4c55fc1847e4
 *
 * Copyright (C) 2000-2008, 2017, Intel Corporation, all rights reserved.
 * Copyright (C) 2009, Willow Garage Inc., all rights reserved.
 * Copyright (C) 2014-2015, Itseez Inc., all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. The name of the copyright holders may not be used to endorse or promote
 *    products derived from this software without specific prior written permission.
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
#include "detect_preprocess.h"
#include <math.h>
#include <string.h>

#define COEF_SCALE 2048u
#define PLANE (320u * 192u)
typedef struct {
    unsigned char rows[2][MEIKI_PREPROCESS_MAX_WIDTH * 4u];
    uint16_t x[320], a[320][2];
} Scratch;
typedef char scratch_size_matches[(sizeof(Scratch) == MEIKI_DETECT_SCRATCH_BYTES) ? 1 : -1];

static unsigned round_even(double value)
{
    unsigned integer = (unsigned)value;
    double fraction = value - integer;
    return integer + (fraction > 0.5 || (fraction == 0.5 && (integer & 1u)));
}

static int is_cancelled(const MeikiRgbaImage *image)
{
    return image->cancelled && image->cancelled(image->ud);
}

int meiki_detect_preprocess_rgba(const MeikiRgbaImage *image, const MeikiCrop *crop,
                         float *output, size_t output_elements,
                         void *scratch, size_t scratch_bytes,
                         MeikiDetectPreprocessInfo *info)
{
    unsigned w, h, dy, dx;
    Scratch *s = (Scratch *)scratch;
    int cached_y[2] = {-1, -1};
    if (info) memset(info, 0, sizeof(*info));
    if (!image || !crop || !output || !s || !info || !image->rows ||
        output_elements < MEIKI_DETECT_ELEMENTS ||
        scratch_bytes < sizeof(*s) || ((uintptr_t)s & 1u) || ((uintptr_t)output & 3u) ||
        !image->width || !image->height ||
        image->width > MEIKI_PREPROCESS_MAX_WIDTH ||
        image->height > MEIKI_PREPROCESS_MAX_HEIGHT || !crop->w || !crop->h ||
        crop->x > image->width || crop->w > image->width - crop->x ||
        crop->y > image->height || crop->h > image->height - crop->y)
        return MEIKI_PREPROCESS_INVALID;
    if ((uintptr_t)output > UINTPTR_MAX - MEIKI_DETECT_ELEMENTS * sizeof(*output) ||
        (uintptr_t)s > UINTPTR_MAX - sizeof(*s) ||
        ((uintptr_t)output < (uintptr_t)s + sizeof(*s) &&
         (uintptr_t)s < (uintptr_t)output + MEIKI_DETECT_ELEMENTS * sizeof(*output)))
        return MEIKI_PREPROCESS_INVALID;
    if (is_cancelled(image)) return MEIKI_PREPROCESS_CANCELLED;
    double scale_w = 320.0 / crop->w, scale_h = 192.0 / crop->h;
    double scale = scale_w < scale_h ? scale_w : scale_h;
    w = (unsigned)(crop->w * scale);
    h = (unsigned)(crop->h * scale);
    if (!w || !h) return MEIKI_PREPROCESS_INVALID;
    memset(output, 0, MEIKI_DETECT_ELEMENTS * sizeof(*output));
    for (dx = 0; dx < w; dx++) {
        float fx = (float)((dx + 0.5) * ((double)crop->w / w) - 0.5);
        int sx = (int)floorf(fx);
        fx -= sx;
        if (sx < 0) { sx = 0; fx = 0.0f; }
        if (sx >= (int)crop->w - 1) { sx = (int)crop->w - 1; fx = 0.0f; }
        s->x[dx] = (uint16_t)sx;
        s->a[dx][0] = (uint16_t)round_even((1.0f - fx) * COEF_SCALE);
        s->a[dx][1] = (uint16_t)round_even(fx * COEF_SCALE);
    }
    for (dy = 0; dy < h; dy++) {
        float fy = (float)((dy + 0.5) * ((double)crop->h / h) - 0.5);
        int sy = (int)floorf(fy), source_y[2], i;
        unsigned b[2];
        fy -= sy;
        b[0] = round_even((1.0f - fy) * COEF_SCALE);
        b[1] = round_even(fy * COEF_SCALE);
        source_y[0] = sy < 0 ? 0 : sy;
        source_y[1] = sy + 1;
        if (source_y[1] >= (int)crop->h) source_y[1] = (int)crop->h - 1;
        if (source_y[0] >= (int)crop->h) source_y[0] = (int)crop->h - 1;
        if (source_y[1] < 0) source_y[1] = 0;
        if (is_cancelled(image)) return MEIKI_PREPROCESS_CANCELLED;
        for (i = 0; i < 2; i++) {
            if (source_y[i] != cached_y[i]) {
                if (is_cancelled(image)) return MEIKI_PREPROCESS_CANCELLED;
                if (image->rows(image->ud, crop->y + (unsigned)source_y[i], 1, s->rows[i]) != 1)
                    return MEIKI_PREPROCESS_SOURCE;
                cached_y[i] = source_y[i];
                if (is_cancelled(image)) return MEIKI_PREPROCESS_CANCELLED;
            }
        }
        for (dx = 0; dx < w; dx++) {
            unsigned sx = s->x[dx], sx1 = sx + (sx + 1 < crop->w);
            unsigned channel;
            for (channel = 0; channel < 3; channel++) {
                unsigned c = 2u - channel;
                unsigned p = (crop->x + sx) * 4u + c;
                unsigned q = (crop->x + sx1) * 4u + c;
                unsigned t0 = s->rows[0][p] * s->a[dx][0] + s->rows[0][q] * s->a[dx][1];
                unsigned t1 = s->rows[1][p] * s->a[dx][0] + s->rows[1][q] * s->a[dx][1];
                unsigned pixel = (((b[0] * (t0 >> 4)) >> 16) +
                                  ((b[1] * (t1 >> 4)) >> 16) + 2) >> 2;
                output[channel * PLANE + dy * 320u + dx] = pixel / 255.0f;
            }
        }
    }
    info->effective_w = w;
    info->effective_h = h;
    info->target_width = (int32_t)(320.0 / scale);
    info->target_height = (int32_t)(192.0 / scale);
    return MEIKI_PREPROCESS_OK;
}
