/* screen.c - see screen.h. */
#include "screen.h"

#include <psp2/display.h>
#include <psp2/kernel/sysmem.h>
#include <string.h>

#define FB_BYTES (2u * 1024u * 1024u) /* 960 * 544 * 4 rounded up to CDRAM's 256 KiB */
#define LOG_Y (4 + SCR_HEAD_ROWS * SCR_ROW + 8)
#define BAR_H 36
#define LOG_ROWS ((SCR_H - BAR_H - LOG_Y) / SCR_ROW)
#define LOG_TEXT 200

typedef struct {
    uint32_t cp;
    uint16_t width, pad;
    uint32_t off;
} Glyph;

static uint32_t *fb;
static const unsigned char *font;
static const Glyph *glyphs;
static uint32_t n_glyphs, font_h;
static size_t font_bytes;
static char log_text[LOG_ROWS][LOG_TEXT];
static uint32_t log_color[LOG_ROWS];
static int log_n;

static uint32_t rd32(const unsigned char *p)
{
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}

int scr_init(void)
{
    SceDisplayFrameBuf param;
    void *base = NULL;
    SceUID uid = sceKernelAllocMemBlock("vocr_fb", SCE_KERNEL_MEMBLOCK_TYPE_USER_CDRAM_RW, FB_BYTES, NULL);
    int rc;
    if (uid < 0) return uid;
    rc = sceKernelGetMemBlockBase(uid, &base);
    if (rc < 0) return rc;
    fb = (uint32_t *)base;
    memset(fb, 0, FB_BYTES);
    memset(&param, 0, sizeof param);
    param.size = sizeof param;
    param.base = fb;
    param.pitch = SCR_W;
    param.pixelformat = SCE_DISPLAY_PIXELFORMAT_A8B8G8R8;
    param.width = SCR_W;
    param.height = SCR_H;
    return sceDisplaySetFrameBuf(&param, SCE_DISPLAY_SETBUF_NEXTFRAME);
}

/* font.bin: "VFNT", u32 version 1, u32 glyph count, u32 height, then per glyph
 * {u32 codepoint, u16 width, u16 0, u32 offset} sorted by codepoint, then the bitmaps:
 * height rows of ceil(width / 8) bytes each, most significant bit first. */
int scr_load_font(const void *data, size_t n)
{
    const unsigned char *b = (const unsigned char *)data;
    uint32_t i, count, h;
    if (!b || n < 16 || memcmp(b, "VFNT", 4) || rd32(b + 4) != 1) return -1;
    count = rd32(b + 8);
    h = rd32(b + 12);
    if (!count || h != SCR_ROW || count > (n - 16) / sizeof(Glyph) || ((uintptr_t)b & 3u)) return -1;
    glyphs = (const Glyph *)(const void *)(b + 16);
    for (i = 0; i < count; i++) {
        const Glyph *g = &glyphs[i];
        if ((g->width != 8 && g->width != 16) || g->off > n ||
            (size_t)h * (g->width / 8u) > n - g->off || (i && g->cp <= glyphs[i - 1].cp))
            return -1;
    }
    font = b;
    font_bytes = n;
    n_glyphs = count;
    font_h = h;
    return 0;
}

int scr_has_font(void) { return font != NULL; }

void scr_fill(int x, int y, int w, int h, uint32_t color)
{
    int i, j;
    if (!fb) return;
    if (x < 0) w += x, x = 0;
    if (y < 0) h += y, y = 0;
    if (x + w > SCR_W) w = SCR_W - x;
    if (y + h > SCR_H) h = SCR_H - y;
    for (j = 0; j < h; j++)
        for (i = 0; i < w; i++) fb[(y + j) * SCR_W + x + i] = color;
}

static const Glyph *find_glyph(uint32_t cp)
{
    uint32_t lo = 0, hi = n_glyphs;
    while (lo < hi) {
        uint32_t mid = (lo + hi) / 2;
        if (glyphs[mid].cp < cp) lo = mid + 1;
        else hi = mid;
    }
    return lo < n_glyphs && glyphs[lo].cp == cp ? &glyphs[lo] : NULL;
}

/* Next code point of a NUL-terminated UTF-8 string (U+FFFD for a bad sequence). */
static uint32_t next_cp(const unsigned char **s)
{
    const unsigned char *p = *s;
    uint32_t c = *p++;
    int n = c >= 0xF0 ? 3 : c >= 0xE0 ? 2 : c >= 0xC0 ? 1 : 0;
    if (c >= 0x80 && !n) c = 0xFFFD;
    else if (n) c &= 0x3Fu >> n;
    while (n--) {
        if ((*p & 0xC0) != 0x80) {
            c = 0xFFFD;
            break;
        }
        c = c << 6 | (*p++ & 0x3Fu);
    }
    *s = p;
    return c;
}

int scr_text(int x, int y, uint32_t color, const char *utf8, int max_px)
{
    const unsigned char *s = (const unsigned char *)utf8;
    int x0 = x;
    if (!fb || !font || !s) return 0;
    while (*s) {
        uint32_t cp = next_cp(&s);
        const Glyph *g = find_glyph(cp);
        int w = g ? g->width : cp < 0x100 ? 8 : 16, r, c;
        if (x + w - x0 > max_px || x + w > SCR_W) break;
        if (g) {
            const unsigned char *bits = font + g->off;
            const int bpr = w / 8;
            for (r = 0; r < (int)font_h && y + r < SCR_H; r++)
                for (c = 0; c < w; c++)
                    if (bits[r * bpr + c / 8] & (0x80u >> (c & 7))) fb[(y + r) * SCR_W + x + c] = color;
        } else if (cp != ' ') { /* a hollow box for a character the font lacks */
            scr_fill(x + 1, y + 2, w - 2, 1, color);
            scr_fill(x + 1, y + 13, w - 2, 1, color);
            scr_fill(x + 1, y + 2, 1, 12, color);
            scr_fill(x + w - 2, y + 2, 1, 12, color);
        }
        x += w;
    }
    return x - x0;
}

void scr_head(int row, uint32_t color, const char *utf8)
{
    if (row < 0 || row >= SCR_HEAD_ROWS) return;
    scr_fill(0, 4 + row * SCR_ROW, SCR_W, SCR_ROW, SCR_BLACK);
    scr_text(8, 4 + row * SCR_ROW, color, utf8, SCR_W - 16);
}

void scr_status(uint32_t color, const char *utf8, unsigned done, unsigned total)
{
    const int y = SCR_H - BAR_H + 4;
    int fill = total ? (int)((unsigned long long)(SCR_W - 16) * done / total) : 0;
    scr_fill(0, SCR_H - BAR_H, SCR_W, BAR_H, SCR_BLACK);
    scr_fill(8, y, SCR_W - 16, 6, 0xFF303030u);
    scr_fill(8, y, fill, 6, color);
    scr_text(8, y + 12, color, utf8, SCR_W - 16);
}

/* Copies whole UTF-8 sequences of src that fit in cap - 1 bytes. */
static void copy_utf8(char *dst, size_t cap, const char *src)
{
    size_t i = 0, k;
    while (src[i]) {
        unsigned char c = (unsigned char)src[i];
        size_t n = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1;
        for (k = 1; k < n; k++)
            if (!src[i + k]) n = k; /* truncated sequence: copy what is there */
        if (i + n > cap - 1) break;
        memcpy(dst + i, src + i, n);
        i += n;
    }
    dst[i] = 0;
}

void scr_log(uint32_t color, const char *utf8)
{
    int i;
    if (log_n == LOG_ROWS) {
        memmove(log_text[0], log_text[1], sizeof log_text[0] * (LOG_ROWS - 1));
        memmove(log_color, log_color + 1, sizeof log_color[0] * (LOG_ROWS - 1));
        log_n--;
    }
    copy_utf8(log_text[log_n], LOG_TEXT, utf8);
    log_color[log_n++] = color;
    scr_fill(0, LOG_Y, SCR_W, LOG_ROWS * SCR_ROW, SCR_BLACK);
    for (i = 0; i < log_n; i++) scr_text(8, LOG_Y + i * SCR_ROW, log_color[i], log_text[i], SCR_W - 16);
}
