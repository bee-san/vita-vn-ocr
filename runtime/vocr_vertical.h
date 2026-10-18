/* Vertical text for the Vita/PSP VN OCR (issue #99, docs/vertical.md): a column finder for regions of vertical
 * Japanese (manga speech balloons, vertical dialogue boxes, NVL pages) and the column rotation the dedicated
 * vertical recognizer reads.
 *
 * vocr_find_columns is the GRAD line finder of vocr_lines.h run on the region's transpose: the same contrast
 * mask, runs, bands, clusters, growth, narrow-line drop and padding, so it finds text columns where GRAD finds
 * rows; then boxes wider than 5/4 of their height (horizontal text) are dropped and the columns are put in
 * reading order, right to left. synth/vertical_twin.py find_columns() is its bit-exact Python twin.
 *
 * vocr_column_rotate turns a column crop 90 degrees anticlockwise (the column's top to the left), so a CTC
 * recognizer trained on such strips (VOCR5-V) reads it as one horizontal line with vocr_run, unchanged.
 *
 * C99, no allocation, no recursion, integer arithmetic only. The caller passes an arena of
 * vocr_columns_arena_bytes() bytes (no alignment needed). Nothing here changes the horizontal finders:
 * vocr_lines.c and vocr_lines_rows.c are only called for the polarity flag (vocr_line_dark, _rows), so link them.
 */
#ifndef VOCR_VERTICAL_H
#define VOCR_VERTICAL_H

#include <stddef.h>

#include "vocr_lines.h"
#include "vocr_lines_rows.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Flags of vocr_find_columns. 0 is GRAD transposed exactly (synth/vertical_twin.py variant 0). */
#define VOCR_COLUMNS_VALLEYS 0x1u /* split bands where a column's ink falls under a quarter of the band's
                                     median, and grow a column only over rows dense enough to be text
                                     (variant 1) */
#define VOCR_COLUMNS_NO_DARK 0x2u /* leave every dark flag 0 and read no column box: the caller computes
                                     vocr_line_dark on each column's pixels (the same flag, bit for bit), e.g.
                                     on the copy it reads for recognition. With a row callback this saves one
                                     more read of every column's rows (a column spans the region's strips) */
#define VOCR_COLUMNS_FLAGS 0x3u   /* every defined flag */

/* Exact arena bytes vocr_find_columns(_rows) needs for any region up to max_width x max_height (0 if either is
 * 0): 83,619 bytes at 960 x 544 and 26,467 at 480 x 272, about GRAD's vocr_lines_arena_bytes. */
size_t vocr_columns_arena_bytes(unsigned max_width, unsigned max_height);

/* The columns of `im` in reading order (right to left; columns sharing a right edge top to bottom), as
 * VocrLines: x, y, w, h the padded box, dark the polarity flag of vocr_line_dark. At most VOCR_LINES_MAX;
 * out->truncated is 1 when more were found. VOCR_LINES_OK, VOCR_LINES_E_ARG (bad image, flags or a NULL pointer)
 * or VOCR_LINES_E_ARENA; out->count is 0 on an error. The arena is scratch only. */
int vocr_find_columns(const VocrLinesImage *im, unsigned flags, void *arena, size_t arena_bytes, VocrLineSet *out);

/* The same through a row callback (vocr_lines_rows.h): two passes over the region's rows in increasing order,
 * then each column's box once for its dark flag (none with VOCR_COLUMNS_NO_DARK). The same columns for the same
 * pixels, or VOCR_LINES_E_SOURCE (and no columns) when a row could not be read. */
int vocr_find_columns_rows(const VocrLinesRows *src, unsigned flags, void *arena, size_t arena_bytes,
                           VocrLineSet *out);

/* Turns a w x h crop 90 degrees anticlockwise into grey: dst is w rows of h bytes, dst_stride apart, and
 * dst[i][j] = grey(src[j][w - 1 - i]) (grey = (77 R + 150 G + 29 B) >> 8 for VOCR_LINES_RGBA). The rotated strip
 * is what VOCR5-V reads with vocr_run (VOCR_GREY8, | VOCR_INVERT when the column is dark). VOCR_LINES_OK or
 * VOCR_LINES_E_ARG. */
int vocr_column_rotate(const unsigned char *src, size_t src_stride, unsigned bpp, unsigned w, unsigned h,
                       unsigned char *dst, size_t dst_stride);

#ifdef __cplusplus
}
#endif

#endif
