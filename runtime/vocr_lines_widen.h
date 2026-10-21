/* vocr_lines_widen.h - a line the region's left or right edge cuts, continued past the edge (issue #109).
 *
 * A region drawn tight against its text cuts the first or last character of a line, and the recognizer then
 * reads what is left of it: on the owner's console, CLANNAD's 「九ヶ月です」 lost its 「 and its 九 became ル.
 * The finder (vocr_lines.h) and the refinement only see the region, so they cannot give back what the region's
 * edge cut off; a caller that holds the frame around the region can (Rubi's capture reads the region alone:
 * vita-vn-ocr #110). vocr_line_widen() takes a
 * found line's rows over the region and a margin on each side, and continues its box past the region's edge
 * through the ink that the edge cuts: a box that starts at the region's first column (or ends at its last) with
 * ink in the edge's two columns is extended column by column while there is ink, across gaps of fewer than
 * max(2, h / 8) columns, plus the finder's 3 px padding, provided that the ink ends in such a gap inside the
 * margin: ink that runs on to the margin's end (a box frame, a textured scene) is not a cut character, and that
 * side stays as the finder returned it. A column holds ink when the range
 * (max - min) of its grey values over the line's rows is at least max(16, C / 3), C the largest range over the
 * box's columns inside the region. Every other line is left as the finder returned it (only moved to the
 * margin's coordinates), so a region with room around its text gives exactly the finder's crops.
 *
 * Plain C99, no allocation, no libm. synth/lines_widen.py is its Python twin (tests/lines_widen_parity.py).
 * Off unless the caller calls it; docs/counters.md has the development measurements.
 */
#ifndef VOCR_LINES_WIDEN_H
#define VOCR_LINES_WIDEN_H

#include "vocr_lines.h"

#ifdef __cplusplus
extern "C" {
#endif

enum { VOCR_WIDEN_LEFT = 1, VOCR_WIDEN_RIGHT = 2 };

/* im: the line's rows (im->height = the box's height, row 0 its top row) over the region widened by a margin:
 * the region's columns are [rx, rx + rw) of im, the margin [0, rx) and [rx + rw, im->width). *x, *w: the
 * finder's box columns, in region coordinates on entry (x + w <= rw), in im's coordinates on return. Returns
 * the VOCR_WIDEN_* sides it extended (0: none, the box only moved by rx), or VOCR_LINES_E_ARG (a null pointer,
 * an empty image, an unknown format, or a box outside the region). */
int vocr_line_widen(const VocrLinesImage *im, unsigned rx, unsigned rw, unsigned *x, unsigned *w);

#ifdef __cplusplus
}
#endif
#endif
