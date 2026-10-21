# vocr_lines_widen.mk - checks of the edge continuation of cut lines (vocr_lines_widen.c, issue #109), included by
# runtime/Makefile.
#
#   make check-widen
#     1. vocr_lines_widen.c and its driver vocr_lines_widen_test.c compiled on their own with LINES_STRICT
#     2. vocr_lines_widen_test built with $(CC) $(CFLAGS) ... $(LDFLAGS) (in CI: ASan + UBSan) and its self-test
#        run: cut glyphs, gaps, regions at the image's edge, faint texture, errors
#     3. tests/lines_widen_parity.py: random strips and regions through that build and the twin
#        (synth/lines_widen.py): every box and side must agree. WIDEN_ARGS sets the size of the run.
#
# Uses CC, CPPFLAGS, CFLAGS, LDFLAGS, PYTHON, OUT, SAN_ENV and LINES_STRICT from the including Makefile
# and vocr_lines.mk.

WIDEN_ARGS ?= --cases 3000
WIDEN_OUT = $(abspath $(OUT))/widen

.PHONY: check-widen

check-widen:
	@mkdir -p $(WIDEN_OUT)
	$(CC) $(LINES_STRICT) -O2 -c vocr_lines_widen.c -o $(WIDEN_OUT)/vocr_lines_widen_strict.o
	$(CC) $(LINES_STRICT) -O2 -c vocr_lines_widen_test.c -o $(WIDEN_OUT)/vocr_lines_widen_test_strict.o
	$(CC) $(CPPFLAGS) -std=c99 -Wall -Wextra -Werror $(CFLAGS) -o $(WIDEN_OUT)/vocr_lines_widen_test \
	    vocr_lines_widen.c vocr_lines_widen_test.c $(LDFLAGS)
	$(SAN_ENV) $(WIDEN_OUT)/vocr_lines_widen_test
	$(SAN_ENV) $(PYTHON) ../tests/lines_widen_parity.py --bin $(WIDEN_OUT)/vocr_lines_widen_test $(WIDEN_ARGS)
