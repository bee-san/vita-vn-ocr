# vocr_vertical.mk - checks of the vertical-text column finder and rotation (vocr_vertical.c), included by
# runtime/Makefile.
#
#   make check-vertical
#     1. vocr_vertical.c and its driver vocr_vertical_test.c compiled on their own with LINES_STRICT
#        (vocr_lines.mk: -std=c99 -pedantic ... -Wconversion -Wsign-conversion ... -Werror)
#     2. vocr_vertical_test, the host driver, built with $(CC) $(CFLAGS) ... $(LDFLAGS) (in CI: ASan + UBSan,
#        ci/runtime.sh) with vocr_lines.c and vocr_lines_rows.c (the dark flags), then its --selftest on
#        VERTICAL_SELFTEST procedural regions (in-memory, row and RGBA forms agree, exact arena, error paths,
#        the rotation)
#     3. tests/vertical_parity.py: that build against the Python twin (synth/vertical.py) on VERTICAL_CASES
#        procedural regions, both flag sets, both forms, and the rotation
#
# Uses CC, CPPFLAGS, CFLAGS, LDFLAGS, PYTHON, OUT, SAN_ENV and LINES_STRICT from the including Makefile and
# vocr_lines.mk. The driver is rebuilt on every run, so a change of CC or flags always takes effect.

VERTICAL_SELFTEST ?= 200
VERTICAL_CASES ?= 120
VERTICAL_OUT = $(abspath $(OUT))/vertical

.PHONY: check-vertical

check-vertical:
	@mkdir -p $(VERTICAL_OUT)
	$(CC) $(LINES_STRICT) -O2 -c vocr_vertical.c -o $(VERTICAL_OUT)/vocr_vertical_strict.o
	$(CC) $(LINES_STRICT) -O2 -c vocr_vertical_test.c -o $(VERTICAL_OUT)/vocr_vertical_test_strict.o
	$(CC) $(CPPFLAGS) -std=c99 -Wall -Wextra -Werror $(CFLAGS) -o $(VERTICAL_OUT)/vocr_vertical_test \
	    vocr_lines.c vocr_lines_rows.c vocr_vertical.c vocr_vertical_test.c $(LDFLAGS)
	$(SAN_ENV) $(VERTICAL_OUT)/vocr_vertical_test --selftest $(VERTICAL_SELFTEST)
	$(SAN_ENV) $(PYTHON) ../tests/vertical_parity.py --bin $(VERTICAL_OUT)/vocr_vertical_test \
	    --cases $(VERTICAL_CASES)
