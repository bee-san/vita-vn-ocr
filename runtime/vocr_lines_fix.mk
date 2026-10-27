# vocr_lines_fix.mk - checks of the finder with #129's fixes (vocr_lines_fix.c), included by runtime/Makefile.
#
#   make check-lines-fix
#     1. vocr_lines_fix.c compiled on its own with LINES_STRICT (-Werror)
#     2. tests/test_vocr_lines_fix.c built with $(CC) $(CFLAGS) ... $(LDFLAGS) (in CI: ASan + UBSan) and run on
#        LINES_FIX_CASES random regions plus the edge sizes: fix 0 against the frozen vocr_lines.c for every
#        finder, the in-memory and row forms against each other for random fix flags, a failing row callback,
#        one region drawn for each fix, the ornament test's counts against their definition, argument checks
#
# Uses CC, CPPFLAGS, CFLAGS, LDFLAGS, OUT, SAN_ENV and LINES_STRICT from the including Makefile and vocr_lines.mk.

LINES_FIX_CASES ?= 600
LINES_FIX_OUT = $(abspath $(OUT))/lines_fix

.PHONY: check-lines-fix

check-lines-fix:
	@mkdir -p $(LINES_FIX_OUT)
	$(CC) $(LINES_STRICT) -O2 -c vocr_lines_fix.c -o $(LINES_FIX_OUT)/vocr_lines_fix_strict.o
	$(CC) $(CPPFLAGS) -std=c99 -Wall -Wextra -Werror $(CFLAGS) -o $(LINES_FIX_OUT)/test_vocr_lines_fix \
	    vocr_lines.c tests/test_vocr_lines_fix.c $(LDFLAGS)
	$(SAN_ENV) $(LINES_FIX_OUT)/test_vocr_lines_fix $(LINES_FIX_CASES)
