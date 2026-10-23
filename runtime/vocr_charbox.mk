# vocr_charbox.mk - checks of the per-character positions (vocr_charbox.c, issue #114), included by runtime/Makefile.
#
#   make check-charbox
#     1. vocr_charbox.c and its driver vocr_charbox_test.c compiled on their own with LINES_STRICT
#        (vocr_lines.mk: -std=c99 -pedantic ... -Wconversion -Wsign-conversion ... -Werror)
#     2. vocr_charbox_test, the host driver, built with $(CC) $(CFLAGS) ... $(LDFLAGS) (in CI: ASan + UBSan,
#        ci/runtime.sh) with vocr.c (vocr_class_text, vocr_resized_width), and its self-test run
#     3. tests/charbox_parity.py: random lines through vocr_charbox_find and random text pairs through
#        vocr_charbox_align, in that build and in the twin (eval/charbox.py): every number must agree.
#        CHARBOX_ARGS sets the size of the run (CI: a few seconds).
#
# Uses CC, CPPFLAGS, CFLAGS, LDFLAGS, PYTHON, OUT, SAN_ENV and LINES_STRICT from the including Makefile
# and vocr_lines.mk. The driver is rebuilt on every run, so a change of CC or flags always takes effect.

CHARBOX_ARGS ?= --lines 1500 --aligns 1500
CHARBOX_OUT = $(abspath $(OUT))/charbox

.PHONY: check-charbox

check-charbox:
	@mkdir -p $(CHARBOX_OUT)
	$(CC) $(LINES_STRICT) -O2 -c vocr_charbox.c -o $(CHARBOX_OUT)/vocr_charbox_strict.o
	$(CC) $(LINES_STRICT) -O2 -c vocr_charbox_test.c -o $(CHARBOX_OUT)/vocr_charbox_test_strict.o
	$(CC) $(CPPFLAGS) -std=c99 -Wall -Wextra -Werror $(CFLAGS) -o $(CHARBOX_OUT)/vocr_charbox_test \
	    vocr.c vocr_charbox.c vocr_charbox_test.c $(LDFLAGS) -lm
	$(SAN_ENV) $(CHARBOX_OUT)/vocr_charbox_test
	$(SAN_ENV) $(PYTHON) ../tests/charbox_parity.py --bin $(CHARBOX_OUT)/vocr_charbox_test $(CHARBOX_ARGS)
