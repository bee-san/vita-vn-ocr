# vocr_textspot7.mk - checks of textspot7's fallback and post-pass (vocr_textspot7.c, issue #128), included by
# runtime/Makefile.
#
#   make check-textspot7
#     1. vocr_textspot7.c and its driver vocr_textspot7_test.c compiled on their own with LINES_STRICT
#        (vocr_lines.mk), and vocr_textspot7.c once more with -DVOCR_NO_NEON (the PSP's scalar build)
#     2. vocr_textspot7_test, the host driver, built with $(CC) $(CFLAGS) ... $(LDFLAGS) (in CI: ASan + UBSan,
#        ci/runtime.sh) with vocr_lines.c and vocr_lines_refine.c
#     3. tests/textspot7_parity.py through it: F's choice, D's post-pass and its row-streamed form against the twin
#        (synth/textspot7_twin.py) on random inputs
#
# Uses CC, CPPFLAGS, CFLAGS, LDFLAGS, PYTHON, OUT, SAN_ENV and LINES_STRICT from the including Makefile.

TEXTSPOT7_OUT = $(abspath $(OUT))/textspot7
TEXTSPOT7_SRC = vocr_lines.c vocr_lines_refine.c vocr_textspot7.c vocr_textspot7_test.c

.PHONY: check-textspot7

check-textspot7:
	@mkdir -p $(TEXTSPOT7_OUT)
	$(CC) $(LINES_STRICT) -O2 -c vocr_textspot7.c -o $(TEXTSPOT7_OUT)/vocr_textspot7_strict.o
	$(CC) $(LINES_STRICT) -O2 -DVOCR_NO_NEON -c vocr_textspot7.c -o $(TEXTSPOT7_OUT)/vocr_textspot7_scalar_strict.o
	$(CC) $(LINES_STRICT) -O2 -c vocr_textspot7_test.c -o $(TEXTSPOT7_OUT)/vocr_textspot7_test_strict.o
	$(CC) $(CPPFLAGS) -std=c99 -Wall -Wextra -Werror $(CFLAGS) -o $(TEXTSPOT7_OUT)/vocr_textspot7_test \
	    $(TEXTSPOT7_SRC) $(LDFLAGS)
	$(SAN_ENV) $(PYTHON) ../tests/textspot7_parity.py --bin $(TEXTSPOT7_OUT)/vocr_textspot7_test
