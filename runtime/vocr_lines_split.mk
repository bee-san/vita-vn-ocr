# vocr_lines_split.mk - checks of the stroke split (vocr_lines_split.c), included by runtime/Makefile.
#
#   make check-split
#     1. vocr_lines_split.c and its driver vocr_lines_split_test.c compiled on their own with LINES_STRICT
#        (vocr_lines.mk: -std=c99 -pedantic ... -Wconversion -Wsign-conversion ... -Werror)
#     2. vocr_lines_split_test, the host driver, built with $(CC) $(CFLAGS) ... $(LDFLAGS) (in CI: ASan + UBSan,
#        ci/runtime.sh) with vocr_lines.c (the finder) and vocr_lines_refine.c (the refinement after the split),
#        then its --selftest on SPLIT_CASES procedural regions (flags 0 is the identity, the in-memory and
#        row-callback forms, the per-box strip form and the minimum misaligned arena agree, lines stay inside
#        their box, errors leave the set unchanged, the arena never exceeds the finder's)
#     3. tests/lines_split_parity.py: that build against the Python twin (synth/lines_split.py, then
#        synth/lines_refine.py) on SPLIT_PARITY_CASES procedural and as many busy regions, both forms
#
# Uses CC, CPPFLAGS, CFLAGS, LDFLAGS, PYTHON, OUT, SAN_ENV and LINES_STRICT from the including Makefile and
# vocr_lines.mk. The driver is rebuilt on every run, so a change of CC or flags always takes effect.

SPLIT_CASES ?= 500
SPLIT_PARITY_CASES ?= 200
SPLIT_OUT = $(abspath $(OUT))/split

.PHONY: check-split

check-split:
	@mkdir -p $(SPLIT_OUT)
	$(CC) $(LINES_STRICT) -O2 -c vocr_lines_split.c -o $(SPLIT_OUT)/vocr_lines_split_strict.o
	$(CC) $(LINES_STRICT) -O2 -c vocr_lines_split_test.c -o $(SPLIT_OUT)/vocr_lines_split_test_strict.o
	$(CC) $(CPPFLAGS) -std=c99 -Wall -Wextra -Werror $(CFLAGS) -o $(SPLIT_OUT)/vocr_lines_split_test \
	    vocr_lines.c vocr_lines_refine.c vocr_lines_split.c vocr_lines_split_test.c $(LDFLAGS)
	$(SAN_ENV) $(SPLIT_OUT)/vocr_lines_split_test --selftest $(SPLIT_CASES)
	$(SAN_ENV) $(PYTHON) ../tests/lines_split_parity.py --bin $(SPLIT_OUT)/vocr_lines_split_test \
	    --cases $(SPLIT_PARITY_CASES)
	$(SAN_ENV) $(PYTHON) ../tests/lines_split_parity.py --bin $(SPLIT_OUT)/vocr_lines_split_test \
	    --cases $(SPLIT_PARITY_CASES) --seed 2 --rows
