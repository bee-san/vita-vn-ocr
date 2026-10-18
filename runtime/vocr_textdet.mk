# vocr_textdet.mk - checks of the learned text detector (vocr_textdet.c), included by runtime/Makefile.
#
#   make check-textdet
#     1. vocr_textdet.c and its driver vocr_textdet_test.c compiled on their own with LINES_STRICT
#        (vocr_lines.mk: -std=c99 -pedantic ... -Wconversion -Wsign-conversion ... -Werror)
#     2. vocr_textdet_test, the host driver, built with $(CC) $(CFLAGS) ... $(LDFLAGS) (in CI: ASan +
#        UBSan, ci/runtime.sh) with vocr_lines.c and vocr_lines_rows.c (the dark flags and the finder mode)
#        and vocr_lines_refine.c (det+refine: the refinement on the detector's horizontal lines)
#     3. tests/textdet_parity.py: random blobs of every shape in train/textdet_model.SHAPES
#        (train/textdet_export.py's functions), probe and wild-requantisation blobs, procedural regions
#        through that build and the Python twin (synth/textdet_twin.py; det+refine: then
#        synth/lines_refine.py): maps, boxes, dark, vertical,
#        order and truncated must agree; it also runs the driver's --selftest on TEXTDET_SELFTEST
#        regions of a few blobs. TEXTDET_ARGS sets the size of the run (CI: a few seconds; locally, e.g.
#        TEXTDET_ARGS="--seeds 6 --regions 40 --probe-regions 60 --selftest 40 --selftest-blobs 8").
#
# Uses CC, CPPFLAGS, CFLAGS, LDFLAGS, PYTHON, OUT, SAN_ENV and LINES_STRICT from the including Makefile
# and vocr_lines.mk. The driver is rebuilt on every run, so a change of CC or flags always takes effect.

TEXTDET_SELFTEST ?= 3
TEXTDET_ARGS ?= --seeds 2 --regions 6 --probe-regions 8 --screen-every 12 --maps-every 2 --selftest-blobs 2
TEXTDET_OUT = $(abspath $(OUT))/textdet

.PHONY: check-textdet

check-textdet:
	@mkdir -p $(TEXTDET_OUT)
	$(CC) $(LINES_STRICT) -O2 -c vocr_textdet.c -o $(TEXTDET_OUT)/vocr_textdet_strict.o
	$(CC) $(LINES_STRICT) -O2 -c vocr_textdet_test.c -o $(TEXTDET_OUT)/vocr_textdet_test_strict.o
	$(CC) $(CPPFLAGS) -std=c99 -Wall -Wextra -Werror $(CFLAGS) -o $(TEXTDET_OUT)/vocr_textdet_test \
	    vocr_lines.c vocr_lines_rows.c vocr_lines_refine.c vocr_textdet.c vocr_textdet_test.c $(LDFLAGS)
	$(SAN_ENV) $(PYTHON) ../tests/textdet_parity.py --bin $(TEXTDET_OUT)/vocr_textdet_test \
	    --selftest $(TEXTDET_SELFTEST) $(TEXTDET_ARGS)
