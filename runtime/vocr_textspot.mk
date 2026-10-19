# vocr_textspot.mk - checks of the full-frame text spotter (vocr_textspot.c), included by runtime/Makefile.
#
#   make check-textspot
#     1. vocr_textspot.c and its driver vocr_textspot_test.c compiled on their own with LINES_STRICT
#        (vocr_lines.mk: -std=c99 -pedantic ... -Wconversion -Wsign-conversion ... -Werror), and vocr_textspot.c
#        once more with -DVOCR_NO_NEON (the PSP's scalar path; on a host it is the same code)
#     2. vocr_textspot_test, the host driver, built with $(CC) $(CFLAGS) ... $(LDFLAGS) (in CI: ASan + UBSan,
#        ci/runtime.sh) with vocr_lines.c and vocr_lines_rows.c (the dark flags); vocr_textdet.c is not linked: the
#        spotter is self-contained
#     3. tests/textspot_parity.py: random VTDT networks of every family B shape (eval/textspot_protocol.json) and of
#        train/textdet_model.SHAPES, wild-requantisation and probe networks, each wrapped at half_above 0 and 600
#        (some with conf_min > 1), on procedural full frames (960x544, 480x272, 725x544, 870x544), edge and odd sizes
#        and designed frames of more than 64 candidates, GREY and RGBA, both forms, VOCR_TEXTSPOT_NO_DARK, through
#        that build and the twin (synth/textspot_twin.py): maps, boxes, blocks, order, pick and truncated must
#        agree; it also runs the driver's --selftest on TEXTSPOT_SELFTEST frames (the edge sizes first) of a few
#        blobs. TEXTSPOT_ARGS sets the size of the run (CI: a few seconds; locally, e.g.
#        TEXTSPOT_ARGS="--seeds 3 --frames 8 --probe-frames 12 --full-every 2" TEXTSPOT_SELFTEST=40).
#
# Uses CC, CPPFLAGS, CFLAGS, LDFLAGS, PYTHON, OUT, SAN_ENV and LINES_STRICT from the including Makefile and
# vocr_lines.mk. The driver is rebuilt on every run, so a change of CC or flags always takes effect.

TEXTSPOT_SELFTEST ?= 10
TEXTSPOT_ARGS ?= --seeds 1 --frames 3 --probe-frames 4 --full-every 8 --maps-every 2 --selftest-blobs 2
TEXTSPOT_OUT = $(abspath $(OUT))/textspot

.PHONY: check-textspot

check-textspot:
	@mkdir -p $(TEXTSPOT_OUT)
	$(CC) $(LINES_STRICT) -O2 -c vocr_textspot.c -o $(TEXTSPOT_OUT)/vocr_textspot_strict.o
	$(CC) $(LINES_STRICT) -O2 -DVOCR_NO_NEON -c vocr_textspot.c -o $(TEXTSPOT_OUT)/vocr_textspot_scalar_strict.o
	$(CC) $(LINES_STRICT) -O2 -c vocr_textspot_test.c -o $(TEXTSPOT_OUT)/vocr_textspot_test_strict.o
	$(CC) $(CPPFLAGS) -std=c99 -Wall -Wextra -Werror $(CFLAGS) -o $(TEXTSPOT_OUT)/vocr_textspot_test \
	    vocr_lines.c vocr_lines_rows.c vocr_textspot.c vocr_textspot_test.c $(LDFLAGS)
	$(SAN_ENV) $(PYTHON) ../tests/textspot_parity.py --bin $(TEXTSPOT_OUT)/vocr_textspot_test \
	    --selftest $(TEXTSPOT_SELFTEST) $(TEXTSPOT_ARGS)
