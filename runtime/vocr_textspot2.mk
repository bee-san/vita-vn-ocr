# vocr_textspot2.mk - checks of the spotter of VTSP version 1 and 2 (d2s) blobs (vocr_textspot2.c, issue #106),
# included by runtime/Makefile.
#
#   make check-textspot2
#     1. vocr_textspot2.c and its driver vocr_textspot2_test.c compiled on their own with LINES_STRICT
#        (vocr_lines.mk: -std=c99 -pedantic ... -Wconversion -Wsign-conversion ... -Werror), and vocr_textspot2.c
#        once more with -DVOCR_NO_NEON (the PSP's scalar path; on a host it is the same code)
#     2. vocr_textspot2_test, the host driver, and textspot's frozen driver vocr_textspot_test (vocr_textspot.c:
#        the version-1 reference), both built with $(CC) $(CFLAGS) ... $(LDFLAGS) (in CI: ASan + UBSan,
#        ci/runtime.sh) with vocr_lines.c and vocr_lines_rows.c (the dark flags)
#     3. tests/textspot2_parity.py through both: random version-1 networks of eval/textspot2_protocol.json's
#        "kroa" shapes (and a few of train/textdet_model.SHAPES), random version-2 networks of its "kroa_x2" (d2s)
#        shapes, wild-requantisation and probe networks of both versions (designed frames of more than 64
#        candidates; every d2s probe must give its stride-4 probe's result), at half_above 0 and 600 (some with
#        conf_min > 1), on procedural full frames, edge and odd sizes, GREY and RGBA, both forms,
#        VOCR_TEXTSPOT_NO_DARK, through the driver and the twin (synth/textspot2_twin.py): maps, boxes, blocks,
#        order, pick and truncated must agree, every version-1 batch must print what vocr_textspot_test prints and
#        every version-1 arena must be its arena; the C loader must refuse what the twin refuses; it also runs the
#        driver's --selftest on TEXTSPOT2_SELFTEST frames of a few blobs of both versions. TEXTSPOT2_ARGS sets the
#        size of the run (CI: a few seconds; locally, e.g.
#        TEXTSPOT2_ARGS="--seeds 3 --frames 8 --probe-frames 12 --full-every 2" TEXTSPOT2_SELFTEST=40).
#
# Uses CC, CPPFLAGS, CFLAGS, LDFLAGS, PYTHON, OUT, SAN_ENV and LINES_STRICT from the including Makefile and
# vocr_lines.mk. The drivers are rebuilt on every run, so a change of CC or flags always takes effect.

TEXTSPOT2_SELFTEST ?= 6
TEXTSPOT2_ARGS ?= --seeds 1 --frames 2 --probe-frames 3 --full-every 8 --maps-every 2 --selftest-blobs 4 \
                  --arena-every 4
TEXTSPOT2_OUT = $(abspath $(OUT))/textspot2

.PHONY: check-textspot2

check-textspot2:
	@mkdir -p $(TEXTSPOT2_OUT)
	$(CC) $(LINES_STRICT) -O2 -c vocr_textspot2.c -o $(TEXTSPOT2_OUT)/vocr_textspot2_strict.o
	$(CC) $(LINES_STRICT) -O2 -DVOCR_NO_NEON -c vocr_textspot2.c -o $(TEXTSPOT2_OUT)/vocr_textspot2_scalar_strict.o
	$(CC) $(LINES_STRICT) -O2 -c vocr_textspot2_test.c -o $(TEXTSPOT2_OUT)/vocr_textspot2_test_strict.o
	$(CC) $(CPPFLAGS) -std=c99 -Wall -Wextra -Werror $(CFLAGS) -o $(TEXTSPOT2_OUT)/vocr_textspot2_test \
	    vocr_lines.c vocr_lines_rows.c vocr_textspot2.c vocr_textspot2_test.c $(LDFLAGS)
	$(CC) $(CPPFLAGS) -std=c99 -Wall -Wextra -Werror $(CFLAGS) -o $(TEXTSPOT2_OUT)/vocr_textspot_test \
	    vocr_lines.c vocr_lines_rows.c vocr_textspot.c vocr_textspot_test.c $(LDFLAGS)
	$(SAN_ENV) $(PYTHON) ../tests/textspot2_parity.py --bin $(TEXTSPOT2_OUT)/vocr_textspot2_test \
	    --v1-bin $(TEXTSPOT2_OUT)/vocr_textspot_test --selftest $(TEXTSPOT2_SELFTEST) $(TEXTSPOT2_ARGS)
