# vocr_textspot3.mk - checks of the spotter of VTSP version 1, 2 and 3 blobs (vocr_textspot3.c, issue #108), included
# by runtime/Makefile.
#
#   make check-textspot3
#     1. vocr_textspot3.c and its driver vocr_textspot3_test.c compiled on their own with LINES_STRICT
#        (vocr_lines.mk: -std=c99 -pedantic ... -Wconversion -Wsign-conversion ... -Werror), and vocr_textspot3.c
#        once more with -DVOCR_NO_NEON (the PSP's scalar path; on a host it is the same code)
#     2. vocr_textspot3_test, the host driver, and textspot2's frozen driver vocr_textspot2_test (vocr_textspot2.c:
#        the version-1 and version-2 reference), both built with $(CC) $(CFLAGS) ... $(LDFLAGS) (in CI: ASan +
#        UBSan, ci/runtime.sh) with vocr_lines.c and vocr_lines_rows.c (the dark flags)
#     3. tests/textspot3_parity.py through both: random version-3 networks of train/textspot3_model.PROBES and of
#        the protocol's registered M shapes, random op graphs outside the grammar, wild-requantisation networks and
#        version-3 probe networks (which must give #97's stride-4 probe's maps and results), at the exporter's
#        half_above 480 / 272 and at other scales (some with conf_min > 1); version-1 and version-2 blobs of
#        textspot2's parity (every one's batches and arenas must be vocr_textspot2_test's); procedural full frames,
#        edge and odd sizes, designed frames of more than 64 candidates, GREY and RGBA, both forms,
#        VOCR_TEXTSPOT_NO_DARK, padded strides, odd arenas, the up-to arena; maps, boxes, blocks, order, pick and
#        truncated against the twin (synth/textspot3_twin.py); the C loader against the twin's refusals; and the
#        driver's --selftest on TEXTSPOT3_SELFTEST frames of a few blobs of every version. TEXTSPOT3_ARGS sets the
#        size of the run (CI: a few seconds; locally, e.g. TEXTSPOT3_ARGS="--seeds 3 --frames 8 --probe-frames 12
#        --full-every 2" TEXTSPOT3_SELFTEST=40).
#
# Uses CC, CPPFLAGS, CFLAGS, LDFLAGS, PYTHON, OUT, SAN_ENV and LINES_STRICT from the including Makefile and
# vocr_lines.mk. The drivers are rebuilt on every run, so a change of CC or flags always takes effect.

TEXTSPOT3_SELFTEST ?= 4
TEXTSPOT3_ARGS ?= --seeds 1 --frames 2 --probe-frames 3 --full-every 6 --maps-every 2 --selftest-blobs 4 \
                  --arena-every 4 --probe-shapes pS1,pM3 --graphs 3
TEXTSPOT3_OUT = $(abspath $(OUT))/textspot3

.PHONY: check-textspot3

check-textspot3:
	@mkdir -p $(TEXTSPOT3_OUT)
	$(CC) $(LINES_STRICT) -O2 -c vocr_textspot3.c -o $(TEXTSPOT3_OUT)/vocr_textspot3_strict.o
	$(CC) $(LINES_STRICT) -O2 -DVOCR_NO_NEON -c vocr_textspot3.c -o $(TEXTSPOT3_OUT)/vocr_textspot3_scalar_strict.o
	$(CC) $(LINES_STRICT) -O2 -c vocr_textspot3_test.c -o $(TEXTSPOT3_OUT)/vocr_textspot3_test_strict.o
	$(CC) $(CPPFLAGS) -std=c99 -Wall -Wextra -Werror $(CFLAGS) -o $(TEXTSPOT3_OUT)/vocr_textspot3_test \
	    vocr_lines.c vocr_lines_rows.c vocr_textspot3.c vocr_textspot3_test.c $(LDFLAGS)
	$(CC) $(CPPFLAGS) -std=c99 -Wall -Wextra -Werror $(CFLAGS) -o $(TEXTSPOT3_OUT)/vocr_textspot2_test \
	    vocr_lines.c vocr_lines_rows.c vocr_textspot2.c vocr_textspot2_test.c $(LDFLAGS)
	$(SAN_ENV) $(PYTHON) ../tests/textspot3_parity.py --bin $(TEXTSPOT3_OUT)/vocr_textspot3_test \
	    --v2-bin $(TEXTSPOT3_OUT)/vocr_textspot2_test --selftest $(TEXTSPOT3_SELFTEST) $(TEXTSPOT3_ARGS)
