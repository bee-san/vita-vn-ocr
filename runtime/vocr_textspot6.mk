# vocr_textspot6.mk - checks of the spotter of VTSP version 1, 2, 3 and 4 blobs (vocr_textspot6.c, issue #128), included
# by runtime/Makefile.
#
#   make check-textspot6
#     1. vocr_textspot6.c and its driver vocr_textspot6_test.c compiled on their own with LINES_STRICT
#        (vocr_lines.mk: -std=c99 -pedantic ... -Wconversion -Wsign-conversion ... -Werror), and vocr_textspot6.c
#        once more with -DVOCR_NO_NEON (the PSP's scalar path; on a host it is the same code), with
#        -DVOCR_TEXTSPOT6_NO_MS (no VTMS path) and with both
#     2. vocr_textspot6_test, the host driver, the same with -DVOCR_TEXTSPOT6_NO_MS, and textspot3's frozen driver
#        vocr_textspot3_test (vocr_textspot3.c: the version-1, 2 and 3 reference), all built with $(CC) $(CFLAGS) ...
#        $(LDFLAGS) (in CI: ASan + UBSan, ci/runtime.sh) with vocr_lines.c and vocr_lines_rows.c (the dark flags)
#     3. tests/textspot6_parity.py through them: random version-4 blobs of both forms (a tiny trunk with tags 1-4, a
#        tiny multi-scale network with tags 2 and 4 by default), wild-requantisation networks, random op graphs with a
#        tagged head, probe networks on designed maps (the twin tests' hand-made maps, tie-breaks, random designed
#        maps; f = 1 and f = 2), version-1, 2 and 3 blobs (every --batch3 line, map and arena vocr_textspot3_test's),
#        procedural full frames, edge and odd sizes, GREY and RGBA, both forms, VOCR_TEXTSPOT_NO_DARK, padded
#        strides, odd arenas, the up-to arena; maps, boxes with tag and tconf, blocks with tag, order, pick,
#        truncated and presence against the twin (synth/textspot6_twin.py); the C loaders (also the NO_MS one)
#        against the twin's refusals and field mutants; and the drivers' --selftest. TEXTSPOT6_ARGS sets the size of
#        the run (CI: a few seconds; locally, e.g. TEXTSPOT6_ARGS="--t-shapes tinyt,t64n8c16 --m-shapes
#        tinym,qS4,qM4,qL5 --seeds 2 --frames 4" TEXTSPOT6_SELFTEST=30).
#
# Uses CC, CPPFLAGS, CFLAGS, LDFLAGS, PYTHON, OUT, SAN_ENV and LINES_STRICT from the including Makefile and
# vocr_lines.mk. The drivers are rebuilt on every run, so a change of CC or flags always takes effect.

TEXTSPOT6_SELFTEST ?= 3
TEXTSPOT6_ARGS ?= --t-shapes tinyt --m-shapes tinym --t-tags 1,2,3,4 --m-tags 2,4 --frames 1 --probe-frames 1 \
                  --probe-variants 0,2 --probe-mc 1 --probe-cells 30x60 --full-every 8 --maps-every 2 --graphs 1 \
                  --mutants 8 --selftest-blobs 9 --arena-every 6 --arena-frames 1 --v3-probe-shapes tiny3 \
                  --v3-graphs 1 --v3-probes 1 --v12-probes 1 --v1-shapes s4_c4 --d2s-shapes s8_c24n4x2
TEXTSPOT6_OUT = $(abspath $(OUT))/textspot6
TEXTSPOT6_SRC = vocr_lines.c vocr_lines_rows.c vocr_textspot6.c vocr_textspot6_test.c

.PHONY: check-textspot6

check-textspot6:
	@mkdir -p $(TEXTSPOT6_OUT)
	$(CC) $(LINES_STRICT) -O2 -c vocr_textspot6.c -o $(TEXTSPOT6_OUT)/vocr_textspot6_strict.o
	$(CC) $(LINES_STRICT) -O2 -DVOCR_NO_NEON -c vocr_textspot6.c -o $(TEXTSPOT6_OUT)/vocr_textspot6_scalar_strict.o
	$(CC) $(LINES_STRICT) -O2 -DVOCR_TEXTSPOT6_NO_MS -c vocr_textspot6.c -o $(TEXTSPOT6_OUT)/vocr_textspot6_noms_strict.o
	$(CC) $(LINES_STRICT) -O2 -DVOCR_NO_NEON -DVOCR_TEXTSPOT6_NO_MS -c vocr_textspot6.c \
	    -o $(TEXTSPOT6_OUT)/vocr_textspot6_scalar_noms_strict.o
	$(CC) $(LINES_STRICT) -O2 -c vocr_textspot6_test.c -o $(TEXTSPOT6_OUT)/vocr_textspot6_test_strict.o
	$(CC) $(CPPFLAGS) -std=c99 -Wall -Wextra -Werror $(CFLAGS) -o $(TEXTSPOT6_OUT)/vocr_textspot6_test \
	    $(TEXTSPOT6_SRC) $(LDFLAGS)
	$(CC) $(CPPFLAGS) -std=c99 -Wall -Wextra -Werror $(CFLAGS) -DVOCR_TEXTSPOT6_NO_MS \
	    -o $(TEXTSPOT6_OUT)/vocr_textspot6_test_noms $(TEXTSPOT6_SRC) $(LDFLAGS)
	$(CC) $(CPPFLAGS) -std=c99 -Wall -Wextra -Werror $(CFLAGS) -o $(TEXTSPOT6_OUT)/vocr_textspot3_test \
	    vocr_lines.c vocr_lines_rows.c vocr_textspot3.c vocr_textspot3_test.c $(LDFLAGS)
	$(SAN_ENV) $(PYTHON) ../tests/textspot6_parity.py --bin $(TEXTSPOT6_OUT)/vocr_textspot6_test \
	    --v3-bin $(TEXTSPOT6_OUT)/vocr_textspot3_test --noms-bin $(TEXTSPOT6_OUT)/vocr_textspot6_test_noms \
	    --selftest $(TEXTSPOT6_SELFTEST) $(TEXTSPOT6_ARGS)
