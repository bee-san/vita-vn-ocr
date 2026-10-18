# vocr_textgate.mk - checks of the text-presence gate (vocr_textgate.c), included by runtime/Makefile.
#
#   make check-textgate
#     1. vocr_textgate.c and its driver vocr_textgate_test.c compiled on their own with LINES_STRICT
#        (vocr_lines.mk: -std=c99 -pedantic ... -Wconversion -Wsign-conversion ... -Werror)
#     2. vocr_textgate_test, the host driver, built with $(CC) $(CFLAGS) ... $(LDFLAGS) (in CI: ASan +
#        UBSan, ci/runtime.sh) with vocr_lines.c and vocr_lines_rows.c (the finder mode); vocr_textdet.c
#        is not linked: the gate is self-contained
#     3. tests/textgate_parity.py: random blobs of presence shapes (train/textgate_train.SHAPES) and #97
#        shapes (train/textdet_model.SHAPES), wild-requantisation, probe and random-chain blobs, procedural
#        regions in every form and both modes, through that build and the twin (eval/textgate.py on
#        synth/textdet_twin.py): every score, K map and finder-mode result must agree; it also runs the
#        driver's --selftest on TEXTGATE_SELFTEST regions (the small edge sizes first) of a few blobs.
#        TEXTGATE_ARGS sets the size of the run (CI: a few seconds; locally, e.g. TEXTGATE_ARGS="--seeds 2
#        --regions 10 --wild 4 --chains 24 --probes 6" TEXTGATE_SELFTEST=40).
#
# Uses CC, CPPFLAGS, CFLAGS, LDFLAGS, PYTHON, OUT, SAN_ENV and LINES_STRICT from the including Makefile
# and vocr_lines.mk. The driver is rebuilt on every run, so a change of CC or flags always takes effect.

TEXTGATE_SELFTEST ?= 10
TEXTGATE_ARGS ?= --seeds 1 --regions 3 --shapes p8_c4,p8_c12,p4_c4,p4_c8,s4_c4,s8_c8,s4p_c4 --wild 1 --chains 3 \
                 --probes 1 --screen-every 0 --selftest-blobs 2
TEXTGATE_OUT = $(abspath $(OUT))/textgate

.PHONY: check-textgate

check-textgate:
	@mkdir -p $(TEXTGATE_OUT)
	$(CC) $(LINES_STRICT) -O2 -c vocr_textgate.c -o $(TEXTGATE_OUT)/vocr_textgate_strict.o
	$(CC) $(LINES_STRICT) -O2 -c vocr_textgate_test.c -o $(TEXTGATE_OUT)/vocr_textgate_test_strict.o
	$(CC) $(CPPFLAGS) -std=c99 -Wall -Wextra -Werror $(CFLAGS) -o $(TEXTGATE_OUT)/vocr_textgate_test \
	    vocr_lines.c vocr_lines_rows.c vocr_textgate.c vocr_textgate_test.c $(LDFLAGS)
	$(SAN_ENV) $(PYTHON) ../tests/textgate_parity.py --bin $(TEXTGATE_OUT)/vocr_textgate_test \
	    --selftest $(TEXTGATE_SELFTEST) $(TEXTGATE_ARGS)
