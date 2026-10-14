# vocr_lines.mk - checks of the dialogue line finder (vocr_lines.c), included by runtime/Makefile.
#
#   make check-lines
#     1. vocr_lines.c compiled on its own with the strict warning set below (-Werror)
#     2. vocr_lines_test, the host driver, built with $(CC) $(CFLAGS) ... $(LDFLAGS) (in CI:
#        ASan + UBSan, ci/runtime.sh), then its --selftest
#     3. tests/linefind_parity.py: that build against the Python twin (synth/common.py) on
#        LINES_CASES procedural regions, both finders
#     4. vocr_lines_rows.c (the row-callback form) strict, then tests/test_vocr_lines_rows: it
#        against vocr_lines.c on LINES_ROWS_CASES procedural regions, every finder, same flags
#     5. vocr_lines_refine.c (the post-finder refinement) strict, then vocr_lines_refine_test, its
#        driver, built with the same flags: --selftest on REFINE_CASES procedural regions (flags 0
#        is the identity, in-memory and row-callback forms and both arena sizes agree), then
#        tests/lines_refine_parity.py against its Python twin (synth/lines_refine.py), both forms
#
# Uses CC, CPPFLAGS, CFLAGS, LDFLAGS, PYTHON, OUT and SAN_ENV from the including Makefile.
# SAN_ENV also prefixes the Python parity driver, whose child is the sanitized driver, so it
# may set library paths (LD_LIBRARY_PATH) but must not preload a sanitizer runtime.
# The programs are rebuilt on every run, so a change of CC or flags always takes effect.

LINES_STRICT = -std=c99 -pedantic -Wall -Wextra -Werror -Wshadow -Wconversion -Wsign-conversion \
               -Wstrict-prototypes -Wmissing-prototypes -Wcast-align
LINES_CASES ?= 400
LINES_ROWS_CASES ?= 300
REFINE_CASES ?= 200
LINES_OUT = $(abspath $(OUT))/lines

.PHONY: check-lines

check-lines:
	@mkdir -p $(LINES_OUT)
	$(CC) $(LINES_STRICT) -O2 -c vocr_lines.c -o $(LINES_OUT)/vocr_lines_strict.o
	$(CC) $(CPPFLAGS) -std=c99 -Wall -Wextra -Werror $(CFLAGS) -o $(LINES_OUT)/vocr_lines_test \
	    vocr_lines.c vocr_lines_test.c $(LDFLAGS)
	$(SAN_ENV) $(LINES_OUT)/vocr_lines_test --selftest
	$(SAN_ENV) $(PYTHON) ../tests/linefind_parity.py --bin $(LINES_OUT)/vocr_lines_test --cases $(LINES_CASES)
	$(CC) $(LINES_STRICT) -O2 -c vocr_lines_rows.c -o $(LINES_OUT)/vocr_lines_rows_strict.o
	$(CC) $(CPPFLAGS) -std=c99 -Wall -Wextra -Werror $(CFLAGS) -o $(LINES_OUT)/test_vocr_lines_rows \
	    vocr_lines.c vocr_lines_rows.c tests/test_vocr_lines_rows.c $(LDFLAGS)
	$(SAN_ENV) $(LINES_OUT)/test_vocr_lines_rows $(LINES_ROWS_CASES)
	$(CC) $(LINES_STRICT) -O2 -c vocr_lines_refine.c -o $(LINES_OUT)/vocr_lines_refine_strict.o
	$(CC) $(CPPFLAGS) -std=c99 -Wall -Wextra -Werror $(CFLAGS) -o $(LINES_OUT)/vocr_lines_refine_test \
	    vocr_lines.c vocr_lines_refine.c vocr_lines_refine_test.c $(LDFLAGS)
	$(SAN_ENV) $(LINES_OUT)/vocr_lines_refine_test --selftest $(REFINE_CASES)
	$(SAN_ENV) $(PYTHON) ../tests/lines_refine_parity.py --bin $(LINES_OUT)/vocr_lines_refine_test --cases $(LINES_CASES)
	$(SAN_ENV) $(PYTHON) ../tests/lines_refine_parity.py --bin $(LINES_OUT)/vocr_lines_refine_test \
	    --cases $(REFINE_CASES) --seed 2 --rows
