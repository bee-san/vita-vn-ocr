# vocr_post.mk - checks of the counter post-processing (vocr_post.c, issue #109), included by runtime/Makefile.
#
#   make check-post
#     1. vocr_post.c and its driver vocr_post_test.c compiled on their own with LINES_STRICT
#        (vocr_lines.mk: -std=c99 -pedantic ... -Wconversion -Wsign-conversion ... -Werror)
#     2. vocr_post_test, the host driver, built with $(CC) $(CFLAGS) ... $(LDFLAGS) (in CI: ASan + UBSan,
#        ci/runtime.sh) with vocr.c (vocr_class_text), and its self-test run: the rule's corpus contexts, the
#        re-decode's thresholds, windows, merges and bracket, the greedy text
#     3. tests/post_parity.py: random texts through the rule and random top-K lines through the re-decode, in
#        that build and in the twin (eval/vocr_post.py): every count, text and edited class must agree.
#        POST_ARGS sets the size of the run (CI: a few seconds).
#
# Uses CC, CPPFLAGS, CFLAGS, LDFLAGS, PYTHON, OUT, SAN_ENV and LINES_STRICT from the including Makefile
# and vocr_lines.mk. The driver is rebuilt on every run, so a change of CC or flags always takes effect.

POST_ARGS ?= --rules 3000 --lines 2000
POST_OUT = $(abspath $(OUT))/post

.PHONY: check-post

check-post:
	@mkdir -p $(POST_OUT)
	$(CC) $(LINES_STRICT) -O2 -c vocr_post.c -o $(POST_OUT)/vocr_post_strict.o
	$(CC) $(LINES_STRICT) -O2 -c vocr_post_test.c -o $(POST_OUT)/vocr_post_test_strict.o
	$(CC) $(CPPFLAGS) -std=c99 -Wall -Wextra -Werror $(CFLAGS) -o $(POST_OUT)/vocr_post_test \
	    vocr.c vocr_post.c vocr_post_test.c $(LDFLAGS) -lm
	$(SAN_ENV) $(POST_OUT)/vocr_post_test
	$(SAN_ENV) $(PYTHON) ../tests/post_parity.py --bin $(POST_OUT)/vocr_post_test $(POST_ARGS)
