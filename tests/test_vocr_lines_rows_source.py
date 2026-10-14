"""runtime/vocr_lines_rows.c is vocr_lines.c in row-callback form; vocr_lines.c is frozen
by pre-registered protocols. The rows file names the vocr_lines.c it was derived from, so a
change to vocr_lines.c fails here until the rows file is brought along (and
runtime/tests/test_vocr_lines_rows.c, in `make check-lines`, shows the two agree)."""
import hashlib
import re
from pathlib import Path

RUNTIME = Path(__file__).resolve().parents[1] / "runtime"


def test_rows_file_follows_the_frozen_finder():
    rows = (RUNTIME / "vocr_lines_rows.c").read_text()
    m = re.search(r"derived-from vocr_lines\.c sha256 ([0-9a-f]{64})", rows)
    assert m, "vocr_lines_rows.c must name the vocr_lines.c it was derived from"
    actual = hashlib.sha256((RUNTIME / "vocr_lines.c").read_bytes()).hexdigest()
    assert m.group(1) == actual, (
        "runtime/vocr_lines.c changed: bring runtime/vocr_lines_rows.c along (same change, "
        "pixel reads through pix_row) and update its derived-from sha256")


def test_frozen_files_are_untouched_by_the_rows_form():
    # the rows form adds files only; the protocol-pinned finder keeps its own API
    header = (RUNTIME / "vocr_lines.h").read_text()
    assert "VocrLinesRows" not in header and "vocr_find_lines_rows" not in header
