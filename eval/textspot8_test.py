#!/usr/bin/env python3
"""textspot8's test reads (eval/textspot8_protocol.json "test"): eval/textspot6_test.py, unchanged in what it reads and
how, for textspot8's pick, once each, after the selection and confirmation records are committed and pushed:

  textspot8_test.py {title,clannad,tests,record} --selection eval/textspot8_dev_20261027.json --final RUN ...
      (textspot6_test.py's arguments; the final weights are the pick retrained on train + guard + hval)

The only differences: a real read writes under $PV/vocr6-spotter2/protected (textspot6_test's guard, with this
study's private directory), and the selection record is textspot8's (kind "textspot8-selection": textspot6's
schema, written by eval/textspot8_select.py), checked as textspot6_test checks its own (committed, unmodified since,
pushed). --dry-run reads textspot6_test's non-protected stand-ins.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
import textspot6_test as T  # noqa: E402

T.PROTECTED_OUT = T.PRIV / "vocr6-spotter2" / "protected"


def load_selection(path, dry, research=False):
    """textspot6_test.load_selection's checks on textspot8's record (no research release: textspot8 has a pick)."""
    path = Path(path)
    sel = json.loads(path.read_text(encoding="utf-8"))
    if sel.get("kind") != "textspot8-selection":
        raise SystemExit(f"{path}: not a textspot8 selection record")
    if research:
        raise SystemExit("textspot8 has a pick: --research does not apply")
    if not sel.get("pick"):
        raise SystemExit(f"{path}: no pick: nothing is read")
    rec = {"path": str(path), "sha256": T.sha256_file(path), "pick": sel["pick"]}
    if dry:
        return sel, dict(rec, dry_run=True)
    if T.git("ls-files", "--error-unmatch", str(path.resolve())).returncode:
        raise SystemExit(f"{path}: the selection record must be committed before any test read")
    if T.git("diff", "--quiet", "HEAD", "--", str(path.resolve())).returncode:
        raise SystemExit(f"{path}: modified since its commit")
    commit, when = T.git("log", "-1", "--format=%H %cI", "--", str(path.resolve())).stdout.split()
    if not T.git("branch", "-r", "--contains", commit).stdout.strip():
        raise SystemExit(f"{path}: its commit {commit[:12]} is not pushed")
    return sel, dict(rec, commit=commit, committed=when)


T.load_selection = load_selection

if __name__ == "__main__":
    sys.exit(T.main())
