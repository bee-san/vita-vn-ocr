"""realplus/manifests: the committed realplus pool manifest is text-free, training-side only, hashed and
small. Skipped until the manifest is committed; needs no network, screenshots or shared artifacts."""
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for d in ("realplus", "synth", ""):
    sys.path.insert(0, str(ROOT / d) if d else str(ROOT))

import shots as SH  # noqa: E402

from common import is_test_vn  # noqa: E402

MAN = ROOT / "realplus" / "manifests"
SHOT_FILES = sorted(MAN.glob("shots_*.jsonl"))
KEYS = {"shot", "vn", "url", "sha256", "pixel_sha256", "dims", "release", "session", "lines", "crops"}
HEX64 = re.compile(r"^[0-9a-f]{64}$")
URL = re.compile(r"^https://t\.vndb\.org/sf/(\d\d)/(\d+)\.jpg$")
# Every VNDB relation of the fresh native textbox titles (any relation kind, from the 2026-10-12 dump).
FRESH_RELATIONS = {"v15053", "v52836", "v25385", "v28146", "v57035"}

pytestmark = pytest.mark.skipif(not SHOT_FILES, reason="no committed realplus manifest yet")


def rows():
    out = []
    for p in SHOT_FILES:
        out += [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    return out


def test_rows_are_text_free_and_hashed():
    seen = set()
    for r in rows():
        assert set(r) == KEYS, r.get("shot")
        m = URL.match(r["url"])
        assert m and int(m.group(2)) % 100 == int(m.group(1)), r["url"]
        assert r["shot"] == f"{r['vn']}_{m.group(2)}" and re.match(r"^v\d+$", r["vn"])
        assert HEX64.match(r["sha256"]) and HEX64.match(r["pixel_sha256"])
        assert r["lines"] >= 1 and r["crops"] >= r["lines"] and r["session"].startswith(r["vn"] + "/")
        assert r["shot"] not in seen
        seen.add(r["shot"])


def test_files_are_small():
    for p in [*SHOT_FILES, MAN / "stats.json"]:
        assert p.stat().st_size < 2 * 1024 * 1024, p


def test_only_training_side_vns_and_screenshots():
    test_all = set(SH.test_games()["all"])
    protected = test_all | SH.fresh_native_vns() | FRESH_RELATIONS | SH.CLANNAD | SH.GENERAL_VN_DEV
    eval_images = set()
    for p in [*(ROOT / "eval").rglob("*.json"), *(ROOT / "eval").rglob("*.jsonl")]:
        eval_images |= {int(x) for x in SH.IMAGE_ID.findall(p.read_text(encoding="utf-8", errors="replace"))}
    bad = [r["shot"] for r in rows() if r["vn"] in protected or is_test_vn(r["vn"])
           or int(r["shot"].split("_")[1]) in eval_images]
    assert not bad, bad[:20]


def test_stats_pin_the_npz_files_and_the_local_manifests():
    stats = json.loads((MAN / "stats.json").read_text(encoding="utf-8"))
    files = stats["build"]["files"]
    assert {"realplus_train_ink.npz", "realplus_train_grad.npz"} <= set(files)
    for f in files.values():
        assert HEX64.match(f["sha256"]) and f["crops"] >= f["lines"] > 0
    assert all(HEX64.match(v) for v in stats["local_manifests"].values())
    assert stats["screenshot_manifests"] == [p.name for p in SHOT_FILES]
    assert sum(r["lines"] for r in rows()) == stats["build"]["lines"]
    assert stats["review"]["reviewer_kind"] == "assistant"
