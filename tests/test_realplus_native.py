"""realplus/native.py: the VNDB PSP / PS Vita enumeration, the native frames a training pool may relabel,
their order, and the known-line filter. A tiny fake dump; no network, no screenshots."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for d in ("realplus", "synth", "tests"):
    sys.path.insert(0, str(ROOT / d))

import native as N  # noqa: E402
import shots as S  # noqa: E402

from test_realplus import train_side, write_dump  # noqa: E402


def fake(tmp_path):
    a, b, c, d, e = train_side(5, start=80000)
    vns = {v: "ja" for v in (a, b, c, d, e)}
    vns[e] = "zh"
    shots = [(a, 1, "r1", 480, 272, 0, 0), (a, 2, "r1", 480, 270, 0, 0),      # PSP-only title: native + odd size
             (b, 3, "r2", 960, 544, 0, 0), (c, 4, "r3", 960, 544, 0, 0),      # Vita-only, multi-platform
             (c, 5, "r4", 1280, 720, 0, 0), (d, 6, "r5", 480, 272, 0.5, 0),   # a PC shot; a rated frame
             (e, 7, "r6", 480, 272, 0, 0)]                                    # not a Japanese original
    releases = {"r1": (["psp"], ["ja"], [a]), "r2": (["psv"], ["ja"], [b]), "r3": (["psv"], ["ja"], [c]),
                "r4": (["win"], ["ja"], [c]), "r5": (["psp"], ["ja"], [d]), "r6": (["psp"], ["ja"], [e])}
    write_dump(tmp_path, vns, shots, releases)
    return S.load_dump(tmp_path), (a, b, c, d, e)


def test_native_kind_and_title_groups(tmp_path):
    d, (a, b, c, dd, e) = fake(tmp_path)
    assert N.native_kind({"psp"}, 480, 272) == "psp" and N.native_kind({"psv"}, 960, 544) == "vita"
    assert N.native_kind({"psp"}, 960, 544) is None and N.native_kind({"win"}, 960, 544) is None
    g = N.title_groups(d)
    assert g[a] == "psp-only" and g[b] == "vita-only" and g[c] == "multi-platform" and g[dd] == "psp-only"


def test_classify_reasons_and_selection_order(tmp_path):
    d, (a, b, c, dd, e) = fake(tmp_path)
    rows = N.classify(d, {}, {3: {"realdata", "teacher"}, 4: {"linefind", "realdata"}}, set())
    why = {r["image"]: r["why"] for r in rows}
    assert 5 not in why                                        # PC release: not a console screenshot at all
    assert why[1] == "new" and why[2] == "not-native-size" and why[3] == "pooled"
    assert why[4] == "set:linefind" and why[6] == "rating" and why[7] == "not-japanese-original"
    files = {f"{b}_3": Path("/x/b3.jpg")}
    keep = N.select_rows(rows, files)
    assert [r["image"] for r in keep] == [1, 3]                # PSP-only titles first
    assert keep[0]["source"] == "vndb-new" and keep[1]["path"] == "/x/b3.jpg" and keep[1]["kind"] == "vita"
    rows2 = N.classify(d, {a: "owner-game"}, {}, set())
    assert {r["image"]: r["why"] for r in rows2}[1] == "vn:owner-game"


def test_known_lines_overlap_rule():
    assert N.overlaps([10, 100, 200, 120], [12, 102, 180, 121])            # same line, a little shifted
    assert not N.overlaps([10, 100, 200, 120], [10, 125, 200, 145])        # the next line
    assert not N.overlaps([10, 100, 60, 120], [80, 100, 200, 120])         # same row, no horizontal overlap
    assert N.is_known({"ink": [10, 100, 200, 120]}, [[0, 0, 5, 5], [12, 102, 180, 121]])
    assert not N.is_known({"ink": None}, [[12, 102, 180, 121]])


def test_check_sample_is_stratified_and_seeded():
    both, ink = {"ink": [0, 0, 5, 5], "grad": [0, 0, 6, 6]}, {"ink": [0, 0, 5, 5]}
    man = [{"id": f"v1_{i}_0", "how": "auto" if i % 3 == 0 else "consensus", "crops": both if i % 2 else ink}
           for i in range(300)]
    a, b = N.check_sample(man, 7), N.check_sample(man, 7)
    assert a == b and len(a) == sum(N.CHECK_STRATA.values())
    assert sum(e["stratum"] == "auto" for e in a) == N.CHECK_STRATA["auto"]
    by = {m["id"]: m for m in man}
    assert all(e["crop"] in by[e["id"]]["crops"] for e in a) and len({e["id"] for e in a}) == len(a)


def test_upscale_2x_tells_adrenaline_captures_from_native_frames():
    import cv2
    import numpy as np
    import owner_frames as OF
    rng = np.random.default_rng(0)
    psp = cv2.GaussianBlur((rng.random((272, 480)) * 255).astype(np.float32), (0, 0), 0.6)
    psp[100:120, 40:440] = 255 * (rng.random((20, 400)) > 0.5)                 # some hard-edged "text"
    psp = psp.clip(0, 255).astype(np.uint8)
    native = cv2.resize(psp, (960, 544), interpolation=cv2.INTER_LINEAR)
    native = cv2.addWeighted(native, 0.5, (rng.random((544, 960)) * 255).astype(np.uint8), 0.5, 0)  # full-res detail
    assert OF.upscale_2x(cv2.resize(psp, (960, 544), interpolation=cv2.INTER_NEAREST))[:2] == ("480x272", "nearest")
    assert OF.upscale_2x(cv2.resize(psp, (960, 544), interpolation=cv2.INTER_LINEAR))[:2] == ("480x272", "smooth")
    assert OF.upscale_2x(native)[:2] == ("960x544", None)


def test_owner_folders_map_to_test_and_training_titles():
    import owner_frames as OF
    assert OF.folder_info("memories-off-TEST") == {"vn": "v1152", "role": "test", "eligible": True, "why": None}
    assert OF.folder_info("psp-428-v1299-TEST")["eligible"]
    assert OF.folder_info("vita-aiyoku-no-eustia-v3770-TEST")["role"] == "test"
    assert not OF.folder_info("psp-kanon-v33-TEST")["eligible"]                     # not an owner test title
    ok = OF.folder_info("vita-chaos-child-v14018", reserved={})
    assert ok["role"] == "training" and ok["eligible"] and ok["vn"] == "v14018"
    no = OF.folder_info("psp-kanon-v33", reserved={"v33": "clannad-relation"})
    assert not no["eligible"] and no["why"] == "reserved: clannad-relation"
    assert not OF.folder_info("loose-shots")["eligible"]


def test_committed_native_manifests_are_text_free_and_consistent():
    import json
    d = ROOT / "realplus" / "manifests" / "native"
    shots = [json.loads(x) for x in (d / "shots.jsonl").read_text(encoding="utf-8").splitlines() if x]
    lines = [json.loads(x) for x in (d / "lines_v1.jsonl").read_text(encoding="utf-8").splitlines() if x]
    stats = json.loads((d / "stats.json").read_text(encoding="utf-8"))
    assert stats["build"]["lines"] == len(lines) == 557
    assert {f: v["crops"] for f, v in stats["build"]["files"].items()} == {
        "native_train_ink_v1.npz": 557, "native_train_grad_v1.npz": 411}
    usable = {s["shot"] for s in shots if s["dropped"] is None}
    assert all(m["shot"] in usable and set(m["crops"]) <= {"ink", "grad"} for m in lines)
    assert not any(k in m for m in lines for k in ("label", "text", "reads", "name"))       # no text
    assert all(s["url"].startswith("https://t.vndb.org/") and len(s["sha256"]) == len(s["pixel_sha256"]) == 64
               for s in shots)
    owner = set(json.loads((ROOT / "data" / "manifests" / "synth3_exclude_owner_games.json").read_text())["vns"])
    assert not owner & ({s["vn"] for s in shots} | {m["vn"] for m in lines})


def test_expand_box_clamps_and_owner_decisions_are_narrow():
    import owner_frames as OF
    assert N.expand_box([10, 200, 300, 220], 8, 480, 272) == [2, 192, 308, 228]
    assert N.expand_box([2, 260, 478, 270], 8, 480, 272) == [0, 252, 480, 272]
    assert OF.owner_allowed("v34", "shared-release")
    assert not OF.owner_allowed("v34", "test-set") and not OF.owner_allowed("v4", "shared-release")
    ok = OF.folder_info("psp-planetarian-v34", reserved={"v34": "shared-release"})
    assert ok["eligible"] and ok["owner_decision"].startswith("owner 2026-10-13 19:21Z")
    no = OF.folder_info("psp-planetarian-v34", reserved={"v34": "clannad-relation"})
    assert not no["eligible"]


def test_owner_publish_writes_aggregates_only(tmp_path, monkeypatch):
    import argparse
    import json

    import owner_frames as OF
    out, pool = tmp_path / "build", tmp_path / "build" / "pool"
    (pool / "manifests").mkdir(parents=True)
    (pool / "logs").mkdir()
    (pool / "reads").mkdir()
    (out / "summary.json").write_text(json.dumps({"psp-x-v57": {"role": "training", "vn": "v57", "eligible": True,
                                                                 "why_not": None, "files": 2, "unique": 2}}))
    (pool / "logs" / "leak_test_sets.log").write_text('header\n{"tests": 3, "pool": 2, "pairs": 0}\n')
    (pool / "reads" / "reads.jsonl").write_text('{"shot": "v57_owner0000"}\n{"shot": "v57_owner0001"}\n')
    (pool / "labels.jsonl").write_text(json.dumps({"id": "v57_owner0000_0", "status": "kept", "how": "auto",
                                                   "label": "「秘密の台詞」"}) + "\n")
    (pool / "manifests" / "build_stats_t.json").write_text(json.dumps({"lines": 1}))
    (pool / "manifests" / "lines_t.jsonl").write_text(json.dumps({"id": "v57_owner0000_0", "vn": "v57",
                                                                  "crops": {"ink": [0, 0, 9, 9]}}) + "\n")
    monkeypatch.setattr(OF, "ROOT", tmp_path)
    OF.cmd_publish(argparse.Namespace(out=str(out), pool=str(pool), tag="t", check="pre"))
    text = (tmp_path / "realplus" / "manifests" / "owner" / "stats.json").read_text(encoding="utf-8")
    d = json.loads(text)
    assert d["frames_read"] == 2 and d["lines_by_vn"] == {"v57": {"lines": 1, "ink": 1}}
    assert d["labels_by_vn"] == {"v57": {"kept:auto": 1}} and d["leak"]["test_sets"]["pairs"] == 0
    assert "秘密" not in text and "v57_owner0000" not in text          # no label text, no frame or line id


def test_margin_variants_never_take_in_a_neighbouring_line():
    long_above = [10, 20, 300, 34]                         # a long line just above a short last line
    short = [10, 40, 60, 54]
    g = N.clamp_margin(short, 8, 480, 272, [long_above])
    assert g == [2, 33, 68, 62]                            # grown below and sideways, at most 1 px into the line
    assert min(g[3], long_above[3]) - max(g[1], long_above[1]) <= 1
    assert N.clamp_margin(short, 8, 480, 272, [[40, 44, 90, 50]]) is None      # a box overlapping the line
    assert N.clamp_margin(short, 4, 480, 272, []) == N.expand_box(short, 4, 480, 272)
    name_left = [0, 38, 8, 56]                             # a box on the same row, to the left
    assert N.clamp_margin(short, 8, 480, 272, [name_left])[0] == 7
