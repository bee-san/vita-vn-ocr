"""Owner pool v2 (docs/owner-pool.md, "Version 2"): the 2026-10-14 drop's layout in synth/owner_drop.py (reader,
dedupe against owner-games, leak drops, later-drop file names), owner_frames.py pool-prep --skip-pool/--prefix and
realplus/owner_pool.py (the confirm rule, leak logs, the merge of two pools, the text-free publish). Fast, offline,
tiny synthetic frames only."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for d in ("synth", "realplus", "tests", ""):
    p = str(ROOT / d) if d else str(ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)

import owner_frames as OF  # noqa: E402
import owner_pool as OP  # noqa: E402

import owner_drop as OD  # noqa: E402
from devnative import build as B  # noqa: E402


def png(path, seed, size=(480, 272)):
    from PIL import Image
    a = np.random.default_rng(seed).integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(a).save(path)
    return path


def games_drop(tmp_path, frames):
    """A drop in the 2026-10-14 layout: frames {folder: [(seed, category)]}, slots psp-npjh5000N."""
    drop = tmp_path / "drop"
    titles = []
    for i, (folder, rows) in enumerate(sorted(frames.items())):
        man = []
        for k, (seed, cat) in enumerate(rows, 1):
            p = png(drop / "games" / folder / "screenshots" / f"{k:06d}.png", seed)
            r = {"image": f"screenshots/{k:06d}.png", "sha256": OD.sha256_file(p), "dimensions": [480, 272],
                 "sequence": k, "source_run": "run", "source_slot": f"psp-npjh5000{i}"}
            if cat == "dialogue":
                r["visually_reviewed_new_dialogue"] = True
            else:
                r["category"] = cat
            man.append(r)
        (drop / "games" / folder / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in man))
        titles.append({"folder": f"games/{folder}", "source_slot": f"psp-npjh5000{i}", "canonical_vndb_ids": []})
    (drop / "dataset-index.json").write_text(json.dumps({"titles": titles}))
    return drop


def test_games_layout_reader_and_drop_tags(tmp_path):
    drop = games_drop(tmp_path, {"Alpha": [(1, "dialogue"), (2, "title_screen")], "Beta": [(3, "dialogue")]})
    assert OD.layout_of(drop) == "games" and OD.drop_tag(drop) == "20261014"
    rows = OD.read_samples(drop)
    assert [(r["title"], r["category"], r["platform"]) for r in rows] == [
        ("Alpha", "dialogue", "psp"), ("Alpha", "title_screen", "psp"), ("Beta", "dialogue", "psp")]
    assert rows[0]["image"] == "games/Alpha/screenshots/000001.png"
    (drop / "dataset-index.json").write_text(json.dumps({"titles": [{"folder": "../x", "source_slot": "psp-a"}]}))
    with pytest.raises(ValueError, match="unsafe"):
        OD.read_games(drop)
    assert OD.dest_name({"image": "games/A/screenshots/000007.png"}, "20261014") == "d20261014_000007.png"
    assert OD.dest_name({"image": "psp/A/000007.png"}) == "000007.png"          # the first drop keeps its names


def test_every_2026_10_14_folder_is_identified_once():
    keys = list(OD.GAMES_20261014)
    assert len(keys) == len(set(keys)) == 87
    assert all(g["vn"].startswith("v") and g["slug"] and g["title"] and g["release"]
               for g in OD.GAMES_20261014.values())
    assert OD.GAMES_20261014[("psp", "Tantei-Jinguuji-Saburo-Hai-to-Diamond")]["vn"] == "v19010"  # no id in the drop
    pub = json.loads((ROOT / "data" / "manifests" / "owner_drop_20261014.json").read_text(encoding="utf-8"))
    assert pub["frames"] == 5050 and sum(g["frames"] for g in pub["games"]) == 5050
    assert {g["vn"] for g in pub["games"] if g["role"] == "test"} == {"v3", "v10856", "v16460", "v18160"}
    assert pub["by_role"] == {"training": 4642, "test": 408}
    assert ".png" not in json.dumps(pub) and '"image"' not in json.dumps(pub)


def test_check_dedupes_against_owner_games_and_drops_leaks(tmp_path, monkeypatch):
    drop = games_drop(tmp_path, {"Alpha": [(1, "dialogue"), (2, "dialogue"), (1, "dialogue")],
                                 "Gamma": [(4, "dialogue")]})
    og = tmp_path / "owner-games"
    png(og / "psp-alpha-v101" / "000001.png", 2)                     # the first drop already has frame 2
    monkeypatch.setattr(OD, "GAMES_20261014", {("psp", "Alpha"): {"vn": "v101", "slug": "alpha", "title": "A",
                                                                  "release": "r1"},
                                               ("psp", "Gamma"): {"vn": "v4", "slug": "gamma", "title": "G",
                                                                  "release": "r2"}})
    test = png(tmp_path / "test" / "v9_1.jpg", 4)                    # Gamma's frame is a test screenshot copy
    plan = OD.check(drop, why={"v4": "clannad"}, why_strict={}, sets={}, owner_games=og, leak=True,
                    groups={"vita-test": [test]})
    fr = {(f["title"], f["image"].rsplit("/", 1)[-1]): f for f in plan["frames"]}
    assert fr[("Alpha", "000002.png")]["duplicate_of"] == "psp-alpha-v101/000001.png"
    assert fr[("Alpha", "000003.png")]["duplicate_of"] == "drop:games/Alpha/screenshots/000001.png"
    assert "duplicate_of" not in fr[("Alpha", "000001.png")]
    g = {x["drop_title"]: x for x in plan["games"]}
    assert g["Alpha"]["role"] == "training" and g["Alpha"]["folder"] == "psp-alpha-v101"     # the folder is reused
    assert g["Alpha"]["duplicates"] == 2 and g["Alpha"]["duplicates_of_owner_games"] == 1
    assert g["Alpha"]["new_frames"] == 1
    assert g["Gamma"]["role"] == "test" and g["Gamma"]["folder"] == "psp-gamma-v4-TEST"
    assert not fr[("Alpha", "000001.png")]["leak_dropped"]
    done = OD.sort(plan, og)
    assert done == {"psp-alpha-v101": 1, "psp-gamma-v4-TEST": 1}
    assert sorted(p.name for p in (og / "psp-alpha-v101").glob("*.png")) == ["000001.png", "d20261014_000001.png"]
    man = [json.loads(x) for x in (og / "psp-alpha-v101" / "manifest_20261014.jsonl").read_text().splitlines()]
    assert [r["skipped"] for r in man] == [None, "duplicate", "duplicate"]
    assert json.loads((og / "psp-alpha-v101" / "ingest_20261014.json").read_text())["new_frames"] == 1
    # after the sort, a re-run of check ignores the drop's own linked frames: same duplicates, same new frames
    again = OD.check(drop, why={"v4": "clannad"}, why_strict={}, sets={}, owner_games=og, leak=True,
                     groups={"vita-test": [test]})
    assert [f.get("duplicate_of") for f in again["frames"]] == [f.get("duplicate_of") for f in plan["frames"]]
    # a training frame at or above MIN_CORR is left out, not linked, and does not block its game
    monkeypatch.setitem(OD.GAMES_20261014, ("psp", "Gamma"), {"vn": "v102", "slug": "gamma", "title": "G",
                                                              "release": "r2"})
    leak = OD.check(drop, why={}, why_strict={}, sets={}, owner_games=tmp_path / "none", leak=True,
                    groups={"vita-test": [test]})
    gl = {x["drop_title"]: x for x in leak["games"]}["Gamma"]
    assert gl["role"] == "training" and gl["leak_frames"] == 1 and gl["new_frames"] == 0


def test_pool_prep_skips_frames_an_earlier_pool_read(tmp_path):
    root, out = tmp_path / "owner-games", tmp_path / "build"
    a, b = png(root / "psp-a-v101" / "1.png", 1), png(root / "psp-a-v101" / "2.png", 2)
    rows = [OF.frame_row(p, "psp-a-v101", {"vn": "v101", "role": "training", "eligible": True, "why": None})
            for p in (a, b)]
    out.mkdir()
    (out / "inventory.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    old = tmp_path / "old-pool"
    (old / "manifests").mkdir(parents=True)
    (old / "manifests" / "frames.jsonl").write_text(json.dumps({"shot": "v101_owner0000",
                                                                "pixel_sha256": rows[0]["pixel_sha256"]}) + "\n")
    import argparse
    pool = tmp_path / "pool2"
    OF.cmd_pool_prep(argparse.Namespace(out=str(out), pool=str(pool), root=str(root), folder=None, prefix="p2f",
                                        skip_pool=[str(old)]))
    reads = [json.loads(x) for x in (pool / "reads" / "shots.jsonl").read_text().splitlines()]
    assert [r["shot"] for r in reads] == ["v101_p2f0001"]          # frame 1 was read by the old pool; ids stay fixed


def label(lid, status="kept", p1=True, conf=None, how="consensus"):
    r = {"id": lid, "status": status, "why": None, "how": how, "label": "あ"}
    if p1:
        r["pass1"] = {"decision": "pick"}
    if conf is not None:
        r["confirm"] = {"agree": conf}
    return r


def test_confirm_rule_needs_an_agreeing_second_look():
    rows = [label("a", p1=False, how="auto"), label("b"), label("c", conf=True), label("d", status="dropped")]
    c = OP.confirm_rule(rows)
    assert [r["status"] for r in rows] == ["kept", "pending", "kept", "dropped"]
    assert rows[1]["why"] == "confirm-missing" and all(r["reviewer_kinds"] == ["assistant"] for r in rows)
    assert c == {"kept:auto": 1, "pending:confirm-missing": 1, "kept:consensus": 1, "dropped:None": 1}


def test_leak_logs_become_a_drop_list():
    log = ("157 test screenshots, 9 pool images\n"
           "  0.952 cross-vn v20326_1.jpg  ~  /x/shots/vita/v17863_p2f0201.png\n"
           '{"tests": 157, "pool": 9, "min_corr": 0.95, "pairs": 1, "same-vn": 0, "cross-vn": 1}\n')
    pairs, summ = OP.leak_pairs(log)
    assert pairs == [(0.952, "cross-vn", "v20326_1.jpg", "v17863_p2f0201")] and summ["pairs"] == 1


def npz(path, rows):
    crops = [np.full((4, 6), i, np.uint8) for i, _ in enumerate(rows)]
    B._save(path, crops, [lab for _, lab in rows], [{"id": lid, "vn": lid.split("_")[0], "target": "psp"}
                                                    for lid, _ in rows])


def test_merge_keeps_every_crop_kind_of_a_line_under_one_cap(tmp_path, monkeypatch):
    old, new = tmp_path / "old", tmp_path / "new"
    for d, rows in ((old, [("v1_owner0000_0", "あ"), ("v1_owner0001_0", "い")]),
                    (new, [("v2_p2f0000_0", "あ"), ("v2_p2f0001_0", "う")])):
        d.mkdir()
        for crop in OP.CROPS:
            npz(d / f"native_train_{crop}_t.npz", rows)
    parts = [(n, *OP.read_npz(d / "native_train_ink_t.npz")) for n, d in (("old", old), ("new", new))]
    crops, labels, metas, dropped = OP.merge_rows(parts, cap=1)
    assert labels == ["あ", "い", "う"] and dropped == {("new", "v2_p2f0000_0")}
    assert [m["owner_pool"] for m in metas] == ["old", "old", "new"] and crops[2].tolist() == [[1] * 6] * 4
    monkeypatch.setattr(OP, "OUT", new)
    (new / "manifests").mkdir()
    import argparse
    OP.cmd_merge(argparse.Namespace(tag="t", old=f"{old}:t"))
    st = json.loads((new / "manifests" / "merge_stats_t.json").read_text())
    assert all(v["crops"] == 4 and v["by_pool"] == {"old": 2, "new": 2} for v in st["files"].values())
    with pytest.raises(SystemExit, match="overwrite"):
        OP.cmd_merge(argparse.Namespace(tag="t", old=f"{old}:t"))


def test_per_game_table_counts_only():
    frames = [{"vn": "v2", "kind": "psp", "folder": "psp-b-v2"}, {"vn": "v2", "kind": "psp", "folder": "psp-b-v2"}]
    labels = [{"id": "v2_p2f0000_0", "status": "kept"}, {"id": "v2_p2f0000_1", "status": "dropped"}]
    man = [{"vn": "v2", "name": True, "crops": {"ink": [0, 0, 1, 1]}},
           {"vn": "v2", "name": False, "crops": {"ink": [0, 0, 1, 1], "grad": [0, 0, 1, 1]}}]
    t = OP.per_game(frames, labels, man, [{"vn": "v2", "title": "B"}])
    assert t == {"v2": {"platform": "psp", "folders": ["psp-b-v2"], "title": "B", "frames": 2, "candidates": 2,
                        "kept_labels": 1, "lines": 2, "dialogue_lines": 1, "crops_ink": 2, "crops_grad": 1}}


def test_name_ids_take_every_look_and_the_same_label_in_the_same_vn(monkeypatch):
    from devnative import decide as D
    labels = [{"id": "v1_p2f0000_0", "vn": "v1", "status": "kept", "label": "岡部", "name": True},   # pass 1 flag
              {"id": "v1_p2f0003_0", "vn": "v1", "status": "kept", "label": "岡部", "how": "auto"},  # same VN, same key
              {"id": "v2_p2f0000_0", "vn": "v2", "status": "kept", "label": "岡部", "how": "auto"},  # another VN
              {"id": "v1_p2f0004_1", "vn": "v1", "status": "kept", "label": "まゆり", "how": "auto"},  # check flag
              {"id": "v1_p2f0005_0", "vn": "v1", "status": "kept", "label": "トゥットゥルー♪"},
              {"id": "v1_p2f0006_0", "vn": "v1", "status": "dropped", "label": None}]
    monkeypatch.setattr(OP.PL, "done_batches", lambda kind: [kind] if kind in ("check", "edge") else [])
    looks = {"check": {"v1_p2f0004_1@ink_m4": {"decision": "text", "name": True}},
             "edge": {"v1_p2f0005_0": {"decision": "pick", "name": False}}}
    monkeypatch.setattr(D, "collect", lambda bs: looks[bs[0]])
    assert OP.name_ids(labels) == {"v1_p2f0000_0", "v1_p2f0003_0", "v1_p2f0004_1"}
