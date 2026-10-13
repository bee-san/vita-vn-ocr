"""The owner's test-only series Memories Off (owner 2026-10-13 10:51Z): realplus/shots.reserved_vns
reserves every VN of the series with its direct relations and release partners, and synth3 never uses
their names, scenes or text (data/manifests/synth3_exclude_owner_games.json). CHAOS;HEAD NOAH, the
owner's training-eligible game, stays on the training side. No network, no screenshots: a fake dump."""
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for d in ("realplus", "synth"):
    sys.path.insert(0, str(ROOT / d))

import shots as S  # noqa: E402

import owner_games as OG  # noqa: E402
import synth3  # noqa: E402
from test_realplus import train_side, write_dump  # noqa: E402

MANIFEST = ROOT / "data" / "manifests" / "synth3_exclude_owner_games.json"


def fake(tmp_path):
    """v1152 with a sequel chain (two hops), a title-only spin-off, a crossover of the sequel (relation
    'char', listed on the crossover's side only) and that crossover's own release partner, a compilation
    of the first sequel with an unrelated game, and an unrelated training-side VN."""
    seq1, seq2, spin, cross, crossmate, boxmate, other = train_side(7, start=90000)
    vns = {v: "ja" for v in ("v1152", seq1, seq2, spin, cross, crossmate, boxmate, other)}
    releases = {"r1": (["psp"], ["ja"], ["v1152"]), "r2": (["win"], ["ja"], [seq1, boxmate]),
                "r3": (["win"], ["ja"], [cross, crossmate]), "r9": (["psv"], ["ja"], [other])}
    shots = [("v1152", 1, "r1", 480, 272, 0, 0), (other, 2, "r9", 960, 544, 0, 0)]
    write_dump(tmp_path, vns, shots, releases, relations=[("v1152", seq1, "seq"), (seq1, seq2, "seq")],
               titles=[("v1152", "メモリーズオフ"), (spin, "Memories Off Festa"), (other, "Ao")])
    with open(tmp_path / "vn_relations", "a", encoding="utf-8") as f:
        f.write(f"{cross}\t{seq2}\tchar\tf\n")
    return S.load_dump(tmp_path), dict(seq1=seq1, seq2=seq2, spin=spin, cross=cross, crossmate=crossmate,
                                       boxmate=boxmate, other=other)


def test_owner_test_vns_reserve_the_whole_series(tmp_path):
    d, v = fake(tmp_path)
    got = S.owner_test_vns(d)
    assert got["v1152"] == got[v["seq1"]] == got[v["seq2"]] == got[v["spin"]] == "owner-test-series"
    assert got[v["cross"]] == "owner-test-relation" and got[v["boxmate"]] == "owner-test-shared-release"
    assert v["crossmate"] not in got and v["other"] not in got     # a relation's own partners stay free


def test_reserved_vns_reserves_the_series_and_keeps_chaos_head_noah_eligible(tmp_path):
    d, v = fake(tmp_path)
    why = S.reserved_vns(d, test_all=[], fresh=[], dev=[])
    assert why["v1152"] == why[v["seq2"]] == "owner-test-series" and why[v["cross"]] == "owner-test-relation"
    assert v["other"] not in why
    rows, _ = S.select(d, why, {}, set())
    assert [r["vn"] for r in rows] == [v["other"]]           # the series' PSP frame is never selected
    assert S.OWNER_TRAINING == {"v22505": "CHAOS;HEAD NOAH"} and "v22505" not in S.OWNER_TEST_SERIES


def test_name_hashes_match_synth3_name_rows():
    """A VNDB name 'family given' drops the names.tsv row (src, role, family, given) by hash."""
    assert synth3.name_hash(["三上", "智也"]) in OG.name_hashes("三上 智也")
    assert OG.name_hashes("ノエル") == {hashlib.sha256("ノエル".encode()).hexdigest()}
    assert OG.name_hashes("") == set()


def test_markers_keep_full_names_and_rare_names_only():
    chars = [("c1", "ja", "三上 智也", ["v1152"]), ("c2", "ja", "ボタン", ["v5"]), ("c3", "ja", "かがり", ["v1153"])]
    always, rare = OG.marker_candidates(["Memories Off", "AIR"], chars, roots_chars={"c1"})
    assert "Memories Off" in always and "三上智也" in always and "AIR" not in always
    assert rare == {"ボタン", "かがり", "智也"}
    hits = {"ボタン": 900, "かがり": 3, "智也": 70}
    markers, common = OG.markers_of(["Memories Off"], chars, {"c1"}, count=lambda ss: {s: hits[s] for s in ss})
    assert "ボタン" not in markers and set(common) == {"ボタン", "智也"} and "かがり" in markers
    m2, _ = OG.markers_of(["Memories Off"], chars, {"c1"})       # no pools: no one-part names at all
    assert "かがり" not in m2 and "三上智也" in m2


def test_committed_manifest_excludes_the_series():
    doc = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert doc["kind"] == "synth3-exclusion" and doc["protected"]["psp"] == ["v1152"]
    vns, series = set(doc["vns"]), set(doc["protected"]["series"])
    assert {"v1152", "v1153", "v1331", "v21964", "v23725"} <= series <= vns
    assert "v22505" not in vns and doc["training_eligible"] == {"v22505": "CHAOS;HEAD NOAH"}
    assert {r["id"] for rs in doc["relations"].values() for r in rs} <= vns
    assert {u for us in doc["vns_sharing_a_release"].values() for u in us} <= vns
    assert set(doc["reasons"]) == vns
    assert all(re.fullmatch(r"[0-9a-f]{64}", h) for k in ("sha256", "pixel_sha256") for h in doc["screenshots"][k])
    assert doc["names"]["characters"] > 100 and all(re.fullmatch(r"[0-9a-f]{64}", h)
                                                    for h in doc["names"]["name_sha256"])
    m = doc["text_markers"]["strings"]
    assert {"Memories Off", "メモリーズオフ"} <= set(m) and "ボタン" not in m
    ex = synth3.load_exclusions([MANIFEST])
    assert ex["vns"] == vns and synth3.marked("メモリーズオフの話", ex["markers"])
    assert MANIFEST in synth3.EXCLUDE_MANIFESTS
    import synth3_stats
    assert "v1152" in synth3_stats.reserved_vns(synth3_stats.DEFAULT_EXCLUDE)
