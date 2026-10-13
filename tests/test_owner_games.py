"""The owner's own Vita games (STEINS;GATE v2002, planetarian v34) are test only (owner 2026-10-13):
realplus/shots.reserved_vns reserves them with every relation and release partner, and synth3 never
uses their names, scenes or text (data/manifests/synth3_exclude_owner_games.json). No network, no
screenshots: a tiny fake dump."""
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
    """v2002 and v34 with: a fan disc of v2002 (relation 'fan'), a parody that names it (relation
    'char', listed only on the parody's side), a compilation release of v34 with an unrelated game,
    a release of the fan disc shared with a third game, and an unrelated training-side VN."""
    fan, parody, boxmate, fanmate, other = train_side(5, start=70000)
    vns = {v: "ja" for v in ("v2002", "v34", fan, parody, boxmate, fanmate, other)}
    shots = [("v2002", 1, "r1", 960, 544, 0, 0), (other, 2, "r9", 960, 544, 0, 0)]
    releases = {"r1": (["psv"], ["ja"], ["v2002"]), "r2": (["win"], ["ja"], ["v34", boxmate]),
                "r3": (["win"], ["ja"], [fan, fanmate]), "r9": (["psv"], ["ja"], [other])}
    write_dump(tmp_path, vns, shots, releases, relations=[("v2002", fan, "fan")])
    with open(tmp_path / "vn_relations", "a", encoding="utf-8") as f:
        f.write(f"{parody}\tv2002\tchar\tf\n")          # one direction only
    return S.load_dump(tmp_path), dict(fan=fan, parody=parody, boxmate=boxmate, fanmate=fanmate, other=other)


def test_owner_games_vns_cover_relations_and_release_partners(tmp_path):
    d, v = fake(tmp_path)
    got = S.owner_games_vns(d)
    assert got["v2002"] == got["v34"] == "owner-game"
    assert got[v["fan"]] == got[v["parody"]] == "owner-game-relation"
    assert got[v["boxmate"]] == got[v["fanmate"]] == "owner-game-shared-release"
    assert v["other"] not in got


def test_reserved_vns_reserves_the_owner_games(tmp_path):
    d, v = fake(tmp_path)
    why = S.reserved_vns(d, test_all=[], fresh=[], dev=[])
    assert why["v2002"] == "owner-game" and why["v34"] == "owner-game"
    assert why[v["fan"]] == "owner-game-relation" and why[v["boxmate"]] == "owner-game-shared-release"
    assert v["other"] not in why
    rows, _ = S.select(d, why, {}, set())
    assert [r["vn"] for r in rows] == [v["other"]]           # STEINS;GATE's Vita frame is never selected


def test_name_hashes_match_synth3_name_rows():
    """A VNDB name 'family given' drops the names.tsv row (src, role, family, given) by hash."""
    assert synth3.name_hash(["岡部", "倫太郎"]) in OG.name_hashes("岡部 倫太郎")
    assert OG.name_hashes("ほしのゆめみ") == {hashlib.sha256("ほしのゆめみ".encode()).hexdigest()}
    assert OG.name_hashes("") == set()


def test_markers_keep_full_names_and_rare_names_only():
    chars = [("c1", "ja", "岡部 倫太郎", ["v2002"]), ("c2", "ja", "ボタン", ["v5"]), ("c3", "ja", "かがり", ["v17102"])]
    always, rare = OG.marker_candidates(["STEINS;GATE", "AIR"], chars, roots_chars={"c1"})
    assert "STEINS;GATE" in always and "岡部倫太郎" in always and "AIR" not in always
    assert rare == {"ボタン", "かがり", "倫太郎"}
    hits = {"ボタン": 900, "かがり": 3, "倫太郎": 7}
    markers, common = OG.markers_of(["STEINS;GATE"], chars, {"c1"}, count=lambda ss: {s: hits[s] for s in ss})
    assert "ボタン" not in markers and common == ["ボタン"] and {"かがり", "倫太郎"} <= set(markers)
    m2, _ = OG.markers_of(["STEINS;GATE"], chars, {"c1"})       # no pools: no one-part names at all
    assert "かがり" not in m2 and "岡部倫太郎" in m2


def test_committed_manifest_excludes_the_owner_games():
    doc = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert doc["kind"] == "synth3-exclusion" and set(doc["protected"]["vita"]) == {"v2002", "v34"}
    vns = set(doc["vns"])
    assert {"v2002", "v34", "v17102", "v27747", "v4"} <= vns          # S;G 0, Snow Globe, a Key box partner
    assert {r["id"] for rs in doc["relations"].values() for r in rs} <= vns
    assert {u for us in doc["vns_sharing_a_release"].values() for u in us} <= vns
    assert set(doc["reasons"]) == vns
    assert all(re.fullmatch(r"[0-9a-f]{64}", h) for k in ("sha256", "pixel_sha256")
               for h in doc["screenshots"][k])
    assert doc["names"]["characters"] > 100 and all(re.fullmatch(r"[0-9a-f]{64}", h)
                                                    for h in doc["names"]["name_sha256"])
    m = doc["text_markers"]["strings"]
    assert {"STEINS;GATE", "planetarian", "岡部倫太郎", "牧瀬紅莉栖", "ほしのゆめみ"} <= set(m)
    assert "ボタン" not in m and "ボタン" in doc["text_markers"]["common_words_not_markers"]
    ex = synth3.load_exclusions([MANIFEST])
    assert ex["vns"] == vns and synth3.marked("岡部倫太郎は言った", ex["markers"])
    assert MANIFEST in synth3.EXCLUDE_MANIFESTS
    import synth3_stats
    assert "v2002" in synth3_stats.reserved_vns(synth3_stats.DEFAULT_EXCLUDE)
