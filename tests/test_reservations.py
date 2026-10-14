"""The 2026-10-14 reservation (docs/reservations.md; owner 13:25Z): the committed eval/manifests/reserved.json, the
helpers in synth/common.py, the development sample, the leak check (eval/tools/reserve_check.py) and the pool split
(realplus/reclaim.py --split). Fast, no network, no screenshots."""
import collections
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for d in ("realplus", "synth", "eval/tools", ""):
    p = str(ROOT / d) if d else str(ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)

import reclaim  # noqa: E402
import reserve_check as RC  # noqa: E402
import shots as S  # noqa: E402

import common  # noqa: E402

RES = json.loads((ROOT / "eval" / "manifests" / "reserved.json").read_text(encoding="utf-8"))
TG = json.loads((ROOT / "eval" / "manifests" / "test_games.json").read_text(encoding="utf-8"))
SETS = set(TG["vita-test"]) | set(TG["psp-test"])


def test_reservation_holds_only_the_test_sets_clannad_and_release_partners():
    test = RES["test"]
    assert RES["kind"] == "vita-vn-ocr-reservation" and RES["policy"].startswith("owner 2026-10-14 13:25Z")
    assert len(SETS) == 28 and all(test[v] == "test-set" for v in SETS)
    assert set(test.values()) <= {"test-set", "test-shared-release", "clannad", "clannad-family", "clannad-relation",
                                  "clannad-shared-release"}
    assert test["v4"] == "clannad" and S.CLANNAD_FAMILY <= set(test)
    assert "v34" not in test                                   # planetarian: owner decision, training data
    assert set(RES["test_shared_releases"]) == {v for v, r in test.items() if r == "test-shared-release"}
    assert set(RES["sessions"]) == set(TG["sessions"]) and len(RES["sessions"]) == 31
    assert set(TG["all"]) == SETS | set(RES["test_shared_releases"])
    # nothing of the old hash splits is reserved for being in them
    old = set(TG["all_20261012"])
    assert len(old) == 172 and (set(test) & old) - SETS <= set(RES["test_shared_releases"])
    hashed = [f"v{i}" for i in range(1, 3000) if common.is_test_vn(f"v{i}")]
    assert sum(v in test for v in hashed) < 0.05 * len(hashed)


def test_development_sample_is_small_fixed_and_off_the_owner_games_and_pools():
    dev = RES["dev_sample"]
    core = [v for v, r in dev.items() if r == "dev-sample"]
    assert len(core) == S.DEV_SAMPLE_SIZE == RES["dev_sample_size"] == 100
    assert set(dev.values()) <= {"dev-sample", "dev-sample-shared-release"}
    assert not set(dev) & set(RES["test"])
    # every owner screenshot trains: the carve's owner list is pinned in the file (the live owner-games folders grow
    # with every drop, so they are never read here), and it holds every owner VN the repo names
    owner = RES["owner_vns"]
    assert owner == sorted(set(owner), key=lambda v: int(v[1:])) and all(v.startswith("v") for v in owner)
    named = set(S.OWNER_TRAINING) | set(S.OWNER_ALLOWED) | set(S.OWNER_TEST_SERIES)
    for drop in sorted((ROOT / "data" / "manifests").glob("owner_drop_*.json")):
        named |= {g["vn"] for g in json.loads(drop.read_text(encoding="utf-8"))["games"]}
    assert named <= set(owner)
    assert not set(dev) & set(owner)
    assert not set(core) & set(S.training_pool_vns())          # never trained on so far
    rows = RES["dev_sample_vns"]
    assert {r["vn"] for r in rows} == set(core) and len(rows) == len(core)
    assert [r["tier"] for r in rows] == sorted(r["tier"] for r in rows)
    assert all(r["lines"] for r in rows)


def test_common_helpers_follow_the_committed_file():
    assert common.test_set_vns() == RES["test"] and common.dev_sample_vns() == RES["dev_sample"]
    assert common.reserved_sessions() == set(RES["sessions"])
    dv = next(v for v, r in RES["dev_sample"].items() if r == "dev-sample")
    assert common.is_reserved_vn("v4") and common.is_reserved_vn(dv) and not common.is_reserved_vn(dv, dev=False)
    hashed = next(f"v{i}" for i in range(5, 500) if common.is_test_vn(f"v{i}") and f"v{i}" not in RES["test"]
                  and f"v{i}" not in RES["dev_sample"])
    assert not common.is_reserved_vn(hashed)                  # the old split is history only
    assert common.is_test_vn("v3396")                         # the frozen hash itself is unchanged


def write_dump(base, vns, releases, relations=(), titles=()):
    base.mkdir(parents=True, exist_ok=True)

    def put(name, rows):
        (base / name).write_text("".join("\t".join(map(str, r)) + "\n" for r in rows), encoding="utf-8")
    put("vn", [(v, "\\N", "\\N", "ja", 0, 0, 0, 0, 0, 0, 0, "", "") for v in vns])
    put("vn_titles", [(v, "ja", "t", t, "\\N") for v, t in titles])
    put("images", [])
    put("vn_screenshots", [])
    put("releases_platforms", [(r, "win") for r in releases])
    put("releases_titles", [(r, "ja", "f", "x", "\\N") for r in releases])
    put("releases_vn", [(r, v, "complete") for r, vs in releases.items() for v in vs])
    put("vn_relations", [(a, b, k, "t") for a, b, k in relations] + [(b, a, k, "t") for a, b, k in relations])


def test_dev_sample_tiers_and_candidate_rules(tmp_path):
    vns = [f"v{900000 + i}" for i in range(12)]
    a, b, c, dd, e, f, g, h, own, near, pool, twin = vns
    write_dump(tmp_path, vns, {"r1": [own, near], "r2": [pool, twin]})
    d = S.load_dump(tmp_path)
    counts = {a: {"devnative:psp": 3}, b: {"teacher_val:vita": 2}, c: {"teacher_val:hires": 5},
              dd: {"real_val:hires": 4}, e: {"teacher_val:psp": 1, "devnative:vita": 1}, f: {"teacher_val:hires": 1},
              own: {"devnative:psp": 9}, near: {"devnative:psp": 9}, twin: {"devnative:psp": 9},
              g: {"teacher_val:hires": 1}, h: {"teacher_val:hires": 1}}
    counts = {v: collections.Counter(x) for v, x in counts.items()}
    rows, why = S.dev_sample(d, {g: "test-set"}, counts=counts, size=5, owner={own}, pools={pool: {"realplus"}})
    got = [(r["vn"], r["tier"]) for r in rows]
    assert {v for v, _ in got[:2]} == {a, e} and got[2] == (b, 2)        # devnative first, then native teacher_val
    assert {v for v, t in got if t == 3} <= {c, f, h} and len(got) == 5 and dd not in {v for v, _ in got}
    assert why["test"] == 1 and why["owner-game"] == 1 and why["owner-shared-release"] == 1
    assert why["two-vn-release-with-a-training-pool-vn"] == 1 and why["not-drawn"] == 2
    again, _ = S.dev_sample(d, {g: "test-set"}, counts=counts, size=5, owner={own}, pools={pool: {"realplus"}})
    assert again == rows                                                 # seeded: the same draw every time


def test_row_facts_reads_every_pool_layout():
    assert RC.row_facts({"id": "v12_345_0", "session": "v12/r67/0"})[:3] == ("v12", 345, "r67")
    assert RC.row_facts({"shot": "v9_8", "vn": "v9", "release": "r5", "sha256": "a" * 64})[:4] == \
        ("v9", 8, "r5", {"a" * 64})
    assert RC.row_facts({"url": "https://t.vndb.org/sf/44/112044.jpg", "vn": "v3"})[:2] == ("v3", 112044)
    assert RC.row_facts({"image": 7, "vn": "v1", "session": "owner/psp-x-v1"})[:3] == ("v1", 7, None)


def test_check_pool_blocks_test_sessions_screenshots_and_the_dev_sample():
    prot = {"test": {"v4": "clannad", "v3396": "test-set"}, "dev": {"v5837": "dev-sample"}, "sessions": {"r14071"},
            "images": {111}, "urls": {"https://t.vndb.org/sf/11/111.jpg"}, "sha256": {"b" * 64}}
    rows = [{"id": "v4_1_0"}, {"id": "v10_2_0", "session": "v10/r14071/0"}, {"shot": "v11_111"},
            {"vn": "v12", "sha256": "b" * 64}, {"id": "v5837_9_0"}, {"id": "v20_3_0"}]
    r = RC.check_pool(rows, prot)
    assert r["blockers"] == ["development sample", "test VN", "test screenshot", "test session"]
    assert r["checks"]["test screenshot"]["rows"] == 2 and r["checks"]["test VN"]["vn_list"] == ["v4"]
    info = RC.check_pool(rows, prot, dev_mode="info")
    assert "development sample" not in info["blockers"] and "development sample (info)" in info["checks"]
    clean = RC.check_pool([{"id": "v20_3_0"}, {"id": "v21_4_0"}], prot)
    assert clean["blockers"] == [] and clean["rows"] == 2
    dump = ({"v77": "test-shared-release"}, {"r999"})
    assert RC.check_pool([{"id": "v78_1_0", "session": "v78/r999/0"}], prot, dump=dump)["blockers"] == \
        ["test VN (dump)"]


def test_protected_reads_the_committed_test_sets():
    prot = RC.protected()
    assert prot["test"] == RES["test"] and prot["sessions"] == set(RES["sessions"])
    rows = [json.loads(x) for x in (ROOT / "eval" / "manifests" / "psp-test.jsonl").read_text(encoding="utf-8")
            .splitlines() if x.strip()]
    r = RC.check_pool(rows, prot)
    assert {"test VN", "test screenshot", "test session"} <= set(r["blockers"])


def test_load_pool_reads_npz_jsonl_and_screenshot_dirs(tmp_path):
    from devnative import build as B
    B._save(tmp_path / "p.npz", [np.zeros((4, 6), np.uint8), np.ones((5, 3), np.uint8)], ["あ", "い"],
            [{"id": "v1_2_0", "vn": "v1"}, {"id": "v4_5_0", "vn": "v4"}])
    assert [m["vn"] for m in RC.load_pool(tmp_path / "p.npz")] == ["v1", "v4"]
    (tmp_path / "l.jsonl").write_text('{"shot": "v3_9", "vn": "v3"}\n\n', encoding="utf-8")
    assert RC.load_pool(tmp_path / "l.jsonl") == [{"shot": "v3_9", "vn": "v3"}]
    (tmp_path / "shots").mkdir()
    (tmp_path / "shots" / "v8_77.jpg").write_bytes(b"x")
    assert RC.load_pool(tmp_path / "shots")[0]["vn"] == "v8"


def test_split_npz_cuts_by_vn_and_drops_test_rows(tmp_path):
    from devnative import build as B
    crops = [np.full((3, 4), i, np.uint8) for i in range(4)]
    B._save(tmp_path / "src.npz", crops, ["a", "b", "c", "d"],
            [{"vn": "v1"}, {"vn": "v2"}, {"vn": "v4"}, {"vn": "v1"}])
    st = reclaim.split_npz(tmp_path / "src.npz", tmp_path / "t.npz", tmp_path / "d.npz", held={"v2"}, test={"v4"})
    assert st["train"]["rows"] == 2 and st["dev"]["rows"] == 1 and st["dropped_test"] == {"v4": 1}
    z = np.load(tmp_path / "t.npz")
    assert json.loads(str(z["labels"])) == ["a", "d"] and z["shapes"].tolist() == [[3, 4], [3, 4]]
    assert z["data"][12:].tolist() == [3] * 12                             # the fourth crop's pixels, in order


def test_committed_reclaim_manifests_add_up():
    doc = json.loads((ROOT / "realplus" / "manifests" / "reclaim_20261014.json").read_text(encoding="utf-8"))
    assert doc["reserved"]["before_realplus_2026_10_12_file"] == 18805
    assert doc["reserved"]["after"] == len(RES["test"]) + len(RES["dev_sample"])
    rp, na = doc["realplus"], doc["native"]
    assert rp["screenshots"] == sum(k["screenshots"] for k in rp["by_kind"].values()) == sum(rp["by_old_why"].values())
    assert na["frames"] == sum(na["by_kind"].values()) == sum(na["by_new_class"].values())
    assert doc["distinct_vndb_screenshots"]["realplus_or_native"] <= rp["screenshots"] + na["frames"]
    assert doc["owner_frames"]["vita-clannad-v4-TEST"]["role"] == "test"
    # an owner folder is test-only exactly when its VN is a test VN (the 2026-10-14 drop brought four non-CLANNAD
    # ones: Time Travelers, Utawarerumono, Collar x Malice, Sweet Clown); every other owner frame trains
    assert all((v["role"] == "test") == (v["vn"] in RES["test"]) for v in doc["owner_frames"].values())
    splits = json.loads((ROOT / "realplus" / "manifests" / "reclaim_20261014_splits.json").read_text(encoding="utf-8"))
    assert set(splits["sets"]["teacher_train"]["dropped_test"]) <= set(RES["test"])
    assert all(not v["dropped_test"] for k, v in splits["sets"].items() if k.endswith("_val") or "devnative" in k)


def test_background_exclusions_follow_the_reservation():
    import bg_exclude
    ex = bg_exclude.load_exclusions()
    assert set(RES["test"]) <= ex and set(RES["dev_sample"]) <= ex
    listed = (ROOT / "data" / "manifests" / "synth2_bg_exclude.txt").read_text(encoding="utf-8")
    rows = [x.split() for x in listed.splitlines() if x.strip() and not x.startswith("#")]
    assert "research-real" not in listed
    assert {v for v, tag in rows if tag == "eval-test"} == set(RES["test"])
    assert {v for v, tag in rows if tag == "dev-sample"} == set(RES["dev_sample"]) - set(RES["test"])


def test_reserve_check_joins_screenshot_hashes_and_reports_coverage(tmp_path):
    """Pool rows carry no hashes: the leak check finds them through the shot key in the screenshot manifests, or
    decodes a local copy, and reports how many rows it could hash-check (audit F1)."""
    from PIL import Image
    a = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
    Image.fromarray(a).save(tmp_path / "v7_70.png")
    Image.fromarray(a).save(tmp_path / "copy.png", compress_level=1)           # same pixels, other bytes
    pix = S.pixel_sha256(tmp_path / "v7_70.png")
    man = tmp_path / "shots.jsonl"
    man.write_text(json.dumps({"shot": "v5_50", "vn": "v5", "sha256": "f" * 64}) + "\n" +
                   json.dumps({"vn": "v6", "url": "https://t.vndb.org/sf/60/60.jpg", "sha256": "e" * 64}) + "\n" +
                   json.dumps({"shot": "v8_owner0001", "vn": "v8", "path": str(tmp_path / "copy.png")}) + "\n",
                   encoding="utf-8")
    idx = RC.shot_index(sources=[(man, "rows")])
    assert set(idx) == {"v5_50", "v6_60", "v8_owner0001"} and idx["v5_50"]["file"] == {"f" * 64}
    assert RC.shot_key({"id": "v8_owner0001_3"}) == "v8_owner0001" and RC.shot_key({"id": "v1_2_0"}) == "v1_2"
    prot = {"test": {}, "dev": {}, "sessions": set(), "images": set(), "urls": set(), "sha256": {"e" * 64}}
    rows = [{"id": "v5_50_0", "vn": "v5"}, {"id": "v6_60_1", "vn": "v6"}, {"id": "v8_owner0001_0", "vn": "v8"},
            {"id": "v9_90_0", "vn": "v9"}]
    r = RC.check_pool(rows, prot, pixels={pix: {"owner-clannad:x"}}, index=idx)
    assert r["checks"]["test screenshot"]["vn_list"] == ["v6"]                # by file sha256, through the join
    assert r["checks"]["test pixels"]["vn_list"] == ["v8"]                    # by decoded pixels of the local copy
    assert r["hash_coverage"] == {"rows": 4, "hash_checked": 3, "pixel_checked": 1, "share": 0.75}
    strict = RC.check_pool(rows, prot, pixels={}, index=idx, require=True)
    assert strict["checks"]["hash coverage"]["vn_list"] == ["v9"] and "hash coverage" in strict["blockers"]
