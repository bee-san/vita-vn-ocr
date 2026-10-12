"""realplus/: screenshot selection from a VNDB dump, the reserved VNs and screenshots, pixel hashes.
Fast, no network, no screenshots: a tiny fake dump in a temporary directory."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for d in ("realplus", "synth"):
    sys.path.insert(0, str(ROOT / d))

import shots as S  # noqa: E402

from common import is_test_vn  # noqa: E402


def train_side(n, start=1):
    """n VN ids that are neither is_test_vn nor in the val bucket."""
    out, v = [], start
    while len(out) < n:
        vid = f"v{v}"
        if not is_test_vn(vid) and not S.val_bucket(vid):
            out.append(vid)
        v += 1
    return out


def write_dump(base, vns, shots, releases, relations=(), titles=()):
    """vns: {vn: olang}; shots: [(vn, image, rid, w, h, sexual, violence)];
    releases: {rid: (platforms, langs, vns)}; relations: [(a, b, kind)]; titles: [(vn, title)]."""
    base.mkdir(parents=True, exist_ok=True)

    def put(name, rows):
        (base / name).write_text("".join("\t".join(map(str, r)) + "\n" for r in rows), encoding="utf-8")
    put("vn", [(v, "\\N", "\\N", o, 0, 0, 0, 0, 0, 0, 0, "", "") for v, o in vns.items()])
    put("vn_titles", [(v, "ja", "t", t, "\\N") for v, t in titles])
    put("images", [(f"sf{i}", w, h, 1, int(sx * 100), 0, int(vi * 100), 0, 1) for _, i, _, w, h, sx, vi in shots])
    put("vn_screenshots", [(v, f"sf{i}", rid) for v, i, rid, *_ in shots])
    put("releases_platforms", [(r, p) for r, (ps, _, _) in releases.items() for p in ps])
    put("releases_titles", [(r, lang, "f", "x", "\\N") for r, (_, ls, _) in releases.items() for lang in ls])
    put("releases_vn", [(r, v, "complete") for r, (_, _, vs) in releases.items() for v in vs])
    put("vn_relations", [(a, b, k, "t") for a, b, k in relations] + [(b, a, k, "t") for a, b, k in relations])


@pytest.mark.parametrize("plats, dims, kind", [
    (["psv"], (960, 544), "vita"), (["psv"], (960, 540), "hires"), (["psp"], (480, 272), "psp"),
    (["win"], (800, 600), "hires"), (["win"], (1280, 720), "hires"), (["win"], (600, 480), None),
    (["p98"], (640, 400), "hires"), (["win"], (640, 399), None), (["sfc"], (256, 224), None),
    (["win", "psv"], (1280, 720), "hires"), (["and"], (1280, 720), "hires"), (["win"], (1000, 500), None),
])
def test_kind_of(plats, dims, kind):
    assert S.kind_of(plats, *dims) == kind


def test_url_and_image_ids(tmp_path):
    assert S.url_of(112044) == "https://t.vndb.org/sf/44/112044.jpg"
    assert S.url_of(5) == "https://t.vndb.org/sf/05/5.jpg"
    p = tmp_path / "m.jsonl"
    p.write_text('{"url": "https://t.vndb.org/sf/44/112044.jpg"}\n{"shot": "v12_777.jpg"}\n')
    assert S.image_ids_in(p) == {112044, 777}


def test_reserved_vns_family_and_shared_releases(tmp_path):
    a, b, c, dd, e, f, g, h = train_side(8, 1000)
    test_vn = next(f"v{i}" for i in range(1, 500) if is_test_vn(f"v{i}"))
    val_vn = next(f"v{i}" for i in range(1, 500) if S.val_bucket(f"v{i}") and not is_test_vn(f"v{i}"))
    vns = {v: "ja" for v in (a, b, c, dd, e, f, g, h, test_vn, val_vn, "v4")}
    releases = {
        "r1": (["win"], ["ja"], [a, b]),               # b on a release with a development VN
        "r2": (["win"], ["ja"], [test_vn, c]),         # a pair with an is_test_vn VN
        "r3": (["win"], ["ja"], [test_vn, dd, e]),      # a compilation of unrelated games
        "r4": (["win"], ["ja"], [val_vn, f, g]),       # a compilation, but f is val_vn's sequel
    }
    write_dump(tmp_path, vns, [], releases, relations=[(a, h, "seq"), (val_vn, f, "fan"), ("v4", g, "char")],
               titles=[(g, "Some game")])
    d = S.load_dump(tmp_path)
    why = S.reserved_vns(d, test_all=set(), fresh=set(), dev={a})
    assert why[a] == "development" and why["v4"] == "clannad"
    assert why[h] == "family"                       # a's sequel
    assert why[b] == "shared-release"               # any release shared with a development VN
    assert why[c] == "shared-release"               # a two-VN release with a test VN
    assert dd not in why and e not in why            # an unrelated compilation with a test VN: allowed
    assert why[f] == "shared-release" and g not in why   # ("char" is not a FAMILY relation)
    assert why[test_vn] == "test-vn" and why[val_vn] == "val-bucket"


def test_clannad_by_title(tmp_path):
    a, b = train_side(2, 100)
    write_dump(tmp_path, {a: "ja", b: "ja"}, [], {}, titles=[(a, "CLANNAD Side Stories"), (b, "Other")])
    why = S.reserved_vns(S.load_dump(tmp_path), test_all=set(), fresh=set(), dev=set())
    assert why[a] == "clannad" and b not in why


def test_select_filters_order_and_sessions(tmp_path):
    a, b, c = train_side(3, 200)
    vns = {a: "ja", b: "ja", c: "en"}
    shots = [(a, 1000, "r1", 800, 600, 0, 0), (a, 1001, "r1", 800, 600, 0, 0), (a, 1100, "r1", 800, 600, 0, 0),
             (a, 1002, "r1", 800, 600, 0.5, 0),        # rating
             (a, 1003, "r2", 800, 600, 0, 0),          # Japanese and Chinese
             (a, 1005, "r4", 800, 600, 0, 0),          # Japanese and English: allowed
             (a, 1004, "r3", 960, 544, 0, 0),          # vita
             (b, 2000, "r1b", 640, 480, 0, 0), (b, 2001, "r1b", 640, 480, 0, 0),
             (b, 2002, "rt", 640, 480, 0, 0),          # a test session
             (b, 2003, "r1b", 640, 480, 0, 0),         # already in a pool
             (c, 3000, "rc", 800, 600, 0, 0)]          # not a Japanese original
    releases = {"r1": (["win"], ["ja"], [a]), "r2": (["win"], ["ja", "zh-Hans"], [a]),
                "r3": (["psv"], ["ja"], [a]), "r4": (["win"], ["ja", "en"], [a]),
                "r1b": (["win"], ["ja"], [b]), "rt": (["win"], ["ja"], [b]), "rc": (["win"], ["ja"], [c])}
    write_dump(tmp_path, vns, shots, releases)
    d = S.load_dump(tmp_path)
    keep, why = S.select(d, {}, {2003: {"realdata"}}, {"rt"})
    got = {r["image"] for r in keep}
    assert got == {1000, 1001, 1100, 1004, 1005, 2000, 2001}
    assert why["rating"] == 1 and why["language"] == 1 and why["test-session"] == 1
    assert why["in-pool:realdata"] == 1 and why["not-japanese-original"] == 1
    assert keep[0]["kind"] == "vita"                                    # native kinds first
    hires = [r for r in keep if r["kind"] == "hires"]
    assert [r["rank"] for r in hires] == sorted(r["rank"] for r in hires)  # round-robin over VNs
    assert len({r["vn"] for r in hires[:2]}) == 2
    sess = {r["image"]: r["session"] for r in keep}
    assert sess[1000] == sess[1001] == f"{a}/r1/0" and sess[1100] == f"{a}/r1/1"   # id gap >= 50
    assert all(r["url"] == S.url_of(r["image"]) for r in keep)
    capped, why2 = S.select(d, {}, {}, set(), per_vn_hires=1)
    assert sum(r["vn"] == a and r["kind"] == "hires" for r in capped) == 1 and why2["hires-cap"] == 6
    # the cap keeps the same shot whatever else is in the dump
    keep_a = {r["image"] for r in capped if r["vn"] == a and r["kind"] == "hires"}
    d["shots"] = [s for s in d["shots"] if s[0] == a]
    assert keep_a == {r["image"] for r in S.select(d, {}, {}, set(), per_vn_hires=1)[0] if r["kind"] == "hires"}


def test_reserved_vn_rows_are_dropped(tmp_path):
    a, b = train_side(2, 300)
    write_dump(tmp_path, {a: "ja", b: "ja"}, [(a, 1, "r1", 800, 600, 0, 0), (b, 2, "r2", 800, 600, 0, 0)],
               {"r1": (["win"], ["ja"], [a]), "r2": (["win"], ["ja"], [b])})
    keep, why = S.select(S.load_dump(tmp_path), {a: "development"}, {}, set())
    assert [r["vn"] for r in keep] == [b] and why["vn:development"] == 1


def test_pixel_sha256_ignores_encoding(tmp_path):
    from PIL import Image
    rng = np.random.default_rng(0)
    a = rng.integers(0, 255, (20, 30, 3), dtype=np.uint8)
    Image.fromarray(a).save(tmp_path / "a.png")
    Image.fromarray(a).save(tmp_path / "b.png", compress_level=1)
    Image.fromarray(a).convert("RGBA").save(tmp_path / "c.png")
    hs = {S.pixel_sha256(tmp_path / n) for n in ("a.png", "b.png", "c.png")}
    assert len(hs) == 1
    assert S.file_sha256(tmp_path / "a.png") != S.file_sha256(tmp_path / "b.png")
    b = a.copy()
    b[0, 0, 0] ^= 1
    Image.fromarray(b).save(tmp_path / "d.png")
    assert S.pixel_sha256(tmp_path / "d.png") not in hs


def test_reserved_sets_cover_the_repo_sets():
    """Every committed eval/linefind/realdata/teacher screenshot is reserved, and the protected
    VN lists include CLANNAD, the fresh native titles and the development VNs."""
    ids = S.reserved_screenshots()
    for path in [ROOT / "eval" / "manifests" / "vita-test.jsonl", ROOT / "eval" / "manifests" / "psp-test.jsonl",
                 *sorted((ROOT / "eval" / "linefind" / "manifests").glob("*.jsonl")),
                 *[ROOT / "realdata" / "manifests" / f"shots_{k}.jsonl" for k in S.KINDS]]:
        assert S.image_ids_in(path) <= set(ids), path
    assert {"v14909", "v20519"} <= S.fresh_native_vns()
    assert S.GENERAL_VN_DEV <= S.dev_vns() and "v4" in S.dev_vns()
    tg = json.loads((ROOT / "eval" / "manifests" / "test_games.json").read_text())
    assert tg["all"] and tg["sessions"]


def test_fresh_native_titles_and_all_their_relations(tmp_path):
    a, b, c = train_side(3, 2000)
    fresh = "v14909"
    vns = {v: "ja" for v in (a, b, c, fresh)}
    write_dump(tmp_path, vns, [], {"r9": (["win"], ["ja"], [fresh, c, b, a])},
               relations=[(fresh, a, "char"), (fresh, b, "set")])
    why = S.reserved_vns(S.load_dump(tmp_path), test_all=set(), fresh=None, dev=set())
    assert why[fresh] == "fresh-native"
    assert why[a] == why[b] == "fresh-native-relation"      # every relation kind, not only FAMILY
    assert why[c] == "shared-release"                         # any release shared with a fresh title
    assert S.FRESH_NATIVE == set("v14909 v25459 v29454 v21289 v21976 v20519 v16509 v19140".split())
    assert S.FRESH_NATIVE <= S.fresh_native_vns()


def test_fresh_native_hashes_and_rgb_hash(tmp_path):
    from PIL import Image
    hs = S.fresh_native_hashes()
    assert len(hs) > 100 and all(len(h) == 64 for h in hs)
    a = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    Image.fromarray(a).save(tmp_path / "a.png")
    import hashlib
    assert S.rgb_sha256(tmp_path / "a.png") == hashlib.sha256(a.tobytes()).hexdigest()
    assert S.pixel_sha256(tmp_path / "a.png") == hashlib.sha256(b"4x5x3" + a.tobytes()).hexdigest()


def test_dedupe_prefers_read_screenshots():
    rows = [{"image": 1, "vn": "v1", "order": 0, "pixel_sha256": "a", "sha256": "f1"},
            {"image": 2, "vn": "v2", "order": 1, "pixel_sha256": "a", "sha256": "f2"},
            {"image": 3, "vn": "v3", "order": 2, "pixel_sha256": "b", "sha256": "f3"},
            {"image": 4, "vn": "v4", "order": 3, "pixel_sha256": "c", "sha256": "f4", "rgb_sha256": "r4"},
            {"image": 5, "vn": "v5", "order": 4}]
    assert S.dedupe(rows, {"b": {"realdata"}}, {"r4"}) == {"2": "realplus:1", "3": "pool:realdata",
                                                           "4": "pool:fresh-native"}
    assert S.dedupe(rows, {}, set(), read={"v2_2"}) == {"1": "realplus:2"}
