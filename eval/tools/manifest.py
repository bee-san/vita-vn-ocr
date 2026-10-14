#!/usr/bin/env python3
"""Write the committed manifests from the hand-check decisions (build tool).

  manifest.py vita|psp

Reads data/eval/<platform>/review.jsonl (decisions) and shots.jsonl (URLs and
hashes) and writes eval/manifests/<platform>-test.jsonl plus the games table
eval/manifests/test_games.json. Lines decided "ok" are scored; "uncertain" ones
are kept as don't-care. Regions are the union of a screenshot's lines that sit
together (vertical gap under 1.6 line heights), plus a margin (Vita 12 px, PSP
6 px), unless the review overrides them.

Each line records its game (vn, the VNDB VN id) and its capture session
(session, the VNDB release the screenshot was uploaded under). A release is the
unit of capture: VNDB links every screenshot to exactly one release, and
screenshots taken in one sitting are uploaded together under one release. It is
the conservative choice, since one release can hold several sittings.

test_games.json's "all" reserves (owner 2026-10-14 13:25Z, docs/reservations.md) the set games and
every VN that shares a VNDB release with one: the VNs on a set screenshot's release and the release
partners eval/manifests/reserved.json records ("test_shared_releases", from the VNDB dump by
realplus/shots.py reserve). "family" lists each set game's direct VNDB relations of the kinds in
FAMILY for information; they are training data. Until 2026-10-14 "all" also held every is_test_vn()
game with Vita or PSP screenshots and that family; "all_20261012" keeps that list as history.
Relations come from tools/relations.py.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent
sys.path.insert(0, str(EVAL))
from fetch import data_dir  # noqa: E402

MARGIN = {"vita": 12, "psp": 6}
FAMILY = {"alt", "orig", "fan", "par", "side", "seq", "preq", "ser"}  # VNDB relation kinds reserved
SMALL_PX = {"vita": 16, "psp": 16}


def load(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def groups(lines):
    lines = sorted(lines, key=lambda r: (r["box"][1], r["box"][0]))
    out = []
    for r in lines:
        x, y, w, h = r["box"]
        if out:
            g = out[-1]
            bottom = max(q["box"][1] + q["box"][3] for q in g)
            hmax = max(q["box"][3] for q in g + [r])
            if y - bottom <= 1.6 * hmax:
                g.append(r)
                continue
        out.append([r])
    return out


def auto_styles(r, which):
    tags = set()
    hint = r.get("hint") or {}
    bf = hint.get("bright_frac")
    if bf is not None:
        tags.add("bright-on-dark" if bf >= 0.5 else "dark-on-light")
    auto_furi = bool(r.get("furigana")) and not r.get("box_override")
    if auto_furi:
        tags.add("furigana")
    main_h = r["box"][3]
    if auto_furi:
        main_h = r["box"][1] + r["box"][3] - max(f[3] for f in r["furigana"])
    if main_h < SMALL_PX[which]:
        tags.add("small")
    tags |= set(r.get("tags_add") or [])
    tags -= set(r.get("tags_del") or [])
    if "coloured" in tags and "name" in tags:
        tags.add("coloured-name")
    return sorted(tags)


def main(which):
    data = data_dir()
    rows = load(data / which / "review.jsonl")
    shots = {s["shot"]: s for s in load(data / which / "shots.jsonl")}
    kept = [r for r in rows if r.get("status") in ("ok", "uncertain")]
    for r in kept:
        if r.get("box_override"):
            r["box"] = r["box_override"]
    by_shot = {}
    for r in kept:
        by_shot.setdefault(r["shot"], []).append(r)
    out = []
    for shot in sorted(by_shot, key=lambda s: (int(s.split("_")[0][1:]), s)):
        s = shots[shot]
        W, H = s["dims"]
        for g in groups(by_shot[shot]):
            ov = next((r["region_override"] for r in g if r.get("region_override")), None)
            if ov:
                region = ov
            else:
                m = MARGIN[which]
                x0 = max(0, min(r["box"][0] for r in g) - m)
                y0 = max(0, min(r["box"][1] for r in g) - m)
                x1 = min(W, max(r["box"][0] + r["box"][2] for r in g) + m)
                y1 = min(H, max(r["box"][1] + r["box"][3] for r in g) + m)
                region = [x0, y0, x1 - x0, y1 - y0]
            for k, r in enumerate(g):
                row = {"id": f"{which}-{Path(shot).stem}-{r['box'][1]:03d}-{r['box'][0]:03d}",
                       "vn": s["vn"], "title": s["title"], "url": s["url"], "sha256": s["sha256"],
                       "session": s["release"], "dims": s["dims"], "region": region, "box": r["box"],
                       "text": r["text"] or "",
                       "styles": auto_styles(r, which), "uncertain": r["status"] == "uncertain"}
                if r.get("note"):
                    row["note"] = r["note"]
                out.append(row)
    ids = [r["id"] for r in out]
    assert len(ids) == len(set(ids)), "duplicate line ids"
    path = EVAL / "manifests" / f"{which}-test.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    scored = [r for r in out if not r["uncertain"]]
    games = {}
    for r in scored:
        g = games.setdefault(r["vn"], {"title": r["title"], "lines": 0, "shots": set(), "sessions": set()})
        g["lines"] += 1
        g["shots"].add(r["url"])
        g["sessions"].add(r["session"])
    print(f"{path}: {len(scored)} scored lines ({sum(len(r['text']) for r in scored)} chars), "
          f"{len(out) - len(scored)} uncertain, {len(games)} VNs, "
          f"{len({r['url'] for r in out})} screenshots")
    tg_path = EVAL / "manifests" / "test_games.json"
    tg = json.loads(tg_path.read_text(encoding="utf-8")) if tg_path.exists() else {}
    tg[f"{which}-test"] = {vn: {"title": g["title"], "lines": g["lines"], "screenshots": len(g["shots"]),
                                "sessions": sorted(g["sessions"], key=lambda x: int(x[1:]))}
                           for vn, g in sorted(games.items(), key=lambda kv: int(kv[0][1:]))}
    # Every VN reserved for eval: the games in the sets and every VN sharing a release with one.
    listing = []
    for plat in ("psv", "psp"):
        for name in (f"vndb_{plat}.jsonl", f"vndb_{plat}_any.jsonl"):
            p = data / "vndb" / name
            if p.exists():
                listing += [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    used = set(tg.get("vita-test", {})) | set(tg.get("psp-test", {}))
    sessions = {s for k in ("vita-test", "psp-test") for g in tg.get(k, {}).values() for s in g["sessions"]}
    # A release (session) can belong to several VNs (compilations): reserve all of them.
    sharing = {x["vn"] for x in listing if x.get("release") in sessions}
    rel_path = data / "vndb" / "relations.json"
    if not rel_path.exists():
        sys.exit(f"{rel_path} missing: run eval/tools/relations.py first")
    relations = json.loads(rel_path.read_text(encoding="utf-8"))
    missing = sorted(used - set(relations))
    if missing:
        sys.exit(f"no relations for {missing}: run eval/tools/relations.py {' '.join(missing)}")
    family = {vn: sorted({x["id"] for x in relations[vn] if x["relation"] in FAMILY}, key=lambda v: int(v[1:]))
              for vn in sorted(used, key=lambda v: int(v[1:]))}
    reserved = EVAL / "manifests" / "reserved.json"
    partners = set(json.loads(reserved.read_text(encoding="utf-8")).get("test_shared_releases", {})) \
        if reserved.exists() else set()
    sys.path.insert(0, str(EVAL.parent / "realplus"))
    from shots import TEST_GAMES_RULE
    tg["rule"] = TEST_GAMES_RULE
    tg["all"] = sorted(used | sharing | partners, key=lambda v: int(v[1:]))
    tg["sessions"] = sorted(sessions, key=lambda x: int(x[1:]))
    tg["family"] = {vn: ids for vn, ids in family.items() if ids}
    keys = ("rule", "all", "sessions", "family", "vita-test", "psp-test", "all_20261012")
    tg = {k: tg[k] for k in keys if k in tg}
    tg_path.write_text(json.dumps(tg, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{tg_path}: {len(tg['all'])} reserved VN ids ({len(used)} in the sets, "
          f"{len((sharing | partners) - used)} more sharing a release), "
          f"{len(sessions)} reserved sessions")


if __name__ == "__main__":
    main(sys.argv[1])
