#!/usr/bin/env python3
"""The owner's own PS Vita games (STEINS;GATE v2002, planetarian v34): test-only titles.

Owner, 2026-10-13 09:25Z: the owner supplies both games; their frames are a fresh native console test
set and on-console overlay test material only. Never train, tune, calibrate or render fonts from
them, and never send their frames to Codex. "Exclude both, and their VNDB relations and shared
releases, from your harvest and from any font or render source."

  owner_games.py manifest --dump DB_DIR [--chars vndb_chars.tsv] [--pools DIR] --out JSON
        data/manifests/synth3_exclude_owner_games.json: a synth3 exclusion manifest (synth3.py
        load_exclusions) with every VN of realplus/shots.owner_games_vns() (the two games, their VNDB
        relations of any kind in either direction, every VN on a release with one of them or with
        such a relation), the relations themselves, the name hashes of every VNDB character of those
        VNs (synth3_text.py's name rows), text markers (titles and character full names) and the
        file / decoded-RGB sha256 of every local copy of a screenshot of those VNs, plus every file
        the owner dropped under shared/private/owner-games/.
  owner_games.py exposure --manifest JSON --out JSON
        counts (no text, no pixels) of rows of those VNs in every existing training, development
        and evaluation pool, and of synth3 scene candidates, name rows and text-pool lines the
        manifest's names and markers hit: what existing data already holds of these titles.

The manifest's VN list is recomputed from today's dump by realplus/shots.reserved_vns, which
reserves the same VNs (reasons owner-game, owner-game-relation, owner-game-shared-release).
"""
import argparse
import collections
import hashlib
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (HERE, ROOT / "realplus"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import shots as SH  # noqa: E402

HERD = Path(os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr"))
ART = HERD / "shared" / "artifacts"
PRIVATE = HERD / "shared" / "private" / "owner-games"
S3 = ART / "synth3"
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
JP = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
# Character names shorter than this (joined parts) are no text markers: 2-character names such as
# 綯 or ダル occur inside unrelated words; their full names still drop name rows by hash.
MIN_MARKER = 3


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rgb_sha256(path):
    """synth3.pixel_sha256: sha256 of PIL's decoded RGB bytes (eval/fresh_native_textbox.py's)."""
    import numpy as np
    from PIL import Image
    with Image.open(path) as im:
        a = np.ascontiguousarray(np.asarray(im.convert("RGB"), dtype=np.uint8))
    return hashlib.sha256(a.tobytes()).hexdigest()


def norm_vn(s):
    """synth3_text.norm_vn, imported lazily (it loads the vocabulary module)."""
    from synth3_text import norm_vn as f
    return f(s)


def name_hashes(name):
    """The exclusion manifests' name keys for one VNDB name (synth3_exclude_fresh_native.json's
    definition): norm_vn() of each space-separated part, the first two parts joined, and all parts
    joined when there are more."""
    parts = [norm_vn(x) for x in (name or "").split(" ") if x]
    if not parts:
        return set()
    out = {hashlib.sha256("".join(parts[:2]).encode("utf-8")).hexdigest()}
    if len(parts) > 2:
        out.add(hashlib.sha256("".join(parts).encode("utf-8")).hexdigest())
    return out


def relation_rows(base):
    """[(vn, other, kind, official)] from the dump's vn_relations table."""
    out = []
    for r in SH.rows("vn_relations", base):
        out.append((r[0], r[1], r[2], (r[3] if len(r) > 3 else "") in ("t", "1")))
    return out


def characters(chars_tsv, vns):
    """[(char_id, lang, name, vn_ids)] of every character of any VN in `vns`, any language."""
    out = []
    with open(chars_tsv, encoding="utf-8") as f:
        col = {k: i for i, k in enumerate(f.readline().rstrip("\n").split("\t"))}
        for line in f:
            p = line.rstrip("\n").split("\t")
            cv = set(p[col["vn_ids"]].split(","))
            if cv & vns:
                out.append((p[col["char_id"]], p[col["lang"]], p[col["name"]], sorted(cv & vns)))
    return out


def marker_candidates(titles, chars, roots_chars=()):
    """(always, if_rare): titles (>= 5 Latin or >= 3 Japanese characters) and every character's full
    name of two or more parts (norm_vn of the parts joined, >= MIN_MARKER characters, Japanese
    script) are always markers. A one-part name, and the given name of a character of the owner's
    games themselves (`roots_chars`), is a marker only if it is rare in the text pools (if_rare):
    ボタン, ポテト or 謎の少女 are VNDB character names too, and common words."""
    always, rare = set(), set()
    for t in titles:
        t = unicodedata.normalize("NFC", t or "").strip()
        if not t:
            continue
        if JP.search(t) and len(t) >= 3 or not JP.search(t) and len(t) >= 5:
            always.add(t)
            full = unicodedata.normalize("NFKC", t)
            if full != t and len(full) >= 3:
                always.add(full)
    for cid, _, name, _ in chars:
        parts = [norm_vn(x) for x in (name or "").split(" ") if x]
        s = "".join(parts)
        if len(s) < MIN_MARKER or not JP.search(s):
            continue
        (always if len(parts) >= 2 else rare).add(s)
        if cid in roots_chars and len(parts) >= 2 and len(parts[-1]) >= 2 and JP.search(parts[-1]):
            rare.add(parts[-1])
    return always, rare - always


MAX_RARE_HITS = 50       # a one-part name hitting more pool lines than this is a common word


def markers_of(titles, chars, roots_chars=(), count=None):
    """The manifest's text markers and the candidates left out as common words. count(strings) ->
    {string: lines} over the text pools; without it only the always-markers are used."""
    always, rare = marker_candidates(titles, chars, roots_chars)
    if count is None:
        return sorted(always), sorted(rare)
    hits = count(sorted(rare))
    keep = {m for m in rare if hits.get(m, 0) <= MAX_RARE_HITS}
    return sorted(always | keep), sorted(rare - keep)


def pool_counter(pools_dir):
    """count(strings) -> {string: lines holding it} over every *.txt text pool in pools_dir."""
    def count(strings):
        total = collections.Counter()
        for p in sorted(Path(pools_dir).glob("*.txt")):
            total += marker_hits(p, strings)[1]
        return dict(total)
    return count


def local_screenshot_dirs():
    """Local copies of VNDB screenshots any pool or synth3 scene source may use."""
    dirs = [d for _, d in SH.screenshot_dirs()]
    dirs += [ART / "realplus" / "shots", ART / "eval", ART / "native-frames" / "shots"]
    return [d for d in dict.fromkeys(dirs) if d.exists()]


SHOT_VN = re.compile(r"^(v\d+)_\d+\.(?:jpg|jpeg|png)$", re.I)


def owner_files(vns, dirs=None, private=PRIVATE):
    """Image files of these VNs in the local screenshot dirs (by the <vn>_<id> file name) and every
    image the owner dropped under shared/private/owner-games/."""
    found = []
    for d in dirs if dirs is not None else local_screenshot_dirs():
        for p in sorted(Path(d).rglob("*")):
            m = SHOT_VN.match(p.name)
            if m and m.group(1) in vns and p.is_file():
                found.append(p)
    if private and Path(private).exists():
        found += sorted(p for p in Path(private).rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXT)
    return list(dict.fromkeys(found))


def build_manifest(d, base, chars_tsv=None, dirs=None, private=PRIVATE, dump_info=None, pools=None):
    owner = SH.owner_games_vns(d)
    vns = set(owner)
    rels = relation_rows(base)
    title = {v: (d["title"].get(v) or ["?"])[0] for v in vns}
    relations = collections.defaultdict(list)
    for a, b, kind, official in sorted(rels):
        if a in SH.OWNER_GAMES:
            relations[a].append({"id": b, "relation": kind, "official": official, "title": title.get(b, "?")})
    sharing = collections.defaultdict(set)
    for rid, rv in sorted(d["rel_vns"].items()):
        for v in rv & vns:
            if owner.get(v) in ("owner-game", "owner-game-relation"):
                sharing[v] |= rv - {v}
    chars = characters(chars_tsv, vns) if chars_tsv and Path(chars_tsv).exists() else []
    hashes = set()
    for _, _, name, _ in chars:
        hashes |= name_hashes(name)
    titles = [t for v in sorted(vns) for t in d["title"].get(v, [])]
    roots_chars = {c[0] for c in chars if set(c[3]) & set(SH.OWNER_GAMES)}
    count = pool_counter(pools) if pools and Path(pools).exists() else None
    markers, common = markers_of(titles, chars, roots_chars, count)
    files = owner_files(vns, dirs, private)
    shot = {"sha256": set(), "pixel_sha256": set()}
    by_src = collections.Counter()
    for p in files:
        try:
            shot["sha256"].add(sha256_file(p))
            shot["pixel_sha256"].add(rgb_sha256(p))
        except OSError:
            continue
        by_src["owner-private" if private and str(p).startswith(str(private)) else p.parent.name] += 1
    doc = {
        "kind": "synth3-exclusion", "schema": 1,
        "name": "the owner's own PS Vita games (STEINS;GATE v2002, planetarian v34): test only",
        "rule": ("Owner 2026-10-13 09:25Z: never train, tune, calibrate, select or render fonts from these titles; "
                 "their frames are a fresh native console test set and on-console overlay test material only, never "
                 "sent to Codex. 'vns' = the two games, every direct VNDB relation of any kind in either direction, "
                 "and every VN sharing a VNDB release with one of them or with such a relation "
                 "(realplus/shots.owner_games_vns). synth3 never uses their names, scenes or text markers."),
        "protected": {"vita": sorted(SH.OWNER_GAMES)},
        "source": dump_info or {},
        "relations": {k: v for k, v in sorted(relations.items())},
        "vns_sharing_a_release": {k: sorted(v) for k, v in sorted(sharing.items())},
        "reasons": {k: owner[k] for k in sorted(owner, key=lambda v: int(v[1:]))},
        "vns": sorted(vns, key=lambda v: int(v[1:])),
        "titles": {v: title[v] for v in sorted(vns, key=lambda v: int(v[1:]))},
        "screenshots": {"sources": len(files), "by_source": dict(sorted(by_src.items())),
                        "sha256": sorted(shot["sha256"]), "pixel_sha256": sorted(shot["pixel_sha256"]),
                        "note": "local copies of VNDB screenshots of these VNs (by file name) and every file under "
                                "shared/private/owner-games/ when the manifest was built; VN ids exclude every "
                                "VNDB screenshot of them anyway"},
        "names": {"source": "VNDB characters of any VN in 'vns' (synth3 text/vndb_chars.tsv), any language",
                  "sha256_of_tsv": sha256_file(chars_tsv) if chars and chars_tsv else None,
                  "characters": len(chars), "definition": "as in synth3_exclude_fresh_native.json",
                  "name_sha256": sorted(hashes)},
        "text_markers": {"strings": markers,
                         "rule": ("a text-pool line (or focus context) holding any of these strings is never used: "
                                  "every title of these VNs (>= 5 Latin or >= 3 Japanese characters), every "
                                  f"character's full name of two or more parts (>= {MIN_MARKER} characters), and "
                                  "one-part names and the given names of the two games' own characters when they "
                                  f"hit at most {MAX_RARE_HITS} text-pool lines (more: a common word, listed under "
                                  "'common_words_not_markers'). Lines that name no character or title cannot be "
                                  "identified and remain."),
                         "pools_counted": bool(count),
                         "common_words_not_markers": common},
    }
    return doc


# ----------------------------------------------------------------------------- exposure

POOLS = ("teacher/teacher_train.npz", "teacher/teacher_val.npz", "teacher/teacher_distill_train.npz",
         "teacher/teacher_distill_val.npz", "realdata/real_train.npz", "realdata/real_val.npz",
         "realdata/real_unlabelled.npz", "realplus/realplus_train_ink.npz", "realplus/realplus_train_grad.npz",
         "realplus/realplus_train_ink_v2.npz", "realplus/realplus_train_grad_v2.npz",
         "devnative/devnative_tv_native_ink.npz", "devnative/devnative_strict_ink.npz",
         "devnative/v3/devnative_tv_native_ink.npz", "devnative/v3/devnative_tv_native_grad.npz",
         "devnative/v3/devnative_strict_ink.npz", "devnative/v3/devnative_strict_grad.npz")


def pool_rows(path, vns):
    import numpy as np
    z = np.load(path)
    meta = json.loads(str(z["meta"]))
    return dict(collections.Counter(m.get("vn") for m in meta if m.get("vn") in vns)), len(meta)


def marker_hits(path, markers):
    """{marker: lines} of one text pool, by plain substring search over the whole file."""
    import bisect
    text = Path(path).read_text(encoding="utf-8")
    starts, i = [0], text.find("\n")
    while i >= 0:
        starts.append(i + 1)
        i = text.find("\n", i + 1)
    lines = set()
    per = collections.Counter()
    for m in markers:
        j = text.find(m)
        while j >= 0:
            ln = bisect.bisect_right(starts, j) - 1
            if (m, ln) not in lines:
                lines.add((m, ln))
                per[m] += 1
            j = text.find(m, j + 1)
    return len({ln for _, ln in lines}), per


def exposure(doc, pools_dir=S3 / "text" / "pools3", text=True):
    vns = set(doc["vns"])
    out = {"kind": "owner-games-exposure", "schema": 1, "vns": len(vns), "pools": {}, "scenes": {},
           "names": {}, "text_pools": {}}
    for rel in POOLS:
        p = ART / rel
        if p.exists():
            rows, n = pool_rows(p, vns)
            out["pools"][rel] = {"rows": n, "owner_rows": sum(rows.values()), "by_vn": rows}
    bg = ART / "synth2" / "backgrounds.json"
    if bg.exists():
        b = json.loads(bg.read_text())
        for k in ("vita_shots", "psp_shots"):
            hit = collections.Counter(s.split("_")[0] for s in b.get(k, []) if s.split("_")[0] in vns)
            out["scenes"][f"synth2 backgrounds.json {k}"] = {"files": len(b.get(k, [])), "owner_files": dict(hit)}
    for kind in ("vita", "psp", "hires"):
        d = ART / "realdata" / "shots" / kind
        if d.exists():
            hit = collections.Counter(p.name.split("_")[0] for p in d.glob("*.jpg") if p.name.split("_")[0] in vns)
            out["scenes"][f"realdata/shots/{kind} (synth3 scene candidates)"] = {"owner_files": dict(hit)}
    d = HERD / "shared" / "data" / "vndb" / "shots"
    if d.exists():
        hit = collections.Counter(p.name.split("_")[0] for p in d.glob("*.jpg") if p.name.split("_")[0] in vns)
        out["scenes"]["shared/data/vndb/shots (synth2 / synth3 scene candidates)"] = {"owner_files": dict(hit)}
    bad = set(doc["names"]["name_sha256"])
    for split in ("train", "dev"):
        p = pools_dir / f"names.{split}.tsv"
        if p.exists():
            n = hit = 0
            for line in p.read_text(encoding="utf-8").splitlines():
                parts = line.split("\t")
                n += 1
                hit += hashlib.sha256("".join(parts[2:]).encode("utf-8")).hexdigest() in bad
            out["names"][f"names.{split}.tsv"] = {"rows": n, "owner_name_rows": hit}
    if text and pools_dir.exists():
        markers = doc["text_markers"]["strings"]
        total = collections.Counter()
        for p in sorted(pools_dir.glob("*.txt")):
            n, per = marker_hits(p, markers)
            out["text_pools"][p.name] = n
            total += per
        out["text_markers_top"] = dict(total.most_common(25))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("manifest")
    p.add_argument("--dump", type=Path, required=True, help="the VNDB dump's db/ directory")
    p.add_argument("--chars", type=Path, default=S3 / "text" / "vndb_chars.tsv")
    p.add_argument("--pools", type=Path, default=S3 / "text" / "pools3",
                   help="synth3's text pools, to tell rare one-part names from common words")
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("exposure")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--pools", type=Path, default=S3 / "text" / "pools3")
    p.add_argument("--no-text", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.cmd == "manifest":
        d = SH.load_dump(a.dump)
        stamp = a.dump.parent / "TIMESTAMP"
        info = {"vndb_dump": stamp.read_text().strip() if stamp.exists() else None}
        doc = build_manifest(d, a.dump, a.chars, dump_info=info, pools=a.pools)
        a.out.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"{len(doc['vns'])} VNs, {doc['names']['characters']} characters, {len(doc['names']['name_sha256'])} "
              f"name hashes, {len(doc['text_markers']['strings'])} markers, {doc['screenshots']['sources']} "
              f"screenshot files -> {a.out}", flush=True)
    else:
        doc = json.loads(a.manifest.read_text(encoding="utf-8"))
        out = exposure(doc, a.pools, text=not a.no_text)
        a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(json.dumps({k: out[k] for k in ("pools", "names", "text_pools")}, ensure_ascii=False)[:3000], flush=True)


if __name__ == "__main__":
    main()
