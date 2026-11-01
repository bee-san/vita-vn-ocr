#!/usr/bin/env python3
"""Audit of the hooked corpus import (data/import_hooked.py, docs/vocr6-data.md).

  hooked_audit.py sample --pools POOLS --out SAMPLE.jsonl [--n 320] [--seed 20261026]
      the label spot-check sample: lines from every title, image_rows lines, furigana lines, names, ruby and
      vertical columns oversampled (QUOTA), the rest random glyph_cells lines; drawn from the pools' kept rows
  hooked_audit.py sheets --sample SAMPLE.jsonl --pools POOLS --out DIR [--per 20]
      contact sheets (<= 1280 px a side, one PNG per 20 lines): each crop in screen polarity, scaled to 28 px tall
      (2x for small text), its label under it in BIZ UDGothic, and its number; DIR/sheets.json maps numbers to ids
  hooked_audit.py tally --sheets DIR --decisions DECISIONS.jsonl --out AUDIT.json
      the reviewer's verdicts (one JSON line per sheet line: {"sheet", "n", "verdict": ok|wrong|unsure, "note"})
      as counts per stratum, with a 95% interval on the wrong-label rate (Wilson)
  hooked_audit.py stats --pools POOLS --pagepool PAGEPOOL --splits SPLITS.json --out STATS.json
      per title, platform, kind and source: lines, characters, label lengths, text heights, polarity, duplicates
      and filter drops; character and kanji coverage against data/vocab_vn_5631.txt's 5,142 kanji (present,
      rare and absent classes; characters of the data outside the vocab); counts only, no label text
  hooked_audit.py overlap --stats STATS.json --vndb-map MAP.json --out OVERLAP.json [--vocr5-pools PIN]
      each title's VNDB VN(s) against eval/manifests/reserved.json (test: vita-test, psp-test, the CLANNAD family and
      their shared releases; dev_sample), the vita-test / psp-test manifests, the owner VNs and the VNs of every
      VOCR5 fine-tune pool (rows per VN): the "seen" flags every VOCR6 report carries

Everything it reads is the owner's private game data; the sheets stay under $HERD (never committed, never sent to a
third party). STATS.json and OVERLAP.json hold counts, ids, characters and hashes only.
"""
import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "data", ROOT / "synth"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import import_hooked as IH  # noqa: E402

FONT = Path("/local/home/skerraut/herd-vita-ocr/shared/data/fonts/bizudgothic/BIZUDGothic-Regular.ttf")
QUOTA = {"image_rows": 100, "furigana": 40, "name": 40, "ruby": 20, "vertical": 20}
SHEET_W, ROW_H, CROP_H, TEXT_PX = 1280, 62, 28, 22


def is_han(c):
    return "\u4e00" <= c <= "\u9fff" or "\u3400" <= c <= "\u4dbf" or "\uf900" <= c <= "\ufaff"


def pool_files(pools, split, pool):
    return sorted((Path(pools) / split / pool).glob("*.npz"))


def kept_rows(pools, splits=("train", "guard", "hval", "test"), pools_kinds=("line", "name", "ruby", "vertical")):
    """[(split, pool, title, index, meta, label, crop)] of every kept row (hval_fold repeats hval: left out)."""
    out = []
    for split in splits:
        for kind in pools_kinds:
            for f in pool_files(pools, split, kind):
                for i, (m, lab, g) in enumerate(IH.read_npz(f)):
                    out.append((split, kind, f.stem, i, m, lab, g))
    return out


# ----------------------------------------------------------------------------- sample and sheets

def stratum(kind, m):
    if kind == "vertical":
        return "vertical"
    if kind == "ruby":
        return "ruby"
    if kind == "name":
        return "name"
    if m.get("furigana"):
        return "furigana"
    if m.get("source") == "image_rows":
        return "image_rows"
    return "glyph_cells"


def sample_group(r):
    """The round-robin group of a kept row: its title, and its capture run when it is a later one (page >= 100000)."""
    page = r[4].get("page") or "0"
    run = int(page) // IH.RUN_STRIDE if page.isdigit() else 0
    return f"{r[2]}~r{run}" if run else r[2]


def cmd_sample(a):
    rows = kept_rows(a.pools)
    quota = json.loads(a.quota) if a.quota else QUOTA
    if a.exclude_v1:                                # pools v2: only records that pools v1 did not hold
        old = set()
        for f in sorted((Path(a.exclude_v1) / "decisions").glob("*.jsonl")):
            old.update(r["id"] for r in IH.read_jsonl(f))
        n0 = len(rows)
        rows = [r for r in rows if r[4]["id"] not in old]
        print(f"{n0 - len(rows)} kept rows of pools v1's records left out; {len(rows)} new rows")
    rng = np.random.default_rng(a.seed)
    by = defaultdict(list)
    for r in rows:
        by[stratum(r[1], r[4])].append(r)
    picked = []
    for s, q in quota.items():                      # oversampled strata: spread over titles round robin
        cand = by.get(s, [])
        per_title = defaultdict(list)
        for r in cand:
            per_title[sample_group(r)].append(r)
        order = {t: list(rng.permutation(len(v))) for t, v in per_title.items()}
        titles = sorted(per_title)
        while len([p for p in picked if p[0] == s]) < min(q, len(cand)):
            for t in titles:
                if order[t] and len([p for p in picked if p[0] == s]) < q:
                    picked.append((s, per_title[t][order[t].pop()]))
    rest = a.n - len(picked)
    gc = by["glyph_cells"]
    per_title = defaultdict(list)
    for r in gc:
        per_title[sample_group(r)].append(r)
    titles = sorted(per_title)
    order = {t: list(rng.permutation(len(v))) for t, v in per_title.items()}
    n0 = len(picked)
    while len(picked) - n0 < min(rest, len(gc)):    # at least one line of every title, then round robin
        for t in titles:
            if order[t] and len(picked) - n0 < rest:
                picked.append(("glyph_cells", per_title[t][order[t].pop()]))
    out = [{"stratum": s, "split": r[0], "pool": r[1], "title_id": r[2], "index": r[3], "id": r[4]["id"],
            "source": r[4].get("source"), "platform": r[4]["platform"]} for s, r in picked]
    IH.write_jsonl(a.out, out)
    print(json.dumps({"lines": len(out), "strata": Counter(o["stratum"] for o in out),
                      "titles": len({o["title_id"] for o in out})}, default=dict))


def render_label(text, font, width):
    from PIL import Image, ImageDraw
    img = Image.new("L", (width, TEXT_PX + 8), 255)
    ImageDraw.Draw(img).text((0, 2), text, font=font, fill=0)
    return np.asarray(img)


def cmd_sheets(a):
    import cv2
    from PIL import ImageFont
    font = ImageFont.truetype(str(FONT), TEXT_PX)
    small = ImageFont.truetype(str(FONT), 14)
    sample = IH.read_jsonl(a.sample)
    cache = {}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    index = {}
    for s0 in range(0, len(sample), a.per):
        chunk = sample[s0:s0 + a.per]
        sheet_no = s0 // a.per + 1
        canvas = np.full((ROW_H * len(chunk) + 4, SHEET_W), 255, np.uint8)
        for k, e in enumerate(chunk):
            f = Path(a.pools) / e["split"] / e["pool"] / f"{e['title_id']}.npz"
            if f not in cache:
                cache.clear()
                cache[f] = IH.read_npz(f)
            m, lab, g = cache[f][e["index"]]
            assert m["id"] == e["id"]
            shown = "as stored"
            if e["pool"] == "vertical":                            # a column upright: its re-flowed copy (vreflow)
                rf = Path(a.pools) / e["split"] / "vreflow" / f"{e['title_id']}.npz"
                hit = [x for x in (IH.read_npz(rf) if rf.exists() else []) if x[0]["id"] == e["id"]]
                if hit and hit[0][1] == lab:
                    m, lab, g = hit[0]
                    shown = "vreflow"
            view = 255 - g if m.get("inverted") else g             # as on screen
            h, w = view.shape
            scale = CROP_H / h                                     # every crop 28 px tall (small text enlarged)
            nw = max(1, min(SHEET_W - 60, int(round(w * scale))))
            crop = cv2.resize(view, (nw, CROP_H), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
            y = 4 + k * ROW_H
            canvas[y:y + CROP_H, 56:56 + nw] = crop
            lab_img = render_label(lab, font, SHEET_W - 56)
            canvas[y + CROP_H:y + CROP_H + lab_img.shape[0], 56:] = lab_img[:, :SHEET_W - 56]
            num = render_label(f"{k + 1}", small, 50)
            canvas[y + 6:y + 6 + num.shape[0], 2:52] = num
            index[f"S{sheet_no:02d}:{k + 1}"] = {**e, "label_chars": len(lab), "crop_shape": [h, w], "shown": shown}
        cv2.imwrite(str(out / f"S{sheet_no:02d}.png"), canvas)
    IH.write_json(out / "sheets.json", index)
    print(f"{(len(sample) + a.per - 1) // a.per} sheets -> {out}")


def wilson(k, n, z=1.96):
    if not n:
        return None
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round(max(0.0, (c - r) / d), 5), round(min(1.0, (c + r) / d), 5)]


def cmd_tally(a):
    index = json.loads((Path(a.sheets) / "sheets.json").read_text(encoding="utf-8"))
    dec = {f"{d['sheet']}:{d['n']}": d for d in IH.read_jsonl(a.decisions)}
    missing = sorted(set(index) - set(dec))
    if missing:
        raise SystemExit(f"{len(missing)} sheet lines without a verdict, e.g. {missing[:5]}")
    by = defaultdict(Counter)
    notes = []
    for k, e in index.items():
        v = dec[k]["verdict"]
        for s in ("all", e["stratum"], f"platform:{e['platform']}", f"title:{e['title_id']}"):
            by[s][v] += 1
        if v != "ok":
            notes.append({"line": k, "id": e["id"], "stratum": e["stratum"], "verdict": v,
                          "note": dec[k].get("note", "")})
    res = {"kind": "vocr6-hooked-label-audit", "reviewer_kind": "assistant", "lines": len(index),
           "strata": {s: dict(c) for s, c in sorted(by.items())},
           "wrong_rate_ci95": {s: wilson(c.get("wrong", 0), sum(c.values())) for s, c in sorted(by.items())
                               if not s.startswith("title:")},
           "not_ok": notes}
    IH.write_json(a.out, res)
    print(json.dumps({s: dict(c) for s, c in by.items() if not s.startswith("title:")}))


# ----------------------------------------------------------------------------- stats

def quantiles(xs, qs=(0.05, 0.5, 0.95)):
    if not xs:
        return None
    return [float(np.quantile(xs, q)) for q in qs]


def cmd_stats(a):
    vocab = IH.load_vocab_set(IH.VOCAB)
    vocab_s = IH.load_vocab_set(IH.VOCAB_S)
    vk = {c for c in vocab if is_han(c)}
    man = json.loads((Path(a.pools) / "manifest.json").read_text(encoding="utf-8"))
    splits = json.loads(Path(a.splits).read_text(encoding="utf-8"))
    titles = {}
    chars_train, chars_all = Counter(), Counter()
    plat = defaultdict(Counter)
    for split in ("train", "guard", "hval", "hval_fold", "test"):
        for kind in ("line", "name", "ruby", "vertical"):
            for f in pool_files(a.pools, split, kind):
                tid = f.stem
                t = titles.setdefault(tid, {"rows": Counter(), "chars": Counter(), "label_len": [], "height": [],
                                            "dark": 0, "dialogue_rows": 0, "sources": Counter(),
                                            "furigana": 0, "s_vocab_rows": 0})
                for m, lab, g in IH.read_npz(f):
                    t["rows"][f"{split}:{kind}"] += 1
                    t["chars"][f"{split}:{kind}"] += len(lab)
                    if split == "hval_fold":      # counted for the per-title totals only (it repeats hval rows)
                        continue
                    t["sources"][m["source"]] += 1
                    t["s_vocab_rows"] += bool(m.get("s_vocab"))
                    plat[m["platform"]][f"{split}:{kind}:rows"] += 1
                    plat[m["platform"]][f"{split}:{kind}:chars"] += len(lab)
                    if kind == "line":
                        t["dialogue_rows"] += 1
                        t["label_len"].append(len(lab))
                        t["height"].append(int(m["box"][3]))
                        t["dark"] += int(m["dark"])
                        t["furigana"] += bool(m.get("furigana"))
                    for c in lab:
                        chars_all[c] += 1
                        if split in ("train", "guard", "hval"):
                            chars_train[c] += 1
    out_titles = {}
    for tid, t in sorted(titles.items()):
        info = splits["titles"][tid]
        pc = man["titles"][tid]["counts"]
        out_titles[tid] = {
            "platform": info["platform"], "title_split": info["title_split"], "pages": info["pages"],
            "page_splits": info["counts"], "records": pc.get("records", 0),
            "rows": dict(sorted(t["rows"].items())), "chars": dict(sorted(t["chars"].items())),
            "sources": dict(t["sources"]), "label_len_q05_q50_q95": quantiles(t["label_len"]),
            "box_height_q05_q50_q95": quantiles(t["height"]),
            "dark_on_light_share": round(t["dark"] / t["dialogue_rows"], 4) if t["dialogue_rows"] else None,
            "furigana_lines": t["furigana"],
            "duplicates_removed": {k[4:]: v for k, v in pc.items() if k.startswith("dup:")},
            "filters": {k: v for k, v in pc.items() if k.endswith((":oov", ":ctc", ":geometry", ":fails_vocab",
                                                                   ":fails_ctc", ":fails_geometry"))},
            "s_vocab_share": round(t["s_vocab_rows"] / max(1, sum(v for k, v in t["rows"].items()
                                                                   if not k.startswith("hval_fold:"))), 4)}
    for tid in sorted(set(splits["titles"]) - set(titles)):
        out_titles[tid] = {"platform": splits["titles"][tid]["platform"], "pages": splits["titles"][tid]["pages"],
                           "title_split": splits["titles"][tid]["title_split"],
                           "page_splits": splits["titles"][tid]["counts"],
                           "records": 0, "rows": {}, "note": "no verified lines"}
    # coverage of the 5,142 vocab kanji by the training side (train + guard + hval: what a final model sees)
    present = {c for c in vk if chars_train[c] > 0}
    bands = Counter()
    for c in vk:
        n = chars_train[c]
        bands["0" if n == 0 else "1-9" if n < 10 else "10-79" if n < 80 else "80+"] += 1
    data_han = {c for c in chars_all if is_han(c)}
    oov_rows = Counter()
    for f in sorted((Path(a.pools) / "drop" / "oov").glob("*/*.npz")):
        for m, lab, g in IH.read_npz(f):
            for c in set(lab):
                if c not in vocab:
                    oov_rows[c] += 1
    stats = {
        "kind": "vocr6-hooked-stats", "pools_manifest_sha256": IH.sha256_file(Path(a.pools) / "manifest.json"),
        "splits_sha256": IH.sha256_file(a.splits),
        "pagepool_manifest_sha256": IH.sha256_file(Path(a.pagepool) / "manifest.json"),
        "titles": out_titles,
        "platforms": {p: dict(sorted(c.items())) for p, c in sorted(plat.items())},
        "vocab": {"file": "data/vocab_vn_5631.txt", "kanji": len(vk),
                  "kanji_in_training_side": len(present),
                  "kanji_training_occurrence_bands": dict(sorted(bands.items())),
                  "kanji_absent": "".join(sorted(vk - present)),
                  "kanji_rare_1_9": "".join(sorted(c for c in vk if 0 < chars_train[c] < 10)),
                  "distinct_chars_all_splits": len(chars_all), "distinct_kanji_all_splits": len(data_han),
                  "kanji_outside_vocab_in_kept_rows": "".join(sorted(data_han - vk)),
                  "oov_chars_dropped_lines": dict(sorted(oov_rows.items(), key=lambda kv: (-kv[1], kv[0]))),
                  "outside_small_vocab_in_kept_rows": "".join(sorted(c for c in chars_all if c not in vocab_s)),
                  "training_side_characters": sum(chars_train.values()),
                  "training_side_kanji_characters": sum(v for c, v in chars_train.items() if is_han(c))}}
    IH.write_json(a.out, stats)
    v = stats["vocab"]
    print(json.dumps({k: v[k] for k in ("kanji", "kanji_in_training_side", "kanji_training_occurrence_bands",
                                        "distinct_chars_all_splits", "distinct_kanji_all_splits",
                                        "training_side_characters")}, ensure_ascii=False))


# ----------------------------------------------------------------------------- overlap

def pool_vns(pin):
    """{VN: rows} over the VOCR5 fine-tune pool files a sha256 pin lists (their meta's vn field)."""
    out, files = Counter(), []
    for line in Path(pin).read_text().splitlines():
        if not line.strip():
            continue
        sha, path = line.split(None, 1)
        p = Path(path.strip())
        z = np.load(p)
        metas = json.loads(str(z["meta"]))
        out.update(m.get("vn") for m in metas)
        files.append({"file": p.name, "sha256": sha, "rows": len(metas)})
    return out, files


def shared_releases(dump, vns, protected):
    """{hooked VN: [[release, [protected VNs]]]} from a VNDB dump's releases_vn table: the releases a hooked VN shares
    with a protected one (vita-test, psp-test, CLANNAD v4)."""
    rel = defaultdict(set)
    with open(Path(dump) / "db" / "releases_vn", encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            rel[p[0]].add(p[1])
    out = defaultdict(list)
    for r, members in sorted(rel.items()):
        for v in sorted(members & vns):
            hit = sorted((members & protected) - {v})
            if hit:
                out[v].append([r, hit])
    return out


def vn_relations(dump, vns):
    """{VN: [[related VN, relation]]} from a VNDB dump's vn_relations table, for the given VNs."""
    out = defaultdict(list)
    with open(Path(dump) / "db" / "vn_relations", encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if p[0] in vns:
                out[p[0]].append([p[1], p[2]])
    return out


def read_vn_set(spec):
    """NAME=FILE (one VN id per line) -> (name, {VNs})."""
    name, path = spec.split("=", 1)
    return name, {x.strip() for x in Path(path).read_text().split() if x.strip()}


def cmd_overlap(a):
    res = json.loads((ROOT / "eval" / "manifests" / "reserved.json").read_text(encoding="utf-8"))
    vt = {json.loads(x)["vn"] for x in (ROOT / "eval" / "manifests" / "vita-test.jsonl").read_text().splitlines()
          if x.strip()}
    pt = {json.loads(x)["vn"] for x in (ROOT / "eval" / "manifests" / "psp-test.jsonl").read_text().splitlines()
          if x.strip()}
    vmap = json.loads(Path(a.vndb_map).read_text(encoding="utf-8"))["titles"]
    v5, v5files = pool_vns(a.vocr5_pools)
    stats = json.loads(Path(a.stats).read_text(encoding="utf-8"))
    all_vns = {v for m in vmap.values() for v in m["vns"]}
    shared = shared_releases(a.vndb_dump, all_vns, vt | pt | {"v4"}) if a.vndb_dump else {}
    # the hooked test titles (the titles index.json marks test): their VNs, and the VNs related to them
    test_vns = {v for t, info in stats["titles"].items() if info.get("title_split") == "test"
                for v in vmap[t]["vns"]} if getattr(a, "hooked_test", False) else set()
    test_shared = shared_releases(a.vndb_dump, all_vns, test_vns) if a.vndb_dump and test_vns else {}
    rel = vn_relations(a.vndb_dump, all_vns) if a.vndb_dump and getattr(a, "relations", False) else {}
    vn_sets = dict(read_vn_set(s) for s in (getattr(a, "vn_set", None) or []))
    v1 = json.loads(Path(a.v1_index).read_text(encoding="utf-8")) if getattr(a, "v1_index", "") else None
    v1_titles = set(v1["titles"]) if v1 else set()
    v1_vns = {v for t in v1_titles if t in vmap for v in vmap[t]["vns"]}
    out = {"kind": "vocr6-hooked-overlap", "reserved_sha256": IH.sha256_file(ROOT / "eval" / "manifests" /
                                                                             "reserved.json"),
           "vndb_dump": (Path(a.vndb_dump) / "TIMESTAMP").read_text().strip() if a.vndb_dump else None,
           "vocr5_pools": v5files, "titles": {}}
    if vn_sets:
        out["vn_sets"] = {k: {"vns": len(v), "file_sha256": IH.sha256_file(s.split("=", 1)[1])}
                          for (k, v), s in zip(vn_sets.items(), a.vn_set)}
    if v1:
        out["v1_index_sha256"] = IH.sha256_file(a.v1_index)
    if test_vns:
        out["hooked_test_vns"] = sorted(test_vns)
    for tid, info in sorted(stats["titles"].items()):
        m = vmap[tid]
        flags, vns = [], m["vns"]
        for v in vns:
            if v in vt:
                flags.append(f"{v}:vita-test")
            if v in pt:
                flags.append(f"{v}:psp-test")
            if v == "v4":
                flags.append(f"{v}:clannad")
            if v in res["test"]:
                flags.append(f"{v}:reserved-test:{res['test'][v]}")
            for r, hit in shared.get(v, []):
                flags.append(f"{v}:shares-{r}-with:{'+'.join(hit)}")
            if v in res["dev_sample"]:
                flags.append(f"{v}:dev_sample")
            if v in res.get("owner_vns", []):
                flags.append(f"{v}:owner_vns")
            for name, s in vn_sets.items():
                if v in s:
                    flags.append(f"{v}:{name}")
            if info.get("title_split") != "test":
                if v in test_vns:
                    flags.append(f"{v}:hooked-test-vn")
                for r, hit in test_shared.get(v, []):
                    flags.append(f"{v}:shares-{r}-with-hooked-test:{'+'.join(hit)}")
                for w, how in rel.get(v, []):
                    if w in test_vns:
                        flags.append(f"{v}:related-to-hooked-test:{how}:{w}")
        rows5 = {v: v5.get(v, 0) for v in vns}
        protected = any(":vita-test" in f or ":psp-test" in f or ":clannad" in f or ":reserved-test" in f or
                        (":shares-" in f and "-with:" in f) for f in flags)
        out["titles"][tid] = {"vns": vns, "vndb_source": m["source"], "platform": info["platform"],
                              "title_split": info.get("title_split"), "flags": flags,
                              "vocr5_pool_rows": rows5, "seen_by_vocr5": any(rows5.values()),
                              "seen_protected_family": protected,
                              "dev_sample": any(":dev_sample" in f for f in flags)}
        if vn_sets:
            out["titles"][tid]["vn_sets"] = sorted({f.split(":")[1] for f in flags if f.split(":")[1] in vn_sets})
        if test_vns:
            out["titles"][tid]["hooked_test_related"] = any("hooked-test" in f for f in flags)
        if v1:
            out["titles"][tid]["in_pools_v1"] = tid in v1_titles
            out["titles"][tid]["vn_in_pools_v1"] = bool(set(vns) & v1_vns)
    out["seen_protected_family"] = sorted(t for t, v in out["titles"].items() if v["seen_protected_family"])
    out["dev_sample_titles"] = sorted(t for t, v in out["titles"].items() if v["dev_sample"])
    out["seen_by_vocr5"] = sorted(t for t, v in out["titles"].items() if v["seen_by_vocr5"])
    for name in vn_sets:
        out[f"{name}_titles"] = sorted(t for t, v in out["titles"].items() if name in v.get("vn_sets", []))
    if test_vns:
        out["hooked_test_related_titles"] = sorted(t for t, v in out["titles"].items() if v["hooked_test_related"])
    IH.write_json(a.out, out)
    for tid, t in out["titles"].items():
        if t["flags"] or t["seen_by_vocr5"]:
            print(tid, t["vns"], t["flags"], t["vocr5_pool_rows"])


def _pct(v, nd=2):
    return "" if v is None else f"{v:.{nd}f}%"


def _ci(s, nd=2):
    if not s or s.get("cer") is None:
        return ""
    ci = s.get("ci95")
    return f"{s['cer']:.{nd}f}%" + (f" [{ci[0]:.{nd}f}, {ci[1]:.{nd}f}]" if ci else "")


def corpus_tables(index, stats, overlap, audit):
    """{section: markdown} of the corpus: splits, filters, titles, coverage, audit."""
    sec = {}
    pc = index["pool_counts"]
    sc = index["split_counts"]
    rows = Counter()
    for f, v in index["pool_files"].items():
        parts = f.split("/")
        rows[(parts[0], parts[1])] += v["rows"]
    out = ["| rows | train | guard | hval | hval_fold | test |", "|---|---:|---:|---:|---:|---:|",
           f"| pages | {sc.get('train', 0):,} | {sc.get('guard', 0):,} | {sc.get('hval', 0):,} | | "
           f"{sc.get('test', 0):,} |"]
    for pool in ("line", "name", "ruby", "vertical", "vreflow", "furimain", "line_x2", "name_x2"):
        cells = [rows.get((s, pool), 0) for s in ("train", "guard", "hval", "hval_fold", "test")]
        out.append(f"| {pool} | " + " | ".join(f"{c:,}" for c in cells) + " |")
    sec["splits"] = "\n".join(out)
    drops = {k: v for k, v in pc.items() if k.split(":")[-1] in ("oov", "geometry", "ctc") and
             k.split(":")[0] in ("line", "name", "ruby", "vertical")}
    dup = Counter()
    for k, v in pc.items():
        if k.startswith("dup:"):
            _, _, pool, kind = k.split(":")
            dup[(pool, kind)] += v
    out = ["| kind | records | label in doubt | outside the vocab | under 8 px | CTC-infeasible | duplicates "
           "(exact / near) | pixel conflicts | hval copies of a guard line | usable (all splits) |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for kind, rec_key, pool in (("dialogue line", "records:line", "line"), ("vertical column",
                                "records:line:vertical", "vertical"), ("speaker name", "records:name", "name"),
                                ("ruby reading", "records:ruby", "ruby")):
        usable = sum(rows.get((s, pool), 0) for s in ("train", "guard", "hval_fold", "test"))
        doubt = pc.get("record:label_doubt", 0) if pool == "line" else 0
        # an hval line whose duplicate cluster is represented in guard: in the screening set (one per cluster), in no
        # pool a final model reads (it reads the guard copy)
        screen_only = rows.get(("hval", pool), 0) - rows.get(("hval_fold", pool), 0)
        out.append(f"| {kind} | {pc.get(rec_key, 0):,} | {doubt} | {drops.get(f'{pool}:oov', 0)} | "
                   f"{drops.get(f'{pool}:geometry', 0)} | {drops.get(f'{pool}:ctc', 0)} | "
                   f"{dup.get((pool, 'exact'), 0):,} / {dup.get((pool, 'near'), 0):,} | "
                   f"{dup.get((pool, 'conflict'), 0)} | {screen_only} | {usable:,} |")
    sec["filters"] = "\n".join(out)
    out = ["| title | platform | split | VN | pages (hval) | records | usable dialogue lines | dialogue chars | "
           "duplicates removed | text height (px, median) | dark-on-light | image_rows share | seen |",
           "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for tid, t in sorted(stats["titles"].items(), key=lambda kv: (kv[1]["platform"], kv[0])):
        o = overlap["titles"][tid]
        r = t.get("rows", {})
        final = ("train:line", "guard:line", "hval_fold:line", "test:line")    # each duplicate cluster once
        usable = sum(r.get(k, 0) for k in final)
        ch = sum(t.get("chars", {}).get(k, 0) for k in final)
        dups = sum(v for k, v in (t.get("duplicates_removed") or {}).items() if not k.endswith(":conflict")
                   and k.split(":")[1] in ("line", "name", "ruby", "vertical"))
        src = t.get("sources", {})
        ir = src.get("image_rows", 0) / max(1, sum(src.values())) if src else 0
        seen = []
        if o["seen_protected_family"]:
            seen.append("protected family")
        if o["dev_sample"]:
            seen.append("dev sample")
        if o["seen_by_vocr5"]:
            seen.append(f"VOCR5 pools ({sum(o['vocr5_pool_rows'].values()):,} rows)")
        hv = t.get("page_splits", {}).get("hval", 0)
        h = t.get("box_height_q05_q50_q95")
        dl = t.get("dark_on_light_share")
        out.append(f"| {tid} | {t['platform']} | {index['titles'][tid]['title_split']} | {', '.join(o['vns'])} | "
                   f"{t['pages']:,} ({hv}) | {t.get('records', 0):,} | {usable:,} | {ch:,} | {dups:,} | "
                   f"{'' if not h else int(h[1])} | {'' if dl is None else _pct(100 * dl, 1)} | "
                   f"{100 * ir:.0f}% | {'; '.join(seen)} |")
    sec["titles"] = "\n".join(out)
    v = stats["vocab"]
    b = v["kanji_training_occurrence_bands"]
    sec["coverage"] = (
        f"Of the {v['kanji']:,} kanji of `data/vocab_vn_5631.txt`, {v['kanji_in_training_side']:,} occur on the "
        f"training side (train + guard + hval): {b.get('80+', 0):,} at least 80 times, {b.get('10-79', 0):,} 10-79 "
        f"times and {b.get('1-9', 0):,} 1-9 times. {b.get('0', 0):,} never occur. The kept rows hold "
        f"{v['distinct_chars_all_splits']:,} distinct characters ({v['distinct_kanji_all_splits']:,} kanji). "
        f"{len(v['oov_chars_dropped_lines'])} characters outside the vocab removed "
        f"{pc.get('line:oov', 0) + pc.get('ruby:oov', 0)} records; "
        f"{len(v['outside_small_vocab_in_kept_rows'])} kept characters are outside Small's 4,987-class vocab, "
        f"which trains them as `<unk>`.")
    out = ["| label audit stratum | lines | ok | wrong | wrong-label rate, 95% interval (Wilson) |",
           "|---|---:|---:|---:|---|"]
    for s in ("glyph_cells", "image_rows", "furigana", "name", "ruby", "vertical", "platform:vita", "platform:psp",
              "all"):
        c = audit["strata"].get(s, {})
        n = sum(c.values())
        ci = audit["wrong_rate_ci95"].get(s)
        out.append(f"| {s} | {n} | {c.get('ok', 0)} | {c.get('wrong', 0)} | "
                   f"{'' if ci is None else f'{100 * ci[0]:.1f}-{100 * ci[1]:.1f}%'} |")
    pf = audit.get("post_fix_check")
    if pf:
        out.append(f"| after the fixes (pools v1) | {pf['of']} | {pf['of'] - pf['wrong_after_fixes']} | "
                   f"{pf['wrong_after_fixes']} | {100 * pf['wrong_rate_ci95_wilson'][0]:.1f}-"
                   f"{100 * pf['wrong_rate_ci95_wilson'][1]:.1f}% |")
    sec["audit"] = "\n".join(out)
    return sec


def baseline_tables(res):
    """{section: markdown} of the baseline: every tier and decode; Normal's shipped path per title."""
    out = ["| tier | decode | hval clean, Vita | hval clean, PSP | hval clean, all | "
           "hval page GRAD+refine, console view (pooled; title macro) | the same, regions without a seen line | "
           "PSP page at native size | hval page textspot3, dialogue | textspot3 F1 | dev score [CI] | "
           "dev score without v12188, v18432 | test clean | test page GRAD+refine | test page textspot3 |",
           "|---|---|---|---|---|---|---|---|---|---:|---|---|---|---|---|"]
    hv, te, dv = (res["sets"].get(s, {}) for s in ("hval", "test", "dev"))
    for t in ("S", "N", "L"):
        for d in ("greedy", "rubi", "rubi_cascade"):
            h = hv.get("tiers", {}).get(t, {}).get("decodes", {}).get(d)
            x = te.get("tiers", {}).get(t, {}).get("decodes", {}).get(d)
            v = dv.get("dev", {}).get(t, {}).get(d)
            if not h:
                continue
            c, f, s = h["clean"]["line"], h["finder"], h["spot"]
            con = f["refine|console|base"]["all"]
            ex = (v or {}).get("without_v12188_v18432")
            row = [t, d, _ci(c.get("vita")), _ci(c.get("psp")), _ci(c.get("all")),
                   f"{_ci(con)}; {con.get('macro_cer', 0):.2f}%",
                   _ci(f.get("refine|console|base|unseen", {}).get("all")),
                   _ci(f.get("refine|x1|base", {}).get("psp")), _ci(s["dialogue"]["all"]),
                   f"{s['detection']['all']['f1']:.3f}",
                   "" if not v else f"{v['score']:.3f} [{v['score_ci95'][0]:.2f}, {v['score_ci95'][1]:.2f}]",
                   "" if not ex else f"{ex['score']:.3f} [{ex['score_ci95'][0]:.2f}, {ex['score_ci95'][1]:.2f}]",
                   "" if not x else _ci(x["clean"]["line"]["all"]),
                   "" if not x else _ci(x["finder"]["refine|console|base"]["all"]),
                   "" if not x else _ci(x["spot"]["dialogue"]["all"])]
            out.append("| " + " | ".join(row) + " |")
    sec = {"baseline": "\n".join(out)}
    h = hv.get("tiers", {}).get("N", {}).get("decodes", {}).get("rubi_cascade")
    if h:
        out = ["| title | hval clean dialogue CER [page CI] | page GRAD+refine, console view | "
               "page textspot3 dialogue | clean ref. chars | page-path pages |", "|---|---|---|---|---:|---:|"]
        c, f, s = h["clean"]["line"]["per_title"], h["finder"]["refine|console|base"]["per_title"], \
            h["spot"]["dialogue"]["per_title"]
        for tid in sorted(set(c) | set(f) | set(s)):
            out.append(f"| {tid} | {_ci(c.get(tid))} | {_ci(f.get(tid))} | {_ci(s.get(tid))} | "
                       f"{(c.get(tid) or {}).get('chars', '')} | {(f.get(tid) or {}).get('clusters', '')} |")
        sec["baseline_titles"] = "\n".join(out)
    return sec


MARK = re.compile(r"(<!-- generated:(\w+) -->\n)(.*?)(<!-- /generated:\2 -->)", re.S)


def splice(doc, sections):
    """The doc with every generated block's body replaced by its section; KeyError for an unknown section."""
    return MARK.sub(lambda m: m.group(1) + sections[m.group(2)] + "\n" + m.group(4), doc)


def table_sections(index, stats, overlap, audit, baseline=None):
    sec = corpus_tables(index, stats, overlap, audit)
    if baseline is not None:
        sec.update(baseline_tables(baseline))
    return sec


# ----------------------------------------------------------------------------- v2's tables (docs/vocr6-data-v2.md)

FINAL = ("train", "guard", "hval_fold", "test")      # each duplicate cluster once: what a final model reads (+ test)


def corpus_numbers(index, stats):
    """{platform (vita, psp, all): {quantity: n}} of one version's committed index and stats."""
    out = defaultdict(Counter)
    rows = defaultdict(Counter)
    for f, v in index["pool_files"].items():
        split, pool, name = f.split("/")[-3:] if f.startswith("drop/") else f.split("/")
        rows[name[:-4]][(split, pool)] += v["rows"]
    for tid, t in index["titles"].items():
        pc, st = t.get("pool_counts", {}), stats["titles"].get(tid, {})
        for plat in (t["platform"], "all"):
            o = out[plat]
            o["titles"] += 1
            o["runs"] += len(t.get("runs") or [None])
            o["pages"] += t["pages"]
            o["test pages"] += t["counts"].get("test", 0)
            for k, name in (("records:line", "dialogue lines"), ("records:line:vertical", "vertical columns"),
                            ("records:name", "speaker names"), ("records:ruby", "ruby readings"),
                            ("records:ruby:vertical", "ruby readings")):
                o[f"records: {name}"] += pc.get(k, 0)
            for pool, name in (("line", "dialogue lines"), ("vertical", "vertical columns"),
                               ("name", "speaker names"), ("ruby", "ruby readings"), ("line_x2", "PSP lines at 2x")):
                o[f"usable: {name}"] += sum(rows[tid].get((s, pool), 0) for s in FINAL)
            o["usable dialogue characters"] += sum((st.get("chars") or {}).get(f"{s}:line", 0) for s in FINAL)
            o["label in doubt (dropped)"] += pc.get("record:label_doubt", 0)
    return out


def compare_table(index1, stats1, index2, stats2):
    a, b = corpus_numbers(index1, stats1), corpus_numbers(index2, stats2)
    keys = ["titles", "runs", "pages", "test pages", "records: dialogue lines", "records: vertical columns",
            "records: speaker names", "records: ruby readings", "label in doubt (dropped)", "usable: dialogue lines",
            "usable dialogue characters", "usable: vertical columns", "usable: speaker names", "usable: ruby readings",
            "usable: PSP lines at 2x"]
    out = ["| | Vita, v1 | Vita, v2.1 | PSP, v1 | PSP, v2.1 | all, v1 | all, v2.1 | added |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for k in keys:
        cells = [a["vita"][k], b["vita"][k], a["psp"][k], b["psp"][k], a["all"][k], b["all"][k]]
        out.append(f"| {k} | " + " | ".join(f"{c:,}" for c in cells) + f" | {b['all'][k] - a['all'][k]:+,} |")
    return "\n".join(out)


def titles2_table(index1, index2, stats2, overlap2):
    rows1 = Counter()
    for f, v in index1["pool_files"].items():
        p = f.split("/")
        if p[0] in FINAL and p[1] == "line":
            rows1[p[2][:-4]] += v["rows"]
    out = ["| title | platform | split | VN | runs | pages v1 → v2.1 (hval) | usable dialogue lines v1 → v2.1 | "
           "dialogue chars | dropped as in doubt | seen / overlaps |",
           "|---|---|---|---|---:|---|---|---:|---:|---|"]
    for tid, t in sorted(index2["titles"].items(), key=lambda kv: (kv[1]["platform"], kv[0])):
        st, o = stats2["titles"][tid], overlap2["titles"][tid]
        r = st.get("rows", {})
        usable = sum(r.get(f"{s}:line", 0) for s in FINAL)
        chars = sum(st.get("chars", {}).get(f"{s}:line", 0) for s in FINAL)
        t1 = index1["titles"].get(tid)
        pages = f"{t1['pages']:,} → {t['pages']:,}" if t1 else f"new: {t['pages']:,}"
        lines = f"{rows1[tid]:,} → {usable:,}" if t1 else f"new: {usable:,}"
        flags = []
        if o["seen_protected_family"]:
            flags.append("protected family")
        for name, label in (("dev_units", "dev units"), ("confirm", "confirmation set")):
            if name in o.get("vn_sets", []):
                flags.append(label)
        if o.get("hooked_test_related"):
            flags.append("related to the test title")
        if o["seen_by_vocr5"]:
            flags.append(f"VOCR5 pools ({sum(o['vocr5_pool_rows'].values()):,} rows)")
        nruns = len(t.get("runs") or [1])
        out.append(f"| {tid} | {t['platform']} | {t['title_split']} | {', '.join(o['vns'])} | {nruns} "
                   f"| {pages} ({t['counts'].get('hval', 0)}) | {lines} | {chars:,} | "
                   f"{t['pool_counts'].get('record:label_doubt', 0)} | {'; '.join(flags)} |")
    return "\n".join(out)


RULES = (("undrawn_stripped", "undrawn code removed, kept"), ("undrawn_shifted", "undrawn code, shifted"),
         ("remapped", "remapped glyph"), ("doubled_cells", "doubled cells"), ("audited_wrong", "audited wrong"))


def rules_table(index2):
    out = ["| title | split | " + " | ".join(n for _, n in RULES) + " | dropped (label in doubt) | pages in doubt "
           "(no regions) |", "|---|---|" + "---:|" * (len(RULES) + 2)]
    tot = Counter()
    for tid, t in sorted(index2["titles"].items()):
        pc = t["pool_counts"]
        if not any(pc.get(f"label_rule:{k}") for k, _ in RULES):
            continue
        cells = [pc.get(f"label_rule:{k}", 0) for k, _ in RULES]
        tot.update({k: c for (k, _), c in zip(RULES, cells)})
        tot["drop"] += pc.get("record:label_doubt", 0)
        nore = sum(v for k, v in (t.get("pagepool_counts") or {}).items() if k.startswith("untrusted:"))
        tot["pages"] += nore
        out.append(f"| {tid} | {t['title_split']} | " + " | ".join(f"{c:,}" for c in cells) +
                   f" | {pc.get('record:label_doubt', 0):,} | {nore} |")
    out.append("| all | | " + " | ".join(f"{tot[k]:,}" for k, _ in RULES) + f" | {tot['drop']:,} | {tot['pages']:,} |")
    return "\n".join(out)


def _cell(g):
    if not g or g.get("cer") is None:
        return ""
    ci = g.get("ci95")
    return f"{g['cer']:.2f}%" + (f" [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "")


FINDER_NAMES = {"spot": "textspot8 spotter (Rubi's default)", "fix1": "Rubi's GRAD path (`rubi,fix=1`)",
                "oracle": "clean crops (each line's own box)", "oraclenf": "clean crops, line filter off",
                "spotlinespunct": "textspot8, filter lines+punct", "spotnone": "textspot8, line filter off"}


def baseline2_tables(base):
    sec = {}
    hv = base["sets"].get("hval_v21", {})
    out = ["| finder | tier | new titles, Vita | new titles, PSP | v1 titles, Vita | v1 titles, PSP | all hval |",
           "|---|---|---|---|---|---|---|"]
    for f in ("spot", "spotlinespunct", "spotnone", "fix1", "oracle", "oraclenf"):
        for t in ("S", "N", "L"):
            g = (hv.get(f) or {}).get(t, {}).get("groups")
            if g:
                out.append(f"| {FINDER_NAMES[f]} | {t} | " + " | ".join(_cell(g.get(k)) for k in (
                    "new:vita", "new:psp", "v1:vita", "v1:psp", "all")) + " |")
    sec["v2_baseline_hval"] = "\n".join(out)
    na = base["sets"].get("new_all_base_v21", {})
    out = ["| finder | tier | Vita | PSP | all |", "|---|---|---|---|---|"]
    for f in ("spot", "spotlinespunct", "spotnone", "oracle", "oraclenf"):
        for t in ("S", "N", "L"):
            g = (na.get(f) or {}).get(t, {}).get("groups")
            if g:
                out.append(f"| {FINDER_NAMES[f]} | {t} | " + " | ".join(_cell(g.get(k)) for k in (
                    "new:vita", "new:psp", "new")) + " |")
    sec["v2_baseline_new"] = "\n".join(out)
    sp, orc = ((na.get(f) or {}).get("N", {}).get("per_title", {}) for f in ("spot", "oracle"))
    out = ["| title | page path, textspot8 (N) | clean crops (N) | reference chars | units |",
           "|---|---:|---:|---:|---:|"]
    for tid in sorted(sp, key=lambda k: (-sp[k]["errors"], k)):
        s, o = sp[tid], orc.get(tid, {})
        out.append(f"| {tid} | {s['cer']:.2f}% | {'' if o.get('cer') is None else format(o['cer'], '.2f') + '%'} | "
                   f"{s['chars']:,} | {s['units']:,} |")
    sec["v2_baseline_titles"] = "\n".join(out)
    return sec


def v2_sections(index1, stats1, index2, stats2, overlap2, audit2, baseline=None):
    """{section: markdown} of docs/vocr6-data-v2.md's generated blocks."""
    base = corpus_tables(index2, stats2, overlap2, audit2)
    sec = {"v2_compare": compare_table(index1, stats1, index2, stats2), "v2_splits": base["splits"],
           "v2_filters": base["filters"], "v2_rules": rules_table(index2), "v2_coverage": base["coverage"],
           "v2_audit": base["audit"].replace("after the fixes (pools v1)", "after the fixes (pools v2.1)"),
           "v2_titles": titles2_table(index1, index2, stats2, overlap2)}
    if baseline is not None:
        sec.update(baseline2_tables(baseline))
    return sec


def cmd_tables2(a):
    load = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))  # noqa: E731
    sec = v2_sections(load(a.index_v1), load(a.stats_v1), load(a.index), load(a.stats), load(a.overlap),
                      load(a.audit), load(a.baseline) if a.baseline else None)
    doc = Path(a.doc).read_text(encoding="utf-8")
    new = splice(doc, sec)
    if a.check:
        if new != doc:
            raise SystemExit(f"{a.doc}: a generated table differs from the committed records")
        print(f"{a.doc}: every generated table matches")
        return
    Path(a.doc).write_text(new, encoding="utf-8")
    print(f"-> {a.doc}: {len(MARK.findall(doc))} generated blocks")


def cmd_tables(a):
    load = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))  # noqa: E731
    sec = table_sections(load(a.index), load(a.stats), load(a.overlap), load(a.audit),
                         load(a.baseline) if a.baseline else None)
    if a.out:
        Path(a.out).write_text("\n\n".join(f"<!-- {k} -->\n{v}" for k, v in sec.items()) + "\n", encoding="utf-8")
    if a.doc:
        doc = Path(a.doc).read_text(encoding="utf-8")
        new = splice(doc, sec)
        if a.check:
            if new != doc:
                raise SystemExit(f"{a.doc}: a generated table differs from the committed records")
            print(f"{a.doc}: every generated table matches")
            return
        Path(a.doc).write_text(new, encoding="utf-8")
        print(f"-> {a.doc}: {len(MARK.findall(doc))} generated blocks")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("tables")
    for k in ("index", "stats", "overlap", "audit"):
        p.add_argument(f"--{k}", required=True)
    p.add_argument("--baseline", default="")
    p.add_argument("--out", default="")
    p.add_argument("--doc", default="", help="replace the generated blocks of this markdown file")
    p.add_argument("--check", action="store_true", help="with --doc: fail if a block differs")
    p = sub.add_parser("tables2", help="docs/vocr6-data-v2.md's generated blocks (pools v2.1 against v1)")
    for k in ("index", "stats", "overlap", "audit", "index-v1", "stats-v1", "doc"):
        p.add_argument(f"--{k}", required=True)
    p.add_argument("--baseline", default="")
    p.add_argument("--check", action="store_true", help="fail if a block differs")
    p = sub.add_parser("sample")
    p.add_argument("--pools", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=320)
    p.add_argument("--seed", type=int, default=20261026)
    p.add_argument("--quota", default="", help='JSON {stratum: lines} replacing QUOTA (e.g. \'{"name": 60}\')')
    p.add_argument("--exclude-v1", default="", help="pools v1's directory: leave out the records it holds")
    p = sub.add_parser("sheets")
    p.add_argument("--sample", required=True)
    p.add_argument("--pools", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--per", type=int, default=20)
    p = sub.add_parser("tally")
    p.add_argument("--sheets", required=True)
    p.add_argument("--decisions", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("stats")
    p.add_argument("--pools", required=True)
    p.add_argument("--pagepool", required=True)
    p.add_argument("--splits", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("overlap")
    p.add_argument("--stats", required=True)
    p.add_argument("--vndb-map", required=True)
    p.add_argument("--vocr5-pools", default=str(ROOT / "eval" / "vocr5_pools.sha256"))
    p.add_argument("--vndb-dump", default="", help="a VNDB dump directory (db/releases_vn): shared releases")
    p.add_argument("--vn-set", action="append", default=[], help="NAME=FILE: flag the titles whose VN is listed")
    p.add_argument("--v1-index", default="", help="pools v1's committed index.json: titles and VNs VOCR6 trained on")
    p.add_argument("--hooked-test", action="store_true",
                   help="flag train titles that share a VN, a release or a VNDB relation with a hooked test title")
    p.add_argument("--relations", action="store_true", help="with --hooked-test: use the dump's vn_relations")
    p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    {"sample": cmd_sample, "sheets": cmd_sheets, "tally": cmd_tally, "stats": cmd_stats,
     "overlap": cmd_overlap, "tables": cmd_tables, "tables2": cmd_tables2}[a.cmd](a)


if __name__ == "__main__":
    main()
