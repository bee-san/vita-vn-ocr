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


def cmd_sample(a):
    rows = kept_rows(a.pools)
    rng = np.random.default_rng(a.seed)
    by = defaultdict(list)
    for r in rows:
        by[stratum(r[1], r[4])].append(r)
    picked = []
    for s, q in QUOTA.items():                      # oversampled strata: spread over titles round robin
        cand = by.get(s, [])
        per_title = defaultdict(list)
        for r in cand:
            per_title[r[2]].append(r)
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
        per_title[r[2]].append(r)
    titles = sorted(per_title)
    order = {t: list(rng.permutation(len(v))) for t, v in per_title.items()}
    n0 = len(picked)
    while len(picked) - n0 < rest:                  # at least one line of every title, then round robin
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
            index[f"S{sheet_no:02d}:{k + 1}"] = {**e, "label_chars": len(lab), "crop_shape": [h, w]}
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
    out = {"kind": "vocr6-hooked-overlap", "reserved_sha256": IH.sha256_file(ROOT / "eval" / "manifests" /
                                                                             "reserved.json"),
           "vndb_dump": (Path(a.vndb_dump) / "TIMESTAMP").read_text().strip() if a.vndb_dump else None,
           "vocr5_pools": v5files, "titles": {}}
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
        rows5 = {v: v5.get(v, 0) for v in vns}
        protected = any(":vita-test" in f or ":psp-test" in f or ":clannad" in f or ":reserved-test" in f or
                        ":shares-" in f for f in flags)
        out["titles"][tid] = {"vns": vns, "vndb_source": m["source"], "platform": info["platform"],
                              "title_split": info.get("title_split"), "flags": flags,
                              "vocr5_pool_rows": rows5, "seen_by_vocr5": any(rows5.values()),
                              "seen_protected_family": protected,
                              "dev_sample": any(":dev_sample" in f for f in flags)}
    out["seen_protected_family"] = sorted(t for t, v in out["titles"].items() if v["seen_protected_family"])
    out["dev_sample_titles"] = sorted(t for t, v in out["titles"].items() if v["dev_sample"])
    out["seen_by_vocr5"] = sorted(t for t, v in out["titles"].items() if v["seen_by_vocr5"])
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
    p = sub.add_parser("sample")
    p.add_argument("--pools", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=320)
    p.add_argument("--seed", type=int, default=20261026)
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
    p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    {"sample": cmd_sample, "sheets": cmd_sheets, "tally": cmd_tally, "stats": cmd_stats,
     "overlap": cmd_overlap, "tables": cmd_tables}[a.cmd](a)


if __name__ == "__main__":
    main()
