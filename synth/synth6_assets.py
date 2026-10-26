"""synth6 assets: what synth6 learns from the owner's hooked corpus before it renders anything.

From the train titles' pages only (the `train` pages of `vocr6-ingest`'s split; never an hval or guard page, never
a test title) it derives, per title:

  layout       the text rows the game draws dialogue on (y band, left edges, right limit) and its name rows,
               clustered from the verified lines' boxes, plus every train page's boxes (to mask them)
  style        the dialogue text as drawn, measured on real line crops: ink and cell height, pen advance, fill
               colour, outline width and colour, shadow offset and colour, polarity; and the dialogue box's colour
               and opacity, from the text rows a page leaves empty (their variation across pages against the
               scene's)
  font         the title's verified dialogue font and its native size or scale and letter spacing (the font
               pack's fidelity records), if the farm harvested one
  text         the train pages' hook text (dialogue and names) for synth6's text mix, and the vocab characters'
               counts in it (rare and absent characters get targeted lines)

`build` writes `titles.json`, `hooktext.jsonl`, `real_counts.json`, `test_text.json` (the test titles' lines)
and `hval_text.json` (the train titles' hval and guard lines) under --out; the last two only guard the renderer.
Nothing here is committed: the derived per-title numbers are summarised in `data/manifests/synth6_styles.json`
(`synth6.py manifests`).
"""
import io
import json
import math
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent


def read_jsonl_zst(path):
    """Rows of a .jsonl.zst file (the corpus's pages and lines files), through the zstd CLI."""
    out = subprocess.run(["zstd", "-dc", str(path)], capture_output=True, check=True).stdout
    for line in io.TextIOWrapper(io.BytesIO(out), encoding="utf-8"):
        if line.strip():
            yield json.loads(line)


def title_dirs(hooked):
    """{title_id: (folder, metadata)} of every title folder (<platform>/<TITLE_ID> <name>/) of the corpus mirror."""
    out = {}
    for plat in ("vita", "psp"):
        root = Path(hooked) / plat
        if not root.exists():
            continue
        for d in sorted(root.iterdir()):
            mf = d / "metadata.json"
            if d.is_dir() and mf.exists():
                md = json.loads(mf.read_text(encoding="utf-8"))
                out[md["title_id"]] = (d, md)
    return out


def page_split(splits, title):
    """{page id: train|hval|guard|test} of one title, from vocr6-ingest's split file."""
    t = splits["titles"].get(title)
    return dict(t["assign"]) if t else {}


# ----------------------------------------------------------------------------- layout

def cluster_rows(boxes, tol=4):
    """Text rows from line boxes [(x, y, w, h)]: boxes whose top and height agree within `tol` px join one row.
    Returns rows sorted by count: {"y0", "h", "x0": [left edges by count], "x1": right limit (p99 of right
    edges), "count"}."""
    rows = []
    for x, y, w, h in sorted(boxes, key=lambda b: (b[1], b[3])):
        for r in rows:
            if abs(r["y0s"][0] - y) <= tol and abs(r["hs"][0] - h) <= tol:
                r["y0s"].append(y)
                r["hs"].append(h)
                r["x0s"].append(x)
                r["x1s"].append(x + w)
                break
        else:
            rows.append({"y0s": [y], "hs": [h], "x0s": [x], "x1s": [x + w]})
    out = []
    for r in rows:
        x0c = Counter(int(round(v)) for v in r["x0s"])
        out.append({"y0": int(np.median(r["y0s"])), "h": int(np.median(r["hs"])),
                    "x0": [[int(k), int(n)] for k, n in x0c.most_common(6)],
                    "x1": int(np.percentile(r["x1s"], 99)), "count": len(r["y0s"])})
    return sorted(out, key=lambda r: -r["count"])


def layout_of(lines, min_share=0.02, min_lines=8):
    """The title's dialogue rows and name rows from its verified horizontal lines (train pages): rows holding at
    least `min_lines` lines and `min_share` of the kind's lines (a layout the game uses, not a stray box), in
    screen order."""
    out = {}
    for kind in ("line", "name"):
        bx = [tuple(r["box"]) for r in lines if r["kind"] == kind and not r.get("vertical")]
        if not bx:
            out[kind] = []
            continue
        rows = [r for r in cluster_rows(bx) if r["count"] >= max(min_lines, min_share * len(bx))]
        out[kind] = sorted(rows, key=lambda r: r["y0"])
    return out


# ----------------------------------------------------------------------------- style measurement

def luma(rgb):
    """Integer luma as the console computes it (common.to_gray), as int32."""
    r, g, b = (rgb[..., i].astype(np.int32) for i in range(3))
    return (77 * r + 150 * g + 29 * b) >> 8


OFFSETS = ((1, 1), (2, 2), (3, 3), (0, 2), (2, 0), (1, 2), (2, 1))


def measure_line(rgb, box, cells=None, margin=5):
    """The drawn style of one real line, from its crop (box + `margin` px). None when the crop has no clear text.

    Polarity from common.line_dark; v = grey with the text bright (dark text inverted). The glyph mask is every
    pixel a quarter of the way from the background to the fill (anti-aliased edges included), the fill the
    Otsu-bright core. With the far background bg (pixels 5+ px from any glyph), ring k (distance k - 1 to k px
    outside the glyphs) deviates by d_k = bg - median(ring k): an outline or halo darker than the box (in v)
    makes d_k > 0. The shadow is the asymmetry between the glyph mask shifted below-right and above-left, outside
    the halo."""
    from common import line_dark
    H, W = rgb.shape[:2]
    x, y, w, h = (int(round(v)) for v in box)
    x0, y0, x1, y1 = max(0, x - margin), max(0, y - margin), min(W, x + w + margin), min(H, y + h + margin)
    if x1 - x0 < 10 or y1 - y0 < 8:
        return None
    crop = rgb[y0:y1, x0:x1, :3]
    g = luma(crop).astype(np.uint8)
    dark = line_dark(g, (0, 0, g.shape[1], g.shape[0]))
    v = ((255 - g) if dark else g).astype(np.float32)
    t, _ = cv2.threshold(v.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    core = v > t
    if core.sum() < 12:
        return None
    fillv = float(np.median(v[core]))
    D0 = cv2.distanceTransform((~core).astype(np.uint8), cv2.DIST_L2, 3)
    far = D0 >= 6
    if far.sum() < 30:
        return None
    bgv = float(np.median(v[far]))
    contrast = fillv - bgv
    if contrast < 24:
        return None
    M = v > bgv + 0.25 * contrast           # glyph mask with the anti-aliased edge
    D = cv2.distanceTransform((~M).astype(np.uint8), cv2.DIST_L2, 3)
    d, ring_rgb = [], []
    for k in (1, 2, 3, 4):
        r = (D > k - 1) & (D <= k)
        if r.sum() < 6:
            d.append(0.0)
            ring_rgb.append(None)
            continue
        d.append(round(bgv - float(np.median(v[r])), 1))
        ring_rgb.append([int(c) for c in np.median(crop[r], axis=0)])
    halo = int(np.argmax(d)) + 1
    asym = {}
    beyond = D > max(1, sum(1 for x_ in d if x_ > 0.5 * max(max(d), 1e-6)))
    for dx, dy in OFFSETS:
        sh = np.zeros_like(M)
        sh[dy:, dx:] = M[:M.shape[0] - dy, :M.shape[1] - dx]
        op = np.zeros_like(M)
        op[:M.shape[0] - dy, :M.shape[1] - dx] = M[dy:, dx:]
        a_, b_ = sh & ~M & beyond, op & ~M & beyond
        if a_.sum() >= 10 and b_.sum() >= 10:
            asym[f"{dx},{dy}"] = (round(float(np.median(v[b_]) - np.median(v[a_])), 1),
                                  [int(c) for c in np.median(crop[a_], axis=0)])
    band = core[max(0, y - y0 - 1):y - y0 + h + 1]          # the line's own rows: no neighbour's glyphs
    ys = np.nonzero(band)[0] if band.any() else np.nonzero(core)[0]
    out = {"dark": int(dark), "fill_rgb": [int(c) for c in np.median(crop[core], axis=0)],
           "bg_v": round(bgv, 1), "fill_v": round(fillv, 1), "contrast": round(contrast, 1), "ring_d": d,
           "halo_k": halo, "halo_rgb": ring_rgb[halo - 1], "asym": asym,
           "ink_h": int(np.percentile(ys, 99) - np.percentile(ys, 1) + 1), "box_h": int(h),
           "bg_rgb": [int(c) for c in np.median(crop[far], axis=0)]}
    if cells and len(cells) >= 2:
        xs = sorted(c[0] for c in cells)
        out["advance"] = float(np.median(np.diff(xs)))
    return out


def summarise_styles(ms):
    """Per-title style from the per-line measurements, and the render parameters they imply.

    halo (outline or glow): present when the median ring deviation peaks above max(6, 6% of the contrast); its
    radius is the peak ring, its width the rings above half the peak, its opacity the peak over the background's
    distance from black (in v), its colour the peak ring's. shadow: the offset with the largest median asymmetry
    when that exceeds max(6, 5% of the contrast), opacity likewise. The fill: the median colour, with the share
    of lines whose fill differs (coloured names, emphasis) and up to 12 of those colours."""
    ms = [m for m in ms if m]
    if len(ms) < 3:
        return None
    dark = float(np.mean([m["dark"] for m in ms]))
    fills = np.array([m["fill_rgb"] for m in ms], np.float32)
    main = np.median(fills, axis=0)
    plain = np.abs(fills - main).max(axis=1) <= 40
    ring = np.median(np.array([m["ring_d"] for m in ms], np.float32), axis=0)
    contrast = float(np.median([m["contrast"] for m in ms]))
    bgv = float(np.median([m["bg_v"] for m in ms]))
    out = {"lines": len(ms), "dark_share": round(dark, 3), "fill_rgb": [int(c) for c in main],
           "fill_plain_share": round(float(plain.mean()), 3),
           "accent_fills": [[int(c) for c in f] for f in fills[~plain][:12]],
           "contrast_p50": round(contrast, 1), "bg_v_p50": round(bgv, 1),
           "ring_d_p50": [round(float(x), 1) for x in ring],
           "bg_rgb_p50": [int(c) for c in np.median(np.array([m["bg_rgb"] for m in ms]), axis=0)],
           "ink_h_p10": int(np.percentile([m["ink_h"] for m in ms], 10)),
           "ink_h_p50": int(np.median([m["ink_h"] for m in ms])),
           "ink_h_p90": int(np.percentile([m["ink_h"] for m in ms], 90)),
           "box_h_p50": int(np.median([m["box_h"] for m in ms]))}
    adv = [m["advance"] for m in ms if m.get("advance")]
    if adv:
        out["advance_p50"] = round(float(np.median(adv)), 2)
    peak = float(ring.max())
    if peak > max(6.0, 0.06 * contrast):
        k = int(np.argmax(ring)) + 1
        cols = [m["halo_rgb"] for m in ms if m.get("halo_rgb") and m["halo_k"] == k] or \
            [m["halo_rgb"] for m in ms if m.get("halo_rgb")]
        out["halo"] = {"radius": k, "width": int(sum(1 for x in ring if x > 0.5 * peak)),
                       "alpha": round(float(np.clip(peak / max(bgv, 8.0), 0.05, 1.0)), 2),
                       "rgb": [int(c) for c in np.median(np.array(cols), axis=0)] if cols else None,
                       "hard": bool(peak > 0.45 * max(bgv, 8.0))}
    best, bo = 0.0, None
    for off in [f"{dx},{dy}" for dx, dy in OFFSETS]:
        vals = [m["asym"][off][0] for m in ms if off in m["asym"]]
        if len(vals) >= max(3, len(ms) // 3) and float(np.median(vals)) > best:
            best, bo = float(np.median(vals)), off
    if bo and best > max(6.0, 0.05 * contrast):
        cols = [m["asym"][bo][1] for m in ms if bo in m["asym"]]
        out["shadow"] = {"offset": [int(v) for v in bo.split(",")], "asym": round(best, 1),
                         "alpha": round(float(np.clip(best / max(bgv, 8.0), 0.05, 1.0)), 2),
                         "rgb": [int(c) for c in np.median(np.array(cols), axis=0)]}
    return out


def box_opacity(pages, layout, read, n=80, rng=None):
    """The dialogue box's colour and opacity from the text rows the pages leave empty.

    For each dialogue row, a fixed patch inside the row (its left edge to the right limit, inset 3 px) is
    collected from up to `n` train pages on which no box touches it, and the same-size patch from the scene above
    the box (the frame's upper part). With a box of colour C and opacity a over scenes S, the across-page variance
    of the row patch is (1 - a)^2 times the scene's: a = 1 - sqrt(var_row / var_scene), C from the means."""
    rng = rng or np.random.default_rng(0)
    rows = layout.get("line") or []
    if not rows:
        return None

    def free(r, boxes):
        x0 = min(v for v, _ in r["x0"])
        return all(bx + bw + 4 <= x0 or by + bh + 4 <= r["y0"] or by - 4 >= r["y0"] + r["h"]
                   for bx, by, bw, bh in boxes)
    best = max(rows, key=lambda r: sum(free(r, b) for _, b in pages))
    x0 = min(v for v, _ in best["x0"])
    xa, xb = x0 + 3, min(best["x1"], x0 + 3 + 320) - 3
    ya, yb = best["y0"] + 3, best["y0"] + best["h"] - 3
    if xb - xa < 16 or yb - ya < 4:
        return None
    patches, scenes = [], []
    order = list(pages)
    rng.shuffle(order)
    for pid, boxes in order:
        if len(patches) >= n:
            break
        if any(not (bx + bw + 4 <= xa or bx - 4 >= xb or by + bh + 4 <= ya or by - 4 >= yb)
               for bx, by, bw, bh in boxes):
            continue
        img = read(pid)
        if img is None:
            continue
        H = img.shape[0]
        sy = max(0, int(0.18 * H))
        if any(not (bx + bw <= xa or bx >= xb or by + bh <= sy or by >= sy + (yb - ya))
               for bx, by, bw, bh in boxes):
            continue
        patches.append(img[ya:yb, xa:xb, :3].astype(np.float32))
        scenes.append(img[sy:sy + (yb - ya), xa:xb, :3].astype(np.float32))
    if len(patches) < 8:
        return {"pages": len(patches)}
    P, S = np.stack(patches), np.stack(scenes)
    vp = float(P.var(axis=0).mean())
    vs = float(S.var(axis=0).mean())
    a = float(np.clip(1.0 - math.sqrt(vp / max(vs, 1e-6)), 0.0, 1.0)) if vs > 25 else None
    mp, ms = P.mean(axis=(0, 1, 2)), S.mean(axis=(0, 1, 2))
    if a is not None and a > 0.15:
        col = np.clip((mp - (1 - a) * ms) / a, 0, 255)
    else:
        col = mp
    tex = float(P.mean(axis=0).std())          # the box's own texture: spatial spread of the across-page mean
    return {"pages": len(patches), "alpha": None if a is None else round(a, 3), "rgb": [int(c) for c in col],
            "row_var": round(vp, 1), "scene_var": round(vs, 1), "texture_sd": round(tex, 2),
            "patch": [xa, ya, xb - xa, yb - ya]}


# ----------------------------------------------------------------------------- build

def page_boxes(rec):
    """A page record's text boxes as integer (x, y, w, h), rounded outward."""
    out = []
    for b in rec.get("boxes") or []:
        x0, y0, x1, y1 = b["xyxy"]
        x0, y0 = int(math.floor(x0)), int(math.floor(y0))
        out.append([x0, y0, int(math.ceil(x1)) - x0, int(math.ceil(y1)) - y0])
    return out


def build_title(tid, folder, md, splits, pages_root, harvest_rows, n_style=160, n_name=60, seed=0):
    """Everything synth6 needs of one train title (see the module docstring), and its train hook text."""
    import synth6_fonts as F6
    assign = page_split(splits, tid)
    if md.get("split") != "train" or not assign:
        return None, [], Counter()
    W, H = md.get("resolution") or (960, 544)
    pages = {}
    for rec in read_jsonl_zst(folder / "pages.jsonl.zst"):
        if assign.get(rec["id"]) == "train":
            pages[rec["id"]] = page_boxes(rec)
    lines = []
    lf = folder / "lines.jsonl.zst"
    if lf.exists():
        lines = [r for r in read_jsonl_zst(lf) if assign.get(r["page"]) == "train"]
    lay = layout_of(lines)
    pdir = Path(pages_root) / tid / "pages"

    def read(pid):
        im = cv2.imread(str(pdir / f"{pid}.png"), cv2.IMREAD_COLOR)
        return None if im is None else im[..., ::-1]

    rng = np.random.default_rng([seed, int.from_bytes(tid.encode()[:8].ljust(8, b"\0"), "little") % (1 << 31)])
    styles = {}
    for kind, n in (("line", n_style), ("name", n_name)):
        cand = [r for r in lines if r["kind"] == kind and not r.get("vertical")]
        if not cand:
            continue
        pick = sorted(rng.choice(len(cand), size=min(n, len(cand)), replace=False))
        by_page = defaultdict(list)
        for i in pick:
            by_page[cand[i]["page"]].append(cand[i])
        ms = []
        for pid, rs in by_page.items():
            img = read(pid)
            if img is None:
                continue
            for r in rs:
                ms.append(measure_line(img, r["box"], r.get("cell_boxes")))
        styles[kind] = summarise_styles(ms)
    box = box_opacity(list(pages.items()), lay, read, rng=rng)
    fonts = F6.dialogue_fonts(harvest_rows, {tid})
    hook = []
    by_page = defaultdict(list)
    for r in lines:
        by_page[r["page"]].append(r)
    counts = Counter()
    for pid, rs in by_page.items():
        rs.sort(key=lambda r: r["id"])
        body = [r["text"] for r in rs if r["kind"] == "line" and not r.get("vertical")]
        if body:
            hook.append({"title": tid, "page": pid, "kind": "line", "text": "".join(body), "lines": body})
        for r in rs:
            if r["kind"] == "name":
                hook.append({"title": tid, "page": pid, "kind": "name", "text": r["text"]})
            counts.update(r["text"])
    vert = sum(1 for r in lines if r.get("vertical"))
    info = {"title": tid, "name": md.get("title"), "platform": md.get("platform"), "dims": [int(W), int(H)],
            "vndb": (md.get("vndb") or {}).get("vn"), "pages_train": len(pages), "lines_train": len(lines),
            "vertical_lines": vert, "layout": lay, "style": styles, "box": box,
            "font": None if tid not in fonts else dict(zip(("font", "size_or_scale", "spacing", "corr"), fonts[tid])),
            "pages": pages}
    return info, hook, counts


def _build_one(job):
    tid, folder, md, splits_path, pages_root, harvest_rows, seed = job
    splits = json.loads(Path(splits_path).read_text(encoding="utf-8"))
    cv2.setNumThreads(1)
    return tid, build_title(tid, Path(folder), md, splits, pages_root, harvest_rows, seed=seed)


def build(hooked, pages_root, splits_path, harvest, out, workers=4, titles=None, seed=0):
    import multiprocessing as mp

    import synth6_fonts as F6
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    dirs = title_dirs(hooked)
    splits = json.loads(Path(splits_path).read_text(encoding="utf-8"))
    train = {t for t, (_, md) in dirs.items() if md.get("split") == "train"}
    rows = F6.inventory(harvest, train_titles=train)
    jobs = [(t, str(d), md, str(splits_path), str(pages_root), rows, seed) for t, (d, md) in sorted(dirs.items())
            if (titles is None or t in titles) and md.get("split") == "train" and t in splits["titles"]]
    infos, counts = {}, Counter()
    with open(out / "hooktext.jsonl", "w", encoding="utf-8") as fh, \
            mp.get_context("fork").Pool(workers) as pool:
        for tid, (info, hook, cnt) in pool.imap_unordered(_build_one, jobs):
            if info is None:
                continue
            infos[tid] = info
            counts.update(cnt)
            for h in hook:
                fh.write(json.dumps(h, ensure_ascii=False) + "\n")
            st = info["style"].get("line") or {}
            print(f"{tid} {info['platform']}: {info['pages_train']} train pages, {info['lines_train']} lines, rows "
                  f"{[(r['y0'], r['h']) for r in info['layout']['line']]}, font "
                  f"{(info['font'] or {}).get('font')}, ink {st.get('ink_h_p50')} outline {st.get('outline_share')} "
                  f"shadow {st.get('shadow_share')} dark {st.get('dark_share')} box {info['box']}", flush=True)
    test = sorted(t for t, (_, md) in dirs.items() if md.get("split") == "test")
    doc = {"kind": "synth6-titles", "schema": 1, "splits_sha256": F6.sha256_file(splits_path),
           "splits_file": str(splits_path), "test_titles": test, "titles": infos}
    (out / "titles.json").write_text(json.dumps(doc, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "real_counts.json").write_text(json.dumps({"kind": "synth6-real-counts", "schema": 1,
                                                      "titles": sorted(infos), "chars": dict(counts.most_common())},
                                                     ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "fonts_inventory.json").write_text(json.dumps(rows, ensure_ascii=False, indent=0) + "\n",
                                              encoding="utf-8")
    # the hval and guard pages' line text of every train title, for the renderer's screening guard only (a
    # synthetic line never repeats an hval line: synth6 text must not teach the screening set's lines)
    held = set()
    for t, (d, md) in dirs.items():
        assign = page_split(splits, t)
        lf = d / "lines.jsonl.zst"
        if md.get("split") == "train" and lf.exists():
            held |= {r["text"] for r in read_jsonl_zst(lf) if assign.get(r["page"]) in ("hval", "guard")}
    (out / "hval_text.json").write_text(json.dumps({"splits_sha256": F6.sha256_file(splits_path),
                                                    "lines": sorted(held)}, ensure_ascii=False) + "\n",
                                        encoding="utf-8")
    # test-title text, for the renderer's decontamination filter only (never rendered)
    tt = []
    for t in test:
        lf = dirs[t][0] / "lines.jsonl.zst"
        if lf.exists():
            tt += [r["text"] for r in read_jsonl_zst(lf)]
    (out / "test_text.json").write_text(json.dumps({"titles": test, "lines": tt}, ensure_ascii=False) + "\n",
                                        encoding="utf-8")
    print(f"BUILD_DONE {len(infos)} titles -> {out}", flush=True)
    return doc
