"""The owner's Manga109 download (issue #159): the 109 volume covers as NOBORU views, their vertical columns' labels
and a training shard of the Manga109-s ones.

  manga109_covers.py views  --root DIR/manga109 --inventory INV.json --out DIR
  manga109_covers.py labels --reads DIR/reads.jsonl --review DIR/review --out LABELS.jsonl
  manga109_covers.py shard  --reads DIR/reads.jsonl --labels LABELS.jsonl --out SHARD.npz [--half]

The download is Manga109's super-resolution benchmark version (Originals/ and ValidOriginals/: one colour cover per
volume, 810 x 1080; HR/LR tiles cut from them), with no interior pages and no annotations. `views` shows each cover
as NOBORU's fit-to-width mode shows a page (eval/vertical_data.noboru_views: scaled to 960 px wide, cut into
544-row windows) and writes eval/vertical_data.py's pages format (origin "manga109", plat "manga", split "m109s" for
the 87 Manga109-s volumes and "m109only" for the 22 others, which may never be training data). Then
`eval/vertical_data.py read` reads every vertical column (MeikiOCR's detector and page read, MeikiOCR on the crop,
PP-OCRv6 small on the turned crop) and `eval/vertical_data.py sheets` draws the review sheets.

`labels`: one review pass (Claude; reviewer_kind "assistant") decides each column; a column is labelled only when
the decided text equals one engine's reading (canon form), as the fleet's labelling rule asks of training labels.
`shard`: the labelled Manga109-s columns' crops (column_box's padded crop of the view) in the console's polarity,
turned 90 degrees anticlockwise, in train/train2.py's format; --half adds each at half scale.

Licence (manga109.org): research use, no redistribution. Views, reads, labels and shards stay in the private
directory; the repo gets this code and counts only.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "synth"))


def sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_views(a):
    import cv2
    import vertical_data as vd
    inv = {r["path"]: r for r in json.loads(Path(a.inventory).read_text(encoding="utf-8"))["files"]}
    out = Path(a.out)
    (out / "views").mkdir(parents=True, exist_ok=True)
    rows = []
    for d in ("Originals", "ValidOriginals"):
        for p in sorted((Path(a.root) / d).glob("*.jpg")):
            rec = inv[f"{d}/{p.name}"]
            rgb = cv2.cvtColor(cv2.imread(str(p), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            stem = re.sub(r"[^0-9A-Za-z]+", "_", p.stem)
            for view, img, (s, ox, oy) in vd.noboru_views(rgb):
                pid = f"m109_{stem}_{view}"
                path = out / "views" / f"{pid}.png"
                cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                rows.append({"page": pid, "origin": "manga109", "vn": f"m109:{rec['title']}", "kind": "vita",
                             "url": None, "sha256": sha256_file(path), "path": str(path), "release": None,
                             "dims": [960, 544], "cluster": f"m109:{rec['title']}",
                             "split": "m109s" if rec["manga109s"] else "m109only", "plat": "manga",
                             "manga": {"member": f"{d}/{p.name}", "member_sha256": rec["sha256"], "view": view,
                                       "scale": s, "ox": ox, "oy": oy, "page_dims": [rgb.shape[1], rgb.shape[0]]}})
    vd.write_jsonl(out / "pages_m109.jsonl", rows)
    print(len(rows), "views of", len({r["vn"] for r in rows}), "covers ->", out / "pages_m109.jsonl")


def engine_texts(col):
    import vertical_data as vd
    return {vd.canon(t) for t in (col.get("reads") or {}).values() if t}


def cmd_labels(a):
    """Labels from one review pass: the decided text, kept only when an engine read the same."""
    import vertical_data as vd
    by_id = {}
    for r in vd.jsonl(a.reads):
        for c in r.get("columns", []):
            by_id[vd.col_id(r["page"], c["k"])] = (r, c)
    dec = {}
    for f in sorted(Path(a.review).glob("decisions*.jsonl")):
        for d in vd.jsonl(f):
            sh = json.loads((Path(a.review) / f"{d['sheet']}.json").read_text(encoding="utf-8"))
            e = next(x for x in sh["entries"] if x["n"] == d["n"])
            cand = {c["c"]: c["text"] for c in e["candidates"]}
            t = cand[d["pick"]] if "pick" in d else d.get("text")
            dec[e["id"]] = {"text": vd.canon(t) if t else None, "skip": d.get("skip"), "unsure": bool(d.get("unsure")),
                            "name": bool(d.get("name"))}
    out, stats = [], {}
    for cid, (r, c) in by_id.items():
        z = dec.get(cid)
        if z is None:
            why = "undecided"
        elif z["skip"] or z["unsure"] or not z["text"]:
            why = z["skip"] or "unsure"
        elif z["text"] not in engine_texts(c):
            why = "no_engine_agrees"
        else:
            why = "label"
        stats[(r["split"], why)] = stats.get((r["split"], why), 0) + 1
        row = {"id": cid, "page": r["page"], "split": r["split"], "why": why, "reviewer_kind": "assistant"}
        if why == "label":
            row["text"] = z["text"]
        out.append(row)
    vd.write_jsonl(a.out, out)
    print(json.dumps({f"{k[0]}:{k[1]}": v for k, v in sorted(stats.items())}))


def cmd_shard(a):
    import cv2
    import vertical_data as vd

    from common import line_dark
    labels = {r["id"]: r for r in vd.jsonl(a.labels) if r["why"] == "label" and r["split"] == "m109s"}
    crops, texts, metas = [], [], []
    for r in vd.jsonl(a.reads):
        if r.get("split") != "m109s" or r.get("error"):
            continue
        img = None
        for c in r["columns"]:
            cid = vd.col_id(r["page"], c["k"])
            if cid not in labels:
                continue
            if img is None:
                img = vd.to_grey(vd.load_page(r))
            # as eval/vertical_data.py cmd_shard cuts #103's real columns
            for sc in ((1.0, 0.5) if a.half else (1.0,)):
                size = (max(1, round(img.shape[1] * sc)), max(1, round(img.shape[0] * sc)))
                gg = img if sc == 1.0 else cv2.resize(img, size, interpolation=cv2.INTER_AREA)
                bx = [int(round(v * sc)) for v in c["crop"]]
                if sc != 1.0:            # the PSP pad: 2 px around the halved ink box
                    ix0, iy0, ix1, iy1 = [int(round(v * sc)) for v in c["ink"]]
                    bx = [max(0, ix0 - 2), max(0, iy0 - 2), min(gg.shape[1], ix1 + 2), min(gg.shape[0], iy1 + 2)]
                w, h = bx[2] - bx[0], bx[3] - bx[1]
                if w < 4 or h < 8:
                    continue
                crop = gg[bx[1]:bx[3], bx[0]:bx[2]]
                dark = line_dark(gg, (bx[0], bx[1], w, h))
                if dark:
                    crop = 255 - crop
                crops.append(np.ascontiguousarray(np.rot90(crop, 1)))
                texts.append(labels[cid]["text"])
                metas.append({"real": True, "pool": "manga109s-covers", "id": cid, "page": r["page"],
                              "member_sha256": r["manga"]["member_sha256"], "dark": int(dark), "scale": sc})
    shapes = np.array([c.shape for c in crops], np.int32).reshape(-1, 2)
    offs = np.concatenate([[0], np.cumsum([c.size for c in crops])]).astype(np.int64)
    data = np.concatenate([c.ravel() for c in crops]) if crops else np.zeros(0, np.uint8)
    np.savez(a.out, data=data, offsets=offs, shapes=shapes, labels=json.dumps(texts, ensure_ascii=False),
             meta=json.dumps(metas, ensure_ascii=False))
    print(len(crops), "crops ->", a.out)


def iou(a, b):
    """IoU of two [x, y, w, h] boxes."""
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    u = a[2] * a[3] + b[2] * b[3] - ix * iy
    return ix * iy / u if u else 0.0


def matched(refs, dets, thr=0.5):
    """How many refs a detection matches one to one at IoU >= thr (greedy by IoU)."""
    cand = sorted(((iou(r, d), i, j) for i, r in enumerate(refs) for j, d in enumerate(dets)), reverse=True)
    ur, ud, n = set(), set(), 0
    for v, i, j in cand:
        if v < thr or i in ur or j in ud:
            continue
        ur.add(i)
        ud.add(j)
        n += 1
    return n


_DET = {}


def _det_init(blob):
    import textspot6_twin as T6
    _DET["model"] = T6.Model(Path(blob).read_bytes())


def detect_view(r):
    """One view's vertical boxes: MeikiOCR's (the reads), textspot8's (the spotter on the 960x544 view, its vertical
    flag) and the column finder P1's (runtime/vocr_vertical.c's twin on the whole view), and the labelled columns."""
    import vertical_data as vd

    import textspot6_twin as T6
    import vertical_twin as vt
    rgb = vd.load_page(r)
    meiki = [[c["ink"][0], c["ink"][1], c["ink"][2] - c["ink"][0], c["ink"][3] - c["ink"][1]] for c in r["columns"]]
    m = _DET["model"]
    res = T6.spot(m, np.ascontiguousarray(rgb), scale=T6.scale_of(m, 960, 544))
    spot_v = [[b.x, b.y, b.w, b.h] for b in res.boxes if b.vertical]
    spot_h = sum(1 for b in res.boxes if not b.vertical)
    cols, _, _ = vt.find_columns(rgb, 1)
    return {"page": r["page"], "split": r["split"], "meiki": meiki, "spot_v": spot_v, "spot_h": spot_h,
            "p1": [list(b) for b in cols], "label_ids": [vd.col_id(r["page"], c["k"]) for c in r["columns"]]}


def cmd_detect(a):
    """Report only: how the spotter and the column finder see the covers' vertical text (counts and matches)."""
    import multiprocessing as mp

    import vertical_data as vd
    labels = {r["id"]: r for r in vd.jsonl(a.labels)} if a.labels else {}
    rows = [r for r in vd.jsonl(a.reads) if not r.get("error")]
    with mp.get_context("spawn").Pool(a.procs, initializer=_det_init, initargs=(a.blob,)) as pool:
        views = list(pool.imap(detect_view, rows, chunksize=4))
    out = {}
    for split in ("m109s", "m109only", "all"):
        vs = [v for v in views if split == "all" or v["split"] == split]
        lab = [(v, i) for v in vs for i, cid in enumerate(v["label_ids"]) if labels.get(cid, {}).get("why") == "label"]
        out[split] = {
            "views": len(vs), "meiki_vertical": sum(len(v["meiki"]) for v in vs),
            "spot_vertical": sum(len(v["spot_v"]) for v in vs), "spot_horizontal": sum(v["spot_h"] for v in vs),
            "p1_columns": sum(len(v["p1"]) for v in vs),
            "meiki_matched_by_spot": sum(matched(v["meiki"], v["spot_v"]) for v in vs),
            "meiki_matched_by_p1": sum(matched(v["meiki"], v["p1"]) for v in vs),
            "labelled": len(lab),
            "labelled_matched_by_spot": sum(matched([v["meiki"][i]], v["spot_v"]) for v, i in lab),
            "labelled_matched_by_p1": sum(matched([v["meiki"][i]], v["p1"]) for v, i in lab)}
    text = json.dumps(out, indent=1)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8")
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("detect")
    p.add_argument("--reads", required=True)
    p.add_argument("--blob", required=True, help="the textspot8 blob (v0.3.10-textspot8.1)")
    p.add_argument("--labels")
    p.add_argument("--procs", type=int, default=5)
    p.add_argument("--out")
    p = sub.add_parser("views")
    p.add_argument("--root", required=True)
    p.add_argument("--inventory", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("labels")
    p.add_argument("--reads", required=True)
    p.add_argument("--review", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("shard")
    p.add_argument("--reads", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--half", action="store_true")
    a = ap.parse_args()
    {"views": cmd_views, "labels": cmd_labels, "shard": cmd_shard, "detect": cmd_detect}[a.cmd](a)


if __name__ == "__main__":
    main()
