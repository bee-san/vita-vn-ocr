#!/usr/bin/env python3
"""The owner's hooked PS Vita / PSP corpus (B2 bucket `data-from-psp-and-vita`, issue #125) in the repo's pool
formats (docs/vocr6-data.md).

  import_hooked.py extract --root MIRROR --out PAGES [--procs 5]
      every title's pages.tar.zst -> PAGES/<title id>/pages/NNNNNN.png; each PNG's sha256 is checked against its
      page record (pages.jsonl.zst) and every line record that cites it
  import_hooked.py split --root MIRROR --out SPLITS.json [--share 0.04] [--guard 15] [--seed 20261026]
      the page split of every title (split_title): a `test` title whole; every `train` title keeps ~4% of its
      pages as contiguous hval blocks with a guard band either side (guard pages leave screening runs, like hval,
      and come back with it for final models); everything else is train
  import_hooked.py lines --root MIRROR --pages PAGES --splits SPLITS.json --out POOLS [--procs 5]
      every line record -> a grey crop (box + pad), its label, the filters (geometry, vocab, CTC feasibility),
      exact and near-duplicate clusters, and the pools POOLS/<split>/<pool>/<title id>.npz (the shard format of
      train/train2.py: data, offsets, shapes, labels, meta); POOLS/decisions/<title id>.jsonl records every
      record's outcome; POOLS/manifest.json every file's sha256 and every filter's counts
  import_hooked.py pages --root MIRROR --pages PAGES --splits SPLITS.json --pools POOLS --out PAGEPOOL
      the page-level pool: PAGEPOOL/<title id>.jsonl, every page with its split, its renderer boxes, its verified
      lines, whether every box is accounted for (`complete`), and the whole-dialogue regions of
      eval/finder_tighten.py's rule
  import_hooked.py manifests --splits SPLITS.json --pools POOLS --pagepool PAGEPOOL --out data/manifests/vocr6_hooked
      the committed manifests: per title its pages (id, sha256, split) and each line's pool or drop code, no text;
      index.json with the sources' and pools' sha256 and the counts

Pools (`<pool>`): line (horizontal dialogue and narration), name (speaker names drawn in a nameplate), ruby (furigana
readings as lines of their own), vertical (tategaki columns turned 90 degrees anticlockwise, as VOCR5-V reads them),
vreflow (those columns re-flowed left to right from their cell boxes, the label unchanged), furimain (lines with
furigana cut at box_main, the main text alone), and line_x2 / name_x2 (PSP lines cut from the page upscaled 2x
bilinear: the Adrenaline view Rubi reads on a Vita). Splits (`<split>`): train, guard, hval (hval lines that are not
duplicates of a train line and are their cluster's first hval copy: the screening set), hval_fold (the hval lines a
final model adds: cluster representatives), test. Lines that fail the vocab policy go to drop/oov.

Crops are stored as the console hands them to the recognizer: integer luma (synth/common.to_gray), dark text
inverted (synth/common.line_polarity; meta `inverted`), at native size (train2 resizes to height 32).

Capture runs (pools v2, docs/vocr6-data-v2.md). A title can have several capture runs: one folder per run, or a
later run uploaded into the earlier run's folder, whose files the bucket then keeps only as object versions
(recovered read-only into MIRROR/versions/<platform>/<folder>/<stamp>/). title_runs() takes every title folder
(<platform>/<folder>/metadata.json, as v1 did) and every recovered run that the mirror's index.json lists, in
`packaged` order. One title = all its runs: run 0 keeps its page ids, run r >= 1's page NNNNNN becomes
r * 100000 + NNNNNN (six digits), its record ids TITLE:PAGE:kindN follow, and its rows keep their own as src_id /
src_page. Each run is split as a title of its own (run 0 with v1's seed, so a v1 title's split is v1's), and a
page of another run that repeats a held-out page goes to guard (cross_run_guard). A one-run mirror (v1's) imports
exactly as v1 did: `lines --pool-tag hooked-v1` from the frozen v1 root rebuilds v1's pool files byte for byte.

Everything read or written here is the owner's private, copyrighted game data: it stays under $HERD and is never
committed; `manifests` writes ids, hashes, geometry and codes only.
"""
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "synth", ROOT / "teacher"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from common import line_polarity, to_gray  # noqa: E402

SCHEMA = "vocr6-hooked/2"                             # /2: titles of several capture runs (title_runs)
POOL_TAG = "hooked-v2"                                # meta `pool` of every row (v1's: hooked-v1, --pool-tag)
RUN_STRIDE = 100000                                   # run r >= 1's page NNNNNN is r * RUN_STRIDE + NNNNNN
CROSS_MIN = 6                                         # cross_run_guard: page texts / line labels this long or longer
VOCAB = ROOT / "data" / "vocab_vn_5631.txt"           # the Normal / Large vocab: its 5,142 kanji
VOCAB_S = ROOT / "data" / "vocab_4987_proof.txt"      # Small's (lines outside it train its <unk> class)
UNK = "〓"
SPLIT_RANK = {"train": 0, "guard": 1, "hval": 2, "test": 3}
MIN_SIDE = 8                                          # import_native's geometry floor
NEAR_SIG_H = 16                                       # near-duplicate signature height
NEAR_MAD = 8.0                                        # near duplicate: signature mean abs. difference <= this
NEAR_SHAPE = 2                                        # ... between crops whose height and width differ <= 2 px
REFLOW_GAP = 2                                        # px between re-flowed cells
UNTRUSTED = ("conflict", "empty_label", "box_outside_page", "label_doubt")   # a page's text then is in doubt
DERIVED = {"line_x2": "line", "name_x2": "name", "vreflow": "vertical", "furimain": "line"}
OPENING, CLOSING = "「『（〔【《〈［｛", "」』）〕】》〉］｝"
TALL_BAND = 0.9          # a "furigana" band above box_main at least this x the main text's height is a text line


def tall_band(rec):
    """A furigana line whose band above box_main is as tall as a text line (the audit: PCSG00807's five, 1.4-1.6x,
    hold a short dialogue line the label lacks; real ruby bands are 0.19-0.63x)."""
    if not (rec.get("furigana") and rec.get("box_main")):
        return False
    band = rec["box_main"][1] - rec["box"][1]
    return band >= TALL_BAND * rec["box_main"][3]


def label_doubt(rec, label):
    """An image_rows label whose first character the crop may not hold (the audit, docs/vocr6-data.md): the row's
    count needed a lead cell outside its cited cells (`uncited_lead_cells`) and the label opens with a bracket (the
    bracket is drawn left of the row's box), or the label opens with a closing bracket (it ends the row above).
    glyph_cells rows are the renderer's own cells: a line of a lone 」 there is what the game drew."""
    if rec.get("source") != "image_rows":
        return False
    if label[:1] in CLOSING:
        return True
    lead = ((rec.get("checks") or {}).get("count") or {}).get("uncited_lead_cells")
    return bool(lead and label[:1] in OPENING)


# pools v2's label audit (b2-ingest2, docs/vocr6-data-v2.md); `lines --label-rules v1` leaves all three out (v1's
# rebuild)
REMAPPED = {   # hook codepoints these titles draw as other glyphs: such a record's label is not what is drawn
    "NPJH50001": "㊤㊥㊦",            # drawn as ―
    "NPJH50002": "㊨",               # drawn as 鬼
    "NPJH50382": "├┤≪≫",             # drawn as ―
    "NPJH50621": "├┤",               # drawn as ―
    "NPJH50882": "ⅥⅧ⑮⑯⑰⑱⑲",       # Ⅵ a dakuten, Ⅷ ―, ⑮-⑲ ～
    "PCSG00476": "ゞ",               # drawn as a !? ligature
}
UNDRAWN = {"PCSG00122": "^"}       # hook codes these titles never draw (each takes a cell of its own)
DOUBLED = 0.85                     # doubled cells: consecutive cells overlapping this much along and across the line
AUDITED_WRONG = {"PCSG00367:000562:line2"}   # the audit's single wrong image_rows label: ！ drawn as ／
LABEL_DROPS = ("undrawn_shifted", "remapped", "doubled_cells", "audited_wrong")   # -> drop label_doubt


def remapped(tid, label):
    return any(c in label for c in REMAPPED.get(tid, ""))


def cell_overlap(a, b, vertical):
    """min over the two axes of the overlap of two cells [x0, y0, x1, y1], as a share of the smaller one."""
    out = []
    for i0, i1 in ((1, 3), (0, 2)) if vertical else ((0, 2), (1, 3)):
        w = min(a[i1] - a[i0], b[i1] - b[i0])
        out.append((min(a[i1], b[i1]) - max(a[i0], b[i0])) / w if w > 0 else 0.0)
    return min(out)


def doubled_cells(rec):
    """A glyph_cells record of which at least half the cells coincide with the next one (DOUBLED): one glyph drawn
    twice under two atlas keys, so a text twice as long as what is drawn matched the cell count (the audit:
    PCSG00926, a ？？？ nameplate labelled with the line below it)."""
    cb = rec.get("cell_boxes") or []
    if rec.get("source") != "glyph_cells" or len(cb) < 2:
        return False
    n = sum(1 for i in range(len(cb) - 1) if cell_overlap(cb[i], cb[i + 1], rec.get("vertical")) >= DOUBLED)
    return n >= len(cb) // 2


def cell_inked(con, t, box, shrink=0.25):
    """Whether a cell holds ink: a pixel of the page's 3x3 contrast `con` at or above `t` in the cell's central part
    (the cell shrunk by `shrink` of its size on every side, so a neighbouring glyph's overlap does not count)."""
    x0, y0, x1, y1 = box
    dx, dy = (x1 - x0) * shrink, (y1 - y0) * shrink
    xa, ya = max(0, int(math.floor(x0 + dx))), max(0, int(math.floor(y0 + dy)))
    xb, yb = int(math.ceil(x1 - dx)), int(math.ceil(y1 - dy))
    sub = con[ya:yb, xa:xb]
    return bool(sub.size and (sub >= t).any())


def undrawn_check(rec, grey, codes):
    """For a record with an undrawn code: (record without the code cells, its label) when every code cell is blank
    and every other cell inked (the code took a blank cell of its own, so the rest is aligned), else None (the
    assignment is shifted: a drawn glyph sits under a code). The ink threshold is GRAD's: max(40, 2/5 of the 99th
    percentile of the line box's 3x3 contrast)."""
    from common import contrast3
    cells, chars = rec.get("cell_boxes") or [], rec.get("cell_chars") or []
    if len(cells) != len(chars) or not cells:
        return None
    x, y, w, h = (int(v) for v in rec["box"])
    con = contrast3(grey)
    sub = con[max(0, y):y + h, max(0, x):x + w]
    t = max(40, 2 * int(np.percentile(sub, 99)) // 5) if sub.size else 255
    keep = []
    for c, b in zip(chars, cells):
        ink = cell_inked(con, t, b)
        if c in codes:
            if ink:
                return None
        elif c.strip() and not ink:
            return None
        else:
            keep.append((c, b))
    text = "".join(c for c in rec["text"] if c not in codes)
    out = dict(rec, text=text, cell_chars=[c for c, _ in keep], cell_boxes=[b for _, b in keep],
               cells=len(keep), undrawn_removed=sum(1 for c in chars if c in codes))
    return out, canon(text)


# ----------------------------------------------------------------------------- small helpers

def sha256_file(path, bufsize=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(bufsize), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def pixel_sha(g):
    """data/import_native.py's crop identity: the shape and the bytes."""
    return hashlib.sha256(np.asarray(g.shape, "<i4").tobytes() + np.ascontiguousarray(g).tobytes()).hexdigest()


def zstd_lines(path):
    """The JSON records of a .jsonl.zst file (the zstd binary; no Python zstd module in the pinned envs)."""
    out = subprocess.run(["zstd", "-dc", str(path)], check=True, capture_output=True).stdout
    return [json.loads(line) for line in out.decode("utf-8").splitlines() if line.strip()]


def write_json(path, doc, indent=1):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=indent) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
    tmp.replace(path)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def load_vocab_set(path):
    chars = [c for c in Path(path).read_text(encoding="utf-8").split("\n") if c]
    return {c for c in chars if c != UNK}


def canon(text):
    import vote
    return vote.canon(text)


# ----------------------------------------------------------------------------- the mirror

def index_key(doc):
    """What identifies a run between the mirror's index.json (an entry per run) and a run's metadata.json."""
    return (doc["title_id"], doc["pages"], doc["characters"], (doc.get("lines") or {}).get("verified_lines"))


def title_runs(root):
    """{title id: [run folder, ...]} in capture order (metadata `packaged`). The runs: every title folder
    (<platform>/<TITLE_ID> <title>/ with a metadata.json, as v1 took them) and every run recovered from the bucket's
    object versions (versions/<platform>/<folder>/<stamp>/) that the mirror's index.json lists (an entry with its
    title id, pages, characters and verified lines). A recovered run index.json does not list is a superseded export
    and is left out. Two runs of one title must agree on platform, resolution and split (load_title_runs)."""
    root = Path(root)
    runs = defaultdict(list)
    for m in sorted(root.glob("*/*/metadata.json")):
        doc = json.loads(m.read_text(encoding="utf-8"))
        runs[doc["title_id"]].append((doc.get("packaged") or "", str(m.parent), m.parent))
    versions = sorted(root.glob("versions/*/*/*/metadata.json"))
    if versions:
        idx = root / "index.json"
        listed = Counter(index_key(e) for e in json.loads(idx.read_text(encoding="utf-8"))["titles"]) \
            if idx.exists() else Counter()
        for m in versions:
            doc = json.loads(m.read_text(encoding="utf-8"))
            if listed.get(index_key(doc)):
                runs[doc["title_id"]].append((doc.get("packaged") or "", str(m.parent), m.parent))
    return {tid: [f for _, _, f in sorted(rs)] for tid, rs in sorted(runs.items())}


def load_title(folder):
    folder = Path(folder)
    meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    pages = zstd_lines(folder / "pages.jsonl.zst") if (folder / "pages.jsonl.zst").exists() else []
    lines = zstd_lines(folder / "lines.jsonl.zst") if (folder / "lines.jsonl.zst").exists() else []
    return meta, pages, lines


def page_id(rec):
    """A page record's or line record's page number (pages/NNNNNN.png -> NNNNNN)."""
    return Path(rec["image"]).stem


def moved_page(pid, run):
    """Run `run`'s page id in its title: run 0 keeps its own, run r >= 1's NNNNNN is r * RUN_STRIDE + NNNNNN."""
    if run == 0:
        return pid
    n = int(pid)
    if not (0 < run < 10 and 0 <= n < RUN_STRIDE and len(pid) == 6):
        raise ValueError(f"run {run} page {pid!r}: outside the page-id scheme")
    return f"{run * RUN_STRIDE + n:06d}"


def move_record(rec, run, tid):
    """A page or line record of run `run` >= 1 with its page id moved (moved_page): image, page and the line id
    TITLE:PAGE:kindN follow; src_image / src_page / src_id keep the run's own."""
    old = page_id(rec)
    new = moved_page(old, run)
    out = dict(rec, image=f"pages/{new}.png", src_image=rec["image"], src_page=old, run_index=run)
    if "page" in rec:
        out["page"] = new
    if rec.get("id") == old:                         # a page record's id is its page number
        out["id"], out["src_id"] = new, old
    elif "id" in rec and rec["id"].count(":") == 2 and rec["id"].split(":")[0] == tid:
        t, p, kind = rec["id"].split(":")
        if p != old:
            raise ValueError(f"{rec['id']}: its page part is not its image's page {old}")
        out["id"], out["src_id"] = f"{t}:{new}:{kind}", rec["id"]
    return out


def run_info(r, folder, meta, root=None):
    f = Path(folder)
    return {"index": r, "run": meta.get("run"), "folder": str(f.relative_to(root)) if root else f.name,
            "packaged": meta.get("packaged"), "pages": meta.get("pages"),
            "page_offset": r * RUN_STRIDE, "metadata_sha256": sha256_file(f / "metadata.json")}


def load_title_runs(folders, root=None):
    """(meta, pages, lines, runs) of a title from its run folders (title_runs order): run 0's metadata, every run's
    page and line records with run r >= 1's moved (move_record), and [run_info] of every run. The runs must agree
    on title id, platform, resolution and split."""
    out_pages, out_lines, runs, meta0 = [], [], [], None
    for r, folder in enumerate(folders):
        meta, pages, lines = load_title(folder)
        if meta0 is None:
            meta0 = meta
        else:
            for k in ("title_id", "platform", "resolution", "split"):
                if meta.get(k) != meta0.get(k):
                    raise ValueError(f"{meta0['title_id']}: run {r} ({folder}) has {k} {meta.get(k)!r}, run 0 "
                                     f"{meta0.get(k)!r}")
        tid = meta0["title_id"]
        if r:
            pages = [move_record(p, r, tid) for p in pages]
            lines = [move_record(x, r, tid) for x in lines]
        out_pages += pages
        out_lines += lines
        runs.append(run_info(r, folder, meta, root))
    ids = [page_id(p) for p in out_pages]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{meta0['title_id']}: page ids repeat across its runs")
    return meta0, out_pages, out_lines, runs


def title_vn(meta, vndb_map=None):
    """(vn, vns) of a title: its metadata's VNDB id (a list's first), else the map's (vndb_map.json `titles`)."""
    v = (meta.get("vndb") or {}).get("vn")
    vns = [x for x in (v if isinstance(v, list) else [v]) if x]
    if not vns and vndb_map and meta["title_id"] in vndb_map:
        vns = list(vndb_map[meta["title_id"]]["vns"])
    return (vns[0] if vns else None), vns


# ----------------------------------------------------------------------------- extract

def extract_title(args):
    """Every run's pages.tar.zst of one title into OUT/<title id>/pages/, run r >= 1's pages at their moved ids
    (tar --transform: pages/0NNNNN.png -> pages/rNNNNN.png); every PNG's sha256 checked against its page record and
    every line record that cites it."""
    folders, out = [Path(f) for f in args[0]], Path(args[1])
    meta0, pages, lines, runs = load_title_runs(folders)
    tid = meta0["title_id"]
    dst = out / tid
    want = {page_id(p): p["image_sha256"] for p in pages}
    for r in lines:
        if want.get(page_id(r), r["image_sha256"]) != r["image_sha256"]:
            raise ValueError(f"{tid}: line {r['id']} cites another image than its page record")
        want.setdefault(page_id(r), r["image_sha256"])
    done = dst / ".extracted.json"
    have = json.loads(done.read_text()) if done.exists() else {}
    tars = []
    for r, folder in enumerate(folders):
        meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        tars.append(meta["files"]["pages.tar.zst"]["sha256"])
        prev = (have.get("runs") or [{}] * len(folders))
        if len(prev) == len(folders) and prev[r].get("pages_tar_sha256") == tars[-1]:
            continue
        dst.mkdir(parents=True, exist_ok=True)
        tar = folder / "pages.tar.zst"
        cmd = ["tar", "-x", "-C", str(dst), "--wildcards", "pages/*.png"]
        if r:
            cmd += [f"--transform=s,^pages/0,pages/{r},"]       # moved_page: r * 100000 + NNNNN for NNNNN < 100000
        p1 = subprocess.Popen(["zstd", "-dc", str(tar)], stdout=subprocess.PIPE)
        subprocess.run(cmd, stdin=p1.stdout, check=True)
        p1.stdout.close()
        if p1.wait() != 0:
            raise RuntimeError(f"{tid}: zstd failed on {tar}")
    bad = [p for p, h in sorted(want.items()) if not (dst / "pages" / f"{p}.png").exists()
           or sha256_file(dst / "pages" / f"{p}.png") != h]
    if bad:
        raise ValueError(f"{tid}: {len(bad)} page images missing or differing from their sha256, e.g. {bad[:3]}")
    extra = sorted(set(p.stem for p in (dst / "pages").glob("*.png")) - set(want))
    write_json(done, {"title_id": tid, "runs": [{"pages_tar_sha256": t, "page_offset": i * RUN_STRIDE}
                                                for i, t in enumerate(tars)],
                      "pages": len(want), "extra_pngs": extra}, indent=None)
    return tid, len(want), len(extra)


def cmd_extract(a):
    runs = title_runs(a.root)
    jobs = [([str(f) for f in runs[t]], a.out) for t in sorted(runs) if not a.titles or t in a.titles.split(",")]
    with ProcessPoolExecutor(a.procs) as ex:
        for tid, n, extra in ex.map(extract_title, jobs):
            print(f"{tid}: {n} pages, sha256 ok, {extra} extra png", flush=True)


# ----------------------------------------------------------------------------- split

def title_seed(seed, tid):
    return int(hashlib.sha256(f"{seed}:{tid}".encode()).hexdigest()[:8], 16)


def split_title(page_ids, title_split, tid, share=0.04, guard=15, seed=20261026, min_pages=200, block_max=40):
    """{page: split} and the hval blocks of one title (page ids sorted numerically = capture order).

    test title: every page test. train title with fewer than `min_pages` pages: every page train. Otherwise
    H = round(share * pages) hval pages in k = ceil(H / block_max) contiguous blocks (lengths differing by at most
    one); the title is cut into k equal segments and block j sits in segment j at an offset drawn from
    default_rng(sha256(seed:title id)) such that its guard band (`guard` pages either side, clipped at the ends of
    the title) stays inside the segment. Guard pages are "guard"."""
    pages = sorted(page_ids, key=int)
    n = len(pages)
    if title_split == "test":
        return {p: "test" for p in pages}, []
    if title_split != "train":
        raise ValueError(f"{tid}: unknown split {title_split!r}")
    out = {p: "train" for p in pages}
    if n < min_pages:
        return out, []
    H = max(1, round(share * n))
    k = max(1, math.ceil(H / block_max))
    sizes = [H // k + (j < H % k) for j in range(k)]
    rng = np.random.default_rng(title_seed(seed, tid))
    blocks = []
    for j, b in enumerate(sizes):
        s0, s1 = j * n // k, (j + 1) * n // k
        lo = s0 + (guard if j > 0 else 0)
        hi = s1 - b - (guard if j < k - 1 else 0)
        if hi < lo:
            raise ValueError(f"{tid}: segment {j} too short for a {b}-page block with its guard bands")
        start = int(rng.integers(lo, hi + 1))
        blocks.append((start, start + b))
    for start, end in blocks:
        for i in range(max(0, start - guard), min(n, end + guard)):
            if out[pages[i]] == "train":
                out[pages[i]] = "guard"
        for i in range(start, end):
            out[pages[i]] = "hval"
    return out, [[pages[s], pages[e - 1]] for s, e in blocks]


def cross_run_guard(run_pages, assign, texts, labels, guard=15, min_len=CROSS_MIN):
    """Pages of a title's other runs that repeat a held-out page become guard (in place), so a near-identical frame
    of an hval page never trains while hval is screened. run_pages: [[page ids of run r]] in capture order; texts:
    {page: its page text, whitespace removed}; labels: {page: its verified horizontal and vertical line labels}.
    A page of run B matches a page P of run A != B when both have the same page text (at least `min_len`
    characters), or, for P in hval, when they share a line label of at least `min_len` characters. Every train page
    that matches a held-out page (hval or guard) is guard, and so are the `guard` pages either side of a page that
    matches an hval page, within its run. Returns the pages it moved, sorted."""
    run_of = {p: r for r, ids in enumerate(run_pages) for p in ids}
    by_text, by_label = defaultdict(set), defaultdict(set)
    for p, t in texts.items():
        if len(t) >= min_len:
            by_text[t].add(p)
    for p, ls in labels.items():
        for lab in ls:
            if len(lab) >= min_len:
                by_label[lab].add(p)
    hit, hit_hval = set(), set()
    for p, s in assign.items():
        if s not in ("hval", "guard"):
            continue
        cand = set(by_text.get(texts.get(p, ""), ())) if len(texts.get(p, "")) >= min_len else set()
        if s == "hval":
            for lab in labels.get(p, ()):
                cand |= by_label.get(lab, set())
        cand = {q for q in cand if run_of[q] != run_of[p]}
        hit |= cand
        if s == "hval":
            hit_hval |= cand
    for r, ids in enumerate(run_pages):
        pos = {p: i for i, p in enumerate(ids)}
        for q in hit_hval:
            if run_of[q] == r:
                i = pos[q]
                hit.update(ids[max(0, i - guard):i + guard + 1])
    moved = sorted(q for q in hit if assign[q] == "train")
    for q in moved:
        assign[q] = "guard"
    return moved


def split_runs(run_pages, title_split, tid, texts, labels, share=0.04, guard=15, seed=20261026, min_pages=200,
               block_max=40):
    """{page: split}, the hval blocks and the cross-run guard pages of a title of one or more runs: each run split
    by split_title as a title of its own (run 0 seeded by the title id, as v1; run r >= 1 by "<title id>:run<r>"),
    then cross_run_guard over the runs. One run: split_title's result exactly."""
    assign, blocks = {}, []
    for r, ids in enumerate(run_pages):
        a, b = split_title(ids, title_split, tid if r == 0 else f"{tid}:run{r}", share, guard, seed, min_pages,
                           block_max)
        assign.update(a)
        blocks += b
    cross = cross_run_guard(run_pages, assign, texts, labels, guard) if len(run_pages) > 1 else []
    return assign, blocks, cross


def page_texts_labels(pages, lines):
    """({page: page text without whitespace}, {page: {verified line labels}}) of a title (cross_run_guard)."""
    texts = {page_id(p): "".join((p.get("text_clean", p.get("text")) or "").split()) for p in pages}
    labels = defaultdict(set)
    for r in lines:
        if r["kind"] == "line" and r.get("text"):
            labels[page_id(r)].add(r["text"])
    return texts, dict(labels)


def cmd_split(a):
    runs = title_runs(a.root)
    doc = {"schema": SCHEMA, "kind": "vocr6-hooked-splits", "rule": split_title.__doc__.split("\n\n")[1].strip(),
           "runs_rule": split_runs.__doc__.strip() + "\n" + cross_run_guard.__doc__.strip(),
           "params": {"share": a.share, "guard": a.guard, "seed": a.seed, "min_pages": a.min_pages,
                      "block_max": a.block_max, "run_stride": RUN_STRIDE, "cross_min": CROSS_MIN},
           "titles": {}}
    for tid, folders in sorted(runs.items()):
        meta, pages, lines, rinfo = load_title_runs(folders, Path(a.root))
        ids = [page_id(p) for p in pages]
        run_pages = [[page_id(p) for p in pages if p.get("run_index", 0) == r] for r in range(len(folders))]
        texts, labels = page_texts_labels(pages, lines)
        assign, blocks, cross = split_runs(run_pages, meta["split"], tid, texts, labels, a.share, a.guard, a.seed,
                                           a.min_pages, a.block_max)
        c = Counter(assign.values())
        doc["titles"][tid] = {"title_split": meta["split"], "platform": meta["platform"], "pages": len(ids),
                              "hval_blocks": blocks, "counts": dict(sorted(c.items())),
                              "pages_sha256": sha256_text("\n".join(f"{p}:{h['image_sha256']}" for p, h in
                                                                    sorted(zip(ids, pages)))),
                              "runs": rinfo, "cross_run_guard": cross, "assign": assign}
        print(f"{tid} {meta['split']:5} {len(ids):5} pages {dict(c)} blocks {blocks}"
              + (f" runs {len(folders)} cross-run guard {len(cross)}" if len(folders) > 1 else ""), flush=True)
    tot = Counter()
    for t in doc["titles"].values():
        tot.update(t["counts"])
    doc["counts"] = dict(sorted(tot.items()))
    write_json(a.out, doc)
    print(json.dumps(doc["counts"]))


# ----------------------------------------------------------------------------- crops

def crop_rect(box, pad, W, H):
    x, y, w, h = box
    return max(0, x - pad), max(0, y - pad), min(W, x + w + pad), min(H, y + h + pad)


def chars_in(rec, label, x0, y0, transform=None):
    """Per-character boxes [[c, x0, y0, x1, y1]] relative to the crop, when the record's cells are its label."""
    cells, chars = rec.get("cell_boxes") or [], rec.get("cell_chars") or []
    if len(cells) != len(chars) or "".join(chars) != label:
        return None
    out = []
    for c, (a, b, cc, d) in zip(chars, cells):
        bx = [a - x0, b - y0, cc - x0, d - y0]
        if transform is not None:
            bx = transform(bx)
        out.append([c] + [round(float(v), 2) for v in bx])
    return out


def reflow(grey, rec, label, pad):
    """A vertical column re-flowed left to right: each cell of cell_boxes (cut between neighbouring cells at the
    midpoint of their overlap or gap) laid side by side with REFLOW_GAP px of the column's background between them.
    Returns (crop, chars) or (None, None) when the record's cells are not its label's characters."""
    cells = rec.get("cell_boxes") or []
    if not cells or len(cells) != len(label):
        return None, None
    H, W = grey.shape
    x, y, w, h = rec["box"]
    xa, xb = max(0, x - pad), min(W, x + w + pad)
    ys = []
    for i, (_, c_y0, _, c_y1) in enumerate(cells):
        top = c_y0 if i == 0 else (c_y0 + cells[i - 1][3]) / 2
        bot = c_y1 if i == len(cells) - 1 else (c_y1 + cells[i + 1][1]) / 2
        ys.append((max(0, int(math.floor(top))), min(H, int(math.ceil(bot)))))
    parts = [grey[a:b, xa:xb] for a, b in ys if b > a]
    if len(parts) != len(cells):
        return None, None
    col = grey[max(0, y - pad):min(H, y + h + pad), xa:xb]
    bg = int(np.median(np.concatenate([col[0], col[-1], col[:, 0], col[:, -1]])))
    ph = max(p.shape[0] for p in parts)
    out, chars, cx = [], [], 0
    for c, cell in zip(label, parts):                       # each glyph upright, as it stands on the page
        if cell.shape[0] < ph:                               # centre a shorter cell on the strip's height
            t = (ph - cell.shape[0]) // 2
            cell = np.pad(cell, ((t, ph - cell.shape[0] - t), (0, 0)), constant_values=bg)
        if out:
            out.append(np.full((ph, REFLOW_GAP), bg, np.uint8))
            cx += REFLOW_GAP
        out.append(cell)
        chars.append([c, float(cx), 0.0, float(cx + cell.shape[1]), float(ph)])
        cx += cell.shape[1]
    return np.ascontiguousarray(np.concatenate(out, axis=1)), chars


def turn_box(width):
    """Box transform of np.rot90(k=1) on a crop `width` px wide: (x, y) -> (y, width - x)."""
    return lambda b: [b[1], width - b[2], b[3], width - b[0]]


def polarity_store(g):
    dark = int(line_polarity(g))
    return (255 - g if dark else g), dark


def cut_variants(rec, grey, grey2, label):
    """[(pool, crop, crop rect, extra meta)] of one line record (before the filters and the polarity rule)."""
    H, W = grey.shape
    pad = int(rec["pad"])
    tall = tall_band(rec)
    box = [int(v) for v in (rec["box_main"] if tall else rec["box"])]   # a tall band is cut off: the main line
    x0, y0, x1, y1 = crop_rect(box, pad, W, H)
    g = np.ascontiguousarray(grey[y0:y1, x0:x1])
    out = []
    if rec.get("vertical"):
        turned = np.ascontiguousarray(np.rot90(g, 1))
        out.append(("vertical", turned, [x0, y0, x1, y1],
                    {"chars": chars_in(rec, label, x0, y0, turn_box(x1 - x0)), "turned": "rot90_ccw"}))
        rf, rchars = reflow(grey, rec, label, pad)
        if rf is not None:
            out.append(("vreflow", rf, [x0, y0, x1, y1],
                        {"chars": [[c, *(round(v, 2) for v in b)] for c, *b in rchars], "reflow": True}))
        return out
    pool = {"line": "line", "name": "name", "ruby": "ruby"}[rec["kind"]]
    chars = chars_in(rec, label, x0, y0)
    out.append((pool, g, [x0, y0, x1, y1], {"chars": chars, **({"tall_band": True} if tall else {})}))
    if rec.get("furigana") and rec.get("box_main") and not tall:
        bm = [int(v) for v in rec["box_main"]]
        m0, n0, m1, n1 = crop_rect(bm, pad, W, H)
        out.append(("furimain", np.ascontiguousarray(grey[n0:n1, m0:m1]), [m0, n0, m1, n1],
                    {"chars": chars_in(rec, label, m0, n0)}))
    if grey2 is not None and pool in ("line", "name"):
        # the page upscaled 2x: the crop rectangle doubles exactly (max(0, 2x - 2p) = 2 max(0, x - p))
        out.append((pool + "_x2", np.ascontiguousarray(grey2[2 * y0:2 * y1, 2 * x0:2 * x1]),
                    [2 * x0, 2 * y0, 2 * x1, 2 * y1],
                    {"chars": None if chars is None else [[c, *(2 * v for v in b)] for c, *b in chars],
                     "scale": 2}))
    return out


def ctc_feasible(shape, label):
    from render import crop_ctc_feasible
    return crop_ctc_feasible(shape, label)


# ----------------------------------------------------------------------------- duplicates

def signature(g):
    h, w = g.shape
    sw = max(4, min(512, round(NEAR_SIG_H * w / h)))
    return cv2.resize(g, (sw, NEAR_SIG_H), interpolation=cv2.INTER_AREA).astype(np.float32)


def near(a, b, mad=NEAR_MAD):
    """Two crops of one label are near duplicates: heights and widths within NEAR_SHAPE px and the mean absolute
    difference of their signatures (height 16, the first crop's width) at most `mad` grey levels."""
    (ha, wa), (hb, wb) = a.shape, b.shape
    if abs(ha - hb) > NEAR_SHAPE or abs(wa - wb) > NEAR_SHAPE:
        return False
    sa = signature(a)
    sb = cv2.resize(b, (sa.shape[1], NEAR_SIG_H), interpolation=cv2.INTER_AREA).astype(np.float32)
    return float(np.abs(sa - sb).mean()) <= mad


def clusters(items, mad=NEAR_MAD):
    """items: [(key, label, crop, split, order)] of one title and pool. Returns {key: (rep key, dup)} with dup None
    for a representative, "exact" for identical pixels and label, "near" for a near duplicate (near()). Members are
    visited by (split rank train < guard < hval < test, order), so a cluster's representative is its first train
    copy when it has one. Identical pixels under different labels are a conflict: every member gets
    (None, "conflict")."""
    by_px = defaultdict(list)
    for it in items:
        by_px[pixel_sha(it[2])].append(it)
    conflict = set()
    for group in by_px.values():
        if len({it[1] for it in group}) > 1:
            conflict.update(it[0] for it in group)
    out = {k: (None, "conflict") for k in conflict}
    by_label = defaultdict(list)
    for it in items:
        if it[0] not in conflict:
            by_label[it[1]].append(it)
    for group in by_label.values():
        group.sort(key=lambda it: (SPLIT_RANK[it[3]], it[4]))
        reps = []                                  # (key, crop, pixel sha)
        for key, _, g, _, _ in group:
            px = pixel_sha(g)
            hit = None
            for rk, rg, rpx in reps:
                if rpx == px:
                    hit = (rk, "exact")
                    break
            if hit is None:
                for rk, rg, rpx in reps:
                    if near(rg, g, mad):
                        hit = (rk, "near")
                        break
            if hit is None:
                reps.append((key, g, px))
                out[key] = (key, None)
            else:
                out[key] = hit
    return out


# ----------------------------------------------------------------------------- lines

def page_grey(pages_dir, tid, page, sha_want, cache):
    if page not in cache:
        p = Path(pages_dir) / tid / "pages" / f"{page}.png"
        data = p.read_bytes()
        if hashlib.sha256(data).hexdigest() != sha_want:
            raise ValueError(f"{tid}:{page}: the page image differs from its sha256")
        bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        rgb = np.ascontiguousarray(bgr[..., ::-1])
        cache.clear()                              # records come page by page: keep one page
        cache[page] = (to_gray(rgb), rgb)
    return cache[page]


def import_title(args):
    """One title's crops, filters, clusters and pool files. Returns its decisions and counts. args: (run folders or
    one folder, pages dir, {page: split}, out dir, vocab, Small's vocab, near-duplicate MAD[, pool tag, vndb map])."""
    folder, pages_dir, assign, out, vocab, vocab_s, mad = args[:7]
    pool_tag = args[7] if len(args) > 7 else POOL_TAG
    vndb_map = args[8] if len(args) > 8 else None
    folders = list(folder) if isinstance(folder, (list, tuple)) else [folder]
    meta, _, lines, runs = load_title_runs(folders)
    tid, platform = meta["title_id"], meta["platform"]
    vn, vns = title_vn(meta, vndb_map)
    label_rules = args[9] if len(args) > 9 else "v2"
    cache, rows, items, crops = {}, [], [], {}
    counts = Counter()
    for order, rec in enumerate(lines):
        page = page_id(rec)
        split = assign[page]
        grey, rgb = page_grey(pages_dir, tid, page, rec["image_sha256"], cache)
        H, W = grey.shape
        if [W, H] != list(rec.get("dimensions") or [W, H]):
            raise ValueError(f"{rec['id']}: dimensions {rec.get('dimensions')} but the page is {W}x{H}")
        label = canon(rec["text"] or "")
        rule = None                    # the v2 label rule that decided this record (label_rule)
        if label_rules == "v2":
            codes = UNDRAWN.get(tid, "")
            if codes and any(c in (rec["text"] or "") for c in codes):
                chk = undrawn_check(rec, grey, codes)
                if chk is None:
                    rule = "undrawn_shifted"
                else:
                    rec, label = chk
                    rule = "undrawn_stripped"
            if rule != "undrawn_shifted":
                if rec["id"] in AUDITED_WRONG:
                    rule = "audited_wrong"
                elif remapped(tid, rec["text"] or "") or remapped(tid, label):
                    rule = "remapped"
                elif doubled_cells(rec):
                    rule = "doubled_cells"
        row = {"id": rec["id"], "page": page, "split": split, "kind": rec["kind"], "vertical": bool(rec["vertical"]),
               "source": rec["source"], "text": label, "canon_changed": label != rec["text"], "pools": {}}
        if rec.get("run_index"):                                   # a later capture run's record (load_title_runs)
            row["src_id"] = rec.get("src_id")
        if rule:
            row["label_rule"] = rule
            counts[f"label_rule:{rule}"] += 1
        if rule == "undrawn_stripped":
            row["undrawn_removed"] = rec["undrawn_removed"]
        counts["records"] += 1
        counts[f"records:{rec['kind']}{':vertical' if rec['vertical'] else ''}"] += 1
        counts["canon_changed"] += row["canon_changed"]
        x, y, w, h = (int(v) for v in rec["box"])
        drop = None
        if not label or any(c.isspace() for c in label):
            drop = "empty_label"
        elif w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > W or y + h > H:
            drop = "box_outside_page"
        elif label_doubt(rec, label) or rule in LABEL_DROPS:
            drop = "label_doubt"
        elif any(c not in vocab for c in label):
            drop = "oov"
            row["oov"] = sorted({c for c in label if c not in vocab})
        row["s_vocab"] = all(c in vocab_s for c in label)
        grey2 = None
        if platform == "psp" and drop is None:
            key2 = ("x2", page)
            if key2 not in cache:
                cache[key2] = to_gray(cv2.resize(rgb, (2 * W, 2 * H), interpolation=cv2.INTER_LINEAR))
            grey2 = cache[key2]
        variants = cut_variants(rec, grey, grey2, label) if drop in (None, "oov") else []
        for pool, g, rect, extra in variants:
            geom_ok = min(g.shape) >= MIN_SIDE
            ctc_ok = bool(ctc_feasible(g.shape, label))
            vdrop = drop or (None if geom_ok else "geometry") or (None if ctc_ok else "ctc")
            stored, dark = polarity_store(g)
            key = f"{rec['id']}|{pool}"
            row["pools"][pool] = {"pool": pool, "drop": vdrop, "shape": list(stored.shape), "dark": dark,
                                  "geometry_ok": geom_ok, "ctc_ok": ctc_ok}
            counts[f"{pool}:{vdrop or 'kept'}"] += 1
            counts[f"{pool}:fails_geometry"] += not geom_ok      # independent of the other filters
            counts[f"{pool}:fails_ctc"] += not ctc_ok
            counts[f"{pool}:fails_vocab"] += drop == "oov"
            if vdrop in (None, "oov"):
                crops[key] = (stored, rect, dark, extra)
            if vdrop is None:
                items.append((key, label, stored, split, order))
        if not variants:
            row["drop"] = drop
            counts[f"record:{drop}"] += 1
        rows.append(row)
    # clusters per base pool; a derived pool's copy of a record follows its base copy's decision (its cluster's
    # representative is the representative's copy in the derived pool), so line and line_x2 (and vertical and
    # vreflow, line and furimain) keep and drop the same records. A derived copy whose base copy was dropped, or
    # whose representative has no derived copy, is its own representative.
    by_pool = defaultdict(list)
    for it in items:
        by_pool[it[0].split("|")[1]].append(it)
    cl = {}
    for pool, its in by_pool.items():
        if pool not in DERIVED:
            cl.update(clusters(its, mad))
    keys = {it[0] for it in items}
    for key, *_ in items:
        rid, pool = key.split("|")
        if pool not in DERIVED:
            continue
        rep, dup = cl.get(f"{rid}|{DERIVED[pool]}", (key, None))
        if rep is None:
            cl[key] = (None, dup)
            continue
        rep_key = f"{rep.split('|')[0]}|{pool}"
        cl[key] = (rep_key, dup) if rep_key in keys else (key, None)
    # per cluster: whether it has a train copy, and its first hval copy (the screening set's member)
    members = defaultdict(list)
    for key, label, g, split, order in items:
        rep, dup = cl[key]
        if rep is not None:
            members[rep].append((SPLIT_RANK[split], order, key, split))
    seen_train, first_hval = {}, set()
    for rep, ms in members.items():
        ms.sort()
        seen_train[rep] = any(m[3] == "train" for m in ms)
        hv = [m for m in ms if m[3] == "hval"]
        if hv:
            first_hval.add(hv[0][2])
    out_rows = defaultdict(list)
    by_id = {r["id"]: r for r in rows}
    for key, label, g, split, order in sorted(items, key=lambda it: it[4]):
        rid, pool = key.split("|")
        rep, dup = cl[key]
        entry = by_id[rid]["pools"][pool]
        entry.update(cluster=rep, dup=dup)
        targets = []
        if dup == "conflict":
            entry["drop"] = "conflict"
        elif split == "train":
            targets = ["train"] if dup is None else []
        elif split == "guard":
            targets = ["guard"] if dup is None else []
        elif split == "hval":
            if dup is None:
                targets.append("hval_fold")
            if key in first_hval and not seen_train[rep]:
                targets.append("hval")
            entry["seen_train"] = bool(seen_train[rep])
        elif split == "test":
            targets = ["test"] if dup is None else []
        entry["to"] = targets
        for t in targets:
            out_rows[(t, pool)].append((key, label))
            counts[f"to:{t}:{pool}"] += 1
        if not targets:
            counts[f"dup:{split}:{pool}:{entry.get('drop') or dup}"] += 1
    for key, (stored, rect, dark, extra) in crops.items():
        rid, pool = key.split("|")
        if by_id[rid]["pools"][pool]["drop"] == "oov":
            out_rows[("drop/oov", pool)].append((key, by_id[rid]["text"]))
    # write the pools
    files = {}
    recs = {r["id"]: r for r in lines}
    for (split, pool), keyed in sorted(out_rows.items()):
        data, labels, metas = [], [], []
        for key, label in keyed:
            rid = key.split("|")[0]
            rec, row = recs[rid], by_id[rid]
            stored, rect, dark, extra = crops[key]
            entry = row["pools"][pool]
            m = {"real": True, "pool": pool_tag, "id": rid, "title_id": tid, "vn": vn, "platform": platform,
                 "kind": platform, "target": platform, "screen": [int(v) for v in rec["dimensions"]],
                 "page": row["page"], "split": row["split"], "line_kind": rec["kind"], "name": rec["kind"] == "name",
                 "source": rec["source"], "source_check": rec["source_check"], "box": rec["box"],
                 "pad": rec["pad"], "crop_box": rect, "vertical": bool(rec["vertical"]),
                 "furigana": bool(rec.get("furigana")), "box_main": rec.get("box_main"), "dark": dark,
                 "inverted": bool(dark), "pol": "dark-on-light" if dark else "bright-on-dark",
                 "image_sha256": rec["image_sha256"], "pixel_sha256": pixel_sha(stored), "s_vocab": row["s_vocab"],
                 "cluster": entry.get("cluster"), "dup": entry.get("dup"), "seen_train": entry.get("seen_train"),
                 "hand_checked": False, "text_source": rec.get("text_source"), **extra}
            if len(vns) > 1:
                m["vns"] = vns
            if rec.get("run_index"):          # a later capture run's row: its run and its own record id and page
                m.update(run=runs[rec["run_index"]]["run"], src_id=rec.get("src_id"), src_page=rec.get("src_page"))
            data.append(stored)
            labels.append(label)
            metas.append(m)
        path = Path(out) / split / pool / f"{tid}.npz"
        write_npz(path, data, labels, metas)
        files[f"{split}/{pool}/{tid}.npz"] = {"rows": len(labels), "sha256": sha256_file(path),
                                             "bytes": path.stat().st_size}
    write_jsonl(Path(out) / "decisions" / f"{tid}.jsonl", rows)
    return tid, dict(counts), files


def write_npz(path, crops, labels, metas):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    shapes = np.asarray([g.shape for g in crops], np.int32).reshape(-1, 2)
    offsets = np.concatenate([[0], np.cumsum(shapes[:, 0] * shapes[:, 1])]).astype(np.int64)
    data = np.concatenate([g.ravel() for g in crops]) if crops else np.zeros(0, np.uint8)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, data=data.astype(np.uint8), offsets=offsets, shapes=shapes,
                        labels=np.asarray(json.dumps(labels, ensure_ascii=False)),
                        meta=np.asarray(json.dumps(metas, ensure_ascii=False)))
    tmp.replace(path)


def read_npz(path):
    """[(meta, label, crop)] of a pool file."""
    z = np.load(path)
    data, offs, shapes = z["data"], z["offsets"], z["shapes"]
    labels, metas = json.loads(str(z["labels"])), json.loads(str(z["meta"]))
    return [(m, lab, data[offs[i]:offs[i + 1]].reshape(int(shapes[i][0]), int(shapes[i][1])))
            for i, (lab, m) in enumerate(zip(labels, metas))]


def cmd_lines(a):
    runs = title_runs(a.root)
    splits = json.loads(Path(a.splits).read_text(encoding="utf-8"))
    vocab, vocab_s = load_vocab_set(VOCAB), load_vocab_set(VOCAB_S)
    vndb_map = json.loads(Path(a.vndb_map).read_text(encoding="utf-8"))["titles"] if a.vndb_map else None
    titles = [t for t in sorted(runs) if not a.titles or t in a.titles.split(",")]
    jobs = [([str(f) for f in runs[t]], a.pages, splits["titles"][t]["assign"], a.out, vocab, vocab_s, a.mad,
             a.pool_tag, vndb_map, a.label_rules) for t in titles]
    man_path = Path(a.out) / "manifest.json"
    man = json.loads(man_path.read_text()) if man_path.exists() and a.titles else {"titles": {}, "files": {}}
    with ProcessPoolExecutor(a.procs) as ex:
        for tid, counts, files in ex.map(import_title, jobs):
            man["files"] = {k: v for k, v in man["files"].items() if not k.endswith(f"/{tid}.npz")}
            man["files"].update(files)
            man["titles"][tid] = {"counts": dict(sorted(counts.items())),
                                  "decisions_sha256": sha256_file(Path(a.out) / "decisions" / f"{tid}.jsonl")}
            print(f"{tid}: {counts.get('records')} records; " + ", ".join(
                f"{k[3:]} {v}" for k, v in sorted(counts.items()) if k.startswith("to:")), flush=True)
    tot = Counter()
    for t in man["titles"].values():
        tot.update(t["counts"])
    man.update({"schema": SCHEMA, "kind": "vocr6-hooked-pools", "pool_tag": a.pool_tag,
                "importer_sha256": sha256_file(__file__), "splits_sha256": sha256_file(a.splits),
                "vndb_map_sha256": sha256_file(a.vndb_map) if a.vndb_map else None,
                **({"label_rules": {"rules": "v2", "remapped": REMAPPED, "undrawn": UNDRAWN, "doubled": DOUBLED,
                                    "audited_wrong": sorted(AUDITED_WRONG)}} if a.label_rules == "v2" else {}),
                "vocab": {"file": str(VOCAB.relative_to(ROOT)), "sha256": sha256_file(VOCAB)},
                "vocab_s": {"file": str(VOCAB_S.relative_to(ROOT)), "sha256": sha256_file(VOCAB_S)},
                "near_duplicate": {"signature_height": NEAR_SIG_H, "mad": a.mad, "shape_px": NEAR_SHAPE},
                "counts": dict(sorted(tot.items()))})
    man["files"] = dict(sorted(man["files"].items()))
    write_json(man_path, man)


# ----------------------------------------------------------------------------- pages

def page_boxes(prec):
    """The renderer's text boxes of a page record: [(index 1.., [x0, y0, x1, y1] floats, kind)]."""
    return [(i + 1, [float(v) for v in b["xyxy"]], b.get("kind")) for i, b in enumerate(prec.get("boxes") or [])]


def xywh_to_xyxy(b):
    return [b[0], b[1], b[0] + b[2], b[1] + b[3]]


def regions_of(prec, recs, platform, W, H):
    """Whole-dialogue regions of a page (eval/finder_tighten.py dialogue_units' rule: verified horizontal dialogue
    lines grouped by eval's vertical-gap rule, their union plus 12 px (Vita) / 6 px (PSP) and +4 / +8 variants,
    every other renderer box kept out: names, unverified boxes, vertical columns; a line's own furigana rows are
    part of its box). [{"variant", "region" [x0, y0, x1, y1], "lines": [line ids top to bottom]}]."""
    sys.path.insert(0, str(ROOT / "eval"))
    import finder_tighten as FT
    dl = [r for r in recs if r["kind"] == "line" and not r["vertical"]]
    if not dl:
        return []
    own = set()
    for r in dl:
        own.update(r.get("box_index") or [])
    ruby_idx = set()
    for r in recs:
        if r["kind"] == "ruby":
            ruby_idx.update(r.get("box_index") or [])
    others = []
    for i, xyxy, _ in page_boxes(prec):
        if i in own or i in ruby_idx:
            continue
        others.append([int(math.floor(xyxy[0])), int(math.floor(xyxy[1])), int(math.ceil(xyxy[2])),
                       int(math.ceil(xyxy[3]))])
    for r in recs:                                   # verified names / columns without a renderer box index
        if r not in dl and r["kind"] != "ruby" and not (r.get("box_index") or []):
            others.append(xywh_to_xyxy(r["box"]))
    for r in dl:                                     # a tall band holds an unlabelled text line: keep it out
        if tall_band(r):
            x, y, w, _ = r["box"]
            others.append([x, y, x + w, r["box_main"][1]])
    members = [(xywh_to_xyxy(r["box_main"] if tall_band(r) else r["box"]), r["id"], r["text"]) for r in dl]
    grey = np.zeros((H, W), np.uint8)
    out = []
    for c in FT.dialogue_units("p", platform, W, H, members, others, grey, "hooked", None):
        out.append({"variant": c["variant"], "region": c["region"], "lines": [ln["id"] for ln in c["lines"]]})
    return out


def import_pages(args):
    folder, assign, pools_dir, out = args
    folders = list(folder) if isinstance(folder, (list, tuple)) else [folder]
    meta, pages, lines, runs = load_title_runs(folders)
    tid, platform = meta["title_id"], meta["platform"]
    W, H = meta["resolution"]
    dec = {r["id"]: r for r in read_jsonl(Path(pools_dir) / "decisions" / f"{tid}.jsonl")}
    by_page = defaultdict(list)
    for r in lines:
        by_page[page_id(r)].append(r)
    rows, counts = [], Counter()
    for p in sorted(pages, key=lambda r: int(page_id(r))):
        pid = page_id(p)
        recs = by_page.get(pid, [])
        boxes = page_boxes(p)
        covered = set()
        for r in recs:
            covered.update(r.get("box_index") or [])
        # complete: every renderer box is some verified record's, and no record's label is in doubt (a filter that
        # drops a line for the recognizer, vocab, CTC or size, leaves its text known; a pixel conflict does not)
        untrusted = [r for r in recs if dec[r["id"]].get("drop") in UNTRUSTED or
                     any(e.get("drop") in UNTRUSTED for e in dec[r["id"]]["pools"].values())]
        complete = bool(boxes) and all(i in covered for i, _, _ in boxes) and not untrusted and \
            not any(tall_band(r) for r in recs)
        lines_out = []
        for r in recs:
            d = dec[r["id"]]
            drops = [e["drop"] for e in d["pools"].values() if e.get("drop")]
            cb = r.get("cell_boxes")
            if d.get("undrawn_removed") and cb and len(cb) == len(r.get("cell_chars") or []):
                cb = [b for b, c in zip(cb, r["cell_chars"]) if c not in UNDRAWN.get(tid, "")]   # the code's blanks
            lines_out.append({"id": r["id"], "kind": r["kind"], "vertical": bool(r["vertical"]), "box": r["box"],
                              "box_main": r.get("box_main"), "furigana": bool(r.get("furigana")),
                              "tall_band": tall_band(r),
                              "box_index": r.get("box_index"), "pad": r["pad"], "text": d["text"],
                              "source": r["source"], "host_line": r.get("host_line"),
                              "drop": d.get("drop") or (drops[0] if drops else None),
                              "cell_boxes": cb, **({"label_rule": d["label_rule"]} if d.get("label_rule") else {})})
        # regions: every page with verified horizontal dialogue and no label in doubt; the rule keeps every other box
        # (an unlabelled nameplate, UI text, an unverified row) out of the region, or drops the group it overlaps
        regs = regions_of(p, recs, platform, W, H) if not untrusted else []
        uncovered = [i for i, _, _ in boxes if i not in covered]
        row = {"title_id": tid, "page": pid, "image": f"{tid}/pages/{pid}.png", "image_sha256": p["image_sha256"],
               "dims": [W, H], "platform": platform, "split": assign[pid], "complete": complete,
               "untrusted": bool(untrusted), "uncovered": uncovered,
               "text_free": not boxes, "roi": (p.get("calibration") or {}).get("dialogue_roi"),
               "boxes": [{"index": i, "xyxy": xyxy, "kind": k} for i, xyxy, k in boxes],
               "lines": lines_out, "regions": regs, "text": p.get("text_clean", p.get("text")),
               "speaker": (p.get("clean") or {}).get("speaker")}
        if p.get("run_index"):                       # a later capture run's page: its run and its own page id
            row.update(run=runs[p["run_index"]]["run"], src_page=p["src_page"])
        rows.append(row)
        counts[f"pages:{assign[pid]}"] += 1
        counts[f"complete:{assign[pid]}"] += complete
        counts[f"with_lines:{assign[pid]}"] += bool(recs)
        counts[f"untrusted:{assign[pid]}"] += bool(untrusted)
        counts[f"text_free:{assign[pid]}"] += not boxes
        counts[f"pages_with_regions:{assign[pid]}"] += any(g["variant"] == "base" for g in regs)
        counts[f"regions_base:{assign[pid]}"] += sum(1 for g in regs if g["variant"] == "base")
    path = Path(out) / f"{tid}.jsonl"
    write_jsonl(path, rows)
    return tid, dict(counts), sha256_file(path)


def cmd_pages(a):
    runs = title_runs(a.root)
    splits = json.loads(Path(a.splits).read_text(encoding="utf-8"))
    titles = [t for t in sorted(runs) if not a.titles or t in a.titles.split(",")]
    jobs = [([str(f) for f in runs[t]], splits["titles"][t]["assign"], a.pools, a.out) for t in titles]
    man_path = Path(a.out) / "manifest.json"
    man = json.loads(man_path.read_text()) if man_path.exists() and a.titles else {"titles": {}}
    with ProcessPoolExecutor(a.procs) as ex:
        for tid, counts, sha in ex.map(import_pages, jobs):
            man["titles"][tid] = {"counts": dict(sorted(counts.items())), "sha256": sha}
            print(tid, dict(sorted(counts.items())), flush=True)
    tot = Counter()
    for t in man["titles"].values():
        tot.update(t["counts"])
    man.update({"schema": SCHEMA, "kind": "vocr6-hooked-pagepool", "pages_dir": str(a.pages),
                "importer_sha256": sha256_file(__file__), "splits_sha256": sha256_file(a.splits),
                "counts": dict(sorted(tot.items()))})
    write_json(man_path, man)


# ----------------------------------------------------------------------------- committed manifests

DROP_CODES = {"oov": "o", "ctc": "c", "geometry": "g", "conflict": "x", "empty_label": "e", "box_outside_page": "b",
              "label_doubt": "d"}


def line_code(d):
    """A record's outcome in the committed manifest: its pools joined by '+', each <pool>:<targets or drop>."""
    if d.get("drop"):
        return DROP_CODES[d["drop"]]
    parts = []
    for pool, e in d["pools"].items():
        if e.get("drop"):
            parts.append(f"{pool}:{DROP_CODES[e['drop']]}")
        elif e.get("to"):
            parts.append(f"{pool}:{','.join(e['to'])}")
        else:
            parts.append(f"{pool}:dup-{e.get('dup')}")
    return "+".join(parts)


def cmd_manifests(a):
    splits = json.loads(Path(a.splits).read_text(encoding="utf-8"))
    pools = json.loads((Path(a.pools) / "manifest.json").read_text(encoding="utf-8"))
    pagepool = json.loads((Path(a.pagepool) / "manifest.json").read_text(encoding="utf-8"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    index = {"schema": SCHEMA, "kind": "vocr6-hooked-manifest-index",
             "note": "ids, hashes, geometry and codes only: no hooked text, no pixels (docs/vocr6-data.md, "
                     "docs/vocr6-data-v2.md)",
             "sources": json.loads(Path(a.sources).read_text()) if a.sources else None,
             "split_params": splits["params"], "split_rule": splits["rule"], "split_counts": splits["counts"],
             "split_runs_rule": splits.get("runs_rule"), "pool_tag": pools.get("pool_tag"),
             "vndb_map_sha256": pools.get("vndb_map_sha256"),
             "pool_counts": pools["counts"], "pool_files": pools["files"], "pagepool_counts": pagepool["counts"],
             "importer_sha256": pools["importer_sha256"], "vocab": pools["vocab"], "vocab_s": pools["vocab_s"],
             "near_duplicate": pools["near_duplicate"], "line_codes": {
                 "drops": DROP_CODES, "form": "<pool>:<targets> joined by '+'; dup-exact / dup-near: a duplicate "
                                             "of its cluster's representative, in no pool"},
             "titles": {}}
    for tid, t in sorted(splits["titles"].items()):
        dec = {r["id"]: r for r in read_jsonl(Path(a.pools) / "decisions" / f"{tid}.jsonl")}
        pages = read_jsonl(Path(a.pagepool) / f"{tid}.jsonl")
        rows = []
        for p in pages:
            row = {"page": p["page"], "sha256": p["image_sha256"], "split": p["split"], "complete": p["complete"],
                   "lines": {r["id"].rsplit(":", 1)[1]: line_code(dec[r["id"]]) for r in p["lines"]}}
            if p.get("src_page"):                    # a later capture run's page: <run index>:<its own page id>
                row["src"] = f"{int(p['page']) // RUN_STRIDE}:{p['src_page']}"
            rows.append(row)
        path = out / f"{tid}.jsonl"
        write_jsonl(path, rows)
        index["titles"][tid] = {"title_split": t["title_split"], "platform": t["platform"], "pages": t["pages"],
                                "hval_blocks": t["hval_blocks"], "counts": t["counts"],
                                "runs": t.get("runs"), "cross_run_guard": len(t.get("cross_run_guard") or []),
                                "pool_counts": pools["titles"][tid]["counts"],
                                "pagepool_counts": pagepool["titles"][tid]["counts"],
                                "manifest_sha256": sha256_file(path)}
    write_json(out / "index.json", index)
    print(f"{len(index['titles'])} title manifests -> {out}")


# ----------------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("extract")
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--titles", default="")
    p.add_argument("--procs", type=int, default=5)
    p = sub.add_parser("split")
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--share", type=float, default=0.04)
    p.add_argument("--guard", type=int, default=15)
    p.add_argument("--seed", type=int, default=20261026)
    p.add_argument("--min-pages", type=int, default=200)
    p.add_argument("--block-max", type=int, default=40)
    p = sub.add_parser("lines")
    p.add_argument("--root", required=True)
    p.add_argument("--pages", required=True)
    p.add_argument("--splits", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--titles", default="")
    p.add_argument("--mad", type=float, default=NEAR_MAD)
    p.add_argument("--procs", type=int, default=5)
    p.add_argument("--pool-tag", default=POOL_TAG, help="every row's meta `pool` (v1's: hooked-v1)")
    p.add_argument("--vndb-map", default="", help="vndb_map.json: the VN of a title whose metadata has none")
    p.add_argument("--label-rules", default="v2", choices=("v1", "v2"),
                   help="v2 (default): pools v2's audit rules (undrawn codes, remapped glyphs, doubled cells); v1: "
                        "none of them (pools v1's rebuild)")
    p = sub.add_parser("pages")
    p.add_argument("--root", required=True)
    p.add_argument("--pages", required=True)
    p.add_argument("--splits", required=True)
    p.add_argument("--pools", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--titles", default="")
    p.add_argument("--procs", type=int, default=5)
    p = sub.add_parser("manifests")
    p.add_argument("--splits", required=True)
    p.add_argument("--pools", required=True)
    p.add_argument("--pagepool", required=True)
    p.add_argument("--sources", default="", help="the download's verification summary (JSON) to record")
    p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    cv2.setNumThreads(1)
    {"extract": cmd_extract, "split": cmd_split, "lines": cmd_lines, "pages": cmd_pages,
     "manifests": cmd_manifests}[a.cmd](a)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
