#!/usr/bin/env python3
"""The hand-checked full-frame development and test sets of the text spotter (issue #104, docs/textspot.md,
eval/textspot_protocol.json "sets" and "labels", which this file implements and may not change).

  textspot_sets.py dev-frames                  the development inventory and selection -> $SETS/dev/frames.jsonl
  textspot_sets.py test-frames [--fetch]       the test frames (eval's screenshots, text-free candidates fetched from
                                               t.vndb.org at most 1 request per second) -> $SETS/test/frames.jsonl
  textspot_sets.py propose --split S           MeikiOCR + PP-OCRv6 small on every candidate frame (teacher/.venv,
                                               through heavy.sh) -> $SETS/<split>/proposals.jsonl
  textspot_sets.py reread --split S            refine and read the boxes the reviewers drew (teacher/.venv)
  textspot_sets.py sheets --split S --stage p1b|p1t|p2b|p2t|p3     the next review tasks
  textspot_sets.py claim|done|check|zoom ...   the reviewers' side of the review (see the batch READMEs)
  textspot_sets.py decide --split S            every decision -> $SETS/<split>/labels.jsonl (+ the open work)
  textspot_sets.py assemble --split S          -> eval/manifests/textspot-<split>.jsonl and $SETS/<split>/PIN.json
  textspot_sets.py feed [--loop]               keep the pipeline going: rereads, next tasks, decide
  textspot_sets.py status                      tasks made / claimed / done

Frames. A development frame is one of the dev_sample VNs' local VNDB screenshots (realdata's and devnative's
copies, deduplicated by VNDB screenshot id and by file sha256): a native 960x544 PS Vita screenshot (part vita), a
native 480x272 PSP one (psp), a PSP one scaled 2x with cv2.INTER_LINEAR (adrenaline), a PC one scaled to fit 960x544
with its aspect, cv2.INTER_AREA (pc), or one with no text at all (text-free). A frame's rank is
sha256("textspot-dev:" + its VNDB screenshot id, e.g. "sf57781"), a VN's sha256("textspot-dev-vn:" + vn); test
frames rank by sha256("textspot-test:" + id). Every choice is the protocol's (select_dev, test_inventory).

Labels are made on the *label frame*: the frame itself, except an Adrenaline frame, which is labelled as its native
480x272 PSP screenshot and whose references are those boxes times 2 exactly. Everything in the review is in label
frame pixels.

Nothing of any arm (GRAD, refine, gate, the #97 detector, the column finder, a spotter candidate) runs here: only the
labelling engines (MeikiOCR, PP-OCRv6 small) and Claude reviewers.
"""
import argparse
import collections
import contextlib
import datetime
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (ROOT, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

HERD = Path(os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr"))
ART = HERD / "shared" / "artifacts"
SETS = Path(os.environ.get("TEXTSPOT_SETS") or ART / "textspot" / "sets")
REALDATA = ART / "realdata"
TEACHER_IN = ART / "teacher" / "inputs"
DEVNATIVE = ART / "devnative"
EVAL_DATA = Path(os.environ.get("VOCR_EVAL_DATA") or ART / "eval")
DUMP = ART / "realplus" / "vndb" / "dump" / "db"
FONT = HERD / "shared" / "data" / "fonts" / "bizudgothic" / "BIZUDGothic-Regular.ttf"
MANIFESTS = ROOT / "eval" / "manifests"
UA = "vita-vn-ocr-textspot/0.1 (OCR evaluation set, low volume; github.com/bee-san/vita-vn-ocr)"

VITA, PSP = (960, 544), (480, 272)
PAD = {"vita": 4, "psp": 2}
CLASSES = ("dialogue", "name", "choice", "ui", "scene")
# why an area is ignored (protocol "labels.ignore"); "undecided": two looks could not agree; "uncertain": an eval line
# eval itself left uncertain (test frames only)
IGNORE_WHY = ("furigana", "logo", "small", "unreadable", "watermark", "script", "rotated", "undecided", "uncertain")
REVIEW_WHY = IGNORE_WHY[:7]          # the whys a reviewer may give
PARTS = ("vita", "psp", "adrenaline", "pc", "text-free")
SEXUAL_MAX, VIOLENCE_MAX = 0.3, 0.5
CAPS = {"vita": 5, "psp": 2, "adrenaline_vns": 20, "pc_vns": 40, "text_free": 25}
TF_MAX_DETS = 1                      # a text-free candidate: MeikiOCR finds at most one box
STAGES = ("p1b", "p1t", "p2b", "p2t", "p3")
IMAGE_BUDGET = 48                    # images a reviewer session may claim (+ a few zoom views, under 55)


# ----------------------------------------------------------------------------- small helpers

def utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def jsonl(path):
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_review_json(path, obj):
    """A review file for a reader: the top level one key per line, each element of a list on its own line."""
    lines = ["{"]
    items = list(obj.items())
    for i, (k, v) in enumerate(items):
        end = "," if i < len(items) - 1 else ""
        if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
            lines.append(f" {json.dumps(k)}: [")
            for j, x in enumerate(v):
                lines.append("  " + json.dumps(x, ensure_ascii=False) + ("," if j < len(v) - 1 else ""))
            lines.append(" ]" + end)
        else:
            lines.append(f" {json.dumps(k)}: {json.dumps(v, ensure_ascii=False)}{end}")
    lines.append("}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rank_key(prefix, key):
    return hashlib.sha256(f"{prefix}{key}".encode()).hexdigest()


def frame_rank(split, sid):
    return rank_key("textspot-dev:" if split == "dev" else "textspot-test:", sid)


def vn_rank(vn):
    return rank_key("textspot-dev-vn:", vn)


_NUM = re.compile(r"/(\d+)\.(?:jpg|jpeg|png)$")


def image_num(url):
    m = _NUM.search(url)
    if not m:
        raise ValueError(f"no VNDB screenshot number in {url!r}")
    return int(m.group(1))


def url_of(num):
    return f"https://t.vndb.org/sf/{num % 100:02d}/{num}.jpg"


def xyxy(b):
    return [b[0], b[1], b[0] + b[2], b[1] + b[3]]


def xywh(b):
    return [b[0], b[1], b[2] - b[0], b[3] - b[1]]


def inter(a, b):
    """Intersection area of two x0, y0, x1, y1 boxes."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return w * h if w > 0 and h > 0 else 0


def area(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def union(boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


def clip(b, W, H):
    return [max(0, min(W, int(b[0]))), max(0, min(H, int(b[1]))), max(0, min(W, int(b[2]))), max(0, min(H, int(b[3])))]


def canon(t):
    """Text in the label form (eval's convention): NFKC, no spaces, full-width ！？～, … for ellipses."""
    from vertical_data import canon as vc
    return vc(t)


def norm(t):
    from metrics import norm as mn
    return mn(t or "")


@contextlib.contextmanager
def locked(path):
    import fcntl
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


# ----------------------------------------------------------------------------- reservation and eval

def reservation():
    return json.loads((MANIFESTS / "reserved.json").read_text(encoding="utf-8"))


def eval_rows(name):
    return jsonl(MANIFESTS / f"{name}.jsonl")


def eval_images():
    """{VNDB image number: (set, sha256)} of vita-test's and psp-test's screenshots."""
    out = {}
    for s in ("vita-test", "psp-test"):
        for r in eval_rows(s):
            out[image_num(r["url"])] = (s, r["sha256"])
    return out


def test_games():
    tg = json.loads((MANIFESTS / "test_games.json").read_text(encoding="utf-8"))
    return sorted(set(tg["vita-test"]) | set(tg["psp-test"]), key=lambda v: int(v[1:])), tg


# ----------------------------------------------------------------------------- frames

def read_rgb(path):
    import cv2
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return None if bgr is None else np.ascontiguousarray(bgr[:, :, ::-1])


def frame_spec(part, src_dims):
    """(frame dims, scale) of a part's frame from a source of src_dims; scale None for a native frame."""
    w, h = src_dims
    if part in ("vita", "psp") or (part == "text-free" and tuple(src_dims) in (VITA, PSP)):
        return [w, h], None
    if part == "adrenaline":
        return [2 * w, 2 * h], {"method": "cv2.INTER_LINEAR", "factor": 2}
    s = min(VITA[0] / w, VITA[1] / h)
    return [max(1, int(round(w * s))), max(1, int(round(h * s)))], {"method": "cv2.INTER_AREA",
                                                                      "factor": round(s, 6)}


def build_frame(rgb, dims, scale):
    """The frame's pixels from the source's RGB (frame_spec's dims and scale)."""
    import cv2
    if scale is None:
        if [rgb.shape[1], rgb.shape[0]] != list(dims):
            raise ValueError(f"native frame of size {rgb.shape[1]}x{rgb.shape[0]}, expected {dims}")
        return rgb
    interp = {"cv2.INTER_LINEAR": cv2.INTER_LINEAR, "cv2.INTER_AREA": cv2.INTER_AREA}[scale["method"]]
    return np.ascontiguousarray(cv2.resize(rgb, tuple(int(v) for v in dims), interpolation=interp))


def label_spec(row):
    """(label dims, label scale, ref_scale) of a frame row: an Adrenaline frame is labelled at its native size."""
    if row["part"] == "adrenaline":
        return list(row["src_dims"]), None, 2
    return list(row["dims"]), row["scale"], 1


def pixel_sha256(rgb):
    a = np.ascontiguousarray(rgb, dtype=np.uint8)
    h = hashlib.sha256(f"{a.shape[0]}x{a.shape[1]}x3".encode())
    h.update(a.tobytes())
    return h.hexdigest()


_SRC_OK = {}


def load_label_frame(shot):
    """RGB of a shot's label frame (shot: a row of shots.jsonl / proposals with path, sha256, label dims, scale)."""
    path = shot["path"]
    if path not in _SRC_OK:
        got = sha256_file(path)
        if got != shot["sha256"]:
            raise RuntimeError(f"{path}: sha256 {got} != {shot['sha256']}")
        _SRC_OK[path] = True
    rgb = read_rgb(path)
    return build_frame(rgb, shot["ldims"], shot["lscale"])


def platform_of(dims):
    return "psp" if dims[0] <= PSP[0] and dims[1] <= PSP[1] else "vita"


# ----------------------------------------------------------------------------- development inventory and selection

def dev_inventory():
    """Every eligible development screenshot: [{shot, sid, num, vn, kind, url, release, sha256, src_dims, path,
    sexual, violence, rank}] and the count of every reason one was left out."""
    res = reservation()
    dev, test, sessions = set(res["dev_sample"]), set(res["test"]), set(res["sessions"])
    ev = eval_images()
    ev_sha = {s for _, s in ev.values()}
    raw = []
    for kind in ("vita", "psp", "hires"):
        for r in jsonl(TEACHER_IN / f"shots_{kind}.jsonl"):
            raw.append(dict(r, src="realdata",
                            path=str(REALDATA / "shots" / r["kind"] / f"{r['vn']}_{r['url'].rsplit('/', 1)[-1]}")))
    for r in jsonl(DEVNATIVE / "shots.jsonl"):
        raw.append(dict(r, src="devnative"))
    why = collections.Counter()
    by_num = {}
    for r in raw:
        if r["vn"] not in dev:
            continue
        num = image_num(r["url"])
        if num in by_num:
            if by_num[num]["sha256"] != r["sha256"]:
                why["same-id-other-sha256"] += 1
            why["duplicate-id"] += 1
            continue
        if r["vn"] in test:
            why["test-vn"] += 1
            continue
        if r.get("release") in sessions:
            why["test-session"] += 1
            continue
        if num in ev or r["sha256"] in ev_sha:
            why["eval-screenshot"] += 1
            continue
        sx, vi = r.get("sexual"), r.get("violence")
        if (sx is not None and sx >= SEXUAL_MAX) or (vi is not None and vi >= VIOLENCE_MAX):
            why["content-flags"] += 1
            continue
        dims = tuple(r["dims"])
        kind = r["kind"]
        if kind == "vita" and dims != VITA or kind == "psp" and dims != PSP:
            why["not-native"] += 1
            continue
        if kind not in ("vita", "psp", "hires"):
            why["kind"] += 1
            continue
        if not os.path.exists(r["path"]):
            why["no-local-file"] += 1
            continue
        by_num[num] = {"shot": f"{r['vn']}_{num}", "sid": f"sf{num}", "num": num, "vn": r["vn"], "kind": kind,
                       "url": r["url"], "release": r.get("release"), "sha256": r["sha256"],
                       "src_dims": list(dims), "path": r["path"], "sexual": sx, "violence": vi, "src": r["src"],
                       "rank": frame_rank("dev", f"sf{num}")}
    inv = sorted(by_num.values(), key=lambda r: r["rank"])
    seen_sha, out = set(), []
    for r in inv:                      # a picture uploaded twice (two ids, one file): the lower rank stays
        if r["sha256"] in seen_sha:
            why["duplicate-sha256"] += 1
            continue
        seen_sha.add(r["sha256"])
        out.append(r)
    return out, dict(why)


def label_shot_of(r):
    """The label-frame spec of an inventory screenshot (vita / psp native, PC scaled to fit the Vita)."""
    part = {"vita": "vita", "psp": "psp", "hires": "pc"}[r["kind"]]
    dims, scale = frame_spec(part, r["src_dims"])
    return dict(r, ldims=dims, lscale=scale)


def select_dev(inv, drops=(), tf=None, dets=None):
    """The development selection (protocol sets.dev.composition), a pure function of the inventory, the shots the
    reviewers marked not a game screen (drops), the text-free looks (tf: {shot: "yes" | "no" | None}) and
    MeikiOCR's detector box counts (dets: {shot: n}). Returns (frames, text-free candidates in rank order)."""
    drops, tf, dets = set(drops), tf or {}, dets or {}
    by_vn = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in inv:
        by_vn[r["vn"]][r["kind"]].append(r)
    for d in by_vn.values():
        for k in d:
            d[k].sort(key=lambda r: r["rank"])
    vns = sorted(by_vn, key=vn_rank)
    frames, used = [], set()

    def add(r, part):
        used.add(r["shot"])
        dims, scale = frame_spec(part, r["src_dims"])
        fid = r["shot"] + ("_x2" if part == "adrenaline" else "")
        frames.append({"id": fid, "vn": r["vn"], "part": part, "shot": r["shot"], "sid": r["sid"], "url": r["url"],
                       "session": r["release"], "sha256": r["sha256"], "src_dims": r["src_dims"], "scale": scale,
                       "dims": dims, "rank": r["rank"], "vn_rank": vn_rank(r["vn"]), "path": r["path"],
                       "text_free": part == "text-free"})

    def first(vn, kind, n):
        out = []
        for r in by_vn[vn].get(kind, []):
            if len(out) >= n:
                break
            if r["shot"] in drops or r["shot"] in used:
                continue
            out.append(r)
        return out

    for vn in vns:
        for r in first(vn, "vita", CAPS["vita"]):
            add(r, "vita")
    for vn in vns:
        for r in first(vn, "psp", CAPS["psp"]):
            add(r, "psp")
    psp_vns = [vn for vn in vns if by_vn[vn].get("psp")]
    for vn in psp_vns[:CAPS["adrenaline_vns"]]:
        for r in first(vn, "psp", 1):
            add(r, "adrenaline")
    hires_vns = [vn for vn in vns if by_vn[vn].get("hires")]
    for vn in hires_vns[:CAPS["pc_vns"]]:
        for r in first(vn, "hires", 1):
            add(r, "pc")
    cands = [r for r in inv if r["shot"] not in used and r["shot"] not in drops and r["shot"] in dets
             and dets[r["shot"]] <= TF_MAX_DETS]
    n_tf = 0
    for r in cands:                     # rank order; the first CAPS confirmed by two looks
        if n_tf >= CAPS["text_free"]:
            break
        if tf.get(r["shot"]) == "yes":
            add(r, "text-free")
            n_tf += 1
    return frames, cands


def tf_candidates_to_look(cands, tf, need=CAPS["text_free"], spare=8):
    """The text-free candidates that still need a look so that the first `need` confirmed ones by rank are known:
    walking in rank order, every undecided candidate until `need` confirmed plus pending ones (and `spare` more
    for the expected rejections) are covered."""
    out, yes, pending = [], 0, 0
    for r in cands:
        st = tf.get(r["shot"])
        if st == "yes":
            yes += 1
        elif st == "no":
            continue
        else:
            out.append(r["shot"])
            pending += 1
        if yes >= need or yes + pending >= need + spare:
            break
    return out


# ----------------------------------------------------------------------------- test inventory

def dump_rows(name):
    with open(DUMP / name, encoding="utf-8") as f:
        for line in f:
            yield line.rstrip("\n").split("\t")


def test_inventory():
    """The test frames' sources: every vita-test / psp-test screenshot, and the 28 test games' other native-size
    screenshots in the VNDB dump (text-free candidates; flags sexual < 0.3, violence < 0.5)."""
    games, tg = test_games()
    ev = {}
    for s, plat in (("vita-test", "vita"), ("psp-test", "psp")):
        for r in eval_rows(s):
            num = image_num(r["url"])
            ev.setdefault(num, {"shot": f"{r['vn']}_{num}", "sid": f"sf{num}", "num": num, "vn": r["vn"],
                                "kind": plat, "set": s, "url": r["url"], "release": r["session"],
                                "sha256": r["sha256"], "src_dims": list(r["dims"]),
                                "path": str(EVAL_DATA / plat / "shots" / f"{r['vn']}_{r['url'].rsplit('/', 1)[-1]}"),
                                "rank": frame_rank("test", f"sf{num}")})
    img = {}
    for r in dump_rows("images"):
        if r[0].startswith("sf"):
            img[int(r[0][2:])] = (int(r[1]), int(r[2]), int(r[4]) / 100, int(r[6]) / 100)
    plat = collections.defaultdict(set)
    for r in dump_rows("releases_platforms"):
        plat[r[0]].add(r[1])
    gs = set(games)
    extra, why = [], collections.Counter()
    for vn, scr, rid in dump_rows("vn_screenshots"):
        if vn not in gs:
            continue
        num = int(scr[2:])
        if num in ev:
            continue
        if num not in img:
            why["no-image-row"] += 1
            continue
        w, h, sx, vi = img[num]
        if sx >= SEXUAL_MAX or vi >= VIOLENCE_MAX:
            why["content-flags"] += 1
            continue
        p = plat.get(rid, set())
        kind = "vita" if "psv" in p and (w, h) == VITA else "psp" if "psp" in p and (w, h) == PSP else None
        if kind is None:
            why["not-native"] += 1
            continue
        extra.append({"shot": f"{vn}_{num}", "sid": f"sf{num}", "num": num, "vn": vn, "kind": kind, "set": None,
                      "url": url_of(num), "release": None if rid == "\\N" else rid, "sha256": None,
                      "src_dims": [w, h], "sexual": sx, "violence": vi,
                      "path": str(SETS / "test" / "shots" / f"{vn}_{num}.jpg"), "rank": frame_rank("test", f"sf{num}")})
    extra.sort(key=lambda r: r["rank"])
    return sorted(ev.values(), key=lambda r: r["rank"]), extra, dict(why)


def fetch_extra(extra, log):
    """Downloads every text-free candidate of the test games from t.vndb.org, at most one request per second, and
    records its sha256 (fetched.jsonl). Already-fetched files are kept."""
    import urllib.request
    done = {r["shot"]: r for r in jsonl(log)}
    last = 0.0
    for r in extra:
        if r["shot"] in done and os.path.exists(r["path"]):
            continue
        Path(r["path"]).parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(4):
            wait = 1.0 - (time.time() - last)
            if wait > 0:
                time.sleep(wait)
            last = time.time()
            try:
                req = urllib.request.Request(r["url"], headers={"User-Agent": UA})
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = resp.read()
                break
            except Exception as e:                      # noqa: BLE001 (retry with backoff on any network error)
                print(f"{r['url']}: {e}; retry {attempt + 1}", flush=True)
                time.sleep(2 ** attempt)
        else:
            print(f"{r['url']}: giving up", flush=True)
            continue
        Path(r["path"]).write_bytes(data)
        rec = {"shot": r["shot"], "url": r["url"], "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
               "fetched_utc": utc()}
        with open(log, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        done[r["shot"]] = rec
        print(f"fetched {r['shot']} ({len(data)} B)", flush=True)
    return done


# ----------------------------------------------------------------------------- shots files (what propose reads)

def shots_path(split):
    return SETS / split / "shots.jsonl"


def load_shots(split):
    return {r["shot"]: r for r in jsonl(shots_path(split))}


def cmd_dev_frames(args):
    inv, why = dev_inventory()
    shots = [label_shot_of(r) for r in inv]
    out = SETS / "dev"
    write_jsonl(shots_path("dev"), shots)
    st = review_state("dev")
    frames, cands = select_dev(inv, st["drops"], st["tf"], st["dets"])
    write_jsonl(out / "frames.jsonl", frames)
    write_jsonl(out / "tf_candidates.jsonl", [{"shot": r["shot"], "rank": r["rank"], "kind": r["kind"],
                                               "dets": st["dets"].get(r["shot"]), "look": st["tf"].get(r["shot"])}
                                              for r in cands])
    c = collections.Counter(f["part"] for f in frames)
    print(f"inventory {len(inv)} screenshots of {len({r['vn'] for r in inv})} VNs; left out {why}")
    print("kinds", dict(collections.Counter(r["kind"] for r in inv)))
    print("frames", dict(c), "total", len(frames), "| text-free candidates", len(cands),
          "| proposals for", len(st["dets"]), "shots")
    write_json(out / "inventory_stats.json", {"utc": utc(), "inventory": len(inv), "left_out": why,
                                              "kinds": dict(collections.Counter(r["kind"] for r in inv)),
                                              "frames": dict(c), "tf_candidates": len(cands)})


def cmd_test_frames(args):
    ev, extra, why = test_inventory()
    out = SETS / "test"
    out.mkdir(parents=True, exist_ok=True)
    bad = [r["shot"] for r in ev if not os.path.exists(r["path"]) or sha256_file(r["path"]) != r["sha256"]]
    if bad:
        sys.exit(f"eval screenshots missing or with another sha256: {bad[:5]}")
    if args.fetch:
        fetch_extra(extra, out / "fetched.jsonl")
    got = {r["shot"]: r for r in jsonl(out / "fetched.jsonl")}
    shots = []
    for r in ev:
        dims, scale = frame_spec(r["kind"], r["src_dims"])
        shots.append(dict(r, ldims=dims, lscale=scale))
    n_extra = 0
    for r in extra:
        g = got.get(r["shot"])
        if not g or not os.path.exists(r["path"]):
            continue
        rgb = read_rgb(r["path"])
        if rgb is None or [rgb.shape[1], rgb.shape[0]] != r["src_dims"]:
            why["fetched-other-size"] = why.get("fetched-other-size", 0) + 1
            continue
        shots.append(dict(r, sha256=g["sha256"], ldims=list(r["src_dims"]), lscale=None))
        n_extra += 1
    write_jsonl(shots_path("test"), shots)
    st = review_state("test")
    frames = []
    for r in ev:
        part = r["kind"]
        dims, scale = frame_spec(part, r["src_dims"])
        frames.append({"id": r["shot"], "vn": r["vn"], "part": part, "shot": r["shot"], "sid": r["sid"],
                       "url": r["url"], "session": r["release"], "sha256": r["sha256"], "src_dims": r["src_dims"],
                       "scale": scale, "dims": dims, "rank": r["rank"], "path": r["path"], "set": r["set"],
                       "text_free": False})
    for r in ev:
        if r["kind"] == "psp":
            dims, scale = frame_spec("adrenaline", r["src_dims"])
            frames.append({"id": r["shot"] + "_x2", "vn": r["vn"], "part": "adrenaline", "shot": r["shot"],
                           "sid": r["sid"], "url": r["url"], "session": r["release"], "sha256": r["sha256"],
                           "src_dims": r["src_dims"], "scale": scale, "dims": dims, "rank": r["rank"],
                           "path": r["path"], "set": r["set"], "text_free": False, "report_only": True})
    cands = [s for s in shots if s.get("set") is None and st["dets"].get(s["shot"], 99) <= TF_MAX_DETS]
    cands.sort(key=lambda s: s["rank"])
    n_tf = 0
    for s in cands:
        if n_tf >= CAPS["text_free"]:
            break
        if st["tf"].get(s["shot"]) == "yes":
            frames.append({"id": s["shot"], "vn": s["vn"], "part": "text-free", "shot": s["shot"], "sid": s["sid"],
                           "url": s["url"], "session": s["release"], "sha256": s["sha256"],
                           "src_dims": s["src_dims"], "scale": None, "dims": s["src_dims"], "rank": s["rank"],
                           "path": s["path"], "set": None, "text_free": True})
            n_tf += 1
    write_jsonl(out / "frames.jsonl", frames)
    write_jsonl(out / "tf_candidates.jsonl", [{"shot": s["shot"], "rank": s["rank"], "kind": s["kind"],
                                               "dets": st["dets"].get(s["shot"]), "look": st["tf"].get(s["shot"])}
                                              for s in cands])
    print(f"eval screenshots {len(ev)}; other native screenshots of the test games {len(extra)} "
          f"(fetched and usable {n_extra}); left out {why}")
    print("frames", dict(collections.Counter(f["part"] for f in frames)), "| text-free candidates", len(cands))


# ----------------------------------------------------------------------------- geometry

def to_grey(rgb):
    r, g, b = (rgb[..., i].astype(np.uint16) for i in range(3))
    return ((77 * r + 150 * g + 29 * b) >> 8).astype(np.uint8)


def _runs(on):
    """[start, end) runs of a boolean vector."""
    runs, s = [], None
    for i, v in enumerate(list(on) + [False]):
        if v and s is None:
            s = i
        elif not v and s is not None:
            runs.append([s, i])
            s = None
    return runs


def rough_ink(rgb, rect, vertical=False):
    """The ink box of a rough rectangle a reviewer drew around one instance ([x0, y0, x1, y1], exclusive), or None.

    The rectangle only shrinks: a pixel is ink when it differs from the median of the rectangle's border ring by
    more than max(28, 0.4 x its 99th percentile) (teacher/geometry.refine's rule, either polarity), and the
    rectangle loses every border row and column with less ink than 2% of its length (at least one pixel). A
    reviewer draws it a little loose around the whole text; a rectangle that cuts text off stays cut (pass 2
    checks every box). None (no trim) when the contrast is too low to tell ink from background (the 99th
    percentile of the difference under 60 grey levels: ghosted buttons, text over a pattern)."""
    H, W = rgb.shape[:2]
    x0, y0, x1, y1 = clip(rect, W, H)
    g = to_grey(rgb[y0:y1, x0:x1]).astype(np.float32)
    if g.shape[0] < 3 or g.shape[1] < 3:
        return None
    ring = np.concatenate([g[0], g[-1], g[:, 0], g[:, -1]])
    d = np.abs(g - float(np.median(ring)))
    if float(np.percentile(d, 99)) < 60:
        return None
    t = max(28.0, 0.4 * float(np.percentile(d, 99)))
    m = d > t
    rows, cols = m.sum(1), m.sum(0)
    ry = np.flatnonzero(rows >= max(1, 0.02 * m.shape[1]))
    rx = np.flatnonzero(cols >= max(1, 0.02 * m.shape[0]))
    if not len(ry) or not len(rx):
        return None
    return [x0 + int(rx[0]), y0 + int(ry[0]), x0 + int(rx[-1]) + 1, y0 + int(ry[-1]) + 1]


def refine_box(rgb, rect, vertical=False, clamp=False):
    """A reviewer's rough rectangle -> the instance's box: the rectangle trimmed to its ink (rough_ink; it never
    grows and never cuts into a line of ink), or the rectangle itself when no clear ink is found. (Rule 2 since
    10:55Z; the first rule also ran teacher/geometry.refine on the trimmed box, which can cut glyph tops; the boxes
    it made that pass 2 had already been shown keep it.) clamp: a piece of a split box stays inside its own range
    along the line (the cut positions)."""
    H, W = rgb.shape[:2]
    rect = clip(rect, W, H)
    if rect[2] - rect[0] < 2 or rect[3] - rect[1] < 2:
        return rect
    ib = rough_ink(rgb, rect, vertical)
    if ib is None:
        return rect
    if clamp:
        if vertical:
            ib = [ib[0], max(ib[1], rect[1]), ib[2], min(ib[3], rect[3])]
        else:
            ib = [max(ib[0], rect[0]), ib[1], min(ib[2], rect[2]), ib[3]]
        if ib[2] - ib[0] < 1 or ib[3] - ib[1] < 1:
            return rect
    return ib


def iou_xyxy(a, b):
    i = inter(a, b)
    u = area(a) + area(b) - i
    return i / u if u > 0 else 0.0


def refine_det(rgb, det):
    """teacher/geometry.refine of a MeikiOCR line box, with one guard: refine keeps the smallest of its three masks'
    boxes, so a dark frame line next to a line of bright text can win with a box that holds only a strip of the
    text. When refine's box overlaps the detector's box at IoU < 0.5, the mask box with the highest IoU with it is
    taken instead (the detector's box when none reaches 0.5)."""
    import cv2

    from teacher import geometry as G
    ib, _ = G.refine(rgb, det)
    ib = [int(v) for v in ib]
    if iou_xyxy(ib, det) >= 0.5:
        return extend_along(rgb, ib, det)
    H, W = rgb.shape[:2]
    x0, y0, x1, y1 = det
    wx0, wy0, wx1, wy1 = max(0, x0 - 3), max(0, y0 - 3), min(W, x1 + 3), min(H, y1 + 3)
    win = rgb[wy0:wy1, wx0:wx1]
    g = G.to_gray(win).astype(np.float32)
    if g.shape[0] < 6 or g.shape[1] < 6:
        return [int(v) for v in det]
    ring = np.concatenate([g[0], g[-1], g[:, 0], g[:, -1]])
    d = np.abs(g - float(np.median(ring)))
    t = max(28.0, 0.4 * float(np.percentile(d, 99)))
    k = np.ones((3, 3), np.uint8)
    masks = [d > t, cv2.dilate((win.min(2) >= 200).astype(np.uint8), k).astype(bool),
             cv2.dilate((win.max(2) <= 70).astype(np.uint8), k).astype(bool)]
    boxes = [b for b in (G._ink_box(m, det, (wx0, wy0)) for m in masks) if b is not None]
    best = max(boxes, key=lambda b: iou_xyxy(b, det), default=None)
    if best is not None and iou_xyxy(best, det) >= 0.5:
        return extend_along(rgb, [int(v) for v in best], det)
    return [int(v) for v in det]


def extend_along(rgb, ib, det):
    """refine keeps its smallest mask's box, which can leave out an end bracket or punctuation mark (「 」 。) drawn
    thinner or greyer than the text; the box is widened along the line to the contrast-mask ink (refine's first
    mask) in its own rows inside the detector's box, while each gap is under 0.6 line heights."""
    H, W = rgb.shape[:2]
    h = ib[3] - ib[1]
    if h < 3:
        return ib
    wx0, wx1 = max(0, min(det[0], ib[0])), min(W, max(det[2], ib[2]))
    wy0, wy1 = max(0, ib[1] - 3), min(H, ib[3] + 3)
    g = to_grey(rgb[wy0:wy1, wx0:wx1]).astype(np.float32)
    if g.shape[0] < 4 or g.shape[1] < 4:
        return ib
    ring = np.concatenate([g[0], g[-1], g[:, 0], g[:, -1]])
    d = np.abs(g - float(np.median(ring)))
    m = d > max(28.0, 0.4 * float(np.percentile(d, 99)))
    cols = np.flatnonzero(m[ib[1] - wy0:ib[3] - wy0].any(0)) + wx0
    if not len(cols):
        return ib
    x0, x1 = ib[0], ib[2]
    left = cols[cols < x0]
    for c in left[::-1]:                  # walk outwards while the gaps stay small
        if x0 - c > 0.6 * h:
            break
        x0 = int(c)
    right = cols[cols >= x1]
    for c in right:
        if c - (x1 - 1) > 0.6 * h:
            break
        x1 = int(c) + 1
    return [x0, ib[1], x1, ib[3]]


def split_parts(ink, cuts, vertical):
    """The pieces of an ink box cut at frame positions `cuts` (x for horizontal, y for vertical)."""
    a0, a1 = (ink[1], ink[3]) if vertical else (ink[0], ink[2])
    edges = [a0] + sorted(int(c) for c in cuts if a0 < c < a1) + [a1]
    out = []
    for s, e in zip(edges, edges[1:]):
        out.append([ink[0], s, ink[2], e] if vertical else [s, ink[1], e, ink[3]])
    return out


def reading_order(boxes_vert):
    """Order of instances for numbering: top to bottom, then left to right (by the box's top and left)."""
    return sorted(range(len(boxes_vert)), key=lambda i: (boxes_vert[i][0][1] // 8, boxes_vert[i][0][0]))


# ----------------------------------------------------------------------------- engines (teacher/.venv)

_ENG = {}


def _init_engines():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from teacher import engines
    _ENG.update(engines.load(["meiki", "v6s"], n_threads=1))


def read_crop(bgr, ink, vertical, pad, eng=None):
    """MeikiOCR's recognizer and PP-OCRv6 small on an instance's crop (ink + pad); a vertical crop is turned 90
    degrees anticlockwise for PP-OCRv6 small (MeikiOCR reads a tall crop with its vertical recognizer)."""
    eng = eng or _ENG
    H, W = bgr.shape[:2]
    x0, y0, x1, y1 = clip([ink[0] - pad, ink[1] - pad, ink[2] + pad, ink[3] + pad], W, H)
    c = np.ascontiguousarray(bgr[y0:y1, x0:x1])
    out = {}
    if c.size == 0 or c.shape[0] < 2 or c.shape[1] < 2:
        return out, [x0, y0, x1, y1]
    try:
        out["meikic"] = eng["meiki"](c)[0]
    except Exception as e:                              # noqa: BLE001 (record and go on)
        out["meikic_error"] = str(e)[:200]
    v = eng["v6s"](np.ascontiguousarray(np.rot90(c)) if vertical else c)
    out["v6s"], out["v6s_conf"] = v[0], [v[1], v[2]]
    return out, [x0, y0, x1, y1]


def propose_shot(shot, eng=None):
    """MeikiOCR's proposals on one label frame (see cmd_propose)."""
    import cv2
    from vertical_data import column_box

    from teacher import geometry as G
    eng = eng or _ENG
    t0 = time.perf_counter()
    rgb = load_label_frame(shot)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    H, W = rgb.shape[:2]
    pad = PAD[platform_of([W, H])]
    dets = eng["meiki"].ocr(bgr)
    horiz = G.merge_rows([dict(d) for d in dets if not d["vertical"]])
    props = []
    for ln in horiz:
        ink = refine_det(rgb, [int(v) for v in ln["det"]])
        props.append({"det": [int(v) for v in ln["det"]], "parts": ln.get("parts", [ln["det"]]),
                      "det_conf": ln["det_conf"], "vertical": False, "ink": [int(v) for v in ink],
                      "reads": {"meiki": ln["text"]}})
    grey = to_grey(rgb)
    for d in dets:
        if not d["vertical"]:
            continue
        cb = column_box(grey, d["det"], pad)
        ink = cb[0] if cb else [int(v) for v in d["det"]]
        props.append({"det": [int(v) for v in d["det"]], "parts": [d["det"]], "det_conf": d["det_conf"],
                      "vertical": True, "ink": [int(v) for v in ink], "reads": {"meiki": d["text"]}})
    # furigana hints (teacher/geometry.is_furigana: small kana just above a bigger line); the host's ink box loses
    # rows the ruby took
    for i, p in enumerate(props):
        if p["vertical"]:
            continue
        for j, o in enumerate(props):
            if i != j and not o["vertical"] and G.is_furigana(p["ink"], o["ink"], p["reads"]["meiki"], p["det"]):
                p["furi_of"] = j
                h = o["ink"][3] - o["ink"][1]
                if p["ink"][3] > o["ink"][1] and p["ink"][3] < o["ink"][3] - 0.4 * h:
                    o["ink"][1] = p["ink"][3]
                break
    for p in props:
        r, crop = read_crop(bgr, p["ink"], p["vertical"], pad, eng)
        p["reads"].update(r)
        p["crop"] = crop
    order = sorted(range(len(props)), key=lambda i: (props[i]["ink"][1], props[i]["ink"][0]))
    remap = {old: new for new, old in enumerate(order)}
    props = [props[i] for i in order]
    for k, p in enumerate(props):
        p["k"] = k
        if "furi_of" in p:
            p["furi_of"] = remap[p["furi_of"]]
    row = {"shot": shot["shot"], "vn": shot["vn"], "dims": [W, H], "n_dets": len(dets), "props": props,
           "secs": round(time.perf_counter() - t0, 3)}
    if shot.get("set"):
        row["eval"] = eval_lines(shot, rgb, props)
    return row


def eval_lines(shot, rgb, props):
    """eval's hand-checked lines of a test screenshot as prefilled instances: text as eval has it, class name when
    tagged name else dialogue, uncertain lines as ignore areas; the reference box is the line's main-text ink
    (eval's box holds its furigana, which becomes an ignore area). MeikiOCR proposals lying mostly inside an eval
    line's box are marked as its duplicates."""
    H, W = rgb.shape[:2]
    out = []
    for r in eval_rows(shot["set"]):
        if image_num(r["url"]) != shot["num"]:
            continue
        eb = clip(xyxy(r["box"]), W, H)
        styles = r.get("styles") or []
        furi = None
        main = list(eb)
        if "furigana" in styles:
            rubies = [p for p in props if "furi_of" in p and inter(p["ink"], eb) >= 0.5 * area(p["ink"])]
            if rubies:
                fb = union([p["ink"] for p in rubies])
                main = [eb[0], max(eb[1], fb[3]), eb[2], eb[3]]
                furi = [eb[0], eb[1], eb[2], max(eb[1] + 1, fb[3])]
            else:
                g = to_grey(rgb[eb[1]:eb[3], eb[0]:eb[2]]).astype(np.float32)
                ring = np.concatenate([g[0], g[-1], g[:, 0], g[:, -1]])
                d = np.abs(g - float(np.median(ring)))
                m = d > max(28.0, 0.4 * float(np.percentile(d, 99)))
                runs = _runs(m.sum(1) >= 1)
                if len(runs) >= 2:
                    big = max(runs, key=lambda q: q[1] - q[0])
                    if big[0] > 0:
                        main = [eb[0], eb[1] + big[0], eb[2], eb[1] + big[1]]
                        furi = [eb[0], eb[1], eb[2], eb[1] + big[0]]
        ink = main
        if area(main) > 0:
            rb = refine_det(rgb, main)
            # eval's box is the hand-checked extent of the line and its furigana: the main text lies inside it
            rb = [max(rb[0], main[0]), max(rb[1], main[1]), min(rb[2], main[2]), min(rb[3], main[3])]
            if rb[2] - rb[0] >= 0.75 * (main[2] - main[0]) and rb[3] - rb[1] >= 0.6 * (main[3] - main[1]):
                ink = rb
        out.append({"eid": r["id"], "box": r["box"], "ink": ink, "furi": furi, "text": r["text"],
                    "cls": "name" if "name" in styles else "dialogue", "uncertain": bool(r.get("uncertain")),
                    "styles": styles})
    for p in props:
        for e in out:
            if inter(p["ink"], xyxy(e["box"])) >= 0.5 * max(1, area(p["ink"])):
                p["dup_eval"] = e["eid"]
                break
    return out


def _propose_job(shot):
    try:
        return propose_shot(shot)
    except Exception as e:                              # noqa: BLE001 (one bad frame must not stop the run)
        return {"shot": shot["shot"], "error": repr(e)[:300]}


def cmd_propose(args):
    import multiprocessing as mp
    shots = list(load_shots(args.split).values())
    path = SETS / args.split / "proposals.jsonl"
    done = {r["shot"] for r in jsonl(path) if not r.get("error")}
    if done:                              # rewrite without error rows so they are retried
        write_jsonl(path, [r for r in jsonl(path) if not r.get("error")])
    todo = [s for s in shots if s["shot"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(shots)} shots, {len(done)} proposed, {len(todo)} to do, {args.procs} workers", flush=True)
    write_json(SETS / args.split / "engines.json", engine_pins())
    t0 = time.time()
    with open(path, "a", encoding="utf-8") as fo, mp.get_context("spawn").Pool(args.procs,
                                                                               initializer=_init_engines) as pool:
        for i, row in enumerate(pool.imap_unordered(_propose_job, todo, chunksize=1)):
            fo.write(json.dumps(row, ensure_ascii=False) + "\n")
            fo.flush()
            if i % 25 == 0:
                print(f"{i + 1}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    print(f"PROPOSE_DONE {len(todo)} in {time.time() - t0:.0f}s", flush=True)


def engine_pins():
    import importlib.metadata as md

    from teacher import engines
    out = {}
    for name in ("meiki_det", "meiki_rec", "meiki_vrec", "v6s"):
        p = engines.model_path(name)
        out[name] = {"file": p.name, "sha256": sha256_file(p), "pinned": engines.PINS[name][2]}
        if out[name]["sha256"] != out[name]["pinned"]:
            sys.exit(f"{p}: sha256 differs from the pin")
    out["packages"] = {k: md.version(k) for k in ("meikiocr", "onnxruntime", "numpy", "opencv-python-headless")}
    return out


def load_proposals(split):
    return {r["shot"]: r for r in jsonl(SETS / split / "proposals.jsonl") if not r.get("error")}


# ----------------------------------------------------------------------------- review state

def review_dir(split, stage=None, batch=None):
    d = SETS / split / "review"
    if stage:
        d = d / stage
    if batch:
        d = d / batch
    return d


def tasks(split, stage):
    d = review_dir(split, stage)
    if not d.exists():
        return []
    return sorted(p for p in d.iterdir() if (p / "task.json").exists())


def task_of(path):
    return json.loads((Path(path) / "task.json").read_text(encoding="utf-8"))


def decisions_of(batch_dir):
    """[(label, decision)] of a batch, in file order (a later line for the same key replaces an earlier one)."""
    out = []
    for f in sorted(Path(batch_dir).glob("decisions-*.jsonl")):
        label = f.stem[len("decisions-"):]
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            try:
                out.append((label, json.loads(line)))
            except json.JSONDecodeError:
                out.append((label, {"_bad_line": i + 1}))
    return out


def claims_path():
    return SETS / "review_claims.jsonl"


def claims():
    """{task key: {"label", "claimed", "done"}} from the claim log."""
    st = {}
    for c in jsonl(claims_path()):
        k = c["task"]
        if c["event"] == "claim":
            st[k] = {"label": c["label"], "claimed": c["t"], "done": None, "images": c.get("images", 0)}
        elif c["event"] == "done" and k in st:
            st[k]["done"] = c["t"]
        elif c["event"] == "release" and k in st:
            del st[k]
    return st


def task_key(split, stage, batch):
    return f"{split}/{stage}/{batch}"


def done_tasks(split, stage):
    cl = claims()
    return [t for t in tasks(split, stage) if (cl.get(task_key(split, stage, t.name)) or {}).get("done")]


def p1b_decisions(split):
    """{shot: (label, decision, task path)} of the finished p1b tasks."""
    out = {}
    for t in done_tasks(split, "p1b"):
        meta = task_of(t)
        fmap = {f["F"]: f["shot"] for f in meta["frames"]}
        for label, d in decisions_of(t):
            if d.get("frame") in fmap:
                out[fmap[d["frame"]]] = (label, d, t)
    return out


def p2b_decisions(split):
    out = {}
    for t in done_tasks(split, "p2b"):
        meta = task_of(t)
        fmap = {f["F"]: f for f in meta["frames"]}
        for label, d in decisions_of(t):
            if d.get("frame") in fmap:
                out[fmap[d["frame"]]["shot"]] = (label, d, t, fmap[d["frame"]])
    return out


def text_decisions(split, stage):
    """{instance key: (label, decision, entry)} of the finished text tasks of a stage (p1t, p2t, or p3 sheets)."""
    out = {}
    for t in done_tasks(split, stage):
        for sheet in sorted(t.glob("S*.json")):
            sh = json.loads(sheet.read_text(encoding="utf-8"))
            ents = {e["n"]: e for e in sh["entries"]}
            for label, d in decisions_of(t):
                if d.get("sheet") == sh["sheet"] and d.get("n") in ents:
                    e = ents[d["n"]]
                    out[e["key"]] = (label, d, e)
    return out


def text_of(d, e):
    """A text decision -> (text in label form | None, status): status "text", "unsure" or "bad"."""
    if "pick" in d:
        cand = {c["c"]: c["text"] for c in e["candidates"]}
        return canon(cand.get(d["pick"], "")), "text"
    if "text" in d and isinstance(d["text"], str):
        return canon(d["text"]), "text"
    if d.get("bad"):
        return None, "bad"
    return None, "unsure"


def review_state(split):
    """What the finished review says so far: drops (not a game screen), text-free looks, MeikiOCR box counts."""
    props = load_proposals(split)
    dets = {s: r["n_dets"] for s, r in props.items()}
    p1 = p1b_decisions(split)
    p2 = p2b_decisions(split)
    p3 = p3_box_decisions(split)
    # a frame either pass marks as not a game screen is dropped (protocol: "dropped by the reviewer"), and the next
    # by rank of its part and VN is taken
    drops = {s for s, (_, d, _) in p1.items() if d.get("screen") is False}
    drops |= {s for s, v in p2.items() if v[1].get("screen") is False}
    tf = {}
    for s, (_, d, _) in p1.items():
        if d.get("screen") is False:
            continue
        a = d.get("text_free")
        if a is not True:
            if a is False or d.get("boxes") or d.get("missed"):
                tf[s] = "no" if (a is False or any("cls" in b for b in d.get("boxes", []))
                                 or any("cls" in m for m in d.get("missed", []))) else None
            continue
        b = p2.get(s)
        if b is None:
            tf[s] = None
            continue
        if b[1].get("text_free") is True:
            tf[s] = "yes"
            continue
        z = p3.get(("tf", s))
        tf[s] = None if z is None else ("yes" if z[1].get("text_free") is True else "no")
    return {"drops": drops, "tf": tf, "dets": dets, "p1": p1, "p2": p2, "p3": p3}


# ----------------------------------------------------------------------------- pass-1 instances

def p1_instances(shot, prop, dec, rereads):
    """Pass 1's instances of a frame from its proposals and its p1b decision: [{iid, ink, vertical, cls | ignore,
    src, reads, needs_text}]. Boxes the reviewer drew come refined from `rereads` ({(shot, iid): row}); one not
    reread yet has "pending": True."""
    out = []
    plist = review_proposals(prop)
    byn = {p["n"]: p for p in plist}
    merged_into = {}
    for b in dec.get("boxes", []):
        for m in b.get("merge", []) or []:
            merged_into[m] = b["n"]

    def reread(iid, rect, vertical):
        rr = rereads.get((shot, iid))
        if rr and rr["rect"] == list(rect) and bool(rr["vertical"]) == bool(vertical):
            return rr["ink"], rr["reads"], False
        return list(rect), {}, True

    for b in dec.get("boxes", []):
        p = byn.get(b.get("n"))
        if p is None or b["n"] in merged_into:
            continue
        vert = bool(b.get("vertical", p["vertical"]))
        if b.get("drop"):
            continue
        base = {"src": p["src"], "n1": p["n"]}
        if p["src"] == "eval":
            base["eid"] = p["eid"]
            base["eval_text"] = p["eval_text"]
        if "ignore" in b:
            out.append(dict(base, iid=f"{p['iid']}", ink=list(p["ink"]), vertical=vert, ignore=b["ignore"],
                            reads={}, needs_text=False))
            continue
        cls = b.get("cls")
        if "split" in b:
            parts = split_parts(p["ink"], b["split"], vert)
            cl = cls if isinstance(cls, list) else [cls] * len(parts)
            for i, part in enumerate(parts):
                iid = f"{p['iid']}.s{i}"
                ink, reads, pend = reread(iid, part, vert)
                out.append(dict(base, iid=iid, ink=ink, vertical=vert, cls=cl[i] if i < len(cl) else cl[-1],
                                reads=reads, pending=pend, needs_text=p["src"] != "eval", rect=part))
            continue
        if b.get("merge"):
            members = [p] + [byn[m] for m in b["merge"] if m in byn]
            rect = union([q["ink"] for q in members])
            iid = f"{p['iid']}.m"
            ink, reads, pend = reread(iid, rect, vert)
            joined = "".join(q["reads"].get("meiki") or "" for q in sorted(
                members, key=lambda q: (q["ink"][1], q["ink"][0]) if vert else (q["ink"][0], q["ink"][1])))
            reads = dict(reads, meiki=joined)
            out.append(dict(base, iid=iid, ink=ink, vertical=vert, cls=cls, reads=reads, pending=pend,
                            needs_text=p["src"] != "eval", rect=rect, members=[q["n"] for q in members]))
            continue
        if "box" in b:
            rect = xyxy(b["box"])
            iid = f"{p['iid']}.b"
            ink, reads, pend = reread(iid, rect, vert)
            out.append(dict(base, iid=iid, ink=ink, vertical=vert, cls=cls, reads=reads, pending=pend,
                            needs_text=p["src"] != "eval", rect=rect))
            continue
        if vert != p["vertical"]:
            iid = f"{p['iid']}.o"
            ink, reads, pend = reread(iid, p["ink"], vert)
            out.append(dict(base, iid=iid, ink=ink, vertical=vert, cls=cls, reads=reads, pending=pend,
                            needs_text=p["src"] != "eval", rect=list(p["ink"])))
            continue
        out.append(dict(base, iid=p["iid"], ink=list(p["ink"]), vertical=vert, cls=cls, reads=dict(p["reads"]),
                        needs_text=p["src"] != "eval"))
    for j, m in enumerate(dec.get("missed", []) or []):
        rect = xyxy(m["box"])
        vert = bool(m.get("vertical"))
        iid = f"m1.{j}"
        if "ignore" in m:
            out.append({"src": "missed", "iid": iid, "ink": rect, "vertical": vert, "ignore": m["ignore"],
                        "reads": {}, "needs_text": False})
            continue
        ink, reads, pend = reread(iid, rect, vert)
        out.append({"src": "missed", "iid": iid, "ink": ink, "vertical": vert, "cls": m.get("cls"), "reads": reads,
                    "pending": pend, "needs_text": True, "rect": rect})
    for e in prop.get("eval") or []:          # eval's furigana and uncertain lines: ignore areas, not reviewed
        if e["furi"]:
            out.append({"src": "eval-furigana", "iid": f"e{e['eid']}.f", "ink": e["furi"], "vertical": False,
                        "ignore": "furigana", "reads": {}, "needs_text": False, "fixed": True})
    return out


def review_proposals(prop):
    """The numbered proposals a p1b reviewer sees: eval's lines (test frames; uncertain ones are fixed ignore
    areas, not shown as proposals), then MeikiOCR's boxes that are not an eval line's duplicate, numbered in
    reading order (top to bottom, then left to right)."""
    items = []
    for e in prop.get("eval") or []:
        if e["uncertain"]:
            continue
        items.append({"src": "eval", "iid": f"e{e['eid']}", "eid": e["eid"], "ink": list(e["ink"]),
                      "vertical": False, "reads": {}, "eval_text": e["text"], "eval_cls": e["cls"],
                      "furi_unsplit": "furigana" in (e.get("styles") or []) and not e.get("furi")})
    for p in prop["props"]:
        if p.get("dup_eval"):
            continue
        items.append({"src": "meiki", "iid": f"p{p['k']}", "k": p["k"], "ink": list(p["ink"]),
                      "vertical": p["vertical"], "reads": p["reads"], "det_conf": p["det_conf"],
                      "furi_of": p.get("furi_of")})
    items.sort(key=lambda q: (q["ink"][1], q["ink"][0]))
    kmap = {}
    for n, q in enumerate(items, 1):
        q["n"] = n
        if q["src"] == "meiki":
            kmap[q["k"]] = n
    for q in items:
        if q.get("furi_of") is not None:
            q["furi_of_n"] = kmap.get(q["furi_of"])
    return items


def fixed_ignores(prop):
    """Ignore areas a test frame has before any review: eval's uncertain lines (their whole box)."""
    out = []
    for e in prop.get("eval") or []:
        if e["uncertain"]:
            out.append({"src": "eval-uncertain", "iid": f"e{e['eid']}", "ink": xyxy(e["box"]), "vertical": False,
                        "ignore": "uncertain", "fixed": True})
    return out


def load_rereads(split):
    return {(r["shot"], r["iid"]): r for r in jsonl(SETS / split / "rereads.jsonl")}


# ----------------------------------------------------------------------------- p3 decisions

def p3_box_decisions(split):
    """{(kind, shot, ...): (label, decision, item)} of the finished p3 box / text-free items."""
    out = {}
    for t in done_tasks(split, "p3"):
        meta = task_of(t)
        items = {it["X"]: it for it in meta.get("items", [])}
        for label, d in decisions_of(t):
            it = items.get(d.get("item"))
            if it is None:
                continue
            key = ("tf", it["shot"]) if it["kind"] == "tf" else (it["kind"], it["shot"], it["ref"])
            out[key] = (label, d, it)
    return out


# ----------------------------------------------------------------------------- drawing

RULER_L, RULER_T = 34, 16
CLS_COLOUR = {"dialogue": (0, 150, 255), "name": (0, 200, 0), "choice": (230, 0, 230), "ui": (255, 140, 0),
              "scene": (240, 220, 0)}
IGNORE_COLOUR = (150, 150, 150)
_FONTS = {}


def font(size):
    from PIL import ImageFont
    if size not in _FONTS:
        _FONTS[size] = ImageFont.truetype(str(FONT), size)
    return _FONTS[size]


def zoom_of(dims):
    return 2 if dims[0] <= PSP[0] and dims[1] <= PSP[1] else 1


def _rulers(dr, ox, oy, W, H, z):
    f = font(10)
    step = 50 if z == 2 else 100
    for x in range(0, W + 1, 10):
        X = ox + x * z
        ln = 6 if x % step == 0 else 4 if x % 50 == 0 else 2
        dr.line([X, oy - ln, X, oy - 1], fill=(0, 0, 0))
        if x % step == 0 and x < W:
            dr.text((X + 2, oy - RULER_T + 1), str(x), fill=(0, 0, 0), font=f)
    for y in range(0, H + 1, 10):
        Y = oy + y * z
        ln = 6 if y % step == 0 else 4 if y % 50 == 0 else 2
        dr.line([ox - ln, Y, ox - 1, Y], fill=(0, 0, 0))
        if y % step == 0 and y < H:
            dr.text((1, Y + 1), str(y), fill=(0, 0, 0), font=f)


def _dashed(dr, b, colour, dash=4):
    x0, y0, x1, y1 = b
    for x in range(int(x0), int(x1), 2 * dash):
        dr.line([x, y0, min(x + dash, x1), y0], fill=colour)
        dr.line([x, y1, min(x + dash, x1), y1], fill=colour)
    for y in range(int(y0), int(y1), 2 * dash):
        dr.line([x0, y, x0, min(y + dash, y1)], fill=colour)
        dr.line([x1, y, x1, min(y + dash, y1)], fill=colour)


def _tag(dr, b, text, colour, vertical, W_disp, ox, oy):
    """A small number tag next to a display box b (left of a horizontal box, above a vertical one)."""
    f = font(11)
    tw, th = int(dr.textlength(text, font=f)) + 4, 13
    x0, y0, x1, y1 = b
    if vertical:
        tx, ty = (x0 + x1) // 2 - tw // 2, y0 - th - 1
        if ty < oy:
            ty = y1 + 1
    else:
        tx, ty = x0 - tw - 1, (y0 + y1) // 2 - th // 2
        if tx < ox:
            tx, ty = x0, y0 - th - 1
            if ty < oy:
                ty = y1 + 1
    tx = max(ox, min(ox + W_disp - tw, tx))
    dr.rectangle([tx, ty, tx + tw - 1, ty + th - 1], fill=colour)
    lum = 0.3 * colour[0] + 0.59 * colour[1] + 0.11 * colour[2]
    dr.text((tx + 2, ty), text, fill=(0, 0, 0) if lum > 120 else (255, 255, 255), font=f)


def frame_canvas(rgb, marks, title=""):
    """The review image of one frame: the frame with the marks drawn (top) and the clean frame (below), each at
    zoom 1 (2 for a 480x272 frame), with rulers in frame pixels. marks: [{"box": x0 y0 x1 y1 (frame px), "tag",
    "colour", "vertical", "style": "solid" | "dashed"}]."""
    from PIL import Image, ImageDraw
    H, W = rgb.shape[:2]
    z = zoom_of([W, H])
    pw, ph = W * z, H * z
    cw, chh = RULER_L + pw + 4, 2 * (RULER_T + ph) + 8 + (14 if title else 0)
    canvas = Image.new("RGB", (cw, chh), (255, 255, 255))
    dr = ImageDraw.Draw(canvas)
    big = Image.fromarray(rgb).resize((pw, ph), Image.NEAREST) if z != 1 else Image.fromarray(rgb)
    top = 14 if title else 0
    if title:
        dr.text((2, 0), title, fill=(0, 0, 0), font=font(11))
    for k, oy in enumerate((top + RULER_T, top + 2 * RULER_T + ph + 8)):
        canvas.paste(big, (RULER_L, oy))
        _rulers(dr, RULER_L, oy, W, H, z)
        if k == 1:
            continue
        for m in marks:
            x0, y0, x1, y1 = m["box"]
            b = [RULER_L + x0 * z - 2, oy + y0 * z - 2, RULER_L + x1 * z + 1, oy + y1 * z + 1]
            if m.get("style") == "dashed":
                _dashed(dr, b, m["colour"])
            else:
                dr.rectangle([b[0] - 1, b[1] - 1, b[2] + 1, b[3] + 1], outline=(0, 0, 0))
                dr.rectangle(b, outline=m["colour"])
            if m.get("tag"):
                _tag(dr, b, m["tag"], m["colour"], m.get("vertical"), pw, RULER_L, oy)
    return canvas


def text_sheet(entries, title):
    """A contact sheet (<= 1280 x 1280) of instance crops: entries [{"n", "rgb" (the label frame), "ink",
    "vertical"}], each shown with a little context and a thin red frame around its padded ink box (zoom 2 or 3 for
    small text). Returns (PIL image, entries placed); stops when the sheet is full."""
    from PIL import Image, ImageDraw
    SH = 1280
    canvas = Image.new("RGB", (SH, SH), (255, 255, 255))
    dr = ImageDraw.Draw(canvas)
    dr.text((4, 2), title, fill=(0, 0, 0), font=font(12))
    x, y, rowh = 0, 18, 0
    placed = []
    for e in entries:
        rgb, ink, vert = e["rgb"], e["ink"], e["vertical"]
        H, W = rgb.shape[:2]
        pad = PAD[platform_of([W, H])]
        across = (ink[2] - ink[0]) if vert else (ink[3] - ink[1])
        z = 3 if across < 9 else 2 if across < 16 else 1
        ctx = max(4, int(round(0.4 * across)))
        cx0, cy0 = max(0, ink[0] - pad - ctx), max(0, ink[1] - pad - ctx)
        cx1, cy1 = min(W, ink[2] + pad + ctx), min(H, ink[3] + pad + ctx)
        cw, chh = cx1 - cx0, cy1 - cy0
        while z > 1 and (cw * z > SH - 40 or chh * z > SH - 60):
            z -= 1
        crop = Image.fromarray(np.ascontiguousarray(rgb[cy0:cy1, cx0:cx1]))
        if cw > SH - 40 or chh > SH - 60:
            s = min((SH - 40) / cw, (SH - 60) / chh)
            crop = crop.resize((max(1, int(cw * s)), max(1, int(chh * s))), Image.LANCZOS)
            zz = s
        else:
            crop = crop.resize((cw * z, chh * z), Image.NEAREST) if z > 1 else crop
            zz = z
        lab = str(e["n"])
        lw = 26
        ew, eh = lw + crop.width + 10, crop.height + 8
        if x + ew > SH:
            x, y, rowh = 0, y + rowh, 0
        if y + eh > SH:
            break
        dr.text((x + 2, y + max(0, eh // 2 - 8)), lab, fill=(200, 0, 0), font=font(14))
        canvas.paste(crop, (x + lw, y + 4))
        fx0 = x + lw + (ink[0] - pad - cx0) * zz - 1
        fy0 = y + 4 + (ink[1] - pad - cy0) * zz - 1
        fx1 = x + lw + (ink[2] + pad - cx0) * zz
        fy1 = y + 4 + (ink[3] + pad - cy0) * zz
        dr.rectangle([fx0, fy0, fx1, fy1], outline=(255, 0, 0))
        dr.line([x + 1, y + eh - 1, x + ew - 3, y + eh - 1], fill=(220, 220, 220))
        placed.append(e)
        x += ew
        rowh = max(rowh, eh)
        if vert is False and crop.width > SH // 2:      # long lines: one per row
            x, y, rowh = 0, y + rowh, 0
    return canvas, placed


# ----------------------------------------------------------------------------- READMEs (the exact formats)

COMMON = """Conventions (eval/textspot_protocol.json "labels"; do not change them):
- An INSTANCE is one horizontal line of text (characters on one baseline) or one vertical column. A gap wider than
  about 1.5 characters, or a separate UI element (two buttons side by side, a label and its value in separate
  boxes), starts a new instance.
- CLASSES: dialogue = story text (ADV and NVL dialogue and narration, monologue, balloon text); name = speaker name
  plates; choice = in-story choice options; ui = menus, buttons, system and HUD labels, dates, chapter titles, save
  slots, the game's own title on a menu, ...; scene = text that is part of the picture (signs, posters, letters,
  phone or TV screens, ...).
- The BOX is the main-text ink box in frame pixels: fill, outline and shadow. NOT part of it: furigana (small kana
  above a line or right of a column), emphasis dots, page cursors (▼ ▶ ...), icons, button glyphs (○ × △ □ L R).
- IGNORE AREAS (neither found nor missed), why = one of: furigana (the ruby over a line) | logo (logos, stylised art
  text) | small (text under 6 px tall in frame pixels) | unreadable | watermark (watermarks, capture overlays) |
  script (scripts other than Japanese and Latin) | rotated (rotated or curved text).
- TEXT (eval's convention): the main text only; full-width ！ ？ ～; … for an ellipsis (one … per three dots); no
  spaces; keep 「」『』（）【】 and every punctuation mark on screen; Latin letters and digits as written. Small kana
  (ゃ っ ァ), dakuten / handakuten (は ば ぱ), ー vs 一, 、 vs 。 and similar kanji matter: check each one. Never
  guess: mark it unsure instead.
- Frame pixels: the rulers on the image count frame pixels (a 480x272 PSP frame is shown at 2x, so its rulers count
  0..480 over 960 screen pixels). A box is [x, y, w, h] with (x, y) its top-left corner, in frame pixels.
"""

README_P1B = """# textspot review: pass 1, BOXES ({split}, task {batch}, {n} frames)

Each `Fxxx.png` shows ONE frame of a PS Vita / PSP / PC visual-novel screenshot TWICE: on top with MeikiOCR's
proposals drawn as thin numbered boxes (magenta = horizontal, orange = vertical, yellow = small kana that may be
furigana{eval_note}), below the same frame clean. `Fxxx.json` (read it as text first) lists every proposal: its
number `n`, box [x, y, w, h] (frame pixels), orientation and the engines' readings (a hint of what the box holds;
the readings may be wrong).

{common}
For each frame decide EVERY proposal, add every MISSED text instance, and write ONE line to
`decisions-<your label>.jsonl` in this directory (append; one JSON object per frame), e.g.

{{"frame": "F001", "screen": true, "text_free": false, "boxes": [{{"n": 1, "cls": "name"}}, {{"n": 2, "cls": "dialogue"}}, {{"n": 3, "drop": true}}], "missed": [{{"box": [880, 500, 60, 22], "cls": "ui"}}]}}

"boxes" holds one entry per proposal n (each n exactly once):
  {{"n": 1, "cls": "dialogue"}}                      keep it: one instance, box right, with its class
  {{"n": 1, "cls": "ui", "vertical": true}}          keep, but the orientation is wrong (true = a vertical column)
  {{"n": 1, "cls": "dialogue", "box": [x, y, w, h]}} keep, but the box is wrong (cuts characters off, or holds
                                                    furigana, an icon or other text): a rough rectangle around
                                                    the right main text (a few px loose is fine; code shrinks it
                                                    to the ink) that holds no other text
  {{"n": 2, "drop": true}}                           not text (an icon, a pattern, a face), or nothing there
  {{"n": 3, "ignore": "furigana"}}                   an ignore area (why: furigana | logo | small | unreadable |
                                                    watermark | script | rotated)
  {{"n": 4, "split": [312], "cls": ["ui", "ui"]}}    more than one instance: cut positions in frame px (x for a
                                                    horizontal box, y for a vertical one), one class per piece
  {{"n": 5, "merge": [6, 7], "cls": "dialogue"}}     5, 6 and 7 are pieces of ONE instance (do not list 6 and 7
                                                    again)
"missed" (may be []): every text instance or ignore area no proposal covers, as a rough rectangle:
  {{"box": [x, y, w, h], "cls": "ui"}}   (+ "vertical": true for a column)   or   {{"box": [x, y, w, h], "ignore": "logo"}}
"screen": false when the frame is NOT a game screen (box art, a photo, a mostly unreadable thumbnail); then
  nothing else is needed: {{"frame": "F002", "screen": false}}
"text_free": true when the frame holds NO text at all (then every proposal is "drop" and "missed" is []); false
  otherwise. Required on frames marked "text_free_candidate" in their JSON; optional elsewhere.
{eval_rules}
Be thorough: every readable line of every kind counts (menus, buttons, dates, signs, small UI labels). One frame
per tool call, never several images in one call; append the frame's line before viewing the next frame. After
each line, `{check}` reports format problems; fix them by appending a corrected line for that frame (the last line
per frame counts).
"""  # noqa: E501

EVAL_NOTE = ", green = eval's hand-checked line, grey dashed = a fixed ignore area"
EVAL_RULES = """
Test frames: green boxes are eval's hand-checked dialogue / name lines; their TEXT is fixed (in the JSON) and is not
reviewed again, but their BOX and CLASS are: keep them with their class (eval's guess is in the JSON; an in-story
choice option is "choice"), or fix the box. Never split or merge an eval line. Grey dashed areas are fixed ignore
areas (eval's uncertain lines, the furigana of eval's lines): not numbered, nothing to decide.
"""

README_TEXT = """# textspot review: {title} ({split}, task {batch}, {n} entries on {sheets} sheets)

Each sheet `Sxxx.png` (<= 1280 x 1280 px) shows numbered entries. An entry is one text INSTANCE of a visual-novel
frame (a horizontal line, or a vertical column shown upright: read it top to bottom), with a little of the screen
around it; the thin red frame marks the instance that gets the label (text around the frame is context, NOT part
of the label). Small text is shown at 2x or 3x. `Sxxx.json` lists each entry's numbered candidate readings{cand_note};
any of them, or none, may be right.

{common}
Write `decisions-<your label>.jsonl` in this directory: one JSON object per entry, for every entry of every sheet:

{{"sheet": "S001", "n": 1, "pick": 2}}              candidate 2 is EXACTLY right (every character)
{{"sheet": "S001", "n": 2, "text": "「ええと……」"}}     the correct text, typed (when no candidate is exactly right)
{{"sheet": "S001", "n": 3, "unsure": true}}         you cannot settle it from the image
{{"sheet": "S001", "n": 4, "bad": "cut"}}           the red frame is wrong: cut (it cuts off part of the main text)
                                                   | two (it holds more than one instance) | not_text | furigana
                                                   (only furigana inside)

A candidate that is right except for one character is NOT right: type the corrected text. The text is the main text
only (never furigana, never the context outside the frame). View each sheet image once, one image per tool call;
append the sheet's decisions before viewing the next sheet. `{check}` reports format problems.
"""

README_P2B = """# textspot review: pass 2, BOX CHECK ({split}, task {batch}, {n} frames)

Each `Fxxx.png` shows ONE frame TWICE: on top with the first reviewer's instances drawn as thin numbered boxes,
coloured by class (blue = dialogue, green = name, magenta = choice, orange = ui, yellow = scene; a "v" after the
number = a vertical column) and its ignore areas as grey dashed boxes (tag "i" + number); below, the same frame
clean. `Fxxx.json` lists them: n, box [x, y, w, h] (frame pixels), class or ignore why, orientation. Some frames
are text-free candidates (no instances drawn): there the only question is whether the frame holds ANY text.

{common}
Check every box: is it one instance, is the box right (the main-text ink; nothing cut off; no furigana, icon or
other text inside), the class and the orientation; is every ignore area right; is any text instance or ignore
area MISSING? Write ONE line per frame to `decisions-<your label>.jsonl` (append):

{{"frame": "F001", "screen": true, "ok": true}}                     every box and ignore area is right, none missing
{{"frame": "F002", "screen": true, "flags": [ ... ], "missed": [ ... ]}}   otherwise (boxes not flagged count as right)

flags (one per wrong box n; give the RIGHT value):
  {{"n": 3, "cls": "ui"}}                         wrong class
  {{"n": 3, "vertical": true}}                    wrong orientation
  {{"n": 3, "box": [x, y, w, h], "cuts": false}}  wrong box: a rough rectangle of the right one; "cuts": true when the
                                                 drawn box cuts characters off (then also "text": its full text)
  {{"n": 3, "not_text": true}}                    not text: should not be an instance
  {{"n": 3, "ignore": "logo"}}                    should be an ignore area (why as above)
  {{"n": 3, "unignore": true, "cls": "ui", "text": "設定"}}   an ignore area that is in fact a readable instance
  {{"n": 3, "split": [312], "cls": ["ui", "ui"], "texts": ["セーブ", "ロード"]}}   more than one instance (cut positions)
  {{"n": 3, "merge": [4], "cls": "dialogue", "text": "..."}}   3 and 4 are pieces of one instance (its full text)
  Flags may combine keys for one n, e.g. {{"n": 3, "cls": "ui", "vertical": false}}.
missed: {{"box": [x, y, w, h], "cls": "ui", "text": "設定"}} (+ "vertical": true)  or  {{"box": [x, y, w, h], "ignore": "small"}}
"screen": false when the frame is not a game screen. "text_free": true / false is REQUIRED on text-free candidates
(true = no text at all anywhere in the frame).
{eval_rules}
One frame per tool call; append the frame's line before viewing the next. `{check}` reports format problems.
"""  # noqa: E501

EVAL_RULES_P2 = """
Test frames: some boxes are eval's hand-checked lines (their text is fixed and not shown); check their box and
class like any other. Never split or merge them. Grey dashed boxes without a number are fixed ignore areas.
"""

README_P3 = """# textspot review: THIRD LOOK ({split}, task {batch}: {nx} box items, {ns} text sheets)

Two earlier reviewers disagreed about these items; you decide. You see no one's name; the order of the versions
is arbitrary.

{common}
BOX ITEMS `Xxxx.png` + `Xxxx.json`: one frame shown twice (on top with the versions drawn, below clean). Version A
is drawn in cyan (solid), version B in yellow (dashed); each version is one or more boxes or ignore areas, described
in the JSON (class, orientation, ignore why). Decide which version is RIGHT under the conventions:
  {{"item": "X001", "pick": "A"}}  or  "B"  or  {{"item": "X001", "pick": "neither"}}
When you pick a version whose JSON says "needs_text": k, add "texts": [k strings], the full text of each of its
instances that needs one, in the JSON's order: {{"item": "X002", "pick": "B", "texts": ["設定"]}}.
TEXT-FREE ITEMS (kind "tf"): is there any text at all in the frame? {{"item": "X003", "text_free": true}} (no text
anywhere) or false.
TEXT SHEETS `Sxxx.png` + `Sxxx.json`: as in the text passes: {{"sheet": "S001", "n": 1, "pick": 2}} | "text": "..." |
"unsure": true | "bad": "cut" | "two" | "not_text" | "furigana". The candidates are unattributed and shuffled.

Write `decisions-<your label>.jsonl` here, one line per item / entry; one image per tool call; append after each
image. `{check}` reports format problems.
"""


# ----------------------------------------------------------------------------- task builders

TOOL = f"{ROOT}/.venv-ci/bin/python {ROOT}/eval/textspot_sets.py"
BATCH_FRAMES = 16
SHEET_MAX = 40


def check_cmd(split, stage, batch):
    return f"{TOOL} check --task {split}/{stage}/{batch} --label <your label>"


def next_batch_name(split, stage):
    d = review_dir(split, stage)
    n = len([p for p in d.iterdir() if p.is_dir()]) if d.exists() else 0
    return f"b{n + 1:03d}"


def shots_in_tasks(split, stage):
    out = {}
    for t in tasks(split, stage):
        for s in task_of(t).get("shots", []):
            out.setdefault(s, t.name)
    return out


def platform_label(dims, part):
    if part == "adrenaline":
        return "PSP 480x272 (an Adrenaline frame, labelled at its native size)"
    if list(dims) == list(PSP):
        return "PSP 480x272"
    if list(dims) == list(VITA):
        return "PS Vita 960x544"
    return f"PC screenshot scaled to {dims[0]}x{dims[1]}"


def write_task(split, stage, batch, meta, readme):
    d = review_dir(split, stage, batch)
    d.mkdir(parents=True, exist_ok=True)
    (d / "README.md").write_text(readme, encoding="utf-8")
    meta = dict(meta, stage=stage, split=split, batch=batch, created=utc())
    write_json(d / "task.json", meta)
    return d


def _readings(reads):
    out = {}
    for k in ("meiki", "meikic", "v6s"):
        v = reads.get(k)
        if v:
            out[k] = v
    return out


def make_p1b(split, shots, ctx, size=BATCH_FRAMES):
    """New p1b tasks for `shots` (shot keys in review order) not yet in any p1b task."""
    have = shots_in_tasks(split, "p1b")
    todo = [s for s in shots if s not in have]
    made = []
    for i in range(0, len(todo), size):
        part = todo[i:i + size]
        batch = next_batch_name(split, "p1b")
        d = review_dir(split, "p1b", batch)
        d.mkdir(parents=True, exist_ok=True)
        frames = []
        for j, s in enumerate(part, 1):
            F = f"F{j:03d}"
            shot, prop = ctx["shots"][s], ctx["props"][s]
            rgb = load_label_frame(shot)
            plist = review_proposals(prop)
            marks = []
            for q in plist:
                col = ((0, 200, 0) if q["src"] == "eval" else (240, 220, 0) if q.get("furi_of_n") else
                       (255, 140, 0) if q["vertical"] else (230, 0, 230))
                marks.append({"box": q["ink"], "tag": str(q["n"]), "colour": col, "vertical": q["vertical"]})
            fixed = fixed_ignores(prop) + [{"ink": e["furi"], "ignore": "furigana"}
                                           for e in prop.get("eval") or [] if e["furi"]]
            for f in fixed:
                marks.append({"box": f["ink"], "colour": IGNORE_COLOUR, "style": "dashed"})
            part_ = ctx["part_of"].get(s, "text-free candidate")
            tfc = s in ctx["tf_cands"]
            img = frame_canvas(rgb, marks, title=f"{F}  {platform_label(prop['dims'], part_)}"
                               + ("  (text-free candidate)" if tfc else ""))
            img.save(d / f"{F}.png")
            js = {"frame": F, "shot": s, "platform": platform_label(prop["dims"], part_), "dims": prop["dims"],
                  "zoom": zoom_of(prop["dims"]), "text_free_candidate": tfc, "proposals": []}
            for q in plist:
                e = {"n": q["n"], "box": xywh(q["ink"]), "vertical": q["vertical"], "src": q["src"]}
                if q["src"] == "eval":
                    e.update(eval_text=q["eval_text"], eval_class=q["eval_cls"])
                    if q.get("furi_unsplit"):
                        e["hint"] = ("eval tags this line with furigana, and no ruby was cut off its box: if the box "
                                     "holds the ruby, give the main text's box and add the ruby to missed as "
                                     "{\"box\": [...], \"ignore\": \"furigana\"}")
                else:
                    e["readings"] = _readings(q["reads"])
                    if q.get("furi_of_n"):
                        e["hint"] = f"small kana just above proposal {q['furi_of_n']}: furigana?"
                js["proposals"].append(e)
            js["fixed_ignore"] = [{"box": xywh(f["ink"]), "why": f["ignore"]} for f in fixed]
            write_review_json(d / f"{F}.json", js)
            frames.append({"F": F, "shot": s, "n": [q["iid"] for q in plist], "tfc": tfc})
        ev = any(ctx["props"][s].get("eval") for s in part)
        readme = README_P1B.format(split=split, batch=batch, n=len(part), common=COMMON,
                                   eval_note=EVAL_NOTE if ev else "", eval_rules=EVAL_RULES if ev else "",
                                   check=check_cmd(split, "p1b", batch))
        write_task(split, "p1b", batch, {"frames": frames, "shots": part, "images": len(part)}, readme)
        made.append(batch)
    return made


def text_entries(split, shot, prop, insts):
    """The text-review entries of a shot's pass-1 instances: [{"key", "shot", "iid", "ink", "vertical", "reads",
    "src"}] for every instance that needs a text."""
    out = []
    for q in insts:
        if q.get("needs_text") and "ignore" not in q:
            out.append({"key": f"{shot}|{q['iid']}", "shot": shot, "iid": q["iid"], "ink": q["ink"],
                        "vertical": q["vertical"], "reads": q.get("reads") or {}, "src": q["src"]})
    return out


def candidates_of(e, mode, extra=()):
    """Numbered candidates of a text entry: pass 1 attributed (most supported first); pass 2 shuffled and
    unattributed, none for a missed instance; third look: `extra` (the two looks' texts) and the engines',
    shuffled and unattributed."""
    order = ("meiki", "meikic", "v6s")
    sup = {}
    if not (mode == "p2t" and e["src"] == "missed"):
        for k in order:
            t = canon(e["reads"].get(k))
            if t:
                sup.setdefault(t, []).append(k)
    for t in extra:
        t = canon(t)
        if t:
            sup.setdefault(t, [])
    items = sorted(sup.items(), key=lambda kv: (-len(kv[1]), min([order.index(k) for k in kv[1]] or [9])))
    if mode != "p1t":
        random.Random(e["key"] + mode).shuffle(items)
    return [{"c": i + 1, "text": t} | ({"engines": ks} if mode == "p1t" else {}) for i, (t, ks) in enumerate(items)]


def write_text_task(split, stage, entries, ctx, title, readme_title, cand_note, extra=None, items=None,
                    item_images=0):
    """Contact sheets of text entries (horizontal ones, then vertical ones on their own sheets) -> a task dir."""
    batch = next_batch_name(split, stage)
    d = review_dir(split, stage, batch)
    d.mkdir(parents=True, exist_ok=True)
    frames_rgb = {}
    sheets = []
    for vert in (False, True):
        todo = [e for e in entries if bool(e["vertical"]) == vert]
        while todo:
            name = f"S{len(sheets) + 1:03d}"
            batch_e = []
            for i, e in enumerate(todo[:SHEET_MAX]):
                if e["shot"] not in frames_rgb:
                    frames_rgb[e["shot"]] = load_label_frame(ctx["shots"][e["shot"]])
                batch_e.append(dict(e, n=i + 1, rgb=frames_rgb[e["shot"]]))
            img, placed = text_sheet(batch_e, f"{name}  {title}")
            if not placed:
                raise RuntimeError(f"entry {todo[0]['key']} does not fit a sheet")
            img.save(d / f"{name}.png")
            ents = []
            for e in placed:
                ex = (extra or {}).get(e["key"], ())
                ents.append({"n": e["n"], "key": e["key"], "candidates": candidates_of(e, stage, ex)})
            write_review_json(d / f"{name}.json", {"sheet": name, "entries": ents})
            sheets.append(name)
            todo = todo[len(placed):]
    readme = README_TEXT.format(title=readme_title, split=split, batch=batch, n=len(entries), sheets=len(sheets),
                                common=COMMON, cand_note=cand_note, check=check_cmd(split, stage, batch))
    shots = sorted({e["shot"] for e in entries})
    meta = {"sheets": sheets, "keys": [e["key"] for e in entries], "shots": shots,
            "images": len(sheets) + item_images}
    if items is not None:
        meta["items"] = items
    write_task(split, stage, batch, meta, readme)
    return batch


def make_p2b(split, shots, ctx, size=BATCH_FRAMES):
    have = shots_in_tasks(split, "p2b")
    todo = [s for s in shots if s not in have]
    made = []
    for i in range(0, len(todo), size):
        part = todo[i:i + size]
        batch = next_batch_name(split, "p2b")
        d = review_dir(split, "p2b", batch)
        d.mkdir(parents=True, exist_ok=True)
        frames = []
        for j, s in enumerate(part, 1):
            F = f"F{j:03d}"
            shot, prop = ctx["shots"][s], ctx["props"][s]
            rgb = load_label_frame(shot)
            insts = ctx["inst1"][s]
            fixed = fixed_ignores(prop)
            shown = [q for q in insts if not q.get("fixed")]
            fixed += [q for q in insts if q.get("fixed")]
            marks, js_inst = [], []
            for n, q in enumerate(shown, 1):
                if "ignore" in q:
                    marks.append({"box": q["ink"], "tag": f"i{n}", "colour": IGNORE_COLOUR, "style": "dashed"})
                    js_inst.append({"n": n, "box": xywh(q["ink"]), "ignore": q["ignore"], "vertical": q["vertical"]})
                else:
                    tag = f"{n}v" if q["vertical"] else str(n)
                    marks.append({"box": q["ink"], "tag": tag, "colour": CLS_COLOUR[q["cls"]],
                                  "vertical": q["vertical"]})
                    e = {"n": n, "box": xywh(q["ink"]), "cls": q["cls"], "vertical": q["vertical"]}
                    if q["src"] == "eval":
                        e["eval_line"] = True
                    js_inst.append(e)
            for f in fixed:
                marks.append({"box": f["ink"], "colour": IGNORE_COLOUR, "style": "dashed"})
            part_ = ctx["part_of"].get(s, "text-free candidate")
            tfc = s in ctx["tf_cands"] and s not in ctx["part_of_main"]
            img = frame_canvas(rgb, marks, title=f"{F}  {platform_label(prop['dims'], part_)}"
                               + ("  (text-free candidate)" if tfc else ""))
            img.save(d / f"{F}.png")
            write_review_json(d / f"{F}.json", {"frame": F, "shot": s, "platform": platform_label(prop["dims"], part_),
                                         "dims": prop["dims"], "zoom": zoom_of(prop["dims"]),
                                         "text_free_candidate": tfc, "instances": js_inst,
                                         "fixed_ignore": [{"box": xywh(f["ink"]), "why": f["ignore"]}
                                                          for f in fixed]})
            frames.append({"F": F, "shot": s, "n": [q["iid"] for q in shown], "tfc": tfc})
        ev = any(ctx["props"][s].get("eval") for s in part)
        readme = README_P2B.format(split=split, batch=batch, n=len(part), common=COMMON,
                                   eval_rules=EVAL_RULES_P2 if ev else "", check=check_cmd(split, "p2b", batch))
        write_task(split, "p2b", batch, {"frames": frames, "shots": part, "images": len(part)}, readme)
        made.append(batch)
    return made


# ----------------------------------------------------------------------------- validation

def _is_box(b, dims):
    return (isinstance(b, list) and len(b) == 4 and all(isinstance(v, (int, float)) for v in b)
            and b[2] > 0 and b[3] > 0 and b[0] >= -2 and b[1] >= -2 and b[0] + b[2] <= dims[0] + 2
            and b[1] + b[3] <= dims[1] + 2)


def _cls_ok(c, n_parts=1):
    if isinstance(c, list):
        return len(c) == n_parts and all(x in CLASSES for x in c)
    return c in CLASSES


def validate_p1b(meta, task_dir, decs):
    errs = []
    last = {}
    for label, d in decs:
        if "_bad_line" in d:
            errs.append(f"{label}: line {d['_bad_line']} is not JSON")
            continue
        last[d.get("frame")] = d
    for f in meta["frames"]:
        F = f["F"]
        d = last.get(F)
        if d is None:
            errs.append(f"{F}: no decision")
            continue
        js = json.loads((task_dir / f"{F}.json").read_text(encoding="utf-8"))
        dims = js["dims"]
        if d.get("screen") is False:
            continue
        if d.get("screen") is not True:
            errs.append(f"{F}: \"screen\" must be true or false")
        props = {p["n"]: p for p in js["proposals"]}
        seen = {}
        for b in d.get("boxes", []) or []:
            n = b.get("n")
            if n not in props:
                errs.append(f"{F}: box n={n} is not a proposal")
                continue
            for m in [n] + list(b.get("merge", []) or []):
                if m in seen:
                    errs.append(f"{F}: proposal {m} decided twice")
                seen[m] = True
            acts = [k for k in ("drop", "ignore", "split", "merge") if b.get(k)]
            if len(acts) > 1:
                errs.append(f"{F}: n={n} has more than one of drop / ignore / split / merge")
            if b.get("drop"):
                continue
            if "ignore" in b:
                if b["ignore"] not in REVIEW_WHY:
                    errs.append(f"{F}: n={n} ignore why {b['ignore']!r} not in {REVIEW_WHY}")
                continue
            p = props[n]
            if p["src"] == "eval" and (b.get("split") or b.get("merge")):
                errs.append(f"{F}: n={n} is an eval line: never split or merge it")
            if b.get("split"):
                cuts = b["split"]
                vert = bool(b.get("vertical", p["vertical"]))
                x, y, w, h = p["box"]
                lo, hi = (y, y + h) if vert else (x, x + w)
                if not isinstance(cuts, list) or not cuts or not all(isinstance(c, (int, float)) and lo < c < hi
                                                                     for c in cuts):
                    errs.append(f"{F}: n={n} split cuts must lie inside the box ({lo}..{hi})")
                elif not _cls_ok(b.get("cls"), len(cuts) + 1) and b.get("cls") not in CLASSES:
                    errs.append(f"{F}: n={n} split needs cls: one class or a list of {len(cuts) + 1}")
                continue
            if b.get("merge"):
                if not all(m in props and m != n for m in b["merge"]):
                    errs.append(f"{F}: n={n} merge members must be other proposals")
            if b.get("cls") not in CLASSES:
                errs.append(f"{F}: n={n} needs cls in {CLASSES} (or drop / ignore)")
            if "box" in b and not _is_box(b["box"], dims):
                errs.append(f"{F}: n={n} box must be [x, y, w, h] inside the frame")
            if "vertical" in b and not isinstance(b["vertical"], bool):
                errs.append(f"{F}: n={n} vertical must be true or false")
        missing = [n for n in props if n not in seen]
        if missing:
            errs.append(f"{F}: proposals without a decision: {missing}")
        for m in d.get("missed", []) or []:
            if not _is_box(m.get("box"), dims):
                errs.append(f"{F}: missed box {m.get('box')} must be [x, y, w, h] inside the frame")
            elif "ignore" in m:
                if m["ignore"] not in REVIEW_WHY:
                    errs.append(f"{F}: missed ignore why {m['ignore']!r} not in {REVIEW_WHY}")
            elif m.get("cls") not in CLASSES:
                errs.append(f"{F}: missed entry needs cls or ignore")
        if js.get("text_free_candidate") and not isinstance(d.get("text_free"), bool):
            errs.append(f"{F}: text-free candidate: \"text_free\" must be true or false")
        if d.get("text_free") is True:
            if any(not b.get("drop") for b in d.get("boxes", []) or []) or any(
                    "cls" in m for m in d.get("missed", []) or []):
                errs.append(f"{F}: text_free true but some box or missed entry is kept as text")
    return errs


def validate_text(meta, task_dir, decs, stage):
    errs = []
    last = {}
    for label, d in decs:
        if "_bad_line" in d:
            errs.append(f"{label}: line {d['_bad_line']} is not JSON")
            continue
        if "sheet" in d:
            last[(d.get("sheet"), d.get("n"))] = d
    for name in meta.get("sheets", []):
        sh = json.loads((task_dir / f"{name}.json").read_text(encoding="utf-8"))
        for e in sh["entries"]:
            d = last.get((name, e["n"]))
            if d is None:
                errs.append(f"{name} n={e['n']}: no decision")
                continue
            ncand = len(e["candidates"])
            keys = [k for k in ("pick", "text", "unsure", "bad") if k in d]
            if len(keys) != 1:
                errs.append(f"{name} n={e['n']}: exactly one of pick / text / unsure / bad")
            elif "pick" in d and not (isinstance(d["pick"], int) and 1 <= d["pick"] <= ncand):
                errs.append(f"{name} n={e['n']}: pick must be 1..{ncand}")
            elif "text" in d and not (isinstance(d["text"], str) and canon(d["text"])):
                errs.append(f"{name} n={e['n']}: text must be a non-empty string")
            elif "bad" in d and d["bad"] not in ("cut", "two", "not_text", "furigana"):
                errs.append(f"{name} n={e['n']}: bad must be cut | two | not_text | furigana")
    return errs


def validate_p2b(meta, task_dir, decs):
    errs = []
    last = {}
    for label, d in decs:
        if "_bad_line" in d:
            errs.append(f"{label}: line {d['_bad_line']} is not JSON")
            continue
        last[d.get("frame")] = d
    for f in meta["frames"]:
        F = f["F"]
        d = last.get(F)
        if d is None:
            errs.append(f"{F}: no decision")
            continue
        js = json.loads((task_dir / f"{F}.json").read_text(encoding="utf-8"))
        dims = js["dims"]
        if js.get("text_free_candidate") and not isinstance(d.get("text_free"), bool):
            errs.append(f"{F}: text-free candidate: \"text_free\" must be true or false")
        if d.get("screen") is False:
            continue
        if d.get("screen") is not True:
            errs.append(f"{F}: \"screen\" must be true or false")
        insts = {q["n"]: q for q in js["instances"]}
        if d.get("ok") is True:
            if d.get("flags") or d.get("missed"):
                errs.append(f"{F}: \"ok\": true with flags or missed entries")
            continue
        if "flags" not in d and "missed" not in d and "text_free" not in d:
            errs.append(f"{F}: needs \"ok\": true, or \"flags\" / \"missed\"")
        seen = set()
        for g in d.get("flags", []) or []:
            n = g.get("n")
            if n not in insts:
                errs.append(f"{F}: flag n={n} is not an instance")
                continue
            if n in seen:
                errs.append(f"{F}: n={n} flagged twice (combine the keys in one flag)")
            seen.add(n)
            q = insts[n]
            keys = set(g) - {"n", "note"}
            if not keys:
                errs.append(f"{F}: flag n={n} says nothing")
            if "cls" in g and not _cls_ok(g["cls"], len(g.get("split", [])) + 1) and g["cls"] not in CLASSES:
                errs.append(f"{F}: n={n} cls must be in {CLASSES}")
            if "box" in g:
                if not _is_box(g["box"], dims):
                    errs.append(f"{F}: n={n} box must be [x, y, w, h] inside the frame")
                if g.get("cuts") and not (isinstance(g.get("text"), str) and canon(g["text"])) and not q.get(
                        "eval_line") and "ignore" not in q:
                    errs.append(f"{F}: n={n} cuts: true needs \"text\": the instance's full text")
            if "ignore" in g and g["ignore"] not in REVIEW_WHY:
                errs.append(f"{F}: n={n} ignore why must be in {REVIEW_WHY}")
            if g.get("unignore"):
                if "ignore" not in q:
                    errs.append(f"{F}: n={n} unignore is for ignore areas")
                if g.get("cls") not in CLASSES or not (isinstance(g.get("text"), str) and canon(g["text"])):
                    errs.append(f"{F}: n={n} unignore needs cls and text")
            if g.get("split"):
                if q.get("eval_line"):
                    errs.append(f"{F}: n={n} is an eval line: never split it")
                cuts = g["split"]
                x, y, w, h = q["box"]
                vert = bool(g.get("vertical", q["vertical"]))
                lo, hi = (y, y + h) if vert else (x, x + w)
                if not isinstance(cuts, list) or not all(isinstance(c, (int, float)) and lo < c < hi for c in cuts):
                    errs.append(f"{F}: n={n} split cuts must lie inside the box ({lo}..{hi})")
                else:
                    txt = g.get("texts")
                    if not (isinstance(txt, list) and len(txt) == len(cuts) + 1 and all(
                            isinstance(t, str) and canon(t) for t in txt)):
                        errs.append(f"{F}: n={n} split needs \"texts\": {len(cuts) + 1} strings")
            if g.get("merge"):
                if q.get("eval_line"):
                    errs.append(f"{F}: n={n} is an eval line: never merge it")
                if not all(m in insts and m != n for m in g["merge"]):
                    errs.append(f"{F}: n={n} merge members must be other instances")
                if not (isinstance(g.get("text"), str) and canon(g["text"])):
                    errs.append(f"{F}: n={n} merge needs \"text\": the full text")
            if "vertical" in g and not isinstance(g["vertical"], bool):
                errs.append(f"{F}: n={n} vertical must be true or false")
        for m in d.get("missed", []) or []:
            if not _is_box(m.get("box"), dims):
                errs.append(f"{F}: missed box {m.get('box')} must be [x, y, w, h] inside the frame")
            elif "ignore" in m:
                if m["ignore"] not in REVIEW_WHY:
                    errs.append(f"{F}: missed ignore why must be in {REVIEW_WHY}")
            elif m.get("cls") not in CLASSES or not (isinstance(m.get("text"), str) and canon(m["text"])):
                errs.append(f"{F}: missed entry needs cls and text (or ignore)")
    return errs


def validate_p3(meta, task_dir, decs):
    errs = validate_text(meta, task_dir, decs, "p3")
    last = {}
    for label, d in decs:
        if "item" in d:
            last[d["item"]] = d
    for it in meta.get("items", []):
        d = last.get(it["X"])
        if d is None:
            errs.append(f"{it['X']}: no decision")
            continue
        if it["kind"] == "tf":
            if not isinstance(d.get("text_free"), bool):
                errs.append(f"{it['X']}: text_free must be true or false")
            continue
        if d.get("pick") not in ("A", "B", "neither"):
            errs.append(f"{it['X']}: pick must be A, B or neither")
            continue
        need = it.get("needs_text", {}).get(d["pick"], 0) if d["pick"] in ("A", "B") else 0
        if need:
            t = d.get("texts")
            if not (isinstance(t, list) and len(t) == need and all(isinstance(x, str) and canon(x) for x in t)):
                errs.append(f"{it['X']}: pick {d['pick']} needs \"texts\": {need} strings")
    return errs


def validate(task_dir):
    meta = task_of(task_dir)
    decs = decisions_of(task_dir)
    st = meta["stage"]
    if st == "p1b":
        return validate_p1b(meta, task_dir, decs)
    if st == "p2b":
        return validate_p2b(meta, task_dir, decs)
    if st == "p3":
        return validate_p3(meta, task_dir, decs)
    return validate_text(meta, task_dir, decs, st)


# ----------------------------------------------------------------------------- the reviewers' commands

STAGE_PRIO = {"p3": 0, "p1b": 1, "p2b": 2, "p1t": 3, "p2t": 4}
PASS_OF = {"p1b": "p1", "p1t": "p1", "p2b": "p2", "p2t": "p2", "p3": "p3"}
SPLIT_PRIO = {"dev": 0, "test": 1}


def all_tasks():
    out = []
    for split in ("dev", "test"):
        for stage in STAGES:
            for t in tasks(split, stage):
                out.append((split, stage, t))
    return out


def roles(cl, metas):
    """{(split, shot): {"p1": labels, "p2": labels, "p3": labels}} of every claimed task."""
    out = collections.defaultdict(lambda: {"p1": set(), "p2": set(), "p3": set()})
    for (split, stage, t), meta in metas.items():
        c = cl.get(task_key(split, stage, t.name))
        if not c:
            continue
        for s in meta.get("shots", []):
            out[(split, s)][PASS_OF[stage]].add(c["label"])
    return out


def allowed(label, split, stage, meta, rl):
    mine = PASS_OF[stage]
    others = {"p1": ("p2", "p3"), "p2": ("p1", "p3"), "p3": ("p1", "p2")}[mine]
    for s in meta.get("shots", []):
        r = rl.get((split, s))
        if r and any(label in r[o] for o in others):
            return False
    return True


def feed_open():
    st = SETS / "feed_state.json"
    if not st.exists():
        return True
    return bool(json.loads(st.read_text(encoding="utf-8")).get("open", True))


def cmd_claim(args):
    label = args.label
    with locked(SETS / "review.lock"):
        cl = claims()
        metas = {(sp, stg, t): task_of(t) for sp, stg, t in all_tasks()}
        for (sp, stg, t), meta in metas.items():          # an unfinished task of this label first (a resumed session)
            c = cl.get(task_key(sp, stg, t.name))
            if c and c["label"] == label and not c["done"]:
                print(f"RESUME {t}")
                print(f"README: {t}/README.md  decisions: {t}/decisions-{label}.jsonl  images: {meta['images']}")
                return
        used = sum(c["images"] for c in cl.values() if c["label"] == label)
        rl = roles(cl, metas)
        free = [(SPLIT_PRIO[sp], STAGE_PRIO[stg], t.name, sp, stg, t) for (sp, stg, t) in metas
                if task_key(sp, stg, t.name) not in cl]
        free.sort()
        fits = False
        for _, _, _, sp, stg, t in free:
            meta = metas[(sp, stg, t)]
            if not allowed(label, sp, stg, meta, rl):
                continue
            if used + meta["images"] > IMAGE_BUDGET:
                fits = fits or False
                continue
            fits = True
            with open(claims_path(), "a", encoding="utf-8") as f:
                f.write(json.dumps({"t": utc(), "label": label, "task": task_key(sp, stg, t.name),
                                    "images": meta["images"], "event": "claim"}) + "\n")
            print(f"TASK {t}")
            print(f"README: {t}/README.md  decisions: {t}/decisions-{label}.jsonl  images: {meta['images']}"
                  f"  (your images so far: {used + meta['images']} of {IMAGE_BUDGET})")
            return
        if any(used + metas[(sp, stg, t)]["images"] <= IMAGE_BUDGET for _, _, _, sp, stg, t in free) or (
                feed_open() and used + 6 <= IMAGE_BUDGET):
            print("WAIT no task for you right now; more are being made. Sleep 120 s and claim again.")
        else:
            print(f"STOP no more tasks for you (images claimed {used} of {IMAGE_BUDGET}, or the review is finished)")


def resolve_task(arg):
    sp, stg, b = arg.strip("/").split("/")[-3:]
    return sp, stg, b, review_dir(sp, stg, b)


def cmd_check(args):
    sp, stg, b, d = resolve_task(args.task)
    errs = validate(d)
    meta = task_of(d)
    n = len(meta.get("frames") or []) + sum(len(json.loads((d / f"{s}.json").read_text(encoding="utf-8"))
                                                ["entries"]) for s in meta.get("sheets", [])) + len(
        meta.get("items") or [])
    if errs:
        print(f"{len(errs)} problems (of {n} frames / entries / items):")
        for e in errs[:40]:
            print("  " + e)
        if len(errs) > 40:
            print(f"  ... {len(errs) - 40} more")
    else:
        print(f"OK: all {n} frames / entries / items of {sp}/{stg}/{b} decided and well-formed")


def cmd_done(args):
    sp, stg, b, d = resolve_task(args.task)
    with locked(SETS / "review.lock"):
        cl = claims()
        c = cl.get(task_key(sp, stg, b))
        if not c or c["label"] != args.label:
            sys.exit(f"{sp}/{stg}/{b} is not claimed by {args.label}")
        if c["done"]:
            print("DONE (already)")
            return
        if not (d / f"decisions-{args.label}.jsonl").exists():
            sys.exit(f"no {d}/decisions-{args.label}.jsonl")
        errs = validate(d)
        if errs:
            print(f"NOT DONE: {len(errs)} problems:")
            for e in errs[:40]:
                print("  " + e)
            sys.exit(1)
        with open(claims_path(), "a", encoding="utf-8") as f:
            f.write(json.dumps({"t": utc(), "label": args.label, "task": task_key(sp, stg, b),
                                "event": "done"}) + "\n")
    print(f"DONE {sp}/{stg}/{b}")


def cmd_release(args):
    """Takes a task back from a reviewer that stopped (its decisions file is kept aside as released-*.jsonl)."""
    sp, stg, b, d = resolve_task(args.task)
    with locked(SETS / "review.lock"):
        c = claims().get(task_key(sp, stg, b))
        if not c:
            sys.exit("not claimed")
        f = d / f"decisions-{c['label']}.jsonl"
        if f.exists():
            os.replace(f, d / f"released-{c['label']}-{int(time.time())}.jsonl")
        with open(claims_path(), "a", encoding="utf-8") as fo:
            fo.write(json.dumps({"t": utc(), "label": c["label"], "task": task_key(sp, stg, b),
                                 "event": "release"}) + "\n")
    print(f"released {sp}/{stg}/{b} from {c['label']}")


def cmd_zoom(args):
    """A closer look at part of a frame of a task: the region [x, y, w, h] (frame pixels) at 2-4x, <= 1280 px."""
    from PIL import Image
    sp, stg, b, d = resolve_task(args.task)
    js = json.loads((d / f"{args.frame}.json").read_text(encoding="utf-8"))
    shot = load_shots(sp)[js["shot"]]
    rgb = load_label_frame(shot)
    H, W = rgb.shape[:2]
    x, y, w, h = args.box
    x0, y0, x1, y1 = clip([x, y, x + w, y + h], W, H)
    if x1 - x0 < 2 or y1 - y0 < 2:
        sys.exit("empty region")
    z = max(1, min(4, 1280 // max(x1 - x0, y1 - y0)))
    img = Image.fromarray(np.ascontiguousarray(rgb[y0:y1, x0:x1])).resize(((x1 - x0) * z, (y1 - y0) * z),
                                                                          Image.NEAREST)
    out = Path("/tmp/textspot-zoom")
    out.mkdir(exist_ok=True)
    p = out / f"{args.label}-{int(time.time() * 1000) % 10**9}.png"
    img.save(p)
    print(f"{p}  ({img.width}x{img.height}: frame region {[x0, y0, x1 - x0, y1 - y0]} at {z}x)")


def cmd_status(args):
    cl = claims()
    rows = collections.Counter()
    for sp, stg, t in all_tasks():
        c = cl.get(task_key(sp, stg, t.name))
        rows[(sp, stg, "done" if c and c["done"] else "claimed" if c else "free")] += 1
    for k in sorted(rows):
        print(k, rows[k])
    by = collections.defaultdict(list)
    for k, c in cl.items():
        by[c["label"]].append((k, "done" if c["done"] else "open", c["images"]))
    for label in sorted(by):
        print(label, sum(x[2] for x in by[label]), "images:", ", ".join(f"{k}({s})" for k, s, _ in by[label]))


# ----------------------------------------------------------------------------- context, pass-2 versions, rereads

def select_test(shots, st):
    """The test frames: every eval screenshot (vita, psp; psp also as the report-only Adrenaline part) and the first
    CAPS text-free candidates by rank that two looks confirm."""
    frames = []
    ev = sorted((s for s in shots.values() if s.get("set")), key=lambda s: s["rank"])
    for s in ev:
        dims, scale = frame_spec(s["kind"], s["src_dims"])
        frames.append({"id": s["shot"], "vn": s["vn"], "part": s["kind"], "shot": s["shot"], "sid": s["sid"],
                       "url": s["url"], "session": s["release"], "sha256": s["sha256"], "src_dims": s["src_dims"],
                       "scale": scale, "dims": dims, "rank": s["rank"], "path": s["path"], "set": s["set"],
                       "text_free": False})
    for s in ev:
        if s["kind"] == "psp":
            dims, scale = frame_spec("adrenaline", s["src_dims"])
            frames.append({"id": s["shot"] + "_x2", "vn": s["vn"], "part": "adrenaline", "shot": s["shot"],
                           "sid": s["sid"], "url": s["url"], "session": s["release"], "sha256": s["sha256"],
                           "src_dims": s["src_dims"], "scale": scale, "dims": dims, "rank": s["rank"],
                           "path": s["path"], "set": s["set"], "text_free": False, "report_only": True})
    cands = sorted((s for s in shots.values() if not s.get("set") and st["dets"].get(s["shot"], 99) <= TF_MAX_DETS),
                   key=lambda s: s["rank"])
    n_tf = 0
    for s in cands:
        if n_tf >= CAPS["text_free"]:
            break
        if st["tf"].get(s["shot"]) == "yes":
            frames.append({"id": s["shot"], "vn": s["vn"], "part": "text-free", "shot": s["shot"], "sid": s["sid"],
                           "url": s["url"], "session": s["release"], "sha256": s["sha256"],
                           "src_dims": s["src_dims"], "scale": None, "dims": s["src_dims"], "rank": s["rank"],
                           "path": s["path"], "set": None, "text_free": True})
            n_tf += 1
    return frames, cands


def build_ctx(split):
    shots = load_shots(split)
    props = load_proposals(split)
    st = review_state(split)
    if split == "dev":
        frames, cands = select_dev(sorted(shots.values(), key=lambda r: r["rank"]), st["drops"], st["tf"],
                                   st["dets"])
    else:
        frames, cands = select_test(shots, st)
    part_of = {}
    for f in frames:
        part_of.setdefault(f["shot"], f["part"])
    main = {f["shot"] for f in frames if f["part"] != "text-free"}
    rereads = load_rereads(split)
    inst1 = {}
    for s, (label, d, t) in st["p1"].items():
        if d.get("screen") is False or s not in props:
            continue
        inst1[s] = p1_instances(s, props[s], d, rereads) + fixed_ignores(props[s])
    return {"split": split, "shots": shots, "props": props, "st": st, "frames": frames, "cands": cands,
            "part_of": part_of, "part_of_main": main, "tf_cands": {c["shot"] for c in cands}, "inst1": inst1,
            "rereads": rereads}


def p2_view(ctx, shot):
    """Pass 2's p2b decision of a shot as versions: {"flags": {iid: {"flag", "B": [instances], "members"}},
    "missed": [{"iid", "B": [instances]}], "dec", "label"} or None. B instances that pass 2 gave a text for carry
    "text2"; B boxes come refined from the rereads (pending until reread)."""
    p2 = ctx["st"]["p2"].get(shot)
    if p2 is None:
        return None
    label, d, t, fmeta = p2
    insts = {q["iid"]: q for q in ctx["inst1"].get(shot, [])}
    n2 = {n + 1: iid for n, iid in enumerate(fmeta["n"])}
    rr = ctx["rereads"]
    out = {"flags": {}, "missed": [], "dec": d, "label": label, "pending": False}

    def rer(iid, rect, vert):
        r = rr.get((shot, iid))
        if r and r["rect"] == list(rect) and bool(r["vertical"]) == bool(vert):
            return r["ink"], r["reads"]
        out["pending"] = True
        out.setdefault("todo", []).append((iid, list(rect), bool(vert)))
        return list(rect), {}
    if d.get("screen") is False or d.get("ok") is True:
        return out
    consumed = set()
    for g in d.get("flags", []) or []:
        iid = n2.get(g.get("n"))
        if iid is None or iid in consumed:
            continue
        A = insts[iid]
        vert = bool(g.get("vertical", A["vertical"]))
        base = {k: A[k] for k in ("src", "eid", "eval_text") if k in A}
        if g.get("not_text"):
            B = []
        elif "ignore" in g:
            B = [dict(base, iid=f"{iid}.B", ink=list(A["ink"]), vertical=vert, ignore=g["ignore"])]
        elif g.get("unignore"):
            ink, reads = rer(f"{iid}.B", A["ink"], vert)
            B = [dict(base, iid=f"{iid}.B", ink=ink, vertical=vert, cls=g["cls"], text2=g["text"], reads=reads)]
        elif g.get("split"):
            parts = split_parts(A["ink"], g["split"], vert)
            cl = g.get("cls") if isinstance(g.get("cls"), list) else [g.get("cls") or A.get("cls")] * len(parts)
            B = []
            for i, p in enumerate(parts):
                ink, reads = rer(f"{iid}.B.s{i}", p, vert)
                B.append(dict(base, iid=f"{iid}.B.s{i}", ink=ink, vertical=vert, cls=cl[min(i, len(cl) - 1)],
                              text2=g["texts"][i], reads=reads))
        elif g.get("merge"):
            members = [n2[m] for m in g["merge"] if m in n2]
            consumed.update(members)
            rect = union([A["ink"]] + [insts[m]["ink"] for m in members])
            ink, reads = rer(f"{iid}.B.m", rect, vert)
            B = [dict(base, iid=f"{iid}.B.m", ink=ink, vertical=vert, cls=g.get("cls") or A.get("cls"),
                      text2=g["text"], reads=reads)]
            out["flags"][iid] = {"flag": g, "B": B, "members": members}
            continue
        else:
            q = dict(base, iid=f"{iid}.B", ink=list(A["ink"]), vertical=vert)
            if "ignore" in A:
                q["ignore"] = A["ignore"]
            else:
                q["cls"] = g.get("cls", A.get("cls"))
            if "box" in g:
                ink, reads = rer(f"{iid}.B", xyxy(g["box"]), vert)
                q["ink"], q["reads"] = ink, reads
                if g.get("cuts") and A["src"] != "eval" and "ignore" not in A:
                    q["text2"] = g["text"]
            B = [q]
        out["flags"][iid] = {"flag": g, "B": B, "members": []}
    for j, m in enumerate(d.get("missed", []) or []):
        iid = f"m2.{j}"
        vert = bool(m.get("vertical"))
        if "ignore" in m:
            B = [{"src": "missed2", "iid": iid, "ink": clip(xyxy(m["box"]), *ctx["props"][shot]["dims"]),
                  "vertical": vert, "ignore": m["ignore"]}]
        else:
            ink, reads = rer(iid, xyxy(m["box"]), vert)
            B = [{"src": "missed2", "iid": iid, "ink": ink, "vertical": vert, "cls": m["cls"], "text2": m["text"],
                  "reads": reads}]
        out["missed"].append({"iid": iid, "B": B})
    return out


def pending_rereads(ctx):
    """(shot, iid, rect, vertical) of every reviewer box not refined and read yet (pass 1 and pass 2)."""
    todo = []
    for s, insts in ctx["inst1"].items():
        for q in insts:
            if q.get("pending"):
                todo.append((s, q["iid"], list(q["rect"]), bool(q["vertical"])))
    for s in ctx["st"]["p2"]:
        v = p2_view(ctx, s)
        for iid, rect, vert in (v or {}).get("todo", []):
            todo.append((s, iid, rect, vert))
    return todo


def cmd_reread(args):
    import cv2
    ctx = build_ctx(args.split)
    todo = pending_rereads(ctx)
    print(f"{len(todo)} boxes to refine and read", flush=True)
    if not todo:
        print("REREAD_DONE 0")
        return
    _init_engines()
    path = SETS / args.split / "rereads.jsonl"
    by = collections.defaultdict(list)
    for t in todo:
        by[t[0]].append(t)
    with open(path, "a", encoding="utf-8") as fo:
        for s, items in by.items():
            rgb = load_label_frame(ctx["shots"][s])
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            pad = PAD[platform_of([rgb.shape[1], rgb.shape[0]])]
            for _, iid, rect, vert in items:
                ink = refine_box(rgb, rect, vert, clamp=".s" in iid)
                reads, _ = read_crop(bgr, ink, vert, pad)
                fo.write(json.dumps({"shot": s, "iid": iid, "rect": rect, "vertical": vert, "ink": ink,
                                     "reads": reads, "rule": 2, "t": utc()}, ensure_ascii=False) + "\n")
    print(f"REREAD_DONE {len(todo)}", flush=True)


# ----------------------------------------------------------------------------- decide

def _agree(t1, t2, z3):
    """Text from two looks (text_of results) and an optional third look: (text | None, outcome)."""
    if t1 is None or t2 is None:
        return None, "open"
    (a, sa), (b, sb) = t1, t2
    if sa == "text" and sb == "text" and norm(a) == norm(b):
        return a, "agree"
    if sa != "text" and sb != "text":
        return None, "both-unsure"
    if z3 is None:
        return None, "need-p3"
    c, sc = z3
    if sc == "text" and sa == "text" and norm(c) == norm(a):
        return a, "p3-first"
    if sc == "text" and sb == "text" and norm(c) == norm(b):
        return b, "p3-second"
    return None, "p3-none"


def decide_split(ctx):
    """Every shot's labels so far: {shot: {"status": final | open | ..., "refs", "open": [items], "stats"}}."""
    split = ctx["split"]
    st = ctx["st"]
    t1s, t2s, t3s = text_decisions(split, "p1t"), text_decisions(split, "p2t"), text_decisions(split, "p3")
    p3 = st["p3"]
    out = {}
    need = set(ctx["part_of"]) | set(st["p1"])
    for s in sorted(need):
        r = {"status": None, "refs": [], "open": [], "stats": collections.Counter()}
        out[s] = r
        p1 = st["p1"].get(s)
        if p1 is None:
            r["status"] = "need-p1b"
            continue
        if s in st["drops"]:
            r["status"] = "dropped"
            continue
        tfs = st["tf"].get(s)
        if ctx["part_of"].get(s) == "text-free" or (s in ctx["tf_cands"] and s not in ctx["part_of_main"]):
            if tfs == "yes":
                r["status"] = "final"
                r["text_free"] = True
            elif tfs == "no":
                r["status"] = "not-text-free"
            else:
                r["status"] = "open"
                if p1[1].get("text_free") is True and s in st["p2"] and st["p2"][s][1].get("text_free") is False:
                    r["open"].append(("tf", s))
            continue
        insts = ctx["inst1"].get(s, [])
        if any(q.get("pending") for q in insts):
            r["status"] = "need-reread"
            continue
        v = p2_view(ctx, s)
        if v is None:
            r["status"] = "need-p2b"
            continue
        if v["pending"]:
            r["status"] = "need-reread"
            continue
        refs, opened = [], []
        consumed = set()
        for iid, f in v["flags"].items():
            z = p3.get(("box", s, iid))
            if z is not None and z[1].get("pick") == "B":
                consumed.update(f["members"])
        for q in insts:
            if q["iid"] in consumed:
                continue
            if q.get("fixed"):
                refs.append(_ref(q, None))
                continue
            f = v["flags"].get(q["iid"])
            versions, z = [q], None
            if f:
                r["stats"]["box_flagged"] += 1
                z = p3.get(("box", s, q["iid"]))
                if z is None:
                    opened.append(("box", s, q["iid"]))
                    continue
                pick = z[1].get("pick")
                r["stats"][f"box_p3_{pick}"] += 1
                if pick == "B":
                    versions = f["B"]
                elif pick != "A":
                    refs.append(_ref(q, "undecided"))
                    continue
            else:
                r["stats"]["box_approved"] += 1
            k_text = 0
            for vq in versions:
                if "ignore" in vq:
                    refs.append(_ref(vq, None))
                elif vq.get("src") == "eval":
                    refs.append(_ref(vq, None, text=vq["eval_text"]))
                elif vq.get("text2") is not None:
                    texts = (z[1].get("texts") if z else None) or []
                    t3 = canon(texts[k_text]) if k_text < len(texts) else None
                    k_text += 1
                    if t3 and norm(t3) == norm(vq["text2"]):
                        refs.append(_ref(vq, None, text=canon(vq["text2"])))
                        r["stats"]["text_new_agree"] += 1
                    else:
                        refs.append(_ref(vq, "undecided"))
                        r["stats"]["text_new_undecided"] += 1
                else:
                    key = f"{s}|{q['iid']}"
                    a = t1s.get(key)
                    b = t2s.get(key)
                    c = t3s.get(key)
                    txt, how = _agree(text_of(a[1], a[2]) if a else None, text_of(b[1], b[2]) if b else None,
                                      text_of(c[1], c[2]) if c else None)
                    r["stats"][f"text_{how}"] += 1
                    if how in ("open",):
                        opened.append(("text-wait", s, q["iid"]))
                    elif how == "need-p3":
                        opened.append(("text", s, q["iid"]))
                    elif txt:
                        refs.append(_ref(vq, None, text=txt))
                    else:
                        refs.append(_ref(vq, "undecided"))
        for m in v["missed"]:
            if len(m["B"]) == 1 and "ignore" in m["B"][0]:
                # an ignore area pass 2 adds is taken as it is: a doubtful area left out of the scoring is the
                # protocol's own outcome for text the looks do not agree on
                refs.append(_ref(m["B"][0], None))
                r["stats"]["missed2_ignore"] += 1
                continue
            r["stats"]["missed2"] += 1
            z = p3.get(("missed", s, m["iid"]))
            if z is None:
                opened.append(("missed", s, m["iid"]))
                continue
            pick = z[1].get("pick")
            r["stats"][f"missed2_p3_{pick}"] += 1
            if pick == "A":
                continue
            if pick != "B":
                refs.append(_ref(m["B"][0], "undecided"))
                continue
            for vq in m["B"]:
                if "ignore" in vq:
                    refs.append(_ref(vq, None))
                else:
                    texts = z[1].get("texts") or []
                    t3 = canon(texts[0]) if texts else None
                    if t3 and norm(t3) == norm(vq["text2"]):
                        refs.append(_ref(vq, None, text=canon(vq["text2"])))
                    else:
                        refs.append(_ref(vq, "undecided"))
        r["refs"] = sorted(refs, key=lambda x: (x["ink"][1], x["ink"][0]))
        r["open"] = opened
        r["status"] = "open" if opened else "final"
    return out


def _ref(q, why, text=None):
    """A reference from an instance: why None keeps it as it is (its own ignore why, or a labelled instance)."""
    ign = why is not None or "ignore" in q
    return {"ink": [int(v) for v in q["ink"]], "cls": q.get("cls"), "vertical": bool(q.get("vertical")),
            "text": None if ign else text, "ignore": ign, "why": why or q.get("ignore") if ign else None,
            "iid": q.get("iid")}


# ----------------------------------------------------------------------------- p3 builder

def make_p3(ctx, dec, size=24):
    """A third-look task for the open disputes of shots whose pass-1 and pass-2 tasks are all done."""
    split = ctx["split"]
    have = set()
    for t in tasks(split, "p3"):
        meta = task_of(t)
        have |= {(it["kind"], it["shot"], it.get("ref")) for it in meta.get("items", [])}
        have |= {("text", *k.split("|")) for k in meta.get("keys", [])}
    items, texts = [], []
    for s, r in dec.items():
        for o in r["open"]:
            kind = o[0]
            if kind == "tf" and ("tf", s, None) not in have:
                items.append({"kind": "tf", "shot": s, "ref": None})
            elif kind in ("box", "missed") and (kind, s, o[2]) not in have:
                items.append({"kind": kind, "shot": s, "ref": o[2]})
            elif kind == "text" and ("text", s, o[2]) not in have:
                texts.append((s, o[2]))
    if not items and not texts:
        return None
    items = items[:size]
    batch = next_batch_name(split, "p3")
    d = review_dir(split, "p3", batch)
    d.mkdir(parents=True, exist_ok=True)
    metas = []
    for i, it in enumerate(items, 1):
        X = f"X{i:03d}"
        s = it["shot"]
        rgb = load_label_frame(ctx["shots"][s])
        dims = ctx["props"][s]["dims"]
        js = {"item": X, "kind": it["kind"], "shot": s, "platform": platform_label(dims, ctx["part_of"].get(s, "")),
              "dims": dims, "zoom": zoom_of(dims)}
        marks = []
        if it["kind"] == "tf":
            js["question"] = "Does the frame hold any text at all (any script, any size)? text_free true = none."
            need = {}
        else:
            v = p2_view(ctx, s)
            insts = {q["iid"]: q for q in ctx["inst1"][s]}
            if it["kind"] == "box":
                A = [insts[it["ref"]]] + [insts[m] for m in v["flags"][it["ref"]]["members"]]
                B = v["flags"][it["ref"]]["B"]
            else:
                A = []
                B = next(m["B"] for m in v["missed"] if m["iid"] == it["ref"])

            def desc(q):
                e = {"box": xywh(q["ink"]), "vertical": bool(q.get("vertical"))}
                if "ignore" in q:
                    e["ignore"] = q["ignore"]
                else:
                    e["cls"] = q.get("cls")
                if q.get("src") == "eval":
                    e["eval_line"] = True
                return e
            js["A"] = [desc(q) for q in A] or "nothing: no instance and no ignore area here"
            js["B"] = [desc(q) for q in B] or "nothing: not text, no instance here"
            need = {"A": 0, "B": sum(1 for q in B if q.get("text2") is not None)}
            if need["B"]:
                js["needs_text"] = {"B": need["B"]}
            for q in A:
                marks.append({"box": q["ink"], "tag": "A", "colour": (0, 220, 220), "vertical": q.get("vertical")})
            for q in B:
                marks.append({"box": q["ink"], "tag": "B", "colour": (255, 230, 0), "vertical": q.get("vertical"),
                              "style": "dashed"})
            if not B:
                for q in A:
                    marks.append({"box": [q["ink"][0] - 3, q["ink"][1] - 3, q["ink"][2] + 3, q["ink"][3] + 3],
                                  "colour": (255, 230, 0), "style": "dashed", "tag": "B: none"})
        img = frame_canvas(rgb, marks, title=f"{X}  third look ({it['kind']})")
        img.save(d / f"{X}.png")
        write_review_json(d / f"{X}.json", js)
        metas.append(dict(it, X=X, needs_text=need))
    entries, extra = [], {}
    t1s, t2s = text_decisions(split, "p1t"), text_decisions(split, "p2t")
    for s, iid in texts[:SHEET_MAX * 2]:
        q = next(x for x in ctx["inst1"][s] if x["iid"] == iid)
        key = f"{s}|{iid}"
        ex = []
        for src in (t1s.get(key), t2s.get(key)):
            if src:
                t, stt = text_of(src[1], src[2])
                if t:
                    ex.append(t)
        extra[key] = ex
        entries.append({"key": key, "shot": s, "iid": iid, "ink": q["ink"], "vertical": q["vertical"],
                        "reads": q.get("reads") or {}, "src": q["src"]})
    # the text sheets and the README go into the same task dir
    frames_rgb, sheets = {}, []
    for vert in (False, True):
        todo = [e for e in entries if bool(e["vertical"]) == vert]
        while todo:
            name = f"S{len(sheets) + 1:03d}"
            be = []
            for j, e in enumerate(todo[:SHEET_MAX]):
                if e["shot"] not in frames_rgb:
                    frames_rgb[e["shot"]] = load_label_frame(ctx["shots"][e["shot"]])
                be.append(dict(e, n=j + 1, rgb=frames_rgb[e["shot"]]))
            img, placed = text_sheet(be, f"{name}  third look")
            img.save(d / f"{name}.png")
            write_review_json(d / f"{name}.json", {"sheet": name, "entries": [
                {"n": e["n"], "key": e["key"], "candidates": candidates_of(e, "p3", extra.get(e["key"], ()))}
                for e in placed]})
            sheets.append(name)
            todo = todo[len(placed):]
    readme = README_P3.format(split=split, batch=batch, nx=len(metas), ns=len(sheets), common=COMMON,
                              check=check_cmd(split, "p3", batch))
    shots = sorted({it["shot"] for it in metas} | {e["shot"] for e in entries})
    write_task(split, "p3", batch, {"items": metas, "sheets": sheets, "keys": [e["key"] for e in entries],
                                    "shots": shots, "images": len(metas) + len(sheets)}, readme)
    return batch


# ----------------------------------------------------------------------------- feed (keeps the review going)

THRESH = {"p1t": 60, "p2t": 60, "p2b": 15, "p3": 10}
REREAD_CMD = ["timeout", "1800", "unshare", "-rn", f"{ROOT}/teacher/.venv/bin/python", f"{ROOT}/eval/textspot_sets.py",
              "reread", "--split"]


def unfinished(split, stage):
    cl = claims()
    return [t for t in tasks(split, stage) if not (cl.get(task_key(split, stage, t.name)) or {}).get("done")]


def log(msg):
    line = f"{utc()} {msg}"
    print(line, flush=True)
    with open(SETS / "feed.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def keys_in(split, stage):
    out = set()
    for t in tasks(split, stage):
        out |= set(task_of(t).get("keys", []))
    return out


def snapshot(ctx):
    split = ctx["split"]
    write_jsonl(SETS / split / "frames.jsonl", ctx["frames"])
    write_jsonl(SETS / split / "tf_candidates.jsonl", [{"shot": c["shot"], "rank": c["rank"], "kind": c["kind"],
                                                        "dets": ctx["st"]["dets"].get(c["shot"]),
                                                        "look": ctx["st"]["tf"].get(c["shot"])} for c in ctx["cands"]])


def feed_split(split, reread=True):
    ctx = build_ctx(split)
    snapshot(ctx)
    st = ctx["st"]
    order = [f["shot"] for f in ctx["frames"] if f["part"] != "text-free" and f["shot"] in ctx["props"]]
    order = list(dict.fromkeys(order))
    tf_look = [s for s in tf_candidates_to_look(ctx["cands"], st["tf"]) if s in ctx["props"]]
    made = make_p1b(split, order + [s for s in tf_look if s not in order], ctx)
    if made:
        log(f"{split}: p1b tasks {made}")
    if reread and pending_rereads(ctx):
        n = len(pending_rereads(ctx))
        log(f"{split}: rereading {n} boxes")
        import subprocess
        rc = subprocess.run(["env", "HEAVY_SLOTS=12", str(HERD / "heavy.sh")] + REREAD_CMD + [split],
                            stdin=subprocess.DEVNULL, capture_output=True, text=True)
        log(f"{split}: reread rc={rc.returncode} {rc.stdout.strip().splitlines()[-1:] if rc.stdout else ''}"
            f" {rc.stderr.strip()[-300:] if rc.returncode else ''}")
        ctx = build_ctx(split)
    p1_open = bool(unfinished(split, "p1b"))
    ready = [s for s in ctx["part_of_main"] if s in ctx["inst1"]
             and not any(q.get("pending") for q in ctx["inst1"][s])]
    ready.sort(key=lambda s: order.index(s) if s in order else 10**6)
    for stage in ("p1t", "p2t"):
        have = keys_in(split, stage)
        ents = [e for s in ready for e in text_entries(split, s, ctx["props"][s], ctx["inst1"][s])
                if e["key"] not in have]
        while ents and (len(ents) >= THRESH[stage] or not p1_open):
            chunk = ents[:SHEET_MAX * 4]
            if stage == "p1t":
                b = write_text_task(split, "p1t", chunk, ctx, "pass 1 text", "pass 1, TEXT",
                                    " with the engines that read them (MeikiOCR page read `meiki`, MeikiOCR on the "
                                    "crop `meikic`, PP-OCRv6 small `v6s`)")
            else:
                b = write_text_task(split, "p2t", chunk, ctx, "pass 2 text (blind)", "pass 2, TEXT (blind)",
                                    " (shuffled and unattributed; an instance that no engine found has none)")
            log(f"{split}: {stage} task {b} ({len(chunk)} entries)")
            ents = ents[len(chunk):]
            if len(ents) < THRESH[stage] and p1_open:
                break
    have = shots_in_tasks(split, "p2b")
    p2b_todo = [s for s in ready if s not in have]
    for s in tf_look + [c["shot"] for c in ctx["cands"]]:
        p1 = st["p1"].get(s)
        if p1 and p1[1].get("text_free") is True and p1[1].get("screen") is not False and s not in have \
                and s not in p2b_todo and s not in ctx["part_of_main"]:
            p2b_todo.append(s)
    if p2b_todo and (len(p2b_todo) >= THRESH["p2b"] or not p1_open):
        made = make_p2b(split, p2b_todo, ctx)
        log(f"{split}: p2b tasks {made}")
    ctx = build_ctx(split)
    dec = decide_split(ctx)
    upstream_open = any(unfinished(split, stg) for stg in ("p1b", "p1t", "p2b", "p2t"))
    n_open = sum(1 for r in dec.values() for o in r["open"] if o[0] in ("box", "missed", "text", "tf"))
    if n_open and (n_open >= THRESH["p3"] or not upstream_open):
        b = make_p3(ctx, dec)
        if b:
            log(f"{split}: p3 task {b}")
            dec = decide_split(build_ctx(split))
    summary = collections.Counter(r["status"] for r in dec.values())
    sel = {f["shot"] for f in ctx["frames"]}
    final_sel = sum(1 for s in sel if dec.get(s, {}).get("status") == "final")
    write_json(SETS / split / "decide_summary.json", {"utc": utc(), "status": dict(summary), "selected": len(sel),
                                                      "selected_final": final_sel,
                                                      "parts": dict(collections.Counter(
                                                          f["part"] for f in ctx["frames"]))})
    return {"selected": len(sel), "final": final_sel, "status": dict(summary),
            "complete": final_sel == len(sel) and not tf_candidates_to_look(ctx["cands"], st["tf"], spare=0)
            and not any(unfinished(split, stg) for stg in STAGES)}


def cmd_feed(args):
    splits = args.splits.split(",")
    while True:
        res = {}
        for sp in splits:
            try:
                res[sp] = feed_split(sp)
            except Exception as e:                       # noqa: BLE001 (log, keep the loop alive)
                import traceback
                log(f"{sp}: feed error {e!r}\n{traceback.format_exc()[-2000:]}")
        write_json(SETS / "feed_state.json", {"utc": utc(), "splits": res,
                                              "open": not all(r.get("complete") for r in res.values()) if res else
                                              True})
        print(json.dumps(res), flush=True)
        if not args.loop:
            break
        time.sleep(args.every)


# ----------------------------------------------------------------------------- assemble and pin

def looks_of(split, shot):
    cl = claims()
    out = []
    for stage in STAGES:
        for t in tasks(split, stage):
            c = cl.get(task_key(split, stage, t.name))
            if c and c["done"] and shot in task_of(t).get("shots", []):
                e = {"pass": stage, "by": c["label"]}
                if e not in out:
                    out.append(e)
    return out


def manifest_rows(ctx, dec, partial=False):
    rows, problems = [], []
    for f in sorted(ctx["frames"], key=lambda f: (PARTS.index(f["part"]), f["rank"], f["id"])):
        r = dec.get(f["shot"])
        if not r or r["status"] != "final":
            problems.append((f["id"], (r or {}).get("status")))
            if not partial:
                continue
        k = 2 if f["part"] == "adrenaline" else 1
        refs = []
        for x in (r or {}).get("refs", []):
            b = xywh(x["ink"])
            ref = {"box": [v * k for v in b], "cls": x["cls"], "vertical": x["vertical"], "text": x["text"],
                   "ignore": x["ignore"]}
            if x["ignore"]:
                ref["why"] = x["why"]
            refs.append(ref)
        rows.append({"id": f["id"], "vn": f["vn"], "part": f["part"], "session": f["session"], "url": f["url"],
                     "sha256": f["sha256"], "src_dims": f["src_dims"], "scale": f["scale"], "dims": f["dims"],
                     "text_free": bool(f["part"] == "text-free"), "refs": refs,
                     "looks": looks_of(ctx["split"], f["shot"]), "reviewer_kind": "assistant"}
                    | ({"report_only": True} if f.get("report_only") else {}))
    return rows, problems


def frame_pixels(row):
    """The frame of a manifest row rebuilt from its source file (path via the local inventories)."""
    return build_frame(read_rgb(row["_path"]), row["dims"], row["scale"])


def pin_stats(split, rows, dec, ctx, manifest_path):
    sha = sha256_file(manifest_path)
    refs = [(r, x) for r in rows for x in r["refs"]]
    lab = [(r, x) for r, x in refs if not x["ignore"]]
    ign = [(r, x) for r, x in refs if x["ignore"]]
    st = collections.Counter()
    for f in ctx["frames"]:
        st.update((dec.get(f["shot"]) or {}).get("stats") or {})
    tf = ctx["st"]["tf"]
    approved, flagged = st["box_approved"], st["box_flagged"]
    agree = st["text_agree"]
    disagree = st["text_p3-first"] + st["text_p3-second"] + st["text_p3-none"]
    cl = claims()
    by_stage = collections.Counter()
    labels = collections.defaultdict(set)
    for k, c in cl.items():
        sp, stg, _ = k.split("/")
        if sp == split and c["done"]:
            by_stage[stg] += 1
            labels[stg].add(c["label"])
    out = {"kind": "textspot-sets-pin", "split": split, "utc": utc(), "manifest": os.path.relpath(manifest_path, ROOT),
           "sha256": sha, "frames": len(rows),
           "frames_by_part": dict(collections.Counter(r["part"] for r in rows)),
           "vns": len({r["vn"] for r in rows}), "vns_by_part": {p: len({r["vn"] for r in rows if r["part"] == p})
                                                                for p in PARTS if any(r["part"] == p for r in rows)},
           "sessions": len({r["session"] for r in rows}),
           "text_free_frames": sum(1 for r in rows if r["text_free"]),
           "refs": {"labelled": len(lab), "by_class": dict(collections.Counter(x["cls"] for _, x in lab)),
                    "by_part": dict(collections.Counter(r["part"] for r, _ in lab)),
                    "by_orientation": dict(collections.Counter("vertical" if x["vertical"] else "horizontal"
                                                               for _, x in lab))},
           "ignore_areas": {"total": len(ign), "by_why": dict(collections.Counter(x["why"] for _, x in ign))},
           "characters": {"total": sum(len(norm(x["text"])) for _, x in lab),
                          "by_class": dict(collections.Counter({c: sum(len(norm(x["text"])) for _, x in lab
                                                                       if x["cls"] == c) for c in CLASSES}))},
           "review": {"reviewer_kind": "assistant", "tasks_done": dict(by_stage),
                      "sessions": {k: sorted(v) for k, v in labels.items()},
                      "boxes": {"pass1_instances_checked": approved + flagged, "pass2_approved": approved,
                                "pass2_flagged": flagged,
                                "pass2_approval_rate": round(approved / (approved + flagged), 4)
                                if approved + flagged else None,
                                "third_look": {k[7:]: v for k, v in st.items() if k.startswith("box_p3_")},
                                "pass2_missed": st["missed2"], "pass2_missed_ignore_areas": st["missed2_ignore"],
                                "pass2_missed_third_look": {k[11:]: v for k, v in st.items()
                                                            if k.startswith("missed2_p3_")}},
                      "text": {"two_looks_agree": agree, "disagree_to_third_look": disagree,
                               "agreement_rate": round(agree / (agree + disagree), 4) if agree + disagree else None,
                               "third_look": {"first": st["text_p3-first"], "second": st["text_p3-second"],
                                              "neither": st["text_p3-none"]},
                               "both_unsure": st["text_both-unsure"],
                               "new_extent_agree": st["text_new_agree"],
                               "new_extent_undecided": st["text_new_undecided"]},
                      "text_free": {"candidates": len(ctx["cands"]),
                                    "looked": sum(1 for c in ctx["cands"] if c["shot"] in ctx["st"]["p1"]),
                                    "confirmed": sum(1 for c in ctx["cands"] if tf.get(c["shot"]) == "yes"),
                                    "rejected": sum(1 for c in ctx["cands"] if tf.get(c["shot"]) == "no")},
                      "dropped_not_a_game_screen": sorted(ctx["st"]["drops"])}}
    return out


def cmd_assemble(args):
    split = args.split
    ctx = build_ctx(split)
    dec = decide_split(ctx)
    rows, problems = manifest_rows(ctx, dec, partial=args.partial)
    if problems and not args.partial:
        print(f"{len(problems)} frames not final: {problems[:10]}")
        sys.exit(1)
    path = MANIFESTS / f"textspot-{split}.jsonl" if not args.out else Path(args.out)
    for r in rows:                                      # sanity: every ref inside its frame
        W, H = r["dims"]
        for x in r["refs"]:
            b = x["box"]
            if b[0] < 0 or b[1] < 0 or b[0] + b[2] > W or b[1] + b[3] > H or b[2] <= 0 or b[3] <= 0:
                sys.exit(f"{r['id']}: ref {b} outside the frame {W}x{H}")
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
    pin = pin_stats(split, rows, dec, ctx, path)
    if not args.partial:
        write_json(SETS / split / "PIN.json", pin)
    print(json.dumps({k: pin[k] for k in ("sha256", "frames", "frames_by_part", "vns", "text_free_frames", "refs",
                                          "ignore_areas", "characters")}, ensure_ascii=False))
    print("review", json.dumps(pin["review"], ensure_ascii=False))
    if problems:
        print(f"PARTIAL: {len(problems)} frames not final")


# ----------------------------------------------------------------------------- spot check

def _hatch(dr, b, colour, step=6):
    x0, y0, x1, y1 = b
    for k in range(int(x0 - (y1 - y0)), int(x1), step):
        a = (max(x0, k), y0 + max(0, x0 - k))
        e = (min(x1, k + (y1 - y0)), y0 + min(y1 - y0, x1 - k))
        if a[0] < e[0]:
            dr.line([a, e], fill=colour)


def spot_canvas(rgb, refs, title):
    """A frame with its references drawn (class colours; ignore areas grey and hatched; numbered), above the clean
    frame, with rulers (frame pixels; a 480x272 frame at 2x)."""
    from PIL import ImageDraw
    canvas = frame_canvas(rgb, [], title=title)
    dr = ImageDraw.Draw(canvas)
    H, W = rgb.shape[:2]
    z = zoom_of([W, H])
    oy = 14 + RULER_T
    for n, x in enumerate(refs, 1):
        bx, by, bw, bh = x["box"]
        b = [RULER_L + bx * z - 1, oy + by * z - 1, RULER_L + (bx + bw) * z, oy + (by + bh) * z]
        if x["ignore"]:
            _hatch(dr, b, IGNORE_COLOUR)
            dr.rectangle(b, outline=IGNORE_COLOUR)
            _tag(dr, b, f"i{n}", IGNORE_COLOUR, x["vertical"], W * z, RULER_L, oy)
        else:
            col = CLS_COLOUR[x["cls"]]
            dr.rectangle([b[0] - 1, b[1] - 1, b[2] + 1, b[3] + 1], outline=(0, 0, 0))
            dr.rectangle(b, outline=col)
            _tag(dr, b, f"{n}v" if x["vertical"] else str(n), col, x["vertical"], W * z, RULER_L, oy)
    return canvas


def source_paths(split):
    """{url: (local path, sha256)} of every source screenshot the split's shots.jsonl knows."""
    return {r["url"]: (r["path"], r["sha256"]) for r in load_shots(split).values()}


def cmd_spot(args):
    path = Path(args.manifest) if args.manifest else MANIFESTS / f"textspot-{args.split}.jsonl"
    rows = jsonl(path)
    if args.parts:
        rows = [r for r in rows if r["part"] in args.parts.split(",")]
    pick = random.Random(args.seed).sample(rows, min(args.n, len(rows)))
    src = source_paths(args.split)
    out = SETS / args.split / f"spot_{args.tag}"
    out.mkdir(parents=True, exist_ok=True)
    listing = []
    for i, r in enumerate(pick, 1):
        p, sha = src[r["url"]]
        if sha256_file(p) != r["sha256"]:
            sys.exit(f"{p}: sha256 differs from the manifest")
        rgb = build_frame(read_rgb(p), r["dims"], r["scale"])
        name = f"K{i:03d}"
        spot_canvas(rgb, r["refs"], f"{name}  {r['id']}  {r['part']}  {r['dims'][0]}x{r['dims'][1]}").save(
            out / f"{name}.png")
        refs = [{"n": n, "box": x["box"], "cls": x["cls"], "vertical": x["vertical"],
                 **({"ignore": x["why"]} if x["ignore"] else {"text": x["text"]})} for n, x in enumerate(r["refs"], 1)]
        write_review_json(out / f"{name}.json", {"spot": name, "id": r["id"], "part": r["part"], "dims": r["dims"],
                                                 "text_free": r["text_free"], "refs": refs})
        listing.append({"spot": name, "id": r["id"], "part": r["part"], "refs": len(r["refs"])})
    write_json(out / "index.json", {"manifest": str(path), "sha256": sha256_file(path), "seed": args.seed,
                                    "n": len(pick), "frames": listing})
    print(f"{len(pick)} frames -> {out}")


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("dev-frames")
    p = sub.add_parser("test-frames")
    p.add_argument("--fetch", action="store_true")
    for name in ("propose", "reread"):
        p = sub.add_parser(name)
        p.add_argument("--split", required=True, choices=("dev", "test"))
        p.add_argument("--procs", type=int, default=5)
        p.add_argument("--limit", type=int, default=0)
    p = sub.add_parser("feed")
    p.add_argument("--splits", default="dev")
    p.add_argument("--loop", action="store_true")
    p.add_argument("--every", type=int, default=60)
    for name in ("claim", "done", "check", "release"):
        p = sub.add_parser(name)
        p.add_argument("--label", default="")
        if name != "claim":
            p.add_argument("--task", required=True)
    p = sub.add_parser("zoom")
    p.add_argument("--task", required=True)
    p.add_argument("--frame", required=True, help="Fxxx or Xxxx of the task")
    p.add_argument("--box", type=int, nargs=4, required=True, metavar=("X", "Y", "W", "H"))
    p.add_argument("--label", default="reviewer")
    sub.add_parser("status")
    p = sub.add_parser("spot")
    p.add_argument("--split", required=True, choices=("dev", "test"))
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="check")
    p.add_argument("--parts", default="")
    p.add_argument("--manifest", default="")
    p = sub.add_parser("assemble")
    p.add_argument("--split", required=True, choices=("dev", "test"))
    p.add_argument("--partial", action="store_true", help="write what is final so far (a preview; no PIN)")
    p.add_argument("--out", default="")
    args = ap.parse_args()
    {"dev-frames": cmd_dev_frames, "test-frames": cmd_test_frames, "propose": cmd_propose, "reread": cmd_reread,
     "feed": cmd_feed, "claim": cmd_claim, "done": cmd_done, "check": cmd_check, "release": cmd_release,
     "zoom": cmd_zoom, "status": cmd_status, "assemble": cmd_assemble, "spot": cmd_spot}[args.cmd](args)


if __name__ == "__main__":
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = "1"
    main()
