#!/usr/bin/env python3
"""Turn the teacher's readings of the training screenshots into labelled crops.

  build.py decide [--rule teacher/rule.json] [--reads DIR]   reads -> teacher/manifests/*, then render
  build.py render                                            manifests + screenshots -> npz (no OCR)

decide applies vote.accept() with the rule chosen on eval's hand-checked sets
(teacher/rule.json, written from measure.py's report) to every line label.py read, and keeps
a label only if every character is in data/vocab.txt. Test games are dropped again here, in
case eval's list changed after the screenshots were selected: every VN with
synth/common.is_test_vn() true, every id in eval's test_games.json ("all"), and every
screenshot (URL or sha256) of eval's manifests. A label is kept at most twice per VN and 8
times in all (UI strings repeat on every screenshot of a game).

Split by game, as realdata: a VN is validation when 15 <= md5(vn) % 100 < 25, otherwise
training (< 15 is the test hash). A capture session (realdata's <vn>/<release>/<n>) belongs
to one VN, so no session straddles the split.

render is deterministic: native Vita and PSP screenshots are cropped as they are; PC shots
are scaled to fit each target screen it is legible on (cv2.INTER_AREA). The crop is the
line's ink box, with furigana and widened to the labelled characters' boxes, plus 4 px (Vita)
or 2 px (PSP), as eval crops its test lines. Grey crops (integer luma) go into render.py's
shard format, so train/train.py --real loads them as they are; per-crop meta holds the
character boxes (relative to the crop) and confidences.
"""
import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import geometry as G  # noqa: E402
import vote  # noqa: E402
from label import OUT, REALDATA, load_shots  # noqa: E402

MAN = Path(os.environ.get("TEACHER_MANIFESTS") or HERE / "manifests")
RULE = HERE / "rule.json"
PER_VN_CAP, GLOBAL_CAP = 2, 8
LINE_FIELDS = ("id", "crop", "label", "how", "targets", "split")
DISTILL_FIELDS = ("id", "crop", "targets", "split", "why")
EVAL_GAMES = [ROOT / "eval" / "manifests" / "test_games.json", OUT / "inputs" / "eval" / "test_games.json",
              OUT / "inputs" / "realdata_eval_exclude.json"]


def is_test_vn(vn):
    return int(hashlib.md5(vn.encode()).hexdigest(), 16) % 100 < 15


def split_of(vn):
    return "val" if 15 <= int(hashlib.md5(vn.encode()).hexdigest(), 16) % 100 < 25 else "train"


def eval_exclusions():
    """(VN ids, release ids, screenshot URLs, sha256s) of eval's test sets: its reserved games
    and sessions plus every screenshot in its manifests (the merged copy in this checkout,
    else the copies in $TEACHER_OUT/inputs)."""
    vns, rels, urls, shas = set(), set(), set(), set()
    for p in EVAL_GAMES:
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            vns |= set(d.get("all", []))
            rels |= set(d.get("sessions", []))
    for d in (ROOT / "eval" / "manifests", OUT / "inputs" / "eval"):
        for p in d.glob("*-test.jsonl"):
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    vns.add(r["vn"])
                    urls.add(r["url"])
                    shas.add(r["sha256"])
                    if r.get("session"):
                        rels.add(r["session"])
    return vns, rels, urls, shas


def excluded(r, ex):
    vns, rels, urls, shas = ex
    return (is_test_vn(r["vn"]) or r["vn"] in vns or r.get("release") in rels or r["url"] in urls
            or r.get("sha256") in shas)


def load_reads(reads_dir):
    out = {}
    for p in sorted(Path(reads_dir).glob("train.*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:            # a torn last line from a killed run
                continue
            out[r["shot"]] = r
    return out


def load_vocab():
    return {c for c in (ROOT / "data" / "vocab.txt").read_text(encoding="utf-8").split("\n") if c}


# Rejected lines that are still real text lines of the right size: kept, unlabelled, for logit
# (soft) distillation. Lines rejected for geometry or for not being Japanese text are not.
DISTILL_WHY = {"disagree", "low-conf", "low-det", "confusable-dissent", "short"}


def readings(ln):
    """{engine: text} of every reading of a line."""
    out = {}
    for e in vote.VOTERS:
        t = vote.reading(ln, e)
        if t is not None:
            out[e] = t[0]
    return out


def crop_of(ln, rule):
    """Source-pixel box of the training crop: ink (+ furigana), widened to the labelled
    characters' boxes (the primary reading's), so every labelled glyph is inside, but never
    beyond the view the voters read."""
    c = list(ln.get("crop") or ln["ink"])
    chars = vote.reading(ln, rule.get("primary", vote.DEFAULT["primary"]))[1]
    if chars:
        v = ln.get("view") or [c[0], c[1], c[2], c[3]]
        c[0] = max(min(v[0], c[0]), min(c[0], min(ch[1] for ch in chars)))
        c[2] = min(max(v[2], c[2]), max(c[2], max(ch[3] for ch in chars)))
    return c


def decide(args):
    rule = json.loads(Path(args.rule).read_text(encoding="utf-8"))["rule"]
    vocab = load_vocab()
    ex = eval_exclusions()
    shots = load_reads(args.reads)
    collection = {r["url"]: r for r in load_shots("train")}
    why, kept, shot_rows, pool = Counter(), [], {}, []
    for sk, r in sorted(shots.items()):
        if excluded({**collection.get(r["url"], {}), **r}, ex):
            why["test-game"] += len(r["lines"])
            continue
        shot_rows[sk] = r
        for ln in G.refurigana(r["lines"]):
            lab, how = vote.accept(ln, rule)
            if lab is None:
                why[how] += 1
                if how in DISTILL_WHY and any(vote.japanese(t) for t in readings(ln).values()):
                    pool.append({"id": f"{sk}_{ln['k']}", "shot": sk, "url": r["url"], "vn": r["vn"],
                                 "kind": r["kind"], "session": r.get("session"), "crop": crop_of(ln, rule),
                                 "targets": ln["targets"], "split": split_of(r["vn"]), "why": how,
                                 "reads": readings(ln), "det_conf": ln.get("det_conf")})
                continue
            if any(c not in vocab for c in lab):
                why["oov"] += 1
                continue
            m = vote.reading(ln, rule.get("primary", vote.DEFAULT["primary"]))
            kept.append({"id": f"{sk}_{ln['k']}", "shot": sk, "vn": r["vn"], "kind": r["kind"],
                         "session": r.get("session"), "crop": crop_of(ln, rule), "label": lab, "how": how,
                         "targets": ln["targets"], "split": split_of(r["vn"]), "pol": G.polarity(ln.get("hint", {})),
                         "chars": m[1], "det_conf": ln.get("det_conf")})
    rng = random.Random(20261009)
    rng.shuffle(kept)
    per_vn, glob, out = Counter(), Counter(), []
    for k in kept:
        if per_vn[(k["vn"], k["label"])] >= PER_VN_CAP or glob[k["label"]] >= GLOBAL_CAP:
            why["repeat-cap"] += 1
            continue
        per_vn[(k["vn"], k["label"])] += 1
        glob[k["label"]] += 1
        out.append(k)
    out.sort(key=lambda k: (int(k["vn"][1:]), k["id"]))
    pool = [p for p in pool if p["targets"]]
    pool.sort(key=lambda k: (int(k["vn"][1:]), k["id"]))
    used = {k["shot"] for k in out} | {p["shot"] for p in pool}
    MAN.mkdir(parents=True, exist_ok=True)
    for kind in ("vita", "psp", "hires"):
        rows = [k for k in out if k["kind"] == kind]
        (MAN / f"lines_{kind}.jsonl").write_text(
            "".join(json.dumps({f: k[f] for f in LINE_FIELDS}, ensure_ascii=False) + "\n" for k in rows),
            encoding="utf-8")
        rows = [p for p in pool if p["kind"] == kind]       # geometry only: no text in git
        (MAN / f"distill_{kind}.jsonl").write_text(
            "".join(json.dumps({f: p[f] for f in DISTILL_FIELDS}, ensure_ascii=False) + "\n" for p in rows),
            encoding="utf-8")
    (OUT / "distill_full.jsonl").write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pool),
                                            encoding="utf-8")
    shots_out = []
    for sk in sorted(used, key=lambda s: (int(s.split("_")[0][1:]), s)):
        r = shot_rows[sk]
        c = collection.get(r["url"], {})
        shots_out.append({"shot": sk, "vn": r["vn"], "kind": r["kind"], "url": r["url"], "sha256": r.get("sha256"),
                          "dims": r["dims"], "release": r.get("release") or c.get("release"),
                          "session": r.get("session") or c.get("session"), "split": split_of(r["vn"])})
    (MAN / "shots.jsonl").write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in shots_out),
                                     encoding="utf-8")
    games = defaultdict(lambda: {"lines": 0, "shots": set(), "sessions": set()})
    for k in out:
        g = games[k["vn"]]
        g["lines"] += 1
        g["shots"].add(k["shot"])
        g["sessions"].add(k["session"])
        g["split"] = k["split"]
    (MAN / "games.json").write_text(json.dumps(
        {"note": "VNDB VNs with teacher-labelled lines (train or val): eval must never test on these.",
         "games": {v: {"split": g["split"], "lines": g["lines"], "shots": len(g["shots"]),
                       "sessions": sorted(s for s in g["sessions"] if s)}
                   for v, g in sorted(games.items(), key=lambda kv: int(kv[0][1:]))}},
        ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    n_lines = sum(len(r["lines"]) for r in shots.values())
    stats = {"rule": rule, "screenshots_read": len(shots), "lines_read": n_lines, "accepted": len(out),
             "rejected": dict(why.most_common()), "by_kind": dict(Counter(k["kind"] for k in out)),
             "by_split": dict(Counter(k["split"] for k in out)),
             "crops_by_split_target": dict(Counter(f"{k['split']}/{t}" for k in out for t in k["targets"])),
             "by_how": dict(Counter(k["how"] for k in out).most_common()),
             "by_polarity": dict(Counter(k["pol"] for k in out)),
             "vns": dict(Counter(g["split"] for g in games.values())),
             "sessions": len({k["session"] for k in out}), "screenshots": len({k["shot"] for k in out}),
             "unique_labels": len({k["label"] for k in out}), "chars": sum(len(k["label"]) for k in out),
             "distill_pool": dict(Counter(f"{p['split']}/{t}" for p in pool for t in p["targets"])),
             "distill_why": dict(Counter(p["why"] for p in pool))}
    (MAN / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    full = OUT / "lines_full.jsonl"            # with character boxes: artifacts only (too big to commit)
    full.write_text("".join(json.dumps(k, ensure_ascii=False) + "\n" for k in out), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=1), flush=True)
    render(args)


def git_sha():
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                             check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
        return sha + (" (with uncommitted changes)" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def shot_path(s):
    for base in (REALDATA / "shots" / s["kind"],):
        p = base / f"{s['vn']}_{s['url'].rsplit('/', 1)[-1]}"
        if p.exists():
            return p
    return None


def _load(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()] \
        if path.exists() else []


def _save(path, crops, labels, metas):
    shapes = np.array([c.shape for c in crops], np.int32)
    offsets = np.concatenate([[0], np.cumsum(shapes[:, 0] * shapes[:, 1])]).astype(np.int64)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, data=np.concatenate([c.ravel() for c in crops]), offsets=offsets, shapes=shapes,
                        labels=np.array(json.dumps(labels, ensure_ascii=False)),
                        meta=np.array(json.dumps(metas, ensure_ascii=False)))
    tmp.replace(path)


def render(args):
    """teacher_{train,val}.npz from manifests/lines_<kind>.jsonl, and teacher_distill_{train,val}.npz
    (empty labels: for logit distillation) from manifests/distill_<kind>.jsonl."""
    import cv2
    shots = {s["shot"]: s for s in _load(MAN / "shots.jsonl")}
    extra = {k["id"]: k for k in _load(OUT / "lines_full.jsonl") + _load(OUT / "distill_full.jsonl")}
    pools = {"teacher": "lines", "teacher_distill": "distill"}
    rows = {name: defaultdict(list) for name in pools}
    for name, prefix in pools.items():
        for kind in ("vita", "psp", "hires"):
            for r in _load(MAN / f"{prefix}_{kind}.jsonl"):
                rows[name][r["id"].rsplit("_", 1)[0]].append(r)
    out = {(name, split): ([], [], []) for name in pools for split in ("train", "val")}
    missing = 0
    for sk in sorted(set(rows["teacher"]) | set(rows["teacher_distill"])):
        s = shots[sk]
        path = shot_path(s)
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR) if path else None
        if bgr is None:
            missing += 1
            continue
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        for t in ("vita", "psp"):
            todo = [(name, r) for name in pools for r in rows[name][sk] if t in r["targets"]]
            if not todo:
                continue
            sc = G.target_scale(rgb.shape[1], rgb.shape[0], t) if s["kind"] == "hires" else 1.0
            img = G.scale_image(rgb, sc)
            for name, r in todo:
                x0, y0, x1, y1 = G.crop_box(r["crop"], sc, G.PAD[t], img.shape[1], img.shape[0])
                crops, labels, metas = out[(name, r["split"])]
                crops.append(G.to_gray(img[y0:y1, x0:x1]))
                f = extra.get(r["id"], {})
                meta = {"real": True, "teacher": True, "id": r["id"], "vn": s["vn"], "session": s["session"],
                        "kind": s["kind"], "target": t, "box": [x0, y0, x1, y1]}
                if name == "teacher":
                    labels.append(r["label"])
                    meta.update(how=r["how"], pol=f.get("pol"), chars=[
                        [c[0], round(c[1] * sc - x0, 1), round(c[2] * sc - y0, 1), round(c[3] * sc - x0, 1),
                         round(c[4] * sc - y0, 1), c[5]] for c in f.get("chars", [])])
                else:
                    labels.append("")
                    meta.update(why=r["why"], reads=f.get("reads", {}))
                metas.append(meta)
    counts = {}
    for (name, split), (crops, labels, metas) in out.items():
        counts[f"{name}_{split}"] = dict(Counter(m["target"] for m in metas))
        if crops:
            _save(OUT / f"{name}_{split}.npz", crops, labels, metas)
            print(f"{OUT / f'{name}_{split}.npz'}: {len(crops)} crops {counts[f'{name}_{split}']}", flush=True)
    if missing:
        print(f"WARNING: {missing} screenshots missing (run realdata/vndb.py fetch)", flush=True)
    write_readme(counts)
    print("BUILD_DONE", flush=True)


def write_readme(counts):
    stats = json.loads((MAN / "stats.json").read_text(encoding="utf-8"))
    spot = MAN / "spot.jsonl"
    spot_txt = "not run yet"
    if spot.exists():
        rows = [json.loads(x) for x in spot.read_text(encoding="utf-8").splitlines() if x.strip()]
        bad = [r for r in rows if not r["ok"]]
        spot_txt = (f"{len(bad)} wrong of {len(rows)} random accepted lines checked by eye "
                    f"({100 * len(bad) / len(rows):.1f}%)")
    text = f"""teacher: real VN lines relabelled by the MeikiOCR-first, multi-engine teacher
(owner: teacher agent; code: teacher/ in bee-san/vita-vn-ocr, git {git_sha()})

teacher_train.npz, teacher_val.npz
    Grey line crops (integer luma, synth/common.to_gray) in render.py's shard format (data, offsets,
    shapes, labels, meta), so train/train.py --real loads them as they are. Crops by split and target:
    {json.dumps(counts)}. meta per crop: id, vn, session, kind (vita | psp | hires), target (vita | psp),
    how (the engines that agreed), pol, chars ([char, x0, y0, x1, y1, conf] relative to the crop:
    MeikiOCR's character boxes and confidences).
    Split by VNDB VN (val = 15 <= md5(vn) % 100 < 25), so no game or capture session is in both.
    Use teacher_val for model selection only; eval's vita-test / psp-test games are never here.

teacher_distill_train.npz, teacher_distill_val.npz
    The real text lines the rule did NOT accept (engines disagree, low confidence, ...), same
    format and crops but with empty labels: unlabelled real crops for logit (soft CTC)
    distillation from a strong local teacher (PP-OCRv6 small measured best, see rule.json).
    meta: why (the rejection reason) and reads (every engine's raw reading), plus box (crop box
    in the target-resolution screenshot) so RGB crops can be re-cut from the screenshots.
    train/train.py drops lines with empty labels, so this file is never mixed into supervised
    training by accident.

Source: realdata's screenshot collection (native Vita 960x544, native PSP 480x272, PC/console
shots scaled to both), read at native resolution: MeikiOCR finds and reads every horizontal text
line (dialogue, names, menus); PP-OCRv6 small, manga-ocr and Chrome Screen AI (local, run with no
network) vote. Rule (chosen on eval's hand-checked sets, teacher/measure.py):
    {json.dumps(stats['rule'])}
Spot check: {spot_txt}.

lines_full.jsonl  every accepted line with its character boxes (source pixels) and the vote
reads/            label.py's raw output: every line, every engine's reading and box
models/           the pinned engine files (label.py fetch)
inputs/           realdata's shots manifests and eval's manifests as copied (see SOURCE files)

stats: {json.dumps(stats, ensure_ascii=False)}
"""
    (OUT / "README").write_text(text, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("decide")
    p.add_argument("--rule", default=str(RULE))
    p.add_argument("--reads", default=str(OUT / "reads"))
    sub.add_parser("render")
    args = ap.parse_args()
    {"decide": decide, "render": render}[args.cmd](args)


if __name__ == "__main__":
    main()
