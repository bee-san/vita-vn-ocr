#!/usr/bin/env python3
"""Turn voter readings into labelled real training lines, and render the npz shards.

  build.py decide [--rule NAME]   labels/*.jsonl + hand checks -> realdata/manifests/lines_<kind>.jsonl,
                                  unlabelled_<kind>.jsonl, stats.json, games.json; then renders (below)
  build.py render                 realdata/manifests/{lines,unlabelled}_<kind>.jsonl + screenshots ->
                                  $REALDATA/real_train.npz, real_val.npz, real_unlabelled.npz

`decide` applies the voting rule (vote.RULES) to every line read by label.py, lets a
hand-checked truth (manifests/handcheck.jsonl) override the vote, and keeps a label only
if it is Japanese and inside data/vocab.txt. A label is kept at most twice per VN and
eight times overall (UI strings like セーブ repeat on every screenshot). Test games
(synth/common.is_test_vn, eval's reserved list) and eval's test capture sessions (VNDB
release ids) are dropped again here in case the lists changed after selection, as are
lines of screenshots that `vndb.py prune` removed. Split by VN: 15 <= md5(vn) % 100 < 25
is validation (about 10% of VNs; < 15 is the test hash), everything else train.

`decide` also lists every other line with Japanese text from a training-split game (the
readers disagreed, so it has no label) in manifests/unlabelled_<kind>.jsonl, with the four
readings, and `render` cuts those into real_unlabelled.npz: crops for soft / logit
distillation from a teacher, with empty labels (train.py --real drops them, so they can't
be trained on by mistake) and the four readings in meta as hints.

`render` is deterministic and needs no OCR: it rescales each screenshot to the line's
target (vita 960x544 / psp 480x272; native Vita and PSP shots stay as they are, PC shots
give one crop per target where the glyphs are legible), crops the padded line box, and
stores integer-luma grey crops in render.py's shard format, with per-line meta. It keeps
the console's polarity contract (docs/linefind.md): a crop whose text common.line_dark()
calls dark is inverted (255 - g), so every crop shows light text, as synth/render.py's
shards do; --keep-polarity writes the crops as they are.
"""
import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import lines as L  # noqa: E402
import vote  # noqa: E402
from vndb import KINDS, excluded_sessions, excluded_vns, load_shots, realdata_dir, shot_key, shot_path  # noqa: E402

sys.path.insert(0, str(ROOT / "synth"))
from common import is_test_vn, line_dark, load_vocab, to_gray  # noqa: E402

MAN = HERE / "manifests"
LINES = {k: MAN / f"lines_{k}.jsonl" for k in KINDS}
UNLABELLED = {k: MAN / f"unlabelled_{k}.jsonl" for k in KINDS}
HAND = MAN / "handcheck.jsonl"
FIELDS = ("id", "ink", "targets", "label", "how", "pol", "split")
POOL_FIELDS = ("id", "ink", "targets", "pol", "reads")
PER_VN_CAP, GLOBAL_CAP = 2, 8


def split_of(vn):
    return "val" if 15 <= int(hashlib.md5(vn.encode()).hexdigest(), 16) % 100 < 25 else "train"


def targets_of(kind, dims, ink):
    if kind in ("vita", "psp"):
        return [kind]
    ih = ink[3] - ink[1]
    return [t for t in ("vita", "psp")
            if L.MIN_INK[t] <= ih * L.target_scale(dims[0], dims[1], t) <= L.MAX_INK]


def load_reads():
    """Every line read by label.py's full run (labels/lines.*.jsonl; the dev sample is
    only for evaluate_rules.py), one record per line id. A line read twice keeps the
    record with more readings; a torn last line (a killed run) is skipped."""
    recs, torn = {}, 0
    for path in sorted((realdata_dir() / "labels").glob("lines.*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                torn += 1
                continue
            if r["id"] not in recs or len(r["r"]) > len(recs[r["id"]]["r"]):
                recs[r["id"]] = r
    if torn:
        print(f"skipped {torn} torn lines", flush=True)
    return recs


def load_lines():
    rows = []
    for k in KINDS:
        if LINES[k].exists():
            rows += [json.loads(x) for x in LINES[k].read_text(encoding="utf-8").splitlines() if x.strip()]
    return rows


def load_unlabelled():
    """The unlabelled pool (manifests/unlabelled_<kind>.jsonl): Japanese lines of training-split
    games that the vote left without a label, with the four readings."""
    rows = []
    for k in KINDS:
        if UNLABELLED[k].exists():
            rows += [json.loads(x) for x in UNLABELLED[k].read_text(encoding="utf-8").splitlines() if x.strip()]
    return rows


def titles():
    out = {}
    for path in sorted((realdata_dir() / "vndb").glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            out[r["vn"]] = r["title"]
    return out


def load_hand():
    """{"<set>:<line id>": hand row}. Line ids are only unique within one labelling run,
    so a dev check and a spot check can share an id while naming different lines."""
    if not HAND.exists():
        return {}
    return {f"{h['set']}:{h['id']}": h for h in map(json.loads, HAND.read_text(encoding="utf-8").splitlines()) if h}


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / max(union, 1)


def match_hand(hand, recs, min_iou=0.5):
    """{record id: hand row}. A hand check names the line it was made on (id) and that
    line's ink box; it applies to the record of the same screenshot whose ink box overlaps
    it most (IoU >= min_iou), so a change in line geometry or numbering can't move a hand
    label onto a different line."""
    by_shot = defaultdict(list)
    for r in recs.values():
        by_shot[r["id"].rsplit("_", 1)[0]].append(r)
    out, score = {}, {}
    for h in hand.values():
        cands = by_shot.get(h["id"].rsplit("_", 1)[0], [])
        best = max(cands, key=lambda r: iou(r["ink"], h["ink"]), default=None)
        if best is None:
            continue
        o = iou(best["ink"], h["ink"])
        if o >= min_iou and o > score.get(best["id"], 0.0):    # two checks of one line: the closer one
            out[best["id"]], score[best["id"]] = h, o
    return out


def decide(args):
    rule = vote.RULES[args.rule]
    vocab = set(load_vocab())
    extra, sessions = excluded_vns(), excluded_sessions()
    shots = {r["url"]: r for r in load_shots()}
    recs = load_reads()
    hand = match_hand(load_hand(), recs)
    why, kept, pool = Counter(), [], []
    for r in recs.values():
        s = shots.get(r["url"])
        # test games and test capture sessions again (eval's lists grow); a screenshot that
        # `vndb.py prune` removed from the shots manifests is gone for good
        reason = ("test-game" if is_test_vn(r["vn"]) or r["vn"] in extra else
                  "unselected" if s is None else
                  "test-session" if s["release"] in sessions else None)
        if reason:
            why[reason] += 1
            continue
        h = hand.get(r["id"])
        if h is not None:
            lab, how = (vote.canon(h["truth"]), "hand") if h["truth"] else (None, "hand-uncertain")
        else:
            lab, how = rule({k: v[0] for k, v in r["r"].items()})
        if lab is None:
            why[how] += 1
            if how in ("disagree", "confusable-dissent") and split_of(r["vn"]) == "train":
                tg = targets_of(r["kind"], r["dims"], r["ink"])
                if tg:
                    pool.append({"id": r["id"], "kind": r["kind"], "ink": r["ink"], "targets": tg, "pol": r["pol"],
                                 "reads": {k: v[0] for k, v in r["r"].items()}})
            continue
        if not vote.japanese(lab):
            why["not-japanese"] += 1
            continue
        oov = [c for c in lab if c not in vocab]
        if oov:
            why["oov"] += 1
            continue
        tg = targets_of(r["kind"], r["dims"], r["ink"])
        if not tg:
            why["too-small"] += 1
            continue
        kept.append({"id": r["id"], "vn": r["vn"], "url": r["url"], "kind": r["kind"], "ink": r["ink"],
                     "targets": tg, "label": lab, "how": how, "pol": r["pol"],
                     "split": split_of(r["vn"])})
    # caps: hand-checked lines first, then a seeded shuffle
    rng = random.Random(20261009)
    rng.shuffle(kept)
    kept.sort(key=lambda k: k["how"] != "hand")
    per_vn, glob, out = Counter(), Counter(), []
    for k in kept:
        if per_vn[(k["vn"], k["label"])] >= PER_VN_CAP or glob[k["label"]] >= GLOBAL_CAP:
            why["repeat-cap"] += 1
            continue
        per_vn[(k["vn"], k["label"])] += 1
        glob[k["label"]] += 1
        out.append(k)
    out.sort(key=lambda k: (int(k["vn"][1:]), k["id"]))
    MAN.mkdir(parents=True, exist_ok=True)
    for kind in KINDS:
        LINES[kind].write_text("".join(json.dumps({x: k[x] for x in FIELDS}, ensure_ascii=False) + "\n"
                                       for k in out if k["kind"] == kind), encoding="utf-8")
    write_games(out, shots)
    pool.sort(key=lambda u: u["id"])
    for kind in KINDS:
        UNLABELLED[kind].write_text("".join(json.dumps({x: u[x] for x in POOL_FIELDS}, ensure_ascii=False) + "\n"
                                            for u in pool if u["kind"] == kind), encoding="utf-8")
    stats = {"rule": args.rule, "lines_read": len(recs), "accepted": len(out), "rejected": dict(why),
             "sessions": {sp: len({shots[k["url"]]["session"] for k in out if k["split"] == sp})
                          for sp in ("train", "val")},
             "by_split_kind": Counter(f"{k['split']}/{k['kind']}" for k in out),
             "crops_by_split_target": Counter(f"{k['split']}/{t}" for k in out for t in k["targets"]),
             "by_how": Counter(k["how"] for k in out), "by_polarity": Counter(k["pol"] for k in out),
             "vns": {sp: len({k["vn"] for k in out if k["split"] == sp}) for sp in ("train", "val")},
             "unique_labels": len({k["label"] for k in out}),
             "chars": sum(len(k["label"]) for k in out),
             "unlabelled_train_lines": len(pool)}
    (MAN / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=1), flush=True)
    render(args)


def git_sha():
    import subprocess
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                             check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
        return sha + (" (with uncommitted changes)" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def spot_result():
    hand = [h for h in load_hand().values() if h.get("set") == "spot"]
    if not hand:
        return "not run yet"
    bad = [h for h in hand if h.get("pseudo") is not None and vote.key(h["pseudo"]) != vote.key(h["truth"] or "")]
    return f"{len(bad)} wrong of {len(hand)} random accepted lines checked by eye ({len(bad) / len(hand):.1%})"


def write_readme(stats, counts):
    text = f"""realdata: real VN dialogue lines for training the tiny Vita/PSP recognizer
(owner: realdata agent; code: realdata/ in bee-san/vita-vn-ocr, git {git_sha()})

real_train.npz, real_val.npz
    Grey line crops (integer luma, synth/common.to_gray) at native size, in render.py's shard
    format (data, offsets, shapes, labels, meta) so train/train.py --real loads them as they
    are. Dark-text crops (common.line_dark, the console's rule) are inverted to light text,
    as the console and synth/render.py do. meta per crop: id, vn, session, kind (vita | psp |
    hires), target (vita | psp), pol (light-on-dark | dark-on-light, the style before
    inversion), dark and inverted (the console's polarity flag), how (the vote, or "hand").
    {counts}
    Split by VNDB VN: val = 15 <= md5(vn) % 100 < 25. Use real_val for model selection only.
    Train with `train.py --real <this dir>/real_train.npz`, naming that file: a glob over this
    directory (--real '<dir>/*.npz') would also train on real_val.npz.

Sources (VNDB kana API, screenshots of Japanese-only releases rated sexual < 0.3 and
violence < 0.5): native Vita 960x544 (kind vita), native PSP 480x272 (kind psp), and PC /
home-console screenshots (kind hires) scaled to fit 960x544 and 480x272 (cv2.INTER_AREA),
one crop per target where the ink is 14-72 px (Vita) or 10-72 px (PSP) tall.
Test games excluded: synth/common.is_test_vn() and every id in eval's test_games.json ("all"),
and every screenshot of eval's test capture sessions (its "sessions", VNDB release ids).

Labels: PP-OCRv5 mobile det finds the text rows; each row's ink box is refined by gradient
energy (either polarity); crops are padded so the console's grad finder gives the same
boxes (median edge difference 0 px on every side, but only 42-70% of edges within 1 px and
p10-p90 -24..+17 px at x0, -17..+2 px at x1; realdata/check_finder.py on 400 lines).
Four readers vote: PP-OCRv5 server, PP-OCRv6 small, manga-ocr, meiki. Rule
{stats['rule']}: at least three readers support one PP-OCR reading (manga-ocr/meiki may
differ by invented edge brackets, and from v6s by ellipsis cell counts); a fourth reader
that differs only in small kana or ー/一 vetoes. Lines checked by hand use the hand label.
Label form: vote.canon (corpus conventions: full-width ！？～（）, one … per cell, ― for
dashes, no spaces, no page-cursor icons). Spot check: {spot_result()}.

Other files: manifests/ (a copy of realdata/manifests at the git SHA above: shots_<kind>.jsonl
with url, sha256, vn, release and capture session per screenshot, lines_<kind>.jsonl,
unlabelled_<kind>.jsonl (the unlabelled pool, with the four readings), games.json with every
VN's split and sessions, handcheck.jsonl, stats.json), labels/*.jsonl (every line read, with
all readings: label.py output), shots/<kind>/ (the screenshots; copyrighted, never commit or
redistribute), vndb/ (raw VNDB listings).

Rebuild from the repo alone:
    python realdata/vndb.py fetch          # screenshots from VNDB, sha256-checked
    python realdata/build.py render        # all three npz from realdata/manifests/{{lines,unlabelled}}_*.jsonl
Re-label from scratch: vndb.py list/select/fetch, label.py (via heavy.sh), build.py decide.

stats: {json.dumps(stats, ensure_ascii=False)}
"""
    (realdata_dir() / "README").write_text(text, encoding="utf-8")


def write_games(out, shots):
    """manifests/games.json: every VN (game) in the screenshot collection with its title,
    split, kinds, capture sessions, screenshot count and accepted lines. eval must not
    test on any of them; a teacher relabelling the collection can split on the same ids."""
    names, lines = titles(), Counter(k["vn"] for k in out)
    games = {}
    for s in shots.values():
        g = games.setdefault(s["vn"], {"title": names.get(s["vn"]), "split": split_of(s["vn"]), "kinds": set(),
                                       "sessions": set(), "shots": 0})
        g["kinds"].add(s["kind"])
        g["sessions"].add(s["session"])
        g["shots"] += 1
    for vn, g in games.items():
        g.update(kinds=sorted(g["kinds"]), sessions=sorted(g["sessions"]), lines=lines.get(vn, 0))
    note = ("Every VNDB VN (game) in realdata's screenshot collection (manifests/shots_*.jsonl): eval must "
            "not test on these. split: val if 15 <= md5(vn) % 100 < 25, else train (< 15 is the test hash, "
            "excluded); a capture session belongs to one VN, so it never straddles a split.")
    body = ",\n".join(f"  {json.dumps(vn)}: {json.dumps(games[vn], ensure_ascii=False)}"
                       for vn in sorted(games, key=lambda v: int(v[1:])))
    (MAN / "games.json").write_text(f'{{"note": {json.dumps(note)},\n "games": {{\n{body}\n}}}}\n',
                                    encoding="utf-8")


def render(args):
    shots = {shot_key(r): r for r in load_shots()}
    by_shot = defaultdict(list)
    for r in load_lines():
        by_shot[r["id"].rsplit("_", 1)[0]].append(r)
    for u in load_unlabelled():
        by_shot[u["id"].rsplit("_", 1)[0]].append({**u, "label": "", "split": "unlabelled", "how": "none"})
    out = {"train": ([], [], []), "val": ([], [], []), "unlabelled": ([], [], [])}
    for sk in sorted(by_shot):
        s = shots[sk]
        bgr = cv2.imread(str(shot_path(s)), cv2.IMREAD_COLOR)
        if bgr is None:
            raise SystemExit(f"missing screenshot {shot_path(s)}: run vndb.py fetch")
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        for t in ("vita", "psp"):
            todo = [r for r in by_shot[sk] if t in r["targets"]]
            if not todo:
                continue
            sc = L.target_scale(rgb.shape[1], rgb.shape[0], t) if s["kind"] == "hires" else 1.0
            grey = to_gray(L.scale_image(rgb, sc))
            for r in todo:
                x0, y0, x1, y1 = L.train_box(r["ink"], sc, grey.shape[1], grey.shape[0])
                dark = line_dark(grey, (x0, y0, x1 - x0, y1 - y0))
                crop = grey[y0:y1, x0:x1]
                crops, labels, metas = out[r["split"]]
                crops.append(255 - crop if dark and not args.keep_polarity else crop.copy())
                labels.append(r["label"])
                metas.append({"real": True, "id": r["id"], "vn": s["vn"], "session": s["session"],
                              "kind": s["kind"], "target": t, "pol": r["pol"], "dark": dark,
                              "inverted": bool(dark and not args.keep_polarity), "how": r["how"]})
                if "reads" in r:
                    metas[-1]["reads"] = r["reads"]
    dest = realdata_dir()
    for split, (crops, labels, metas) in out.items():
        if not crops:
            print(f"{split}: no lines", flush=True)
            continue
        shapes = np.array([c.shape for c in crops], np.int32)
        offsets = np.concatenate([[0], np.cumsum(shapes[:, 0] * shapes[:, 1])]).astype(np.int64)
        path = dest / f"real_{split}.npz"
        tmp = path.with_name(path.name + ".tmp.npz")
        np.savez_compressed(tmp, data=np.concatenate([c.ravel() for c in crops]), offsets=offsets,
                            shapes=shapes, labels=np.array(json.dumps(labels, ensure_ascii=False)),
                            meta=np.array(json.dumps(metas, ensure_ascii=False)))
        tmp.replace(path)
        print(f"{path}: {len(crops)} crops ({Counter(m['target'] for m in metas)})", flush=True)
    counts = "; ".join(f"{split}: {len(out[split][0])} crops "
                       f"({dict(Counter(m['target'] for m in out[split][2]))})" for split in ("train", "val"))
    counts += (f"\n    real_unlabelled.npz: {len(out['unlabelled'][0])} crops of lines from training-split games "
               "that the readers disagreed on;\n    labels are empty, the four readings are in meta (for logit "
               "distillation from a teacher)")
    stats_path = MAN / "stats.json"
    write_readme(json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {"rule": "?"},
                 counts)
    import shutil
    (dest / "manifests").mkdir(exist_ok=True)
    for f in sorted(MAN.glob("*.json*")):
        shutil.copyfile(f, dest / "manifests" / f.name)
    print("BUILD_DONE", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("decide")
    p.add_argument("--rule", default=vote.DEFAULT_RULE)
    p.add_argument("--keep-polarity", action="store_true", help="don't invert dark-text crops")
    p = sub.add_parser("render")
    p.add_argument("--keep-polarity", action="store_true", help="don't invert dark-text crops")
    args = ap.parse_args()
    {"decide": decide, "render": render}[args.cmd](args)


if __name__ == "__main__":
    main()
