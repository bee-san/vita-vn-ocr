#!/usr/bin/env python3
"""PROVISIONAL real-set evaluation for the synth2 A/B (until eval/'s benchmark merges).

Sets (all local, never committed):

  hand         The research's hand-transcribed real lines (docs/research-findings.md, Q5),
               on the research's own line crops: gt_real.json (32 lines) and
               q5_run_a/gt.json (37 lines). The two sets share 7 screenshots, so lines are
               deduplicated by (screenshot sha256, normalised text): 58 lines, 28 Vita
               screenshots. Tagged light-on-dark / dark-on-light / light-on-light (outlined
               text or light boxes) from the transcribers' notes. Most games are training-
               side: the old shards could use these screenshots as backgrounds; synth2
               excludes them (data/manifests/synth2_bg_exclude.txt).
  hand-tight   The same crops trimmed horizontally to their edge-energy columns plus 4 px
               (the research crops span the whole region width; the overlay's finder hands
               the recognizer tight boxes). Groups [reserved-test-games] and
               [training-side-games] split both by eval's reserved list and is_test_vn.
  pseudo-test  data/real lines where PP-OCRv5 and manga-ocr agree, test-split games
               (is_test_vn): bright-finder crops of bright text, labels not hand-checked.
  pseudo-train The same for training-side games.
  synth-val:<name>  the first --synth-n lines of a validation shard (--synth-val name=path).

CER is the edit distance over the reference length, summed over lines, after NFKC and
whitespace removal (the research convention). Exact = whole line right.

  ab_eval.py --ckpt v1=PATH --ckpt A=PATH ... [--synth-val old=PATH] [--pairs A:B] [--json out.json]
"""
import argparse
import hashlib
import json
import os
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "train"))
from common import DATA, is_test_vn, to_gray  # noqa: E402
from model import TinyVNOCR  # noqa: E402
from train import edit_distance, greedy, make_batch, resize32  # noqa: E402

HERD = Path(os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr"))
RESEARCH = HERD / "shared" / "research" / "scratch" / "real"
EVAL_GAMES = HERD / "shared" / "artifacts" / "eval" / "test_games.json"


def norm(s):
    s = unicodedata.normalize("NFKC", s)
    return "".join(ch for ch in s if not ch.isspace())


def reserved():
    games = set()
    if EVAL_GAMES.exists():
        games = set(json.loads(EVAL_GAMES.read_text())["all"])
    return lambda vn: is_test_vn(vn) or vn in games


def kind_from_notes(notes):
    n = notes.lower()
    if "black on" in n or "dark on" in n:
        return "dark-on-light"
    if "outline" in n or "on light" in n:
        return "light-on-light"
    return "light-on-dark"


def grey(path):
    return to_gray(np.asarray(Image.open(path).convert("RGB")))


def tighten(g, frac=0.12, pad=4):
    """Trim a line crop horizontally to the columns with edge energy above frac x the
    maximum (the column rule of finders.find_lines_grad), keeping pad px each side."""
    f = g.astype(np.float32)
    e = np.zeros_like(f)
    e[:-1, :-1] = np.abs(np.diff(f, axis=1))[:-1, :] + np.abs(np.diff(f, axis=0))[:, :-1]
    col = np.convolve(e.sum(axis=0), np.ones(5) / 5, mode="same")
    cols = np.flatnonzero(col > frac * col.max()) if col.max() > 0 else []
    if not len(cols):
        return g
    x0, x1 = max(0, int(cols[0]) - pad), min(g.shape[1], int(cols[-1]) + 1 + pad)
    return g[:, x0:x1] if x1 - x0 >= 8 else g


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hand():
    """[(crop, text, vn, kind, source)] from the research's two hand-transcribed sets,
    deduplicated by (screenshot sha256, normalised text); the first transcription wins."""
    out, seen = [], set()

    def add(shot_path, crop_path, text, vn, kind, src):
        key = (file_sha(shot_path), norm(text))
        if key not in seen:
            seen.add(key)
            out.append((grey(crop_path), text, vn, kind, src))
    for it in json.loads((RESEARCH / "gt_real.json").read_text(encoding="utf-8")):
        shot = next((RESEARCH / "shots").glob(f"{it['shot']}.*"))
        for j, text in enumerate(it["lines"]):
            add(shot, RESEARCH / "lines" / f"{it['shot']}_{j}.png", text, it["shot"].split("_")[0],
                kind_from_notes(it["notes"]), "gt_real")
    for it in json.loads((RESEARCH / "q5_run_a" / "gt.json").read_text(encoding="utf-8")):
        shot = RESEARCH / "q5_run_a" / "raw" / Path(it["file"]).name
        for j, ln in enumerate(it["lines"]):
            if ln.get("text"):
                add(shot, RESEARCH / "lines" / f"{it['tag']}_{j}.png", ln["text"], it["vn"], it["kind"], "q5")
    return out


def pseudo(split):
    out = []
    for s in (DATA / "real" / "lines.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(s)
        if r["split"] == split and r["agree"] and r["label"]:
            out.append((grey(DATA / "real" / "crops" / f"{r['id']:05d}.png"), r["label"], r["vn"],
                        "bright", "data/real"))
    return out


def synth_val(path, n):
    z = np.load(path)
    data, offs, shapes = z["data"], z["offsets"], z["shapes"]
    labels = json.loads(str(z["labels"]))
    metas = json.loads(str(z["meta"])) if "meta" in z.files else [{}] * len(labels)
    out = []
    for i in range(min(n, len(labels))):
        h, w = (int(v) for v in shapes[i])
        out.append((data[offs[i]:offs[i + 1]].reshape(h, w), labels[i], "", metas[i].get("scheme", "bright"),
                    metas[i].get("platform", "vita")))
    return out


def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = TinyVNOCR(len(ck["vocab"]) + 1, width=ck.get("width", 1.0))
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck["vocab"], ck.get("step")


@torch.no_grad()
def predict(model, vocab, crops, batch=64):
    imgs = [resize32(g) for g in crops]
    order = np.argsort([g.shape[1] for g in imgs])
    dummy = [np.zeros(1, np.int64)] * len(imgs)
    preds = [None] * len(imgs)
    for k in range(0, len(order), batch):
        idx = order[k:k + batch]
        x, in_len, _, _ = make_batch(imgs, dummy, idx)
        for i, seq in zip(idx, greedy(model(x), in_len)):
            preds[i] = "".join(vocab[t - 1] for t in seq)
    return preds


def score(preds, items):
    rows = []
    for p, it in zip(preds, items):
        ref, hyp = norm(it[1]), norm(p)
        rows.append((edit_distance(list(hyp), list(ref)), len(ref)))
    return rows


def summary(rows, idx=None):
    idx = range(len(rows)) if idx is None else idx
    e = sum(rows[i][0] for i in idx)
    n = sum(rows[i][1] for i in idx)
    k = len(list(idx))
    return {"lines": k, "chars": n, "cer": round(e / max(n, 1), 4),
            "exact": round(sum(rows[i][0] == 0 for i in idx) / max(k, 1), 4)}


def bootstrap_diff(ra, rb, idx, reps=4000, seed=0):
    """Paired bootstrap over lines of CER(b) - CER(a): [2.5%, 97.5%] percentiles."""
    rng = np.random.default_rng(seed)
    idx = np.asarray(list(idx))
    ea, eb = np.array([ra[i][0] for i in idx]), np.array([rb[i][0] for i in idx])
    n = np.array([ra[i][1] for i in idx])
    s = rng.integers(0, len(idx), (reps, len(idx)))
    d = eb[s].sum(1) / n[s].sum(1) - ea[s].sum(1) / n[s].sum(1)
    return [round(float(x), 4) for x in np.percentile(d, [2.5, 97.5])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True, help="name=path of a train.py checkpoint")
    ap.add_argument("--synth-val", action="append", default=[], help="name=path of a validation shard")
    ap.add_argument("--synth-n", type=int, default=3000)
    ap.add_argument("--pairs", action="append", default=[], help="A:B, report CER(B)-CER(A) with a 95%% CI")
    ap.add_argument("--json", default="")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("TORCH_NUM_THREADS", 4)))
    ap.add_argument("--dump", default="", help="write per-line predictions (JSONL) for these sets")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    is_res = reserved()
    hand_set = hand()
    tight = [(tighten(it[0]),) + tuple(it[1:]) for it in hand_set]
    sets = {"hand": hand_set, "hand-tight": tight, "pseudo-test": pseudo("test"), "pseudo-train": pseudo("train")}
    for spec in args.synth_val:
        name, path = spec.split("=", 1)
        sets[f"synth-val:{name}"] = synth_val(path, args.synth_n)
    groups = {}          # (set, group) -> line indices
    for name, items in sets.items():
        groups[(name, "all")] = list(range(len(items)))
        by = defaultdict(list)
        for i, it in enumerate(items):
            by[it[3]].append(i)
            if name.startswith("synth-val"):
                by[f"platform={it[4]}"].append(i)
        for k, v in sorted(by.items()):
            if len(by) > 1:
                groups[(name, k)] = v
    res_idx = [i for i, it in enumerate(hand_set) if is_res(it[2])]
    for name in ("hand", "hand-tight"):
        groups[(name, "reserved-test-games")] = res_idx
        groups[(name, "training-side-games")] = [i for i in range(len(hand_set)) if i not in set(res_idx)]
    crops_sha = hashlib.sha256(b"".join(it[0].tobytes() for it in hand_set)).hexdigest()[:16]
    out = {"sets": {k: len(v) for k, v in sets.items()}, "hand_crops_sha256_16": crops_sha,
           "models": {}, "results": {}, "pairs": {}}
    rows = {}
    dump = []
    for spec in args.ckpt:
        name, path = spec.split("=", 1)
        model, vocab, step = load_model(path)
        out["models"][name] = {"path": path, "step": step}
        rows[name] = {}
        for sname, items in sets.items():
            preds = predict(model, vocab, [it[0] for it in items])
            rows[name][sname] = score(preds, items)
            if args.dump and not sname.startswith("synth-val"):
                dump += [{"model": name, "set": sname, "vn": it[2], "kind": it[3], "ref": it[1], "hyp": p}
                         for p, it in zip(preds, items)]
        for (sname, g), idx in groups.items():
            out["results"].setdefault(f"{sname} [{g}]", {})[name] = summary(rows[name][sname], idx)
        print(f"{name}: " + ", ".join(f"{s} CER {summary(rows[name][s])['cer']:.2%}" for s in sets), flush=True)
    for pair in args.pairs:
        a, b = pair.split(":")
        for (sname, g), idx in groups.items():
            if idx:
                out["pairs"].setdefault(f"{b} - {a}", {})[f"{sname} [{g}]"] = {
                    "cer_diff": round(summary(rows[b][sname], idx)["cer"] - summary(rows[a][sname], idx)["cer"], 4),
                    "ci95": bootstrap_diff(rows[a][sname], rows[b][sname], idx)}
    width = max(len(k) for k in out["results"])
    names = list(out["models"])
    print(f"{'set [group]':<{width}}  " + "  ".join(f"{n:>16}" for n in names))
    for key, per in out["results"].items():
        cells = [f"{per[n]['cer']:7.2%} {per[n]['exact']:6.1%}" if n in per else "" for n in names]
        print(f"{key:<{width}}  " + "  ".join(f"{c:>16}" for c in cells) + f"   n={per[names[0]]['lines']}")
    for pair, per in out["pairs"].items():
        print(pair)
        for key, v in per.items():
            print(f"  {key:<{width}}  {v['cer_diff']:+.2%}  CI95 [{v['ci95'][0]:+.2%}, {v['ci95'][1]:+.2%}]")
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    if args.dump:
        Path(args.dump).write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in dump))


if __name__ == "__main__":
    main()
