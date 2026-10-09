#!/usr/bin/env python3
"""Score sweep checkpoints: fp32 and int8 weights, on synthetic and real VN lines.

  sweep_eval.py --run runs/NAME [--ckpt ckpt_last.pt] [--sets synth-val,real-hand,real-pseudo]

Variants: fp32, and int8 = the int8 export's weights (runtime/'s export.py --int8:
symmetric per-output-channel int8 for the pointwise, row-collapse, bottleneck and
head convs, BN folded; stem, depthwise and biases float; see model.int8_conv) with
float activations, i.e. the runtime's W8F mode. For a QAT run the int8 row is the
QAT result; for every other run it is post-training quantisation (PTQ). w8i adds
the runtime's W8I activation rounding as described in runtime/vocr.h (inputs of the
int8 convs quantised per position to int16, symmetric max-abs over the channels),
simulated in float.

Sets from eval's shared benchmark (needs eval/ in this checkout, or $VOCR_EVAL_CODE
pointing at an eval checkout, and $VOCR_EVAL_DATA at its screenshots):
  vita-test, psp-test   eval's hand-checked lines, cropped and scored exactly as
               eval/run.py does (eval/sets.py + eval/metrics.py), crops fed as they
               are; one forward pass per padded width, so the text is identical to
               run.py's one-line-at-a-time inference
  vita-test-inv, psp-test-inv   the same lines with the dark-on-light ones inverted by
               their style tag (an oracle): the v1 recipe renders only light text, so
               this separates recognizer capacity from that data gap
  real-hand-x  real-hand (below) minus the lines from vita-test's games: 17 more games,
               a check that is independent of vita-test
Provisional sets (from before eval/ existed; real-pseudo is all from vita-test games):
  synth-val    the held-out synthetic shard (data/synth, last shard), all lines
  real-hand    hand-transcribed real Vita lines from the research campaign
               (scratch/real/gt.json confident lines, plus gt_real.json lines whose
               text is not already in gt.json), crops as the research cut them
  real-pseudo  data/real lines of held-out test VNs (common.is_test_vn) on which
               PP-OCRv5 server and manga-ocr agree exactly (pseudo-labels, the
               overlay's bright-pixel finder crops)
Dark-text crops are inverted before recognition, since the v1 recipe only renders
light text: --polarity tag (default) inverts the lines the research tagged dark text
on a light background (an oracle; a deployed pipeline needs the line finder to
report polarity), auto guesses from the crop, none feeds every crop as it is.
Scoring: CER = Levenshtein / GT length after label_real.canon, full-width ASCII
folded to ASCII, dash-like glyphs folded to ―, whitespace removed. A reduced
vocab's <unk> (〓) is always an error. Writes <run>/eval_<ckpt stem>.json. --save-int8-ckpt also writes
<run>/<ckpt stem>_int8w.pt, the same checkpoint with every int8 conv's weights
replaced by their quantise-dequantise values: evaluated in fp32 by any tool (e.g.
eval/run.py) it gives exactly the runtime's W8F output.
"""
import argparse
import glob
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "synth"))
sys.path.insert(0, str(ROOT / "data"))
from common import DATA, is_test_vn, to_gray  # noqa: E402
from model import fake_quant_weight, from_checkpoint, int8_conv, set_weight_quant  # noqa: E402
from train import MAX_W, STRIDE, edit_distance, greedy  # noqa: E402
from vocab import RESEARCH  # noqa: E402

DASHES = set("—―─‒–－━-‐")


def canon(s):
    from label_real import canon as c
    return c(s)


def score_norm(s):
    s = canon(s)
    out = []
    for ch in s:
        o = ord(ch)
        if 0xFF01 <= o <= 0xFF5E:
            ch = chr(o - 0xFEE0)
        if ch in DASHES:
            ch = "―"
        if not ch.isspace():
            out.append(ch)
    return "".join(out)


def polarity(g):
    """Invert a crop whose text is darker than its background (median grey)."""
    bg = float(np.median(g))
    ink = g[np.abs(g.astype(np.int16) - bg) > 40]
    if ink.size and ink.mean() < bg:
        return 255 - g
    return g


def eval_modules():
    """eval/'s sets and metrics modules (eval/ here or $VOCR_EVAL_CODE/eval), or None."""
    import os
    for base in (os.environ.get("VOCR_EVAL_CODE"), str(ROOT)):
        if base and (Path(base) / "eval" / "sets.py").exists():
            sys.path.insert(0, str(Path(base) / "eval"))
            import metrics as eval_metrics
            import sets as eval_sets
            return eval_sets, eval_metrics
    return None


def eval_set(name, invert_dark=False):
    """eval's real set as [(grey crop, text, meta)]; crops exactly as eval/run.py cuts them.
    $VOCR_EVAL_MANIFESTS overrides eval's manifest directory (e.g. a not yet merged set)."""
    import os
    mods = eval_modules()
    if mods is None:
        raise SystemExit(f"{name}: eval/ not found (set VOCR_EVAL_CODE to an eval checkout)")
    eval_sets, _ = mods
    if os.environ.get("VOCR_EVAL_MANIFESTS"):
        eval_sets.MANIFESTS = Path(os.environ["VOCR_EVAL_MANIFESTS"])
    from fetch import data_dir
    lines, _ = eval_sets.load(name, data_dir())
    out = []
    for ln in lines:
        g = to_gray(ln.crop_rgb)
        if invert_dark and "dark-on-light" in ln.styles:
            g = 255 - g
        out.append((g, ln.text, {"id": ln.id, "vn": ln.vn, "styles": ln.styles}))
    return out


def real_hand_x(pol="tag"):
    """real-hand without the games of eval's vita-test (independent of it)."""
    mods = eval_modules()
    if mods is None:
        return real_hand(pol)
    games = {r["vn"] for r in mods[0].read_manifest("vita-test")}
    return [it for it in real_hand(pol) if it[2]["vn"] not in games]


def default_sets():
    import os
    mods = eval_modules()
    if mods is None:
        return "synth-val,real-hand,real-pseudo"
    names = ["synth-val", "vita-test", "vita-test-inv", "real-hand-x"]
    manifests = Path(os.environ.get("VOCR_EVAL_MANIFESTS") or mods[0].MANIFESTS)
    if (manifests / "psp-test.jsonl").exists():
        names[3:3] = ["psp-test", "psp-test-inv"]
    return ",".join(names)


def synth_val(limit=None):
    path = sorted(glob.glob(str(DATA / "synth" / "synth_*.npz")))[-1]
    z = np.load(path)
    data, offs, shapes = z["data"], z["offsets"], z["shapes"]
    labels = json.loads(str(z["labels"]))
    meta = json.loads(str(z["meta"]))
    n = len(labels) if limit is None else min(limit, len(labels))
    out = []
    for i in range(n):
        h, w = (int(v) for v in shapes[i])
        out.append((data[offs[i]:offs[i + 1]].reshape(h, w), labels[i],
                    {"cat": meta[i]["cat"], "id": i}))
    return out


def dark_text(kind):
    """Research style tag says dark text on a light background."""
    k = str(kind).lower()
    return k == "dark-on-light" or "dark on light" in k or "black on white" in k


def real_hand(pol="tag"):
    def fix(g, kind):
        if pol == "tag":
            return 255 - g if dark_text(kind) else g
        return polarity(g) if pol == "auto" else g

    items, seen = [], set()
    gt_b = RESEARCH / "gt.json"
    if gt_b.exists():
        for e in json.loads(gt_b.read_text(encoding="utf-8")):
            for k, ln in enumerate(e["lines"]):
                if not ln.get("confident", True):
                    continue
                path = RESEARCH / "lines" / Path(ln["crop"]).name
                rgb = cv2.imread(str(path))[:, :, ::-1]
                g = to_gray(rgb)
                items.append((fix(g, e["kind"]), ln["text"],
                              {"vn": e["vn"], "kind": e["kind"], "id": path.name}))
                seen.add(score_norm(ln["text"]))
    gt_a = RESEARCH / "gt_real.json"
    if gt_a.exists():
        for e in json.loads(gt_a.read_text(encoding="utf-8")):
            for k, text in enumerate(e["lines"]):
                path = RESEARCH / "lines" / f"{e['shot']}_{k}.png"
                if score_norm(text) in seen or not path.exists():
                    continue
                g = to_gray(cv2.imread(str(path))[:, :, ::-1])
                items.append((fix(g, e.get("notes", "")), text,
                              {"vn": e["shot"].split("_")[0], "kind": e.get("notes", ""),
                               "id": path.name}))
                seen.add(score_norm(text))
    return items


def real_pseudo(pol="tag"):
    """The overlay's bright-pixel finder cut these, so every line is light text."""
    items = []
    lines = DATA / "real" / "lines.jsonl"
    for x in lines.read_text(encoding="utf-8").splitlines():
        r = json.loads(x)
        if r["agree"] and r["label"] and is_test_vn(r["vn"]):
            bgr = cv2.imread(str(DATA / "real" / "crops" / f"{r['id']:05d}.png"))
            g = to_gray(bgr[:, :, ::-1])
            items.append((polarity(g) if pol == "auto" else g, r["label"],
                          {"vn": r["vn"], "id": r["id"]}))
    return items


def int16_positions(module, inp):
    """Forward pre-hook: round a conv input per position (over channels) to int16."""
    x = inp[0]
    s = x.abs().amax(dim=1, keepdim=True).clamp(min=1e-30) / 32767
    return (torch.clamp(torch.round(x / s), -32767, 32767) * s,)


def prep(g, height):
    h, w = g.shape
    nw = max(8, min(MAX_W, int(round(w * height / h))))
    return cv2.resize(g, (nw, height), interpolation=cv2.INTER_LINEAR).astype(np.float32)


@torch.no_grad()
def predict(model, crops, height, vocab, batch=64, exact=False):
    """Greedy CTC text per crop. exact=True batches only crops of the same padded width,
    so every crop sees exactly the input of a one-line forward pass (eval/run.py)."""
    gs = [prep(g, height) for g in crops]
    pw = [(g.shape[1] + STRIDE - 1) // STRIDE * STRIDE for g in gs]
    order = np.argsort(pw, kind="stable")
    groups = []
    for i in order:
        if groups and len(groups[-1]) < batch and (not exact or pw[groups[-1][0]] == pw[i]):
            groups[-1].append(i)
        else:
            groups.append([i])
    out = [None] * len(gs)
    for idx in groups:
        W = max(pw[i] for i in idx)
        x = np.empty((len(idx), 1, height, W), np.float32)
        for j, i in enumerate(idx):
            w = gs[i].shape[1]
            x[j, 0, :, :w] = gs[i]
            x[j, 0, :, w:] = gs[i][:, -1:]
        in_len = torch.tensor([math.ceil(gs[i].shape[1] / STRIDE) for i in idx])
        for i, seq in zip(idx, greedy(model(torch.from_numpy(x / 127.5 - 1.0)), in_len)):
            out[i] = "".join(vocab[t - 1] for t in seq)
    return out


def eval_score(preds, items, unk_vocab=None):
    """eval/metrics.score_lines (CER, exact, per style, per game), plus cer_unk_ok."""
    _, eval_metrics = eval_modules()
    rows = [{"id": m["id"], "vn": m["vn"], "styles": m["styles"], "text": t, "pred": p}
            for (_, t, m), p in zip(items, preds)]
    res = eval_metrics.score_lines(rows)
    a = res["all"]
    out = {"cer": a["cer"], "line_acc": a["exact"], "lines": a["lines"], "chars": a["chars"],
           "errors": a["errors"], "styles": {k: v["cer"] for k, v in res["styles"].items()},
           "game_macro_cer": res["game_macro_cer"]}
    if unk_vocab is not None:
        errs = 0
        for (_, t, _), p in zip(items, preds):
            ref = "".join(c if c in unk_vocab else "〓" for c in t)
            errs += eval_metrics.lev(eval_metrics.norm(p), eval_metrics.norm(ref))
        out["cer_unk_ok"] = errs / max(a["chars"], 1)
    return out


def score(preds, refs, unk_vocab=None):
    """CER and exact-line accuracy. With unk_vocab (the set of a reduced vocab, <unk>
    excluded) it also reports cer_unk_ok: the CER when every reference character
    outside that vocab counts as read correctly if the model printed 〓 there, i.e.
    the errors that remain once the vocab's own coverage limit is set aside."""
    errs = chars = exact = errs_unk = 0
    for p, r in zip(preds, refs):
        p, r = score_norm(p), score_norm(r)
        e = edit_distance(list(p), list(r))
        errs, chars, exact = errs + e, chars + len(r), exact + (e == 0)
        if unk_vocab is not None:
            # score_norm folds full-width ASCII, so test membership on both spellings
            ok = lambda c: c in unk_vocab or (0x21 <= ord(c) <= 0x7E  # noqa: E731
                                              and chr(ord(c) + 0xFEE0) in unk_vocab)
            errs_unk += edit_distance(list(p), [c if ok(c) else "〓" for c in r])
    out = {"cer": errs / max(chars, 1), "line_acc": exact / max(len(refs), 1),
           "lines": len(refs), "chars": chars, "errors": errs}
    if unk_vocab is not None:
        out["cer_unk_ok"] = errs_unk / max(chars, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--ckpt", default="ckpt_last.pt")
    ap.add_argument("--sets", default=None,
                    help="comma list (default: eval's sets + real-hand-x if eval/ is found, "
                         "else synth-val,real-hand,real-pseudo)")
    ap.add_argument("--polarity", choices=("tag", "auto", "none"), default="tag")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="first N synth-val lines (tests)")
    ap.add_argument("--save-int8-ckpt", action="store_true")
    args = ap.parse_args()
    from train import default_threads
    torch.set_num_threads(args.threads or default_threads())
    torch.set_flush_denormal(True)
    ck = torch.load(args.run / args.ckpt, map_location="cpu", weights_only=False)
    model, meta = from_checkpoint(ck)
    if args.save_int8_ckpt:
        q = dict(ck)
        q["model"] = {k: v.clone() for k, v in ck["model"].items()}
        for name, m in model.named_modules():
            if int8_conv(m):
                q["model"][f"{name}.weight"] = fake_quant_weight(m.weight.detach()).contiguous()
        q["int8_weights"] = "dequantised per-channel int8 (= runtime W8F)"
        torch.save(q, args.run / f"{Path(args.ckpt).stem}_int8w.pt")
    pol = args.polarity
    loaders = {"synth-val": lambda: synth_val(args.limit), "real-hand": lambda: real_hand(pol),
               "real-pseudo": lambda: real_pseudo(pol), "real-hand-x": lambda: real_hand_x(pol),
               "vita-test": lambda: eval_set("vita-test"),
               "vita-test-inv": lambda: eval_set("vita-test", invert_dark=True),
               "psp-test": lambda: eval_set("psp-test"),
               "psp-test-inv": lambda: eval_set("psp-test", invert_dark=True)}
    sets = {name: loaders[name]() for name in (args.sets or default_sets()).split(",")}
    res = {"run": args.run.name, "ckpt": args.ckpt, "step": ck["step"], "width": meta["width"],
           "bottleneck": meta["bottleneck"], "height": meta["height"],
           "classes": len(meta["vocab"]) + 1, "unk": meta["unk"], "qat": ck.get("qat"),
           "polarity": args.polarity, "results": {}}
    hooks = []
    for variant, bits in (("fp32", 0), ("int8", 8), ("w8i", 8)):
        set_weight_quant(model, bits)
        if variant == "w8i":
            hooks = [m.register_forward_pre_hook(int16_positions)
                     for m in model.modules() if int8_conv(m)]
        for name, items in sets.items():
            t0 = time.time()
            harness = name.startswith(("vita-test", "psp-test"))
            preds = predict(model, [g for g, _, _ in items], meta["height"], meta["vocab"],
                            exact=harness)
            refs = [t for _, t, _ in items]
            unk_vocab = set(meta["vocab"][:-1]) if meta["unk"] else None
            r = eval_score(preds, items, unk_vocab) if harness else score(preds, refs, unk_vocab)
            r["seconds"] = round(time.time() - t0, 1)
            if name != "synth-val":
                r["lines_detail"] = [{"id": str(m["id"]), "ref": t, "pred": p}
                                     for (_, t, m), p in zip(items, preds)]
            else:
                by = {}
                for (_, t, m), p in zip(items, preds):
                    by.setdefault(m["cat"], ([], []))
                    by[m["cat"]][0].append(p)
                    by[m["cat"]][1].append(t)
                r["by_cat"] = {c: score(*v)["cer"] for c, v in sorted(by.items())}
            res["results"][f"{variant}/{name}"] = r
            print(f"{args.run.name} {variant:4s} {name:11s} CER {r['cer'] * 100:6.2f}% "
                  f"line acc {r['line_acc'] * 100:5.1f}% ({r['lines']} lines, {r['chars']} chars, "
                  f"{r['seconds']}s)", flush=True)
    for h in hooks:
        h.remove()
    out = args.run / f"eval_{Path(args.ckpt).stem}{'_limit' if args.limit else ''}.json"
    if out.exists():                       # keep the other sets of an earlier run
        old = json.loads(out.read_text(encoding="utf-8"))
        if old.get("step") == res["step"]:
            res["results"] = {**old.get("results", {}), **res["results"]}
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("EVAL_DONE", out, flush=True)


if __name__ == "__main__":
    main()
