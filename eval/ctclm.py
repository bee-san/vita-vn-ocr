#!/usr/bin/env python3
"""CTC prefix beam search + character n-gram LM (runtime/vocr_lm.c) on development data only.

  ctclm.py extract --blob NAME=PATH [...] --out DIR [--k 16]
      Runs each W8I blob through the C runtime (runtime/pyvocr.py) on every development line in the
      console-polarity view (rv: dark text inverted, as stored), and keeps per line the greedy text
      and, per CTC frame, the K largest logits (descending, ties: lower class first, exactly the
      runtime's own top-K order) and the blank's logit. DIR/NAME.npz.
  ctclm.py decode / tune / report      (below)

Development sets (no protected set: vita-test, psp-test and CLANNAD are never read here):

  tv       teacher_val (VOCR3's copy, 1,643 lines of 180 VNs; the selection cohort of VOCR3/VOCR4)
  dn_ink   devnative v3, hand-cut ink crops (tv_native + strict: 543)
  dn_grad  devnative v3, the console line finder's GRAD crops of the same lines (443; a second view)
  own      the owner pool v1 (owner-pool PR #72: 594 ink crops of the owner's PSP/Vita captures)

Dialogue vs names (the owner's 2026-10-14 13:57Z rule: dialogue CER is primary, names secondary):
a line is a name when devnative tags it `name` (hand-reviewed styles), when the owner pool marks it
`name`, and, in teacher_val (which has no style tags), when it is name-like by the repo's existing rule
(eval/synth3_ablation.strata: at most 6 characters after metrics.norm and no dialogue punctuation),
which also drops teacher_val's UI labels and menu items. Everything else is dialogue.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "runtime"))
from metrics import lev, norm  # noqa: E402

HERD = Path(os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr"))
A = HERD / "shared" / "artifacts"
SEL = A / "synth3" / "vocr3" / "select_pools"
OWNER = HERD / "shared" / "private" / "owner-games" / "_build" / "owner-pool"
SETS = {
    "tv": [SEL / "teacher_val.npz"],
    "dn_ink": [SEL / "devnative_v3" / "devnative_tv_native_ink.npz", SEL / "devnative_v3" / "devnative_strict_ink.npz"],
    "dn_grad": [SEL / "devnative_v3" / "devnative_tv_native_grad.npz",
                SEL / "devnative_v3" / "devnative_strict_grad.npz"],
    "own": [OWNER / "native_train_ink_v2.npz"],
}
PUNCT = set("「」『』。、！？!?…―（）")
PROTECTED_HINTS = ("vita-test", "psp-test", "clannad", "CLANNAD", "fresh")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def namelike(ref):
    return len(ref) <= 6 and not set(ref) & PUNCT


def is_name(set_name, meta, ref):
    if set_name.startswith("dn_"):
        return "name" in (meta.get("styles") or [])
    if set_name == "own":
        return bool(meta.get("name"))
    return namelike(ref)


def load_set(set_name):
    """Rows of one development set: id, vn, ref (metrics.norm), name flag, grey crop as stored (rv)."""
    rows = []
    for path in SETS[set_name]:
        if any(h in str(path) for h in PROTECTED_HINTS):
            raise SystemExit(f"refusing a protected set: {path}")
        z = np.load(path)
        data, offs, shapes = z["data"], z["offsets"], z["shapes"]
        labels = json.loads(str(z["labels"]))
        meta = json.loads(str(z["meta"]))
        for i, (lab, m) in enumerate(zip(labels, meta)):
            if not lab:
                continue
            h, w = (int(v) for v in shapes[i])
            ref = norm(lab)
            rows.append({"set": set_name, "file": path.name, "idx": i, "id": m["id"], "vn": m["vn"], "ref": ref,
                         "name": is_name(set_name, m, ref), "target": m.get("target"), "kind": m.get("kind"),
                         "img": data[offs[i]:offs[i + 1]].reshape(h, w)})
    return rows


def topk_of(logits, k):
    """Per frame, the k largest logits in decreasing order, ties broken by the lower class: a stable
    sort, exactly the order runtime/vocr.c's streaming top-K keeps (strict > comparisons)."""
    order = np.argsort(-logits, axis=1, kind="stable")[:, :k]
    return np.take_along_axis(logits, order, 1), order.astype(np.uint16)


def collapse(best, vocab):
    out, last = [], 0
    for c in best:
        c = int(c)
        if c and c != last:
            out.append(vocab[c])
        last = c
    return "".join(out)


def cmd_extract(args):
    from pyvocr import Vocr, load_lib
    lib = load_lib(args.lib) if args.lib else None
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sets = {s: load_set(s) for s in args.sets.split(",")}
    for spec in args.blob:
        name, _, path = spec.partition("=")
        rt = Vocr(path, mode="w8i", max_width=1024, lib=lib)
        vocab = [rt.class_text(c) for c in range(rt.n_classes)]
        lines, topv, topc, blank = [], [], [], []
        for set_name, rows in sets.items():
            for r in rows:
                text, logits = rt.run(r["img"], logits=True)
                v, c = topk_of(logits, args.k)
                if collapse(c[:, 0], vocab) != text:
                    raise SystemExit(f"{name} {r['id']}: numpy top-1 does not reproduce the runtime's greedy text")
                topv.append(v.astype(np.float32))
                topc.append(c)
                blank.append(logits[:, 0].astype(np.float32))
                lines.append({k: r[k] for k in ("set", "file", "idx", "id", "vn", "ref", "name", "target", "kind")}
                             | {"frames": int(logits.shape[0]), "greedy": text})
            print(f"{name}: {set_name} {len(rows)} lines", flush=True)
        off = np.zeros(len(lines) + 1, np.int64)
        off[1:] = np.cumsum([x["frames"] for x in lines])
        meta = {"kind": "ctclm-extract", "model": name, "blob": str(path), "blob_sha256": sha256(path),
                "n_classes": rt.n_classes, "k": args.k, "view": "rv (console polarity, crops as stored)",
                "sets": {s: [{"file": str(p), "sha256": sha256(p)} for p in SETS[s]] for s in sets},
                "vocab": vocab, "lines": lines}
        np.savez(out / f"{name}.npz", off=off, topv=np.concatenate(topv), topc=np.concatenate(topc),
                 blank=np.concatenate(blank), meta=json.dumps(meta, ensure_ascii=False))
        n = sum(len(x["ref"]) for x in lines)
        e = sum(lev(norm(x["greedy"]), x["ref"]) for x in lines)
        print(f"{name}: {len(lines)} lines, {int(off[-1])} frames, greedy CER {e / n:.4f}", flush=True)


def load_extract(path):
    z = np.load(path)
    meta = json.loads(str(z["meta"]))
    return {"off": z["off"], "topv": z["topv"], "topc": z["topc"], "blank": z["blank"], "meta": meta}


# ---------------------------------------------------------------- confusion families (eval/model_gaps.py, PR #74)

SMALL = {"ぁ": "あ", "ぃ": "い", "ぅ": "う", "ぇ": "え", "ぉ": "お", "っ": "つ", "ゃ": "や", "ゅ": "ゆ", "ょ": "よ",
         "ゎ": "わ", "ァ": "ア", "ィ": "イ", "ゥ": "ウ", "ェ": "エ", "ォ": "オ", "ッ": "ツ", "ャ": "ヤ", "ュ": "ユ",
         "ョ": "ヨ", "ヮ": "ワ", "ヵ": "カ", "ヶ": "ケ"}
FAM_PUNCT = set("、。「」『』（）()［］[]【】〈〉《》〔〕｛｝{}・…‥!?！？~〜～―-—─:;：；,.，．'\"“”‘’")
FAMILIES = ("kanji_kanji", "dakuten", "small_kana", "hira_kata", "dash_like", "punct_punct", "punct_other",
            "other_sub", "insertion", "deletion")


def is_kanji(c):
    o = ord(c)
    return 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF or 0xF900 <= o <= 0xFAFF or c in "々〆ヶ"


def align(ref, pred):
    """One optimal Levenshtein alignment [(op, ref char, pred char)], op in = s d i; ties prefer the
    diagonal, then a deletion, then an insertion (eval/model_gaps.py's align)."""
    n, m = len(ref), len(pred)
    D = [list(range(m + 1))] + [[i] + [0] * m for i in range(1, n + 1)]
    for i in range(1, n + 1):
        Di, Dp, r = D[i], D[i - 1], ref[i - 1]
        for j in range(1, m + 1):
            a = Dp[j - 1] + (r != pred[j - 1])
            b = Dp[j] + 1
            c = Di[j - 1] + 1
            Di[j] = a if a <= b and a <= c else (b if b <= c else c)
    ops, i, j = [], n, m
    while i or j:
        if i and j and D[i][j] == D[i - 1][j - 1] + (ref[i - 1] != pred[j - 1]):
            ops.append(("=" if ref[i - 1] == pred[j - 1] else "s", ref[i - 1], pred[j - 1]))
            i, j = i - 1, j - 1
        elif i and D[i][j] == D[i - 1][j] + 1:
            ops.append(("d", ref[i - 1], None))
            i -= 1
        else:
            ops.append(("i", None, pred[j - 1]))
            j -= 1
    return ops[::-1]


def pair_kind(r, p):
    """The substitution's confusion family (eval/model_gaps.py's pair_kind, PR #74; 'other' -> 'other_sub')."""
    import unicodedata
    br, bp = unicodedata.normalize("NFD", r)[0], unicodedata.normalize("NFD", p)[0]
    if br == bp and r != p:
        return "dakuten"
    if SMALL.get(r) == p or SMALL.get(p) == r:
        return "small_kana"

    def kata(c):
        return chr(ord(c) + 0x60) if 0x3041 <= ord(c) <= 0x3096 else c
    if kata(r) == kata(p):
        return "hira_kata"
    if {r, p} <= set("ー一―-~〜"):
        return "dash_like"
    if r in FAM_PUNCT and p in FAM_PUNCT:
        return "punct_punct"
    if is_kanji(r) and is_kanji(p):
        return "kanji_kanji"
    if (r in FAM_PUNCT) != (p in FAM_PUNCT):
        return "punct_other"
    return "other_sub"


def correct_positions(ref, pred):
    """Per reference character: read correctly (an '=' in align's alignment)?"""
    ok, i = [False] * len(ref), 0
    for op, _, _ in align(ref, pred):
        if op == "i":
            continue
        ok[i] = op == "="
        i += 1
    return ok


def char_group(c, ranks):
    if is_kanji(c):
        return "kanji_rare" if ranks.get(c, 99999) > 2000 else "kanji_common"
    o = ord(c)
    if 0x30A1 <= o <= 0x30FA or c == "ー":
        return "katakana"
    if 0x3041 <= o <= 0x3096:
        return "hiragana"
    if c.isascii() and c.isalnum():
        return "latin_digit"
    if c in FAM_PUNCT:
        return "punct"
    return "other"


def families(ref, pred, pairs=None):
    from collections import Counter
    out = Counter()
    for op, r, p in align(ref, pred):
        if op == "s":
            k = pair_kind(r, p)
            out[k] += 1
            if pairs is not None:
                pairs[(r, p)] += 1
        elif op == "i":
            out["insertion"] += 1
        elif op == "d":
            out["deletion"] += 1
    return out


# ---------------------------------------------------------------- development views

def fold_of(vn):
    """The 2-fold split of the development VNs (by VN id, fixed): fold 0 or 1."""
    return int(hashlib.sha256(f"ctclm-fold:{vn}".encode()).hexdigest(), 16) % 2


class Dev:
    """One model's extracted development lines, with the views the report uses:
      dialogue  tv + dn_ink + own dialogue lines (the primary view; own is left out for a model that trained
                on the owner pool, `in_sample_own`)
      names     the name lines of the same sets (secondary)
      grad      devnative's console-finder GRAD crops, dialogue lines (a second, noisier crop of dn_ink)"""

    def __init__(self, path, in_sample_own=False):
        z = load_extract(path)
        self.path, self.z, self.meta = str(path), z, z["meta"]
        self.lines = self.meta["lines"]
        self.vocab = self.meta["vocab"]
        self.in_sample_own = in_sample_own
        L = self.lines
        prim = ("tv", "dn_ink") + (() if in_sample_own else ("own",))
        self.views = {
            "dialogue": [i for i, x in enumerate(L) if x["set"] in prim and not x["name"]],
            "names": [i for i, x in enumerate(L) if x["set"] in ("tv", "dn_ink", "own") and x["name"]],
            "grad": [i for i, x in enumerate(L) if x["set"] == "dn_grad" and not x["name"]],
            "own": [i for i, x in enumerate(L) if x["set"] == "own" and not x["name"]],
        }
        self.ref = [x["ref"] for x in L]
        self.greedy = [norm(x["greedy"]) for x in L]
        self.greedy_err = np.array([lev(p, r) for p, r in zip(self.greedy, self.ref)], np.int32)
        self.chars = np.array([len(r) for r in self.ref], np.int64)
        self.vn = np.array([x["vn"] for x in L])
        self.fold = np.array([fold_of(v) for v in self.vn])

    def arrays(self, i):
        a, b = int(self.z["off"][i]), int(self.z["off"][i + 1])
        return self.z["topv"][a:b], self.z["topc"][a:b], self.z["blank"][a:b]

    def decode_all(self, dec, idx=None):
        """{line index: normalised prediction} through a pyvocr.VocrLM."""
        out = {}
        for i in (range(len(self.lines)) if idx is None else idx):
            tv, tc, bl = self.arrays(i)
            out[i] = norm(dec.decode(tv, tc, bl))
        return out

    def errors(self, preds):
        e = self.greedy_err.copy()
        for i, p in preds.items():
            if p != self.greedy[i]:
                e[i] = lev(p, self.ref[i])
        return e


def cluster_boot(ea, eb, chars, groups, n_boot=10000, seed=20261014):
    """CER(B) - CER(A) in points with a 95% VN-cluster bootstrap interval (eval/synth3_ablation.bootstrap)."""
    d = (eb.sum() - ea.sum()) / chars.sum() * 100
    ug, inv = np.unique(groups, return_inverse=True)
    ga = np.bincount(inv, weights=ea, minlength=len(ug))
    gb = np.bincount(inv, weights=eb, minlength=len(ug))
    gc = np.bincount(inv, weights=chars, minlength=len(ug))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ug), size=(n_boot, len(ug)))
    diffs = (gb[idx].sum(1) - ga[idx].sum(1)) / gc[idx].sum(1) * 100
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"delta_points": round(float(d), 3), "ci95": [round(float(lo), 3), round(float(hi), 3)],
            "cer_a": round(float(ea.sum() / chars.sum()), 5), "cer_b": round(float(eb.sum() / chars.sum()), 5),
            "errors_a": int(ea.sum()), "errors_b": int(eb.sum()), "chars": int(chars.sum()), "vns": int(len(ug))}


GRID = {"alpha": (150, 250, 350, 500, 700, 1000), "beta": (-500, 0, 500, 1000, 1500, 2500),
        "gate": (1500, 2500, 4000, 6000, 10000), "flags": (0, 1)}


def grid_configs(grid, beam, topk):
    out = []
    for a in grid["alpha"]:
        for b in grid["beta"]:
            for g in grid["gate"]:
                for f in grid["flags"]:
                    for k in grid.get("bracket", [None]):
                        c = {"beam": beam, "topk": topk, "alpha": a, "beta": b, "gate": g, "flags": f}
                        if k is not None:
                            c["bracket"] = k
                        out.append(c)
    return out


def _tune_worker(job):
    from pyvocr import Vocr, VocrLM
    extract, blob, lm_path, cfgs, in_sample_own = job
    dev = Dev(extract, in_sample_own)
    rt = Vocr(blob, mode="w8i", max_width=1024)
    lm = Path(lm_path).read_bytes() if lm_path else None
    rows = []
    for cfg in cfgs:
        dec = VocrLM(rt, lm if cfg["alpha"] else None, **cfg)
        rows.append(dev.errors(dev.decode_all(dec)).astype(np.int16))
    return rows


def cmd_tune(args):
    """Every grid configuration's per-line errors for one model and one LM: DIR/tune_<model>_<lm>.npz."""
    from concurrent.futures import ProcessPoolExecutor
    meta = load_extract(args.extract)["meta"]
    cfgs = [{"beam": 8, "topk": 8, "alpha": 0, "beta": 0, "gate": 0, "flags": 0}]
    cfgs += grid_configs(json.loads(args.grid) if args.grid else GRID, args.beam, args.topk)
    chunks = [cfgs[i::args.workers] for i in range(args.workers)]
    jobs = [(args.extract, meta["blob"], args.lm, ch, args.in_sample_own) for ch in chunks]
    with ProcessPoolExecutor(args.workers) as ex:
        res = list(ex.map(_tune_worker, jobs))
    errs = [None] * len(cfgs)
    for w, rows in enumerate(res):
        for j, r in enumerate(rows):
            errs[w + j * args.workers] = r
    dev = Dev(args.extract, args.in_sample_own)
    if not np.array_equal(errs[0], dev.greedy_err):
        raise SystemExit("alpha = 0 did not reproduce the greedy errors")
    out = Path(args.out)
    np.savez_compressed(out, errors=np.stack(errs), cfgs=json.dumps(cfgs),
                        meta=json.dumps({"extract": args.extract, "extract_sha256": sha256(args.extract),
                                         "lm": args.lm, "lm_sha256": sha256(args.lm) if args.lm else None,
                                         "model": meta["model"], "in_sample_own": args.in_sample_own}))
    d = dev.views["dialogue"]
    base = dev.greedy_err[d].sum()
    best = min(range(len(cfgs)), key=lambda j: (errs[j][d].sum(), j))
    lm_name = Path(args.lm).name if args.lm else "-"
    print(f"{meta['model']} {lm_name}: dialogue greedy {base} -> best {errs[best][d].sum()} "
          f"({cfgs[best]}) of {dev.chars[d].sum()} chars", flush=True)


MODELS = ("v3FL10", "v3H15", "v3W20", "v4FL10", "v4H15", "v4W20")
IN_SAMPLE_OWN = ("v4W20",)        # VOCR4-W20 (arm M) fine-tuned on the owner pool (docs/vocr4.md)
TIERS = (("0.5MB", 500_000), ("1MB", 1_000_000), ("2MB", 2_000_000), ("4MB", 4_000_000))


def load_tune(path):
    z = np.load(path)
    return z["errors"].astype(np.int64), json.loads(str(z["cfgs"])), json.loads(str(z["meta"]))


def tier_of(nbytes):
    for name, cap in TIERS:
        if nbytes <= cap:
            return name
    return None


class Space:
    """Every (LM, configuration) candidate of one tier, with each model's per-line errors."""

    def __init__(self, devs, tune_dir, prefix, lms):
        self.devs, self.cands, self.E = devs, [], {m: [] for m in devs}
        self.greedy = {m: d.greedy_err.astype(np.int64) for m, d in devs.items()}
        for lm in lms:
            for m, d in devs.items():
                e, cfgs, meta = load_tune(Path(tune_dir) / f"{prefix}{m}_{lm}.npz")
                if bool(meta["in_sample_own"]) != d.in_sample_own or meta["extract_sha256"] != sha256(d.path):
                    raise SystemExit(f"{prefix}{m}_{lm}: tuned on another extract or view")
                if not np.array_equal(e[0], self.greedy[m]):
                    raise SystemExit(f"{prefix}{m}_{lm}: configuration 0 is not the greedy baseline")
                self.E[m].append(e[1:])
            self.cands += [(lm, c) for c in cfgs[1:]]
        self.E = {m: np.concatenate(v) for m, v in self.E.items()}

    def pooled(self, view, keep=None):
        """Pooled errors of every candidate over all models' `view` lines (keep: line filter)."""
        tot = np.zeros(len(self.cands), np.int64)
        for m, d in self.devs.items():
            idx = [i for i in d.views[view] if keep is None or keep(d, i)]
            tot += self.E[m][:, idx].sum(1)
        return tot

    def best(self, view="dialogue", keep=None):
        return int(np.argmin(self.pooled(view, keep)))      # the first minimum: grid order breaks ties


def cmd_select(args):
    """Per LM tier: the shared (LM, configuration) with the fewest pooled dialogue errors over the six models,
    its 2-fold VN cross-validated estimate, and every model's result with VN-cluster bootstrap intervals."""
    devs = {m: Dev(Path(args.extract) / f"{m}.npz", in_sample_own=m in IN_SAMPLE_OWN) for m in MODELS}
    lm_bytes = {p.stem: p.stat().st_size for p in Path(args.lm_dir).glob("*.vlm")}
    lms = [x for x in args.lms.split(",") if x]
    out = {"kind": "ctclm-select", "tune": args.prefix, "models": {}, "tiers": {}}
    for m, d in devs.items():
        out["models"][m] = {"extract": d.path, "extract_sha256": sha256(d.path), "blob_sha256": d.meta["blob_sha256"],
                            "in_sample_own": d.in_sample_own,
                            "views": {v: {"lines": len(ix), "chars": int(d.chars[ix].sum()),
                                          "greedy_errors": int(d.greedy_err[ix].sum())} for v, ix in d.views.items()}}
    for tier, cap in TIERS:
        names = [x for x in lms if x in lm_bytes and tier_of(lm_bytes[x]) == tier]
        if not names:
            continue
        sp = Space(devs, args.tune, args.prefix, names)
        j = sp.best()
        lm, cfg = sp.cands[j]
        res = {"lm": lm, "lm_bytes": lm_bytes[lm], "config": cfg, "candidates": len(sp.cands), "lms": names,
               "per_model": {}, "cv": {}}
        cvj = {f: sp.best(keep=lambda d, i, f=f: d.fold[i] != f) for f in (0, 1)}
        res["cv"]["folds"] = {str(f): {"lm": sp.cands[c][0], "config": sp.cands[c][1]} for f, c in cvj.items()}
        for m, d in devs.items():
            r = {}
            for view in ("dialogue", "names", "grad"):
                ix = np.array(d.views[view])
                r[view] = cluster_boot(sp.greedy[m][ix], sp.E[m][j][ix], d.chars[ix], d.vn[ix])
            ix = np.array(d.views["dialogue"])
            cv = np.array([sp.E[m][cvj[int(d.fold[i])]][i] for i in ix])
            r["dialogue_cv"] = cluster_boot(sp.greedy[m][ix], cv, d.chars[ix], d.vn[ix])
            for s in ("tv", "dn_ink", "own"):
                sx = np.array([i for i in ix if d.lines[i]["set"] == s], int)
                if len(sx):
                    r[f"dialogue_{s}"] = cluster_boot(sp.greedy[m][sx], sp.E[m][j][sx], d.chars[sx], d.vn[sx])
            own = np.array(d.views["own"])
            if d.in_sample_own and len(own):
                r["own_in_sample"] = cluster_boot(sp.greedy[m][own], sp.E[m][j][own], d.chars[own], d.vn[own])
            # the per-model optimum (in-sample, for reference only)
            jm = int(np.argmin(sp.E[m][:, ix].sum(1)))
            r["own_best"] = {"lm": sp.cands[jm][0], "config": sp.cands[jm][1],
                             "dialogue": cluster_boot(sp.greedy[m][ix], sp.E[m][jm][ix], d.chars[ix], d.vn[ix])}
            res["per_model"][m] = r
        # pooled over the models (each model's own lines; VN clusters across models share their VN)
        for view in ("dialogue", "names"):
            ga = np.concatenate([sp.greedy[m][devs[m].views[view]] for m in devs])
            gb = np.concatenate([sp.E[m][j][devs[m].views[view]] for m in devs])
            ch = np.concatenate([devs[m].chars[devs[m].views[view]] for m in devs])
            vn = np.concatenate([devs[m].vn[devs[m].views[view]] for m in devs])
            res[f"pooled_{view}"] = cluster_boot(ga, gb, ch, vn)
        # sensitivity: the shared optimum's neighbours (alpha, beta, gate one grid step away) and the gate sweep
        gate_curve = {}
        pd, pn = sp.pooled("dialogue"), sp.pooled("names")
        for k, (lm2, c2) in enumerate(sp.cands):
            same = ("alpha", "beta", "flags", "beam", "topk", "bracket")
            if lm2 == lm and all(c2.get(x) == cfg.get(x) for x in same):
                gate_curve[c2["gate"]] = {"dialogue": int(pd[k]), "names": int(pn[k])}
        res["gate_curve"] = {str(g): v for g, v in sorted(gate_curve.items())}
        res["greedy_pooled"] = {"dialogue": int(sum(devs[m].greedy_err[devs[m].views["dialogue"]].sum() for m in devs)),
                                "names": int(sum(devs[m].greedy_err[devs[m].views["names"]].sum() for m in devs))}
        out["tiers"][tier] = res
        print(f"{tier}: {lm} {cfg} pooled dialogue {res['pooled_dialogue']['errors_a']} -> "
              f"{res['pooled_dialogue']['errors_b']} ({res['pooled_dialogue']['delta_points']:+.3f} points, "
              f"CI {res['pooled_dialogue']['ci95']}); CV folds {res['cv']['folds']}", flush=True)
        for m in devs:
            r = res["per_model"][m]
            print(f"   {m}: dialogue {r['dialogue']['cer_a']:.4f} -> {r['dialogue']['cer_b']:.4f} "
                  f"{r['dialogue']['ci95']}  CV -> {r['dialogue_cv']['cer_b']:.4f} {r['dialogue_cv']['ci95']}  "
                  f"names {r['names']['errors_a']} -> {r['names']['errors_b']}", flush=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def cmd_ablate(args):
    """beam x top-K at one tier's selected parameters (from a select JSON): pooled dialogue errors."""
    from pyvocr import Vocr, VocrLM
    sel = json.loads(Path(args.select).read_text(encoding="utf-8"))["tiers"][args.tier]
    lm_path = Path(args.lm_dir) / f"{sel['lm']}.vlm"
    lm = lm_path.read_bytes()
    out = {"kind": "ctclm-ablate", "tier": args.tier, "lm": sel["lm"], "config": sel["config"], "rows": []}
    devs = {m: Dev(Path(args.extract) / f"{m}.npz", in_sample_own=m in IN_SAMPLE_OWN) for m in MODELS}
    rts = {m: Vocr(d.meta["blob"], mode="w8i", max_width=1024) for m, d in devs.items()}
    for beam in (int(x) for x in args.beams.split(",")):
        for topk in (int(x) for x in args.topks.split(",")):
            cfg = dict(sel["config"], beam=beam, topk=topk)
            row = {"beam": beam, "topk": topk, "dialogue": 0, "names": 0}
            for m, d in devs.items():
                dec = VocrLM(rts[m], lm, **cfg)
                for view in ("dialogue", "names"):
                    ix = d.views[view]
                    row[view] += int(d.errors(d.decode_all(dec, ix))[ix].sum())
            out["rows"].append(row)
            print(row, flush=True)
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")


def cmd_report(args):
    """Confusion families, fixed/introduced errors and the guard analysis at every tier's selection, for each
    model: re-decodes the development lines with the C library (counts and single-character pairs only)."""
    from collections import Counter

    from pyvocr import Vocr, VocrLM
    sel = json.loads(Path(args.select).read_text(encoding="utf-8"))
    ranks = json.loads((ROOT / "data/manifests/fl10_kanji_policy_20261011.json").read_text(encoding="utf-8"))["ranks"]
    out = {"kind": "ctclm-report", "select_sha256": sha256(args.select), "tiers": {}}
    for tier, s in sel["tiers"].items():
        lm = (Path(args.lm_dir) / f"{s['lm']}.vlm").read_bytes()
        res = {}
        for m in MODELS:
            d = Dev(Path(args.extract) / f"{m}.npz", in_sample_own=m in IN_SAMPLE_OWN)
            rt = Vocr(d.meta["blob"], mode="w8i", max_width=1024)
            dec = VocrLM(rt, lm, **s["config"])
            te, tc, _ = load_tune(Path(args.tune) / f"{sel['tune']}{m}_{s['lm']}.npz")
            tj = tc.index(s["config"])
            r = {}
            for view in ("dialogue", "names"):
                ix = d.views[view]
                pred = d.decode_all(dec, ix)
                if not np.array_equal(d.errors(pred)[ix], te[tj][ix]):   # this build == the tuning build
                    raise SystemExit(f"{tier} {m} {view}: the decoder's errors differ from the tuning run's")
                fam_a, fam_b, pa, pb = Counter(), Counter(), Counter(), Counter()
                fixed = introduced = changed = 0
                guard = Counter()
                by_class = {}
                for i in ix:
                    ref, g, b = d.ref[i], d.greedy[i], pred[i]
                    fam_a.update(families(ref, g, pa))
                    fam_b.update(families(ref, b, pb))
                    oka = correct_positions(ref, g)
                    okb = oka if b == g else correct_positions(ref, b)
                    for c, x, y in zip(ref, oka, okb):
                        k = by_class.setdefault(char_group(c, ranks), Counter())
                        k["chars"] += 1
                        k["greedy_correct"] += x
                        k["beam_correct"] += y
                        k["broke"] += x and not y
                        k["fixed"] += y and not x
                    if g != b:
                        changed += 1
                        ea, eb = lev(g, ref), lev(b, ref)
                        fixed += max(0, ea - eb)
                        introduced += max(0, eb - ea)
                        if eb > ea:   # what the LM broke: rare kanji, katakana, Latin/digits
                            rare = any(is_kanji(c) and ranks.get(c, 99999) > 2000 for c in ref)
                            kata = sum(0x30A1 <= ord(c) <= 0x30FA for c in ref) >= max(2, len(ref) // 3)
                            guard["rare_kanji_line"] += rare
                            guard["katakana_line"] += kata
                            guard["latin_digit_line"] += any(c.isascii() and c.isalnum() for c in ref)
                            guard["lines_worse"] += 1
                        elif eb < ea:
                            guard["lines_better"] += 1
                r[view] = {"lines": len(ix), "chars": int(d.chars[ix].sum()), "errors_greedy": int(sum(fam_a.values())),
                           "errors_beam": int(sum(fam_b.values())), "lines_changed": changed,
                           "chars_fixed": fixed, "chars_introduced": introduced, "guard": dict(guard),
                           "by_class": {k: dict(v) for k, v in sorted(by_class.items())},
                           "families_greedy": {k: int(fam_a.get(k, 0)) for k in FAMILIES},
                           "families_beam": {k: int(fam_b.get(k, 0)) for k in FAMILIES},
                           "top_pairs_greedy": [[a, b, n] for (a, b), n in pa.most_common(15)],
                           "top_pairs_beam": [[a, b, n] for (a, b), n in pb.most_common(15)]}
                if sum(fam_a.values()) != int(d.greedy_err[ix].sum()):
                    raise SystemExit(f"{m} {view}: the alignment does not add up to the edit distance")
            res[m] = r
            print(f"{tier} {m}: dialogue families greedy {r['dialogue']['families_greedy']} -> "
                  f"beam {r['dialogue']['families_beam']}; fixed {r['dialogue']['chars_fixed']} "
                  f"introduced {r['dialogue']['chars_introduced']}", flush=True)
        out["tiers"][tier] = res
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def _a9_run(path):
    """One a9_cycles.py measure directory: (counts by address, extras, cache stats, frames, lines)."""
    sys.path.insert(0, str(ROOT / "vita"))
    import a9_cycles
    path = Path(path)
    meta = json.loads((path / "meta.json").read_text())
    counts, extra = a9_cycles.read_counts(path / "counts_w8i.txt")
    cache = a9_cycles.read_cache(path / "cache_w8i.txt")
    return counts, extra, cache, meta["modes"]["w8i"]["frames"], meta["modes"]["w8i"]["lines"], \
        sha256(path / "vocr_cli_vitacc")


def _a9_diff(a, b, c, d):
    """(a - b) - (c - d) of four runs' counts, extras and cache stats (each run: _a9_run's tuple; counts are by
    address, so the runs must come from one binary)."""
    if len({x[5] for x in (a, b, c, d) if len(x) > 5}) > 1:
        raise SystemExit("A9 runs from different vocr_cli builds cannot be differenced")
    keys = set(a[0]) | set(b[0]) | set(c[0]) | set(d[0])
    counts = {k: a[0].get(k, 0) - b[0].get(k, 0) - c[0].get(k, 0) + d[0].get(k, 0) for k in keys}
    bad = {k: v for k, v in counts.items() if v < 0}
    if bad:
        raise SystemExit(f"negative differential counts at {len(bad)} addresses: the runs do not line up")
    ex = {k: a[1].get(k, 0) - b[1].get(k, 0) - c[1].get(k, 0) + d[1].get(k, 0) for k in a[1]}
    cache = {k: max(0, a[2][k] - b[2][k] - c[2][k] + d[2][k]) for k in a[2]}
    return {k: v for k, v in counts.items() if v}, ex, cache


def _a9_cost(dirpath, counts, extra, cache, frames):
    """a9_cycles.model_mode over the runtime's functions for differential counts: cycles per frame (444, 333)."""
    sys.path.insert(0, str(ROOT / "vita"))
    import a9_cycles
    d = Path(dirpath)
    rows = a9_cycles.parse_dis((d / "cli.dis").read_text())
    funcs = set((d / "vocr_functions.txt").read_text().split())
    return a9_cycles.model_mode(rows, counts, extra, funcs, frames, cache)


def cmd_a9(args):
    """The A9 estimate of each component per model (vita/a9_cycles.py's cost model, Vita machine code under
    qemu-arm with the Vita-like cache simulator), from the runs of a9/run_final.sh on a development pack of N
    lines and the same pack twice (2N): the network alone (base), + top-K (topk8), + each tier's beam + LM.
    Per-line costs are slopes, (run at 2N - run at N) / N, so the one-time LM load (CRC, vocr_lm_load,
    vocr_lm_init) drops out; it is reported separately. Differences between two runs on the same pack hold
    only what the second adds (instruction counts are deterministic), including the extra cache misses the
    LM causes in the network's own work."""
    d, out = Path(args.dir), {"kind": "ctclm-a9", "estimate": True, "models": {}}
    clocks = (444, 333)
    for m in args.models.split(","):
        runs = {}
        for p in d.glob(f"f_{m}_*_dev*"):
            name, pack = p.name[len(f"f_{m}_"):].rsplit("_", 1)
            if (p / "meta.json").exists():
                runs[(name, pack)] = _a9_run(p)
        if ("base", "dev40") not in runs:
            continue
        b40, b80 = runs[("base", "dev40")], runs[("base", "dev80")]
        n_lines, frames = b40[4], b40[3]
        if b80[4] != 2 * n_lines or b80[3] != 2 * frames:
            raise SystemExit(f"{m}: the 2N pack is not the N pack twice")
        net = _a9_cost(d / f"f_{m}_base_dev40", b40[0], b40[1], b40[2], frames)
        res = {"lines": n_lines, "frames": frames, "frames_per_line": frames / n_lines,
               "network_ms_640px": {str(c): [x * 80 / (c * 1000) for x in net["cycles_per_frame"][c]] for c in clocks}}
        zero = ({}, {k: 0 for k in b40[1]}, {k: 0 for k in b40[2]}, 0, 0)
        comps = {}
        for name in sorted({n for n, _ in runs if n != "base"}):
            ref40, ref80 = (b40, b80) if name == "topk8" else (runs[("topk8", "dev40")], runs[("topk8", "dev80")])
            r40, r80 = runs[(name, "dev40")], runs[(name, "dev80")]
            c, ex, ca = _a9_diff(r80, r40, ref80, ref40)          # per-line work of N lines
            cost = _a9_cost(d / f"f_{m}_{name}_dev40", c, ex, ca, frames)
            comp = {"insns_per_frame": cost["insns_per_frame"], "l1d_misses_per_frame": cost["l1d_misses_per_frame"],
                    "l2_misses_per_frame": cost["l2_misses_per_frame"],
                    "ms_per_dev_line": {str(k): [x * frames / n_lines / (k * 1000) for x in cost["cycles_per_frame"][k]]
                                        for k in clocks},
                    "ms_640px": {str(k): [x * 80 / (k * 1000) for x in cost["cycles_per_frame"][k]] for k in clocks},
                    "top_functions": cost["top_functions"]}
            if name != "topk8":
                lc, lex, lca = _a9_diff(r40, ref40, (c, ex, ca), zero[:3])   # once: (lm - topk at N) - N lines
                load = _a9_cost(d / f"f_{m}_{name}_dev40", lc, lex, lca, 1)
                comp["load_ms"] = {str(k): [x / (k * 1000) for x in load["cycles_per_frame"][k]] for k in clocks}
            comps[name] = comp
        res["components"] = comps
        out["models"][m] = res
        print(m, json.dumps({k: {kk: [round(x, 2) for x in vv] for kk, vv in v["ms_640px"].items()}
                             for k, v in comps.items()}), flush=True)
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")


def conf_analysis(devs, k=8):
    """conf_min (the smallest softmax-max probability over the frames emitting a greedy character, over each
    frame's top-k logits, as vocr_run's vocr_extra.conf_min) as a predictor of a dialogue line with an error."""
    out = {}
    for m, d in devs.items():
        cmin, err = [], []
        for i in d.views["dialogue"]:
            tv, tc, _ = d.arrays(i)
            v = tv[:, :k].astype(np.float64)
            p = 1 / np.exp(v - v[:, :1]).sum(1)
            c = tc[:, 0].astype(int)
            emit = [f for f in range(len(c)) if c[f] and (f == 0 or c[f] != c[f - 1])]
            cmin.append(float(p[emit].min()) if emit else 1.0)
            err.append(bool(d.greedy_err[i] > 0))
        cmin, err = np.array(cmin), np.array(err)
        a, b = np.sort(cmin[err]), np.sort(cmin[~err])
        left, right = np.searchsorted(b, a, "left"), np.searchsorted(b, a, "right")
        auroc = float(((len(b) - right) + 0.5 * (right - left)).sum() / (len(a) * len(b)))
        rows = []
        for t in (0.5, 0.7, 0.8, 0.9, 0.95):
            f = cmin < t
            rows.append({"threshold": t, "flagged": round(float(f.mean()), 4), "recall": round(float(f[err].mean()), 4),
                         "precision": round(float(err[f].mean()) if f.any() else 0.0, 4)})
        out[m] = {"lines": int(len(err)), "lines_with_error": int(err.sum()), "auroc": round(auroc, 4), "rows": rows}
    return out


def recipe_table(devs, tune_dir, prefix, lm_dir, lms):
    rows = []
    for lm in lms:
        p = Path(tune_dir) / f"{prefix}{devs and next(iter(devs))}_{lm}.npz"
        if not p.exists():
            continue
        sp = Space(devs, tune_dir, prefix, [lm])
        pd, pn = sp.pooled("dialogue"), sp.pooled("names")
        j = int(np.argmin(pd))
        f = {x: sp.best(keep=lambda d, i, x=x: d.fold[i] != x) for x in (0, 1)}
        cv = sum(int(sp.E[m][f[int(d.fold[i])]][i]) for m, d in devs.items() for i in d.views["dialogue"])
        man = json.loads((Path(lm_dir) / f"{lm}.json").read_text(encoding="utf-8"))
        rows.append({"lm": lm, "bytes": man["bytes"], "order": man["order"],
                     "corpora": [c["name"] for c in man["corpora"]],
                     "jiten_share": {x["name"]: x["share"] for x in man["jiten"]},
                     "heldout_nats_per_char": min(x["heldout_nats_per_char"] for x in man["profiles"]),
                     "dialogue_errors": int(pd[j]), "dialogue_errors_cv": cv, "names_errors": int(pn[j]),
                     "config": sp.cands[j][1]})
    return rows


def topk_oracle(devs, k=8):
    """Per model: greedy dialogue substitutions, and how many of them have the reference character among the
    emitting frame's top k (the ones a top-k search could fix at all)."""
    out = {}
    for m, d in devs.items():
        subs = inside = 0
        for i in d.views["dialogue"]:
            if not d.greedy_err[i]:
                continue
            tv, tc, _ = d.arrays(i)
            best = tc[:, 0].astype(int)
            emit = [f for f in range(len(best)) if best[f] and (f == 0 or best[f] != best[f - 1])]
            chars = [norm(d.vocab[best[f]]) for f in emit]
            if "".join(chars) != d.greedy[i] or any(len(c) != 1 for c in chars):
                continue                                  # normalisation merged characters: skip the line
            j = 0
            for op, r, p in align(d.ref[i], d.greedy[i]):
                if op in ("=", "s"):
                    if op == "s":
                        subs += 1
                        inside += r in {norm(d.vocab[int(c)]) for c in tc[emit[j], :k]}
                    j += 1
                elif op == "i":
                    j += 1
        out[m] = {"substitutions": subs, f"reference_in_top{k}": inside}
    return out


def cmd_summary(args):
    """docs/ctc-lm-decode.json, whole, from the selection, report, ablation, A9 and LM build outputs."""
    from pyvocr import Vocr, VocrLM
    sel = json.loads(Path(args.select).read_text(encoding="utf-8"))
    rep = json.loads(Path(args.report).read_text(encoding="utf-8"))
    devs = {m: Dev(Path(args.extract) / f"{m}.npz", in_sample_own=m in IN_SAMPLE_OWN) for m in MODELS}
    out = {"kind": "ctc-lm-decode", "schema": 1,
           "note": "Development data only (teacher_val, devnative v3, the owner pool); no protected set was read. "
                   "Every console time is a vita/a9_cycles.py estimate.",
           "development": {"sets": {s: [{"file": p.name, "sha256": sha256(p)} for p in SETS[s]] for s in SETS},
                           "views": {"dialogue": "tv + dn_ink + own lines that are not names (own left out for "
                                                 "VOCR4-W20, which trained on it)",
                                     "names": "devnative 'name' style, owner-pool name flag, teacher_val name-like",
                                     "grad": "devnative's console-finder GRAD crops of the dialogue lines"},
                           "polarity": "rv: the console polarity (dark text inverted), crops as stored",
                           "folds": "2 folds by sha256('ctclm-fold:' + VN id) % 2",
                           "bootstrap": "10,000 VN-cluster resamples, seed 20261014, 95% percentile interval"},
           "models": {m: {"blob_sha256": d.meta["blob_sha256"], "n_classes": d.meta["n_classes"],
                          "in_sample_own": d.in_sample_own} for m, d in devs.items()},
           "selection": {"rule": "per LM tier, the (configuration) with the fewest pooled development dialogue errors "
                                 "over the six models (first minimum in grid order); 2-fold VN cross-validation "
                                 "re-selects on one fold and scores the other",
                         "grids": json.loads(args.grids) if args.grids else None},
           "tiers": {}, "families": {}, "recipes": None, "ablation": {}, "confidence": conf_analysis(devs),
           "oracle": topk_oracle(devs)}
    for tier, s in sel["tiers"].items():
        t = {k: s[k] for k in ("lm", "lm_bytes", "config", "candidates", "pooled_dialogue", "pooled_names",
                               "greedy_pooled", "gate_curve")}
        t["cv_folds"] = s["cv"]["folds"]
        t["per_model"] = s["per_model"]
        rel = dict(x.split("=") for x in args.release_map.split(","))[tier]
        rblob = (Path(args.release) / f"{rel}.vlm").read_bytes()
        tuned = (Path(args.lm_dir) / f"{s['lm']}.vlm").read_bytes()
        if rblob[64:] != tuned[64:] or rblob[:32] != tuned[:32]:      # only the provenance prefix may differ
            raise SystemExit(f"{tier}: {rel}.vlm is not the tuned {s['lm']}.vlm")
        man = json.loads((Path(args.release) / f"{rel}.json").read_text(encoding="utf-8"))
        t["release"] = {"file": f"{rel}.vlm", "bytes": len(rblob), "sha256": hashlib.sha256(rblob).hexdigest(),
                        "tuned_as": s["lm"], "manifest": {k: v for k, v in man.items() if k not in ("paths",)}}
        out["tiers"][tier] = t
        out["families"][tier] = {m: {v: {k: r[v][k] for k in ("lines", "chars", "errors_greedy", "errors_beam",
                                                              "lines_changed", "chars_fixed", "chars_introduced",
                                                              "guard", "by_class", "families_greedy",
                                                              "families_beam")}
                                     for v in ("dialogue", "names")} for m, r in rep["tiers"][tier].items()}
    for p in args.ablate:
        a = json.loads(Path(p).read_text(encoding="utf-8"))
        out["ablation"][a["tier"]] = {"lm": a["lm"], "config": a["config"], "rows": a["rows"]}
    if args.recipes:
        out["recipes"] = recipe_table(devs, args.tune, args.recipe_prefix, args.lm_dir, args.recipes.split(","))
    if args.a9:
        out["a9"] = json.loads(Path(args.a9).read_text(encoding="utf-8"))
    if args.gate_sweep:
        prefix, lm = args.gate_sweep.split(":")
        sp = Space(devs, args.tune, prefix, [lm])
        pd, pn = sp.pooled("dialogue"), sp.pooled("names")
        greedy = {v: int(sum(d.greedy_err[d.views[v]].sum() for d in devs.values())) for v in ("dialogue", "names")}
        out["gate_sweep"] = {"lm": lm, "config": {k: v for k, v in sp.cands[0][1].items() if k != "gate"},
                             "greedy": greedy,
                             "rows": [{"gate": c["gate"], "dialogue": int(pd[k]), "names": int(pn[k])}
                                      for k, (_, c) in enumerate(sp.cands)]}
    # memory: the LM blob (resident), the decoder workspace (vocr_lm_workspace_bytes) for a 1024-px line, and the
    # caller's top-K buffers (frames x K x (4 + 2) + frames x 4 bytes), per model and tier
    mem = {}
    for tier, s in sel["tiers"].items():
        lm = (Path(args.lm_dir) / f"{s['lm']}.vlm").read_bytes()
        row = {"lm_bytes": len(lm), "workspace": {}, "topk_buffers": {}}
        for m, d in devs.items():
            rt = Vocr(d.meta["blob"], mode="w8i", max_width=1024)
            dec = VocrLM(rt, lm, **s["config"])
            frames = 1024 // rt.stride
            row["workspace"][m] = dec.workspace_bytes
            row["topk_buffers"][m] = frames * s["config"]["topk"] * 6 + frames * 4
        mem[tier] = row
    out["memory"] = mem
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


NAMES = {"v3FL10": "VOCR3-FL10", "v3H15": "VOCR3-H15", "v3W20": "VOCR3-W20", "v4FL10": "VOCR4-FL10",
         "v4H15": "VOCR4-H15", "v4W20": "VOCR4-W20"}


def _pct(x):
    return f"{100 * x:.2f}%"


def _ci(r):
    return f"{r['delta_points']:+.2f} ({r['ci95'][0]:+.2f} to {r['ci95'][1]:+.2f})"


def cmd_tables(args):
    """The markdown tables of docs/ctc-lm-decode.md, from docs/ctc-lm-decode.json."""
    j = json.loads(Path(args.json).read_text(encoding="utf-8"))
    tiers = list(j["tiers"])
    p = []
    p.append("Dialogue CER, greedy → beam + LM (2-fold VN cross-validated in brackets):\n")
    p.append("| model | dialogue chars | greedy | " + " | ".join(f"{t} LM" for t in tiers) + " |")
    p.append("|---|---:|---:|" + "---:|" * len(tiers))
    for m in MODELS:
        r0 = j["tiers"][tiers[0]]["per_model"][m]["dialogue"]
        cells = [f"{_pct(j['tiers'][t]['per_model'][m]['dialogue']['cer_b'])} "
                 f"({_pct(j['tiers'][t]['per_model'][m]['dialogue_cv']['cer_b'])})" for t in tiers]
        p.append(f"| {NAMES[m]} | {r0['chars']:,} | {_pct(r0['cer_a'])} | " + " | ".join(cells) + " |")
    pd = {t: j["tiers"][t]["pooled_dialogue"] for t in tiers}
    p.append(f"| pooled (six models) | {pd[tiers[0]]['chars']:,} | {_pct(pd[tiers[0]]['cer_a'])} | "
             + " | ".join(_pct(pd[t]["cer_b"]) for t in tiers) + " |")
    p.append("\nChange in dialogue CER, points, with the 95% VN-cluster bootstrap interval (in-sample selection; "
             "cross-validated):\n")
    p.append("| model | " + " | ".join(f"{t}" for t in tiers) + " | "
             + " | ".join(f"{t} CV" for t in tiers[-1:]) + " |")
    p.append("|---|" + "---|" * (len(tiers) + 1))
    for m in MODELS:
        pm = {t: j["tiers"][t]["per_model"][m] for t in tiers}
        p.append(f"| {NAMES[m]} | " + " | ".join(_ci(pm[t]["dialogue"]) for t in tiers) + " | "
                 + _ci(pm[tiers[-1]]["dialogue_cv"]) + " |")
    p.append("| pooled | " + " | ".join(_ci(pd[t]) for t in tiers) + " | |")
    p.append("\nSecondary views: names (CER) and the console finder's GRAD crops of devnative's dialogue lines:\n")
    p.append("| model | names greedy | " + " | ".join(f"names {t}" for t in tiers) + " | GRAD greedy | GRAD "
             + tiers[-1] + " |")
    p.append("|---|---:|" + "---:|" * len(tiers) + "---:|---:|")
    for m in MODELS:
        pm = {t: j["tiers"][t]["per_model"][m] for t in tiers}
        p.append(f"| {NAMES[m]} | {_pct(pm[tiers[0]]['names']['cer_a'])} | "
                 + " | ".join(_pct(pm[t]["names"]["cer_b"]) for t in tiers)
                 + f" | {_pct(pm[tiers[0]]['grad']['cer_a'])} | {_pct(pm[tiers[-1]]['grad']['cer_b'])} |")
    # families, pooled over the models
    p.append("\nConfusion families on dialogue lines, summed over the six models (eval/model_gaps.py's pair_kind; "
             "substitutions, then insertions and deletions):\n")
    p.append("| family | greedy | " + " | ".join(tiers) + " |")
    p.append("|---|---:|" + "---:|" * len(tiers))
    for fam in FAMILIES:
        g = sum(j["families"][tiers[0]][m]["dialogue"]["families_greedy"][fam] for m in MODELS)
        p.append(f"| {fam} | {g} | " + " | ".join(
            str(sum(j["families"][t][m]["dialogue"]["families_beam"][fam] for m in MODELS)) for t in tiers) + " |")
    tot = {t: sum(j["families"][t][m]["dialogue"]["errors_beam"] for m in MODELS) for t in tiers}
    g = sum(j["families"][tiers[0]][m]["dialogue"]["errors_greedy"] for m in MODELS)
    p.append(f"| **all** | {g} | " + " | ".join(str(tot[t]) for t in tiers) + " |")
    p.append("\nPer model at the 1MB tier (greedy → beam):\n")
    fams = ("kanji_kanji", "dakuten", "small_kana", "punct_punct", "dash_like", "other_sub", "deletion", "insertion")
    p.append("| model | " + " | ".join(fams) + " |")
    p.append("|---|" + "---|" * len(fams))
    for m in MODELS:
        f = j["families"]["1MB"][m]["dialogue"]
        p.append(f"| {NAMES[m]} | " + " | ".join(f"{f['families_greedy'][k]} → {f['families_beam'][k]}" for k in fams)
                 + " |")
    # guard
    p.append("\nWhat the LM changes, dialogue lines summed over the six models (characters fixed and introduced, "
             "lines better and worse; the last three columns: introduced-error lines with a rare kanji (Jiten rank > "
             "2,000), mostly katakana, or Latin/digits in the reference):\n")
    p.append("| tier | lines changed | chars fixed | chars introduced | lines better | lines worse | rare kanji | "
             "katakana | Latin/digits |")
    p.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for t in tiers:
        f = [j["families"][t][m]["dialogue"] for m in MODELS]
        gs = {k: sum(x["guard"].get(k, 0) for x in f) for k in ("lines_better", "lines_worse", "rare_kanji_line",
                                                                "katakana_line", "latin_digit_line")}
        p.append(f"| {t} | {sum(x['lines_changed'] for x in f)} | {sum(x['chars_fixed'] for x in f)} | "
                 f"{sum(x['chars_introduced'] for x in f)} | {gs['lines_better']} | {gs['lines_worse']} | "
                 f"{gs['rare_kanji_line']} | {gs['katakana_line']} | {gs['latin_digit_line']} |")
    p.append("\nReference characters by class, summed over the six models at the 1MB tier: read correctly by the "
             "greedy text and by the beam, and the characters the beam broke (greedy right, beam wrong) and fixed:\n")
    p.append("| view | class | chars | greedy correct | beam correct | broke | fixed |")
    p.append("|---|---|---:|---:|---:|---:|---:|")
    for view in ("dialogue", "names"):
        groups = sorted({g for m in MODELS for g in j["families"]["1MB"][m][view]["by_class"]})
        for g in groups:
            v = [j["families"]["1MB"][m][view]["by_class"].get(g, {}) for m in MODELS]
            n = {k: sum(x.get(k, 0) for x in v) for k in ("chars", "greedy_correct", "beam_correct", "broke", "fixed")}
            p.append(f"| {view} | {g} | {n['chars']:,} | {n['greedy_correct']:,} | {n['beam_correct']:,} | "
                     f"{n['broke']} | {n['fixed']} |")
    p.append("\nThe same for name lines:\n")
    p.append("| tier | lines changed | chars fixed | chars introduced | lines better | lines worse |")
    p.append("|---|---:|---:|---:|---:|---:|")
    for t in tiers:
        f = [j["families"][t][m]["names"] for m in MODELS]
        p.append(f"| {t} | {sum(x['lines_changed'] for x in f)} | {sum(x['chars_fixed'] for x in f)} | "
                 f"{sum(x['chars_introduced'] for x in f)} | {sum(x['guard'].get('lines_better', 0) for x in f)} | "
                 f"{sum(x['guard'].get('lines_worse', 0) for x in f)} |")
    gp = j["tiers"][tiers[0]]["greedy_pooled"]
    p.append("\nThe gate at each tier's other settings, pooled errors over the six models (dialogue / names; "
             f"greedy {gp['dialogue']} / {gp['names']}):\n")
    gates = sorted({int(g) for t in tiers for g in j["tiers"][t]["gate_curve"]})
    p.append("| tier | " + " | ".join(f"gate {g / 1000:g} nat" for g in gates) + " |")
    p.append("|---|" + "---|" * len(gates))
    for t in tiers:
        gc = j["tiers"][t]["gate_curve"]
        p.append(f"| {t} | " + " | ".join(f"{gc[str(g)]['dialogue']} / {gc[str(g)]['names']}" if str(g) in gc else "–"
                                          for g in gates) + " |")
    if j.get("gate_sweep"):
        g = j["gate_sweep"]
        p.append(f"\nThe gate from 0.5 to 64 nats at the {g['lm']} settings otherwise (pooled errors over the six "
                 f"models; greedy {g['greedy']['dialogue']} dialogue / {g['greedy']['names']} names):\n")
        p.append("| gate (nat) | " + " | ".join(f"{r['gate'] / 1000:g}" for r in g["rows"]) + " |")
        p.append("|---|" + "---:|" * len(g["rows"]))
        p.append("| dialogue errors | " + " | ".join(str(r["dialogue"]) for r in g["rows"]) + " |")
        p.append("| name errors | " + " | ".join(str(r["names"]) for r in g["rows"]) + " |")
    for t, a in j["ablation"].items():
        beams = sorted({r["beam"] for r in a["rows"]})
        ks = sorted({r["topk"] for r in a["rows"]})
        p.append(f"\nBeam width × top-K at the {t} tier's other settings (pooled dialogue errors, six models):\n")
        p.append("| beam | " + " | ".join(f"top-{k}" for k in ks) + " |")
        p.append("|---|" + "---:|" * len(ks))
        for b in beams:
            p.append(f"| {b} | " + " | ".join(str(next(r["dialogue"] for r in a["rows"] if r["beam"] == b
                                                        and r["topk"] == k)) for k in ks) + " |")
    if j.get("recipes"):
        p.append("\nLM recipes (round 2's grid; pooled dialogue errors at each LM's best shared configuration, "
                 "cross-validated, names; held-out cross-entropy on realplus v2 labels):\n")
        p.append("| LM | bytes | order | corpora | Jiten share | held-out nats/char | dialogue | CV | names |")
        p.append("|---|---:|---:|---|---|---:|---:|---:|---:|")
        for r in j["recipes"]:
            js = ", ".join(f"{k} {v:g}" for k, v in r["jiten_share"].items()) or "none"
            cor = "Aozora" if r["corpora"] == ["aozora_dialogue", "aozora_narration"] else \
                "Aozora + Aratako + JESC + Tatoeba" if len(r["corpora"]) == 5 else "+".join(r["corpora"])
            p.append(f"| {r['lm']} | {r['bytes']:,} | {r['order']} | {cor} | {js} | {r['heldout_nats_per_char']:.3f} "
                     f"| {r['dialogue_errors']} | {r['dialogue_errors_cv']} | {r['names_errors']} |")
    if j.get("a9"):
        a9 = j["a9"]["models"]
        p.append("\nEstimated Cortex-A9 cost (vita/a9_cycles.py on the Vita build, development lines), ms per 640-px "
                 "line (80 CTC frames) at 444 MHz, low–high (333 MHz in brackets):\n")
        p.append("| model | network (greedy) | + top-8 | " + " | ".join(f"+ beam + {t} LM" for t in tiers)
                 + " | LM load, once |")
        p.append("|---|---|---|" + "---|" * len(tiers) + "---|")
        for m, r in a9.items():
            def cell(x):
                lo, hi = x["444"]
                lo3, hi3 = x["333"]
                return f"{lo:.1f}–{hi:.1f} ({lo3:.1f}–{hi3:.1f})"
            comps = r["components"]
            loads = "; ".join(f"{t} {comps['lm' + t]['load_ms']['444'][0]:.0f}–"
                              f"{comps['lm' + t]['load_ms']['444'][1]:.0f}" for t in tiers if "lm" + t in comps)
            p.append(f"| {NAMES.get(m, m)} | {cell(r['network_ms_640px'])} | {cell(comps['topk8']['ms_640px'])} | "
                     + " | ".join(cell(comps["lm" + t]["ms_640px"]) if "lm" + t in comps else "–" for t in tiers)
                     + f" | {loads} ms |")
        p.append("\nPer typical full textbox (3 lines of about 640 px) at 444 MHz, the network plus top-K plus "
                 "beam + LM, estimated, against the tier limits (Small ≤ 1 s, Normal ≤ 2 s, Large ≤ 3 s):\n")
        p.append("| model | greedy box | " + " | ".join(f"with the {t} LM" for t in tiers) + " |")
        p.append("|---|---|" + "---|" * len(tiers))
        for m, r in a9.items():
            comps = r["components"]
            net = r["network_ms_640px"]["444"]
            tk = comps["topk8"]["ms_640px"]["444"]
            cells = []
            for t in tiers:
                c = comps.get("lm" + t)
                if not c:
                    cells.append("–")
                    continue
                lo = 3 * (net[0] + tk[0] + c["ms_640px"]["444"][0]) / 1000
                hi = 3 * (net[1] + tk[1] + c["ms_640px"]["444"][1]) / 1000
                cells.append(f"{lo:.2f}–{hi:.2f} s")
            p.append(f"| {NAMES.get(m, m)} | {3 * net[0] / 1000:.2f}–{3 * net[1] / 1000:.2f} s | "
                     + " | ".join(cells) + " |")
    p.append("\nMemory per tier: the LM blob (resident while the game runs, read-only), the decoder workspace "
             "(vocr_lm_workspace_bytes, a 1,024-px line) and the caller's top-K buffers:\n")
    p.append("| tier | LM file | LM bytes | sha256 | workspace (FL10 / H15, W20) | top-8 buffers |")
    p.append("|---|---|---:|---|---|---:|")
    for t in tiers:
        mm = j["memory"][t]
        rel = j["tiers"][t]["release"]
        p.append(f"| {t} | `{rel['file']}` | {rel['bytes']:,} | `{rel['sha256'][:16]}…` | "
                 f"{mm['workspace']['v3FL10']:,} / {mm['workspace']['v3W20']:,} B | "
                 f"{mm['topk_buffers']['v3W20']:,} B |")
    if j.get("oracle"):
        o = j["oracle"]
        p.append("\nGreedy dialogue substitutions whose reference character is among the emitting frame's top 8 (the "
                 "ones a top-8 search can fix at all):\n")
        p.append("| model | substitutions | reference in the top 8 |")
        p.append("|---|---:|---:|")
        for m, r in o.items():
            p.append(f"| {NAMES[m]} | {r['substitutions']} | {r['reference_in_top8']} "
                     f"({100 * r['reference_in_top8'] / max(1, r['substitutions']):.0f}%) |")
    p.append("\nThe greedy path's confidence (vocr_extra.conf_min over top-8) as a flag for a dialogue line with an "
             "error (greedy; AUROC, and the share of lines flagged / errors caught at three thresholds):\n")
    p.append("| model | lines (with an error) | AUROC | < 0.5 | < 0.7 | < 0.9 |")
    p.append("|---|---|---:|---|---|---|")
    for m, c in j["confidence"].items():
        rows = {r["threshold"]: r for r in c["rows"]}
        cell = lambda r: f"{100 * r['flagged']:.0f}% / {100 * r['recall']:.0f}%"  # noqa: E731
        p.append(f"| {NAMES[m]} | {c['lines']:,} ({c['lines_with_error']}) | {c['auroc']:.3f} | {cell(rows[0.5])} | "
                 f"{cell(rows[0.7])} | {cell(rows[0.9])} |")
    print("\n".join(p))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--blob", action="append", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--k", type=int, default=16)
    e.add_argument("--sets", default="tv,dn_ink,dn_grad,own")
    e.add_argument("--lib", help="a built libvocr.so to load (default: runtime/build/libvocr.so, rebuilt if stale)")
    t = sub.add_parser("tune")
    t.add_argument("--extract", required=True)
    t.add_argument("--lm", default="")
    t.add_argument("--out", required=True)
    t.add_argument("--beam", type=int, default=8)
    t.add_argument("--topk", type=int, default=8)
    t.add_argument("--grid", default="", help="JSON {alpha: [...], beta: [...], gate: [...], flags: [...]}")
    t.add_argument("--workers", type=int, default=5)
    t.add_argument("--in-sample-own", action="store_true",
                   help="the model trained on the owner pool: leave it out of the dialogue view")
    s = sub.add_parser("select")
    s.add_argument("--extract", required=True, help="the directory of extract's NAME.npz files")
    s.add_argument("--tune", required=True, help="the directory of tune's npz files")
    s.add_argument("--prefix", default="r2_", help="tune file names: PREFIX<model>_<lm>.npz")
    s.add_argument("--lm-dir", required=True)
    s.add_argument("--lms", required=True, help="candidate LM names (their tier comes from their bytes)")
    s.add_argument("--out", required=True)
    b = sub.add_parser("ablate")
    b.add_argument("--extract", required=True)
    b.add_argument("--select", required=True)
    b.add_argument("--tier", required=True)
    b.add_argument("--lm-dir", required=True)
    b.add_argument("--beams", default="1,2,4,8,16")
    b.add_argument("--topks", default="2,4,8,16")
    b.add_argument("--out", required=True)
    r = sub.add_parser("report")
    r.add_argument("--extract", required=True)
    r.add_argument("--select", required=True)
    r.add_argument("--lm-dir", required=True)
    r.add_argument("--tune", required=True)
    r.add_argument("--out", required=True)
    q = sub.add_parser("a9")
    q.add_argument("--dir", required=True, help="the directory of a9/run_final.sh's runs")
    q.add_argument("--models", default="v3FL10,v3W20,v4W20")
    q.add_argument("--out", required=True)
    y = sub.add_parser("summary")
    y.add_argument("--extract", required=True)
    y.add_argument("--select", required=True)
    y.add_argument("--report", required=True)
    y.add_argument("--ablate", nargs="*", default=[])
    y.add_argument("--a9")
    y.add_argument("--tune", required=True)
    y.add_argument("--lm-dir", required=True)
    y.add_argument("--release", required=True)
    y.add_argument("--release-map", required=True, help="TIER=NAME,... the release file of each tier's LM")
    y.add_argument("--recipes", default="")
    y.add_argument("--recipe-prefix", default="r2_")
    y.add_argument("--grids", default="")
    y.add_argument("--gate-sweep", default="", help="PREFIX:LM of tune files that vary the gate only")
    y.add_argument("--out", required=True)
    w = sub.add_parser("tables")
    w.add_argument("--json", default=str(ROOT / "docs" / "ctc-lm-decode.json"))
    args = ap.parse_args()
    {"extract": cmd_extract, "tune": cmd_tune, "select": cmd_select, "ablate": cmd_ablate,
     "report": cmd_report, "a9": cmd_a9, "summary": cmd_summary, "tables": cmd_tables}[args.cmd](args)


if __name__ == "__main__":
    main()
