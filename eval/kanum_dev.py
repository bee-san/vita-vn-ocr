#!/usr/bin/env python3
"""Development check of the counter post-processing (issue #109, docs/counters.md): runtime/vocr_post.c's
text rule and counter-context re-decode on VOCR5-S/N/L's W8I readings of the fixed development sample, greedy
and with Rubi's beam + LM, every changed line listed.

    kanum_dev.py render --out DIR/synth.npz          the synthetic counter-context set (below; private renders)
    kanum_dev.py cache --out DIR --blob S=VOCR5_S_w8.vocr [--synth DIR/synth.npz]   (heavy: top-8 of every crop)
    kanum_dev.py score --cache DIR --blob S=... --lm vocr_lm_o3_1000k.vlm --out scores.json [--taus 500,...]
    kanum_dev.py summary --counts counts.json --grid S=scores2_S.json ... --final S=scores3_S.json ...
                         --widen S=widen3_S.json ... --out eval/counters_dev_20261021.json   (numbers only)

The cohorts are VOCR5's (eval/vocr5_protocol.json "cohorts"): teacher_val's and devnative's development files
in both views (rv: console polarity as stored, rvn: screen polarity) and the whole-dialogue cohorts tv_dlg and
dn_dlg, whose regions go through the C GRAD finder (flags 0) and the C refinement (VOCR_REFINE_DEFAULT), each
line read with its dark flag as VOCR_INVERT (eval/finder_tighten.py's path). Dialogue only, as VOCR5 scores;
name lines are a separate column. The cache holds per-frame top-K logits, no pixels; it stays private anyway.

Arms (per model): G greedy; L the beam + LM (Rubi's settings: beam 8, top-8, alpha 500, beta 3500, gate 8000,
</s>, bracket 8000); +R the text rule on the result; +A<tau> the re-decode's numeral alone (vocr_post_opts tau,
open = 0) and +B<open> its opening bracket alone (tau = 0, open), 1/1000 nat, on the top-K before the text and
the LM; with --final TAU,OPEN also +A<TAU>B<OPEN>, both.

The synthetic counter-context set (sy) is what the re-decode's threshold is chosen on (issue #109's
pre-registered rule): the development real lines hold no counter context at all. Every line of the development
splits of synth3's text pools (pools3/*.dev.txt, which no model trained on) with a numeral or 何 数 幾 before
ヶ ヵ ケ 箇 and 月 所 年 国 条 gives three texts (from the numeral on, up to 14 characters): as 「...」, bare, and as
a continuation line ...」 (its opening bracket on an earlier line: an inserted 「 would be wrong there),
each drawn once in one of nine open training fonts (white, a dark outline, on a noisy dark box, 24 or 26 px),
and every drawing is cropped seven times: its left edge 3 px before the first ink column (the finder's
padding), at it, and 2, 4, 6, 8 and 10 px into it, as a region drawn into the line cuts it. The reference is
always the whole text.
"""
import argparse
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "eval", ROOT / "synth", ROOT / "runtime", ROOT / "train"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from metrics import lev, norm  # noqa: E402

HERD = Path(os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr"))
ART = HERD / "shared" / "artifacts"
POOLS = ART / "owner-drive-ingest-2" / "reclaim" / "pools"
COHORTS = ART / "vocr5-train" / "cohorts"
TV = "teacher_val_dev.npz"
DN = ("devnative_tv_native_ink_dev.npz", "devnative_tv_native_grad_dev.npz", "devnative_strict_ink_dev.npz",
      "devnative_strict_grad_dev.npz")
WD = ("tv_dlg", "dn_dlg")
K = 8
LM_OPTS = {"beam": 8, "topk": 8, "alpha": 500, "beta": 3500, "gate": 8000, "flags": 2, "bracket": 8000}
TAUS = (500, 1000, 2000, 3000, 4000, 6000, 8000)       # the pre-registered grid (issue #109), 1/1000 nat
TEXT_POOLS = ART / "synth3" / "text" / "pools3"
FONTS = ART / "synth3" / "fonts"
SY_FONTS = ("ibmplexsansjp/IBMPlexSansJP-SemiBold.ttf", "ipaexfont/ipaexg.ttf", "kosugimaru/KosugiMaru-Regular.ttf",
            "koruri/Koruri-Bold.ttf", "vlgothic/VL-Gothic-Regular.ttf", "geneigothicn/GenEiGothicN-Bold.otf",
            "lineseedjp/LINESeedJP-Bold.ttf", "mplusrounded1c/MPLUSRounded1c-ExtraBold.ttf",
            "bizudpmincho/BIZUDPMincho-Bold.ttf")
SY_CUTS = (3, 0, -2, -4, -6, -8, -10)                  # px of the line's left before its first ink column
SY_RE = r"[〇一二三四五六七八九十百千万0-9０-９何数幾][ヶヵケ箇][月所年国条]"


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sy_texts():
    """[(source, text)]: two texts per development pool line holding a counter context (render's doc)."""
    import re
    pat = re.compile(SY_RE)
    out = []
    for f in sorted(TEXT_POOLS.glob("*.dev.txt")):
        for line in f.read_text(encoding="utf-8").splitlines():
            m = pat.search(line)
            if not m:
                continue
            core = line[m.start():m.start() + 14].strip("「」『』 　")
            if len(core) < 3:
                continue
            out.append((f.name, "「" + core + "」"))
            out.append((f.name, core))
            out.append((f.name, core + "」"))
    return out


def find_font(rel):
    p = FONTS / rel
    if not p.exists():
        hits = sorted(FONTS.glob(f"{rel.split('/')[0]}/**/{rel.split('/')[-1]}"))
        if not hits:
            raise SystemExit(f"font {rel} not found")
        p = hits[0]
    return p


def cmd_render(a):
    from PIL import Image, ImageDraw, ImageFilter, ImageFont
    texts = sy_texts()
    imgs, meta = [], []
    for i, (src, text) in enumerate(texts):
        rng = np.random.default_rng(1000 + i)
        rel = SY_FONTS[i % len(SY_FONTS)]
        size = int(rng.choice([24, 26]))
        ft = ImageFont.truetype(str(find_font(rel)), size)
        pad, top = 24, 16
        w, h = int(ft.getlength(text)) + 2 * pad, size + 2 * top
        bg = float(rng.uniform(60, 120))
        im = Image.fromarray(np.clip(rng.normal(bg, 6, (h, w)), 0, 255).astype(np.uint8))
        ImageDraw.Draw(im).text((pad, top), text, font=ft, fill=int(rng.uniform(235, 255)),
                                stroke_width=int(rng.choice([1, 2])), stroke_fill=int(rng.uniform(10, 50)))
        g = np.array(im.filter(ImageFilter.GaussianBlur(float(rng.uniform(0.3, 0.7)))))
        ys, xs = np.where(np.abs(g.astype(int) - int(round(bg))) > 40)
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        for cut in SY_CUTS:
            left = x0 - cut
            crop = g[max(0, y0 - 3):y1 + 4, max(0, left):x1 + 4]
            imgs.append(np.ascontiguousarray(crop))
            meta.append({"id": f"sy{i:04d}c{cut}", "line": i, "source": src, "text": text, "font": rel,
                         "size": size, "cut": cut})
    offs = np.cumsum([0] + [x.size for x in imgs])
    np.savez_compressed(a.out, data=np.concatenate([x.reshape(-1) for x in imgs]), offsets=offs,
                        shapes=np.array([x.shape for x in imgs], np.int32),
                        meta=json.dumps(meta, ensure_ascii=False))
    print(f"{len(texts)} texts, {len(imgs)} crops -> {a.out}")


def sy_rows(path):
    z = np.load(path)
    data, offs, shapes = z["data"], z["offsets"], z["shapes"]      # each z[...] decompresses the array again
    meta = json.loads(str(z["meta"]))
    rows = []
    for i, m in enumerate(meta):
        h, w = (int(v) for v in shapes[i])
        rows.append(dict(m, img=data[offs[i]:offs[i + 1]].reshape(h, w).copy(), ref=norm(m["text"])))
    return rows


def fixed_lines():
    """[(cohort, file, row)] of the fixed-line files: rows with id, vn, ref, img, inverted, name."""
    import synth3_ablation as S
    out = []
    for f in (TV, *DN):
        rows = S.load_dev(POOLS / f)
        if f != TV:
            z = np.load(POOLS / f)
            names = {m["id"]: "name" in (m.get("styles") or []) for m in json.loads(str(z["meta"]))}
        for r in rows:
            r["name"] = names[r["id"]] if f != TV else "namelike" in r["tags"]
            out.append(("tv" if f == TV else "dn", f, r))
    return out


def wd_lines(driver, flags):
    """{cohort: (cases, {mode: [(boxes, dark, org, rows, found)]})}, C finder == twin on every case."""
    import finder_tighten as FT
    res = {}
    for name in WD:
        cases = FT.load_cohort(COHORTS, name)
        imgs = [c["image"] for c in cases]
        res[name] = (cases, {"grad": FT.lines_of(driver, imgs, 0), "refine": FT.lines_of(driver, imgs, flags)})
    return res


def cmd_cache(a):
    import finder_tighten as FT
    from pyvocr import Vocr

    import lines_refine as LR
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ident = FT.build(out / "build", "gcc")
    flags = LR.parse_flags("default")
    lines = [] if a.synth_only else fixed_lines()
    wd = {} if a.synth_only else wd_lines(ident["driver"], flags)
    for spec in a.blob:
        name, _, path = spec.partition("=")
        rt = Vocr(path, mode="w8i", max_width=1024)
        items = []

        def read(img, invert, key):
            o = rt.run_topk(np.ascontiguousarray(img), K, invert=bool(invert))
            items.append({"key": key, "text": o["text"], "topv": o["topv"], "topc": o["topc"], "blank": o["blank"]})

        for coh, f, r in lines:
            g = r["img"]
            read(g, False, (coh, f, r["id"], "rv"))
            read(255 - g if r["inverted"] else g, False, (coh, f, r["id"], "rvn"))
        for r in (sy_rows(a.synth) if a.synth else []):
            read(r["img"], False, ("sy", "synth", r["id"], "rv"))
        for coh, (cases, modes) in wd.items():
            for mode, per in modes.items():
                for ci, (c, (boxes, dark, _, _, _)) in enumerate(zip(cases, per)):
                    for li, ((x, y, w, h), d) in enumerate(zip(boxes, dark)):
                        read(c["image"][y:y + h, x:x + w], d, (coh, mode, ci, li))
        meta = {"blob": Path(path).name, "sha256": sha256(path), "items": len(items), "topk": K}
        with open(out / f"cache_{name}{'_sy' if a.synth_only else ''}.pkl", "wb") as fh:
            pickle.dump({"meta": meta, "items": items}, fh, protocol=4)
        print(f"{name}: {len(items)} crops cached", flush=True)
    (out / f"meta{'_sy' if a.synth_only else ''}.json").write_text(json.dumps({"native": ident, "flags": flags,
                                               "pools": {f: sha256(POOLS / f) for f in (TV, *DN)},
                                               "cohorts": {c: sha256(COHORTS / f"{c}.pkl") for c in WD}},
                                              indent=1) + "\n")


class Arms:
    """The texts of one cached crop under every arm."""

    def __init__(self, rt, lm, taus, final=None):
        import vocr_post as P
        from pyvocr import post_counters
        self.rt, self.lm, self.taus, self.P, self.final = rt, lm, taus, P, final
        self.rule = lambda t: post_counters(t, rt.lib)[0]

    def texts(self, it):
        out = {}
        g = it["text"]
        lm_text = self.lm.decode(it["topv"], it["topc"], it["blank"]) if g else ""
        out["G"], out["L"] = g, lm_text
        out["G+R"], out["L+R"] = self.rule(g), self.rule(lm_text)
        stats = {}
        runs = [(f"A{t}", t, 0) for t in self.taus] + [(f"B{t}", 0, t) for t in self.taus]
        if self.final:
            runs.append((f"A{self.final[0]}B{self.final[1]}", *self.final))
        for tag, tau, opn in runs:
            res, st = self.rt.redecode(it, tau, opn)
            gt = res["text"]
            lt = (self.lm.decode(res["topv"], res["topc"], res["blank"]) if gt else "") if st["edits"] else lm_text
            out[f"G+{tag}+R"], out[f"L+{tag}+R"] = self.rule(gt), self.rule(lt)
            stats[tag] = st
        return out, stats


def cmd_score(a):
    import finder_tighten as FT
    from pyvocr import Vocr, VocrLM

    from general_vn_textbox_guard import transcript_score
    cache = Path(a.cache)
    taus = [int(t) for t in a.taus.split(",")] if a.taus else list(TAUS)
    lines = fixed_lines()
    ref = {(coh, f, r["id"]): r for coh, f, r in lines}
    if a.synth:
        for r in sy_rows(a.synth):
            ref[("sy", "synth", r["id"])] = dict(r, name=False)
    wdc = {name: FT.load_cohort(COHORTS, name) for name in WD}
    report = {"kind": "kanum-fix-dev", "lm": {"file": Path(a.lm).name, "sha256": sha256(a.lm), **LM_OPTS},
              "taus": taus, "models": {}}
    blobs = dict(s.partition("=")[::2] for s in a.blob)
    for name in sorted(blobs):
        c = {"items": []}
        for suffix in ("", "_sy"):
            f = cache / f"cache_{name}{suffix}.pkl"
            if not f.exists():
                continue
            with open(f, "rb") as fh:
                part = pickle.load(fh)
            if part["meta"]["sha256"] != sha256(blobs[name]):
                raise SystemExit(f"{f}: the cache was made with another blob")
            c["meta"] = part["meta"]
            c["items"] += part["items"]
        rt = Vocr(blobs[name], mode="w8i", max_width=1024)
        lm = VocrLM(rt, a.lm, **LM_OPTS)
        final = [int(t) for t in a.final.split(",")] if a.final else None
        arms = Arms(rt, lm, taus, final)
        fixed = {}            # arm -> [(key, errors, chars, name, pred)]
        wd_pred = {}          # (coh, mode, ci) -> {arm: [line texts]}
        triggers = {}
        for it in c["items"]:
            texts, stats = arms.texts(it)
            for t, st in stats.items():
                tt = triggers.setdefault(t, {"contexts": 0, "numerals": 0, "brackets": 0})
                for k in tt:
                    tt[k] += st[k]
            key = it["key"]
            if key[0] in ("tv", "dn", "sy"):
                r = ref[key[:3]]
                for arm, t in texts.items():
                    p = norm(t)
                    fixed.setdefault(arm, []).append((key, lev(p, r["ref"]), len(r["ref"]), r["name"], p))
            else:
                coh, mode, ci, li = key
                d = wd_pred.setdefault((coh, mode, ci), {})
                for arm, t in texts.items():
                    d.setdefault(arm, []).append(norm(t))
        res = {"blob": c["meta"], "triggers": triggers, "arms": {}, "changed": {}}
        tv_shots = {u["shot"] for u in wdc["tv_dlg"]}
        arm_names = list(fixed)
        for arm in arm_names:
            comp = {}
            for coh in ("tv", "dn", "sy"):
                sel = [x for x in fixed[arm] if x[0][0] == coh]
                for part, keep in (("dialogue", lambda x: not x[3]), ("name", lambda x: x[3])):
                    e = sum(x[1] for x in sel if keep(x))
                    n = sum(x[2] for x in sel if keep(x))
                    comp[f"{coh}_{part}"] = [e, n]
            e = n = 0
            for coh in WD:
                for mode in ("grad", "refine"):
                    for ci, case in enumerate(wdc[coh]):
                        if coh == "dn_dlg" and case["shot"] in tv_shots:
                            continue
                        preds = wd_pred.get((coh, mode, ci), {}).get(arm, [])
                        sc = transcript_score(case["lines"], preds)
                        e += sc["all"]["full"]["errors"]
                        n += sc["all"]["full"]["chars"]
            comp["wd"] = [e, n]
            for cut in SY_CUTS:
                sel = [x for x in fixed[arm] if x[0][0] == "sy" and ref[x[0][:3]]["cut"] == cut]
                comp[f"sy_cut{cut}"] = [sum(x[1] for x in sel), sum(x[2] for x in sel)]
            cer = {k: round(100 * v[0] / v[1], 4) if v[1] else None for k, v in comp.items()}
            score = 0.5 * (cer["tv_dialogue"] + cer["dn_dialogue"]) / 2 + 0.5 * cer["wd"]
            res["arms"][arm] = {"errors_chars": comp, "cer": cer, "dev_score": round(score, 4)}
        # every line an arm changes, against its base (G or L)
        for arm in arm_names:
            base = "G" if arm.startswith("G") else "L"
            if arm == base:
                continue
            ch = []
            for (k0, e0, n0, nm, p0), (k1, e1, n1, _, p1) in zip(fixed[base], fixed[arm]):
                if p0 != p1:
                    ch.append({"line": "/".join(map(str, k0)), "ref": ref[k0[:3]]["ref"], "before": p0, "after": p1,
                               "errors": [e0, e1], "name": nm})
            for (coh, mode, ci), d in sorted(wd_pred.items()):
                if d.get(base) != d.get(arm):
                    case = wdc[coh][ci]
                    s0 = transcript_score(case["lines"], d.get(base, []))["all"]["full"]["errors"]
                    s1 = transcript_score(case["lines"], d.get(arm, []))["all"]["full"]["errors"]
                    ch.append({"line": f"{coh}/{mode}/{case['unit']}|{case['variant']}",
                               "ref": " / ".join(norm(x["text"]) for x in case["lines"]),
                               "before": " / ".join(d.get(base, [])), "after": " / ".join(d.get(arm, [])),
                               "errors": [s0, s1]})
            res["changed"][arm] = ch
        res["effect"] = {}
        for arm in arm_names:
            if "+A" not in arm and "+B" not in arm:
                continue
            base = arm[0] + "+R"
            eff = {}
            for coh in ("tv", "dn", "sy"):
                b = [x for x in fixed[base] if x[0][0] == coh]
                c2 = [x for x in fixed[arm] if x[0][0] == coh]
                eff[coh] = {"better": sum(y[1] < x[1] for x, y in zip(b, c2)),
                            "worse": sum(y[1] > x[1] for x, y in zip(b, c2)),
                            "changed": sum(y[4] != x[4] for x, y in zip(b, c2))}
            better = worse = 0
            for (coh, mode, ci), d in wd_pred.items():
                if d.get(base) != d.get(arm):
                    case = wdc[coh][ci]
                    s0 = transcript_score(case["lines"], d.get(base, []))["all"]["full"]["errors"]
                    s1 = transcript_score(case["lines"], d.get(arm, []))["all"]["full"]["errors"]
                    better += s1 < s0
                    worse += s1 > s0
            eff["wd"] = {"better": better, "worse": worse}
            res["effect"][arm] = eff
        report["models"][name] = res
        b = res["arms"]
        print(name, {k: (v["dev_score"], v["cer"]["wd"]) for k, v in b.items() if k in ("G", "L", "G+R", "L+R")},
              {arm: len(v) for arm, v in res["changed"].items() if v}, flush=True)
    Path(a.out).write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def select(grid, kind):
    """The pre-registered rule (issue #109): the largest threshold of the grid with no line worse on any cohort,
    for every tier and both decodings; None if even the smallest makes one worse."""
    ok = []
    for t in grid["taus"]:
        if all(v["worse"] == 0 for m in grid["models"].values() for dec in ("G", "L")
               for v in m[kind][str(t)][dec].values()):
            ok.append(t)
        else:
            break
    return ok[-1] if ok else None


def widen_adopted(widen):
    """Issue #109's rule for the edge continuation: every tier's VN-cluster interval of the standard regions' CER
    change not entirely above 0, and the mean change over the tiers <= 0."""
    d = [m["standard"]["compare"]["delta"] if "compare" in m["standard"] else 0.0 for m in widen.values()]
    lo = [m["standard"]["compare"]["vn_ci95"][0] if "compare" in m["standard"] else 0.0 for m in widen.values()]
    return all(x <= 0 for x in lo) and sum(d) / len(d) <= 0


def cmd_summary(a):
    def pairs(specs):
        return {k: json.loads(Path(v).read_text(encoding="utf-8")) for k, v in (s.partition("=")[::2] for s in specs)}
    counts = json.loads(Path(a.counts).read_text(encoding="utf-8"))
    tab, chars = {}, {}
    numl = ("numk", "numd", "quant")
    for src in counts["sources"]:
        g = src["source"].split(":")[0]
        chars[g] = chars.get(g, 0) + src["chars"]
        for lc, mid, r, n in src["counts"]:
            if r not in "月所年国条村町日" or len(r) != 1:
                continue
            t = tab.setdefault(r, {"tsu_any": 0, "tsu_numeral": 0, "counter_any": 0, "counter_numeral": 0})
            if mid == "ッ":
                t["tsu_any"] += n
                t["tsu_numeral"] += n if lc in numl else 0
            elif mid in "ヶヵケ箇":
                t["counter_any"] += n
                t["counter_numeral"] += n if lc in numl else 0
    grid = {"taus": None, "models": {}}
    for m, rep in pairs(a.grid).items():
        r = rep["models"][m]
        grid["taus"] = rep["taus"]
        g = {"A": {}, "B": {}}
        for kind in "AB":
            for t in rep["taus"]:
                g[kind][str(t)] = {dec: r["effect"][f"{dec}+{kind}{t}+R"] for dec in "GL"}
        g["rule_changes"] = {dec: len(r["changed"].get(f"{dec}+R", [])) for dec in "GL"}
        grid["models"][m] = g
    final = {}
    for m, rep in pairs(a.final).items():
        r = rep["models"][m]
        final[m] = {arm: {k: r["arms"][arm][k] for k in ("dev_score", "cer", "errors_chars")} |
                    ({"effect": r["effect"][arm]} if arm in r["effect"] else {})
                    for arm in r["arms"] if arm in ("G", "L", "G+R", "L+R") or
                    (arm.split("+")[1].startswith("A") and "B" in arm.split("+")[1])}
        final[m]["triggers"] = r["triggers"]
    widen = {}
    for m, rep in pairs(a.widen).items():
        r = rep["models"][m]
        widen[m] = {k: {kk: vv for kk, vv in v.items()} for k, v in r["sets"].items()} | {"counts": r["counts"]}
    out = {"kind": "counters-dev", "issue": 109, "lm": next(iter(pairs(a.final).values()))["lm"],
           "corpus": {"chars": chars, "table": tab},
           "grid": grid, "selection": {"tau": select(grid, "A"), "open": select(grid, "B")},
           "final": final, "widen": widen, "widen_adopted": widen_adopted(widen)}
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(out["selection"], "widen adopted:", out["widen_adopted"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("cache")
    p.add_argument("--out", required=True)
    p.add_argument("--blob", action="append", required=True)
    p.add_argument("--synth")
    p.add_argument("--synth-only", action="store_true")
    p = sub.add_parser("render")
    p.add_argument("--out", required=True)
    p = sub.add_parser("score")
    p.add_argument("--cache", required=True)
    p.add_argument("--blob", action="append", required=True)
    p.add_argument("--lm", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--taus")
    p.add_argument("--synth")
    p.add_argument("--final")
    p = sub.add_parser("summary")
    p.add_argument("--counts", required=True)
    p.add_argument("--grid", action="append", required=True)
    p.add_argument("--final", action="append", required=True)
    p.add_argument("--widen", action="append", required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"cache": cmd_cache, "score": cmd_score, "render": cmd_render, "summary": cmd_summary}[a.cmd](a)


if __name__ == "__main__":
    main()
