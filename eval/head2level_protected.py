#!/usr/bin/env python3
"""The two-level head's protected sets, once, after every tier's development verdict is pushed
(eval/head2level_protocol.json "protected"; docs/head2level.md). Report only: it decides and tunes nothing. A copy of
eval/headgate_protected.py with the two-level arm in place of the gated blob.

  head2level_protected.py tests --data EVAL_DATA --release DIR --arm TIER=BLOB ... --lm LM --out DIR
      vita-test and psp-test fixed lines (eval's hand-checked lines, not uncertain, box + pad as eval/sets.py cuts
      them; console polarity: grey, read with VOCR_INVERT where synth/common.line_polarity calls the text dark).
  head2level_protected.py clannad --work WORK --release DIR --arm TIER=BLOB ... --lm LM --out DIR
      CLANNAD's long run against owner-drive-ingest's frozen references (eval/clannad_refs.py): each page's ROI cut
      from its frame, a reference line a GRAD box of it, read as GRAD + refine inside that box (the refined lines
      from it, each with its dark flag, joined), as eval/textgate_protected.py does.

Each tier, today (the release blob) against its two-level arm ("gated" below: the --arm blob with two_level on, and
head_gate on when it holds a gate; for a tier without an adopted arm, the arm its lower tier's cascade reads), decodes
greedy, rubi and rubi_cascade (eval/headgate_eval.py) and the re-read variants: head-gate's counter-context re-read and
the two-level head's (a line whose reading holds a full-size カ or ケ before a counter character is read again with the
full head and no skipping; eval/head2level_dev.py REREAD_RE). Dialogue (no 'name' style / not a
name reference) primary, names a column; CER with paired line and VN-cluster bootstraps (tests) or page bootstraps
(CLANNAD, one title). Each command's attempt record (OUT/<cmd>.attempt.json) is created exclusively and refuses a
second run. Public outputs hold counts, rates and hashes; per-line records go to OUT/private/.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "eval", ROOT / "synth", ROOT / "runtime", ROOT / "train"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import headgate_eval as E  # noqa: E402
from metrics import lev, norm  # noqa: E402

TIERS = ("S", "N", "L")
REREAD_RE = __import__("re").compile("[カケ][月所年国条]")
ARM_BLOBS = {}
DECODES = ("greedy", "rubi", "rubi_cascade", "rubi_reread", "rubi_cascade_reread")


def once(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:            # exclusive: a second attempt fails here
        f.write(json.dumps(record, indent=1) + "\n")


def runtimes(a):
    from pyvocr import Vocr

    import headgate as HG
    rel = Path(a.release)
    out = {}
    for spec in a.arm:
        t, _, path = spec.partition("=")
        ARM_BLOBS[t] = path
    for t in TIERS:
        if t not in ARM_BLOBS:
            raise SystemExit(f"--arm {t}=BLOB is needed for every tier")
        gate = HG.read_gate(Path(ARM_BLOBS[t]).read_bytes()) is not None
        out[t] = {"today": Vocr(rel / f"VOCR5_{t}_w8.vocr", mode="w8i", max_width=1024),
                  "gated": Vocr(ARM_BLOBS[t], mode="w8i", max_width=1024, two_level=1,
                                head_gate="on" if gate else "off"),
                  "blob": {"today": E.sha256(rel / f"VOCR5_{t}_w8.vocr"), "gated": E.sha256(ARM_BLOBS[t])}}
    return out


def read_all(rts, crops):
    """{tier: [{"today": out, "gated": out, "ungated": out}]} for every (grey crop, invert)."""
    res = {}
    for t in TIERS:
        r = rts[t]
        res[t] = [{"today": r["today"].run_topk(g, E.K, invert=inv),
                   "gated": r["gated"].run_topk(g, E.K, invert=inv),
                   "ungated": r["gated"].run_topk(g, E.K, invert=inv, ungated=True)} for g, inv in crops]
    return res


def decode_all(res, decoders):
    """texts[(tier, arm, decode)] = [text per read], as eval/headgate_eval.py score composes them."""
    texts, confs = {}, {}
    for t in TIERS:
        for arm in E.ARMS:
            dec = decoders[t][arm]                                      # the arm's own vocab (permuted classes)
            g, r, cf, rr, cfr = [], [], [], [], []
            for x in res[t]:
                o = x[arm]
                g.append(o["text"])
                r.append(dec.rubi(o)[0])
                cf.append(E.line_conf(o))
                if arm == "gated":
                    tr, used = dec.rubi(o, ungated=x["ungated"])
                    if used is o and REREAD_RE.search(o["text"]):       # the two-level head's re-read
                        tr, used = dec.rubi(x["ungated"])
                    rr.append(tr)
                    cfr.append(E.line_conf(used))
                else:
                    rr.append(r[-1])
                    cfr.append(cf[-1])
            texts[(t, arm, "greedy")], texts[(t, arm, "rubi")], texts[(t, arm, "rubi_reread")] = g, r, rr
            confs[(t, arm)], confs[(t, arm, "reread")] = cf, cfr
    for t in TIERS:
        nxt = E.CASCADE[t]
        for arm in E.ARMS:
            for d, cd, ck in (("rubi", "rubi_cascade", (t, arm)), ("rubi_reread", "rubi_cascade_reread",
                                                                   (t, arm, "reread"))):
                base = texts[(t, arm, d)]
                if nxt is None:
                    texts[(t, arm, cd)] = list(base)
                else:
                    up, cf = texts[(nxt, arm, d)], confs[ck]
                    texts[(t, arm, cd)] = [u if c < E.CASCADE_CONF else b for b, u, c in zip(base, up, cf)]
    return texts


def cmd_tests(a):
    import sets

    from common import line_polarity, to_gray
    out = Path(a.out)
    once(out / "tests.attempt.json", {"started": E.utc_now(), "protocol_sha256": E.sha256(ROOT / "eval" /
                                                                                      "head2level_protocol.json")})
    rows, crops = [], []
    for name in ("vita-test", "psp-test"):
        lines, _ = sets.load_real(name, a.data)
        for ln in lines:
            g = np.ascontiguousarray(to_gray(ln.crop_rgb))
            rows.append({"set": name, "id": ln.id, "vn": ln.vn, "ref": norm(ln.text), "name": "name" in ln.styles})
            crops.append((g, bool(line_polarity(g))))
    score_tests(a, rows, crops, out)


def score_tests(a, rows, crops, out, sets_=("vita-test", "psp-test")):
    """The fixed-line read and its scores (rows: set, id, vn, ref (normalised), name; crops: (grey, invert))."""
    import finder_tighten as FT
    import vocr3_report as V
    out.mkdir(parents=True, exist_ok=True)
    rts = runtimes(a)
    res = read_all(rts, crops)
    decs = {t: {"today": E.Decoder(Path(a.release) / f"VOCR5_{t}_w8.vocr", a.lm),
                "gated": E.Decoder(ARM_BLOBS[t], a.lm)} for t in TIERS}
    texts = decode_all(res, decs)
    result = {"kind": "head2level-protected-tests", "created": E.utc_now(), "lines": len(rows),
              "inverted": sum(inv for _, inv in crops), "models": {t: rts[t]["blob"] for t in TIERS},
              "lm_sha256": E.sha256(a.lm), "bootstrap": {"reps": 10000, "seed": 0}, "tiers": {}}
    priv = {}
    for t in TIERS:
        tr = {"skip": {}, "decodes": {}}
        for scope, keep in (("dialogue", [not r["name"] for r in rows]), ("names", [r["name"] for r in rows])):
            tr["skip"][scope] = {"frames": int(sum(x["gated"]["frames"] for x, k in zip(res[t], keep) if k)),
                                 "skipped": int(sum(x["gated"]["gated"] for x, k in zip(res[t], keep) if k))}
        for d in DECODES:
            e = {arm: [lev(norm(p), r["ref"]) for p, r in zip(texts[(t, arm, d)], rows)] for arm in E.ARMS}
            priv[f"{t}/{d}"] = [{"set": r["set"], "id": r["id"], "today": e["today"][i], "gated": e["gated"][i],
                                 "n": len(r["ref"])} for i, r in enumerate(rows)]
            ent = {}
            for scope in ("dialogue", "names"):
                for part in (*sets_, "pooled"):
                    idx = [i for i, r in enumerate(rows) if (r["name"] == (scope == "names"))
                           and (part == "pooled" or r["set"] == part)]
                    ga = {f"{rows[i]['set']}:{rows[i]['id']}": (e["gated"][i], len(rows[i]["ref"])) for i in idx}
                    ta = {f"{rows[i]['set']}:{rows[i]['id']}": (e["today"][i], len(rows[i]["ref"])) for i in idx}
                    vn = {f"{rows[i]['set']}:{rows[i]['id']}": rows[i]["vn"] for i in idx}
                    n = sum(len(rows[i]["ref"]) for i in idx)
                    ent[f"{scope}/{part}"] = {
                        "lines": len(idx), "chars": n,
                        "cer_today": round(100 * sum(e["today"][i] for i in idx) / n, 4) if n else None,
                        "cer_gated": round(100 * sum(e["gated"][i] for i in idx) / n, 4) if n else None,
                        "lines_changed": sum(texts[(t, "today", d)][i] != texts[(t, "gated", d)][i] for i in idx),
                        "line_bootstrap": V.paired_bootstrap(ga, ta) if idx else None,
                        "vn_bootstrap": V.cluster_bootstrap(ga, ta, vn) if idx else None}
            tr["decodes"][d] = ent
            k = ent["dialogue/pooled"]
            print(t, d, k["cer_today"], "->", k["cer_gated"], k["line_bootstrap"]["delta"],
                  k["vn_bootstrap"].get("ci95"), flush=True)
        result["tiers"][t] = tr
    FT.assert_public(result)
    (out / "tests.json").write_text(json.dumps(result, indent=1) + "\n")
    (out / "private").mkdir(exist_ok=True)
    (out / "private" / "tests_lines.json").write_text(json.dumps(priv) + "\n")


def cmd_clannad(a):
    out, work = Path(a.out), Path(a.work)
    freeze = json.loads((work / "refs.freeze.json").read_text(encoding="utf-8"))
    if E.sha256(work / "refs.jsonl") != freeze["refs_sha256"]:
        raise SystemExit("refs.jsonl differs from its freeze record")
    once(out / "clannad.attempt.json", {"started": E.utc_now(), "refs_sha256": freeze["refs_sha256"],
                                        "protocol_sha256": E.sha256(ROOT / "eval" / "head2level_protocol.json")})
    score_clannad(a, work, freeze, out)


def score_clannad(a, work, freeze, out):
    """The CLANNAD read and its scores from a clannad_refs work directory (lines, refs, the session's frames)."""
    import clannad_refs as C
    import finder_tighten as FT
    import vocr3_report as V
    from PIL import Image

    import lines_refine as LR
    from common import to_gray
    out.mkdir(parents=True, exist_ok=True)
    rows, meta = C.load_lines(work)
    session = Path(meta["session"])
    refs = {x["id"]: x for x in C.read_jsonl(work / "refs.jsonl")}
    by_page = {}
    for r in rows:
        by_page.setdefault(r["page"], []).append(r)
    ident = FT.build(out / "build", "gcc")
    flags = LR.parse_flags("default")
    pages, imgs = [], []
    for page, rs in sorted(by_page.items()):
        src = rs[0]["source"]
        path = session / src["path"]
        if E.sha256(path) != src["sha256"]:
            raise SystemExit(f"{src['path']}: frame hash differs from the capture journal")
        with Image.open(path) as im:
            rgb = np.asarray(im.convert("RGB"))
        x, y, w, h = rs[0]["roi"]
        pages.append((page, rs))
        imgs.append(np.ascontiguousarray(to_gray(np.ascontiguousarray(rgb[y:y + h, x:x + w]))))
    grad = FT.lines_of(ident["driver"], imgs, 0)
    refd = FT.lines_of(ident["driver"], imgs, flags)
    units, crops = [], []           # units: (ref id, page, [read index ...])
    unmatched = 0
    for (page, rs), img, (gb, _, _, _, _), (rb, rdk, org, _, _) in zip(pages, imgs, grad, refd):
        for r in rs:
            if refs.get(r["id"], {}).get("status") != "ref":
                continue
            box = tuple(r["box"])
            if box not in gb:
                unmatched += 1
                continue
            k = gb.index(box)
            idx = []
            for (xx, yy, ww, hh), dd, o in zip(rb, rdk, org):
                if o == k:
                    idx.append(len(crops))
                    crops.append((np.ascontiguousarray(img[yy:yy + hh, xx:xx + ww]), bool(dd)))
            units.append((r["id"], page, idx))
    rts = runtimes(a)
    res = read_all(rts, crops)
    decs = {t: {"today": E.Decoder(Path(a.release) / f"VOCR5_{t}_w8.vocr", a.lm),
                "gated": E.Decoder(ARM_BLOBS[t], a.lm)} for t in TIERS}
    texts = decode_all(res, decs)
    result = {"kind": "head2level-protected-clannad", "created": E.utc_now(), "refs_sha256": freeze["refs_sha256"],
              "pages": len(pages), "scored_lines": len(units), "unmatched": unmatched, "reads": len(crops),
              "flags": flags, "models": {t: rts[t]["blob"] for t in TIERS}, "lm_sha256": E.sha256(a.lm),
              "bootstrap": {"reps": C.REPS, "seed": C.SEED, "unit": "page"}, "tiers": {}}
    priv = {}
    for t in TIERS:
        tr = {"skip": {"frames": int(sum(x["gated"]["frames"] for x in res[t])),
                       "skipped": int(sum(x["gated"]["gated"] for x in res[t]))}, "decodes": {}}
        for d in DECODES:
            pred = {arm: {uid: "".join(texts[(t, arm, d)][i] for i in idx) for uid, _, idx in units}
                    for arm in E.ARMS}
            ent = {}
            for scope in ("dialogue", "all"):
                sel = [(uid, pg) for uid, pg, _ in units if scope == "all" or not refs[uid].get("name")]
                pg = [p for _, p in sel]
                ids = sorted(set(pg))
                bidx = C.boot_idx(len(ids), C.REPS, C.SEED)
                sums = {arm: C.page_sums([C.line_unit(refs[uid]["text"], pred[arm][uid]) for uid, _ in sel], pg)
                        for arm in E.ARMS}
                cer = V.paired_bootstrap({p: (sums["gated"][p][0], sums["gated"][p][1]) for p in ids},
                                         {p: (sums["today"][p][0], sums["today"][p][1]) for p in ids},
                                         C.REPS, C.SEED)
                ent[scope] = {"today": C.summary(sums["today"], ids, bidx),
                              "gated": C.summary(sums["gated"], ids, bidx),
                              "gated_minus_today": {"cer_delta": cer["delta"], "cer_ci95": cer["ci95"],
                                                    "chars": cer["chars"]},
                              "lines_changed": sum(pred["today"][uid] != pred["gated"][uid] for uid, _ in sel)}
            priv[f"{t}/{d}"] = {uid: [pred["today"][uid] == pred["gated"][uid]] for uid, _, _ in units}
            tr["decodes"][d] = ent
            k = ent["dialogue"]
            print(t, d, k["today"]["cer"], "->", k["gated"]["cer"], k["gated_minus_today"], flush=True)
        result["tiers"][t] = tr
    FT.assert_public(result)
    (out / "clannad.json").write_text(json.dumps(result, indent=1) + "\n")
    (out / "private").mkdir(exist_ok=True)
    (out / "private" / "clannad_same.json").write_text(json.dumps(priv) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("tests", "clannad"):
        p = sub.add_parser(name)
        p.add_argument("--release", required=True)
        p.add_argument("--arm", action="append", required=True, help="TIER=BLOB, every tier of --tiers")
        p.add_argument("--tiers", default="S,N,L", help="the tiers with an adopted arm (and their cascades' tiers)")
        p.add_argument("--lm", required=True)
        p.add_argument("--out", required=True)
        if name == "tests":
            p.add_argument("--data", required=True)
        else:
            p.add_argument("--work", required=True)
    a = ap.parse_args()
    global TIERS
    TIERS = tuple(t for t in a.tiers.split(",") if t)
    for t in TIERS:
        if E.CASCADE[t] is not None and E.CASCADE[t] not in TIERS:
            raise SystemExit(f"{t}'s cascade reads {E.CASCADE[t]}: add it to --tiers")
    {"tests": cmd_tests, "clannad": cmd_clannad}[a.cmd](a)


if __name__ == "__main__":
    main()
