#!/usr/bin/env python3
"""textdet: the protected sets, once, after the selection is committed (eval/textdet_protocol.json "protected").

  textdet_protected.py tests --det NAME=BLOB --model NAME=BLOB ... --data EVAL --out DIR [--cbin DRIVER]
      vita-test + psp-test: eval's dialogue regions (a region holding an uncertain line is left out whole) at the
      base box and grown by 4 and 8 px. GRAD (runtime/vocr_lines.c through its C driver, twin-checked), GRAD + refine
      (VOCR_REFINE_DEFAULT) and the detector (its twin, C-checked with --cbin); every horizontal line each finder
      returns is read by each recognizer (the C runtime, W8I), joined in the finder's order and scored by
      transcript_score (body = dialogue text and all text). Detection against eval's line boxes too (line recall /
      precision / F1 at IoU >= 0.5, region exact). Paired bootstraps over regions and VN clusters.
  textdet_protected.py clannad --det NAME=BLOB --model NAME=BLOB ... --work CLANNAD_WORK --out DIR [--cbin DRIVER]
      CLANNAD's long run against owner-drive-ingest's frozen references (eval/clannad_refs.py): each page's ROI cut
      from its native frame; a reference line is a GRAD box of that ROI (the C finder reproduces every one); GRAD
      reads that box, refine the refined lines from it, the detector every horizontal line whose box lies mostly
      inside it (at least half its area; a line goes to the GRAD box it overlaps most), joined in the detector's
      order. clannad_refs' scoring (metrics.norm, Levenshtein), whole-page bootstraps; dialogue (no speaker-name
      line) and all lines. Also counted: the detector's lines on pages and boxes the references hold no text for.

Each command runs once: its attempt record (OUT/*.attempt.json) is created exclusively and refuses a second run.
Public outputs hold counts, rates and hashes only; per-unit records go to OUT/private/.
"""
import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "eval", ROOT / "synth", ROOT / "runtime", ROOT / "train"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

PROTOCOL = ROOT / "eval" / "textdet_protocol.json"
VARIANTS = {"base": 0, "expanded-4": 4, "expanded-8": 8}
FLAGS = 0x1F


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def once(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:            # exclusive: a second attempt fails here
        f.write(json.dumps(record, indent=1) + "\n")


def parse(spec):
    name, _, path = spec.partition("=")
    if not name or not path:
        raise SystemExit(f"NAME=PATH, not {spec!r}")
    return name, Path(path)


def det_lines(model, images, cbin=None, blob=None, refine=False):
    """The detector's lines on every grey image (twin; with refine, round 2's det+refine), checked against the C
    driver when given."""
    import textdet as TD

    import textdet_twin as TW
    out = []
    for g in images:
        lines, tr = TW.find_lines(model, g)
        out.append((TD.det_refine(lines, g) if refine else lines, tr))
    equal = None
    if cbin:
        c = TD.c_detect(cbin, blob, images, 2 if refine else 0)
        equal = all([list(map(int, ln)) for ln in cl] ==
                    [list(map(int, ln)) for ln in tl if not (refine and ln[5])] and ct == tt
                    for (tl, tt), (cl, ct) in zip(out, c))
        if not equal:
            raise SystemExit("the C detector differs from its twin on a protected region")
    return out, equal


def cmd_tests(args):
    import finder_tighten as FT
    import sets
    import textdet as TD
    import vocr3_report as V

    import textdet_twin as TW
    from common import to_gray
    from general_vn_textbox_guard import transcript_score
    out = Path(args.out)
    dname, dblob = parse(args.det)
    once(out / "tests.attempt.json", {"started": utc_now(), "protocol_sha256": sha256(PROTOCOL), "det": dname,
                                      "finder": "det+refine" if args.refine else "det",
                                      "det_sha256": sha256(dblob),
                                      "models": {n: sha256(p) for n, p in map(parse, args.model)}})
    model = TW.Model(dblob.read_bytes())
    ident = FT.build(out / "build", args.compiler)
    readers = {n: FT.Reader(p, ident["library"]) for n, p in map(parse, args.model)}
    M = TD._linefind("metrics")
    units = {n: {m: [] for m in ("grad", "refine", "det")} for n in readers}
    detection = {m: [] for m in ("grad", "refine", "det")}
    c_equal = []
    for set_name in ("vita-test", "psp-test"):
        regions, excluded = V.dialogue_regions(set_name, args.data)
        boxes = {}
        for path, region, rows in sets.regions(set_name, args.data):
            for r in rows:
                boxes[r["id"]] = r["box"]
        cache = sets.ImageCache()
        cases = []
        for reg in regions:
            img = cache.get(reg["path"])
            dims = [img.shape[1], img.shape[0]]
            for variant, pad in VARIANTS.items():
                x, y, w, h = V.expand(reg["region"], pad, dims)
                cases.append((reg, variant, (x, y), np.ascontiguousarray(to_gray(np.ascontiguousarray(
                    img[y:y + h, x:x + w])))))
        imgs = [c[3] for c in cases]
        found = {"grad": FT.lines_of(ident["driver"], imgs, 0), "refine": FT.lines_of(ident["driver"], imgs, FLAGS)}
        dl, eq = det_lines(model, imgs, args.cbin, dblob, args.refine)
        c_equal.append(eq)
        for (reg, variant, (ox, oy), img), g, r, (dlines, _) in zip(cases, found["grad"], found["refine"], dl):
            per = {"grad": (g[0], g[1], [0] * len(g[0])), "refine": (r[0], r[1], [0] * len(r[0])),
                   "det": ([ln[:4] for ln in dlines], [ln[4] for ln in dlines], [ln[5] for ln in dlines])}
            gt = [(b[0] - ox, b[1] - oy, b[0] - ox + b[2], b[1] - oy + b[3]) for b in
                  (boxes[ln["id"]] for ln in reg["lines"])]
            for mode, (bx, dk, vt) in per.items():
                mt = M.match([(x, y, x + w, y + h) for x, y, w, h in bx], gt, ())
                detection[mode].append({"unit": f"{set_name}|{reg['region_id']}|{variant}", "set": set_name,
                                        "vn": reg["vn"], "variant": variant,
                                        **{k: mt[k] for k in ("tp", "fp", "fn", "merged", "split", "cut")}})
                hz = [(b, d) for b, d, v in zip(bx, dk, vt) if not v]
                for name, rd in readers.items():
                    preds = [rd.read(img[y:y + h, x:x + w], d) for (x, y, w, h), d in hz]
                    sc = transcript_score(reg["lines"], preds)
                    units[name][mode].append({
                        "unit": f"{set_name}|{reg['region_id']}|{variant}", "set": set_name, "vn": reg["vn"],
                        "platform": reg["platform"], "variant": variant, "detections": len(hz),
                        **{f"{a}_{t}_{k}": sc[a][t][k] for a in ("all", "body") for t in ("full", "hira")
                           for k in ("errors", "chars")}})
        print(f"{set_name}: {len(regions)} regions, {len(cases)} cases (excluded {excluded})", flush=True)
    result = {"kind": "textdet-protected-tests", "det": dname, "det_sha256": sha256(dblob),
              "finder": "det+refine" if args.refine else "det",
              "det_bytes": model.blob_bytes, "c_equals_twin": c_equal, "native": ident,
              "models": {n: {"sha256": r.sha256, "blob_bytes": r.blob_bytes} for n, r in readers.items()},
              "bootstrap": FT.BOOT, "dialogue": {}, "detection": {}}
    for name in readers:
        e = {}
        for scope in ("body", "all"):
            for v in list(VARIANTS) + ["all"]:
                for plat in ("vita", "psp", "pooled"):
                    def keep(u, v=v, plat=plat):
                        return (v == "all" or u["variant"] == v) and (plat == "pooled" or u["platform"] == plat)

                    sel = {m: [dict(u, errors=u[f"{scope}_full_errors"], chars=u[f"{scope}_full_chars"])
                               for u in units[name][m]] for m in units[name]}
                    t = {m: FT.totals(sel[m], keep) for m in sel}
                    e[f"{scope}/{v}/{plat}"] = {
                        **{m: {"errors": t[m][0], "chars": t[m][1], "cer": FT.pct(t[m])} for m in t},
                        "det_minus_grad": FT.compare(sel["det"], sel["grad"], keep),
                        "det_minus_refine": FT.compare(sel["det"], sel["refine"], keep)}
        result["dialogue"][name] = e
    for mode, rows in detection.items():
        for plat in ("vita-test", "psp-test", "pooled"):
            sel = [r for r in rows if plat == "pooled" or r["set"] == plat]
            tp, fp, fn = (sum(r[k] for r in sel) for k in ("tp", "fp", "fn"))
            p_ = tp / (tp + fp) if tp + fp else 0.0
            r_ = tp / (tp + fn) if tp + fn else 0.0
            result["detection"][f"{mode}/{plat}"] = {
                "cases": len(sel), "tp": tp, "fp": fp, "fn": fn, "line_recall": round(r_, 4),
                "line_precision": round(p_, 4), "line_f1": round(2 * p_ * r_ / (p_ + r_), 4) if p_ + r_ else 0.0,
                "region_exact": round(sum(r["fp"] == 0 and r["fn"] == 0 for r in sel) / len(sel), 4),
                **{k: sum(r[k] for r in sel) for k in ("merged", "split", "cut")}}
    FT.assert_public(result)
    (out / "tests.json").write_text(json.dumps(result, indent=1) + "\n")
    (out / "private").mkdir(exist_ok=True)
    (out / "private" / "tests_units.json").write_text(json.dumps({"units": units, "detection": detection}) + "\n")
    for name in readers:
        k = result["dialogue"][name]["body/all/pooled"]
        print(name, {m: k[m]["cer"] for m in ("grad", "refine", "det")}, "det-grad", k["det_minus_grad"]["delta"],
              k["det_minus_grad"]["vn_ci95"], "det-refine", k["det_minus_refine"]["delta"],
              k["det_minus_refine"]["vn_ci95"], flush=True)


def assign(det_boxes, grad_boxes):
    """For each detector box, the index of the GRAD box it lies mostly inside (largest overlap, at least half its
    area), or None."""
    out = []
    for x, y, w, h in det_boxes:
        best, area = None, 0
        for k, (gx, gy, gw, gh) in enumerate(grad_boxes):
            ix = max(0, min(x + w, gx + gw) - max(x, gx))
            iy = max(0, min(y + h, gy + gh) - max(y, gy))
            if ix * iy > area:
                best, area = k, ix * iy
        out.append(best if best is not None and 2 * area >= w * h else None)
    return out


def cmd_clannad(args):
    import clannad_refs as C
    import finder_tighten as FT
    import vocr3_report as V
    from PIL import Image

    import textdet_twin as TW
    from common import to_gray
    out = Path(args.out)
    work = Path(args.work)
    dname, dblob = parse(args.det)
    freeze = json.loads((work / "refs.freeze.json").read_text(encoding="utf-8"))
    if sha256(work / "refs.jsonl") != freeze["refs_sha256"]:
        raise SystemExit("refs.jsonl differs from its freeze record")
    once(out / "clannad.attempt.json", {"started": utc_now(), "refs_sha256": freeze["refs_sha256"],
                                        "finder": "det+refine" if args.refine else "det",
                                        "protocol_sha256": sha256(PROTOCOL), "det": dname,
                                        "det_sha256": sha256(dblob)})
    model = TW.Model(dblob.read_bytes())
    rows, meta = C.load_lines(work)
    session = Path(meta["session"])
    refs = {x["id"]: x for x in C.read_jsonl(work / "refs.jsonl")}
    by_page = {}
    for r in rows:
        by_page.setdefault(r["page"], []).append(r)
    ident = FT.build(out / "build", args.compiler)
    readers = {n: FT.Reader(p, ident["library"]) for n, p in map(parse, args.model)}
    pages, imgs = [], []
    for page, rs in sorted(by_page.items()):
        src = rs[0]["source"]
        path = session / src["path"]
        if sha256(path) != src["sha256"]:
            raise SystemExit(f"{src['path']}: frame hash differs from the capture journal")
        with Image.open(path) as im:
            rgb = np.asarray(im.convert("RGB"))
        x, y, w, h = rs[0]["roi"]
        pages.append((page, rs))
        imgs.append(np.ascontiguousarray(to_gray(np.ascontiguousarray(rgb[y:y + h, x:x + w]))))
    grad = FT.lines_of(ident["driver"], imgs, 0)
    refd = FT.lines_of(ident["driver"], imgs, FLAGS)
    dl, c_equal = det_lines(model, imgs, args.cbin, dblob, args.refine)
    preds = {n: {"grad": {}, "refine": {}, "det": {}} for n in readers}
    extra = {"det_lines_on_no_text_pages": 0, "det_lines_outside_ref_boxes": 0, "no_text_pages": 0,
             "pages_with_ref": 0, "det_lines": 0, "det_vertical": 0, "grad_lines_on_no_text_pages": 0}
    matched = unmatched = 0
    for (page, rs), img, (gb, gd, _, _, _), (rb, rdk, org, _, _), (dlines, _) in zip(pages, imgs, grad, refd, dl):
        hz = [ln for ln in dlines if not ln[5]]
        extra["det_lines"] += len(dlines)
        extra["det_vertical"] += len(dlines) - len(hz)
        ref_rows = [r for r in rs if refs.get(r["id"], {}).get("status") == "ref"]
        if not ref_rows:
            extra["no_text_pages"] += 1
            extra["det_lines_on_no_text_pages"] += len(dlines)
            extra["grad_lines_on_no_text_pages"] += len(gb)
            continue
        extra["pages_with_ref"] += 1
        owner = assign([ln[:4] for ln in hz], gb)
        ref_boxes = set()
        for r in ref_rows:
            box = tuple(r["box"])
            if box not in gb:
                unmatched += 1
                continue
            matched += 1
            k = gb.index(box)
            ref_boxes.add(k)
            for n, rd in readers.items():
                x, y, w, h = box
                preds[n]["grad"][r["id"]] = rd.read(img[y:y + h, x:x + w], gd[k])
                preds[n]["refine"][r["id"]] = "".join(rd.read(img[yy:yy + hh, xx:xx + ww], dd)
                                                      for (xx, yy, ww, hh), dd, o in zip(rb, rdk, org) if o == k)
                preds[n]["det"][r["id"]] = "".join(rd.read(img[ln[1]:ln[1] + ln[3], ln[0]:ln[0] + ln[2]], ln[4])
                                                   for ln, o in zip(hz, owner) if o == k)
        extra["det_lines_outside_ref_boxes"] += sum(o not in ref_boxes for o in owner)
    first = preds[next(iter(readers))]["grad"]
    scored = [x for x in refs.values() if x["status"] == "ref" and x["id"] in first]
    result = {"kind": "textdet-clannad", "det": dname, "det_sha256": sha256(dblob),
              "finder": "det+refine" if args.refine else "det",
              "refs_sha256": freeze["refs_sha256"],
              "scored_lines": len(scored), "matched_grad_boxes": matched, "unmatched": unmatched,
              "c_equals_twin": c_equal, "native": ident, "counts": extra,
              "models": {n: {"sha256": r.sha256} for n, r in readers.items()}, "scopes": {}}
    for scope in ("dialogue", "all"):
        sel = [x for x in scored if scope == "all" or not x.get("name")]
        pg = [x["page"] for x in sel]
        ids = sorted(set(pg))
        idx = C.boot_idx(len(ids), C.REPS, C.SEED)
        e = {"lines": len(sel), "pages": len(ids), "systems": {}, "pairs": {}}
        for n in readers:
            sums = {}
            for mode in ("grad", "refine", "det"):
                u = [C.line_unit(x["text"], preds[n][mode][x["id"]]) for x in sel]
                sums[mode] = C.page_sums(u, pg)
                e["systems"][f"{n}/{mode}"] = C.summary(sums[mode], ids, idx)
            for base in ("grad", "refine"):
                a = {p: (sums["det"][p][0], sums["det"][p][1]) for p in ids}
                b = {p: (sums[base][p][0], sums[base][p][1]) for p in ids}
                cer = V.paired_bootstrap(a, b, C.REPS, C.SEED)
                e["pairs"][f"{n}/det:{n}/{base}"] = {"cer_delta": cer["delta"], "cer_ci95": cer["ci95"],
                                                     "candidate_errors": cer["candidate_errors"],
                                                     "baseline_errors": cer["baseline_errors"], "chars": cer["chars"]}
        result["scopes"][scope] = e
    FT.assert_public(result)
    (out / "clannad.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: (v["cer"], v["cer_ci95"]) for k, v in result["scopes"]["dialogue"]["systems"].items()},
                     indent=1))
    print(json.dumps(result["scopes"]["dialogue"]["pairs"], indent=1))
    print(json.dumps(extra))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("tests", "clannad"):
        p = sub.add_parser(name)
        p.add_argument("--det", required=True)
        p.add_argument("--model", action="append", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--cbin", default="")
        p.add_argument("--compiler", default="gcc")
        p.add_argument("--refine", action="store_true", help="round 2's det+refine (protocol amendment 3)")
        if name == "tests":
            p.add_argument("--data", required=True)
        else:
            p.add_argument("--work", required=True)
    args = ap.parse_args()
    {"tests": cmd_tests, "clannad": cmd_clannad}[args.cmd](args)


if __name__ == "__main__":
    main()
