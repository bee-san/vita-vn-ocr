#!/usr/bin/env python3
"""textgate: the protected sets, once, after the selection record is pushed (eval/textgate_protocol.json
"protected").

  textgate_protected.py tests --gate NAME=BLOB --mode MODE --k K --t T --units97 JSON --data EVAL --out DIR
      vita-test + psp-test whole dialogue: eval's dialogue regions (a region holding an uncertain line left out whole)
      at the base box and grown by 4 and 8 px, exactly as eval/textdet_protected.py cut them. The gate runs on every
      case (its twin, eval/textgate.score; GRAD boxes from the C finder for the boxes mode); a case it passes keeps
      #97's GRAD + refine unit for each VOCR5 model (OUT of #97's single protected read,
      $HERD/shared/private/textdet/protected/tests/private/tests_units.json), a case it stops reads nothing. Body
      (dialogue) and all text, VN-cluster intervals, per set.
  textgate_protected.py clannad --gate NAME=BLOB --mode MODE --k K --t T --model NAME=BLOB ... --work DIR --out DIR
      CLANNAD's long run against owner-drive-ingest's frozen references (eval/clannad_refs.py): each page's ROI cut
      from its native frame, a reference line a GRAD box of that ROI; GRAD + refine read every reference line as
      eval/textdet_protected.py does (the refined lines from that box, joined); with the gate, a page it stops reads
      nothing. Whole-page bootstraps; dialogue and all lines; the lines GRAD + refine return on the pages the
      references hold no text for, with and without the gate.

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

PROTOCOL = ROOT / "eval" / "textgate_protocol.json"
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


class Gate:
    def __init__(self, blob, mode, k, t, cgate=None):
        import textdet_twin as TW
        self.blob, self.mode, self.k, self.t = Path(blob), mode, int(k), int(t)
        self.model = TW.Model(self.blob.read_bytes())
        self.cgate = cgate

    def scores(self, images, grad_boxes):
        import textgate as G
        out = [G.score(G.kmap(self.model, g), self.mode, self.k, self.model.stride, b)
               for g, b in zip(images, grad_boxes)]
        c_equal = None
        if self.cgate:
            c = self.cgate(self.blob, self.mode, self.k, images, grad_boxes)
            c_equal = list(c) == list(out)
            if not c_equal:
                raise RuntimeError("the C gate differs from the twin on a protected case")
        return out, c_equal


def c_gate_fn(cbin):
    """The C runtime's batch scores (runtime/vocr_textgate_test --batch), when a driver is given."""
    if not cbin:
        return None
    import textgate_c as GC
    return lambda blob, mode, k, images, boxes: GC.c_scores(cbin, blob, mode, k, images, boxes)


def cmd_tests(args):
    import finder_tighten as FT
    import sets
    import textdet as TD
    import vocr3_report as V

    from common import to_gray
    out = Path(args.out)
    gname, gblob = parse(args.gate)
    units97 = json.loads(Path(args.units97).read_text(encoding="utf-8"))["units"]
    once(out / "tests.attempt.json", {"started": utc_now(), "protocol_sha256": sha256(PROTOCOL), "gate": gname,
                                      "gate_sha256": sha256(gblob), "mode": args.mode, "k": args.k, "t": args.t,
                                      "units97_sha256": sha256(args.units97)})
    gate = Gate(gblob, args.mode, args.k, args.t, c_gate_fn(args.cgate))
    ident = FT.build(out / "build", args.compiler)
    keys, scores, c_equal = [], [], []
    for set_name in ("vita-test", "psp-test"):
        regions, excluded = V.dialogue_regions(set_name, args.data)
        cache = sets.ImageCache()
        cases = []
        for reg in regions:
            img = cache.get(reg["path"])
            dims = [img.shape[1], img.shape[0]]
            for variant, pad in VARIANTS.items():
                x, y, w, h = V.expand(reg["region"], pad, dims)
                cases.append((f"{set_name}|{reg['region_id']}|{variant}",
                              np.ascontiguousarray(to_gray(np.ascontiguousarray(img[y:y + h, x:x + w])))))
        imgs = [g for _, g in cases]
        grad = FT.lines_of(ident["driver"], imgs, 0)
        s, eq = gate.scores(imgs, [g[0] for g in grad])
        keys += [k for k, _ in cases]
        scores += s
        c_equal.append(eq)
        print(f"{set_name}: {len(regions)} regions, {len(cases)} cases (excluded {excluded})", flush=True)
    keep = {k: s >= gate.t for k, s in zip(keys, scores)}
    result = {"kind": "textgate-protected-tests", "gate": gname, "gate_sha256": sha256(gblob), "mode": args.mode,
              "k": args.k, "t": args.t, "c_equals_twin": c_equal, "native": ident, "cases": len(keys),
              "cases_stopped": sum(not v for v in keep.values()),
              "stopped_by_set": {s: sum(not v for k, v in keep.items() if k.startswith(s)) for s in
                                 ("vita-test", "psp-test")},
              "bootstrap": FT.BOOT, "dialogue": {}}
    priv = {}
    for name, modes in units97.items():
        base = modes["refine"]
        if sorted(u["unit"] for u in base) != sorted(keys):
            raise SystemExit(f"{name}: #97's refine units are not this run's cases")
        gated = []
        for u in base:
            g = dict(u)
            if not keep[u["unit"]]:
                for a in ("all", "body"):
                    for t in ("full", "hira"):
                        g[f"{a}_{t}_errors"] = u[f"{a}_{t}_chars"]
                g["gated"] = True
            gated.append(g)
        priv[name] = gated
        e = {}
        for scope in ("body", "all"):
            for v in list(VARIANTS) + ["all"]:
                for plat in ("vita", "psp", "pooled"):
                    def keepf(u, v=v, plat=plat):
                        return (v == "all" or u["variant"] == v) and (plat == "pooled" or u["platform"] == plat)

                    sel = {m: [dict(u, errors=u[f"{scope}_full_errors"], chars=u[f"{scope}_full_chars"]) for u in us]
                           for m, us in (("refine", base), ("gate", gated))}
                    t = {m: FT.totals(sel[m], keepf) for m in sel}
                    e[f"{scope}/{v}/{plat}"] = {**{m: {"errors": t[m][0], "chars": t[m][1], "cer": FT.pct(t[m])}
                                                   for m in t},
                                                "gate_minus_refine": FT.compare(sel["gate"], sel["refine"], keepf)}
        result["dialogue"][name] = e
    FT.assert_public(result)
    (out / "tests.json").write_text(json.dumps(result, indent=1) + "\n")
    (out / "private").mkdir(exist_ok=True)
    (out / "private" / "tests_units.json").write_text(json.dumps({"units": priv, "keep": keep}) + "\n")
    print(json.dumps({"stopped": result["cases_stopped"], "by_set": result["stopped_by_set"]}))
    for name in units97:
        k = result["dialogue"][name]["body/all/pooled"]
        print(name, k["refine"]["cer"], "->", k["gate"]["cer"], k["gate_minus_refine"]["delta"],
              k["gate_minus_refine"]["vn_ci95"], flush=True)
    del TD


def cmd_clannad(args):
    import clannad_refs as C
    import finder_tighten as FT
    import vocr3_report as V
    from PIL import Image

    from common import to_gray
    out = Path(args.out)
    work = Path(args.work)
    gname, gblob = parse(args.gate)
    freeze = json.loads((work / "refs.freeze.json").read_text(encoding="utf-8"))
    if sha256(work / "refs.jsonl") != freeze["refs_sha256"]:
        raise SystemExit("refs.jsonl differs from its freeze record")
    once(out / "clannad.attempt.json", {"started": utc_now(), "refs_sha256": freeze["refs_sha256"],
                                        "protocol_sha256": sha256(PROTOCOL), "gate": gname,
                                        "gate_sha256": sha256(gblob), "mode": args.mode, "k": args.k, "t": args.t})
    gate = Gate(gblob, args.mode, args.k, args.t, c_gate_fn(args.cgate))
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
    gscores, c_equal = gate.scores(imgs, [g[0] for g in grad])
    preds = {n: {"refine": {}, "gate": {}} for n in readers}
    extra = {"pages": len(pages), "pages_stopped": 0, "pages_with_ref": 0, "pages_with_ref_stopped": 0,
             "no_text_pages": 0, "no_text_pages_stopped": 0, "refine_lines_on_no_text_pages": 0,
             "gate_lines_on_no_text_pages": 0, "unmatched": 0}
    for (page, rs), img, (gb, gd, _, _, _), (rb, rdk, org, _, _), sc in zip(pages, imgs, grad, refd, gscores):
        stop = sc < gate.t
        extra["pages_stopped"] += stop
        ref_rows = [r for r in rs if refs.get(r["id"], {}).get("status") == "ref"]
        if not ref_rows:
            extra["no_text_pages"] += 1
            extra["no_text_pages_stopped"] += stop
            extra["refine_lines_on_no_text_pages"] += len(rb)
            extra["gate_lines_on_no_text_pages"] += 0 if stop else len(rb)
            continue
        extra["pages_with_ref"] += 1
        extra["pages_with_ref_stopped"] += stop
        for r in ref_rows:
            box = tuple(r["box"])
            if box not in gb:
                extra["unmatched"] += 1
                continue
            k = gb.index(box)
            for n, rd in readers.items():
                text = "".join(rd.read(img[yy:yy + hh, xx:xx + ww], dd)
                               for (xx, yy, ww, hh), dd, o in zip(rb, rdk, org) if o == k)
                preds[n]["refine"][r["id"]] = text
                preds[n]["gate"][r["id"]] = "" if stop else text
    first = preds[next(iter(readers))]["refine"]
    scored = [x for x in refs.values() if x["status"] == "ref" and x["id"] in first]
    result = {"kind": "textgate-clannad", "gate": gname, "gate_sha256": sha256(gblob), "mode": args.mode,
              "k": args.k, "t": args.t, "refs_sha256": freeze["refs_sha256"], "scored_lines": len(scored),
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
            for mode in ("refine", "gate"):
                u = [C.line_unit(x["text"], preds[n][mode][x["id"]]) for x in sel]
                sums[mode] = C.page_sums(u, pg)
                e["systems"][f"{n}/{mode}"] = C.summary(sums[mode], ids, idx)
            a = {p: (sums["gate"][p][0], sums["gate"][p][1]) for p in ids}
            b = {p: (sums["refine"][p][0], sums["refine"][p][1]) for p in ids}
            cer = V.paired_bootstrap(a, b, C.REPS, C.SEED)
            e["pairs"][f"{n}/gate:{n}/refine"] = {"cer_delta": cer["delta"], "cer_ci95": cer["ci95"],
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
        p.add_argument("--gate", required=True)
        p.add_argument("--mode", required=True, choices=("region", "boxes"))
        p.add_argument("--k", type=int, required=True)
        p.add_argument("--t", type=int, required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--cgate", default="", help="runtime/vocr_textgate_test: the C gate must equal the twin")
        p.add_argument("--compiler", default="gcc")
        if name == "tests":
            p.add_argument("--units97", required=True)
            p.add_argument("--data", required=True)
        else:
            p.add_argument("--model", action="append", required=True)
            p.add_argument("--work", required=True)
    args = ap.parse_args()
    {"tests": cmd_tests, "clannad": cmd_clannad}[args.cmd](args)


if __name__ == "__main__":
    main()
