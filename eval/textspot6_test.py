#!/usr/bin/env python3
"""textspot6's test reads (issue #128, docs/textspot6.md "Test sets, read once after the selection is committed";
eval/textspot6_protocol.json "test"). Each command reads its set ONCE: OUT/<command>.attempt.json is created
exclusively (open(..., "x")) before any test pixel is read, and refuses a second run. Run only after the selection
record (eval/textspot6_dev_<date>.json) is committed and pushed. Outputs: OUT/<command>.json with counts, rates,
intervals and hashes only; the units, rows and boxes stay under OUT/private/.

  textspot6_test.py title --selection SEL.json --blob FINAL.vtsp --decode D --out OUT --pagepath PAGEPATH
      the test title(s) of the page pool (split "test": Dance with Devils, PCSG00807) through eval/pagepath.py:
      GRAD + refine (--finder rubi) against the spotter's boxes in its decode, paired (page clusters), Vita view
  textspot6_test.py clannad --selection SEL.json --blob FINAL.vtsp --decode D --out OUT --pagepath PAGEPATH
                            --work CLANNAD_WORK
      CLANNAD's frozen long run (owner-drive-ingest's references, refs.freeze.json): each page's ROI (the player's
      region) as a unit; GRAD + refine and the spotter read through eval/pagepath.py; then, as textspot's CLANNAD
      read, a reference line reads the boxes that lie mostly inside its GRAD box (textdet_protected.assign), their
      readings joined in the unit's box order; dialogue = the lines not flagged name; page-cluster bootstrap
  textspot6_test.py tests --selection SEL.json --blob FINAL.vtsp --out OUT [--refs3 DIR]
      vita-test + psp-test as whole frames (eval/manifests/textspot-test.jsonl, textspot's pinned test manifest):
      textspot's whole-frame metrics through eval/textspot.py's reading (VOCR5-N W8I greedy) for det and
      det+refine, the tagged dialogue CER and presence on the text-free frames
"""
import argparse
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "eval", ROOT / "synth", ROOT / "train", ROOT / "runtime"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import textspot6 as E6  # noqa: E402

HERD = E6.HERD
ART = E6.ART
TEST_MANIFEST = ROOT / "eval" / "manifests" / "textspot-test.jsonl"
TEST_MANIFEST_SHA256 = None    # read from eval/textspot_protocol.json's pin at run time


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def once(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as f:
        f.write(json.dumps(record, indent=1) + "\n")


def check_selection(sel_path, blob):
    sel = json.loads(Path(sel_path).read_text(encoding="utf-8"))
    if sel.get("kind") != "textspot6-selection":
        raise SystemExit(f"{sel_path}: not a textspot6 selection record")
    rc = subprocess.run(["git", "ls-files", "--error-unmatch", str(sel_path)], cwd=ROOT, capture_output=True)
    if rc.returncode:
        raise SystemExit(f"{sel_path}: the selection record must be committed before any test read")
    return sel, sha256_file(blob)


def pagepath_run(pagepath, units, out_base, extra, pages=None, workers=5):
    A = ART
    cmd = [sys.executable, str(pagepath), "run", "--units", str(units), *(["--pages", str(pages)] if pages else []),
           *extra, "--models", str(A / "vvo-public" / "v030"), "--lm",
           str(A / "vvo-public" / "v030" / "vocr_lm_o3_1000k.vlm"), "--rubi",
           os.environ.get("RUBI", "/home/skerraut/herd-synth3/clones/vocr6-spotter-rubi"),
           "--cache", str(Path(out_base).parent / "reads.sqlite"), "--work", str(Path(out_base).parent / "work"),
           "--workers", str(workers), "--rows", f"{out_base}.rows.jsonl", "--out", f"{out_base}.json"]
    subprocess.run(cmd, check=True, stdin=subprocess.DEVNULL)
    return json.loads(Path(f"{out_base}.json").read_text(encoding="utf-8"))


def paired(rows_a, rows_b, key="cluster"):
    """CER(a) - CER(b) with eval/vocr3_report.cluster_bootstrap over `key` clusters."""
    import vocr3_report as VR
    a = {r["uid"]: r for r in rows_a}
    b = {r["uid"]: r for r in rows_b}
    pa = {k: (a[k]["errors"], a[k]["chars"]) for k in a}
    pb = {k: (b[k]["errors"], b[k]["chars"]) for k in a}
    cl = {k: a[k][key] for k in a}
    n = sum(v[1] for v in pa.values())
    d = 100 * (sum(v[0] for v in pa.values()) - sum(v[0] for v in pb.values())) / n if n else None
    out = VR.cluster_bootstrap(pa, pb, cl)
    ea, eb = sum(v[0] for v in pa.values()), sum(v[0] for v in pb.values())
    return {"delta": round(d, 4) if d is not None else None, "a_cer": round(100 * ea / n, 4) if n else None,
            "b_cer": round(100 * eb / n, 4) if n else None, "chars": n, **out}


# ----------------------------------------------------------------------------- the test title

def cmd_title(a):
    sel, blob_sha = check_selection(a.selection, a.blob)
    out = Path(a.out)
    once(out / "title.attempt.json", {"started": utc(), "selection_sha256": sha256_file(a.selection),
                                      "blob_sha256": blob_sha, "decode": a.decode,
                                      "pagepath_sha256": sha256_file(a.pagepath)})
    priv = out / "private" / "title"
    priv.mkdir(parents=True, exist_ok=True)
    units = priv / "units_test.jsonl"
    subprocess.run([sys.executable, str(a.pagepath), "units", "--pagepool", str(E6.POOL), "--set", "test",
                    "--out", str(units)], check=True, stdin=subprocess.DEVNULL)
    base = pagepath_run(a.pagepath, units, priv / "grad_refine", ["--finder", "rubi"], E6.PAGES)
    boxes = priv / f"boxes_{a.decode}.jsonl"
    if a.decode.startswith("crop"):
        E6.main(["spot-crop", "--blob", str(a.blob), "--set", "hval", "--units", str(units), "--out", str(boxes),
                 "--decode", "detref" if a.decode.endswith("refine") else "det"])
        arm = pagepath_run(a.pagepath, units, priv / "spotter", ["--boxes", str(boxes)], E6.PAGES)
    else:
        raise SystemExit("title: a whole-frame decode reads the test pages' frames; add the frame list first")
    res = {"kind": "textspot6-test-title", "computed": utc(), "selection_sha256": sha256_file(a.selection),
           "blob_sha256": blob_sha, "decode": a.decode, "units": sum(1 for _ in open(units)) - 1,
           "grad_refine": {k: base.get(k) for k in ("summary", "by_group") if k in base},
           "spotter": {k: arm.get(k) for k in ("summary", "by_group") if k in arm},
           "spotter_minus_grad_refine": paired(arm["units"], base["units"], "cluster"),
           "spotter_minus_grad_refine_pages": paired(arm["units"], [dict(u, page=u["uid"].split(":")[1])
                                                                    for u in base["units"]], "cluster")}
    # page clusters (one title: the title-cluster interval is degenerate)
    for u in arm["units"]:
        u["page_cluster"] = ":".join(u["uid"].split(":")[:2])
    for u in base["units"]:
        u["page_cluster"] = ":".join(u["uid"].split(":")[:2])
    res["spotter_minus_grad_refine_pages"] = paired(arm["units"], base["units"], "page_cluster")
    (out / "title.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(res["spotter_minus_grad_refine_pages"]))


# ----------------------------------------------------------------------------- CLANNAD

def clannad_units(work):
    """(units in eval/pagepath.py's dev-unit pickle form: one per page with reference lines, the ROI's grey;
    the reference lines per page with their GRAD boxes in ROI px) of the frozen long run."""
    import clannad_refs as CR
    from PIL import Image

    from common import to_gray
    work = Path(work)
    freeze = json.loads((work / "refs.freeze.json").read_text(encoding="utf-8"))
    if sha256_file(work / "refs.jsonl") != freeze["refs_sha256"]:
        raise SystemExit("refs.jsonl differs from its freeze record")
    refs = {x["id"]: x for x in CR.read_jsonl(work / "refs.jsonl")}
    rows, meta = CR.load_lines(work)
    session = Path(meta["session"])
    by_page = defaultdict(list)
    for r in rows:
        by_page[r["page"]].append(r)
    units, lines = [], {}
    for page, rs in sorted(by_page.items()):
        ref_rows = [r for r in rs if refs.get(r["id"], {}).get("status") == "ref"]
        src = rs[0]["source"]
        path = session / src["path"]
        if sha256_file(path) != src["sha256"]:
            raise SystemExit(f"{src['path']}: frame hash differs from the capture journal")
        with Image.open(path) as im:
            g = to_gray(np.ascontiguousarray(np.asarray(im.convert("RGB"))))
        x, y, w, h = rs[0]["roi"]
        roi = np.ascontiguousarray(g[y:y + h, x:x + w])
        lines[page] = [{"id": r["id"], "box": [r["box"][0] - x, r["box"][1] - y, r["box"][2], r["box"][3]],
                        "text": refs[r["id"]]["text"], "name": bool(refs[r["id"]].get("name"))} for r in ref_rows]
        units.append({"cohort": "clannad", "unit": page, "shot": page, "variant": "base", "vn": "clannad",
                      "kind": "vita", "src_kind": "vita", "scale": 1.0, "region": [x, y, x + w, y + h],
                      "image": roi, "frame_dims": [g.shape[1], g.shape[0]],
                      "lines": [{"id": ln["id"], "text": ln["text"], "styles": ["name"] if ln["name"] else [],
                                 "box": [ln["box"][0], ln["box"][1], ln["box"][0] + ln["box"][2],
                                         ln["box"][1] + ln["box"][3]]} for ln in lines[page]]})
    return units, lines, freeze


def cmd_clannad(a):
    import clannad_refs as CR
    import textdet_protected as TDP
    import vocr3_report as V
    sel, blob_sha = check_selection(a.selection, a.blob)
    out = Path(a.out)
    once(out / "clannad.attempt.json", {"started": utc(), "selection_sha256": sha256_file(a.selection),
                                        "blob_sha256": blob_sha, "decode": a.decode,
                                        "pagepath_sha256": sha256_file(a.pagepath)})
    priv = out / "private" / "clannad"
    priv.mkdir(parents=True, exist_ok=True)
    units, lines, freeze = clannad_units(a.work)
    pkl = priv / "units.pkl"
    with open(pkl, "wb") as f:
        pickle.dump({"units": units}, f)
    pagepath_run(a.pagepath, pkl, priv / "grad_refine", ["--finder", "rubi"])
    import textspot6_twin as T6
    m = T6.Model(Path(a.blob).read_bytes())
    k = "detref" if a.decode.endswith("refine") else "det"
    with open(priv / f"boxes_{a.decode}.jsonl", "w", encoding="utf-8") as f:
        for i, u in enumerate(units):
            r = T6.spot(m, u["image"], scale=T6.scale_of(m, *u["frame_dims"]))
            bx = [[b.x, b.y, b.w, b.h, b.vertical, b.conf, b.dark, b.block, b.tag, b.tconf] for b in r.boxes]
            if k == "detref":
                bx = [[*b[:7], -1, b[7], 0] for b in E6.refined(bx, u["image"])]
            f.write(json.dumps({"unit": f"dev:{i}:{u['unit']}:{u['variant']}",
                                "boxes": [[b[0], b[1], b[2], b[3], b[6]] for b in bx if not b[4]]}) + "\n")
    pagepath_run(a.pagepath, pkl, priv / "spotter", ["--boxes", str(priv / f"boxes_{a.decode}.jsonl")])
    rows = {"grad_refine": {r["uid"]: r for r in E6.read_jsonl(priv / "grad_refine.rows.jsonl")},
            "spotter": {r["uid"]: r for r in E6.read_jsonl(priv / "spotter.rows.jsonl")}}
    preds = {s: {} for s in rows}
    for i, u in enumerate(units):
        uid = f"dev:{i}:{u['unit']}:{u['variant']}"
        gb = [ln["box"] for ln in lines[u["unit"]]]
        for s, rr in rows.items():
            r = rr[uid]
            owner = TDP.assign([tuple(b) for b in r["boxes_xywh"]], gb)
            for j, ln in enumerate(lines[u["unit"]]):
                preds[s][ln["id"]] = "".join(t for t, o in zip(r["texts"], owner) if o == j)
    result = {"kind": "textspot6-clannad", "computed": utc(), "refs_sha256": freeze["refs_sha256"],
              "selection_sha256": sha256_file(a.selection), "blob_sha256": blob_sha, "decode": a.decode,
              "pages": len(units), "pages_with_ref": sum(1 for u in units if lines[u["unit"]]),
              "assignment": "a box goes to the GRAD box of the page's ROI it overlaps most when that box holds at "
                            "least half its area (textdet_protected.assign); a reference line reads its boxes joined "
                            "in the page path's box order", "scopes": {}}
    for scope in ("dialogue", "all"):
        sel_l = [(u["unit"], ln) for u in units for ln in lines[u["unit"]] if scope == "all" or not ln["name"]]
        pg = [p for p, _ in sel_l]
        ids = sorted(set(pg))
        e = {"lines": len(sel_l), "pages": len(ids), "systems": {}}
        idx = CR.boot_idx(len(ids), CR.REPS, CR.SEED)
        sums = {}
        for s in rows:
            us = [CR.line_unit(ln["text"], preds[s][ln["id"]]) for _, ln in sel_l]
            sums[s] = CR.page_sums(us, pg)
            e["systems"][s] = CR.summary(sums[s], ids, idx)
        a_ = {p: (sums["spotter"][p][0], sums["spotter"][p][1]) for p in ids}
        b_ = {p: (sums["grad_refine"][p][0], sums["grad_refine"][p][1]) for p in ids}
        d = V.paired_bootstrap(a_, b_, CR.REPS, CR.SEED)
        e["spotter_minus_grad_refine"] = {"cer_delta": d["delta"], "cer_ci95": d["ci95"],
                                          "spotter_errors": d["candidate_errors"],
                                          "grad_refine_errors": d["baseline_errors"], "chars": d["chars"]}
        result["scopes"][scope] = e
    no_text = [u for u in units if not lines[u["unit"]]]
    result["no_text_pages"] = {"pages": len(no_text),
                               "boxes": {s: sum(len(rows[s][f"dev:{units.index(u)}:{u['unit']}:{u['variant']}"]
                                                    ["boxes_xywh"]) for u in no_text) for s in rows}}
    (out / "clannad.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    for s, v in result["scopes"]["dialogue"]["systems"].items():
        print(f"{s}: dialogue CER {v['cer']} {v['cer_ci95']}")
    print("spotter - GRAD + refine:", result["scopes"]["dialogue"]["spotter_minus_grad_refine"])


# ----------------------------------------------------------------------------- vita-test + psp-test

def cmd_tests(a):
    import textspot as TSE
    import textspot_metrics as M

    import textspot6_twin as T6
    sel, blob_sha = check_selection(a.selection, a.blob)
    proto = json.loads((ROOT / "eval" / "textspot_protocol.json").read_text(encoding="utf-8"))
    pin = TSE.pinned_sha(proto, "eval/manifests/textspot-test.jsonl")
    digest = sha256_file(TEST_MANIFEST)
    if pin != digest:
        raise SystemExit(f"{TEST_MANIFEST}: sha256 {digest} is not textspot's pinned {pin}")
    out = Path(a.out)
    once(out / "tests.attempt.json", {"started": utc(), "selection_sha256": sha256_file(a.selection),
                                      "blob_sha256": blob_sha, "manifest_sha256": digest})
    rows = [json.loads(x) for x in TEST_MANIFEST.read_text().splitlines()]
    report_ids = {r["id"] for r in rows if r.get("report_only") or r.get("part") == "adrenaline"}
    dirs = [Path(d) for d in (a.shots or [])]
    if a.data:
        dirs += [Path(a.data) / "vita" / "shots", Path(a.data) / "psp" / "shots"]
    dirs += [TSE.P / "sets" / "test" / "shots", TSE.T / "sets" / "test" / "shots"]
    frames = TSE.load_frames(rows, dirs)
    m = T6.Model(Path(a.blob).read_bytes())
    cache = TSE.Cache(out / "private" / "cache")
    models = ["VOCR5-N"]
    readers = TSE.c_readers({k: TSE.BLOBS[k] for k in [*models, TSE.V_MODEL]}, out / "private" / "cbuild")
    res = {"kind": "textspot6-tests", "computed": utc(), "manifest_sha256": digest, "blob_sha256": blob_sha,
           "scored_frames": len(rows) - len(report_ids), "decodes": {}}
    for dec in ("det", "detref"):
        arms, tagged_boxes = [], []
        for f in frames:
            r = T6.spot(m, f.grey)
            bx = [[b.x, b.y, b.w, b.h, b.vertical, b.conf, b.dark, b.block, b.tag, b.tconf] for b in r.boxes]
            if dec == "detref":
                ob, bl, pk = T6.group_and_order([(*b[:7], b[7], 0) for b in E6.refined(bx, f.grey)], f.height)
                bx = [[*b[:7], b[9], b[7], b[8]] for b in ob]
                arm = {"boxes": [tuple(b[:8]) for b in bx], "blocks": [tuple(k[:7]) for k in bl], "pick": pk,
                       "truncated": r.truncated}
            else:
                arm = {"boxes": [tuple(b[:8]) for b in bx],
                       "blocks": [(k.x, k.y, k.w, k.h, k.vertical, k.first, k.count) for k in r.blocks],
                       "pick": r.pick, "truncated": r.truncated}
            arms.append({"spot": arm})
            tagged_boxes.append((bx, r.presence))
        TSE.read_all(frames, arms, models, readers, cache, workers=a.workers)
        units, tagged = [], []
        for f, ar, (bx, pres) in zip(frames, arms, tagged_boxes):
            texts = TSE.texts_of(f, ar["spot"]["boxes"], models[0], readers, cache)
            units.append(TSE.frame_unit(f, ar["spot"], texts))
            dets = [{"box": list(b[:4]), "vertical": bool(b[4]), "text": t, "tag": b[8]} for b, t in zip(bx, texts)]
            e, n = E6.tagged_dialogue(dets, f.refs, f.width, f.height)
            tagged.append({"unit": f.id, "vn": f.vn, "errors": e, "chars": n, "presence": pres,
                           "text_free": f.text_free, "report": f.id in report_ids})
        sc = [u for u in units if u["unit"] not in report_ids]
        summ = TSE.summarize(sc)
        tf = [t for t in tagged if t["text_free"] and not t["report"]]
        summ["tagged_dialogue"] = M.cer_of([t for t in tagged if not t["report"]])
        summ["presence_text_free"] = {"frames": len(tf), "with_dialogue_box": sum(1 for t in tf if t["presence"])}
        res["decodes"][dec] = summ
        print(f"{dec}: F1 {summ['detection']['f1']:.4f} CER all {summ['cer_all']['cer']} dialogue "
              f"{summ['cer_dialogue']['cer']} tagged {summ['tagged_dialogue']['cer']}")
    cache.close()
    (out / "tests.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("title", "clannad", "tests"):
        p = sub.add_parser(name)
        p.add_argument("--selection", required=True)
        p.add_argument("--blob", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--workers", type=int, default=5)
        if name != "tests":
            p.add_argument("--decode", required=True, choices=("crop", "crop+refine", "det", "det+refine"))
            p.add_argument("--pagepath", required=True)
        if name == "clannad":
            p.add_argument("--work", required=True)
        if name == "tests":
            p.add_argument("--data", help="eval's data directory (vita/shots, psp/shots)")
            p.add_argument("--shots", nargs="*")
    a = ap.parse_args(argv)
    {"title": cmd_title, "clannad": cmd_clannad, "tests": cmd_tests}[a.cmd](a)


if __name__ == "__main__":
    main()
