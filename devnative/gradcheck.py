#!/usr/bin/env python3
"""GRAD crops: the console finder's boxes for the labelled lines, and their containment check.

  gradcheck.py boxes  --pass1 DIR... [--out $OUT/grad/boxes.jsonl]
  gradcheck.py sheets [--boxes ...] --out BATCH [--chunk N]       containment review sheets
  gradcheck.py apply  [--boxes ...] --batch DIR... [--out $OUT/grad/grad.jsonl]

A screenshot's dialogue regions are built (devnative/lines.regions, eval's rule) from the lines
pass 1 called dialogue: decided (pick or text), unsure, or cut. Lines skipped as ui or bad stay
out, as eval leaves UI out of its regions. The console's GRAD finder runs on each region and its
boxes are matched to the lines (lines.grad_assign). A labelled line gets a GRAD crop only when
its box holds the whole line and no other detected text (status "ok") and, where the box runs
more than a little beyond the reviewed ink crop (left/right by more than max(3 px, 0.3 ink
heights), top/bottom by more than 0.6 ink heights), a containment check on a contact sheet
confirms that the box shows exactly the label: the detector does not see every glyph (card
digits, icons, a line that runs on), so text in the extension would make the label wrong for
that crop. Every decision is the assistant's (reviewer_kind "assistant").
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from devnative import decide as D  # noqa: E402
from devnative import lines as L  # noqa: E402
from devnative import paths as P  # noqa: E402
from devnative import sheets as S  # noqa: E402
from teacher import vote  # noqa: E402

GRAD_DIR = P.OUT / "grad"


def extension(gbox, crop):
    """How far a GRAD box runs beyond the ink crop: [left, top, right, bottom] px (>= 0)."""
    return [max(0, crop[0] - gbox[0]), max(0, crop[1] - gbox[1]), max(0, gbox[2] - crop[2]),
            max(0, gbox[3] - crop[3])]


def needs_check(gbox, crop, ink_h):
    left, top, right, bottom = extension(gbox, crop)
    side = max(3, 0.3 * ink_h)
    return left > side or right > side or top > 0.6 * ink_h or bottom > 0.6 * ink_h


def region_member(rec):
    """Pass-1 decisions whose line belongs in the dialogue regions."""
    return rec["decision"] in ("pick", "text", "unsure") or (rec["decision"] == "skip" and rec["skip"] == "cut")


def boxes(shots, p1):
    """GRAD results for every pass-1-decided (pick/text) line: {id: record}."""
    idx = S.line_index(shots)
    by_shot = {}
    for lid, r in p1.items():
        if region_member(r):
            by_shot.setdefault(idx[lid][0]["shot"], []).append(lid)
    out = {}
    for sk in sorted(by_shot):
        s = shots[sk]
        rgb = P.load_screen(s)
        H, W = rgb.shape[:2]
        ids = sorted(by_shot[sk])
        lines = [idx[i][1] for i in ids]
        for lid, ln, g in zip(ids, lines, L.grad_assign(rgb, lines, s.get("dets", []), s["kind"])):
            if p1[lid]["decision"] not in ("pick", "text"):
                continue
            crop = L.crop_box(ln, s["kind"], W, H)
            rec = {"id": lid, "shot": sk, "region": g["region"], "box": g["box"], "status": g["status"],
                   "dark": g["dark"], "found": g.get("found"), "crop": crop, "label": p1[lid]["text"]}
            if g["status"] == "ok":
                rec["extension"] = extension(g["box"], crop)
                rec["check"] = needs_check(g["box"], crop, L.ink_height(ln))
            out[lid] = rec
    return out


def cmd_boxes(args):
    shots = S.load_reads(args.reads)
    p1 = D.collect(args.pass1)
    res = boxes(shots, p1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for _, r in sorted(res.items())),
                              encoding="utf-8")
    st = Counter(r["status"] + (":check" if r.get("check") else "") for r in res.values())
    print(json.dumps({"lines": len(res), "by": dict(st)}), flush=True)


def load_boxes(path):
    return {r["id"]: r for r in (json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines()
                                 if x.strip())}


def cmd_sheets(args):
    shots = S.load_reads(args.reads)
    idx = S.line_index(shots)
    todo = [r for r in load_boxes(args.boxes).values() if r["status"] == "ok" and r.get("check")]
    todo.sort(key=lambda r: (int(r["shot"].split("_")[0][1:]), r["id"]))
    entries = []
    for r in todo:
        s, ln = idx[r["id"]]
        entries.append((r["id"], s, ln, [{"text": r["label"], "key": vote.key(r["label"]), "engines": []}],
                        r["box"]))
    chunk = args.chunk or len(entries) or 1
    for k, i in enumerate(range(0, len(entries), chunk), 1):
        sub = Path(args.out) / f"b{k:02d}"
        n = S.make_batch(entries[i:i + chunk], sub, "grad", name=f"{Path(args.out).name}/{sub.name}")
        print(f"{sub}: {len(entries[i:i + chunk])} lines, {n} sheets", flush=True)


def accept(rec, decision):
    """(GRAD crop kept, why) for one boxes record and its containment decision (or None)."""
    if rec["status"] != "ok":
        return False, f"grad-{rec['status']}"
    if not rec.get("check"):
        return True, "inside"
    if decision is None:
        return False, "grad-unchecked"
    if decision["decision"] == "pick" and decision.get("pick") == 1:
        return True, "checked"
    if decision["decision"] == "text" and vote.key(decision["text"]) == vote.key(rec["label"]):
        return True, "checked"
    return False, {"text": "grad-extra-text", "skip": f"grad-{decision.get('skip')}",
                   "unsure": "grad-unsure"}.get(decision["decision"], "grad-rejected")


def cmd_apply(args):
    recs = load_boxes(args.boxes)
    dec = D.collect(args.batch or [])
    out, why = [], Counter()
    for lid, r in sorted(recs.items()):
        ok, how = accept(r, dec.get(lid))
        why[how] += 1
        out.append({k: r.get(k) for k in ("id", "region", "box", "status", "dark", "check", "extension")}
                   | {"grad_ok": ok, "grad_why": how, "review": (dec.get(lid) or {}).get("decision")})
    Path(args.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    print(json.dumps({"lines": len(out), "kept": sum(r["grad_ok"] for r in out), "by": dict(why)}), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reads", default=str(P.OUT / "reads" / "reads.jsonl"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("boxes")
    p.add_argument("--pass1", nargs="+", required=True)
    p.add_argument("--out", default=str(GRAD_DIR / "boxes.jsonl"))
    p = sub.add_parser("sheets")
    p.add_argument("--boxes", default=str(GRAD_DIR / "boxes.jsonl"))
    p.add_argument("--out", required=True)
    p.add_argument("--chunk", type=int, default=0)
    p = sub.add_parser("apply")
    p.add_argument("--boxes", default=str(GRAD_DIR / "boxes.jsonl"))
    p.add_argument("--batch", nargs="*")
    p.add_argument("--out", default=str(GRAD_DIR / "grad.jsonl"))
    args = ap.parse_args()
    {"boxes": cmd_boxes, "sheets": cmd_sheets, "apply": cmd_apply}[args.cmd](args)


if __name__ == "__main__":
    main()
