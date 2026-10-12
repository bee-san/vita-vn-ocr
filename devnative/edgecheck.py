#!/usr/bin/env python3
"""Edge check of the ink crops: does each final crop hold the whole line and exactly its label?

  edgecheck.py boxes  [--labels $OUT/labels.jsonl] [--evidence DIR...] [--out $OUT/edge/boxes.jsonl]
  edgecheck.py sheets [--boxes ...] --out BATCH [--chunk N]
  edgecheck.py apply  [--boxes ...] --batch DIR... [--out $OUT/edge/edge.jsonl]

The reviewers decided each line on its ink crop (ink + furigana + 4/2 px). MeikiOCR's
detector sometimes leaves an end 。, 」 or 「 out of its box, so (a) a label read off a wider
view can hold a character that sits outside that crop, and (b) a crop can leave out a bracket
of its line. The final ink crop is widened to the labelled characters' boxes (lines.final_crop,
teacher's rule), and a line is checked on a contact sheet (sheets.py mode "edge": the frame is
the final crop, the one candidate its label) when
  * the final crop differs from the reviewed one by WIDEN px or more on a side, or
  * ink continues the line's rows just beyond the final crop (lines.ink_beyond > max(2, 0.2 ink
    heights) on a side), or
  * a later look (the GRAD check or the blind re-check, --evidence) typed other text or called
    the line cut.
A checked line keeps its label only when the check picks it (the frame shows the whole line and
exactly the label); typed text, cut or unsure drops it (why edge-*). Unflagged lines keep the
reviewed crop's decision. Decisions are the assistant's (reviewer_kind "assistant").
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "synth"))
from common import to_gray  # noqa: E402
from devnative import decide as D  # noqa: E402
from devnative import lines as L  # noqa: E402
from devnative import paths as P  # noqa: E402
from devnative import sheets as S  # noqa: E402
from teacher import vote  # noqa: E402

EDGE_DIR = P.OUT / "edge"
WIDEN = 3


def flags(gray, line, kind, W, H, label):
    """(final crop, reviewed crop, reasons) for one labelled line."""
    rc = L.crop_box(line, kind, W, H)
    fc = L.final_crop(line, kind, W, H, L.label_chars(line, vote.key, label))
    why = []
    if rc[0] - fc[0] >= WIDEN:
        why.append("widened-left")
    if fc[2] - rc[2] >= WIDEN:
        why.append("widened-right")
    bl, br = L.ink_beyond(gray, line, fc)
    m = max(2, round(0.2 * L.ink_height(line)))
    if bl > m:
        why.append("ink-left")
    if br > m:
        why.append("ink-right")
    return fc, rc, why


def evidence(dirs):
    """{line id: [reasons]} from later looks, batch by batch (a line can be in several): a GRAD
    check that typed other text, any look that called the line cut."""
    out = {}
    for d in D.batches(dirs):
        for r in D.parsed(d):
            if r["decision"] == "skip" and r["skip"] == "cut":
                out.setdefault(r["id"], []).append(f"{r['mode']}-cut")
            elif r["decision"] == "text" and r["mode"] == "grad":
                out.setdefault(r["id"], []).append("grad-text")
    return out


def cmd_boxes(args):
    shots = S.load_reads(args.reads)
    idx = S.line_index(shots)
    labels = [json.loads(x) for x in Path(args.labels).read_text(encoding="utf-8").splitlines() if x.strip()]
    ev = evidence(args.evidence or [])
    out, cache = [], {}
    for r in sorted((r for r in labels if r["status"] == "kept"), key=lambda r: r["id"]):
        s, ln = idx[r["id"]]
        if s["shot"] not in cache:
            cache.clear()
            rgb = P.load_screen(s)
            cache[s["shot"]] = (rgb, to_gray(rgb))
        rgb, gray = cache[s["shot"]]
        H, W = rgb.shape[:2]
        fc, rc, why = flags(gray, ln, s["kind"], W, H, r["label"])
        why += ev.get(r["id"], [])
        out.append({"id": r["id"], "shot": s["shot"], "label": r["label"], "crop": fc, "reviewed": rc,
                    "check": bool(why), "why": why})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out), encoding="utf-8")
    print(json.dumps({"lines": len(out), "check": sum(x["check"] for x in out),
                      "why": dict(Counter(w for x in out for w in x["why"]))}), flush=True)


def load_boxes(path):
    return {r["id"]: r for r in (json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines()
                                 if x.strip())}


def cmd_sheets(args):
    shots = S.load_reads(args.reads)
    idx = S.line_index(shots)
    todo = sorted((r for r in load_boxes(args.boxes).values() if r["check"]),
                  key=lambda r: (int(r["shot"].split("_")[0][1:]), r["id"]))
    entries = []
    for r in todo:
        s, ln = idx[r["id"]]
        entries.append((r["id"], s, ln, [{"text": r["label"], "key": vote.key(r["label"]), "engines": []}], r["crop"]))
    chunk = args.chunk or len(entries) or 1
    for k, i in enumerate(range(0, len(entries), chunk), 1):
        sub = Path(args.out) / f"b{k:02d}"
        n = S.make_batch(entries[i:i + chunk], sub, "edge", name=f"{Path(args.out).name}/{sub.name}")
        print(f"{sub}: {len(entries[i:i + chunk])} lines, {n} sheets", flush=True)


def accept(rec, decision):
    """(kept, why) of one boxes record given its edge-check decision (or None)."""
    if not rec["check"]:
        return True, "unflagged"
    if decision is None:
        return False, "edge-unchecked"
    if decision["decision"] == "pick" and decision.get("pick") == 1:
        return True, "checked"
    if decision["decision"] == "text" and vote.key(decision["text"]) == vote.key(rec["label"]):
        return True, "checked"
    return False, {"text": "edge-other-text", "skip": f"edge-{decision.get('skip')}",
                   "unsure": "edge-unsure"}.get(decision["decision"], "edge-rejected")


def cmd_apply(args):
    recs = load_boxes(args.boxes)
    dec = D.collect(args.batch or [])
    out, why = [], Counter()
    for lid, r in sorted(recs.items()):
        ok, how = accept(r, dec.get(lid))
        why[how] += 1
        out.append({"id": lid, "crop": r["crop"], "reviewed": r["reviewed"], "flags": r["why"], "edge_ok": ok,
                    "edge_why": how, "review": (dec.get(lid) or {}).get("decision")})
    Path(args.out).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out), encoding="utf-8")
    print(json.dumps({"lines": len(out), "kept": sum(x["edge_ok"] for x in out), "by": dict(why)}), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reads", default=str(P.OUT / "reads" / "reads.jsonl"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("boxes")
    p.add_argument("--labels", default=str(P.OUT / "labels.jsonl"))
    p.add_argument("--evidence", nargs="*", help="batch dirs of later looks (gradcheck, blind)")
    p.add_argument("--out", default=str(EDGE_DIR / "boxes.jsonl"))
    p = sub.add_parser("sheets")
    p.add_argument("--boxes", default=str(EDGE_DIR / "boxes.jsonl"))
    p.add_argument("--out", required=True)
    p.add_argument("--chunk", type=int, default=0)
    p = sub.add_parser("apply")
    p.add_argument("--boxes", default=str(EDGE_DIR / "boxes.jsonl"))
    p.add_argument("--batch", nargs="*")
    p.add_argument("--out", default=str(EDGE_DIR / "edge.jsonl"))
    args = ap.parse_args()
    {"boxes": cmd_boxes, "sheets": cmd_sheets, "apply": cmd_apply}[args.cmd](args)


if __name__ == "__main__":
    main()
