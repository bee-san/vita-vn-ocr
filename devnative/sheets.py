#!/usr/bin/env python3
"""Contact sheets for adjudicating real line crops (pass 1, the independent pass 2, the blind re-check).

  sheets.py make --mode pass1 --out BATCH [--reads R] [--subsets strict,tv_native,scene_vn]
  sheets.py make --mode pass2 --out BATCH --lines ids.json      (ids from decide.py queue)
  sheets.py make --mode blind --out BATCH --lines ids.json
  (gradcheck.py writes "grad" batches: the frame is the GRAD box, the one candidate the label;
   edgecheck.py writes "edge" batches: the frame is the final ink crop, the candidate the label)

A batch directory holds README.md (the task and the exact output format), S001.png ... (sheets
of at most 1280 x 1280 px) and S001.json ... (per entry: its number, line id and numbered
candidate readings, so text can be copied exactly). The reviewer writes decisions.jsonl there.

Each entry shows the line in context: the crop framed in red (crop_box() of the line; on a raw
reads.jsonl line, whose "crop" read.py already padded, that is ink + furigana + 8/4 px: the
review frame every devnative label was decided on), a line
height of the screenshot to its left and right and a third above and below, zoomed by an
integer factor (nearest neighbour, so small kana and dakuten keep their pixels) to about 44 px
of ink. Below it, the distinct readings (agreement key = eval's scoring normalisation; written
in the training-label form, teacher/vote.canon), numbered by support. Characters that differ
from candidate 1 are red. Pass 1 also shows which engines read each candidate
(M MeikiOCR on its own box, m MeikiOCR on the wider view, V PP-OCRv6 small on the view,
v PP-OCRv6 small on the crop). Pass 2 and the blind re-check hide the engines and shuffle the
candidates (seeded by line id); pass 2 adds pass 1's decision as an unmarked candidate when it
differs from every engine.
"""
import argparse
import difflib
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from devnative import lines as L  # noqa: E402
from devnative import paths as P  # noqa: E402
from teacher import vote  # noqa: E402

ENGINES = (("meiki", "M"), ("meikiv", "m"), ("v6s", "V"), ("v6sc", "v"))
FAMILY = {"meiki": "meiki", "meikiv": "meiki", "v6s": "v6s", "v6sc": "v6s"}
SHEET = 1280
FONT = P.SHARED_DATA / "fonts" / "bizudgothic" / "BIZUDGothic-Regular.ttf"
FONT_BOLD = P.SHARED_DATA / "fonts" / "bizudgothic" / "BIZUDGothic-Bold.ttf"
TEXT_PX, HEAD_PX, ROW, GAP = 22, 16, 28, 10
INK_TARGET = 44


# ----------------------------------------------------------------------------- readings

def readings(line):
    """{engine: raw text} of a line's four readings (missing ones left out)."""
    out = {}
    for e, _ in ENGINES:
        r = (line.get("reads") or {}).get(e)
        if r is not None:
            out[e] = r[0] or ""
    return out


def candidates(line):
    """Distinct readings, most supported first: [{"text": canon form, "key": agreement key,
    "engines": [...]}]. Empty readings are left out."""
    groups = {}
    for e, _ in ENGINES:
        t = readings(line).get(e)
        if not t or not vote.key(t):
            continue
        g = groups.setdefault(vote.key(t), {"text": vote.canon(t), "key": vote.key(t), "engines": []})
        g["engines"].append(e)
    order = [e for e, _ in ENGINES]
    return sorted(groups.values(), key=lambda g: (-len(g["engines"]), order.index(g["engines"][0])))


def is_candidate(line):
    """A line worth adjudicating: horizontal (lines_of drops vertical), not cut by the screen
    edge, ink 10-72 px tall, and at least one reading Japanese (kana or kanji, at most half ASCII)."""
    if line.get("edge"):
        return False, "screen-edge"
    if not L.size_ok(line):
        return False, "size"
    if not any(vote.japanese(t) for t in readings(line).values()):
        return False, "not-japanese"
    return True, None


def support(label, line):
    """Engine families (meiki, v6s) with a reading whose key equals the label's."""
    k = vote.key(label)
    return sorted({FAMILY[e] for e, t in readings(line).items() if t and vote.key(t) == k})


# ----------------------------------------------------------------------------- loading

def load_reads(path):
    shots = {}
    for x in Path(path).read_text(encoding="utf-8").splitlines():
        if x.strip():
            r = json.loads(x)
            shots[r["shot"]] = r
    return shots


def line_index(shots):
    """{line id: (shot row, line)} for every line read; id = <shot>_<k>."""
    out = {}
    for s in shots.values():
        for ln in s.get("lines", []):
            out[f"{s['shot']}_{ln['k']}"] = (s, ln)
    return out


def shot_path(s):
    return P.screen_path(s)


# ----------------------------------------------------------------------------- drawing

def _fonts():
    """(text, bold header, small) fonts: BIZ UDGothic from the shared font pool (a universal-design
    face that keeps lookalikes apart); PIL's built-in font where it is missing (layout only)."""
    from PIL import ImageFont
    if not FONT.exists():
        f = ImageFont.load_default()
        return f, f, f
    return (ImageFont.truetype(str(FONT), TEXT_PX), ImageFont.truetype(str(FONT_BOLD), HEAD_PX),
            ImageFont.truetype(str(FONT), 13))


def context_box(line, kind, W, H, frame=None):
    cx0, cy0, cx1, cy1 = frame or L.crop_box(line, kind, W, H)
    h = max(1, line["ink"][3] - line["ink"][1])
    dx, dy = max(4, round(1.0 * h)), max(3, round(0.35 * h))
    return [max(0, cx0 - dx), max(0, cy0 - dy), min(W, cx1 + dx), min(H, cy1 + dy)]


def zoom_of(line, ctx_w):
    h = max(1, line["ink"][3] - line["ink"][1])
    z = int(min(3, max(1, round(INK_TARGET / h))))
    while z > 1 and ctx_w * z > SHEET - 20:
        z -= 1
    return z


def render_entry(rgb, line, kind, n, cands, show_engines, fonts, frame=None):
    """One entry block (PIL RGB image): header, the framed crop in context, the candidates.
    frame: the box to mark (default: the line's ink crop)."""
    from PIL import Image, ImageDraw
    font, bold, small = fonts
    H, W = rgb.shape[:2]
    ctx = context_box(line, kind, W, H, frame)
    if ctx[2] - ctx[0] > SHEET - 20:          # too wide even at 1x: keep the crop, trim the context
        c = frame or L.crop_box(line, kind, W, H)
        ctx = [max(0, c[0] - 4), ctx[1], min(W, c[2] + 4), ctx[3]]
    z = zoom_of(line, ctx[2] - ctx[0])
    patch = Image.fromarray(np.ascontiguousarray(rgb[ctx[1]:ctx[3], ctx[0]:ctx[2]]))
    if patch.width * z > SHEET - 20:
        f = (SHEET - 20) / patch.width
        patch = patch.resize((SHEET - 20, max(1, round(patch.height * f))), Image.LANCZOS)
        z = f
    else:
        patch = patch.resize((patch.width * z, patch.height * z), Image.NEAREST)
    c = frame or L.crop_box(line, kind, W, H)
    d = ImageDraw.Draw(patch)
    fx0, fy0 = (c[0] - ctx[0]) * z - 1, (c[1] - ctx[1]) * z - 1
    fx1, fy1 = (c[2] - ctx[0]) * z, (c[3] - ctx[1]) * z
    d.rectangle([fx0, fy0, fx1, fy1], outline=(255, 30, 30), width=1)
    tag_w = 54 if show_engines else 0
    text_w = max([int(font.getlength(x["text"])) for x in cands] + [60])
    w = max(patch.width, 34 + tag_w + text_w) + 8
    h = HEAD_PX + 6 + patch.height + 4 + ROW * max(1, len(cands)) + GAP
    img = Image.new("RGB", (min(SHEET, w), h), (255, 255, 255))
    dr = ImageDraw.Draw(img)
    head = f"#{n}" + ("   ≠" if len(cands) > 1 and show_engines else "")
    dr.text((2, 1), head, font=bold, fill=(20, 20, 160))
    img.paste(patch, (4, HEAD_PX + 6))
    y = HEAD_PX + 6 + patch.height + 4
    base = cands[0]["text"] if cands else ""
    for i, cnd in enumerate(cands, 1):
        dr.text((4, y + 2), f"{i}", font=bold, fill=(0, 0, 0))
        x = 26
        if show_engines:
            tags = "".join(t if e in cnd["engines"] else "·" for e, t in ENGINES)
            dr.text((x, y + 7), tags, font=small, fill=(110, 110, 110))
            x += tag_w
        _diff_text(dr, (x, y), cnd["text"], base if i > 1 else cnd["text"], font)
        y += ROW
    if not cands:
        dr.text((26, y), "(no reading)", font=small, fill=(120, 120, 120))
    dr.line([(0, h - 2), (img.width, h - 2)], fill=(200, 200, 200))
    return img


def _diff_text(dr, xy, text, base, font):
    """Draw text with the characters that differ from base in red."""
    x, y = xy
    sm = difflib.SequenceMatcher(a=base, b=text, autojunk=False)
    same = [False] * len(text)
    for tag, _, _, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for j in range(j1, j2):
                same[j] = True
    for ch, ok in zip(text, same):
        dr.text((x, y), ch, font=font, fill=(0, 0, 0) if ok else (220, 0, 0))
        x += font.getlength(ch)


def pack(blocks):
    """Shelf-pack entry blocks into sheets of at most SHEET x SHEET px. Returns
    [[(block index, x, y)], ...] and each sheet's size."""
    sheets, cur, sizes = [], [], []
    x = y = row_h = width = 0
    for i, b in enumerate(blocks):
        bw, bh = b.size
        if x and x + bw > SHEET:
            y += row_h
            x = row_h = 0
        if y + bh > SHEET and cur:
            sheets.append(cur)
            sizes.append((width, y + (row_h if x else 0)))
            cur, x, y, row_h, width = [], 0, 0, 0, 0
        cur.append((i, x, y))
        x += bw + 6
        row_h = max(row_h, bh)
        width = max(width, x - 6)
    if cur:
        sheets.append(cur)
        sizes.append((width, y + row_h))
    return sheets, sizes


# ----------------------------------------------------------------------------- batches

README = """# Adjudication batch {name} ({mode}, {n} lines, {sheets} sheets)

{intro}

Each sheet `Sxxx.png` (at most 1280 x 1280 px) shows numbered entries. An entry is one text
line of a real PS Vita / PSP visual-novel screenshot: the red frame marks the crop that gets the
label; around it is a little of the screen for context. Under the image are numbered candidate
readings. Characters that differ from candidate 1 are red.{engines} `Sxxx.json` lists the same
entries with each candidate's exact text, so you can copy it.

Decide what the framed line says. Label conventions (the benchmark's):
- the main text only: never furigana (small ruby above the line), page-cursor icons (▼ ▽ ◆ ...)
  or button glyphs;
- written as on screen, keeping 「」『』（）, with full-width ！ ？ ～, … for an ellipsis (one … per
  cell of dots), ― for a long dash, no spaces; Latin letters and digits as plain ASCII;
- small kana (ゃ っ ァ), dakuten / handakuten (は ば ぱ), ー versus 一, and similar-looking kanji
  matter: zoom in on the image in your mind and check each one;
- never guess a character you cannot see; mark the line unsure instead.

Write `decisions.jsonl` in this directory, one JSON object per entry, for every entry of every sheet:

```
{{"sheet": "S001", "n": 1, "pick": 2}}             candidate 2 is exactly right
{{"sheet": "S001", "n": 2, "text": "「ええと……」"}}    the correct text (when no candidate is exactly right)
{{"sheet": "S001", "n": 3, "unsure": true}}        you cannot settle it from the image
{{"sheet": "S001", "n": 4, "skip": "ui"}}          not dialogue: ui (menus, buttons, system text,
                                                   logos, credits) | cut (the frame cuts the line
                                                   or text runs on outside it) | bad (not one
                                                   readable horizontal text line)
```

Add `"name": true` when the line is a speaker name (name plate or 【name】). Optional `"note"`.
{default}Do not edit anything else. When every entry has a decision, the batch is done.
"""
INTRO = {
    "pass1": "Pass 1: adjudicate every line between the engines' readings.",
    "pass2": "Independent second pass: these lines were decided once already; decide them again from the "
             "image alone. The candidates are shuffled and unattributed; any of them, or none, may be right.",
    "blind": "Blind re-check: decide each line from the image alone. The candidates are shuffled and "
             "unattributed; any of them, or none, may be right.",
    "grad": "GRAD-crop check. Here the red frame is the box the console's line finder cut for the line, and "
            "the one candidate is the line's decided label. Pick 1 when the frame shows exactly that text and no "
            "other readable character (decoration, plates, background art and unreadable slivers of glyphs cut "
            "off at the frame's edge are fine). Otherwise type exactly what the frame shows (for example the "
            "label plus the digits of a card), skip \"cut\" when the frame cuts characters of the label, or mark "
            "unsure.",
    "edge": "Crop-edge check. The red frame is the final crop of the line and the one candidate is its decided "
            "label. Pick 1 only when the frame holds the whole line and exactly the label: every character of the "
            "label fully inside, and no character, bracket or punctuation mark of this line (or of any other text) "
            "partly or fully inside the frame that the label leaves out. Decoration, plate borders, background "
            "art and unreadable slivers of a neighbouring line's glyphs at the frame's top or bottom edge are "
            "fine. Skip \"cut\" when the frame cuts a character of the label or the line visibly continues just "
            "outside the frame (for example its closing 」 or 。 sits right next to the frame); type the text when "
            "the frame shows exactly some other text; unsure when you cannot tell.",
}
DEFAULT = ("A shortcut for entries with a single candidate: {\"sheet\": \"S001\", \"default\": 1} accepts\n"
           "candidate 1 of every entry of that sheet that has exactly one candidate and no explicit line.\n"
           "Entries with two or more candidates always need their own line.\n")


def shuffled(cands, line_id):
    rng = random.Random(int(hashlib.sha256(line_id.encode()).hexdigest()[:12], 16))
    out = [dict(c) for c in cands]
    rng.shuffle(out)
    return out


def make_batch(entries, out, mode, name=None):
    """entries: [(line id, shot row, line, candidates[, frame box])], numbered 1..N in this order
    (numbers are unique in the batch). Writes the batch directory."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fonts = _fonts()
    show = mode == "pass1"
    blocks, cur, rgb = [], None, None
    for n, (lid, s, ln, cands, *frame) in enumerate(entries, 1):
        if s["shot"] != cur:
            cur, rgb = s["shot"], _load(s)
        blocks.append(render_entry(rgb, ln, s["kind"], n, cands, show, fonts, frame[0] if frame else None))
    sheets, sizes = pack(blocks)
    from PIL import Image
    index = []
    for si, (members, (w, h)) in enumerate(zip(sheets, sizes), 1):
        sid = f"S{si:03d}"
        page = Image.new("RGB", (max(64, w), max(64, h)), (255, 255, 255))
        rows = []
        for bi, x, y in members:
            lid, s, ln, cands = entries[bi][:4]
            page.paste(blocks[bi], (x, y))
            row = {"n": bi + 1, "id": lid, "candidates": [{"c": i, "text": c["text"]} for i, c in enumerate(cands, 1)]}
            if show:
                for i, c in enumerate(cands):
                    row["candidates"][i]["engines"] = c["engines"]
            rows.append(row)
            index.append({"sheet": sid, "n": bi + 1, "id": lid})
        assert page.width <= SHEET and page.height <= SHEET
        page.save(out / f"{sid}.png", optimize=True)
        (out / f"{sid}.json").write_text(json.dumps({"sheet": sid, "entries": rows}, ensure_ascii=False, indent=1)
                                         + "\n", encoding="utf-8")
    (out / "batch.json").write_text(json.dumps({"mode": mode, "lines": len(entries), "sheets": len(sheets),
                                                "index": index}, ensure_ascii=False, indent=1) + "\n",
                                    encoding="utf-8")
    eng = (" Each candidate also shows which engines read it (M MeikiOCR on its own box, m MeikiOCR on a wider "
           "view, V PP-OCRv6 small on that view, v PP-OCRv6 small on the crop; · = not that engine).") if show else ""
    (out / "README.md").write_text(README.format(name=name or out.name, mode=mode, n=len(entries), sheets=len(sheets),
                                                 intro=INTRO[mode], engines=eng,
                                                 default=DEFAULT if mode in ("pass1", "grad", "edge") else ""),
                                   encoding="utf-8")
    return len(sheets)


def _load(s):
    return P.load_screen(s)


def pass1_entries(shots, subsets):
    out = []
    for s in sorted(shots.values(), key=lambda s: (s["subset"], int(s["vn"][1:]), s["shot"])):
        if s["subset"] not in subsets:
            continue
        for ln in sorted(s.get("lines", []), key=lambda ln: (ln["ink"][1], ln["ink"][0])):
            ok, _ = is_candidate(ln)
            if ok:
                out.append((f"{s['shot']}_{ln['k']}", s, ln, candidates(ln)))
    return out


def review_entries(shots, ids, extra=None, frames=None):
    """Entries for pass 2 / blind: candidates shuffled; `extra` {id: text} adds a candidate
    (pass 1's decision) when its key is not among them; `frames` {id: box} marks that box
    (e.g. the final crop) instead of the ink crop."""
    idx = line_index(shots)
    out = []
    for lid in ids:
        s, ln = idx[lid]
        cands = [{"text": c["text"], "key": c["key"], "engines": []} for c in candidates(ln)]
        t = (extra or {}).get(lid)
        if t and vote.key(t) not in {c["key"] for c in cands}:
            cands.append({"text": vote.canon(t), "key": vote.key(t), "engines": []})
        e = (lid, s, ln, shuffled(cands, lid))
        out.append(e + (frames[lid],) if frames and lid in frames else e)
    return out


def cmd_make(args):
    shots = load_reads(args.reads)
    if args.mode == "pass1":
        entries = pass1_entries(shots, set(args.subsets.split(",")))
    else:
        spec = json.loads(Path(args.lines).read_text(encoding="utf-8"))
        ids = spec["ids"] if isinstance(spec, dict) else spec
        entries = review_entries(shots, ids, spec.get("extra") if isinstance(spec, dict) else None,
                                 spec.get("frames") if isinstance(spec, dict) else None)
    if args.chunk:
        k = 0
        for i in range(0, len(entries), args.chunk):
            k += 1
            sub = Path(args.out) / f"b{k:02d}"
            ns = make_batch(entries[i:i + args.chunk], sub, args.mode, name=f"{Path(args.out).name}/{sub.name}")
            print(f"{sub}: {len(entries[i:i + args.chunk])} lines, {ns} sheets", flush=True)
    else:
        ns = make_batch(entries, args.out, args.mode)
        print(f"{args.out}: {len(entries)} lines, {ns} sheets", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("make")
    p.add_argument("--mode", choices=("pass1", "pass2", "blind"), required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--reads", default=str(P.OUT / "reads" / "reads.jsonl"))
    p.add_argument("--subsets", default="strict,tv_native,scene_vn")
    p.add_argument("--lines", help="pass2/blind: JSON list of line ids, or {ids, extra, frames}")
    p.add_argument("--chunk", type=int, default=0, help="split into batch sub-directories of this many lines")
    args = ap.parse_args()
    cmd_make(args)


if __name__ == "__main__":
    main()
