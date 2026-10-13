#!/usr/bin/env python3
"""CLANNAD long run: Claude references, then v1 / v2 / v3 and Google Lens scored against them.

  clannad_refs.py lines   --session DIR --work WORK
  clannad_refs.py engines --work WORK [--procs 5]          (teacher/.venv; run it offline: unshare -rn)
  clannad_refs.py arms    --work WORK --blobs DIR [--compiler gcc]
  clannad_refs.py sheets  --work WORK --round r1|r2|r3 --out DIR [--lines-per-batch 320]
  clannad_refs.py collect --work WORK --round r1|r2|r3 --batches DIR...
  clannad_refs.py refs    --work WORK
  clannad_refs.py score   --work WORK --out RESULT.json [--md RESULT.md]

The protocol is eval/clannad_refs_protocol.json, committed before any arm read a CLANNAD crop. The session is
the owner's long run (eval/clannad_longrun.py; eval/vocr3_clannad.py's published pins), shipped in the owner's
Google Drive drop of 2026-10-13. A unit is one line of a successful page: the stored processed line PNG that
Google Lens and the cached FL10 read in the long run.

lines    binds the session (vocr3_clannad.bound_session: pins, hash chains, one comparison per capture) and
         lists every line of the 2,177 successful pages -> WORK/lines.jsonl (private: Lens's and the cached
         FL10's text).
engines  MeikiOCR's recognizer and PP-OCRv6 small (teacher/engines, pinned files) on every line crop ->
         WORK/engines.jsonl: R1's candidates. Neither engine is a compared arm.
arms     every protocol arm through the C runtime, deployment W8I (vocr3_report.CReader on a library built now),
         on every line crop with invert=False -> WORK/arms.jsonl (private) and arms.meta.json (identities,
         resources, native build, the cache check: VOCR1's readings against the long run's cached FL10). No
         score: there are no references yet.
sheets   contact sheets for one round (R1: the engines' readings; R2: R1's decision, the engines, Lens and every
         arm, shuffled and unattributed; R3: R2's candidates plus R2's decision, for the lines R1 and R2 do not
         agree on), chunked into batch directories of about --lines-per-batch lines (README, Sxxx.png/json).
collect  each batch's decisions.jsonl (devnative/decide.parse_decisions; every entry decided) -> WORK/rN.jsonl.
refs     the reference rule (two reads with one agreement key; else excluded) -> WORK/refs.jsonl and the freeze
         record refs.freeze.json (hashes of the references and of every decision file), created exclusively.
score    once, on the frozen references: CER, exact lines and hiragana per arm, for Lens and for the cached FL10;
         95% intervals from whole-page bootstraps; the protocol's pairs (paired page bootstrap); the Lens proxy
         (each arm's disagreement with Lens); the lens-disagree flags. Public JSON holds counts, rates and hashes
         only; per-line rows go to WORK/score.private.jsonl.

Everything under WORK is private (copyrighted CLANNAD crops, readings and references): never commit it, never send
it to Codex or any third party.
"""
import argparse
import collections
import datetime
import hashlib
import json
import random
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "eval", ROOT / "train", ROOT / "synth", ROOT / "runtime"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

PROTOCOL_PATH = ROOT / "eval" / "clannad_refs_protocol.json"
ROUNDS = ("r1", "r2", "r3")
LENS, CACHED = "Lens", "cached"
# notext: no text in the frame (scene art, box edges, a cursor); multi: more than one text line in the frame;
# cut: the frame cuts through a character; ui: menus and system text; bad: anything else unreadable
SKIPS = ("notext", "multi", "cut", "ui", "bad")
ALIASES = {"p": "pick", "t": "text", "u": "unsure", "s": "skip"}
REPS, SEED = 10000, 0
LINES_PER_BATCH = 320
FORBIDDEN = ("text", "pred", "reference_text", "lens", "readings")


def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def write_jsonl(path, rows, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    return sha256_file(path)


def write_json(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=1, sort_keys=True)
        f.write("\n")
    return sha256_file(path)


def protocol(path=PROTOCOL_PATH):
    p = json.loads(Path(path).read_text(encoding="utf-8"))
    if p.get("kind") != "clannad-refs-protocol" or p.get("schema") != 1:
        raise ValueError("not the CLANNAD references protocol")
    return p


def key(s):
    from teacher import vote
    return vote.key(s or "")


def canon(s):
    from teacher import vote
    return vote.canon(s or "")


# ----------------------------------------------------------------------------- lines

def line_id(capture, i):
    return f"p{int(capture['sequence']):05d}-{i}"


def list_lines(session):
    """Every line of every successful page, in capture order, bound to the published session."""
    import vocr3_clannad as VC
    session = Path(session).resolve()
    lock, ok, excluded = VC.bound_session(session, VC.PUBLISHED)
    roi = lock["roi"]
    rows = []
    for capture, result in ok:
        page = f"p{int(capture['sequence']):05d}"
        for i, (entry, res) in enumerate(zip(capture["lines"], result["lines"], strict=True)):
            rows.append({"id": line_id(capture, i), "page": page, "seq": int(capture["sequence"]), "i": i,
                         "event_sha256": capture["event_sha256"], "line": entry, "source": capture["source"],
                         "roi": roi, "box": entry["box"], "dark": bool(entry["dark"]),
                         "lens": res["google"]["text"], "cached": res["vocr"]["text"]})
    return rows, {"pages": len(ok), "excluded_pages": len(excluded), "lines": len(rows), "roi": roi,
                  "lock_sha256": VC.PUBLISHED["lock_sha256"], "captures_sha256": VC.PUBLISHED["captures_sha256"],
                  "comparisons_sha256": VC.PUBLISHED["comparisons_sha256"]}


def cmd_lines(a):
    import clannad_longrun as LR
    rows, meta = list_lines(a.session)
    session = Path(a.session).resolve()
    for r in rows:
        LR.verified_image(session, r["line"])
    for src in {json.dumps(r["source"], sort_keys=True) for r in rows}:
        LR.verified_image(session, json.loads(src))
    work = Path(a.work)
    meta.update(session=str(session), lines_sha256=write_jsonl(work / "lines.jsonl", rows), created=now(),
                verified_images=len(rows) + meta["pages"])
    write_json(work / "lines.meta.json", meta)
    print(json.dumps({k: meta[k] for k in ("pages", "excluded_pages", "lines", "lines_sha256")}))


def load_lines(work):
    work = Path(work)
    meta = json.loads((work / "lines.meta.json").read_text(encoding="utf-8"))
    if sha256_file(work / "lines.jsonl") != meta["lines_sha256"]:
        raise ValueError("lines.jsonl changed since `lines`")
    return read_jsonl(work / "lines.jsonl"), meta


def load_grey(session, entry):
    """A stored line PNG (mode L) after checking its file hash."""
    from PIL import Image
    path = Path(session) / entry["path"]
    if sha256_file(path) != entry["sha256"]:
        raise ValueError(f"{entry['path']}: file hash differs from the capture journal")
    with Image.open(path) as im:
        return np.asarray(im.convert("L")).copy()


# ----------------------------------------------------------------------------- engines (R1's candidates)

_ENG = {}


def _engine_init():
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from teacher import engines
    _ENG.update(engines.load(["meiki", "v6s"], n_threads=1))


def _engine_read(job):
    session, row = job
    g = load_grey(session, row["line"])
    bgr = np.ascontiguousarray(np.repeat(g[:, :, None], 3, axis=2))
    return {"id": row["id"], "meiki": _ENG["meiki"](bgr)[0], "v6s": _ENG["v6s"](bgr)[0]}


def cmd_engines(a):
    import multiprocessing as mp

    from teacher import engines
    rows, meta = load_lines(a.work)
    jobs = [(meta["session"], r) for r in rows]
    with mp.get_context("spawn").Pool(a.procs, initializer=_engine_init) as pool:
        out = pool.map(_engine_read, jobs, chunksize=16)
    pins = {n: {"file": engines.model_path(n).name, "sha256": sha256_file(engines.model_path(n))}
            for n in ("meiki_det", "meiki_rec", "v6s") if engines.model_path(n).is_file()}
    sha = write_jsonl(Path(a.work) / "engines.jsonl", out)
    write_json(Path(a.work) / "engines.meta.json", {"lines": len(out), "engines_sha256": sha, "models": pins,
                                                    "lines_sha256": meta["lines_sha256"], "created": now()})
    print(f"engines: {len(out)} lines -> {sha[:12]}")


# ----------------------------------------------------------------------------- arms (C runtime, W8I)

def protocol_arms(blobs, proto=None):
    import vocr3_report as R
    proto = proto or protocol()
    return [R.Arm({"name": x["name"], "file": x["file"], "sha256": x["sha256"], "finder": x["finder"]}, blobs)
            for x in proto["arms"]]


def cmd_arms(a):
    import vocr3_report as R
    rows, meta = load_lines(a.work)
    work = Path(a.work)
    if a.extra:
        return extra_arms(a, rows, meta)
    if (work / "refs.freeze.json").exists():
        raise SystemExit("references are frozen; arms ran before them and never run again")
    arms = protocol_arms(a.blobs)
    native = R.build_native(work / "native", a.compiler)
    lib = R.load_library(native)
    readers = {x.name: R.CReader(x, lib) for x in arms}
    out, same_cache = [], 0
    for r in rows:
        g = np.ascontiguousarray(load_grey(meta["session"], r["line"]))
        rec = {"id": r["id"]}
        for name, reader in readers.items():
            rec[name] = reader.text(g, False)
        same_cache += rec["VOCR1"] == r["cached"]
        out.append(rec)
    sha = write_jsonl(work / "arms.jsonl", out)
    info = {"lines": len(out), "arms_sha256": sha, "lines_sha256": meta["lines_sha256"],
            "arms": {x.name: x.identity() for x in arms}, "resources": {x.name: R.resources(x, lib) for x in arms},
            "native": R.public_native(native), "forwards": {n: rd.forwards for n, rd in readers.items()},
            "cache_check": {"arm": "VOCR1", "identical_to_cached_fl10": same_cache, "lines": len(out)},
            "code_sha256": R.code_identity(R.CODE + ("eval/clannad_refs.py", "eval/vocr3_clannad.py",
                                                      "eval/clannad_longrun.py")),
            "environment": R.environment(), "created": now()}
    write_json(work / "arms.meta.json", info)
    print(json.dumps({"lines": len(out), "cache_check": info["cache_check"], "arms_sha256": sha}))


SUFFIX = re.compile(r"[a-z0-9][a-z0-9_-]{0,31}")


def parse_extra(text):
    """NAME=FILE#SHA256 (the sha256 is required: a later model is scored only by its pinned bytes)."""
    m = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9_.+-]{0,63})=([^#]+)#([0-9a-f]{64})(?:@(grad|textbox))?", text)
    if not m:
        raise SystemExit(f"--extra {text!r}: want NAME=FILE#SHA256[@finder]")
    return {"name": m.group(1), "file": m.group(2), "sha256": m.group(3), "finder": m.group(4) or "grad"}


def extra_arms(a, rows, meta):
    """Later blobs (for example VOCR4) read the same crops after the references are frozen: they can never be
    an R2 candidate. Writes arms.<suffix>.jsonl / .meta.json exclusively; the protocol's arms are untouched."""
    import vocr3_report as R
    work = Path(a.work)
    into = Path(a.into or work)            # WORK itself, or another agent's own directory (WORK read-only)
    if not (work / "refs.freeze.json").exists():
        raise SystemExit("extra arms run only after refs.freeze.json exists")
    if not a.suffix or not SUFFIX.fullmatch(a.suffix):
        raise SystemExit("--suffix [a-z0-9_-]{1,32} is required with --extra")
    specs = [parse_extra(x) for x in a.extra]
    taken = {x["name"] for x in protocol()["arms"]} | {LENS, CACHED}
    if {s["name"] for s in specs} & taken or len({s["name"] for s in specs}) != len(specs):
        raise SystemExit(f"extra arm names must be unique and differ from {sorted(taken)}")
    arms = [R.Arm(s, a.blobs) for s in specs]
    native = R.build_native(into / f"native.{a.suffix}", a.compiler)
    lib = R.load_library(native)
    readers = {x.name: R.CReader(x, lib) for x in arms}
    out = []
    for r in rows:
        g = np.ascontiguousarray(load_grey(meta["session"], r["line"]))
        out.append({"id": r["id"], **{n: rd.text(g, False) for n, rd in readers.items()}})
    sha = write_jsonl(into / f"arms.{a.suffix}.jsonl", out, exclusive=True)
    write_json(into / f"arms.{a.suffix}.meta.json", {
        "suffix": a.suffix, "lines": len(out), "arms_sha256": sha, "lines_sha256": meta["lines_sha256"],
        "arms": {x.name: x.identity() for x in arms}, "resources": {x.name: R.resources(x, lib) for x in arms},
        "native": R.public_native(native), "forwards": {n: rd.forwards for n, rd in readers.items()},
        "after_freeze": json.loads((work / "refs.freeze.json").read_text(encoding="utf-8"))["refs_sha256"],
        "code_sha256": R.code_identity(R.CODE + ("eval/clannad_refs.py",)), "environment": R.environment(),
        "created": now()}, exclusive=True)
    print(json.dumps({"suffix": a.suffix, "lines": len(out), "arms": [x.name for x in arms], "sha256": sha}))


def load_extra(into, suffix):
    work = Path(into)
    meta = json.loads((work / f"arms.{suffix}.meta.json").read_text(encoding="utf-8"))
    if sha256_file(work / f"arms.{suffix}.jsonl") != meta["arms_sha256"]:
        raise ValueError(f"arms.{suffix}.jsonl changed since it was written")
    return {r["id"]: r for r in read_jsonl(work / f"arms.{suffix}.jsonl")}, meta


def load_arms(work, proto=None):
    work = Path(work)
    meta = json.loads((work / "arms.meta.json").read_text(encoding="utf-8"))
    if sha256_file(work / "arms.jsonl") != meta["arms_sha256"]:
        raise ValueError("arms.jsonl changed since `arms`")
    proto = proto or protocol()
    want = {x["name"]: x["sha256"] for x in proto["arms"]}
    if {n: x["sha256"] for n, x in meta["arms"].items()} != want:
        raise ValueError("arms.meta.json's blobs differ from the protocol's")
    return {r["id"]: r for r in read_jsonl(work / "arms.jsonl")}, meta


# ----------------------------------------------------------------------------- candidates and decisions

def plausible(s):
    """A reading worth showing: non-empty, and at most half of it ASCII letters or digits (the engines'
    and arms' garbage on crops without text, such as 20IOOEI.0, is left off the sheets)."""
    k = key(s)
    return bool(k) and sum(c.isascii() and c.isalnum() for c in k) <= 0.5 * len(k)


def distinct(texts):
    """[{"text": canon, "key": key, "engines": []}] of the plausible readings, first occurrence first."""
    out, seen = [], set()
    for t in texts:
        k = key(t)
        if k and k not in seen and plausible(t):
            seen.add(k)
            out.append({"text": canon(t), "key": k, "engines": []})
    return out


class DecisionError(ValueError):
    pass


def parse_decisions(text, sheets, batch_name):
    """devnative/decide.parse_decisions' format with this test's skip kinds and no shortcut: one record per
    entry {id, sheet, n, decision, text, skip, name, note}; raises DecisionError listing every problem."""
    errs, got = [], {}
    for i, raw in enumerate(text.splitlines(), 1):
        if not raw.strip():
            continue
        try:
            d = json.loads(raw)
        except ValueError as e:
            errs.append(f"line {i}: not JSON ({e})")
            continue
        d = {ALIASES.get(k, k): v for k, v in d.items()}
        sid, n = d.get("sheet"), d.get("n")
        if sid not in sheets or n not in sheets[sid]:
            errs.append(f"line {i}: no entry {sid!r} #{n!r}")
            continue
        entry = sheets[sid][n]
        kinds = [k for k in ("pick", "text", "unsure", "skip") if d.get(k) not in (None, False, "")]
        if len(kinds) != 1:
            errs.append(f"line {i}: {sid} #{n} needs exactly one of pick / text / unsure / skip, got {kinds}")
            continue
        k = kinds[0]
        rec = {"id": entry["id"], "sheet": sid, "n": n, "decision": k, "text": None, "skip": None,
               "name": bool(d.get("name")), "note": d.get("note"), "batch": batch_name, "reviewer_kind": "assistant"}
        if k == "pick":
            cands = {x["c"]: x["text"] for x in entry["candidates"]}
            if d["pick"] not in cands:
                errs.append(f"line {i}: {sid} #{n} has no candidate {d['pick']!r}")
                continue
            rec["text"], rec["pick"] = canon(cands[d["pick"]]), d["pick"]
        elif k == "text":
            rec["text"] = canon(str(d["text"]))
            if not key(rec["text"]):
                errs.append(f"line {i}: {sid} #{n} text is empty after normalisation")
                continue
        elif k == "skip":
            if d["skip"] not in SKIPS:
                errs.append(f"line {i}: {sid} #{n} skip must be one of {list(SKIPS)}")
                continue
            rec["skip"] = d["skip"]
        if entry["id"] in got and got[entry["id"]] != rec:
            errs.append(f"line {i}: {sid} #{n} decided twice, differently")
            continue
        got[entry["id"]] = rec
    missing = [f"{sid} #{n}" for sid, es in sheets.items() for n, e in es.items() if e["id"] not in got]
    if missing:
        errs.append(f"{len(missing)} entries without a decision: {', '.join(missing[:20])}"
                    + (" ..." if len(missing) > 20 else ""))
    if errs:
        raise DecisionError("\n".join(errs))
    return [got[e["id"]] for sid in sorted(sheets) for _, e in sorted(sheets[sid].items())]


def load_batch(d):
    d = Path(d)
    batch = json.loads((d / "batch.json").read_text(encoding="utf-8"))
    sheets = {}
    for p in sorted(d.glob("S*.json")):
        s = json.loads(p.read_text(encoding="utf-8"))
        sheets[s["sheet"]] = {e["n"]: e for e in s["entries"]}
    return batch, sheets


def batch_dirs(dirs):
    out = []
    for d in map(Path, dirs):
        out += [d] if (d / "batch.json").exists() else sorted(x for x in d.glob("b*") if (x / "batch.json").exists())
    return out


def shuffled(cands, seed_text):
    rng = random.Random(int(hashlib.sha256(seed_text.encode()).hexdigest()[:12], 16))
    out = [dict(c) for c in cands]
    rng.shuffle(out)
    return out


def outcome(rec):
    """('text', key, canon) | ('skip', kind) | ('unsure',) of one parsed decision (None: no read)."""
    if rec is None:
        return None
    if rec["decision"] in ("pick", "text"):
        return ("text", key(rec["text"]), canon(rec["text"]))
    if rec["decision"] == "skip":
        return ("skip", rec["skip"])
    return ("unsure",)


def needs_r3(d1, d2):
    a, b = outcome(d1), outcome(d2)
    if a is None or b is None:
        raise ValueError("R1 and R2 must both have decided the line")
    if a[0] == "text" and b[0] == "text":
        return a[1] != b[1]
    return a[0] == "text" or b[0] == "text"        # one text, one non-text: a third read settles it


def decide_ref(d1, d2, d3=None):
    """(status, text, why): status 'ref' (text = the agreed canon form) or 'excluded'."""
    reads = [o for o in (outcome(d1), outcome(d2), outcome(d3)) if o is not None]
    if not needs_r3(d1, d2):
        a = reads[0]
        if a[0] == "text":
            return "ref", a[2], "r1=r2"
        return "excluded", None, "r1,r2 " + "+".join(sorted({o[0] if o[0] != "skip" else "skip:" + o[1]
                                                               for o in reads[:2]}))
    if d3 is None:
        raise ValueError("R1 and R2 disagree: a third read is needed")
    texts = collections.Counter(o[1] for o in reads if o[0] == "text")
    k, n = texts.most_common(1)[0] if texts else (None, 0)
    if n >= 2:
        return "ref", next(o[2] for o in reads if o[0] == "text" and o[1] == k), "two of three"
    if sum(o[0] != "text" for o in reads) >= 2:
        return "excluded", None, "two of three non-text"
    return "excluded", None, "no two reads agree"


def r1_candidates(eng):
    return distinct([eng.get("meiki"), eng.get("v6s")])


def r2_candidates(lid, d1, eng, lens, arm_texts):
    """R1's decision, both engines, Lens and every arm: distinct, shuffled, unattributed."""
    first = [d1["text"]] if d1 and d1.get("text") else []
    return shuffled(distinct(first + [eng.get("meiki"), eng.get("v6s"), lens] + list(arm_texts)), "r2:" + lid)


def r3_candidates(lid, d1, d2, eng, lens, arm_texts):
    extra = [d["text"] for d in (d1, d2) if d and d.get("text")]
    return shuffled(distinct(extra + [eng.get("meiki"), eng.get("v6s"), lens] + list(arm_texts)), "r3:" + lid)


# ----------------------------------------------------------------------------- sheets

README = """# CLANNAD reference batch {name} (round {round}, {n} lines, {sheets} sheets)

PRIVATE, PROTECTED TEST DATA: CLANNAD (PS Vita), the owner's long run. Do not copy these files anywhere, never
send them or any crop of them to Codex or any other service or model, do not run OCR on them, and do not edit
any repository. You only view the sheets here and write decisions.jsonl in this directory.

{intro}

Each sheet `Sxxx.png` (at most 1280 x 1280 px) shows numbered entries. An entry is one text line that the
console's line finder cut from a CLANNAD screenshot: the red frame marks the crop, shown at full brightness;
around it, dimmed, is a little of the screen for context (dimmed pixels are outside the crop). Under the image
are numbered candidate readings; characters that differ from candidate 1 are red. `Sxxx.json` lists the same
entries with each candidate's exact text, so you can copy it. A wide frame over the scene art often holds just
a speaker's name plate: the dialogue line below it, dimmed, is outside the frame (it is its own entry).

Decide what the framed crop says: the characters inside the red frame, left to right. Conventions (the
benchmark's):
- the main text only: never furigana (small ruby above the line), page-cursor icons (▼ ▽ ◆ and the like) or
  button glyphs;
- written as on screen, keeping 「」『』（）, with full-width ！ ？ ～, … for an ellipsis (one … per cell of
  dots), ― for a long dash, no spaces; Latin letters and digits as plain ASCII;
- small kana (ゃ っ ァ), dakuten / handakuten (は ば ぱ), ー versus 一, and similar-looking kanji matter: look at
  each character; never guess a character you cannot see: mark the line unsure instead.

Write `decisions.jsonl` in this directory, one JSON object per entry, for every entry of every sheet (there is no
shortcut: every entry needs its own line):

```
{{"sheet": "S001", "n": 1, "pick": 2}}             candidate 2 is exactly right
{{"sheet": "S001", "n": 2, "text": "「ええと……」"}}    the correct text (when no candidate is exactly right)
{{"sheet": "S001", "n": 3, "unsure": true}}        you cannot settle it from the image
{{"sheet": "S001", "n": 4, "skip": "notext"}}      no text in the frame (scene art, box edges, a cursor)
{{"sheet": "S001", "n": 5, "skip": "multi"}}       more than one text line inside the frame
{{"sheet": "S001", "n": 6, "skip": "cut"}}         the frame cuts through a character (part of a glyph inside,
                                                   part outside)
{{"sheet": "S001", "n": 7, "skip": "ui"}}          menus, buttons, system text
{{"sheet": "S001", "n": 8, "skip": "bad"}}         anything else that is not one readable horizontal line
```

Many frames hold no text at all (the finder also fires on scene art above the text box): skip those as
"notext", whatever the candidates say. Candidates that are mostly Latin letters or digits are not shown, so an
entry may have no candidate. Add `"name": true` when the line is a speaker name (the small plate above the
dialogue, such as 春原). Optional `"note"`. Text that simply continues outside the frame at a gap between
characters is not "cut": write the characters inside the frame. Glyphs of a neighbouring line peeking in at the
frame's top or bottom edge are not part of this line.
Images: view ONE sheet per tool call, each sheet once, at most ~60 images per session. Append the decisions of
each sheet to decisions.jsonl as soon as that sheet is decided, so a fresh session can resume: if decisions.jsonl
already exists, keep it and continue with the first sheet that has no decisions yet.
"""
INTRO = {
    "r1": "Round 1: decide every line. The candidates are two OCR engines' readings of the crop; either, both or "
          "neither may be right.",
    "r2": "Round 2, an independent second read: decide every line again from the image alone. The candidates are "
          "distinct readings from several sources (an earlier read among them), shuffled and unattributed; any of "
          "them, or none, may be right. How many candidates there are says nothing about which one is right.",
    "r3": "Round 3: two earlier reads of each of these lines disagreed. Decide each line from the image alone, "
          "character by character. The candidates are shuffled and unattributed; any of them, or none, may be "
          "right.",
}


def native_box(row):
    """The line's crop box in the native frame: [x0, y0, x1, y1]."""
    rx, ry = row["roi"][0], row["roi"][1]
    x, y, w, h = row["box"]
    return [x + rx, y + ry, x + w + rx, y + h + ry]


DIM = 0.45      # context outside the frame is drawn at this brightness: what is inside the crop stands out


def dimmed(rgb, box):
    """A copy of the frame with everything outside box darkened (the crop itself unchanged)."""
    out = (rgb.astype(np.float32) * DIM).astype(np.uint8)
    x0, y0, x1, y1 = box
    out[y0:y1, x0:x1] = rgb[y0:y1, x0:x1]
    return out


def make_batch(entries, out, rnd, session, name=None):
    """entries: [(line row, candidates)] numbered 1..N. Writes README, Sxxx.png / Sxxx.json, batch.json."""
    import clannad_longrun as LR
    from PIL import Image

    from devnative import sheets as S
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    fonts = S._fonts()
    blocks, cache = [], {}
    for n, (row, cands) in enumerate(entries, 1):
        src = row["source"]["path"]
        if src not in cache:
            cache.clear()
            cache[src] = LR.verified_image(Path(session), row["source"])
        box = native_box(row)
        blocks.append(S.render_entry(dimmed(cache[src], box), {"ink": box}, "vita", n, cands, False, fonts,
                                     frame=box))
    sheets, sizes = S.pack(blocks)
    index = []
    for si, (members, (w, h)) in enumerate(zip(sheets, sizes), 1):
        sid = f"S{si:03d}"
        page = Image.new("RGB", (max(64, w), max(64, h)), (255, 255, 255))
        rows = []
        for bi, x, y in members:
            row, cands = entries[bi]
            page.paste(blocks[bi], (x, y))
            rows.append({"n": bi + 1, "id": row["id"],
                         "candidates": [{"c": i, "text": c["text"]} for i, c in enumerate(cands, 1)]})
            index.append({"sheet": sid, "n": bi + 1, "id": row["id"]})
        if page.width > S.SHEET or page.height > S.SHEET:
            raise ValueError("sheet larger than 1280 px")
        page.save(out / f"{sid}.png", optimize=True)
        (out / f"{sid}.json").write_text(json.dumps({"sheet": sid, "entries": rows}, ensure_ascii=False, indent=1)
                                         + "\n", encoding="utf-8")
    write_json(out / "batch.json", {"round": rnd, "lines": len(entries), "sheets": len(sheets), "index": index})
    (out / "README.md").write_text(README.format(name=name or out.name, round=rnd.upper(), n=len(entries),
                                                 sheets=len(sheets), intro=INTRO[rnd]), encoding="utf-8")
    return len(sheets)


def round_entries(work, rnd):
    """[(line row, candidates)] of one round, in page order."""
    rows, _ = load_lines(work)
    eng = {r["id"]: r for r in read_jsonl(Path(work) / "engines.jsonl")}
    if rnd == "r1":
        return [(r, r1_candidates(eng[r["id"]])) for r in rows]
    arms, _ = load_arms(work)
    names = [x["name"] for x in protocol()["arms"]]
    d1 = load_round(work, "r1")
    if rnd == "r2":
        return [(r, r2_candidates(r["id"], d1[r["id"]], eng[r["id"]], r["lens"],
                                  [arms[r["id"]][n] for n in names])) for r in rows]
    d2 = load_round(work, "r2")
    return [(r, r3_candidates(r["id"], d1[r["id"]], d2[r["id"]], eng[r["id"]], r["lens"],
                              [arms[r["id"]][n] for n in names]))
            for r in rows if needs_r3(d1[r["id"]], d2[r["id"]])]


def chunks(entries, per):
    """Batches of about `per` lines that never split a page."""
    out, cur = [], []
    for e in entries:
        if cur and len(cur) >= per and e[0]["page"] != cur[-1][0]["page"]:
            out.append(cur)
            cur = []
        cur.append(e)
    if cur:
        out.append(cur)
    return out


def cmd_sheets(a):
    work, out = Path(a.work), Path(a.out)
    if (work / "refs.freeze.json").exists():
        raise SystemExit("references are frozen")
    if a.round == "r2" and a.batch:
        return r2_from_batch(work, Path(a.batch), out)
    entries = round_entries(work, a.round)
    _, meta = load_lines(work)
    made = []
    for bi, chunk in enumerate(chunks(entries, a.lines_per_batch), 1):
        d = out / f"b{bi:02d}"
        n = make_batch(chunk, d, a.round, meta["session"], name=f"clannad-{a.round}/{d.name}")
        made.append({"batch": d.name, "lines": len(chunk), "sheets": n})
        print(f"{d}: {len(chunk)} lines, {n} sheets", flush=True)
    write_json(work / f"{a.round}.sheets.json", {"round": a.round, "out": str(out), "batches": made,
                                                 "lines": len(entries), "ids": [e[0]["id"] for e in entries],
                                                 "created": now()})


def r2_from_batch(work, r1_batch, out):
    """R2 for the lines of one finished R1 batch (other sessions read it), registered in r2.sheets.json."""
    batch, sheets = load_batch(r1_batch)
    if batch.get("round") != "r1":
        raise SystemExit(f"{r1_batch} is not an R1 batch")
    d1 = {r["id"]: dict(r, round="r1") for r in parse_decisions(
        (r1_batch / "decisions.jsonl").read_text(encoding="utf-8"), sheets, r1_batch.name)}
    rows, meta = load_lines(work)
    eng = {r["id"]: r for r in read_jsonl(work / "engines.jsonl")}
    arms, _ = load_arms(work)
    names = [x["name"] for x in protocol()["arms"]]
    ids = [x["id"] for x in batch["index"]]
    by = {r["id"]: r for r in rows}
    entries = [(by[i], r2_candidates(i, d1[i], eng[i], by[i]["lens"], [arms[i][n] for n in names])) for i in ids]
    n = make_batch(entries, out, "r2", meta["session"], name=f"clannad-r2/{out.name}")
    reg_path = work / "r2.sheets.json"
    reg = json.loads(reg_path.read_text(encoding="utf-8")) if reg_path.exists() else \
        {"round": "r2", "batches": [], "ids": [], "lines": 0}
    if set(ids) & set(reg["ids"]):
        raise SystemExit("some of these lines already have an R2 batch")
    reg["batches"].append({"batch": out.name, "dir": str(out), "from": str(r1_batch), "lines": len(ids), "sheets": n})
    reg["ids"] += ids
    reg["lines"] = len(reg["ids"])
    reg["updated"] = now()
    write_json(reg_path, reg)
    print(f"{out}: {len(ids)} lines, {n} sheets (from {r1_batch.name})")


# ----------------------------------------------------------------------------- collect

def cmd_collect(a):
    work = Path(a.work)
    want = json.loads((work / f"{a.round}.sheets.json").read_text(encoding="utf-8"))
    got, files = {}, {}
    for d in batch_dirs(a.batches):
        batch, sheets = load_batch(d)
        if batch.get("round") != a.round:
            raise SystemExit(f"{d}: a {batch.get('round')} batch, not {a.round}")
        text = (d / "decisions.jsonl").read_text(encoding="utf-8")
        files[d.name] = hashlib.sha256(text.encode("utf-8")).hexdigest()
        for r in parse_decisions(text, sheets, d.name):
            if r["id"] in got:
                raise SystemExit(f"{r['id']} decided in two batches")
            r["round"] = a.round
            got[r["id"]] = r
    missing = [i for i in want["ids"] if i not in got]
    extra = sorted(set(got) - set(want["ids"]))
    if missing or extra:
        raise SystemExit(f"{len(missing)} undecided lines (first {missing[:5]}), {len(extra)} unexpected")
    sha = write_jsonl(work / f"{a.round}.jsonl", [got[i] for i in want["ids"]])
    write_json(work / f"{a.round}.collect.json", {"round": a.round, "lines": len(got), "sha256": sha,
                                                  "decision_files_sha256": files, "created": now()})
    c = collections.Counter(r["decision"] for r in got.values())
    print(f"{a.round}: {len(got)} decisions {dict(c)} -> {sha[:12]}")


def load_round(work, rnd):
    work = Path(work)
    meta = json.loads((work / f"{rnd}.collect.json").read_text(encoding="utf-8"))
    if sha256_file(work / f"{rnd}.jsonl") != meta["sha256"]:
        raise ValueError(f"{rnd}.jsonl changed since `collect`")
    return {r["id"]: r for r in read_jsonl(work / f"{rnd}.jsonl")}


# ----------------------------------------------------------------------------- references

def build_refs(rows, d1, d2, d3):
    out = []
    for r in rows:
        a, b, c = d1[r["id"]], d2[r["id"]], d3.get(r["id"])
        status, text, why = decide_ref(a, b, c)
        reads = [x for x in (a, b, c) if x is not None]
        ref = {"id": r["id"], "page": r["page"], "status": status, "text": text, "why": why,
               "name": any(x.get("name") for x in reads if outcome(x)[0] == "text" and key(x["text"]) == key(text))
               if text else False, "dark": r["dark"],
               "reads": [{"round": x["round"], "decision": x["decision"], "skip": x.get("skip"),
                          "key_matches_ref": bool(text) and outcome(x)[0] == "text" and key(x["text"]) == key(text)}
                         for x in reads]}
        if status == "ref":
            ref["lens_disagree"] = key(r["lens"]) != key(text)
            ref["r1_matches_lens"] = outcome(a)[0] == "text" and key(a["text"]) == key(r["lens"])
        ref["notext"] = status == "excluded" and sum(outcome(x) == ("skip", "notext") for x in reads) >= 2
        out.append(ref)
    return out


def cmd_refs(a):
    work = Path(a.work)
    rows, meta = load_lines(work)
    d1, d2 = load_round(work, "r1"), load_round(work, "r2")
    d3 = load_round(work, "r3") if (work / "r3.collect.json").exists() else {}
    refs = build_refs(rows, d1, d2, d3)
    sha = write_jsonl(work / "refs.jsonl", refs, exclusive=True)
    rounds = {rnd: json.loads((work / f"{rnd}.collect.json").read_text(encoding="utf-8"))
              for rnd in ROUNDS if (work / f"{rnd}.collect.json").exists()}
    freeze = {"refs_sha256": sha, "lines_sha256": meta["lines_sha256"], "rounds": rounds,
              "protocol_sha256": sha256_file(PROTOCOL_PATH),
              "counts": dict(collections.Counter(r["why"] for r in refs)),
              "scored_lines": sum(r["status"] == "ref" for r in refs), "frozen": now()}
    write_json(work / "refs.freeze.json", freeze, exclusive=True)
    print(json.dumps({k: freeze[k] for k in ("refs_sha256", "counts", "scored_lines")}, ensure_ascii=False))


# ----------------------------------------------------------------------------- scoring

def line_unit(ref, pred):
    """(errors, chars, exact, hiragana errors, hiragana chars) of one line."""
    import metrics
    t, p = metrics.norm(ref), metrics.norm(pred or "")
    h = metrics.score_hiragana([{"text": ref, "pred": pred or ""}])
    return metrics.lev(t, p), len(t), int(t == p), h["errors"], h["chars"]


def page_sums(units, pages):
    """{page: [errors, chars, exact, lines, hira errors, hira chars]} from per-line units."""
    out = collections.defaultdict(lambda: [0, 0, 0, 0, 0, 0])
    for u, pg in zip(units, pages):
        s = out[pg]
        s[0] += u[0]
        s[1] += u[1]
        s[2] += u[2]
        s[3] += 1
        s[4] += u[3]
        s[5] += u[4]
    return dict(out)


def boot_idx(n, reps=REPS, seed=SEED):
    return np.random.default_rng(seed).integers(0, n, (reps, n))


def rate_ci(num, den, idx):
    """95% percentile interval of sum(num)/sum(den) over resampled units (rows of idx)."""
    num, den = np.asarray(num, np.float64), np.asarray(den, np.float64)
    tot = den[idx].sum(1)
    ok = tot > 0
    r = num[idx].sum(1)[ok] / tot[ok]
    lo, hi = np.percentile(r, [2.5, 97.5])
    return [round(100 * float(lo), 4), round(100 * float(hi), 4)]


def diff_ci(num_a, num_b, den, idx):
    num_a, num_b, den = (np.asarray(x, np.float64) for x in (num_a, num_b, den))
    tot = den[idx].sum(1)
    ok = tot > 0
    d = (num_a[idx].sum(1)[ok] - num_b[idx].sum(1)[ok]) / tot[ok]
    lo, hi = np.percentile(d, [2.5, 97.5])
    return [round(100 * float(lo), 4), round(100 * float(hi), 4)]


def summary(sums, ids, idx):
    e = [sums[p][0] for p in ids]
    n = [sums[p][1] for p in ids]
    x = [sums[p][2] for p in ids]
    m = [sums[p][3] for p in ids]
    he, hn = sum(sums[p][4] for p in ids), sum(sums[p][5] for p in ids)
    return {"pages": len(ids), "lines": sum(m), "chars": sum(n), "errors": sum(e), "exact_lines": sum(x),
            "cer": round(100 * sum(e) / sum(n), 4) if sum(n) else None, "cer_ci95": rate_ci(e, n, idx),
            "exact": round(100 * sum(x) / sum(m), 4) if sum(m) else None, "exact_ci95": rate_ci(x, m, idx),
            "hiragana_errors": he, "hiragana_chars": hn, "hiragana_cer": round(100 * he / hn, 4) if hn else None}


def score_all(rows, refs, arms, names, reps=REPS, seed=SEED, pairs=()):
    """Public scores (counts and rates only) and private per-line rows."""
    import vocr3_report as R
    by = {r["id"]: r for r in rows}
    scored = [x for x in refs if x["status"] == "ref"]
    systems = list(names) + [LENS, CACHED]

    def pred(sysname, lid):
        if sysname == LENS:
            return by[lid]["lens"]
        if sysname == CACHED:
            return by[lid]["cached"]
        return arms[lid][sysname]
    units = {s: [line_unit(x["text"], pred(s, x["id"])) for x in scored] for s in systems}
    pages = [x["page"] for x in scored]
    sums = {s: page_sums(units[s], pages) for s in systems}
    ids = sorted(set(pages))
    idx = boot_idx(len(ids), reps, seed)
    subsets = {"all": lambda x: True, "name": lambda x: x["name"], "not_name": lambda x: not x["name"],
               "inverted": lambda x: x["dark"], "not_inverted": lambda x: not x["dark"]}
    out = {"systems": {}, "pairs": {}, "bootstrap": {"unit": "page", "pages": len(ids), "reps": reps, "seed": seed}}
    for s in systems:
        entry = {"all": summary(sums[s], ids, idx)}
        for name, keep in subsets.items():
            if name == "all":
                continue
            sel = [u for u, x in zip(units[s], scored) if keep(x)]
            sp = [x["page"] for x in scored if keep(x)]
            if sel:
                ss = page_sums(sel, sp)
                sid = sorted(ss)
                entry[name] = summary(ss, sid, boot_idx(len(sid), reps, seed))
        out["systems"][s] = entry
    for cand, base in pairs:
        a = {p: (sums[cand][p][0], sums[cand][p][1]) for p in ids}
        b = {p: (sums[base][p][0], sums[base][p][1]) for p in ids}
        cer = R.paired_bootstrap(a, b, reps, seed)
        ea = [sums[cand][p][2] for p in ids]
        eb = [sums[base][p][2] for p in ids]
        m = [sums[cand][p][3] for p in ids]
        out["pairs"][f"{cand}:{base}"] = {
            "cer_delta": cer["delta"], "cer_ci95": cer["ci95"],
            "exact_delta": round(100 * (sum(ea) - sum(eb)) / sum(m), 4), "exact_ci95": diff_ci(ea, eb, m, idx),
            "candidate_errors": cer["candidate_errors"], "baseline_errors": cer["baseline_errors"],
            "chars": cer["chars"], "pages": cer["units"]}
    private = [{"id": x["id"], "page": x["page"], "ref": x["text"],
                "preds": {s: pred(s, x["id"]) for s in systems},
                "errors": {s: units[s][i][0] for s in systems}} for i, x in enumerate(scored)]
    return out, private


def proxy(rows, arms, names, ids=None):
    """Each system's edit disagreement with Lens (the long run's proxy), on the given lines (default all)."""
    import metrics
    keep = [r for r in rows if ids is None or r["id"] in ids]
    out = {}
    for s in list(names) + [CACHED]:
        e = n = x = 0
        for r in keep:
            t = metrics.norm(r["lens"])
            p = metrics.norm(r["cached"] if s == CACHED else arms[r["id"]][s])
            e += metrics.lev(t, p)
            n += len(t)
            x += t == p
        out[s] = {"lines": len(keep), "reference_chars": n, "edit_disagreements": e,
                  "disagreement_rate": round(100 * e / n, 4) if n else None, "identical_lines": x}
    return out


def notext_output(rows, refs, arms, names):
    """What each system writes on crops two reads found no text in: lines with any output, characters."""
    import metrics
    by = {r["id"]: r for r in rows}
    ids = [x["id"] for x in refs if x.get("notext")]
    out = {}
    for s in list(names) + [LENS, CACHED]:
        lens = [len(metrics.norm(by[i]["lens"] if s == LENS else by[i]["cached"] if s == CACHED else arms[i][s]))
                for i in ids]
        out[s] = {"lines": len(ids), "lines_with_output": sum(n > 0 for n in lens), "output_chars": sum(lens)}
    return out


def flags_summary(refs):
    scored = [x for x in refs if x["status"] == "ref"]
    dis = [x for x in scored if x["lens_disagree"]]
    return {"scored_lines": len(scored), "lens_disagree": len(dis),
            "lens_disagree_r1_had_lens_reading": sum(x["r1_matches_lens"] for x in dis),
            "lens_agree": len(scored) - len(dis),
            "excluded_lines": sum(x["status"] == "excluded" for x in refs),
            "by_rule": dict(collections.Counter(x["why"] for x in refs))}


def assert_public(value, where="result"):
    if isinstance(value, dict):
        for k, v in value.items():
            if k in FORBIDDEN:
                raise ValueError(f"{where}: public output would hold the field {k!r}")
            assert_public(v, f"{where}.{k}")
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            assert_public(v, f"{where}[{i}]")


def cmd_score(a):
    work = Path(a.work)
    proto = protocol()
    freeze = json.loads((work / "refs.freeze.json").read_text(encoding="utf-8"))
    if sha256_file(work / "refs.jsonl") != freeze["refs_sha256"]:
        raise SystemExit("refs.jsonl differs from the freeze record")
    if freeze.get("protocol_sha256", sha256_file(PROTOCOL_PATH)) != sha256_file(PROTOCOL_PATH):
        raise SystemExit(f"{PROTOCOL_PATH.name} differs from the one the references were frozen under")
    tag = f".{a.suffix}" if a.suffix else ""
    if a.suffix and not SUFFIX.fullmatch(a.suffix):
        raise SystemExit("bad --suffix")
    into = Path(a.into or work) if a.suffix else work
    attempt = into / f"score{tag}.attempt.json"
    write_json(attempt, {"started": now(), "refs_sha256": freeze["refs_sha256"]}, exclusive=True)   # once
    rows, meta = load_lines(work)
    arms, ameta = load_arms(work, proto)
    refs = read_jsonl(work / "refs.jsonl")
    names = [x["name"] for x in proto["arms"]]
    pairs = [tuple(p) for p in proto["pairs"]]
    extra_meta = None
    if a.suffix:
        extra, extra_meta = load_extra(into, a.suffix)
        if extra_meta["after_freeze"] != freeze["refs_sha256"]:
            raise SystemExit("the extra arms did not read these frozen references' session")
        arms = {i: {**arms[i], **extra[i]} for i in arms}
        names += list(extra_meta["arms"])
        pairs = [tuple(p.split(":")) for p in a.pair]
        known = set(names) | {LENS, CACHED}
        if not pairs or any(len(p) != 2 or not set(p) <= known for p in pairs):
            raise SystemExit(f"--pair CAND:BASE over {sorted(known)} is required with --suffix")
    scores, private = score_all(rows, refs, arms, names, pairs=pairs)
    scored_ids = {x["id"] for x in refs if x["status"] == "ref"}
    result = {"schema": 1, "kind": "clannad-refs-result", "protocol_sha256": sha256_file(PROTOCOL_PATH),
              "session": {k: meta[k] for k in ("pages", "excluded_pages", "lines", "lock_sha256", "captures_sha256",
                                                "comparisons_sha256")},
              "references": {"freeze": {k: freeze.get(k) for k in ("refs_sha256", "lines_sha256", "protocol_sha256",
                                                                   "counts", "scored_lines", "frozen")},
                             "rounds": {k: {"lines": v["lines"], "sha256": v["sha256"],
                                            "decision_files_sha256": v["decision_files_sha256"]}
                                        for k, v in freeze["rounds"].items()},
                             "agreement": agreement(work), "flags": flags_summary(refs),
                             "reviewer_kind": "assistant"},
              "scores": scores, "notext_output": notext_output(rows, refs, arms, names),
              "lens_proxy": {"scored_lines": proxy(rows, arms, names, scored_ids),
                             "all_lines": proxy(rows, arms, names)},
              "arms": ameta["arms"], "resources": ameta["resources"], "native": ameta["native"],
              "cache_check": ameta["cache_check"], "code_sha256": ameta["code_sha256"],
              "environment": {k: ameta["environment"][k] for k in ("python", "system", "machine", "libc", "packages")},
              "lines_sha256": meta["lines_sha256"], "arms_sha256": ameta["arms_sha256"],
              "training_or_selection_use": False, "created": now()}
    if extra_meta:
        result.update(kind="clannad-refs-extra-result", suffix=a.suffix, pairs_requested=[":".join(p) for p in pairs],
                      extra_arms=extra_meta["arms"], extra_resources=extra_meta["resources"],
                      extra_native=extra_meta["native"], extra_arms_sha256=extra_meta["arms_sha256"],
                      extra_code_sha256=extra_meta["code_sha256"])
    assert_public(result)
    write_jsonl(into / f"score{tag}.private.jsonl", private, exclusive=True)
    write_json(Path(a.out), result, exclusive=True)
    md = markdown(result)
    if a.md:
        Path(a.md).write_text(md + "\n", encoding="utf-8")
    write_json(into / f"score{tag}.done.json", {"finished": now(), "result_sha256": sha256_file(a.out)},
               exclusive=True)
    print(md)


def agreement(work):
    """How often R1 and R2 gave the same key, among lines both read as text."""
    d1, d2 = load_round(work, "r1"), load_round(work, "r2")
    both = [i for i in d1 if outcome(d1[i])[0] == "text" and outcome(d2[i])[0] == "text"]
    same = sum(outcome(d1[i])[1] == outcome(d2[i])[1] for i in both)
    return {"r1_text_r2_text": len(both), "same_key": same, "r1_lines": len(d1),
            "r1_decisions": dict(collections.Counter(r["decision"] for r in d1.values())),
            "r2_decisions": dict(collections.Counter(r["decision"] for r in d2.values()))}


def markdown(result):
    s = result["scores"]["systems"]
    md = ["| system | lines | ref chars | errors | CER % [95% CI] | exact % [95% CI] | hiragana errors / chars |",
          "|---|---:|---:|---:|---|---|---:|"]
    for name, e in s.items():
        a = e["all"]
        md.append(f"| {name} | {a['lines']:,} | {a['chars']:,} | {a['errors']:,} | {a['cer']:.2f} "
                  f"[{a['cer_ci95'][0]:.2f}, {a['cer_ci95'][1]:.2f}] | {a['exact']:.1f} "
                  f"[{a['exact_ci95'][0]:.1f}, {a['exact_ci95'][1]:.1f}] | {a['hiragana_errors']:,} / "
                  f"{a['hiragana_chars']:,} |")
    md += ["", "| pair | CER delta, points [95% CI] | exact delta, points [95% CI] |", "|---|---|---|"]
    for k, p in result["scores"]["pairs"].items():
        md.append(f"| {k} | {p['cer_delta']:+.2f} [{p['cer_ci95'][0]:+.2f}, {p['cer_ci95'][1]:+.2f}] | "
                  f"{p['exact_delta']:+.1f} [{p['exact_ci95'][0]:+.1f}, {p['exact_ci95'][1]:+.1f}] |")
    return "\n".join(md)


def cmd_check(a):
    """A reviewer's self-check: every entry of the batch decided once, in the README's format."""
    batch, sheets = load_batch(a.batch)
    p = Path(a.batch) / "decisions.jsonl"
    try:
        recs = parse_decisions(p.read_text(encoding="utf-8") if p.exists() else "", sheets, Path(a.batch).name)
    except DecisionError as e:
        print("NOT DONE\n" + str(e))
        raise SystemExit(1)
    c = collections.Counter(r["decision"] if r["decision"] != "skip" else "skip:" + r["skip"] for r in recs)
    print(f"OK: {len(recs)} of {batch['lines']} entries decided {dict(c)}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("check")
    p.add_argument("--batch", type=Path, required=True)
    p = sub.add_parser("lines")
    p.add_argument("--session", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    p = sub.add_parser("engines")
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--procs", type=int, default=5)
    p = sub.add_parser("arms")
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--blobs", type=Path, required=True)
    p.add_argument("--compiler", default="gcc")
    p.add_argument("--extra", action="append", default=[],
                   help="NAME=FILE#SHA256[@finder]: a later blob, read after the freeze into arms.<suffix>.jsonl")
    p.add_argument("--suffix", help="with --extra: the output tag (arms.<suffix>.jsonl)")
    p.add_argument("--into", type=Path, help="with --extra: write there instead of WORK (WORK is only read)")
    p = sub.add_parser("sheets")
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--round", choices=ROUNDS, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--lines-per-batch", type=int, default=LINES_PER_BATCH)
    p.add_argument("--batch", type=Path, help="R2 only: build the R2 batch of this finished R1 batch's lines")
    p = sub.add_parser("collect")
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--round", choices=ROUNDS, required=True)
    p.add_argument("--batches", type=Path, nargs="+", required=True)
    p = sub.add_parser("refs")
    p.add_argument("--work", type=Path, required=True)
    p = sub.add_parser("score")
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--md", type=Path)
    p.add_argument("--suffix", help="score the extra arms of arms.<suffix>.jsonl (own attempt and output files)")
    p.add_argument("--pair", action="append", default=[], help="with --suffix: CAND:BASE")
    p.add_argument("--into", type=Path, help="with --suffix: where the extra arms are and the outputs go")
    a = ap.parse_args(argv)
    {"check": cmd_check, "lines": cmd_lines, "engines": cmd_engines, "arms": cmd_arms, "sheets": cmd_sheets,
     "collect": cmd_collect, "refs": cmd_refs, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
