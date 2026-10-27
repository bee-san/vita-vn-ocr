#!/usr/bin/env python3
"""VOCR6 stage 4's final test (issue #125; eval/vocr6_final_test_protocol.json): the blobs chosen to ship and VOCR5
beside them, read once on the protected sets (vita-test, psp-test, CLANNAD's frozen references, the hooked test title
PCSG00807), on two paths: (a) VOCR5's locked path as published, (b) Rubi's shipped path at the final read.

  vocr6_final_test.py stage --s BLOB --n BLOB --l BLOB --out MODELS_DIR [--message TEXT]
      copy the three blobs to MODELS_DIR/{S,N,L}.vocr and write models.json (source, sha256, bytes, heads)
  vocr6_final_test.py rubi-path --rubi RUBI --out RUBI_PATH.json
      path (b)'s settings from a Rubi_for_vita checkout of main (LM, its settings, cascade, counter, head gate,
      refinement, finder fix, textnorm), matched against the protocol's candidates; the finder's sources
  vocr6_final_test.py pcsg-a --family NAME --arms ARMS.json --build-cache DIR --cache DIR --out RESULT.json
      the test title's path (a): eval/vocr6_baseline.py build (once) / read / score with the family's arms, and
      the per-unit rows the pairs need (private)
  vocr6_final_test.py plan --set vita-test|psp-test|clannad|pcsg00807 --rubi-path RUBI_PATH.json --out PLAN.pkl
      path (b)'s crops and units of a set (private): fixed lines, regions or pages through Rubi's finder
  vocr6_final_test.py read --plan PLAN.pkl --blob BLOB --rubi-path RUBI_PATH.json --out DUMP.pkl [--workers 5]
      every crop through Rubi's line read (W8I, the blob's heads, counter re-decode, re-read, beam + LM)
  vocr6_final_test.py score-b --plan PLAN.pkl --dump FAMILY:TIER=DUMP ... --rubi-path R --out RESULT.json
      Rubi's cascade, textnorm and line filter; per-unit errors of every family, tier and stage (private)
  vocr6_final_test.py validate --priv PRIV --out VALIDATION.json
      path (a)'s VOCR5 outputs against the published records: every compared entry must be equal
  vocr6_final_test.py report --priv PRIV --out OUT
      every table: OUT/report.json (counts, rates, intervals, hashes; no text) and OUT/report.md

Everything a model reads here is protected or private (copyrighted frames, the owner's text, hand-checked
references): plans, dumps and per-unit rows stay under $HERD/shared/private/vocr6-final-test; the public outputs
hold integer counts, rates, intervals, ids and hashes only (assert_public). Nothing here selects or tunes anything.
"""
import argparse
import datetime
import hashlib
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "eval", ROOT / "train", ROOT / "runtime", ROOT / "synth", ROOT / "data", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from metrics import lev, norm  # noqa: E402

PROTOCOL = ROOT / "eval" / "vocr6_final_test_protocol.json"
HERD = Path(os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr"))
A = HERD / "shared" / "artifacts"
TIERS = ("S", "N", "L")
NEXT = {"S": "N", "N": "L", "L": None}
FAMILIES = ("VOCR5", "VOCR6")
SETS_B = ("vita-test", "psp-test", "clannad", "pcsg00807")
TOPK = 8
VARIANTS = {"base": 0, "expanded-4": 4, "expanded-8": 8}
STAGES = ("greedy", "rubi_tier", "rubi_cascade", "shipped")
FINDER_RECOVERY = 3                       # vocr_lines.h VOCR_LINES_TEXTBOX_RECOVERY (Rubi's VJO_VOCR_FINDER)
REFINE_ALL = 0x1F                         # VOCR_REFINE_ALL (ocr_refine_lines = psp: PSP titles only)
REPS, SEED, MIN_CLUSTERS = 10000, 0, 5
KANA_COUNTER = re.compile("[カケ][月所年国条]")    # Rubi core/vocr_ocr.c kana_counter
COLUMN_FLAGS = 3                          # Rubi VJO_VOCR_COLUMN_FLAGS: VOCR_COLUMNS_VALLEYS | VOCR_COLUMNS_NO_DARK
ROTATE_BYTES = 56 * 1024                  # Rubi VJO_VOCR_ROTATE_BYTES (= VJO_VOCR_STRIP_BYTES)
MAX_LINES = 16                            # Rubi VJO_VOCR_MAX_LINES
FORBIDDEN = {"text", "texts", "pred", "preds", "predictions", "ref", "reference_text", "lens", "readings", "g",
             "b", "used_g"}


# ----------------------------------------------------------------------------- small helpers

def sha256(path, bufsize=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(bufsize), b""):
            h.update(chunk)
    return h.hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def protocol(path=PROTOCOL):
    p = json.loads(Path(path).read_text(encoding="utf-8"))
    if p.get("kind") != "vocr6-final-test-protocol" or p.get("schema") != 1:
        raise SystemExit("not the VOCR6 final-test protocol")
    return p


def check_inputs(proto, name):
    """The protected set's pinned files, checked against the protocol's sha256 before a read: the CLANNAD work
    directory (its references, lines and freeze record) or the test title's pools, page pool and their manifests."""
    s = proto["sets"][name]
    if name == "clannad":
        w = Path(s["work_dir"])
        for f, k in (("refs.jsonl", "refs_sha256"), ("lines.jsonl", "lines_sha256"),
                     ("refs.freeze.json", "refs_freeze_sha256"), ("lines.meta.json", "lines_meta_sha256")):
            if sha256(w / f) != s[k]:
                raise SystemExit(f"CLANNAD {f} differs from the protocol's sha256")
        if sha256(ROOT / "eval" / "clannad_refs_protocol.json") != s["clannad_refs_protocol_sha256"]:
            raise SystemExit("eval/clannad_refs_protocol.json differs from the protocol's sha256")
        return str(w)
    if name == "pcsg00807":
        for path, k in ((Path(s["pools_dir"]) / "manifest.json", "pools_manifest_sha256"),
                        (Path(s["pagepool_dir"]) / "manifest.json", "pagepool_manifest_sha256"),
                        (Path(s["pools_dir"]) / "test" / "line" / "PCSG00807.npz", "clean_npz_sha256"),
                        (Path(s["pagepool_dir"]) / "PCSG00807.jsonl", "pagepool_title_sha256")):
            if sha256(path) != s[k]:
                raise SystemExit(f"{path} differs from the protocol's sha256")
        return s
    if sha256(ROOT / s["manifest"]) != s["manifest_sha256"]:
        raise SystemExit(f"{s['manifest']} differs from the protocol's sha256")
    return s


def write_json(path, doc, exclusive=False, indent=1):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=indent, sort_keys=True)
        f.write("\n")
    return sha256(path)


def write_pickle(path, doc):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(doc, f, protocol=4)
    os.replace(tmp, path)
    return sha256(path)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def once(path, record):
    """An attempt record, created exclusively: a second attempt of the same read fails here."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        f.write(json.dumps(record, indent=1, sort_keys=True) + "\n")


def assert_public(value, where="report"):
    """Public outputs carry counts, rates, ids and hashes: no field that could hold a protected line's text."""
    if isinstance(value, dict):
        for k, v in value.items():
            if k in FORBIDDEN:
                raise ValueError(f"{where}: public output would hold the field {k!r}")
            assert_public(v, f"{where}.{k}")
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            assert_public(v, f"{where}[{i}]")


def git_head():
    r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return r.stdout.strip()


# ----------------------------------------------------------------------------- Rubi's line read (core/vocr_ocr.c)

def line_conf(text, conf_min):
    """core/vocr_ocr.c line_conf: conf_min x 1000 + 0.5 in float, truncated and clamped; 1000 for an empty line."""
    if not text:
        return 1000
    c = np.float32(np.float32(conf_min) * np.float32(1000.0) + np.float32(0.5))
    return 0 if c <= 0 else 1000 if c >= 1000 else int(c)


def kana_counter(t):
    """1 if t holds a full-size カ or ケ right before 月 所 年 国 条 (core/vocr_ocr.c kana_counter)."""
    return bool(KANA_COUNTER.search(t or ""))


def read_again(o, contexts, two, greedy):
    """core/vocr_ocr.c read_again: the run skipped frames (the head gate's, or with the two-level head frames that ran
    its frequent rows only) and its counter re-decode found a counter context, or the two-level head is on and the
    greedy text holds a full-size カ/ケ before a counter character."""
    skipped = o["gated"] > 0 or (two and o["escaped"] + o["gated"] < o["frames"])
    return bool(skipped and (contexts or (two and kana_counter(greedy))))


def rubi_line(rt, lm, img, inv, two, counter):
    """One crop as finish_line reads it with the job's model: the greedy text (kept for the log), the shown text b
    (the LM's over the re-decoded extras of the read used: the ungated re-read when read_again), the line's
    confidence from the read used, and counts. A line read as empty stays empty."""
    o = rt.run_topk(img, TOPK, invert=bool(inv), blank=True)
    g = o["text"]
    out = {"g": g, "frames": int(o["frames"]), "gated": int(o["gated"]), "escaped": int(o["escaped"]), "reread": 0,
           "edits": 0}
    if not g:
        out.update(b="", conf=1000, used_g="")
        return out
    red, st = rt.redecode(o, *counter)
    used = o
    if read_again(o, st["contexts"], two, g):
        u = rt.run_topk(img, TOPK, invert=bool(inv), blank=True, ungated=True)
        out["reread"] = 1
        used = u
        if not u["text"]:
            out.update(b="", conf=1000, used_g="")
            return out
        red, st = rt.redecode(u, *counter)
    out["edits"] = int(st["edits"])
    out["used_g"] = used["text"]
    out["conf"] = line_conf(used["text"], used["conf_min"])
    out["b"] = lm.decode(red["topv"], red["topc"], red["blank"])
    return out


def compose(own, up, casc_conf):
    """The text finish_line leaves for one crop before textnorm: the tier's own reading, or, when its confidence is
    under casc_conf and a next tier exists, that tier's reading of the same crop (empty when its greedy text is)."""
    if not own["g"]:
        return ""
    if up is not None and own["conf"] < casc_conf:
        return up["b"] if up["g"] else ""
    return own["b"]


def has_japanese(t):
    """Rubi core/utf.h vjo_is_japanese_char over a line (the Shell's non_japanese_filter = lines)."""
    return any(0x3040 <= ord(c) <= 0x309F or 0x30A0 <= ord(c) <= 0x30FF or 0x4E00 <= ord(c) <= 0x9FAF for c in t)


def rows_vertical(lines, w, h):
    """Rubi core/vocr_ocr.c vjo_vocr_rows_vertical (API 47, ocr_vertical = auto): whether the row finder's lines
    [(x, y, w, h)] on a w x h region look like vertical text read as rows (integer arithmetic, as the C)."""
    n = len(lines)
    if not n or n > MAX_LINES:
        return False
    narrow = sum(1 for (_, _, lw, lh) in lines if lw <= 4 * lh)
    small = [(x, x + lw) for (x, _, lw, lh) in lines if 2 * lw <= 7 * lh]
    if h >= w and n >= 2 and 10 * narrow >= 6 * n:
        return True
    if n < 5 or 10 * len(small) < 7 * n:
        return False
    m0 = sorted(a for a, _ in small)[len(small) // 2]
    m1 = sorted(b for _, b in small)[len(small) // 2]
    inside = 0
    for (x, _, lw, lh) in lines:
        a, b = max(x, m0), min(x + lw, m1)
        inside += bool(2 * lw <= 7 * lh and b > a and 5 * (b - a) >= 4 * lw)
    return 10 * inside >= 7 * n


def region_columns(grey):
    """A job of columns on a grey region (Rubi recognize with the vertical model): vita-vn-ocr's column finder with
    Rubi's flags (synth/vertical_twin.find_columns, the bit-exact twin of runtime/vocr_vertical.c), right to left;
    a column over ROTATE_BYTES left out; each column's dark flag from its own pixels (vocr_line_dark) and its crop
    turned 90 degrees anticlockwise (vocr_column_rotate). -> ([(box, dark, rotated crop)], skipped)."""
    import vertical_twin as VT
    from common import line_dark
    boxes, _, _ = VT.find_columns(grey, COLUMN_FLAGS)
    out, skipped = [], 0
    for (x, y, w, h) in boxes:
        if w * h > ROTATE_BYTES:
            skipped += 1
            continue
        out.append(((x, y, w, h), int(line_dark(grey, (x, y, w, h))),
                    np.ascontiguousarray(np.rot90(grey[y:y + h, x:x + w], 1))))
    return out, skipped


def blob_heads(path, lib=None):
    """(head gate present, two-level head present) of a blob, as the C runtime sees it."""
    import ctypes as C

    from pyvocr import Vocr
    rt = Vocr(str(path), mode="w8i", max_width=1024, lib=lib)
    two = False
    if hasattr(rt.lib, "vocr_has_two_level"):
        rt.lib.vocr_has_two_level.argtypes = [C.c_void_p]
        rt.lib.vocr_has_two_level.restype = C.c_int
        two = bool(rt.lib.vocr_has_two_level(rt._model))
    return bool(rt.has_gate), two


# ----------------------------------------------------------------------------- stage

def cmd_stage(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "models.json").exists():
        raise SystemExit(f"{out}/models.json exists: the final blobs are staged once")
    doc = {"kind": "vocr6-final-test-models", "staged": now(), "message": a.message or None, "tiers": {}}
    for t, src in (("S", a.s), ("N", a.n), ("L", a.l)):
        src = Path(src).resolve()
        dst = out / f"{t}.vocr"
        shutil.copyfile(src, dst)
        if sha256(src) != sha256(dst):
            raise SystemExit(f"{src}: the copy differs")
        gate, two = blob_heads(dst)
        doc["tiers"][t] = {"source": str(src), "file": src.name, "sha256": sha256(dst), "bytes": dst.stat().st_size,
                           "head_gate": gate, "two_level": two}
    write_json(out / "models.json", doc, exclusive=True)
    print(json.dumps({t: {k: v for k, v in e.items() if k != "source"} for t, e in doc["tiers"].items()}))


# ----------------------------------------------------------------------------- rubi-path

def parse_lm_row(src, name_macro="VJO_VOCR_LM2_NAME"):
    """core/vocr_ocr.c lms[]'s row of an LM tier: (sha256 hex, beta, tier, alpha or None, flags or None)."""
    m = re.search(r"\{\s*" + name_macro + r"\s*,\s*\w+\s*,\s*\{([^}]*)\}\s*,\s*(\d+)\s*,\s*(\d+)\s*"
                  r"(?:,\s*(\d+)\s*,\s*(VOCR_LM_\w+(?:\s*\|\s*VOCR_LM_\w+)*|\d+)\s*)?\}", src)
    if not m:
        raise SystemExit(f"no lms[] row for {name_macro} in core/vocr_ocr.c")
    bs = [int(x, 16) for x in re.findall(r"0x([0-9a-fA-F]{2})", m.group(1))]
    if len(bs) != 32:
        raise SystemExit(f"{name_macro}: the pinned hash is not 32 bytes")
    flags = m.group(5)
    if flags is not None and not flags.isdigit():
        flags = sum({"VOCR_LM_BOS": 1, "VOCR_LM_EOS": 2}[f.strip()] for f in flags.split("|"))
    return bytes(bs).hex(), int(m.group(2)), int(m.group(3)), (int(m.group(4)) if m.group(4) else None), \
        (int(flags) if flags is not None else None)


def parse_lm_opts(src):
    """lm_opts()'s assignments: {field: int or 'lm->field'}."""
    m = re.search(r"static void lm_opts\([^)]*\)\s*\{(.*?)\n\}", src, re.S)
    if not m:
        raise SystemExit("no lm_opts() in core/vocr_ocr.c")
    out = {}
    for k, v in re.findall(r"o->(\w+)\s*=\s*([^;]+);", m.group(1)):
        v = v.strip()
        out[k] = {"VOCR_LM_EOS": 2, "VOCR_LM_BOS": 1, "VJO_VOCR_LM_TOPK": 8}.get(v, int(v) if v.isdigit() else v)
    return out


def rubi_lm(src):
    """What Rubi main's core/vocr_ocr.c pins for the default LM tier (ocr_lm = 1000k, lms[] row 2, lm_opts):
    (sha256, settings)."""
    sha, beta, tier, alpha, flags = parse_lm_row(src)
    opts = parse_lm_opts(src)
    return sha, {"beam": opts.get("beam"), "topk": opts.get("topk"), "gate": opts.get("gate"),
                 "bracket": opts.get("bracket"), "beta": beta if opts.get("beta") == "lm->beta" else opts.get("beta"),
                 "alpha": alpha if opts.get("alpha") == "lm->alpha" else opts.get("alpha"),
                 "flags": flags if opts.get("flags") == "lm->flags" else opts.get("flags")}


def pinned_lm(proto):
    """Amendment 2: path (b)'s LM is pinned (vocr6-lead's ruling, eval/vocr6_protocol.json stage 4 amendment 3):
    the shipped 1 MB game-text LM with its settings, checked against the release's settings.json."""
    pin = proto["paths"]["b"]["lm"]
    c = proto["paths"]["b"]["lm_candidates"][pin["candidate"]]
    if sha256(c["file"]) != c["sha256"]:
        raise SystemExit(f"{c['file']} differs from its pinned sha256")
    st = json.loads(Path(pin["settings_json"]).read_text(encoding="utf-8"))
    if sha256(pin["settings_json"]) != pin["settings_json_sha256"]:
        raise SystemExit(f"{pin['settings_json']} differs from its pinned sha256")
    tier = st["tiers"][pin["settings_tier"]]
    if tier["sha256"] != c["sha256"] or tier["config"] != c["settings"] or tier["cascade_conf"] != 700:
        raise SystemExit("the release's settings.json does not hold the pinned LM's settings")
    return {"candidate": pin["candidate"], "file": c["file"], "sha256": c["sha256"], "settings": dict(c["settings"])}


def resolve_lm(src, proto):
    """The default LM tier's file and settings Rubi ships (ocr_lm = 1000k, lms[] row 2, lm_opts)."""
    sha, beta, tier, alpha, flags = parse_lm_row(src)
    opts = parse_lm_opts(src)
    got = {"beam": opts.get("beam"), "topk": opts.get("topk"), "gate": opts.get("gate"), "bracket": opts.get("bracket"),
           "beta": beta if opts.get("beta") == "lm->beta" else opts.get("beta"),
           "alpha": alpha if opts.get("alpha") == "lm->alpha" else opts.get("alpha"),
           "flags": flags if opts.get("flags") == "lm->flags" else opts.get("flags")}
    for name, c in proto["paths"]["b"]["lm_candidates"].items():
        if c["sha256"] == sha:
            if got != c["settings"]:
                raise SystemExit(f"Rubi's settings for {name} {got} differ from the protocol's {c['settings']}")
            if sha256(c["file"]) != sha:
                raise SystemExit(f"{c['file']} differs from its pinned sha256")
            return {"candidate": name, "file": c["file"], "sha256": sha, "settings": got, "tier": tier}
    raise SystemExit(f"Rubi's 1000k LM {sha[:16]}... is none of the protocol's candidates")


def c_function(src, name):
    """The text of C function `name` in src (from its signature to its closing brace), or None."""
    m = re.search(r"^(?:static\s+)?(?:__attribute__\(\(\w+\)\)\s+)?[\w\s\*]+?\b" + name + r"\s*\([^;{]*\)\s*\{", src,
                  re.M)
    if not m:
        return None
    i, depth = m.end(), 1
    while depth and i < len(src):
        depth += {"{": 1, "}": -1}.get(src[i], 0)
        i += 1
    return src[m.start():i]


def rubi_functions(rubi):
    """sha256 of the Rubi functions path (b) models (core/vocr_ocr.c's line read and probe, the Shell's line
    filter): {name: sha256 or None}."""
    rubi = Path(rubi)
    src = (rubi / "core" / "vocr_ocr.c").read_text(encoding="utf-8")
    out = {}
    for n in ("line_conf", "counter_redecode", "reread_ungated", "read_full", "kana_counter", "read_again",
              "finish_line", "probe_short_lines", "probe_contrast", "recognize", "vjo_vocr_rows_vertical",
              "median_of"):
        f = c_function(src, n)
        out[n] = hashlib.sha256(f.encode()).hexdigest() if f else None
    f = c_function((rubi / "game_ocr" / "main.c").read_text(encoding="utf-8"), "request_policy")
    out["game_ocr/main.c:request_policy"] = hashlib.sha256(f.encode()).hexdigest() if f else None
    f = c_function((rubi / "core" / "textfilter.c").read_text(encoding="utf-8"), "line_has_japanese")
    out["textfilter.c:line_has_japanese"] = hashlib.sha256(f.encode()).hexdigest() if f else None
    m = re.search(r"static inline int vjo_is_japanese_char\(uint32_t cp\)\s*\{.*?\n\}",
                  (rubi / "core" / "utf.h").read_text(encoding="utf-8"), re.S)
    out["utf.h:vjo_is_japanese_char"] = hashlib.sha256(m.group(0).encode()).hexdigest() if m else None
    return out


def rubi_job_defaults(rubi):
    """Amendment 5 (audit F2): the job defaults path (b) assumes, read from a Rubi checkout: {name: (value, ok)}.
    The finder mode, the text-presence gate off, blank-line skipping off, the non-Japanese line filter on (lines),
    a cascade time ceiling (ocr_cascade_ms > 0) and the game worker's cascade threshold left to the core's (0), and
    the line-finder fix default: absent (Rubi does not ship #129's fixes yet) or on (VOCR_FIX_DEFAULT alone)."""
    rubi = Path(rubi)
    src = (rubi / "core" / "vocr_ocr.c").read_text(encoding="utf-8")
    cfg = (rubi / "core" / "config.c").read_text(encoding="utf-8")
    cfh = (rubi / "core" / "config.h").read_text(encoding="utf-8")
    gm = (rubi / "game_ocr" / "main.c").read_text(encoding="utf-8")
    out = {}
    m = re.search(r"#define VJO_VOCR_FINDER (\w+)", src)
    out["finder"] = (m.group(1) if m else None, bool(m) and m.group(1) == "VOCR_LINES_TEXTBOX_RECOVERY")
    gate = re.search(r"c->ocr_text_gate\s*=\s*(\w+)\s*;", cfg)
    gate_v = gate.group(1) if gate else None
    if gate_v == "VJO_TEXT_GATE_DEFAULT":
        gate_v = c_define(cfh, "VJO_TEXT_GATE_DEFAULT")
    out["ocr_text_gate"] = (gate_v, str(gate_v) == "0")
    blank = re.search(r"c->ocr_skip_blank_lines\s*=\s*(\w+)\s*;", cfg)
    out["ocr_skip_blank_lines"] = (blank.group(1) if blank else None, bool(blank) and blank.group(1) == "0")
    filt = re.search(r"c->non_japanese_filter\s*=\s*(\w+)\s*;", cfg)
    out["non_japanese_filter"] = (filt.group(1) if filt else None, bool(filt) and filt.group(1) == "VJO_FILTER_LINES")
    casc = re.search(r"c->ocr_cascade_ms\s*=\s*(\d+)\s*;", cfg)
    out["ocr_cascade_ms"] = (int(casc.group(1)) if casc else None, bool(casc) and int(casc.group(1)) > 0)
    gc = c_define(gm, "VJO_GAME_CASCADE_CONF")
    out["VJO_GAME_CASCADE_CONF"] = (gc, gc == 0)
    lf = c_define(cfh, "VJO_LINE_FIX_DEFAULT")
    mapped = "VJO_GAME_OCR_S_LINE_FIX ? (unsigned)VOCR_FIX_DEFAULT" in gm
    out["line_fix_default"] = (lf, lf is None or (lf == 1 and mapped))
    # API 47 (amendment 6): ocr_vertical's default; absent (before API 47) or off: rows only; auto: the check
    m = re.search(r"#define VJO_VERTICAL_DEFAULT (\w+)", cfh)
    vd = m.group(1) if m else None
    out["ocr_vertical"] = (vd, vd in (None, "VJO_VERTICAL_OFF", "VJO_VERTICAL_AUTO"))
    return out


def c_define(src, name):
    m = re.search(r"#define\s+" + name + r"\s+(\d+)u?\b", src)
    return int(m.group(1)) if m else None


def vocr_pins(cmake):
    """cmake/VocrOCR.cmake: (pinned commit, {runtime file: sha256})."""
    commit = re.search(r"set\(VJO_VOCR_COMMIT ([0-9a-f]{40})\)", cmake)
    block = re.search(r"set\(_vocr_pins(.*?)\)", cmake, re.S)
    pins = dict(re.findall(r"(\S+\.[ch])\s+([0-9a-f]{64})", block.group(1))) if block else {}
    return (commit.group(1) if commit else None), pins


FINDER_FILES = ("eval/pagepath_find.c", "runtime/vocr_lines.c", "runtime/vocr_lines.h", "runtime/vocr_lines_rows.h",
                "runtime/vocr_lines_refine.c", "runtime/vocr_lines_refine.h", "runtime/vocr_lines_fix.c",
                "runtime/vocr_lines_fix.h")


def finder_sources(dest, ref="origin/feat/vocr6-finder-fix"):
    """The finder's sources from one tree: this checkout if it has runtime/vocr_lines_fix.c (#135 merged), else the
    named ref's (#135's head). Copied to dest with their sha256."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    if (ROOT / "runtime" / "vocr_lines_fix.c").exists():
        tree = {"tree": "checkout", "commit": git_head()}
        for f in FINDER_FILES:
            (dest / Path(f).name).write_bytes((ROOT / f).read_bytes())
    else:
        subprocess.run(["git", "fetch", "-q", "origin"], cwd=ROOT, check=False, capture_output=True)
        commit = subprocess.run(["git", "rev-parse", ref], cwd=ROOT, capture_output=True, text=True,
                                check=True).stdout.strip()
        tree = {"tree": ref, "commit": commit}
        for f in FINDER_FILES:
            data = subprocess.run(["git", "show", f"{commit}:{f}"], cwd=ROOT, capture_output=True, check=True).stdout
            (dest / Path(f).name).write_bytes(data)
    tree["files"] = {Path(f).name: sha256(dest / Path(f).name) for f in FINDER_FILES}
    hdr = (dest / "vocr_lines_fix.h").read_text(encoding="utf-8")
    m = re.search(r"#define VOCR_FIX_DEFAULT\s+(\w+)", hdr)
    flags = {k: int(v, 16) for k, v in re.findall(r"#define (VOCR_FIX_\w+)\s+0x([0-9a-fA-F]+)u", hdr)}
    tree["fix_default"] = flags.get(m.group(1)) if m else None
    tree["fix_flags"] = flags
    return tree


def build_finder(src_dir, out_dir, cc="gcc"):
    """eval/pagepath_find.c with -DPAGEPATH_FIX from one source directory (finder_sources)."""
    src_dir, out_dir = Path(src_dir), Path(out_dir)
    srcs = [src_dir / n for n in ("pagepath_find.c", "vocr_lines_refine.c", "vocr_lines_fix.c", "vocr_lines.c")]
    h = hashlib.sha256(cc.encode())
    for n in sorted(p.name for p in src_dir.iterdir() if p.suffix in (".c", ".h")):
        h.update(n.encode() + (src_dir / n).read_bytes())
    exe = out_dir / f"pagepath_find_fix_{h.hexdigest()[:16]}"
    if not exe.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp = exe.with_suffix(".tmp")
        subprocess.run([cc, "-std=c99", "-O2", "-ffp-contract=off", "-Wall", "-Wextra", "-Werror", "-DPAGEPATH_FIX",
                        f"-I{src_dir}", "-o", str(tmp), *map(str, srcs)], check=True)
        os.replace(tmp, exe)
    return exe, {"driver_sha256": sha256(exe), "sources_sha256": h.hexdigest()}


def cmd_rubi_path(a):
    proto = protocol()
    rubi = Path(a.rubi).resolve()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists")
    src = (rubi / "core" / "vocr_ocr.c").read_text(encoding="utf-8")
    hdr = (rubi / "core" / "vocr_ocr.h").read_text(encoding="utf-8")
    cfg = (rubi / "core" / "config.c").read_text(encoding="utf-8")
    cmake = (rubi / "cmake" / "VocrOCR.cmake").read_text(encoding="utf-8")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=rubi, capture_output=True, text=True).stdout.strip()
    lm = pinned_lm(proto)                      # amendment 2: pinned; Rubi main's own pin is recorded beside it
    rsha, rset = rubi_lm(src)
    known = [n for n, c in proto["paths"]["b"]["lm_candidates"].items() if c["sha256"] == rsha]
    lm["rubi_main"] = {"sha256": rsha, "settings": rset, "candidate": known[0] if known else None,
                       "equal_to_pinned": rsha == lm["sha256"] and rset == lm["settings"]}
    if not re.search(r"c->ocr_lm\s*=\s*2\s*;", cfg):
        raise SystemExit("Rubi's default ocr_lm is not tier 2 (1000k)")
    head_default = re.search(r"c->ocr_head_gate\s*=\s*VJO_HEAD_GATE_DEFAULT", cfg) is not None
    conf_h = (rubi / "core" / "config.h").read_text(encoding="utf-8")
    both = re.search(r"#define VJO_HEAD_GATE_DEFAULT \(VJO_HEAD_GATE \| VJO_HEAD_TWO_LEVEL\)", conf_h) is not None
    if not (head_default and both):
        raise SystemExit("Rubi's default ocr_head_gate is not 'on' (the head gate and the two-level head)")
    if not re.search(r"c->ocr_refine_lines\s*=\s*2\s*;", cfg):
        raise SystemExit("Rubi's default ocr_refine_lines is not psp")
    casc = c_define(hdr, "VJO_VOCR_CASCADE_CONF")
    counter = (c_define(hdr, "VJO_VOCR_COUNTER_TAU"), c_define(hdr, "VJO_VOCR_COUNTER_OPEN"))
    if casc != 700 or counter != (6000, 4000):
        raise SystemExit(f"Rubi's cascade {casc} / counter {counter} differ from the protocol's 700 / (6000, 4000)")
    commit, pins = vocr_pins(cmake)
    fsrc = finder_sources(out.parent / "finder_src")
    # amendment 7: every runtime file Rubi pins must be this checkout's (path (b) reads with this runtime/)
    differ = sorted(f for f, h in pins.items() if not (ROOT / "runtime" / f).exists() or
                    sha256(ROOT / "runtime" / f) != h)
    if differ:
        raise SystemExit(f"Rubi pins runtime files {differ} that differ from this checkout's: review and amend")
    rubi_fix = None
    for f in ("vocr_lines_fix.c", "vocr_lines_fix.h"):
        if f in pins and pins[f] != fsrc["files"][f]:
            raise SystemExit(f"Rubi pins {f} {pins[f][:16]}..., the finder source holds {fsrc['files'][f][:16]}...")
    if "vocr_lines_fix.c" in pins:
        m = re.search(r"VOCR_FIX_(?:DEFAULT|COLGAP|JUNK|VALLEY|ALL)", src + cfg)
        rubi_fix = m.group(0) if m else None
    for f in ("vocr_lines.c", "vocr_lines.h", "vocr_lines_refine.c", "vocr_lines_refine.h"):
        if f in pins and pins[f] != fsrc["files"][f]:
            raise SystemExit(f"Rubi pins {f} {pins[f][:16]}..., the finder source holds {fsrc['files'][f][:16]}...")
    fix = fsrc["fix_default"]
    if fix != fsrc["fix_flags"].get("VOCR_FIX_COLGAP"):
        raise SystemExit(f"VOCR_FIX_DEFAULT is {fix}, not COLGAP: the protocol registered COLGAP")
    exe, dmeta = build_finder(out.parent / "finder_src", out.parent / "bin")
    import pagepath as PP
    norm_exe, nmeta = PP.rubi_norm_tool(str(rubi), out.parent / "bin")
    models = {}
    for m in re.finditer(r"\{(VJO_VOCR5_[SNL]_NAME), \w+,\s*\{([^}]*)\}", src):
        models[m.group(1)] = bytes(int(x, 16) for x in re.findall(r"0x([0-9a-fA-F]{2})", m.group(2))).hex()
    want = {f"VJO_VOCR5_{t}_NAME": proto["models"]["VOCR5"]["gated"][t]["sha256"] for t in TIERS}
    doc = {"kind": "vocr6-final-test-rubi-path", "resolved": now(), "rubi_head": head, "lm": lm,
           "cascade_conf": casc, "counter": list(counter), "head_gate_default": "on (gate and two-level head)",
           "refine": "psp", "probe": 1, "finder_mode": FINDER_RECOVERY,
           "finder": {"fix": fix, "fix_name": "VOCR_FIX_COLGAP", "rubi_pins_fix": "vocr_lines_fix.c" in pins,
                      "rubi_fix_symbol": rubi_fix, "source": fsrc, "driver": str(exe), **dmeta},
           "textnorm": {"exe": str(norm_exe), "exe_sha256": sha256(norm_exe), **nmeta,
                        "textnorm_h_sha256": sha256(rubi / "core" / "textnorm.h")},
           "vocr_pin": {"commit": commit, "files": pins}, "vocr5_in_rubi": models,
           "vocr5_in_rubi_matches_protocol": models == want}
    # amendment 5 (audit F2): the job defaults path (b) assumes
    jd = rubi_job_defaults(rubi)
    doc["job_defaults"] = {k: {"value": v, "ok": ok} for k, (v, ok) in jd.items()}
    doc["finder"]["rubi_ships_fix"] = jd["line_fix_default"][0] == 1 and "vocr_lines_fix.c" in pins
    # API 47 (amendment 6): auto reads a region whose rows look vertical as columns with Rubi's vertical model, which
    # must be the protocol's VOCR5-V; the column finder's files must be the checkout's (its twin's)
    doc["vertical_check"] = jd["ocr_vertical"][0] == "VJO_VERTICAL_AUTO"
    if doc["vertical_check"]:
        mv = re.search(r"\{VJO_VOCR5_V_NAME, \w+,\s*\{([^}]*)\}", src)
        vsha = bytes(int(x, 16) for x in re.findall(r"0x([0-9a-fA-F]{2})", mv.group(1))).hex() if mv else None
        doc["vertical_model"] = {"sha256": vsha, "equal_to_protocol": vsha == proto["models"]["VOCR5"]["vertical"]
                                 ["sha256"]}
        for f in ("vocr_vertical.c", "vocr_vertical.h"):
            if f in pins and pins[f] != sha256(ROOT / "runtime" / f):
                raise SystemExit(f"Rubi pins {f} {pins[f][:16]}..., this checkout holds another")
        if not doc["vertical_model"]["equal_to_protocol"] or \
                c_define(src, "VJO_VOCR_COLUMN_FLAGS") not in (None, COLUMN_FLAGS) or \
                "#define VJO_VOCR_COLUMN_FLAGS (VOCR_COLUMNS_VALLEYS | VOCR_COLUMNS_NO_DARK)" not in src:
            raise SystemExit("Rubi's vertical model or column flags differ from what path (b) models")
    bad = sorted(k for k, (v, ok) in jd.items() if not ok)
    if bad:
        raise SystemExit(f"Rubi's job defaults {bad} differ from what path (b) assumes: review and amend")
    # amendment 3: the Rubi functions path (b) models must be the ones rubicheck compared against (else the model of
    # Rubi's line read may be stale: stop before anything is read); VOCR5's pins are recorded (path (b) reads the
    # protocol's gated VOCR5 blobs whatever Rubi pins by then)
    ref = proto["paths"]["b"]["rubi_functions"]["sha256"]
    got = rubi_functions(rubi)
    doc["rubi_functions"] = {"reference": proto["paths"]["b"]["rubi_functions"]["rubi"], "equal": got == ref,
                             "differ": sorted(k for k in set(ref) | set(got) if ref.get(k) != got.get(k))}
    if got != ref:
        raise SystemExit(f"Rubi's {doc['rubi_functions']['differ']} differ from the code rubicheck compared "
                         "path (b) with: review and amend before the read")
    write_json(out, doc, exclusive=True)
    print(json.dumps({"rubi_head": head, "lm": lm["candidate"], "settings": lm["settings"],
                      "rubi_main_lm": lm["rubi_main"]["candidate"], "fix": fix, "finder_tree": fsrc["tree"],
                      "finder_commit": fsrc["commit"][:12]}))


def load_rubi_path(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if d.get("kind") != "vocr6-final-test-rubi-path":
        raise SystemExit(f"{path} is not a rubi-path record")
    if sha256(d["finder"]["driver"]) != d["finder"]["driver_sha256"]:
        raise SystemExit("the finder driver changed since rubi-path")
    if sha256(d["textnorm"]["exe"]) != d["textnorm"]["exe_sha256"]:
        raise SystemExit("the textnorm filter changed since rubi-path")
    return d


# ----------------------------------------------------------------------------- the finder on regions

def find_boxes(exe, imgs, platforms, fix, probe=1, refine=None):
    """Rubi's line path on grey regions (pagepath_find): [(boxes [(x, y, w, h)], dark flags, rc)]. refine None: the
    refinement on PSP titles (ocr_refine_lines = psp); else these flags everywhere (0: the finder's rows alone)."""
    import pagepath as PP
    jobs = [(img, FINDER_RECOVERY, fix, probe, (REFINE_ALL if plat == "psp" else 0) if refine is None else refine)
            for img, plat in zip(imgs, platforms)]
    out = []
    for i in range(0, len(jobs), 256):
        for rc, found, probed, bx in PP.run_driver(exe, jobs[i:i + 256]):
            # a finder error is Rubi's job failing on the region (VJO_E_OCR_REGION): no line, no text; counted
            out.append(([tuple(b[:4]) for b in bx] if rc == 0 else [], [int(b[4]) for b in bx] if rc == 0 else [],
                        {"found": found, "probed": probed, "rc": rc}))
    return out


def job_crops(rp, imgs, platforms):
    """Each region as Rubi's job of rows reads it, with API 47's auto check when Rubi's default is auto
    (rp["vertical_check"]): the finder's rows (no probe, no refinement) checked by rows_vertical; a region they call
    vertical is read as columns (region_columns, by the vertical model), any other through the line path (find_boxes).
    With a finder fix (Rubi API 50, amendment 7), a region at least as tall as wide is checked on the frozen finder's
    rows (fix 0), any other on the fixed finder's (core/vocr_ocr.c recognize: frozen_check).
    -> [(boxes, dark, finder info, [(crop, invert)], columns flag)]."""
    fix = rp["finder"]["fix"]
    found = find_boxes(rp["finder"]["driver"], imgs, platforms, fix)
    rows = [None] * len(imgs)
    if rp.get("vertical_check"):
        tall = [i for i, img in enumerate(imgs) if fix and img.shape[0] >= img.shape[1]]
        wide = [i for i in range(len(imgs)) if i not in set(tall)]
        for idx, f in ((tall, 0), (wide, fix)):
            if idx:
                got = find_boxes(rp["finder"]["driver"], [imgs[i] for i in idx], [platforms[i] for i in idx], f,
                                 probe=0, refine=0)
                for i, g in zip(idx, got):
                    rows[i] = g
    out = []
    for img, (boxes, dark, st), raw in zip(imgs, found, rows):
        if raw is not None and raw[2].get("rc") == 0 and rows_vertical(raw[0], img.shape[1], img.shape[0]):
            cols, skipped = region_columns(img)
            st = dict(st, vertical=True, rotate_skipped=skipped)
            out.append(([list(b) for b, _, _ in cols], [d for _, d, _ in cols], st, [(c, d) for _, d, c in cols],
                        True))
            continue
        out.append(([list(b) for b in boxes], list(dark), st,
                    [(np.ascontiguousarray(img[y:y + h, x:x + w]), d) for (x, y, w, h), d in zip(boxes, dark)],
                    False))
    return out


def expand(box, pad, dims):
    x, y, w, h = box
    x0, y0 = max(0, x - pad), max(0, y - pad)
    return [x0, y0, min(dims[0], x + w + pad) - x0, min(dims[1], y + h + pad) - y0]


def console_frame(rgb, platform):
    """What Rubi reads on a Vita: a Vita frame as it is, a PSP frame 2x bilinear (cv2.INTER_LINEAR), then grey."""
    import cv2

    from common import to_gray
    s = 2 if platform == "psp" else 1
    if s != 1:
        rgb = cv2.resize(rgb, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(rgb), to_gray(rgb), s


# ----------------------------------------------------------------------------- plans (path b)

class Plan:
    def __init__(self, name):
        self.doc = {"kind": "vocr6-final-test-plan", "set": name, "view": "console", "built": now(), "crops": [],
                    "fixed": [], "units": [], "inputs": {}}

    def crop(self, img, inv, column=False):
        self.doc["crops"].append((np.ascontiguousarray(img, np.uint8), bool(inv)))
        if column:
            self.doc.setdefault("column_crops", []).append(len(self.doc["crops"]) - 1)
        return len(self.doc["crops"]) - 1

    def add_units(self, infos, imgs, rp):
        """Region or page units through job_crops; units read as columns are marked and their crops listed."""
        for info, (boxes, dark, st, crops, cols) in zip(infos, job_crops(rp, imgs, [i["platform"] for i in infos])):
            info.update(boxes=boxes, dark=dark, finder=st, columns=cols,
                        crops=[self.crop(c, d, cols) for c, d in crops])
            self.doc["units"].append(info)


def plan_tests(name, rp, plan, data, rows=None, platform=None, image_of=None):
    """vita-test / psp-test: fixed lines (box + pad as eval/sets.py cuts them, x2 from the 2x frame on PSP; invert =
    line_polarity of the grey crop) and whole-dialogue regions (base / +4 / +8, without the regions that hold an
    uncertain line) through Rubi's finder. rows / platform / image_of: a stand-in set (dryrun) instead of the
    manifest's."""
    import sets
    from fetch import shot_path
    from PIL import Image

    from common import line_polarity, to_gray
    platform = platform or sets.SETS[name]
    pad = sets.PAD[platform]
    stand_in = rows is not None
    rows = rows if stand_in else sets.read_manifest(name)
    image_of = image_of or (lambda r: shot_path(data, platform, r["vn"], r["url"]))
    frames = {}

    def frame(r):
        p = image_of(r)
        if p not in frames:
            if sha256(p) != r["sha256"]:
                raise SystemExit(f"{name} {r['id']}: the screenshot differs from the manifest's sha256")
            with Image.open(p) as im:
                rgb = np.asarray(im.convert("RGB"))
            frames[p] = console_frame(rgb, platform) + (rgb.shape[1], rgb.shape[0])
        return frames[p]

    for r in rows:
        if r.get("uncertain"):
            continue
        rgb2, grey2, s, W, H = frame(r)
        x0, y0, x1, y1 = sets.pad_box(r["box"], pad, W, H)
        c = to_gray(np.ascontiguousarray(rgb2[y0 * s:y1 * s, x0 * s:x1 * s]))
        ci = plan.crop(c, line_polarity(c))
        plan.doc["fixed"].append({"id": r["id"], "vn": r["vn"], "title": r.get("title"), "platform": platform,
                                  "image": r["sha256"], "styles": sorted(r.get("styles") or []),
                                  "name": "name" in (r.get("styles") or []), "text": r["text"], "crop": ci})
    by_reg = {}
    for r in rows:
        if r.get("region"):
            by_reg.setdefault((r["sha256"], tuple(r["region"])), []).append(r)
    imgs, infos = [], []
    excluded = 0
    for (img_sha, region), rs in by_reg.items():
        if any(r.get("uncertain") for r in rs):
            excluded += 1
            continue
        rgb2, grey2, s, W, H = frame(rs[0])
        lines = [{"id": r["id"], "text": r["text"], "styles": r.get("styles") or []}
                 for r in sorted(rs, key=lambda r: (r["box"][1], r["box"][0]))]
        for variant, vpad in VARIANTS.items():
            x, y, w, h = expand(list(region), vpad, (W, H))
            imgs.append(np.ascontiguousarray(grey2[y * s:(y + h) * s, x * s:(x + w) * s]))
            infos.append({"uid": f"{name}|{img_sha[:16]}@{','.join(map(str, region))}|{variant}", "kind": "region",
                          "vn": rs[0]["vn"], "title": rs[0].get("title"), "platform": platform, "image": img_sha,
                          "variant": variant, "lines": lines})
    plan.add_units(infos, imgs, rp)
    plan.doc["inputs"] = {"manifest": "stand-in" if stand_in else f"eval/manifests/{name}.jsonl",
                          "manifest_sha256": None if stand_in else sha256(sets.MANIFESTS / f"{name}.jsonl"),
                          "screenshots": len(frames), "regions_excluded_uncertain": excluded}


def page_reference(statuses):
    """A CLANNAD page's reference: 'complete' when every long-run line is a reference or was excluded as holding no
    text (an empty reference); 'no_reference_line' when none is a reference; else 'incomplete_reference'."""
    if not any(x["status"] == "ref" for x in statuses):
        return "no_reference_line"
    if all(x["status"] == "ref" or (x["status"] == "excluded" and x.get("notext")) for x in statuses):
        return "complete"
    return "incomplete_reference"


def plan_clannad(rp, plan, work):
    """CLANNAD: the 3,048 reference line crops (stored PNGs, invert off) and the pages with a complete reference
    (every long-run line a reference or excluded as no text): the ROI of the frame through Rubi's finder."""
    import clannad_refs as C
    from PIL import Image

    from common import to_gray
    work = Path(work)
    freeze = json.loads((work / "refs.freeze.json").read_text(encoding="utf-8"))
    if sha256(work / "refs.jsonl") != freeze["refs_sha256"]:
        raise SystemExit("refs.jsonl differs from its freeze record")
    rows, meta = C.load_lines(work)
    session = Path(meta["session"])
    refs = {x["id"]: x for x in C.read_jsonl(work / "refs.jsonl")}
    for r in rows:
        x = refs[r["id"]]
        if x["status"] != "ref":
            continue
        g = np.ascontiguousarray(C.load_grey(session, r["line"]))
        plan.doc["fixed"].append({"id": r["id"], "vn": "v4", "page": r["page"], "platform": "vita",
                                  "name": bool(x["name"]), "styles": ["name"] if x["name"] else [],
                                  "text": x["text"], "crop": plan.crop(g, False)})
    by_page = defaultdict(list)
    for r in rows:
        by_page[r["page"]].append(r)
    imgs, infos, skipped = [], [], defaultdict(int)
    for page, rs in sorted(by_page.items()):
        st = [refs[r["id"]] for r in rs]
        why = page_reference(st)
        if why != "complete":
            skipped[why] += 1
            continue
        src = rs[0]["source"]
        path = session / src["path"]
        if sha256(path) != src["sha256"]:
            raise SystemExit(f"{src['path']}: the frame differs from the capture journal")
        with Image.open(path) as im:
            rgb = np.asarray(im.convert("RGB"))
        x, y, w, h = rs[0]["roi"]
        imgs.append(np.ascontiguousarray(to_gray(np.ascontiguousarray(rgb[y:y + h, x:x + w]))))
        order = sorted(zip(rs, st), key=lambda p: (p[0]["box"][1], p[0]["box"][0]))
        infos.append({"uid": f"clannad|{page}", "kind": "page", "vn": "v4", "page": page, "platform": "vita",
                      "variant": "base", "lines": [{"id": r["id"], "text": x["text"] if x["status"] == "ref" else "",
                                                    "styles": ["name"] if x.get("name") else []} for r, x in order]})
    plan.add_units(infos, imgs, rp)
    plan.doc["inputs"] = {"refs_sha256": freeze["refs_sha256"], "lines_sha256": meta["lines_sha256"],
                          "pages": len(by_page), "pages_scored": len(infos), "pages_skipped": dict(skipped)}


def plan_pcsg(rp, plan, pools, pagepool, pages, title="PCSG00807", split="test", vn="v18432"):
    """The test title: its clean line crops (invert off) and eval/pagepath.py's hooked units of the test split, cut
    by Rubi's finder and, for the decomposition, by the reference boxes plus pad (pagepath's oracle)."""
    import pagepath as PP

    import import_hooked as IH
    pools, pagepool, pages = Path(pools), Path(pagepool), Path(pages)
    f = pools / split / "line" / f"{title}.npz"
    for i, (m, lab, g) in enumerate(IH.read_npz(f)):
        plan.doc["fixed"].append({"id": m["id"], "vn": vn, "page": m["page"], "platform": m["platform"],
                                  "name": False, "styles": [], "text": lab, "crop": plan.crop(g, False)})
    units, umeta = PP.hooked_units(pagepool, {split}, titles={title})
    frames = PP.Frames(pages, "console")
    imgs, infos = [], []
    checked = set()
    for u, img, s, o in PP.unit_images(units, frames):
        png = pages / u["title"] / "pages" / f"{u['page']}.png"
        if png not in checked:
            if sha256(png) != u["image_sha256"]:
                raise SystemExit(f"{png}: differs from the page pool's sha256")
            checked.add(png)
        u["_h"], u["_w"] = img.shape
        cells, rboxes = PP.ref_geometry(u, s, o)
        pts = PP.ink_points(img, cells, rboxes)
        imgs.append(img)
        infos.append((u, s, o, pts))
    found = job_crops(rp, imgs, [u["platform"] for u, _, _, _ in infos])
    from common import line_dark
    for img, (u, s, o, pts), (boxes, dark, st, crops, cols) in zip(imgs, infos, found):
        base = {"cluster": u["cluster"], "vn": vn, "page": u["page"], "platform": u["platform"],
                "variant": u["variant"], "lines": [{"id": ln["id"], "text": ln["text"], "styles": []}
                                                   for ln in u["lines"]], "points": pts}
        plan.doc["units"].append(dict(base, uid=u["uid"], kind="page", boxes=boxes, dark=dark, finder=st,
                                      columns=cols, crops=[plan.crop(c, d, cols) for c, d in crops]))
        ob = PP.oracle_boxes(u, s, o)
        od = [line_dark(img, b) for b in ob]
        plan.doc["units"].append(dict(base, uid=u["uid"] + "|oracle", kind="oracle", boxes=[list(b) for b in ob],
                                      dark=od, crops=[plan.crop(img[y:y + h, x:x + w], d)
                                                      for (x, y, w, h), d in zip(ob, od)]))
    plan.doc["inputs"] = {"pools_manifest_sha256": sha256(pools / "manifest.json"), "npz_sha256": sha256(f),
                          **umeta, "units": len(units)}


def cmd_plan(a):
    proto = protocol()
    rp = load_rubi_path(a.rubi_path)
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists")
    plan = Plan(a.set)
    t0 = time.time()
    if a.set in ("vita-test", "psp-test"):
        from fetch import data_dir
        plan_tests(a.set, rp, plan, data_dir(a.data))
    elif a.set == "clannad":
        plan_clannad(rp, plan, check_inputs(proto, "clannad"))
    else:
        s = check_inputs(proto, "pcsg00807")
        plan_pcsg(rp, plan, s["pools_dir"], s["pagepool_dir"], s["pages_dir"])
    plan.doc["finder"] = {k: rp["finder"][k] for k in ("fix", "driver_sha256", "sources_sha256")}
    plan.doc["seconds"] = round(time.time() - t0, 1)
    sha = write_pickle(out, plan.doc)
    print(json.dumps({"set": a.set, "crops": len(plan.doc["crops"]), "fixed": len(plan.doc["fixed"]),
                      "units": len(plan.doc["units"]), "sha256": sha[:16], "seconds": plan.doc["seconds"]}))


# ----------------------------------------------------------------------------- read (path b)

_W = {}


def _init(blob, lm, settings, counter):
    from pyvocr import Vocr, VocrLM
    gate, two = blob_heads(blob)
    rt = Vocr(str(blob), mode="w8i", max_width=1024, head_gate="on" if gate else "off", two_level=1 if two else 0)
    _W.update(rt=rt, lm=VocrLM(rt, str(lm), **settings), two=two, counter=tuple(counter))


def _read(job):
    """One crop through rubi_line; a runtime error (pyvocr's VocrError, or an empty crop) is recorded as the crop's
    error: Rubi's job fails on it (core/vocr_ocr.c run_error), so score-b blanks the crop's whole unit."""
    i, img, inv = job
    from pyvocr import VocrError
    try:
        if img.size == 0:
            raise ValueError("empty crop")
        return i, rubi_line(_W["rt"], _W["lm"], img, inv, _W["two"], _W["counter"])
    except (VocrError, ValueError) as e:
        return i, {"g": "", "b": "", "conf": 1000, "used_g": "", "frames": 0, "gated": 0, "escaped": 0, "reread": 0,
                   "edits": 0, "error": type(e).__name__}


def cmd_read(a):
    rp = load_rubi_path(a.rubi_path)
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists")
    with open(a.plan, "rb") as f:
        plan = pickle.load(f)
    lm, settings = rp["lm"]["file"], dict(rp["lm"]["settings"])
    if sha256(lm) != rp["lm"]["sha256"]:
        raise SystemExit("the LM differs from rubi-path's")
    gate, two = blob_heads(a.blob)
    only = getattr(a, "only", "rows")
    cols = set(plan.get("column_crops") or [])
    jobs = [(i, img, inv) for i, (img, inv) in enumerate(plan["crops"])
            if only == "all" or (only == "columns") == (i in cols)]
    res = [None] * len(plan["crops"])
    t0 = time.time()
    with get_context("fork").Pool(a.workers, initializer=_init,
                                  initargs=(str(a.blob), lm, settings, rp["counter"])) as pool:
        for i, r in pool.imap_unordered(_read, jobs, chunksize=16):
            res[i] = r
    doc = {"kind": "vocr6-final-test-dump", "plan_sha256": sha256(a.plan), "blob": str(a.blob),
           "blob_sha256": sha256(a.blob), "head_gate": gate, "two_level": two, "lm_sha256": rp["lm"]["sha256"],
           "lm_settings": settings, "counter": rp["counter"],
           "library_sha256": sha256(ROOT / "runtime/build/libvocr.so"),
           "repo_head": git_head(), "crops": res, "only": only, "seconds": round(time.time() - t0, 1), "at": now()}
    write_pickle(out, doc)
    print(json.dumps({"crops": len(jobs), "head_gate": gate, "two_level": two,
                      "rereads": sum(r["reread"] for r in res if r),
                      "errors": sum(1 for r in res if r and r.get("error")), "seconds": doc["seconds"]}))


# ----------------------------------------------------------------------------- score (path b)

class Norm:
    """Rubi's core/textnorm.c over many lines in one process (eval/pagepath.py rubi_norm_tool)."""

    def __init__(self, exe):
        self.exe = str(exe)

    def __call__(self, texts):
        import pagepath as PP
        return PP.textnorm(self.exe, list(texts))


def stage_texts(dumps, family, tier, casc_conf, cnorm, columns=()):
    """{stage: [text per crop]} of one family's tier: greedy, rubi_tier, rubi_cascade, shipped; and the crops whose
    read failed (the job tier's, or the cascade tier's for a crop it reads again). columns: the crops of regions
    auto read as columns, which the vertical model reads (dumps[("V", "V")]) for every family and tier, with no
    cascade (a job of columns has none)."""
    own = dumps[(family, tier)]["crops"]
    nt = NEXT[tier]
    up = dumps[(family, nt)]["crops"] if nt else None
    cols = set(columns)
    vert = dumps[("V", "V")]["crops"] if cols else None
    g, rt, rc, failed, casc_failed = [], [], [], set(), 0
    for i, c in enumerate(own):
        if i in cols:
            v = vert[i]
            t = v["b"] if v["g"] else ""
            g.append(v["g"])
            rt.append(t)
            rc.append(t)
            if v.get("error"):
                failed.add(i)
            continue
        g.append(c["g"])
        rt.append(c["b"] if c["g"] else "")
        # a cascade read that failed leaves the job tier's reading (finish_line: the primary's reading stays; Rubi then
        # turns the cascade off for the rest of that job, which is not modelled: it never happened in any read so far)
        u = up[i] if up is not None and not up[i].get("error") else None
        rc.append(compose(c, u, casc_conf))
        if c.get("error"):
            failed.add(i)
        if up is not None and c["g"] and c["conf"] < casc_conf and up[i].get("error"):
            casc_failed += 1
    sh = [t if has_japanese(t) else "" for t in cnorm(rc)]
    return {"greedy": g, "rubi_tier": rt, "rubi_cascade": rc, "shipped": sh, "failed": failed,
            "cascade_failed": casc_failed}


def unit_score(u, texts, failed=()):
    """(dialogue errors, dialogue chars, all errors, all chars) of a region or page: transcript_score's body and all.
    A unit with a failed crop shows nothing (Rubi's job fails)."""
    from general_vn_textbox_guard import transcript_score
    preds = [] if any(c in failed for c in u["crops"]) else [texts[c] for c in u["crops"]]
    sc = transcript_score([{"text": ln["text"], "styles": ln.get("styles") or []} for ln in u["lines"]], preds)
    return sc["body"]["full"]["errors"], sc["body"]["full"]["chars"], sc["all"]["full"]["errors"], \
        sc["all"]["full"]["chars"]


def cmd_score_b(a):
    rp = load_rubi_path(a.rubi_path)
    out = Path(a.out)
    with open(a.plan, "rb") as f:
        plan = pickle.load(f)
    psha = sha256(a.plan)
    dumps = {}
    for spec in a.dump:
        key, _, path = spec.partition("=")
        fam, tier = key.split(":")
        with open(path, "rb") as f:
            d = pickle.load(f)
        if d["plan_sha256"] != psha:
            raise SystemExit(f"{path} was read from another plan")
        dumps[(fam, tier)] = d
    fams = sorted({f for f, _ in dumps if f != "V"})
    cols = plan.get("column_crops") or []
    if cols and ("V", "V") not in dumps:
        raise SystemExit("the plan holds regions auto read as columns: the vertical model's dump (V:V=) is needed")
    cnorm = Norm(rp["textnorm"]["exe"])
    import pagepath as PP
    rows = {"fixed": [], "units": []}
    texts = {(f, t): stage_texts(dumps, f, t, rp["cascade_conf"], cnorm, cols) for f in fams for t in TIERS}
    for fx in plan["fixed"]:
        ref = norm(fx["text"])
        r = {k: fx.get(k) for k in ("id", "vn", "title", "page", "platform", "image", "name", "styles")}
        r["chars"] = len(ref)
        # the line filter's floor (audit F4): a reference line with no Japanese character is deleted by Rubi's
        # non-Japanese line filter however well it is read
        r["nojp_chars"] = len(ref) if ref and not has_japanese(fx["text"]) else 0
        r["errors"] = {f"{f}:{t}:{s}": lev("" if fx["crop"] in texts[(f, t)]["failed"] else
                                           norm(texts[(f, t)][s][fx["crop"]]), ref)
                       for f in fams for t in TIERS for s in STAGES}
        rows["fixed"].append(r)
    for u in plan["units"]:
        r = {k: u.get(k) for k in ("uid", "kind", "vn", "title", "page", "platform", "image", "variant", "cluster")}
        r["columns"] = bool(u.get("columns"))
        r["lines"] = len(u["lines"])
        r["boxes"] = len(u["crops"])
        nojp = [len(norm(ln["text"])) for ln in u["lines"] if "name" not in (ln.get("styles") or []) and
                norm(ln["text"]) and not has_japanese(ln["text"])]
        r["nojp_lines"], r["nojp_chars"] = len(nojp), sum(nojp)
        r["errors"], r["errors_all"] = {}, {}
        for f in fams:
            for t in TIERS:
                for s in STAGES:
                    e, n, ea, na = unit_score(u, texts[(f, t)][s], texts[(f, t)]["failed"])
                    r["errors"][f"{f}:{t}:{s}"] = e
                    r["errors_all"][f"{f}:{t}:{s}"] = ea
                    r["chars"], r["chars_all"] = n, na
        if "points" in u:
            status, kinds, _ = PP.line_stats([tuple(b) for b in u["boxes"]], u["points"])
            r["line_stats"] = {k: status.count(k) for k in ("whole", "cut", "missed", "noink")}
            r["line_stats"].update({k: kinds.count(k) for k in ("merged", "junk")})
        rows["units"].append(r)
    doc = {"kind": "vocr6-final-test-b-rows", "set": plan["set"], "plan_sha256": psha,
           "dumps": {f"{f}:{t}": {k: d[k] for k in ("blob_sha256", "head_gate", "two_level", "lm_sha256",
                                                     "lm_settings", "library_sha256", "seconds")}
                     for (f, t), d in dumps.items()},
           "rereads": {f"{f}:{t}": sum(c["reread"] for c in d["crops"] if c) for (f, t), d in dumps.items()},
           "read_errors": {f"{f}:{t}": sum(1 for c in d["crops"] if c and c.get("error"))
                           for (f, t), d in dumps.items()},
           "vertical_auto": {"units": sum(1 for u in plan["units"] if u["kind"] != "oracle"),
                             "columns": sum(1 for u in plan["units"] if u.get("columns")),
                             "column_crops": len(cols),
                             "rotate_skipped": sum((u.get("finder") or {}).get("rotate_skipped", 0)
                                                   for u in plan["units"]),
                             "check": bool(rp.get("vertical_check"))},
           "cascade_read_errors": {f"{f}:{t}": texts[(f, t)]["cascade_failed"] for f in fams for t in TIERS},
           "finder_errors": sum(1 for u in plan["units"] if (u.get("finder") or {}).get("rc")),
           "cascaded": {f"{f}:{t}": sum(1 for i, c in enumerate(dumps[(f, t)]["crops"])
                                        if c and i not in set(cols) and c["g"] and c["conf"] < rp["cascade_conf"])
                        for f in fams for t in TIERS if NEXT[t]},
           "rubi_path": {k: rp[k] for k in ("rubi_head", "lm", "cascade_conf", "counter")},
           "finder": plan["finder"], "inputs": plan["inputs"], "rows": rows}
    write_pickle(out, doc)
    print(json.dumps({"set": plan["set"], "fixed": len(rows["fixed"]), "units": len(rows["units"])}))


# ----------------------------------------------------------------------------- pcsg-a (the test title's path a)

def pcsg_arms(family, proto, models=None):
    """eval/vocr6_baseline.py ARMS for a family: VOCR5 = its own (the released gated arms); VOCR6 = the staged blobs
    with their own heads (the gate on when present; the two-level head on when present); VOCR5-V in both."""
    import vocr6_baseline as VB
    if family == "VOCR5":
        arms = {t: dict(VB.ARMS[t]) for t in VB.ARMS}
        for t in TIERS:
            if arms[t]["sha256"] != proto["models"]["VOCR5"]["gated"][t]["sha256"]:
                raise SystemExit(f"vocr6_baseline's {t} arm is not the protocol's gated VOCR5 blob")
        return arms
    m = json.loads((Path(models) / "models.json").read_text(encoding="utf-8"))
    arms = {"V": dict(VB.ARMS["V"])}
    for t in TIERS:
        e = m["tiers"][t]
        arms[t] = {"blob": Path(models) / f"{t}.vocr", "gate": e["head_gate"], "two": e["two_level"],
                   "sha256": e["sha256"], "release": f"VOCR6 {t}: {e['file']} (staged)"}
    return arms


def pcsg_rows(VB, cache, lib):
    """Per-unit rows of the test set's reads: clean dialogue lines, the finder's units (mode, scale, variant) and the
    textspot3 frames, each with its page and errors per tier and decode (eval/vocr6_baseline.py's own scoring)."""
    import textspot_metrics as M
    d = Path(cache) / "test"
    reads = {t: r for t in TIERS + ("V",) if (r := VB.load_reads(d, t)) is not None}
    dec = VB.decodes(reads, lib)
    units = json.loads((d / "units.json").read_text(encoding="utf-8"))
    vtexts = dict(zip(reads["V"]["keys"], reads["V"]["greedy"])) if "V" in reads else {}
    from general_vn_textbox_guard import transcript_score
    out = {"clean": [], "finder": [], "spot": []}
    for u in units["clean"]:
        if u["pool"] != "line":
            continue
        out["clean"].append({"key": u["key"], "page": u["page"], "chars": len(u["ref"]),
                             "errors": {f"{t}:{x}": lev(norm(dec[t][x][u["key"]]), u["ref"])
                                        for t in TIERS for x in VB.DECODES if x in dec[t]}})
    for u in units["finder"]:
        e, n = {}, None
        for t in TIERS:
            for x in VB.DECODES:
                if x not in dec[t]:
                    continue
                sc = transcript_score([{"text": s, "styles": []} for s in u["refs"]],
                                      [norm(dec[t][x][k]) for k in u["keys"]])["all"]["full"]
                e[f"{t}:{x}"], n = sc["errors"], sc["chars"]
        out["finder"].append({"mode": u["mode"], "scale": u["scale"], "variant": u["variant"], "page": u["page"],
                              "platform": u["platform"], "gi": u["gi"], "chars": n, "errors": e})
    for u in units["spot"]:
        e, n = {}, None
        for t in TIERS:
            for x in VB.DECODES:
                if x not in dec[t]:
                    continue
                dets = [{"box": list(b[:4]), "vertical": bool(v), "text": (vtexts.get(k, "") if v else
                                                                             dec[t][x][k]) or ""}
                        for (k, v), b in zip(u["keys"], u["boxes"])]
                er = M.e2e_frame(dets, u["refs"], u["width"], u["height"])["dialogue"]
                e[f"{t}:{x}"], n = er[0], er[1]
        out["spot"].append({"frame": u["frame"], "page": u["page"], "chars": n, "errors": e})
    return out


def cmd_pcsg_a(a):
    import vocr6_baseline as VB
    proto = protocol()
    s = check_inputs(proto, "pcsg00807")
    pools, pagepool, pages = s["pools_dir"], s["pagepool_dir"], s["pages_dir"]
    build = Path(a.build_cache)
    if not (build / "test" / "units.json").exists():
        VB.cmd_build(argparse.Namespace(set="test", pools=pools, pagepool=pagepool, pages=pages, cache=str(build),
                                        procs=a.procs))
    cache = Path(a.cache)
    (cache / "test").mkdir(parents=True, exist_ok=True)
    for f in ("crops.pkl", "units.json"):
        dst = cache / "test" / f
        if not dst.exists():
            os.link(build / "test" / f, dst)
    VB.ARMS = pcsg_arms(a.family, proto, a.models)
    for t in TIERS + ("V",):
        if not (cache / "test" / f"reads_{t}.json").exists():
            VB.cmd_read(argparse.Namespace(set="test", tier=t, cache=str(cache), procs=a.procs))
    VB.cmd_score(argparse.Namespace(cache=str(cache), sets="test", pools=pools, pagepool=pagepool, out=a.out))
    from pyvocr import load_lib
    rows = pcsg_rows(VB, cache, load_lib())
    write_pickle(Path(a.out).with_suffix(".rows.pkl"), {"family": a.family, "rows": rows,
                                                        "arms": {t: VB.ARMS[t]["sha256"] for t in VB.ARMS}})
    print(json.dumps({"family": a.family, "clean": len(rows["clean"]), "finder": len(rows["finder"]),
                      "spot": len(rows["spot"])}))


# ----------------------------------------------------------------------------- statistics

def boot_rate(errors, chars, clusters, seed=SEED, reps=REPS):
    """CER (%) with a cluster bootstrap 95% interval (eval/vocr6_baseline.py boot_ratio: sorted cluster ids,
    default_rng(seed).integers, linear 2.5 / 97.5 percentiles)."""
    import vocr6_baseline as VB
    r = VB.boot_ratio(errors, chars, clusters, seed=seed, n=reps)
    return {"cer": r["cer"], "errors": r["errors"], "chars": r["chars"], "clusters": r["clusters"], "ci95": r["ci95"]}


def boot_pair(err_a, err_b, chars, clusters, ids, min_clusters=MIN_CLUSTERS):
    """A - B in points over the same units with whole clusters resampled (eval/vocr3_report.py cluster_bootstrap),
    and the unit-level interval (paired_bootstrap)."""
    import vocr3_report as R
    a = {i: (int(e), int(n)) for i, e, n in zip(ids, err_a, chars)}
    b = {i: (int(e), int(n)) for i, e, n in zip(ids, err_b, chars)}
    cl = dict(zip(ids, clusters))
    units = R.paired_bootstrap(a, b, REPS, SEED)
    clus = R.cluster_bootstrap(a, b, cl, REPS, SEED, min_clusters)
    return {"delta": units["delta"], "ci95": clus.get("ci95"), "clusters": clus["clusters"],
            "unit_ci95": units["ci95"], "units": units["units"], "chars": units["chars"],
            "errors": [units["candidate_errors"], units["baseline_errors"]]}


def table(rows, keys, chars_key="chars", err_key="errors", cluster=lambda r: r["vn"], ident=lambda r: r["id"],
          exact=False, min_clusters=MIN_CLUSTERS):
    """{key: rate} and {tier-pair: pair} over rows (each with err_key {key: errors}); keys are FAMILY:TIER[:STAGE]."""
    ch = [r[chars_key] for r in rows]
    cl = [cluster(r) for r in rows]
    ids = [ident(r) for r in rows]
    rates = {}
    for k in keys:
        e = [r[err_key][k] for r in rows]
        rates[k] = boot_rate(e, ch, cl)
        rates[k]["lines"] = len(rows)
        if exact:
            rates[k]["exact"] = sum(1 for x in e if x == 0)
    pairs = {}
    for k in keys:
        if not k.startswith("VOCR6:"):
            continue
        base = "VOCR5:" + k.split(":", 1)[1]
        if base in keys:
            pairs[k.split(":", 1)[1]] = boot_pair([r[err_key][k] for r in rows], [r[err_key][base] for r in rows],
                                                  ch, cl, ids, min_clusters)
    return {"rates": rates, "pairs": pairs, "units": len(rows), "chars": int(sum(ch)), "clusters": len(set(cl))}


# ----------------------------------------------------------------------------- path (a) inputs

def a_fixed_rows(priv, families):
    """vita-test / psp-test fixed lines of path (a): vocr3_report's per-line records (console polarity, greedy) and
    the LM runs (eval/run.py outputs, re-scored on the manifest), keyed FAMILY:TIER:greedy / :lm."""
    import sets
    rep = json.loads((Path(priv) / "a" / "shared" / "report.json").read_text(encoding="utf-8"))["shared"]
    fl = rep["fixed_line"]["console"]
    man = {s: {r["id"]: r for r in sets.read_manifest(s)} for s in ("vita-test", "psp-test")}
    rows = {}
    for s in ("vita-test", "psp-test"):
        for f in families:
            for t in TIERS:
                arm = f"{f}_{t}"
                for x in fl[arm]["lines"][s]:
                    r = rows.setdefault((s, x["id"]), {"id": f"{s}:{x['id']}", "set": s, "vn": x["vn"],
                                                       "image": man[s][x["id"]]["sha256"],
                                                       "title": man[s][x["id"]].get("title"),
                                                       "platform": sets.SETS[s], "name": "name" in x["styles"],
                                                       "chars": x["n"], "errors": {}})
                    if r["chars"] != x["n"]:
                        raise SystemExit("reference lengths differ between arms")
                    r["errors"][f"{f}:{t}:greedy"] = x["errors"]
                lm = Path(priv) / "a" / "lm" / f"{arm}_{s}.json"
                d = json.loads(lm.read_text(encoding="utf-8"))
                for x in d["sets"][s]["lines"]:
                    ref = norm(man[s][x["id"]]["text"])
                    r = rows[(s, x["id"])]
                    if len(ref) != r["chars"]:
                        raise SystemExit("the LM run's reference differs")
                    r["errors"][f"{f}:{t}:lm"] = lev(norm(x.get("pred", "")), ref)
    return list(rows.values())


def a_dialogue_rows(priv, families):
    """Whole dialogue of path (a): vocr3_report's per region-variant units (body = dialogue; all with names)."""
    import sets
    rep = json.loads((Path(priv) / "a" / "shared" / "report.json").read_text(encoding="utf-8"))["shared"]
    units = rep["dialogue"]["units"]
    titles = {r["vn"]: r.get("title") for s in ("vita-test", "psp-test") for r in sets.read_manifest(s)}
    rows = {}
    for f in families:
        for t in TIERS:
            for u in units[f"{f}_{t}"]:
                img = u["unit"].split("|", 1)[1].split("@")[0]
                r = rows.setdefault((u["unit"], u["variant"]), {
                    "id": f"{u['unit']}|{u['variant']}", "set": u["set"], "vn": u["vn"], "platform": u["platform"],
                    "title": titles.get(u["vn"]), "variant": u["variant"], "image": img, "chars": u["body_full_chars"],
                    "chars_all": u["all_full_chars"], "errors": {}, "errors_all": {}})
                r["errors"][f"{f}:{t}:greedy"] = u["body_full_errors"]
                r["errors_all"][f"{f}:{t}:greedy"] = u["all_full_errors"]
    return list(rows.values())


def a_clannad_rows(priv, families, suffix):
    """CLANNAD of path (a): the extra-arms attempt's private per-line errors (dialogue = not a name reference)."""
    work = Path(check_inputs(protocol(), "clannad"))
    refs = {x["id"]: x for x in read_jsonl(work / "refs.jsonl") if x["status"] == "ref"}
    rows = []
    for x in read_jsonl(Path(priv) / "clannad" / f"score.{suffix}.private.jsonl"):
        n = len(norm(x["ref"]))
        rows.append({"id": x["id"], "page": x["page"], "vn": "v4", "platform": "vita",
                     "name": bool(refs[x["id"]]["name"]),
                     "chars": n, "errors": {f"{f}:{t}:greedy": x["errors"][f"{f}_{t}"] for f in families for t in TIERS
                                            if f"{f}_{t}" in x["errors"]},
                     "errors_lens": x["errors"].get("Lens")})
    return rows


def a_pcsg_rows(priv, families):
    """The test title's path (a): pcsg-a's per-unit rows, keyed FAMILY:TIER:DECODE."""
    out = {}
    for f in families:
        with open(Path(priv) / "a" / "pcsg" / f"{f}.rows.pkl", "rb") as fh:
            d = pickle.load(fh)["rows"]
        for part, rs in d.items():
            for i, r in enumerate(rs):
                key = r.get("key") or r.get("frame") or (r["mode"], r["scale"], r["variant"], r["page"], r["gi"])
                o = out.setdefault((part, key), {"part": part, "id": str(key), "page": r["page"],
                                                 "vn": "v18432", "chars": r["chars"], "errors": {},
                                                 **{k: r[k] for k in ("mode", "scale", "variant") if k in r}})
                if o["chars"] != r["chars"]:
                    raise SystemExit("the test title's references differ between families")
                for k, e in r["errors"].items():
                    o["errors"][f"{f}:{k}"] = e
    return list(out.values())


# ----------------------------------------------------------------------------- validate

def strip_timing(v, drop=("seconds",)):
    """v without its timing fields (and any other keys in drop), at every depth."""
    if isinstance(v, dict):
        return {k: strip_timing(x, drop) for k, x in v.items() if k not in drop}
    if isinstance(v, list):
        return [strip_timing(x, drop) for x in v]
    return v


def first_diff(a, b, where=""):
    if type(a) is not type(b):
        return f"{where}: {type(a).__name__} != {type(b).__name__}"
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                return f"{where}.{k}: missing on one side"
            d = first_diff(a[k], b[k], f"{where}.{k}")
            if d:
                return d
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{where}: lengths {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            d = first_diff(x, y, f"{where}[{i}]")
            if d:
                return d
        return None
    return None if a == b else f"{where}: {a!r} != {b!r}"


def cmd_validate(a):
    import vocr3_report as R
    priv = Path(a.priv)
    checks = []

    def check(name, mine, published):
        d = first_diff(mine, published)
        checks.append({"check": name, "equal": d is None, "first_difference": d})

    pub = json.loads((ROOT / "eval" / "vocr5_report_20261016.json").read_text(encoding="utf-8"))["shared"]
    mine = R.compact(json.loads((priv / "a" / "shared" / "report.json").read_text(encoding="utf-8")))["shared"]
    for t in TIERS:
        arm = f"VOCR5_{t}"
        for pol in ("console", "native"):
            check(f"fixed_line.{pol}.{arm}.summary", mine["fixed_line"][pol][arm]["summary"],
                  pub["fixed_line"][pol][arm]["summary"])
        check(f"dialogue.totals.{arm}", mine["dialogue"]["totals"][arm], pub["dialogue"]["totals"][arm])
        check(f"dialogue.groups.{arm}", mine["dialogue"]["groups"][arm], pub["dialogue"]["groups"][arm])
        check(f"resources.{arm}", mine["resources"][arm], pub["resources"][arm])
    dpub = json.loads((ROOT / "eval" / "vocr5_dialogue_20261016.json").read_text(encoding="utf-8"))["arms"]
    dmine = json.loads((priv / "a" / "shared" / "dialogue.json").read_text(encoding="utf-8"))["arms"]
    for t in TIERS:
        for arm in (f"VOCR5_{t}", f"VOCR5_{t}+LM"):
            check(f"dialogue_tables.console.{arm}", dmine["console"][arm], dpub["console"][arm])
        check(f"dialogue_tables.native.VOCR5_{t}", dmine["native"][f"VOCR5_{t}"], dpub["native"][f"VOCR5_{t}"])
    cpub = json.loads((ROOT / "eval" / "vocr5_clannad_20261016.json").read_text(encoding="utf-8"))["scores"]
    cmine = json.loads((priv / "clannad" / f"clannad_{a.suffix}.json").read_text(encoding="utf-8"))["scores"]
    for t in TIERS:
        check(f"clannad.systems.VOCR5_{t}", cmine["systems"][f"VOCR5_{t}"], cpub["systems"][f"VOCR5_{t}"])
    check("clannad.systems.Lens", cmine["systems"]["Lens"], cpub["systems"]["Lens"])
    bpub = json.loads((ROOT / "eval" / "vocr6_baseline_20261026.json").read_text(encoding="utf-8"))["sets"]["test"]
    bmine = json.loads((priv / "a" / "pcsg" / "VOCR5.json").read_text(encoding="utf-8"))["sets"]["test"]
    # the finder's build identity (the driver and library sha256 of finder_tighten.build, compiled now from this
    # checkout's runtime/) is not a score: recorded beside the checks, not one of them
    ident = ("finder_identity",)
    check("pcsg00807.sets.test", strip_timing(bmine, ("seconds",) + ident), strip_timing(bpub, ("seconds",) + ident))
    info = {"pcsg00807.finder_identity": {"equal": bmine.get("finder_identity") == bpub.get("finder_identity"),
                                          "published": bpub.get("finder_identity"),
                                          "this_read": bmine.get("finder_identity")}}
    doc = {"kind": "vocr6-final-test-validation", "at": now(), "repo_head": git_head(), "checks": checks,
           "identity": info, "all_equal": all(c["equal"] for c in checks)}
    write_json(a.out, doc)
    print(json.dumps({"checks": len(checks), "equal": sum(c["equal"] for c in checks),
                      "differ": [c["check"] for c in checks if not c["equal"]]}))
    if not doc["all_equal"]:
        raise SystemExit(1)


# ----------------------------------------------------------------------------- report

def keys_for(families, stages):
    return [f"{f}:{t}:{s}" for f in families for t in TIERS for s in stages]


def sub(rows, keep):
    return [r for r in rows if keep(r)]


def set_tables(rows, families, stages, cluster, ident, exact=True, split_platform=True, extra=None):
    """dialogue / names tables of fixed lines, per set and pooled."""
    keys = keys_for(families, stages)
    out = {}
    groups = {"pooled": lambda r: True}
    if split_platform:
        groups = {"vita-test": lambda r: r["set"] == "vita-test", "psp-test": lambda r: r["set"] == "psp-test",
                  "pooled": lambda r: True}
    for g, keep in groups.items():
        for part, pk in (("dialogue", lambda r: not r["name"]), ("names", lambda r: r["name"])):
            rs = sub(rows, lambda r: keep(r) and pk(r))
            if rs:
                out[f"{g}|{part}"] = table(rs, keys, cluster=cluster, ident=ident, exact=exact)
    return out


def region_tables(rows, families, stages, cluster):
    keys = keys_for(families, stages)
    out = {}
    for v in VARIANTS:
        for g, keep in (("vita-test", lambda r: r["set"] == "vita-test"),
                        ("psp-test", lambda r: r["set"] == "psp-test"), ("pooled", lambda r: True)):
            rs = sub(rows, lambda r: r["variant"] == v and keep(r))
            if rs:
                out[f"{g}|{v}|dialogue"] = table(rs, keys, cluster=cluster, ident=lambda r: r["id"])
                out[f"{g}|{v}|all"] = table(rs, keys, chars_key="chars_all", err_key="errors_all", cluster=cluster,
                                            ident=lambda r: r["id"])
                nm = [dict(r, chars_names=r["chars_all"] - r["chars"],
                           errors_names={k: r["errors_all"][k] - r["errors"][k] for k in keys}) for r in rs]
                if sum(r["chars_names"] for r in nm):
                    out[f"{g}|{v}|names"] = table(nm, keys, chars_key="chars_names", err_key="errors_names",
                                                  cluster=cluster, ident=lambda r: r["id"])
    return out


def filter_floor(rows, chars_key="chars"):
    """Audit F4: the reference characters on lines with no Japanese character, which Rubi's non-Japanese line filter
    deletes however well they are read; an upper bound on a region or page measure (such a line read in one crop
    with another line is kept), exact on fixed lines. Computed from the references alone."""
    chars = sum(r.get("nojp_chars", 0) for r in rows)
    lines = sum(r.get("nojp_lines", 1 if r.get("nojp_chars") else 0) for r in rows)
    total = sum(r[chars_key] for r in rows)
    return {"lines": lines, "chars": chars, "points": round(100 * chars / total, 4) if total else None}


def per_title(rows, families, stages, image_key="image"):
    """Per VN: dialogue CER per key with screenshot-cluster intervals (no pair intervals: report only)."""
    out = {}
    keys = keys_for(families, stages)
    for vn in sorted({r["vn"] for r in rows}):
        rs = [r for r in rows if r["vn"] == vn and not r.get("name")]
        if not rs:
            continue
        ch = [r["chars"] for r in rs]
        cl = [r[image_key] for r in rs]
        out[vn] = {"set": rs[0].get("set"), "title": rs[0].get("title"), "lines": len(rs), "chars": int(sum(ch)),
                   "screenshots": len(set(cl)), "rates": {k: boot_rate([r["errors"][k] for r in rs], ch, cl)
                                                          for k in keys}}
    return out


def seen_of(proto, vn=None, set_name=None):
    sf = proto["seen_flags"]
    flags = []
    if set_name in ("vita-test", "psp-test") and vn and vn in sf.get(set_name, {}):
        flags.append(sf[set_name][vn])
    return flags


def build_report(priv, out, suffix):
    proto = protocol()
    priv = Path(priv)
    fams = list(FAMILIES)
    rep = {"kind": "vocr6-final-test-report", "schema": 1, "created": now(), "repo_head": git_head(),
           "protocol_sha256": sha256(PROTOCOL), "attempt": json.loads((priv / "attempt.json").read_text()),
           "models": json.loads((priv / "models.json").read_text(encoding="utf-8")),
           "rubi_path": None, "path_a": {}, "path_b": {}, "seen_flags": proto["seen_flags"]}
    # path (a)
    fixed = a_fixed_rows(priv, fams)
    rep["path_a"]["vita_psp_fixed_lines"] = set_tables(fixed, fams, ("greedy", "lm"), cluster=lambda r: r["vn"],
                                                  ident=lambda r: r["id"])
    rep["path_a"]["vita_psp_fixed_lines_per_title"] = per_title(fixed, fams, ("greedy", "lm"))
    dia = a_dialogue_rows(priv, fams)
    rep["path_a"]["vita_psp_whole_dialogue"] = region_tables(dia, fams, ("greedy",), cluster=lambda r: r["vn"])
    rep["path_a"]["vita_psp_whole_dialogue_per_title"] = per_title([r for r in dia if r["variant"] == "base"], fams,
                                                              ("greedy",))
    cl = a_clannad_rows(priv, fams, suffix)
    rep["path_a"]["clannad_lines"] = {
        "dialogue": table(sub(cl, lambda r: not r["name"]), keys_for(fams, ("greedy",)), cluster=lambda r: r["page"],
                          ident=lambda r: r["id"], exact=True, min_clusters=2),
        "names": table(sub(cl, lambda r: r["name"]), keys_for(fams, ("greedy",)), cluster=lambda r: r["page"],
                       ident=lambda r: r["id"], exact=True, min_clusters=2),
        "all": table(cl, keys_for(fams, ("greedy",)), cluster=lambda r: r["page"], ident=lambda r: r["id"],
                     exact=True, min_clusters=2)}
    pc = a_pcsg_rows(priv, fams)
    import vocr6_baseline as VB
    pk = keys_for(fams, VB.DECODES)
    rep["path_a"]["pcsg00807"] = {
        "clean": table(sub(pc, lambda r: r["part"] == "clean"), pk, cluster=lambda r: r["page"],
                       ident=lambda r: r["id"], exact=True, min_clusters=2),
        "page_refine_console_base": table(sub(pc, lambda r: r["part"] == "finder" and r.get("mode") == "refine"
                                              and r.get("variant") == "base"), pk, cluster=lambda r: r["page"],
                                          ident=lambda r: r["id"], min_clusters=2),
        "page_grad_console_base": table(sub(pc, lambda r: r["part"] == "finder" and r.get("mode") == "grad"
                                            and r.get("variant") == "base"), pk, cluster=lambda r: r["page"],
                                        ident=lambda r: r["id"], min_clusters=2),
        "page_textspot3": table(sub(pc, lambda r: r["part"] == "spot"), pk, cluster=lambda r: r["page"],
                                ident=lambda r: r["id"], min_clusters=2)}
    rep["path_a"]["tool_outputs"] = {
        "vocr3_report_pairs": json.loads((priv / "a" / "shared" / "shared.json").read_text(encoding="utf-8"))
        .get("pairs", {}),
        "clannad": json.loads((priv / "clannad" / f"clannad_{suffix}.json").read_text(encoding="utf-8"))["scores"],
        "clannad_dialogue_pairs": json.loads((priv / "clannad" / f"clannad_{suffix}_dialogue.json")
                                             .read_text(encoding="utf-8"))["pairs"],
        "resources": json.loads((priv / "a" / "shared" / "shared.json").read_text(encoding="utf-8"))["resources"]}
    tool = rep["path_a"]["tool_outputs"]
    tool["vocr3_report_pairs"] = R_compact_pairs(tool["vocr3_report_pairs"])
    # path (b)
    bdir = priv / "b"
    if (bdir / "rows_vita-test.pkl").exists():
        rep["rubi_path"] = json.loads((priv / "rubi_path.json").read_text(encoding="utf-8"))
        rep["rubi_path"].pop("vocr_pin", None)
        b = {}
        for s in SETS_B:
            with open(bdir / f"rows_{s}.pkl", "rb") as fh:
                b[s] = pickle.load(fh)
        fx = [dict(r, set=s, id=f"{s}:{r['id']}") for s in ("vita-test", "psp-test") for r in b[s]["rows"]["fixed"]]
        rep["path_b"]["vita_psp_fixed_lines"] = set_tables(fx, fams, STAGES, cluster=lambda r: r["vn"],
                                                      ident=lambda r: r["id"])
        rep["path_b"]["vita_psp_fixed_lines_per_title"] = per_title(fx, fams, ("shipped",))
        ru = [dict(r, set=s, id=r["uid"]) for s in ("vita-test", "psp-test") for r in b[s]["rows"]["units"]]
        rep["path_b"]["vita_psp_whole_dialogue"] = region_tables(ru, fams, STAGES, cluster=lambda r: r["vn"])
        rep["path_b"]["vita_psp_whole_dialogue_per_title"] = per_title([r for r in ru if r["variant"] == "base"], fams,
                                                                  ("shipped",))
        cfx = b["clannad"]["rows"]["fixed"]
        rep["path_b"]["clannad_lines"] = {
            "dialogue": table(sub(cfx, lambda r: not r["name"]), keys_for(fams, STAGES), cluster=lambda r: r["page"],
                              ident=lambda r: r["id"], exact=True, min_clusters=2),
            "names": table(sub(cfx, lambda r: r["name"]), keys_for(fams, STAGES), cluster=lambda r: r["page"],
                           ident=lambda r: r["id"], exact=True, min_clusters=2)}
        cpu = b["clannad"]["rows"]["units"]
        rep["path_b"]["clannad_pages"] = {
            "dialogue": table(cpu, keys_for(fams, STAGES), cluster=lambda r: r["page"], ident=lambda r: r["uid"],
                              min_clusters=2),
            "all": table(cpu, keys_for(fams, STAGES), chars_key="chars_all", err_key="errors_all",
                         cluster=lambda r: r["page"], ident=lambda r: r["uid"], min_clusters=2),
            "inputs": b["clannad"]["inputs"]}
        pfx = b["pcsg00807"]["rows"]["fixed"]
        rep["path_b"]["pcsg00807_clean"] = table(pfx, keys_for(fams, STAGES), cluster=lambda r: r["page"],
                                            ident=lambda r: r["id"], exact=True, min_clusters=2)
        pu = b["pcsg00807"]["rows"]["units"]
        rep["path_b"]["pcsg00807_page"] = {}
        for kind in ("page", "oracle"):
            for v in list(VARIANTS) + ["all-variants"]:
                rs = sub(pu, lambda r: r["kind"] == kind and (v == "all-variants" or r["variant"] == v))
                t = table(rs, keys_for(fams, STAGES), cluster=lambda r: r["page"], ident=lambda r: r["uid"],
                          min_clusters=2)
                if kind == "page":
                    t["line_stats"] = {k: sum(r["line_stats"].get(k, 0) for r in rs)
                                       for k in ("whole", "cut", "missed", "noink", "merged", "junk")}
                    t["lines"] = sum(r["lines"] for r in rs)
                    t["boxes"] = sum(r["boxes"] for r in rs)
                rep["path_b"]["pcsg00807_page"][f"{kind}|{v}"] = t
        rep["path_b"]["reads"] = {s: {k: b[s].get(k) for k in ("dumps", "rereads", "cascaded", "read_errors",
                                                               "cascade_read_errors", "finder_errors", "finder",
                                                               "vertical_auto", "inputs")} for s in SETS_B}
        dlg = lambda r: not r.get("name")  # noqa: E731
        rep["path_b"]["filter_floor"] = {
            "clannad_lines": filter_floor(sub(cfx, dlg)),
            "clannad_pages": filter_floor(cpu),
            "vita_psp_fixed_lines": filter_floor(sub(fx, dlg)),
            "vita-test_fixed_lines": filter_floor(sub(fx, lambda r: dlg(r) and r["set"] == "vita-test")),
            "psp-test_fixed_lines": filter_floor(sub(fx, lambda r: dlg(r) and r["set"] == "psp-test")),
            "vita_psp_whole_dialogue_base": filter_floor(sub(ru, lambda r: r["variant"] == "base")),
            "pcsg00807_clean": filter_floor(pfx),
            "pcsg00807_page_base": filter_floor(sub(pu, lambda r: r["kind"] == "page" and r["variant"] == "base")),
            "note": "reference characters on dialogue lines with no hiragana, katakana or CJK character: Rubi's "
                    "non-Japanese line filter deletes them however they are read (exact on line measures, an upper "
                    "bound on region and page measures); the same for both families"}
    vp = Path(out) / "validation.json"
    if vp.exists():
        v = json.loads(vp.read_text(encoding="utf-8"))
        rep["validation"] = {"all_equal": v["all_equal"], "checks": len(v["checks"]),
                             "equal": sum(c["equal"] for c in v["checks"]), "at": v.get("at")}
    rep["headline"] = headline(rep)
    rep["seen_per_title"] = {s: {vn: seen_of(proto, vn, s) for vn in {r["vn"] for r in fixed if r["set"] == s}
                                 if seen_of(proto, vn, s)} for s in ("vita-test", "psp-test")}
    assert_public(rep)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "report.json", rep)
    (out / "report.md").write_text(markdown(rep, proto) + "\n", encoding="utf-8")
    return rep


def R_compact_pairs(pairs):
    """vocr3_report's pair entries without their per-stratum style rows (kept: all)."""
    out = {}
    for k, v in pairs.items():
        out[k] = {pol: {s: {"all": e["all"]} for s, e in sets_.items()} for pol, sets_ in v.items()}
    return out


def headline(rep):
    """Per tier: VOCR5 / VOCR6 dialogue CER and the pair, on the headline measures of both paths."""
    def pick(tbl, stage):
        res = {}
        for t in TIERS:
            r5, r6 = tbl["rates"].get(f"VOCR5:{t}:{stage}"), tbl["rates"].get(f"VOCR6:{t}:{stage}")
            res[t] = {"VOCR5": r5 and {k: r5[k] for k in ("cer", "ci95", "errors", "chars")},
                      "VOCR6": r6 and {k: r6[k] for k in ("cer", "ci95", "errors", "chars")},
                      "VOCR6_minus_VOCR5": tbl["pairs"].get(f"{t}:{stage}")}
        return res
    h = {"path_a": {}, "path_b": {}}
    a = rep["path_a"]
    h["path_a"]["clannad_lines"] = pick(a["clannad_lines"]["dialogue"], "greedy")
    h["path_a"]["vita_psp_fixed_lines"] = pick(a["vita_psp_fixed_lines"]["pooled|dialogue"], "greedy")
    h["path_a"]["vita_psp_fixed_lines_lm"] = pick(a["vita_psp_fixed_lines"]["pooled|dialogue"], "lm")
    h["path_a"]["vita_psp_whole_dialogue_base"] = pick(a["vita_psp_whole_dialogue"]["pooled|base|dialogue"], "greedy")
    h["path_a"]["pcsg00807_clean"] = pick(a["pcsg00807"]["clean"], "rubi_cascade")
    h["path_a"]["pcsg00807_page"] = pick(a["pcsg00807"]["page_refine_console_base"], "rubi_cascade")
    b = rep.get("path_b") or {}
    if b:
        h["path_b"]["clannad_lines"] = pick(b["clannad_lines"]["dialogue"], "shipped")
        h["path_b"]["clannad_pages"] = pick(b["clannad_pages"]["dialogue"], "shipped")
        h["path_b"]["vita_psp_fixed_lines"] = pick(b["vita_psp_fixed_lines"]["pooled|dialogue"], "shipped")
        h["path_b"]["vita_psp_whole_dialogue_base"] = pick(b["vita_psp_whole_dialogue"]["pooled|base|dialogue"],
                                                           "shipped")
        h["path_b"]["pcsg00807_clean"] = pick(b["pcsg00807_clean"], "shipped")
        h["path_b"]["pcsg00807_page"] = pick(b["pcsg00807_page"]["page|base"], "shipped")
    return h


# ----------------------------------------------------------------------------- markdown

def pct(r):
    return "-" if not r or r.get("cer") is None or not r.get("chars") else \
        f"{100 * r['errors'] / r['chars']:.2f}%"


def ci(c):
    return "" if not c else f" [{c[0]:.2f}, {c[1]:.2f}]"


def cell(r):
    return "-" if not r else pct(r) + ci(r.get("ci95"))


def pcell(p):
    if not p or p.get("delta") is None:
        return "-"
    return f"{p['delta']:+.2f}" + (f" [{p['ci95'][0]:+.2f}, {p['ci95'][1]:+.2f}]" if p.get("ci95") else "")


SEEN_KEYS = {"clannad": ("all_models", "vocr6_training", "clannad", "clannad_glyph_raster"),
             "vita_psp": ("all_models", "vocr6_training", "vita-test"),
             "pcsg00807": ("all_models", "vocr6_training", "pcsg00807")}


def seen_note(kind, lm=None):
    """The seen flags a table carries (their keys in the protocol's seen_flags, printed in full at the end)."""
    keys = list(SEEN_KEYS[kind]) + ([lm] if lm else [])
    return " (seen flags: " + ", ".join(keys) + ")"


def floor_line(fl, bound=False):
    if not fl or fl.get("points") is None:
        return []
    return [f"Line-filter floor: {fl['lines']:,} reference line(s), {fl['chars']:,} characters "
            f"({fl['points']:.2f} points{', an upper bound' if bound else ''}) hold no Japanese character; Rubi's "
            "non-Japanese line filter deletes them when they are read right (both families alike).", ""]


def md_measure(title, tbl, stage, note="", floor=None, bound=False):
    md = [f"**{title}**{note}", "", "| tier | VOCR5 | VOCR6 | VOCR6 - VOCR5 (points) |", "|---|---|---|---|"]
    for t in TIERS:
        md.append(f"| {t} | {cell(tbl['rates'].get(f'VOCR5:{t}:{stage}'))} | "
                  f"{cell(tbl['rates'].get(f'VOCR6:{t}:{stage}'))} | {pcell(tbl['pairs'].get(f'{t}:{stage}'))} |")
    md.append("")
    return md + floor_line(floor, bound)


def markdown(rep, proto):
    md = ["# VOCR6 final test (read once)", "",
          f"Protocol `eval/vocr6_final_test_protocol.json` (sha256 `{rep['protocol_sha256'][:16]}…`), repo "
          f"`{rep['repo_head'][:7]}`, read {rep['attempt'].get('started')}. Dialogue CER (names a secondary column), "
          "95% intervals: game (VN) bootstrap for vita-test / psp-test, page bootstrap for CLANNAD and the test title; "
          "VOCR6 - VOCR5 paired over the same units. Host measurements; nothing ran on a console.", ""]
    m = rep["models"]["tiers"]
    res = (rep.get("path_a") or {}).get("tool_outputs", {}).get("resources", {})
    md += ["| tier | VOCR6 blob | sha256 | bytes | W8I arena (VOCR6 / VOCR5) | head gate | two-level |",
           "|---|---|---|---:|---|---|---|"]
    for t in TIERS:
        e = m[t]
        ar6, ar5 = (res.get(f"VOCR6_{t}") or {}).get("arena_w8i"), (res.get(f"VOCR5_{t}") or {}).get("arena_w8i")
        arena = f"{ar6:,} / {ar5:,} B" if ar6 and ar5 else "-"
        md.append(f"| {t} | `{e['file']}` | `{e['sha256'][:16]}…` | {e['bytes']:,} | {arena} | {e['head_gate']} | "
                  f"{e['two_level']} |")
    md.append("")
    v = rep.get("validation")
    if v:
        md += [f"VOCR5 on path (a) against its published records, in this read: {v['equal']} of {v['checks']} checks "
               f"equal ({'all equal' if v['all_equal'] else 'NOT all equal: see validation.json'}).", ""]
    if rep.get("rubi_path"):
        rp = rep["rubi_path"]
        rm = rp["lm"].get("rubi_main") or {}
        ships_lm, ships_fix = bool(rm.get("equal_to_pinned")), bool(rp["finder"].get("rubi_ships_fix"))
        what = ("Rubi main `{h}`'s shipped line path" if ships_lm and ships_fix else
                "Rubi main `{h}`'s line read with the protocol's pinned "
                + " and ".join(x for x, ok in (("LM", ships_lm), ("finder fix", ships_fix)) if not ok)
                + " (Rubi main " + "; ".join(x for x in (
                    None if ships_lm else f"pins the LM {rm.get('candidate') or rm.get('sha256', '')[:12]}",
                    None if ships_fix else "does not ship #129's COLGAP fix yet") if x) + ")").format(
                        h=rp["rubi_head"][:7])
        md += [f"Path (b) is {what}: LM `{Path(rp['lm']['file']).name}` "
               f"({rp['lm']['candidate']}; alpha {rp['lm']['settings']['alpha']}, beta {rp['lm']['settings']['beta']}, "
               f"flags {rp['lm']['settings']['flags']}), cascade under {rp['cascade_conf']}, finder TEXTBOX_RECOVERY + "
               f"short-line probe + fix {rp['finder']['fix']} (COLGAP; source {rp['finder']['source']['tree']} "
               f"`{rp['finder']['source']['commit'][:7]}`), refinement on PSP, textnorm `"
               f"{rp['textnorm']['textnorm_sha256'][:12]}…`, the non-Japanese line filter; PSP in the 2x console "
               "view.", ""]
        if rp.get("vertical_check"):
            reads = (rep.get("path_b") or {}).get("reads") or {}
            va = {s: (reads.get(s) or {}).get("vertical_auto") or {} for s in SETS_B}
            md += ["Rubi's ocr_vertical = auto (API 47): regions whose rows look like vertical text are read as "
                   "columns by VOCR5-V (both families). Regions or pages read as columns: " +
                   ", ".join(f"{s} {v.get('columns', 0)} of {v.get('units', 0)}" for s, v in va.items()) + ".", ""]
    md += ["## Headline", ""]
    b = rep.get("path_b") or {}
    if b:
        md += ["### Path (b): Rubi's line path (what of it Rubi main ships: above)", ""]
        ff = b.get("filter_floor") or {}
        t = b["clannad_lines"]["dialogue"]
        md += md_measure(f"CLANNAD, {t['units']:,} dialogue lines (the frozen line crops)", t, "shipped",
                         seen_note("clannad", "lm_lm6"), ff.get("clannad_lines"))
        t = b["clannad_pages"]["dialogue"]
        md += md_measure(f"CLANNAD, {t['units']:,} pages (Rubi's finder on the ROI; pages with a complete reference)",
                         t, "shipped", seen_note("clannad", "lm_lm6"), ff.get("clannad_pages"), True)
        t = b["vita_psp_fixed_lines"]["pooled|dialogue"]
        md += md_measure(f"vita-test + psp-test fixed lines, {t['units']:,} dialogue lines", t, "shipped",
                         seen_note("vita_psp", "lm_lm6"), ff.get("vita_psp_fixed_lines"))
        t = b["vita_psp_whole_dialogue"]["pooled|base|dialogue"]
        md += md_measure(f"vita-test + psp-test whole dialogue, {t['units']:,} base regions", t, "shipped",
                         seen_note("vita_psp", "lm_lm6"), ff.get("vita_psp_whole_dialogue_base"), True)
        t = b["pcsg00807_clean"]
        md += md_measure(f"PCSG00807 clean lines ({t['units']:,})", t, "shipped", seen_note("pcsg00807", "lm_lm6"),
                         ff.get("pcsg00807_clean"))
        t = b["pcsg00807_page"]["page|base"]
        md += md_measure(f"PCSG00807 page path, {t['units']:,} base regions", t, "shipped",
                         seen_note("pcsg00807", "lm_lm6"), ff.get("pcsg00807_page_base"), True)
    a = rep["path_a"]
    md += ["### Path (a): VOCR5's locked path", ""]
    md += md_measure("CLANNAD dialogue lines (greedy)", a["clannad_lines"]["dialogue"], "greedy", seen_note("clannad"))
    md += md_measure("vita-test + psp-test fixed lines, dialogue (greedy, console polarity)",
                     a["vita_psp_fixed_lines"]["pooled|dialogue"], "greedy", seen_note("vita_psp"))
    md += md_measure("the same with beam 8 + ctclm.2's 1 MB LM", a["vita_psp_fixed_lines"]["pooled|dialogue"], "lm",
                     seen_note("vita_psp", "lm_ctclm2"))
    md += md_measure("vita-test + psp-test whole dialogue (base, GRAD)", a["vita_psp_whole_dialogue"]
                     ["pooled|base|dialogue"], "greedy", seen_note("vita_psp"))
    md += md_measure("PCSG00807 clean lines (rubi_cascade)", a["pcsg00807"]["clean"], "rubi_cascade",
                     seen_note("pcsg00807", "lm_ctclm2"))
    md += md_measure("PCSG00807 page GRAD + refine (base, rubi_cascade)", a["pcsg00807"]["page_refine_console_base"],
                     "rubi_cascade", seen_note("pcsg00807", "lm_ctclm2"))
    md += ["## Vita and PSP apart", "", "Seen flags: all_models, vocr6_training, vita-test; the decoded rows also "
           "lm_lm6 (path b) or lm_ctclm2 (path a's + LM).", ""]
    hdr = "| measure | tier | " + " | ".join(f"{s} VOCR5 | {s} VOCR6 | {s} Δ" for s in ("vita-test", "psp-test")) + \
          " |"
    md += [hdr, "|---|---|" + "---|" * 6]
    for path, key, stage, tag in (("b", "vita_psp_fixed_lines", "shipped", "(b) fixed lines"),
                                  ("b", "vita_psp_whole_dialogue", "shipped", "(b) whole dialogue base"),
                                  ("a", "vita_psp_fixed_lines", "greedy", "(a) fixed lines greedy"),
                                  ("a", "vita_psp_fixed_lines", "lm", "(a) fixed lines + LM"),
                                  ("a", "vita_psp_whole_dialogue", "greedy", "(a) whole dialogue base")):
        tbls = rep.get(f"path_{path}", {}).get(key)
        if not tbls:
            continue
        for t in TIERS:
            cells = []
            for s in ("vita-test", "psp-test"):
                tb = tbls.get(f"{s}|dialogue") or tbls.get(f"{s}|base|dialogue")
                cells += [cell(tb["rates"].get(f"VOCR5:{t}:{stage}")), cell(tb["rates"].get(f"VOCR6:{t}:{stage}")),
                          pcell(tb["pairs"].get(f"{t}:{stage}"))]
            md.append(f"| {tag} | {t} | " + " | ".join(cells) + " |")
    md.append("")
    md += ["## Names (secondary)", "", "Seen flags as in the headline tables of each set.", "",
           "| measure | tier | VOCR5 | VOCR6 | Δ |", "|---|---|---|---|---|"]
    for path, tb, stage, tag in (("b", (b.get("vita_psp_fixed_lines") or {}).get("pooled|names"), "shipped",
                                  "(b) vita+psp fixed lines"),
                                 ("b", (b.get("vita_psp_whole_dialogue") or {}).get("pooled|base|names"), "shipped",
                                  "(b) vita+psp whole dialogue base"),
                                 ("b", (b.get("clannad_lines") or {}).get("names"), "shipped", "(b) CLANNAD lines"),
                                 ("a", a["vita_psp_fixed_lines"].get("pooled|names"), "greedy", "(a) vita+psp fixed"),
                                 ("a", a["vita_psp_whole_dialogue"].get("pooled|base|names"), "greedy",
                                  "(a) vita+psp whole dialogue base"),
                                 ("a", a["clannad_lines"]["names"], "greedy", "(a) CLANNAD lines")):
        if not tb:
            continue
        for t in TIERS:
            md.append(f"| {tag} | {t} | {cell(tb['rates'].get(f'VOCR5:{t}:{stage}'))} | "
                      f"{cell(tb['rates'].get(f'VOCR6:{t}:{stage}'))} | {pcell(tb['pairs'].get(f'{t}:{stage}'))} |")
    md.append("")
    if b:
        md += ["## Path (b) decomposition (dialogue CER)", "",
               "Seen flags as in the headline tables; the decoded stages also lm_lm6. The shipped stage includes the "
               "line filter's floor (headline tables).", "",
               "| measure | family | tier | greedy | + counter / LM (tier alone) | + cascade | "
               "shipped (textnorm, filter) |",
               "|---|---|---|---|---|---|---|"]
        for tag, tb in (("CLANNAD lines", b["clannad_lines"]["dialogue"]),
                        ("CLANNAD pages", b["clannad_pages"]["dialogue"]),
                        ("vita+psp fixed lines", b["vita_psp_fixed_lines"]["pooled|dialogue"]),
                        ("vita+psp whole dialogue base", b["vita_psp_whole_dialogue"]["pooled|base|dialogue"]),
                        ("PCSG00807 clean", b["pcsg00807_clean"]),
                        ("PCSG00807 page base", b["pcsg00807_page"]["page|base"]),
                        ("PCSG00807 oracle boxes base", b["pcsg00807_page"]["oracle|base"])):
            for f in FAMILIES:
                for t in TIERS:
                    md.append(f"| {tag} | {f} | {t} | " + " | ".join(pct(tb["rates"].get(f"{f}:{t}:{s}"))
                                                                    for s in STAGES) + " |")
        ls = b["pcsg00807_page"]["page|base"].get("line_stats")
        if ls:
            md += ["", f"PCSG00807's base regions through Rubi's finder: {b['pcsg00807_page']['page|base']['lines']:,} "
                   f"reference lines: {ls['whole']:,} whole, {ls['cut']:,} cut, {ls['missed']:,} missed; "
                   f"{b['pcsg00807_page']['page|base']['boxes']:,} boxes, {ls['merged']:,} merged, {ls['junk']:,} "
                   "junk."]
        md.append("")
    md += ["## Per title (vita-test, psp-test; dialogue)", "", "Intervals per title are in report.json. 'seen': the "
           "title's own flag (seen_flags.vita-test); every title also carries all_models and vocr6_training.", ""]
    for path, key, stage in (("b", "vita_psp_fixed_lines_per_title", "shipped"),
                             ("b", "vita_psp_whole_dialogue_per_title", "shipped"),
                             ("a", "vita_psp_fixed_lines_per_title", "greedy"),
                             ("a", "vita_psp_whole_dialogue_per_title", "greedy")):
        pt = rep.get(f"path_{path}", {}).get(key)
        if not pt:
            continue
        md += [f"**({path}) {key.replace('_per_title', '').replace('_', ' ')}, {stage}**", "",
               "| set | VN | title | chars | " + " | ".join(f"{f[-1]}{t}" for t in TIERS for f in FAMILIES) +
               " | seen |", "|---|---|---|---:|" + "---|" * 6 + "---|"]
        for vn, e in sorted(pt.items(), key=lambda kv: (kv[1]["set"] or "", kv[0])):
            flags = seen_of(proto, vn, e["set"])
            md.append(f"| {e['set']} | {vn} | {e['title'] or ''} | {e['chars']:,} | " +
                      " | ".join(pct(e["rates"].get(f"{f}:{t}:{stage}")) for t in TIERS for f in FAMILIES) +
                      f" | {'yes' if flags else ''} |")
        md.append("")
    md += ["## Seen flags and disclosures", ""]
    for k, v in proto["seen_flags"].items():
        if k == "rule":
            continue
        md.append(f"- **{k}**: " + (v if isinstance(v, str) else "; ".join(f"{a}: {b}" for a, b in v.items())))
    md.append("")
    return "\n".join(md)


def cmd_report(a):
    rep = build_report(a.priv, a.out, a.suffix)
    print(json.dumps(rep["headline"]["path_b"] or rep["headline"]["path_a"], ensure_ascii=False)[:4000])


# ----------------------------------------------------------------------------- selftest (development data only)

def _sample(n, k, seed=0):
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(n, size=min(k, n), replace=False).tolist())


def selftest_finder(src_dir, out_dir, plan_path, pages, limit=0):
    """Path (b)'s finder build against vocr6-lead's hval plan v2 (pagepath_find from #135, fix 5, probe 1): every
    unit's region re-cut from its page in the console view, the boxes and dark flags compared."""
    import cv2
    exe, meta = build_finder(src_dir, out_dir)
    with open(plan_path, "rb") as f:
        plan = pickle.load(f)
    units = plan["units"][:limit] if limit else plan["units"]
    cache, imgs, plats = {}, [], []
    for u in units:
        k = (u["title"], u["page"])
        if k not in cache:
            if len(cache) > 64:
                cache.clear()
            bgr = cv2.imread(str(Path(pages) / u["title"] / "pages" / f"{u['page']}.png"), cv2.IMREAD_COLOR)
            cache[k] = console_frame(np.ascontiguousarray(bgr[..., ::-1]), u["platform"])[1]
        x0, y0, x1, y1 = u["region"]
        imgs.append(np.ascontiguousarray(cache[k][y0:y1, x0:x1]))
        plats.append(u["platform"])
    found = find_boxes(exe, imgs, plats, plan["fix"], plan["probe"])
    same = sum(1 for u, (b, d, _) in zip(units, found) if [list(x) for x in b] == u["boxes"] and d == u["dark"])
    return {"units": len(units), "identical": same, "fix": plan["fix"], "probe": plan["probe"],
            "plan_driver_sha256": plan["driver_sha256"], "driver": meta}


T1_SCORER = {"commit": "fb077e7", "branch": "feat/vocr6-t1-stages",
             "sha256": "808b49eedb97446ea76867a7beb2d2f14609680b0f19aa7eabdd7ad4afd7c3bb"}


def t1_module(name, work, pin=T1_SCORER):
    """vocr6-lead's T1 scorer eval/<name>.py at the commit the recorded selftest read (feat/vocr6-t1-stages fb077e7:
    its stage-4 cascade; main's #134 version has none), copied into work and checked by sha256: the selftest's
    reference implementation only."""
    import importlib.util
    subprocess.run(["git", "fetch", "-q", "origin", pin["branch"]], cwd=ROOT, capture_output=True)
    path = Path(work) / f"{name}.py"
    path.write_bytes(subprocess.run(["git", "show", f"{pin['commit']}:eval/{name}.py"], cwd=ROOT, capture_output=True,
                                    check=True).stdout)
    if sha256(path) != pin["sha256"]:
        raise SystemExit(f"{name}.py at {pin['commit']} is not the pinned reference")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def selftest_plain(plan_path, dump_paths, blobs, lm, settings, k, workers, rubi_norm, work):
    """Path (b)'s line read with plain blobs (no head gate) against vocr6-lead's hval dumps of the same plan (eval/
    vocr6_score.py read): greedy, the beam text and the confidence per crop; then the cascade, textnorm and the
    filter against vocr6_score.texts(rubi, up=the next tier)."""
    VS = t1_module("vocr6_score", work)
    with open(plan_path, "rb") as f:
        plan = pickle.load(f)
    dumps = {t: json.loads(Path(p).read_text(encoding="utf-8")) for t, p in dump_paths.items()}
    for t, d in dumps.items():
        if d["plan_sha256"] != sha256(plan_path) or d["blob_sha256"] != sha256(blobs[t]):
            raise SystemExit(f"{dump_paths[t]}: another plan or blob")
    idx = _sample(len(plan["crops"]), k)
    out = {"crops": len(idx)}
    mine = {}
    for t, blob in blobs.items():
        jobs = [(i, plan["crops"][i][0], plan["crops"][i][1]) for i in idx]
        res = {}
        with get_context("fork").Pool(workers, initializer=_init, initargs=(str(blob), lm, settings, (6000, 4000))) \
                as pool:
            for i, r in pool.imap_unordered(_read, jobs, chunksize=16):
                res[i] = r
        d = dumps[t]["crops"]
        out[t] = {"greedy_equal": sum(res[i]["g"] == d[i]["g"] for i in idx),
                  "beam_equal": sum((res[i]["b"] if res[i]["g"] else "") == (d[i]["b"] if d[i]["g"] else "")
                                    for i in idx),
                  "conf_equal": sum(res[i]["conf"] == (int(d[i]["cmin"] * 1000 + 0.5) if d[i]["g"] else 1000)
                                    for i in idx),
                  "rereads": sum(res[i]["reread"] for i in idx)}
        mine[t] = res
    if "N" in mine and "L" in mine:
        cnorm = Norm(rubi_norm)
        theirs = VS.texts(dumps["N"], "rubi", VS.Norm(rubi_norm), up=dumps["L"])
        own = [compose(mine["N"][i], mine["L"][i], 700) for i in idx]
        shipped = [t if has_japanese(t) else "" for t in cnorm(own)]
        out["shipped_equal"] = sum(a == theirs[i] for a, i in zip(shipped, idx))
    return out


def selftest_gated(cache_dir, tiers, k, workers, lm):
    """Path (b)'s line read with the released gated arms against eval/vocr6_baseline.py's hval reads of the same
    crops (greedy, the re-read's beam text before the counter rule, the confidence): a sample plus every crop the
    baseline read again."""
    import head2level_dev as H2
    import vocr6_baseline as VB
    with open(Path(cache_dir) / "crops.pkl", "rb") as f:
        crops = pickle.load(f)
    out = {}
    for t in tiers:
        r = json.loads((Path(cache_dir) / f"reads_{t}.json").read_text(encoding="utf-8"))
        pos = {key: i for i, key in enumerate(r["keys"])}
        rows = [i for i, c in enumerate(crops) if c[0] in pos]
        idx = sorted(set(_sample(len(rows), k)) | {j for j, c in enumerate(rows) if r["reread"][pos[crops[c][0]]]})
        jobs = [(j, crops[rows[j]][1], crops[rows[j]][2]) for j in idx]
        res = {}
        with get_context("fork").Pool(workers, initializer=_init,
                                      initargs=(str(VB.ARMS[t]["blob"]), lm, dict(H2.RUBI_LM), (6000, 4000))) \
                as pool:
            for j, x in pool.imap_unordered(_read, jobs, chunksize=16):
                res[j] = x
        e = {"crops": len(idx), "baseline_rereads": 0, "greedy_equal": 0, "text_equal": 0, "conf_equal": 0,
             "reread_equal": 0, "differ_kana_rule": 0}
        for j in idx:
            p = pos[crops[rows[j]][0]]
            x = res[j]
            e["baseline_rereads"] += r["reread"][p]
            e["greedy_equal"] += x["g"] == r["greedy"][p]
            same = (x["b"] if x["g"] else "") == (r["lm_reread"][p] if r["greedy"][p] else "")
            e["text_equal"] += same
            e["conf_equal"] += x["conf"] == r["conf"][p] or bool(x["reread"])
            e["reread_equal"] += x["reread"] == r["reread"][p]
            if not same and t == "S" and r["reread"][p] and not x["reread"]:
                e["differ_kana_rule"] += 1
        out[t] = e
    return out


def cmd_selftest(a):
    """Development data only (the hooked hval block): the finder, the plain and the gated line reads against the
    records of the tools whose shipped-path parts this harness reuses."""
    proto = protocol()
    out = {"kind": "vocr6-final-test-selftest", "at": now(), "repo_head": git_head()}
    src = Path(a.work) / "finder_src"
    out["finder_source"] = finder_sources(src)
    lead = A / "vocr6-lead"
    if "finder" in a.parts:
        out["finder"] = selftest_finder(src, Path(a.work) / "bin", lead / "plans" / "hval_plan_v2.pkl",
                                        A / "vocr6-ingest" / "pages", a.limit)
        write_json(a.out, out)
    rel = A / "vocr5-train" / "release" / "v0.3.0"
    lmc = proto["paths"]["b"]["lm_candidates"]["ctclm2_1000k"]
    import pagepath as PP
    rubi_norm = PP.rubi_norm_tool(a.rubi, Path(a.work) / "bin")[0]
    if "plain" in a.parts:
        out["plain"] = selftest_plain(lead / "plans" / "hval_plan_v1.pkl",
                                      {"N": lead / "scores" / "dump_hval_VOCR5-N.json",
                                       "L": lead / "scores" / "dump_hval_VOCR5-L.json"},
                                      {"N": rel / "VOCR5_N_w8.vocr", "L": rel / "VOCR5_L_w8.vocr"},
                                      lmc["file"], lmc["settings"], a.k, a.workers, rubi_norm, a.work)
        write_json(a.out, out)
    if "gated" in a.parts:
        out["gated"] = selftest_gated(A / "vocr6-ingest" / "baseline" / "hval", TIERS, a.k, a.workers, lmc["file"])
    write_json(a.out, out)
    print(json.dumps({k: v for k, v in out.items() if k in ("finder", "plain", "gated")}))


# ----------------------------------------------------------------------------- dryrun (development data only)

DRY_TESTS = {"vita-test": ("PCSG00576", "PCSG00146", "PCSG00653"), "psp-test": ("NPJH50001", "ULJM05674")}
DRY_CLANNAD, DRY_PCSG = "PCSG00620", "PCSG00576"


def dry_pages(pagepool, title, n):
    rows = [r for r in read_jsonl(Path(pagepool) / f"{title}.jsonl") if r["split"] == "hval" and r.get("regions")]
    return rows[:n]


def dry_tests_rows(pagepool, titles, n):
    """A vita-test / psp-test-shaped stand-in from hooked hval pages: each base region a manifest region, its
    verified lines the rows."""
    out = []
    for t in titles:
        for r in dry_pages(pagepool, t, n):
            by = {ln["id"]: ln for ln in r["lines"]}
            for g in r["regions"]:
                if g["variant"] != "base":
                    continue
                x0, y0, x1, y1 = g["region"]
                for lid in g["lines"]:
                    ln = by[lid]
                    out.append({"id": lid, "vn": t, "title": t, "session": r["page"], "url": "", "page": r["page"],
                                "sha256": r["image_sha256"], "dims": r["dims"], "region": [x0, y0, x1 - x0, y1 - y0],
                                "box": ln.get("box_main") or ln["box"], "text": ln["text"], "styles": []})
        # a vertical title's pages (no horizontal region): a region drawn around its columns, so that the stand-in
        # exercises API 47's auto switch (the column read by the vertical model) end to end
        rows = [r for r in read_jsonl(Path(pagepool) / f"{t}.jsonl") if r["split"] == "hval" and
                sum(1 for ln in r["lines"] if ln.get("vertical")) >= 2][:n]
        for r in rows:
            v = [ln for ln in r["lines"] if ln.get("vertical")]
            W, H = r["dims"]
            x0, y0 = max(0, min(ln["box"][0] for ln in v) - 12), max(0, min(ln["box"][1] for ln in v) - 12)
            x1 = min(W, max(ln["box"][0] + ln["box"][2] for ln in v) + 12)
            y1 = min(H, max(ln["box"][1] + ln["box"][3] for ln in v) + 12)
            for ln in sorted(v, key=lambda q: -q["box"][0]):
                out.append({"id": ln["id"], "vn": t, "title": t, "session": r["page"], "url": "", "page": r["page"],
                            "sha256": r["image_sha256"], "dims": r["dims"], "region": [x0, y0, x1 - x0, y1 - y0],
                            "box": ln["box"], "text": ln["text"], "styles": []})
    return out


def dry_clannad_work(work, pagepool, pages, title, n):
    """A clannad_refs-shaped work directory from hooked hval pages: a page's first base region is the ROI, its lines
    the long-run lines (stored grey PNG crops), every second page with one extra no-text line (an empty reference)."""
    import cv2
    from PIL import Image

    from common import line_dark, to_gray
    work, session = Path(work), Path(work) / "session"
    (session / "inputs").mkdir(parents=True, exist_ok=True)
    lines, refs = [], []
    for seq, r in enumerate(dry_pages(pagepool, title, n)):
        src = session / "inputs" / f"{seq:06d}-native.png"
        if not src.exists():
            os.symlink(Path(pages) / r["image"], src)
        rgb = np.ascontiguousarray(cv2.imread(str(src), cv2.IMREAD_COLOR)[..., ::-1])
        g = [x for x in r["regions"] if x["variant"] == "base"][0]
        x0, y0, x1, y1 = g["region"]
        roi = [x0, y0, x1 - x0, y1 - y0]
        grey = to_gray(rgb)[y0:y1, x0:x1]
        by = {ln["id"]: ln for ln in r["lines"]}
        boxes = []
        for lid in g["lines"]:
            bx, by_, bw, bh = by[lid].get("box_main") or by[lid]["box"]
            boxes.append((by[lid]["text"], [max(0, bx - x0 - 3), max(0, by_ - y0 - 3), min(bw + 6, x1 - bx), bh + 6]))
        if seq % 2:
            boxes.append(("", [0, 0, max(8, roi[2] // 8), max(8, roi[3] // 8)]))
        for i, (text, b) in enumerate(boxes):
            x, y, w, h = b
            w, h = min(w, roi[2] - x), min(h, roi[3] - y)
            crop = np.ascontiguousarray(grey[y:y + h, x:x + w])
            path = session / "inputs" / f"{seq:06d}-line-{i:02d}.png"
            Image.fromarray(crop).save(path)
            dark = bool(line_dark(grey, (x, y, w, h)))
            lines.append({"id": f"p{seq:05d}-{i}", "page": f"p{seq:05d}", "seq": seq, "i": i, "event_sha256": "",
                          "line": {"box": [x, y, w, h], "dark": dark, "mode": "L",
                                   "path": str(path.relative_to(session)), "sha256": sha256(path), "size": [w, h]},
                          "source": {"mode": "RGB", "path": str(src.relative_to(session)), "sha256": sha256(src),
                                     "size": [rgb.shape[1], rgb.shape[0]]},
                          "roi": roi, "box": [x, y, w, h], "dark": dark, "lens": "", "cached": ""})
            refs.append({"id": f"p{seq:05d}-{i}", "page": f"p{seq:05d}", "status": "ref" if text else "excluded",
                         "text": text, "name": False, "notext": not text, "dark": dark, "reads": [],
                         "why": "r1=r2" if text else "r1,r2 skip:notext"})
    import clannad_refs as C
    meta = {"session": str(session), "lines_sha256": C.write_jsonl(work / "lines.jsonl", lines),
            "pages": len({x["page"] for x in lines}), "lines": len(lines)}
    write_json(work / "lines.meta.json", meta)
    write_json(work / "refs.freeze.json", {"refs_sha256": C.write_jsonl(work / "refs.jsonl", refs)})
    return work


def dry_fake_priv(priv, val, models, rows_b, suffix):
    """A private directory for build_report: the validation's path (a) outputs with every VOCR5 entry copied as
    VOCR6 (the renderer's test only; no model reads anything), the stand-in path (b) rows, a stand-in models.json."""
    priv, val = Path(priv), Path(val)
    for d in ("a/shared", "a/lm", "a/pcsg", "clannad", "b"):
        (priv / d).mkdir(parents=True, exist_ok=True)
    rep = json.loads((val / "a/shared/report.json").read_text(encoding="utf-8"))
    sh = rep["shared"]
    for pol in sh["fixed_line"]:
        for t in TIERS:
            sh["fixed_line"][pol][f"VOCR6_{t}"] = sh["fixed_line"][pol][f"VOCR5_{t}"]
    for t in TIERS:
        sh["dialogue"]["units"][f"VOCR6_{t}"] = sh["dialogue"]["units"][f"VOCR5_{t}"]
    write_json(priv / "a/shared/report.json", rep)
    shutil.copyfile(val / "a/shared/shared.json", priv / "a/shared/shared.json")
    for f in (val / "a/lm").glob("VOCR5_*.json"):
        shutil.copyfile(f, priv / "a/lm" / f.name)
        shutil.copyfile(f, priv / "a/lm" / f.name.replace("VOCR5_", "VOCR6_"))
    rows = read_jsonl(val / "clannad" / f"score.{suffix}.private.jsonl")
    for x in rows:
        for t in TIERS:
            x["errors"][f"VOCR6_{t}"] = x["errors"][f"VOCR5_{t}"]
    with open(priv / "clannad" / f"score.{suffix}.private.jsonl", "w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    for n in (f"clannad_{suffix}.json", f"clannad_{suffix}_dialogue.json"):
        shutil.copyfile(val / "clannad" / n, priv / "clannad" / n)
    for f in ("VOCR5", "VOCR6"):
        shutil.copyfile(val / "a/pcsg/VOCR5.rows.pkl", priv / "a/pcsg" / f"{f}.rows.pkl")
    write_json(priv / "attempt.json", {"mode": "dryrun", "started": now()})
    write_json(priv / "models.json", models)
    for s, p in rows_b.items():
        shutil.copyfile(p, priv / "b" / f"rows_{s}.pkl")


def cmd_dryrun(a):
    """Development data only (the hooked hval block; no protected set is read by a model): path (b)'s plan, read and
    score-b on stand-ins of the four sets, then build_report on the stand-in rows with path (a)'s inputs copied from
    the validation's private outputs (VOCR6 = VOCR5 there): the final read's code paths, end to end."""
    proto = protocol()
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    s = proto["sets"]["pcsg00807"]
    pagepool, pages, pools = s["pagepool_dir"], s["pages_dir"], s["pools_dir"]
    rpath = work / "rubi_path.json"
    if not rpath.exists():
        cmd_rubi_path(argparse.Namespace(rubi=a.rubi, out=str(rpath)))
    rp = load_rubi_path(rpath)
    blobs = {("VOCR5", t): proto["models"]["VOCR5"]["gated"][t]["file"] for t in TIERS}
    for t, spec in zip(TIERS, a.vocr6.split(",")):
        blobs[("VOCR6", t)] = spec
    summary, rows_b = {}, {}
    for name in SETS_B:
        planp = work / f"plan_{name}.pkl"
        if not planp.exists():
            plan = Plan(name)
            if name in DRY_TESTS:
                rows = dry_tests_rows(pagepool, DRY_TESTS[name], a.pages)
                plan_tests(name, rp, plan, None, rows=rows, platform="vita" if name == "vita-test" else "psp",
                           image_of=lambda r: Path(pages) / r["vn"] / "pages" / f"{r['page']}.png")
            elif name == "clannad":
                plan_clannad(rp, plan, dry_clannad_work(work / "clannad_work", pagepool, pages, DRY_CLANNAD,
                                                        a.pages * 2))
            else:
                plan_pcsg(rp, plan, pools, pagepool, pages, title=DRY_PCSG, split="hval", vn=DRY_PCSG)
            plan.doc["finder"] = {k: rp["finder"][k] for k in ("fix", "driver_sha256", "sources_sha256")}
            write_pickle(planp, plan.doc)
        dumps = []
        vd = work / f"dump_{name}_V_V.pkl"
        if not vd.exists():
            cmd_read(argparse.Namespace(plan=str(planp), blob=proto["models"]["VOCR5"]["vertical"]["file"],
                                        rubi_path=str(rpath), workers=a.workers, only="columns", out=str(vd)))
        dumps.append(f"V:V={vd}")
        for (fam, t), blob in blobs.items():
            dp = work / f"dump_{name}_{fam}_{t}.pkl"
            if not dp.exists():
                cmd_read(argparse.Namespace(plan=str(planp), blob=str(blob), rubi_path=str(rpath), workers=a.workers,
                                            only="rows", out=str(dp)))
            dumps.append(f"{fam}:{t}={dp}")
        rows_b[name] = work / f"rows_{name}.pkl"
        if not rows_b[name].exists():
            cmd_score_b(argparse.Namespace(plan=str(planp), dump=dumps, rubi_path=str(rpath), out=str(rows_b[name])))
        with open(rows_b[name], "rb") as f:
            d = pickle.load(f)
        summary[name] = {"fixed": len(d["rows"]["fixed"]), "units": len(d["rows"]["units"]),
                         "read_errors": d["read_errors"], "finder_errors": d["finder_errors"], "rereads": d["rereads"],
                         "vertical_auto": d["vertical_auto"]}
    models = {"kind": "vocr6-final-test-models", "staged": now(), "tiers": {}}
    for t in TIERS:
        b = blobs[("VOCR6", t)]
        gate, two = blob_heads(b)
        models["tiers"][t] = {"source": str(b), "file": Path(b).name, "sha256": sha256(b),
                              "bytes": Path(b).stat().st_size, "head_gate": gate, "two_level": two}
    priv = work / "priv"
    dry_fake_priv(priv, a.validation, models, rows_b, a.suffix)
    shutil.copyfile(rpath, priv / "rubi_path.json")
    rep = build_report(priv, work / "report", a.suffix)
    summary["report_md_lines"] = len((work / "report" / "report.md").read_text(encoding="utf-8").splitlines())
    summary["headline_b_keys"] = sorted(rep["headline"]["path_b"])
    write_json(work / "dryrun.json", summary)
    print(json.dumps(summary)[:3000])


# ----------------------------------------------------------------------------- rubicheck (development data only)

def cmd_rubicheck(a):
    """Development data only (hooked hval regions of the train titles): path (b)'s finder and line read against
    Rubi's own host tool vjo-ocr (host/ocr_tool.c: vjo_vocr_ocr_policy, the game worker's job: the finder with its
    short-line probe, the refinement, the head gate, the two-level head, the counter re-decode, the re-read, the LM and
    the cascade), built from a Rubi checkout, with the models and LM that checkout pins (VOCR5's gated blobs, its 1 MB
    LM) and the finder it runs (fix 0 until Rubi pins #129's fixes). Compared per region and tier: the boxes, the dark
    flags, each line's text and the job's text."""
    import cv2
    from pyvocr import Vocr, VocrLM

    proto = protocol()
    s = proto["sets"]["pcsg00807"]
    pagepool, pages = Path(s["pagepool_dir"]), Path(s["pages_dir"])
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    md = Path(a.models)
    names = {"S": "VOCR5_S_w8g.vocr", "N": "VOCR5_N_w8g2.vocr", "L": "VOCR5_L_w8g2.vocr"}
    for t in TIERS:
        if sha256(md / names[t]) != proto["models"]["VOCR5"]["gated"][t]["sha256"]:
            raise SystemExit(f"{md / names[t]} is not the protocol's gated VOCR5 {t}")
    src = (Path(a.rubi) / "core" / "vocr_ocr.c").read_text(encoding="utf-8")
    lm_sha, lm_set = rubi_lm(src)
    lm_name = {c["sha256"]: Path(c["file"]).name for c in proto["paths"]["b"]["lm_candidates"].values()}[lm_sha]
    lm = md / lm_name
    if sha256(lm) != lm_sha:
        raise SystemExit(f"{lm} is not the LM Rubi pins")
    fsrc = finder_sources(work / "finder_src")
    exe, _ = build_finder(work / "finder_src", work / "bin")
    fix = int(a.fix)
    # a Rubi with #129's fixes (API 44, Rubi #330): vjo-ocr --line-fix on is VOCR_FIX_DEFAULT (audit F2)
    if fix and fix != fsrc["fix_default"]:
        raise SystemExit(f"--fix {fix}: rubicheck compares Rubi's default fix ({fsrc['fix_default']}) or none")
    line_fix_arg = ["--line-fix", "on"] if fix else []
    rts = {}
    for t in TIERS:
        gate, two = blob_heads(md / names[t])
        rt = Vocr(str(md / names[t]), mode="w8i", max_width=1024, head_gate="on" if gate else "off",
                  two_level=1 if two else 0)
        rts[t] = (rt, VocrLM(rt, str(lm), **lm_set), two)
    units, counter_units = [], []
    counter_re = re.compile("[〇一二三四五六七八九十百千万0-9０-９何数幾][ヶヵケカッ箇][月年所国条]")
    for f in sorted(pagepool.glob("*.jsonl")):
        allrows = [r for r in read_jsonl(f) if r.get("regions") and not r.get("untrusted") and
                   r["split"] in ("train", "hval")]
        rows = [r for r in allrows if r["split"] == "hval"]
        rng = np.random.default_rng(int(hashlib.sha256(f"rubicheck:{f.stem}".encode()).hexdigest()[:8], 16))
        for i in sorted(rng.choice(len(rows), size=min(a.per_title, len(rows)), replace=False).tolist()) if rows \
                else []:
            r = rows[i]
            g = [x for x in r["regions"] if x["variant"] == "base"][0]
            units.append((r, g))
        # development regions whose verified text holds a counter (the re-decode's and the re-read's cases)
        for r in allrows:
            texts = {ln["id"]: ln["text"] for ln in r["lines"]}
            for g in r["regions"]:
                if g["variant"] == "base" and any(counter_re.search(texts.get(i, "")) for i in g["lines"]):
                    counter_units.append((r, g))
    rng = np.random.default_rng(20261026)
    if counter_units and a.counter_regions:
        pick = sorted(rng.choice(len(counter_units), size=min(a.counter_regions, len(counter_units)),
                                 replace=False).tolist())
        units += [counter_units[i] for i in pick]
    res = {"regions": 0, "sampled": len(units), "counter_candidates": len(counter_units),
           "tiers": {t: defaultdict(int) for t in TIERS}, "fix": fix, "finder_source": fsrc["tree"],
           "rubi_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=a.rubi, capture_output=True,
                                       text=True).stdout.strip(), "lm": lm_name, "lm_settings": lm_set,
           "differences": []}
    for r, g in units:
        bgr = cv2.imread(str(pages / r["image"]), cv2.IMREAD_COLOR)
        rgb2, grey2, sc = console_frame(np.ascontiguousarray(bgr[..., ::-1]), r["platform"])
        x0, y0, x1, y1 = (v * sc for v in g["region"])
        reg_rgb = np.ascontiguousarray(rgb2[y0:y1, x0:x1])
        grey = np.ascontiguousarray(grey2[y0:y1, x0:x1])
        H, W = grey.shape
        if W > 960 or H > 544:
            continue
        rgba = np.concatenate([reg_rgb, np.full((H, W, 1), 255, np.uint8)], axis=2)
        rp = work / "region.rgba"
        rp.write_bytes(np.ascontiguousarray(rgba).tobytes())
        boxes, dark, st = find_boxes(exe, [grey], [r["platform"]], fix)[0]
        crops = [np.ascontiguousarray(grey[y:y + h, x:x + w]) for (x, y, w, h) in boxes]
        mine = {t: [rubi_line(rts[t][0], rts[t][1], c, d, rts[t][2], (6000, 4000)) for c, d in zip(crops, dark)]
                for t in TIERS}
        res["regions"] += 1
        for t in TIERS:
            nt = NEXT[t]
            comp = [compose(m, mine[nt][i] if nt else None, 700) for i, m in enumerate(mine[t])]
            lp = work / "lines.jsonl"
            if lp.exists():
                lp.unlink()
            argv = [str(a.vjo_ocr), "--backend", "vocr", "--model", names[t], "--lm", lm_name, "--counters", "on",
                    "--head-gate", "on", "--two-level", "on", "--refine", "on" if r["platform"] == "psp" else "off",
                    "--lines", str(lp)] + (["--cascade", names[nt]] if nt else []) + line_fix_arg + \
                [str(md), str(W), str(H), str(rp)]
            out = subprocess.run(argv, capture_output=True, timeout=600)
            theirs = read_jsonl(lp) if lp.exists() else []
            text = out.stdout.decode("utf-8")[:-1] if out.returncode == 0 else None
            e = res["tiers"][t]
            e["regions"] += 1
            tb = [tuple(x["box"]) for x in theirs]
            same_boxes = tb == [tuple(b) for b in boxes] and [x["dark"] for x in theirs] == list(dark)
            e["boxes_equal"] += same_boxes
            e["lines"] += len(boxes)
            if same_boxes:
                eq = [x["text"] == c for x, c in zip(theirs, comp)]
                e["line_text_equal"] += sum(eq)
                e["greedy_equal"] += sum(x["greedy"] == (m["used_g"] if m["g"] else "")
                                         for x, m in zip(theirs, mine[t]))
                e["cascaded"] += sum(1 for x in theirs if x["flags"] & 0x0400)
            job = "\n".join(c for c in comp if c)
            e["job_text_equal"] += text == job
            e["rereads"] += sum(m["reread"] for m in mine[t])
            if not same_boxes or text != job:
                res["differences"].append({"title": r["title_id"], "page": r["page"], "tier": t,
                                           "boxes_equal": same_boxes, "rc": out.returncode,
                                           "lines_theirs": len(theirs), "lines_mine": len(boxes)})
    res["tiers"] = {t: dict(v) for t, v in res["tiers"].items()}
    res["all_equal"] = all(v["boxes_equal"] == v["regions"] and v["job_text_equal"] == v["regions"]
                           for v in res["tiers"].values())
    write_json(a.out, res)
    print(json.dumps({k: v for k, v in res.items() if k != "differences"}))


# ----------------------------------------------------------------------------- pagecheck (development data only)

def dev_regions(pagepool, per_title, counter_regions, vertical_regions, seed=20261026):
    """Development regions of the hooked train titles (train and hval pages; never the test split): per title a
    sample of hval base regions, regions whose verified text holds a counter, and regions drawn around the vertical
    title's columns (PCSG00653: the union of a page's vertical line boxes plus 12 px)."""
    pagepool = Path(pagepool)
    counter_re = re.compile("[〇一二三四五六七八九十百千万0-9０-９何数幾][ヶヵケカッ箇][月年所国条]")
    sample, counters, verticals = [], [], []
    for f in sorted(pagepool.glob("*.jsonl")):
        rows = [r for r in read_jsonl(f) if r["split"] in ("train", "hval") and not r.get("untrusted")]
        hv = [r for r in rows if r["split"] == "hval" and r.get("regions")]
        rng = np.random.default_rng(int(hashlib.sha256(f"pagecheck:{f.stem}".encode()).hexdigest()[:8], 16))
        for i in sorted(rng.choice(len(hv), size=min(per_title, len(hv)), replace=False).tolist()) if hv else []:
            g = [x for x in hv[i]["regions"] if x["variant"] == "base"][0]
            sample.append((hv[i], g["region"], "horizontal"))
        for r in rows:
            texts = {ln["id"]: ln["text"] for ln in r["lines"]}
            for g in r.get("regions") or []:
                if g["variant"] == "base" and any(counter_re.search(texts.get(i, "")) for i in g["lines"]):
                    counters.append((r, g["region"], "counter"))
            v = [ln["box"] for ln in r["lines"] if ln.get("vertical")]
            if len(v) >= 2:
                W, H = r["dims"]
                x0 = max(0, min(b[0] for b in v) - 12)
                y0 = max(0, min(b[1] for b in v) - 12)
                x1 = min(W, max(b[0] + b[2] for b in v) + 12)
                y1 = min(H, max(b[1] + b[3] for b in v) + 12)
                verticals.append((r, [x0, y0, x1, y1], "vertical"))
    rng = np.random.default_rng(seed)
    pick = lambda xs, k: [xs[i] for i in sorted(rng.choice(len(xs), size=min(k, len(xs)), replace=False))]  # noqa
    return sample + pick(counters, counter_regions) + pick(verticals, vertical_regions)


def cmd_pagecheck(a):
    """Development data only: path (b) (job_crops, rubi_line, the cascade, textnorm, the line filter; API 47's auto
    with the column read) against Rubi's own vjo-page-ocr (host/page_ocr.c: the game worker's job on the defaults of
    config.ini, --vertical auto) built from a Rubi checkout, with the models and LM that checkout pins (VOCR5's gated
    blobs, its tier-2 LM, VOCR5-V) and its finder (--fix as that checkout runs it). Compared per region and tier:
    auto's switch, each line's box, dark flag and text, and the text shown."""
    import cv2
    from pyvocr import Vocr, VocrLM

    proto = protocol()
    s = proto["sets"]["pcsg00807"]
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    md = Path(a.models)
    names = {"S": "VOCR5_S_w8g.vocr", "N": "VOCR5_N_w8g2.vocr", "L": "VOCR5_L_w8g2.vocr"}
    for t in TIERS:
        if sha256(md / names[t]) != proto["models"]["VOCR5"]["gated"][t]["sha256"]:
            raise SystemExit(f"{md / names[t]} is not the protocol's gated VOCR5 {t}")
    if sha256(md / "VOCR5_V_w8.vocr") != proto["models"]["VOCR5"]["vertical"]["sha256"]:
        raise SystemExit("the vertical model is not the protocol's VOCR5-V")
    src = (Path(a.rubi) / "core" / "vocr_ocr.c").read_text(encoding="utf-8")
    lm_sha, lm_set = rubi_lm(src)
    lm_name = {c["sha256"]: Path(c["file"]).name for c in proto["paths"]["b"]["lm_candidates"].values()}[lm_sha]
    lm = md / lm_name
    if sha256(lm) != lm_sha:
        raise SystemExit(f"{lm} is not the LM Rubi pins")
    jd = rubi_job_defaults(a.rubi)
    fsrc = finder_sources(work / "finder_src")
    exe, _ = build_finder(work / "finder_src", work / "bin")
    import pagepath as PP
    norm_exe, _ = PP.rubi_norm_tool(a.rubi, work / "bin")
    cnorm = Norm(norm_exe)
    fix = int(a.fix)
    rp = {"finder": {"driver": str(exe), "fix": fix}, "vertical_check": jd["ocr_vertical"][0] == "VJO_VERTICAL_AUTO"}
    rts = {}
    for t, f in list(names.items()) + [("V", "VOCR5_V_w8.vocr")]:
        gate, two = blob_heads(md / f)
        rt = Vocr(str(md / f), mode="w8i", max_width=1024, head_gate="on" if gate else "off",
                  two_level=1 if two else 0)
        rts[t] = (rt, VocrLM(rt, str(lm), **lm_set), two)
    regions = dev_regions(s["pagepool_dir"], a.per_title, a.counter_regions, a.vertical_regions)
    imgs, rgbas, metas = [], [], []
    for r, reg, kind in regions:
        bgr = cv2.imread(str(Path(s["pages_dir"]) / r["image"]), cv2.IMREAD_COLOR)
        rgb2, grey2, sc = console_frame(np.ascontiguousarray(bgr[..., ::-1]), r["platform"])
        x0, y0, x1, y1 = (v * sc for v in reg)
        g = np.ascontiguousarray(grey2[y0:y1, x0:x1])
        if g.shape[1] > 960 or g.shape[0] > 544:
            continue
        imgs.append(g)
        rgbas.append(np.concatenate([np.ascontiguousarray(rgb2[y0:y1, x0:x1]),
                                     np.full((g.shape[0], g.shape[1], 1), 255, np.uint8)], axis=2))
        metas.append((r["title_id"], r["page"], r["platform"], kind))
    mine = job_crops(rp, imgs, [m[2] for m in metas])
    res = {"regions": len(imgs), "kinds": {k: sum(1 for m in metas if m[3] == k) for k in ("horizontal", "counter",
                                                                                         "vertical")},
           "fix": fix, "finder_source": fsrc["tree"], "vertical_check": rp["vertical_check"],
           "rubi_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=a.rubi, capture_output=True,
                                       text=True).stdout.strip(), "lm": lm_name, "lm_settings": lm_set,
           "switched_mine": sum(1 for x in mine if x[4]), "tiers": {}, "differences": []}
    reads = {}
    for t in TIERS + ("V",):
        reads[t] = [[rubi_line(rts[t][0], rts[t][1], c, d, rts[t][2], (6000, 4000)) for c, d in m[3]]
                    if (m[4] == (t == "V")) or (t != "V" and not m[4]) else None for m in mine]
    for t in TIERS:
        jobs = work / f"jobs_{t}.bin"
        with open(jobs, "wb") as f:
            for i, (rgba, (tid, page, plat, kind)) in enumerate(zip(rgbas, metas)):
                f.write(f"r{i}:{tid}:{page} {rgba.shape[1]} {rgba.shape[0]} {1 if plat == 'psp' else 0}\n".encode())
                f.write(np.ascontiguousarray(rgba).tobytes())
        tier_no = {"S": 1, "N": 2, "L": 3}[t]
        with open(jobs, "rb") as fin:
            out = subprocess.run([str(a.page_ocr), str(md), "--tier", str(tier_no), "--lm", "2", "--cascade", "on",
                                  "--vertical", "auto"] + (["--line-fix", "on"] if fix else []),
                                 stdin=fin, capture_output=True, timeout=7200)
        if out.returncode:
            raise SystemExit(f"vjo-page-ocr failed: {out.stderr.decode()[:400]}")
        theirs = [json.loads(x) for x in out.stdout.decode("utf-8").splitlines() if x.strip()]
        e = defaultdict(int)
        nt = NEXT[t]
        for i, (m, th) in enumerate(zip(mine, theirs)):
            boxes, dark, st, crops, cols = m
            if cols:
                comp = [x["b"] if x["g"] else "" for x in reads["V"][i]]
            else:
                comp = [compose(x, reads[nt][i][k] if nt else None, 700) for k, x in enumerate(reads[t][i])]
            shown = [z for z in (q if has_japanese(q) else "" for q in cnorm(comp)) if z]
            tb = [tuple(x["box"]) for x in th["lines"]]
            e["regions"] += 1
            e["switched_equal"] += bool(th["switched"]) == cols
            same_boxes = tb == [tuple(b) for b in boxes] and [x["dark"] for x in th["lines"]] == list(dark)
            e["boxes_equal"] += same_boxes
            e["lines"] += len(boxes)
            if same_boxes:
                e["line_text_equal"] += sum(x["text"] == c for x, c in zip(th["lines"], comp))
            e["shown_equal"] += th["text"] == "\n".join(shown)
            e["columns"] += cols
            if not same_boxes or th["text"] != "\n".join(shown) or bool(th["switched"]) != cols:
                res["differences"].append({"title": metas[i][0], "page": metas[i][1], "kind": metas[i][3],
                                           "tier": t, "switched": [bool(th["switched"]), cols],
                                           "boxes_equal": same_boxes, "lines": [len(th["lines"]), len(boxes)]})
        res["tiers"][t] = dict(e)
    res["all_equal"] = not res["differences"] and all(v["regions"] == len(imgs) for v in res["tiers"].values())
    write_json(a.out, res)
    print(json.dumps({k: v for k, v in res.items() if k != "differences"}))


# ----------------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("stage")
    p.add_argument("--s", required=True)
    p.add_argument("--n", required=True)
    p.add_argument("--l", required=True)
    p.add_argument("--message", default="")
    p.add_argument("--out", required=True)
    p = sp.add_parser("rubi-path")
    p.add_argument("--rubi", required=True)
    p.add_argument("--out", required=True)
    p = sp.add_parser("pcsg-a")
    p.add_argument("--family", required=True, choices=FAMILIES)
    p.add_argument("--models", help="MODELS_DIR (VOCR6)")
    p.add_argument("--build-cache", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--procs", type=int, default=5)
    p.add_argument("--out", required=True)
    p = sp.add_parser("plan")
    p.add_argument("--set", required=True, choices=SETS_B)
    p.add_argument("--rubi-path", required=True)
    p.add_argument("--data", help="eval screenshots (vita-test, psp-test)")
    p.add_argument("--out", required=True)
    p = sp.add_parser("read")
    p.add_argument("--plan", required=True)
    p.add_argument("--blob", required=True)
    p.add_argument("--rubi-path", required=True)
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--only", default="rows", choices=("rows", "columns", "all"),
                   help="rows: every crop but the columns of a region auto read as vertical (a tier's read); columns: "
                        "those alone (the vertical model's read)")
    p.add_argument("--out", required=True)
    p = sp.add_parser("score-b")
    p.add_argument("--plan", required=True)
    p.add_argument("--dump", action="append", required=True, help="FAMILY:TIER=DUMP.pkl")
    p.add_argument("--rubi-path", required=True)
    p.add_argument("--out", required=True)
    p = sp.add_parser("validate")
    p.add_argument("--priv", required=True)
    p.add_argument("--suffix", required=True)
    p.add_argument("--out", required=True)
    p = sp.add_parser("selftest")
    p.add_argument("--work", required=True)
    p.add_argument("--rubi", required=True)
    p.add_argument("--k", type=int, default=3000)
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--parts", default="finder,plain,gated")
    p.add_argument("--out", required=True)
    p = sp.add_parser("dryrun")
    p.add_argument("--work", required=True)
    p.add_argument("--rubi", required=True)
    p.add_argument("--vocr6", required=True, help="S,N,L stand-in blobs (development: any VOCR6 candidates)")
    p.add_argument("--validation", required=True, help="the validation's private directory (path (a) outputs)")
    p.add_argument("--suffix", default="v6ftcheck")
    p.add_argument("--pages", type=int, default=12)
    p.add_argument("--workers", type=int, default=5)
    p = sp.add_parser("rubicheck")
    p.add_argument("--work", required=True)
    p.add_argument("--rubi", required=True)
    p.add_argument("--vjo-ocr", required=True, help="Rubi's host tool (build/vjo-ocr of the --rubi checkout)")
    p.add_argument("--models", required=True, help="a directory with the models and LMs that checkout pins")
    p.add_argument("--fix", default="0", help="the finder fix Rubi's checkout runs (0 until it pins #129's)")
    p.add_argument("--per-title", type=int, default=6)
    p.add_argument("--counter-regions", type=int, default=150)
    p.add_argument("--out", required=True)
    p = sp.add_parser("pagecheck")
    p.add_argument("--work", required=True)
    p.add_argument("--rubi", required=True)
    p.add_argument("--page-ocr", required=True, help="Rubi's host tool vjo-page-ocr (build of the --rubi checkout)")
    p.add_argument("--models", required=True, help="a directory with the models and LMs that checkout pins")
    p.add_argument("--fix", default="0", help="the finder fix Rubi's checkout runs (0 until it pins #129's)")
    p.add_argument("--per-title", type=int, default=4)
    p.add_argument("--counter-regions", type=int, default=60)
    p.add_argument("--vertical-regions", type=int, default=60)
    p.add_argument("--out", required=True)
    p = sp.add_parser("report")
    p.add_argument("--priv", required=True)
    p.add_argument("--suffix", required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    {"stage": cmd_stage, "rubi-path": cmd_rubi_path, "pcsg-a": cmd_pcsg_a, "plan": cmd_plan, "read": cmd_read,
     "score-b": cmd_score_b, "validate": cmd_validate, "selftest": cmd_selftest, "dryrun": cmd_dryrun,
     "rubicheck": cmd_rubicheck, "pagecheck": cmd_pagecheck, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
