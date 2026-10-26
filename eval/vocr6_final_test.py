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
    lm = resolve_lm(src, proto)
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
    if not doc["vocr5_in_rubi_matches_protocol"]:
        raise SystemExit("Rubi's VOCR5 pins differ from the protocol's gated VOCR5 blobs")
    write_json(out, doc, exclusive=True)
    print(json.dumps({"rubi_head": head, "lm": lm["candidate"], "settings": lm["settings"], "fix": fix,
                      "finder_tree": fsrc["tree"], "finder_commit": fsrc["commit"][:12]}))


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

def find_boxes(exe, imgs, platforms, fix, probe=1):
    """Rubi's line path on grey regions (pagepath_find): [(boxes [(x, y, w, h)], dark flags, rc)]."""
    import pagepath as PP
    jobs = [(img, FINDER_RECOVERY, fix, probe, REFINE_ALL if plat == "psp" else 0)
            for img, plat in zip(imgs, platforms)]
    out = []
    for i in range(0, len(jobs), 256):
        for rc, found, probed, bx in PP.run_driver(exe, jobs[i:i + 256]):
            # a finder error is Rubi's job failing on the region (VJO_E_OCR_REGION): no line, no text; counted
            out.append(([tuple(b[:4]) for b in bx] if rc == 0 else [], [int(b[4]) for b in bx] if rc == 0 else [],
                        {"found": found, "probed": probed, "rc": rc}))
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

    def crop(self, img, inv):
        self.doc["crops"].append((np.ascontiguousarray(img, np.uint8), bool(inv)))
        return len(self.doc["crops"]) - 1


def plan_tests(name, rp, plan, data):
    """vita-test / psp-test: fixed lines (box + pad as eval/sets.py cuts them, x2 from the 2x frame on PSP; invert =
    line_polarity of the grey crop) and whole-dialogue regions (base / +4 / +8, without the regions that hold an
    uncertain line) through Rubi's finder."""
    import sets
    from fetch import shot_path
    from PIL import Image

    from common import line_polarity, to_gray
    platform = sets.SETS[name]
    pad = sets.PAD[platform]
    rows = sets.read_manifest(name)
    frames = {}

    def frame(r):
        p = shot_path(data, platform, r["vn"], r["url"])
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
    found = find_boxes(rp["finder"]["driver"], imgs, [i["platform"] for i in infos], rp["finder"]["fix"])
    for img, info, (boxes, dark, st) in zip(imgs, infos, found):
        info.update(boxes=[list(b) for b in boxes], dark=dark, finder=st,
                    crops=[plan.crop(img[y:y + h, x:x + w], d) for (x, y, w, h), d in zip(boxes, dark)])
        plan.doc["units"].append(info)
    plan.doc["inputs"] = {"manifest": f"eval/manifests/{name}.jsonl",
                          "manifest_sha256": sha256(sets.MANIFESTS / f"{name}.jsonl"),
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
    found = find_boxes(rp["finder"]["driver"], imgs, ["vita"] * len(imgs), rp["finder"]["fix"])
    for img, info, (boxes, dark, st) in zip(imgs, infos, found):
        info.update(boxes=[list(b) for b in boxes], dark=dark, finder=st,
                    crops=[plan.crop(img[y:y + h, x:x + w], d) for (x, y, w, h), d in zip(boxes, dark)])
        plan.doc["units"].append(info)
    plan.doc["inputs"] = {"refs_sha256": freeze["refs_sha256"], "lines_sha256": meta["lines_sha256"],
                          "pages": len(by_page), "pages_scored": len(infos), "pages_skipped": dict(skipped)}


def plan_pcsg(rp, plan, pools, pagepool, pages):
    """The test title: its clean line crops (invert off) and eval/pagepath.py's hooked units of the test split, cut
    by Rubi's finder and, for the decomposition, by the reference boxes plus pad (pagepath's oracle)."""
    import pagepath as PP

    import import_hooked as IH
    pools, pagepool, pages = Path(pools), Path(pagepool), Path(pages)
    f = pools / "test" / "line" / "PCSG00807.npz"
    for i, (m, lab, g) in enumerate(IH.read_npz(f)):
        plan.doc["fixed"].append({"id": m["id"], "vn": "v18432", "page": m["page"], "platform": m["platform"],
                                  "name": False, "styles": [], "text": lab, "crop": plan.crop(g, False)})
    units, umeta = PP.hooked_units(pagepool, {"test"})
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
    found = find_boxes(rp["finder"]["driver"], imgs, [u["platform"] for u, _, _, _ in infos], rp["finder"]["fix"])
    from common import line_dark
    for img, (u, s, o, pts), (boxes, dark, st) in zip(imgs, infos, found):
        base = {"cluster": u["cluster"], "vn": "v18432", "page": u["page"], "platform": u["platform"],
                "variant": u["variant"], "lines": [{"id": ln["id"], "text": ln["text"], "styles": []}
                                                   for ln in u["lines"]], "points": pts}
        plan.doc["units"].append(dict(base, uid=u["uid"], kind="page", boxes=[list(b) for b in boxes], dark=dark,
                                      finder=st, crops=[plan.crop(img[y:y + h, x:x + w], d)
                                                        for (x, y, w, h), d in zip(boxes, dark)]))
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
    jobs = [(i, img, inv) for i, (img, inv) in enumerate(plan["crops"])]
    res = [None] * len(jobs)
    t0 = time.time()
    with get_context("fork").Pool(a.workers, initializer=_init,
                                  initargs=(str(a.blob), lm, settings, rp["counter"])) as pool:
        for i, r in pool.imap_unordered(_read, jobs, chunksize=16):
            res[i] = r
    doc = {"kind": "vocr6-final-test-dump", "plan_sha256": sha256(a.plan), "blob": str(a.blob),
           "blob_sha256": sha256(a.blob), "head_gate": gate, "two_level": two, "lm_sha256": rp["lm"]["sha256"],
           "lm_settings": settings, "counter": rp["counter"],
           "library_sha256": sha256(ROOT / "runtime/build/libvocr.so"),
           "repo_head": git_head(), "crops": res, "seconds": round(time.time() - t0, 1), "at": now()}
    write_pickle(out, doc)
    print(json.dumps({"crops": len(res), "head_gate": gate, "two_level": two, "rereads": sum(r["reread"] for r in res),
                      "errors": sum(1 for r in res if r.get("error")), "seconds": doc["seconds"]}))


# ----------------------------------------------------------------------------- score (path b)

class Norm:
    """Rubi's core/textnorm.c over many lines in one process (eval/pagepath.py rubi_norm_tool)."""

    def __init__(self, exe):
        self.exe = str(exe)

    def __call__(self, texts):
        import pagepath as PP
        return PP.textnorm(self.exe, list(texts))


def stage_texts(dumps, family, tier, casc_conf, cnorm):
    """{stage: [text per crop]} of one family's tier: greedy, rubi_tier, rubi_cascade, shipped; and the crops whose
    read failed (the job tier's, or the cascade tier's for a crop it reads again)."""
    own = dumps[(family, tier)]["crops"]
    nt = NEXT[tier]
    up = dumps[(family, nt)]["crops"] if nt else None
    g = [c["g"] for c in own]
    rt = [c["b"] if c["g"] else "" for c in own]
    rc = [compose(c, up[i] if up else None, casc_conf) for i, c in enumerate(own)]
    sh = [t if has_japanese(t) else "" for t in cnorm(rc)]
    failed = {i for i, c in enumerate(own) if c.get("error") or
              (up is not None and c["g"] and c["conf"] < casc_conf and up[i].get("error"))}
    return {"greedy": g, "rubi_tier": rt, "rubi_cascade": rc, "shipped": sh, "failed": failed}


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
    fams = sorted({f for f, _ in dumps})
    cnorm = Norm(rp["textnorm"]["exe"])
    import pagepath as PP
    rows = {"fixed": [], "units": []}
    texts = {(f, t): stage_texts(dumps, f, t, rp["cascade_conf"], cnorm) for f in fams for t in TIERS}
    for fx in plan["fixed"]:
        ref = norm(fx["text"])
        r = {k: fx.get(k) for k in ("id", "vn", "title", "page", "platform", "image", "name", "styles")}
        r["chars"] = len(ref)
        r["errors"] = {f"{f}:{t}:{s}": lev("" if fx["crop"] in texts[(f, t)]["failed"] else
                                           norm(texts[(f, t)][s][fx["crop"]]), ref)
                       for f in fams for t in TIERS for s in STAGES}
        rows["fixed"].append(r)
    for u in plan["units"]:
        r = {k: u.get(k) for k in ("uid", "kind", "vn", "title", "page", "platform", "image", "variant", "cluster")}
        r["lines"] = len(u["lines"])
        r["boxes"] = len(u["crops"])
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
            r["line_stats"] = {k: status.count(k) for k in ("whole", "cut", "missed")}
            r["line_stats"].update({k: kinds.count(k) for k in ("merged", "junk")})
        rows["units"].append(r)
    doc = {"kind": "vocr6-final-test-b-rows", "set": plan["set"], "plan_sha256": psha,
           "dumps": {f"{f}:{t}": {k: d[k] for k in ("blob_sha256", "head_gate", "two_level", "lm_sha256",
                                                     "lm_settings", "library_sha256", "seconds")}
                     for (f, t), d in dumps.items()},
           "rereads": {f"{f}:{t}": sum(c["reread"] for c in d["crops"]) for (f, t), d in dumps.items()},
           "read_errors": {f"{f}:{t}": sum(1 for c in d["crops"] if c.get("error")) for (f, t), d in dumps.items()},
           "finder_errors": sum(1 for u in plan["units"] if (u.get("finder") or {}).get("rc")),
           "cascaded": {f"{f}:{t}": sum(1 for c in dumps[(f, t)]["crops"] if c["g"] and c["conf"] < rp["cascade_conf"])
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
    rep = json.loads((Path(priv) / "a" / "shared" / "report.json").read_text(encoding="utf-8"))["shared"]
    units = rep["dialogue"]["units"]
    rows = {}
    for f in families:
        for t in TIERS:
            for u in units[f"{f}_{t}"]:
                img = u["unit"].split("|", 1)[1].split("@")[0]
                r = rows.setdefault((u["unit"], u["variant"]), {
                    "id": f"{u['unit']}|{u['variant']}", "set": u["set"], "vn": u["vn"], "platform": u["platform"],
                    "variant": u["variant"], "image": img, "chars": u["body_full_chars"],
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
    return out


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
           "rubi_path": None, "a": {}, "b": {}, "seen_flags": proto["seen_flags"]}
    # path (a)
    fixed = a_fixed_rows(priv, fams)
    rep["a"]["vita_psp_fixed_lines"] = set_tables(fixed, fams, ("greedy", "lm"), cluster=lambda r: r["vn"],
                                                  ident=lambda r: r["id"])
    rep["a"]["vita_psp_fixed_lines_per_title"] = per_title(fixed, fams, ("greedy", "lm"))
    dia = a_dialogue_rows(priv, fams)
    rep["a"]["vita_psp_whole_dialogue"] = region_tables(dia, fams, ("greedy",), cluster=lambda r: r["vn"])
    rep["a"]["vita_psp_whole_dialogue_per_title"] = per_title([r for r in dia if r["variant"] == "base"], fams,
                                                              ("greedy",))
    cl = a_clannad_rows(priv, fams, suffix)
    rep["a"]["clannad_lines"] = {
        "dialogue": table(sub(cl, lambda r: not r["name"]), keys_for(fams, ("greedy",)), cluster=lambda r: r["page"],
                          ident=lambda r: r["id"], exact=True, min_clusters=2),
        "names": table(sub(cl, lambda r: r["name"]), keys_for(fams, ("greedy",)), cluster=lambda r: r["page"],
                       ident=lambda r: r["id"], exact=True, min_clusters=2),
        "all": table(cl, keys_for(fams, ("greedy",)), cluster=lambda r: r["page"], ident=lambda r: r["id"],
                     exact=True, min_clusters=2)}
    pc = a_pcsg_rows(priv, fams)
    import vocr6_baseline as VB
    pk = keys_for(fams, VB.DECODES)
    rep["a"]["pcsg00807"] = {
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
    rep["a"]["tool_outputs"] = {
        "vocr3_report_pairs": json.loads((priv / "a" / "shared" / "shared.json").read_text(encoding="utf-8"))
        .get("pairs", {}),
        "clannad": json.loads((priv / "clannad" / f"clannad_{suffix}.json").read_text(encoding="utf-8"))["scores"],
        "clannad_dialogue_pairs": json.loads((priv / "clannad" / f"clannad_{suffix}_dialogue.json")
                                             .read_text(encoding="utf-8"))["pairs"],
        "resources": json.loads((priv / "a" / "shared" / "shared.json").read_text(encoding="utf-8"))["resources"]}
    rep["a"]["tool_outputs"]["vocr3_report_pairs"] = R_compact_pairs(rep["a"]["tool_outputs"]["vocr3_report_pairs"])
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
        rep["b"]["vita_psp_fixed_lines"] = set_tables(fx, fams, STAGES, cluster=lambda r: r["vn"],
                                                      ident=lambda r: r["id"])
        rep["b"]["vita_psp_fixed_lines_per_title"] = per_title(fx, fams, ("shipped",))
        ru = [dict(r, set=s, id=r["uid"]) for s in ("vita-test", "psp-test") for r in b[s]["rows"]["units"]]
        rep["b"]["vita_psp_whole_dialogue"] = region_tables(ru, fams, STAGES, cluster=lambda r: r["vn"])
        rep["b"]["vita_psp_whole_dialogue_per_title"] = per_title([r for r in ru if r["variant"] == "base"], fams,
                                                                  ("shipped",))
        cfx = b["clannad"]["rows"]["fixed"]
        rep["b"]["clannad_lines"] = {
            "dialogue": table(sub(cfx, lambda r: not r["name"]), keys_for(fams, STAGES), cluster=lambda r: r["page"],
                              ident=lambda r: r["id"], exact=True, min_clusters=2),
            "names": table(sub(cfx, lambda r: r["name"]), keys_for(fams, STAGES), cluster=lambda r: r["page"],
                           ident=lambda r: r["id"], exact=True, min_clusters=2)}
        cpu = b["clannad"]["rows"]["units"]
        rep["b"]["clannad_pages"] = {
            "dialogue": table(cpu, keys_for(fams, STAGES), cluster=lambda r: r["page"], ident=lambda r: r["uid"],
                              min_clusters=2),
            "all": table(cpu, keys_for(fams, STAGES), chars_key="chars_all", err_key="errors_all",
                         cluster=lambda r: r["page"], ident=lambda r: r["uid"], min_clusters=2),
            "inputs": b["clannad"]["inputs"]}
        pfx = b["pcsg00807"]["rows"]["fixed"]
        rep["b"]["pcsg00807_clean"] = table(pfx, keys_for(fams, STAGES), cluster=lambda r: r["page"],
                                            ident=lambda r: r["id"], exact=True, min_clusters=2)
        pu = b["pcsg00807"]["rows"]["units"]
        rep["b"]["pcsg00807_page"] = {}
        for kind in ("page", "oracle"):
            for v in list(VARIANTS) + ["all-variants"]:
                rs = sub(pu, lambda r: r["kind"] == kind and (v == "all-variants" or r["variant"] == v))
                t = table(rs, keys_for(fams, STAGES), cluster=lambda r: r["page"], ident=lambda r: r["uid"],
                          min_clusters=2)
                if kind == "page":
                    t["line_stats"] = {k: sum(r["line_stats"][k] for r in rs)
                                       for k in ("whole", "cut", "missed", "merged", "junk")}
                    t["lines"] = sum(r["lines"] for r in rs)
                    t["boxes"] = sum(r["boxes"] for r in rs)
                rep["b"]["pcsg00807_page"][f"{kind}|{v}"] = t
        rep["b"]["reads"] = {s: {k: b[s][k] for k in ("dumps", "rereads", "cascaded", "read_errors", "finder_errors",
                                                      "finder", "inputs")} for s in SETS_B}
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
    h = {"a": {}, "b": {}}
    a = rep["a"]
    h["a"]["clannad_lines"] = pick(a["clannad_lines"]["dialogue"], "greedy")
    h["a"]["vita_psp_fixed_lines"] = pick(a["vita_psp_fixed_lines"]["pooled|dialogue"], "greedy")
    h["a"]["vita_psp_fixed_lines_lm"] = pick(a["vita_psp_fixed_lines"]["pooled|dialogue"], "lm")
    h["a"]["vita_psp_whole_dialogue_base"] = pick(a["vita_psp_whole_dialogue"]["pooled|base|dialogue"], "greedy")
    h["a"]["pcsg00807_clean"] = pick(a["pcsg00807"]["clean"], "rubi_cascade")
    h["a"]["pcsg00807_page"] = pick(a["pcsg00807"]["page_refine_console_base"], "rubi_cascade")
    b = rep.get("b") or {}
    if b:
        h["b"]["clannad_lines"] = pick(b["clannad_lines"]["dialogue"], "shipped")
        h["b"]["clannad_pages"] = pick(b["clannad_pages"]["dialogue"], "shipped")
        h["b"]["vita_psp_fixed_lines"] = pick(b["vita_psp_fixed_lines"]["pooled|dialogue"], "shipped")
        h["b"]["vita_psp_whole_dialogue_base"] = pick(b["vita_psp_whole_dialogue"]["pooled|base|dialogue"], "shipped")
        h["b"]["pcsg00807_clean"] = pick(b["pcsg00807_clean"], "shipped")
        h["b"]["pcsg00807_page"] = pick(b["pcsg00807_page"]["page|base"], "shipped")
    return h


# ----------------------------------------------------------------------------- markdown

def pct(r):
    return "-" if not r or r.get("cer") is None else f"{100 * r['errors'] / max(1, r['chars']):.2f}%"


def ci(c):
    return "" if not c else f" [{c[0]:.2f}, {c[1]:.2f}]"


def cell(r):
    return "-" if not r else pct(r) + ci(r.get("ci95"))


def pcell(p):
    if not p:
        return "-"
    return f"{p['delta']:+.2f}" + (f" [{p['ci95'][0]:+.2f}, {p['ci95'][1]:+.2f}]" if p.get("ci95") else "")


def md_measure(title, tbl, stage, note=""):
    md = [f"**{title}**{note}", "", "| tier | VOCR5 | VOCR6 | VOCR6 - VOCR5 (points) |", "|---|---|---|---|"]
    for t in TIERS:
        md.append(f"| {t} | {cell(tbl['rates'].get(f'VOCR5:{t}:{stage}'))} | "
                  f"{cell(tbl['rates'].get(f'VOCR6:{t}:{stage}'))} | {pcell(tbl['pairs'].get(f'{t}:{stage}'))} |")
    md.append("")
    return md


def markdown(rep, proto):
    md = ["# VOCR6 final test (read once)", "",
          f"Protocol `eval/vocr6_final_test_protocol.json` (sha256 `{rep['protocol_sha256'][:16]}…`), repo "
          f"`{rep['repo_head'][:7]}`, read {rep['attempt'].get('started')}. Dialogue CER (names a secondary column), "
          "95% intervals: game (VN) bootstrap for vita-test / psp-test, page bootstrap for CLANNAD and the test title; "
          "VOCR6 - VOCR5 paired over the same units. Host measurements; nothing ran on a console.", ""]
    m = rep["models"]["tiers"]
    md += ["| tier | VOCR6 blob | sha256 | bytes | head gate | two-level |", "|---|---|---|---:|---|---|"]
    for t in TIERS:
        e = m[t]
        md.append(f"| {t} | `{e['file']}` | `{e['sha256'][:16]}…` | {e['bytes']:,} | {e['head_gate']} | "
                  f"{e['two_level']} |")
    md.append("")
    if rep.get("rubi_path"):
        rp = rep["rubi_path"]
        md += [f"Path (b) is Rubi main `{rp['rubi_head'][:7]}`: LM `{Path(rp['lm']['file']).name}` "
               f"({rp['lm']['candidate']}; alpha {rp['lm']['settings']['alpha']}, beta {rp['lm']['settings']['beta']}, "
               f"flags {rp['lm']['settings']['flags']}), cascade under {rp['cascade_conf']}, finder TEXTBOX_RECOVERY + "
               f"short-line probe + fix {rp['finder']['fix']} (COLGAP; source {rp['finder']['source']['tree']} "
               f"`{rp['finder']['source']['commit'][:7]}`), refinement on PSP, textnorm `"
               f"{rp['textnorm']['textnorm_sha256'][:12]}…`, the non-Japanese line filter; PSP in the 2x console "
               "view.", ""]
    md += ["## Headline", ""]
    b = rep.get("b") or {}
    if b:
        md += ["### Path (b): Rubi's shipped path", ""]
        md += md_measure("CLANNAD, 2,052 dialogue lines (the frozen line crops)", b["clannad_lines"]["dialogue"],
                         "shipped")
        md += md_measure("CLANNAD pages (Rubi's finder on the ROI; pages with a complete reference)",
                         b["clannad_pages"]["dialogue"], "shipped")
        md += md_measure("vita-test + psp-test fixed lines, dialogue", b["vita_psp_fixed_lines"]["pooled|dialogue"],
                         "shipped")
        md += md_measure("vita-test + psp-test whole dialogue (base)",
                         b["vita_psp_whole_dialogue"]["pooled|base|dialogue"], "shipped")
        md += md_measure("PCSG00807 clean lines", b["pcsg00807_clean"], "shipped")
        md += md_measure("PCSG00807 page path (base regions)", b["pcsg00807_page"]["page|base"], "shipped")
    a = rep["a"]
    md += ["### Path (a): VOCR5's locked path", ""]
    md += md_measure("CLANNAD dialogue lines (greedy)", a["clannad_lines"]["dialogue"], "greedy")
    md += md_measure("vita-test + psp-test fixed lines, dialogue (greedy, console polarity)",
                     a["vita_psp_fixed_lines"]["pooled|dialogue"], "greedy")
    md += md_measure("the same with beam 8 + ctclm.2's 1 MB LM", a["vita_psp_fixed_lines"]["pooled|dialogue"], "lm")
    md += md_measure("vita-test + psp-test whole dialogue (base, GRAD)", a["vita_psp_whole_dialogue"]
                     ["pooled|base|dialogue"], "greedy")
    md += md_measure("PCSG00807 clean lines (rubi_cascade)", a["pcsg00807"]["clean"], "rubi_cascade")
    md += md_measure("PCSG00807 page GRAD + refine (base, rubi_cascade)", a["pcsg00807"]["page_refine_console_base"],
                     "rubi_cascade")
    md += ["## Vita and PSP apart", ""]
    hdr = "| measure | tier | " + " | ".join(f"{s} VOCR5 | {s} VOCR6 | {s} Δ" for s in ("vita-test", "psp-test")) + \
          " |"
    md += [hdr, "|---|---|" + "---|" * 6]
    for path, key, stage, tag in (("b", "vita_psp_fixed_lines", "shipped", "(b) fixed lines"),
                                  ("b", "vita_psp_whole_dialogue", "shipped", "(b) whole dialogue base"),
                                  ("a", "vita_psp_fixed_lines", "greedy", "(a) fixed lines greedy"),
                                  ("a", "vita_psp_fixed_lines", "lm", "(a) fixed lines + LM"),
                                  ("a", "vita_psp_whole_dialogue", "greedy", "(a) whole dialogue base")):
        tbls = rep.get(path, {}).get(key)
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
    md += ["## Names (secondary)", "", "| measure | tier | VOCR5 | VOCR6 | Δ |", "|---|---|---|---|---|"]
    for path, tb, stage, tag in (("b", (b.get("vita_psp_fixed_lines") or {}).get("pooled|names"), "shipped",
                                  "(b) vita+psp fixed lines"),
                                 ("b", (b.get("clannad_lines") or {}).get("names"), "shipped", "(b) CLANNAD lines"),
                                 ("a", a["vita_psp_fixed_lines"].get("pooled|names"), "greedy", "(a) vita+psp fixed"),
                                 ("a", a["clannad_lines"]["names"], "greedy", "(a) CLANNAD lines")):
        if not tb:
            continue
        for t in TIERS:
            md.append(f"| {tag} | {t} | {cell(tb['rates'].get(f'VOCR5:{t}:{stage}'))} | "
                      f"{cell(tb['rates'].get(f'VOCR6:{t}:{stage}'))} | {pcell(tb['pairs'].get(f'{t}:{stage}'))} |")
    md.append("")
    if b:
        md += ["## Path (b) decomposition (dialogue CER)", "",
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
    md += ["## Per title (vita-test, psp-test; dialogue)", ""]
    for path, key, stage in (("b", "vita_psp_fixed_lines_per_title", "shipped"),
                             ("b", "vita_psp_whole_dialogue_per_title", "shipped"),
                             ("a", "vita_psp_fixed_lines_per_title", "greedy"),
                             ("a", "vita_psp_whole_dialogue_per_title", "greedy")):
        pt = rep.get(path, {}).get(key)
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
    print(json.dumps(rep["headline"]["b"] or rep["headline"]["a"], ensure_ascii=False)[:4000])


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


def selftest_plain(plan_path, dump_paths, blobs, lm, settings, k, workers, rubi_norm):
    """Path (b)'s line read with plain blobs (no head gate) against vocr6-lead's hval dumps of the same plan (eval/
    vocr6_score.py read): greedy, the beam text and the confidence per crop; then the cascade, textnorm and the
    filter against vocr6_score.texts(rubi, up=the next tier)."""
    import vocr6_score as VS
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
    out["finder"] = selftest_finder(src, Path(a.work) / "bin", lead / "plans" / "hval_plan_v2.pkl",
                                    A / "vocr6-ingest" / "pages", a.limit)
    rel = A / "vocr5-train" / "release" / "v0.3.0"
    lmc = proto["paths"]["b"]["lm_candidates"]["ctclm2_1000k"]
    import pagepath as PP
    rubi_norm = PP.rubi_norm_tool(a.rubi, Path(a.work) / "bin")[0]
    out["plain"] = selftest_plain(lead / "plans" / "hval_plan_v1.pkl",
                                  {"N": lead / "scores" / "dump_hval_VOCR5-N.json",
                                   "L": lead / "scores" / "dump_hval_VOCR5-L.json"},
                                  {"N": rel / "VOCR5_N_w8.vocr", "L": rel / "VOCR5_L_w8.vocr"},
                                  lmc["file"], lmc["settings"], a.k, a.workers, rubi_norm)
    out["gated"] = selftest_gated(A / "vocr6-ingest" / "baseline" / "hval", TIERS, a.k, a.workers, lmc["file"])
    write_json(a.out, out)
    print(json.dumps({k: v for k, v in out.items() if k in ("finder", "plain", "gated")}))


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
    p.add_argument("--out", required=True)
    p = sp.add_parser("report")
    p.add_argument("--priv", required=True)
    p.add_argument("--suffix", required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    {"stage": cmd_stage, "rubi-path": cmd_rubi_path, "pcsg-a": cmd_pcsg_a, "plan": cmd_plan, "read": cmd_read,
     "score-b": cmd_score_b, "validate": cmd_validate, "selftest": cmd_selftest, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
