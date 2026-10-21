#!/usr/bin/env python3
"""textspot2: the scoring wrapper of issue #106 (docs/textspot2.md, eval/textspot2_protocol.json). eval/textspot.py
(the harness), eval/textspot_metrics.py and eval/textspot2_rule.py are imported and used unchanged; nothing here
re-implements a metric or the rule.

  textspot2.py cbuild --out DIR                      the C drivers from this checkout (textspot's, plus the d2s driver
                                                     runtime/vocr_textspot2_test once its sources exist)
  textspot2.py seed-cache [--cache DIR]              copy textspot's development reading cache once (never written)
  textspot2.py fetch-teacher [--models DIR]          meiki.text.detect v0.1 320x192 (and its inference script) at the
                                                     pinned revision, sha256-checked
  textspot2.py run --runs DIR [--names RUN ...] --out DIR --cbin DIR [--models N]
      every finished run (done.json; listed in finished.txt when the file exists) as the arms T2_<run>_det and
      T2_<run>_detref: eval/textspot.py's det / detref candidate specs through its evaluate path (the twins, C parity
      per frame, the reading by VOCR5-N for horizontal boxes and VOCR5-V for vertical ones, frame_unit, summarize),
      on the development set; an arm already scored for its blob is skipped. A VTSP version-2 (d2s) blob goes through
      synth/textspot2_twin.py and runtime/vocr_textspot2_test (v2_dispatch); a version-1 blob through textspot's
      twin and driver, exactly as textspot ran them.
  textspot2.py ceiling --out DIR [--cbin DIR]        the teacher ceiling of amendment 1 (report only, never a
                                                     candidate): meiki960_det / _detref and meiki320_det / _detref
  textspot2.py headline --units DIR [--a9 JSON]      every seed's F1, CER all, dialogue, est. ms and the
                                                     representatives (median seed) as markdown, for the conductor
  textspot2.py select --units DIR ... --a9 JSON --out eval/textspot2_dev_<date>.json
      the seeds, their eligibility inputs (eval/textspot.py eligibility; the A9 rows of textspot's table for
      textspot's shapes, else textspot2's), the representatives (textspot2_rule.median_seed; r0's seeds 1-2 are
      textspot's runs, their units read under $T/eval/famB and famC), textspot2_rule.apply_rule2 once over the
      representatives of every arm except r0, against the frozen baseline and the protocol's ref and fast_ref (all
      sha256-checked); the report: every comparison, the Pareto front, the secondary dialogue + name + choice column,
      what each arm bought, the ceiling, per part and per class. Compact JSON (counts, rates, hashes) and an indented
      copy under $T2.
  textspot2.py tables --dev JSON [--out MD]          markdown for docs/textspot2.md from the selection record

Paths: T2 = $HERD/shared/artifacts/textspot2 (units under T2/eval/units, the ceiling under T2/eval/ceiling, the
reading cache T2/eval/cache, the C drivers T2/eval/cbuild, the frames' pixel hashes T2/eval/frames.json); textspot's
artifacts ($T) and private files are read, never written.

The secondary column (textspot2_rule.dnc_units) needs every reference's errors, which the harness's units do not hold:
they are recomputed from an arm's boxes and the cached readings with textspot_metrics.e2e_frame, and every frame is
checked to give the stored units' all-text and dialogue errors exactly.

The teacher ceiling: teacher/engines.Meiki's detector (MeikiOCR's preprocessing, BGR input) for the 960x544 file
pinned in teacher/engines.py; the 320x192 file of the same repository and revision as its inference.v0.1.320x192.py
runs it (a plain resize to 320x192, sizes [[320, 192]], int(coordinate / ratio)). Boxes of score >= 0.5 with a
positive size, clipped to the frame (the 960 path's np.clip; the same for 320), vertical when taller than wide
(MeikiOCR's rule), padded by the spotter's pad rule (textspot_metrics.pad_box: max(2, (across 18 + 50) / 100) on
every side, clipped); the _detref arms put their horizontal boxes through the refinement (eval/textspot.py
refine_spot_boxes). Blocks, order and pick by textspot_twin.group_and_order, dark flags by common.line_dark, as for
every baseline; read, scored and summarised exactly as the candidates are.
"""
import argparse
import contextlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "eval", ROOT / "synth", ROOT / "runtime", ROOT / "train"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import textspot as TS  # noqa: E402
import textspot2_rule as R  # noqa: E402
import textspot_metrics as M  # noqa: E402

import textspot2_shapes as S2  # noqa: E402

HERD = TS.HERD
T = TS.T                           # textspot's artifacts: read only
P = TS.P                           # textspot's private files: read only
T2 = TS.ART / "textspot2"
P2 = HERD / "shared" / "private" / "textspot2"
EVAL2 = T2 / "eval"
DEFAULT = {"units": EVAL2 / "units", "ceiling": EVAL2 / "ceiling", "cache": EVAL2 / "cache", "cbuild": EVAL2 / "cbuild",
           "rbuild": EVAL2 / "rbuild", "models": EVAL2 / "models", "frames": EVAL2 / "frames.json", "runs": T2 / "runs"}
PROTOCOL2 = ROOT / "eval" / "textspot2_protocol.json"
BASELINE = ROOT / "eval" / "textspot_baseline_20261019.json"
TS_DEV = ROOT / "eval" / "textspot_dev_20261019.json"
TS_A9 = ROOT / "eval" / "textspot_a9_20261019.json"
TS_CACHE = T / "eval" / "cache"
KINDS = ("det", "detref")
DECODE = {"det": "det", "detref": "det+refine"}
MHZ = 333 / 444
MAX_RECORD = 2 * 1024 * 1024       # the committed selection record stays under 2 MiB

# r0's seeds 1 and 2 are textspot's own runs: their development units are read, never re-run
R0_TEXTSPOT = {"r0_h2_s4_c24n4": {1: ("famB", "B_h2_s4_c24n4_seed1"), 2: ("famB", "B_h2_s4_c24n4_seed2")},
               "r0_h2_s4p_c8": {1: ("famC", "C_h2_s4p_c8_seed1"), 2: ("famC", "C_h2_s4p_c8_seed2")}}

# stage 1 as docs/textspot2.md "Arms and stages" lists it (38 runs)
STAGE1 = tuple([f"{r}_h2_s4_c24n4_seed{n}" for r in ("lab", "data", "ema", "cls") for n in (1, 2, 3)]
               + ["r0_h2_s4_c24n4_seed3"]
               + [f"long_h2_{s}_seed{n}" for s in ("s4_c24n4", "s4p_c8") for n in (1, 2, 3)]
               + [f"data_h2_{s}_seed{n}" for s in ("s4p_c8", "s4p_c6n4", "s4p_c8n4", "s4p_c10n4", "s4p_c12", "s4_c32n3")
                  for n in (1, 2, 3)]
               + ["r0_h2_s4p_c8_seed3"])

RUN_RE = re.compile(r"^(?P<recipe>[a-z][a-z0-9]*)_h2_(?P<shape>[a-z0-9_]+?)_seed(?P<seed>[0-9]+)$")


def protocol2(path=PROTOCOL2):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def code_identity():
    """The harness's code identity plus this wrapper's own files."""
    out = TS.code_identity()
    out.update(TS.code_identity(("eval/textspot2.py", "eval/textspot2_protected.py", "eval/textspot2_rule.py",
                                 "eval/textspot2_protocol.json", "train/textspot2_shapes.py",
                                 "synth/textspot2_twin.py", "runtime/vocr_textspot2.c")))
    return out


def parse_run(name):
    """'<recipe>_h2_<shape>_seed<n>' -> {"run", "recipe", "shape", "seed", "arm_id"}; the shape is the protocol's."""
    m = RUN_RE.match(name)
    if m is None or m.group("shape") not in S2.SHAPES:
        raise ValueError(f"{name!r} is not <recipe>_h2_<shape>_seed<n> of a protocol shape")
    recipe, shape, seed = m.group("recipe"), m.group("shape"), int(m.group("seed"))
    return {"run": name, "recipe": recipe, "shape": shape, "seed": seed, "arm_id": f"{recipe}_h2_{shape}"}


def is_under(path, base):
    try:
        Path(path).resolve().relative_to(Path(base).resolve())
        return True
    except ValueError:
        return False


def check_writable(*paths):
    """Nothing is ever written under textspot's artifacts or private files."""
    for p in paths:
        if p and (is_under(p, T) or is_under(p, P)):
            raise SystemExit(f"{p}: textspot's files are read only (write under $T2 / $P2)")


# ----------------------------------------------------------------------------- VTSP versions

TWIN2 = ROOT / "synth" / "textspot2_twin.py"
DRIVER2 = "vocr_textspot2_test"
DRIVER2_SOURCE_SETS = (("vocr_lines.c", "vocr_lines_rows.c", "vocr_textspot2.c", "vocr_textspot2_test.c"),
                       ("vocr_lines.c", "vocr_textspot2.c", "vocr_textspot2_test.c"),
                       ("vocr_textspot2.c", "vocr_textspot2_test.c"))


def blob_version(blob):
    """The VTSP header's version field (1: textspot's format; 2: the d2s extension)."""
    b = bytes(blob[:8])
    if len(b) < 8 or b[:4] != b"VTSP":
        raise ValueError("not a VTSP blob")
    return struct.unpack_from("<I", b, 4)[0]


def twin2():
    if not TWIN2.exists():
        raise SystemExit("a VTSP version-2 (d2s) blob needs synth/textspot2_twin.py (textspot2-runtime), which this "
                         "checkout does not have")
    import textspot2_twin
    return textspot2_twin


def model_of(blob):
    """The twin's model of a blob: textspot's twin for version 1, textspot2_twin for any other version."""
    if blob_version(blob) == 1:
        import textspot_twin as S
        return S.Model(blob)
    return twin2().Model(blob)


def shim_module(real):
    """A stand-in for synth/textspot_twin with every name of the real module, whose Model and spot send a version-1
    blob (and its model) to the real twin, unchanged, and any other version to textspot2_twin."""
    t2 = twin2()
    mod = types.ModuleType("textspot_twin")
    mod.__dict__.update(real.__dict__)

    def Model(blob):  # noqa: N802 (the twin's class name)
        return real.Model(blob) if blob_version(blob) == 1 else t2.Model(blob)

    def spot(model, img):
        return real.spot(model, img) if isinstance(model, real.Model) else t2.spot(model, img)

    mod.Model, mod.spot, mod.TEXTSPOT2_SHIM = Model, spot, True
    return mod


class CDrivers2(TS.CDrivers):
    """eval/textspot.py's C drivers; a version-2 blob goes to runtime/vocr_textspot2_test (textspot's batch protocol),
    a version-1 blob to textspot's own driver, unchanged."""

    def spot(self, greys, blob):
        if blob_version(blob) == 1:
            return super().spot(greys, blob)
        if not self.has(DRIVER2):
            return None
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".vtsp", delete=False) as f:
            f.write(blob)
            bpath = f.name
        try:
            recs = [struct.pack("<4I", g.shape[1], g.shape[0], 1, 0) + g.tobytes() for g in greys]
            lines = self._batch(DRIVER2, ["--batch", bpath], recs)
        finally:
            os.unlink(bpath)
        out = []
        for line in lines:
            v = [int(t) for t in line.split()]
            if v[0]:
                raise RuntimeError(f"{DRIVER2}: rc={v[0]}")
            n, nb, pick, tr = v[1:5]
            bx = [tuple(v[5 + 8 * i:13 + 8 * i]) for i in range(n)]
            o = 5 + 8 * n
            out.append((bx, [tuple(v[o + 7 * i:o + 7 + 7 * i]) for i in range(nb)], pick, tr))
        if len(out) != len(greys):
            raise RuntimeError(f"{DRIVER2}: one result per frame expected")
        return out


@contextlib.contextmanager
def v2_dispatch(active):
    """While active: the harness's `import textspot_twin` resolves to shim_module and its C parity to CDrivers2, so
    a version-2 blob runs through textspot2_twin and its driver. Inactive (only version-1 blobs): nothing changes."""
    if not active:
        yield
        return
    import textspot_twin as real
    if getattr(real, "TEXTSPOT2_SHIM", False):
        yield
        return
    shim, saved = shim_module(real), TS.CDrivers
    sys.modules["textspot_twin"], TS.CDrivers = shim, CDrivers2
    try:
        yield
    finally:
        sys.modules["textspot_twin"], TS.CDrivers = real, saved


def cbuild2(out, compiler="gcc"):
    """eval/textspot.py's cbuild, plus runtime/vocr_textspot2_test (textspot2-runtime's driver for version-2 blobs)
    when its sources exist: the first of DRIVER2_SOURCE_SETS that compiles and links with textspot's flags."""
    out = Path(out)
    check_writable(out)
    rec = TS.cbuild(out, compiler)
    rt = ROOT / "runtime"
    env = {"PATH": os.environ.get("PATH", os.defpath), "LC_ALL": "C", "SOURCE_DATE_EPOCH": "0"}
    rec["driver2"] = None
    for srcs in DRIVER2_SOURCE_SETS:
        if not all((rt / s).exists() for s in srcs):
            continue
        p = subprocess.run([compiler, *TS.C_FLAGS, "-o", str(out / DRIVER2), *srcs], cwd=rt, env=env,
                           capture_output=True, text=True)
        if p.returncode == 0:
            rec["binaries"][DRIVER2] = TS.sha256(out / DRIVER2)
            rec["driver2"] = {"sources": [f"runtime/{s}" for s in srcs]}
            for s in srcs:
                rec["sources_sha256"][f"runtime/{s}"] = TS.sha256(rt / s)
            break
    TS.write_json(out / "build.json", rec, indent=1)
    return rec


# ----------------------------------------------------------------------------- frames, the cache, the readers

def load_frames_rgb(rows, dirs):
    """[(Frame, rgb)]: eval/textspot.py's load_frame, with the frame's RGB kept (the teacher reads colour)."""
    from common import to_gray
    out = []
    for row in rows:
        src = TS.read_rgb(TS.source_path(row, dirs))
        if row.get("src_dims") and [src.shape[1], src.shape[0]] != list(row["src_dims"]):
            raise ValueError(f"{row['id']}: source is {src.shape[1]}x{src.shape[0]}, the row says {row['src_dims']}")
        rgb = TS.scale_frame(row, src)
        out.append((TS.Frame(row, to_gray(rgb), TS.pixel_sha256(rgb)), rgb))
    return out


def frames_doc(frames, manifest):
    return {"kind": "textspot2-frames", "manifest": manifest,
            "frames": {f.id: {"sha": f.sha, "dims": [f.width, f.height], "vn": f.vn, "part": f.part} for f in frames}}


def record_frames(path, frames, manifest):
    """The frames' pixel sha256 (the reading cache's key) in a file, so that select finds the readings of any arm's
    boxes without decoding a screenshot. A later write must agree with what is recorded."""
    if not path:
        return None
    p = Path(path)
    check_writable(p)
    doc = frames_doc(frames, manifest)
    if p.exists():
        old = json.loads(p.read_text(encoding="utf-8"))
        if old["manifest"]["sha256"] == manifest["sha256"]:
            if {k: old["frames"].get(k) for k in doc["frames"]} != doc["frames"]:
                raise SystemExit(f"{p}: these frames' pixels differ from the recorded ones")
            if set(old["frames"]) >= set(doc["frames"]):
                return old
    TS.write_json(p, doc, indent=1)
    return doc


def seed_cache(dst, src=TS_CACHE):
    """The reading cache dst seeded once with a byte-for-byte copy (sha256-checked) of every shard of src
    (textspot's development cache, only read), recorded in dst/SEEDED.json; later calls change nothing."""
    dst = Path(dst)
    check_writable(dst)
    marker = dst / "SEEDED.json"
    if marker.exists():
        return json.loads(marker.read_text(encoding="utf-8"))
    dst.mkdir(parents=True, exist_ok=True)
    files = {}
    for p in sorted(Path(src).glob("*.jsonl")):
        q, h = dst / p.name, TS.sha256(p)
        if not q.exists():
            shutil.copyfile(p, q)
        if TS.sha256(q) != h:
            raise SystemExit(f"{q}: not a copy of {p}")
        files[p.name] = {"sha256": h, "bytes": p.stat().st_size}
    if not files:
        raise SystemExit(f"{src}: no cache shard to copy")
    rec = {"kind": "textspot2-cache-seed", "seeded": TS.utc_now(), "from": TS.rel(src), "files": files}
    TS.write_json(marker, rec, indent=1)
    return rec


def check_recognizers(proto, proto2):
    """Both protocols pin the same four recognizers."""
    for m, r in proto2["reused"]["recognizers"].items():
        if proto["recognizers"][m]["sha256"] != r["sha256"]:
            raise SystemExit(f"{m}: textspot's and textspot2's protocols pin different recognizers")


def readers_for(models, proto, build_dir, compiler="gcc", want_library=None):
    """eval/textspot.py's C readers (the recognizers sha256-checked against the protocol), built into build_dir. The
    library must be the one textspot's units and cache were read with (the frozen baseline's reader)."""
    rd = TS.c_readers(TS.recognizer_paths(models, proto), build_dir, compiler)
    want = want_library or json.loads(BASELINE.read_text(encoding="utf-8"))["reader"]["library_sha256"]
    if rd.identity["library_sha256"] != want:
        raise SystemExit(f"the reader library {rd.identity['library_sha256'][:12]} is not textspot's {want[:12]}: the "
                         "copied cache's readings would come from another build")
    return rd


def resolve_cbin(text, out, compiler="gcc"):
    if not text:
        return None
    if text == "auto":
        cbuild2(Path(out) / "cbuild", compiler)
        return Path(out) / "cbuild"
    if not (Path(text) / "build.json").exists():
        raise SystemExit(f"{text}: no build.json (textspot2.py cbuild --out {text})")
    return Path(text)


def check_dev_manifest(which, manifest, proto2):
    if which == "dev" and manifest["sha256"] != proto2["reused"]["dev_manifest"]["sha256"]:
        raise SystemExit("the development manifest is not the one textspot2's protocol reuses")


# ----------------------------------------------------------------------------- the runs

def finished_list(runs_dir):
    """{run: (sha256, bytes)} of runs_dir/finished.txt (the sweep's lines: name, sha256 of final.vtsp, bytes, ...),
    or None when the file does not exist."""
    f = Path(runs_dir) / "finished.txt"
    if not f.exists():
        return None
    out = {}
    for line in f.read_text(encoding="utf-8").splitlines():
        v = line.split()
        if len(v) >= 3:
            out[v[0]] = (v[1], int(v[2]))
    return out


def finished_runs(runs_dir, names=None):
    """({run: info}, [skipped]) of the finished runs: run/done.json and run/final.vtsp, the blob's sha256 and bytes
    as done.json records them and as finished.txt lists them (when it exists, a run it does not list is not finished
    yet), a name <recipe>_h2_<shape>_seed<n> (others are skipped), and a blob of that shape at half_above 600 with
    conf_min 1."""
    runs_dir = Path(runs_dir)
    listed = finished_list(runs_dir)
    found = sorted(p.parent.name for p in runs_dir.glob("*/done.json"))
    if names:
        missing = sorted(set(names) - set(found))
        if missing:
            raise SystemExit(f"not finished (no done.json): {missing}")
        found = [n for n in found if n in set(names)]
    out, skipped = {}, []
    for name in found:
        if listed is not None and name not in listed:
            if names:
                raise SystemExit(f"{name}: not in finished.txt yet")
            continue
        try:
            info = parse_run(name)
        except ValueError:
            if names:
                raise
            skipped.append(name)
            continue
        d = runs_dir / name
        done = json.loads((d / "done.json").read_text(encoding="utf-8"))
        path = d / "final.vtsp"
        blob = path.read_bytes()
        sha = TS.sha256_bytes(blob)
        if sha != done.get("vtsp_sha256") or len(blob) != done.get("vtsp_bytes"):
            raise SystemExit(f"{name}: final.vtsp is not the blob done.json records")
        if listed is not None and listed[name] != (sha, len(blob)):
            raise SystemExit(f"{name}: final.vtsp is not the blob finished.txt lists")
        m = model_of(blob)
        if (m.name, m.half_above, m.conf_min) != (info["shape"], 600, 1):
            raise SystemExit(f"{name}: the blob is {m.name} at half_above {m.half_above}, conf_min {m.conf_min}")
        out[name] = dict(info, path=path, blob=blob, sha256=sha, bytes=len(blob), version=blob_version(blob),
                         half_above=m.half_above, steps=done.get("steps"))
    return out, skipped


def run_specs(runs):
    """eval/textspot.py's det and detref candidate specs of each run: the arms T2_<run>_det and T2_<run>_detref."""
    out = []
    for name, r in sorted(runs.items()):
        out += TS.candidate_specs(f"T2_{name}", r["blob"], r["path"], "T2", name, r["half_above"], KINDS,
                                  text=f"T2:{name}")
    return out


def is_scored(out, spec, manifest, frame_ids, models, need_c):
    """The arm's boxes are reusable (eval/textspot.py reusable: the same spec, manifest, twins and C check) for every
    frame, and every model's units score those boxes."""
    d = Path(out) / spec.name
    doc = TS.reusable(d / "boxes.json", spec.meta(), manifest, need_c)
    if doc is None or not set(doc["frames"]) >= set(frame_ids):
        return False
    bsha = TS.sha256(d / "boxes.json")
    for m in models or [None]:
        f = d / f"units_{m or 'detection'}.json"
        if not f.exists():
            return False
        meta = json.loads(f.read_text(encoding="utf-8"))["meta"]
        if meta.get("boxes_sha256") != bsha or meta.get("spec_meta") != spec.meta():
            return False
    return True


def pending(out, runs, manifest, frame_ids, models, need_c, force=False):
    """The specs of the runs' arms that are not scored yet."""
    specs = run_specs(runs)
    with v2_dispatch(any(r["version"] != 1 for r in runs.values())):
        return [s for s in specs if force or not is_scored(out, s, manifest, frame_ids, models, need_c)]


def score_runs(frames, manifest, runs, specs, out, models, readers=None, cache=None, cdir=None, workers=5,
               force=False, log=print, proto=None):
    """The specs through eval/textspot.py's evaluate (boxes, C parity, reading, units and summaries), then a sidecar
    per arm, OUT/<arm>/textspot2.json: its run, recipe, shape, seed, arm, blob version and this wrapper's hashes."""
    with v2_dispatch(any(blob_version(s.blob) != 1 for s in specs)):
        TS.evaluate(frames, specs, manifest, out, models, readers, cache, cdir, workers, force, log, proto)
    for s in specs:
        r = runs[s.run]
        side = {"kind": "textspot2-arm", "arm": s.name, "run": s.run, "recipe": r["recipe"], "shape": r["shape"],
                "seed": r["seed"], "arm_id": r["arm_id"], "decode": s.decode, "blob_version": r["version"],
                "blob_sha256": r["sha256"], "code_sha256": code_identity(), "computed": TS.utc_now()}
        if r["version"] != 1 and cdir is not None:
            drv = Path(cdir) / DRIVER2
            side["driver2_sha256"] = TS.sha256(drv) if drv.exists() else None
        TS.write_json(Path(out) / s.name / "textspot2.json", side, indent=1)
    return [s.name for s in specs]


def log_line(s):
    print(s, flush=True)


def cmd_run(args):
    proto, proto2 = TS.protocol(), protocol2()
    check_recognizers(proto, proto2)
    out = Path(args.out)
    cache_dir = Path(args.cache) if args.cache else (DEFAULT["cache"] if args.set == "dev" else out / "cache")
    check_writable(out, cache_dir, args.rbuild, args.frames)
    rows, manifest = TS.load_set(args, proto)
    check_dev_manifest(args.set, manifest, proto2)
    if args.set == "dev" and not args.cbin:
        raise SystemExit("the development set needs --cbin (the rule's C parity per frame)")
    runs, skipped = finished_runs(args.runs, args.names)
    for s in skipped:
        print(f"skipped {s}: not a run name of the protocol", flush=True)
    models = TS.parse_models(args.models)
    cdir = resolve_cbin(args.cbin, out, args.compiler)
    todo = pending(out, runs, manifest, [r["id"] for r in rows], models, cdir is not None, args.force)
    print(f"{len(rows)} frames, {len(runs)} finished runs, {len(todo)} arms to score, models {models or 'none'}",
          flush=True)
    if not todo:
        return []
    frames = TS.load_frames(rows, [Path(d) for d in args.shots] + list(TS.SHOT_DIRS))
    record_frames(args.frames or (DEFAULT["frames"] if args.set == "dev" else None), frames, manifest)
    readers = cache = build = None
    if models:
        if args.set == "dev":
            seed_cache(cache_dir)
        build = Path(args.rbuild) / str(os.getpid())
        readers = readers_for(models, proto, build, args.compiler)
        cache = TS.Cache(cache_dir)
    try:
        return score_runs(frames, manifest, runs, todo, out, models, readers, cache, cdir, args.workers, args.force,
                          log_line, proto)
    finally:
        if cache is not None:
            cache.close()
        if build is not None:
            shutil.rmtree(build, ignore_errors=True)


# ----------------------------------------------------------------------------- the teacher ceiling (amendment 1)

CEILING_SIZES = ("960x544", "320x192")
SCORE_MIN = 0.5                    # the ceiling's boxes: score >= 0.5
RAW_MIN = 0.2                      # the raw detections kept on disk (the labels' threshold; the record's only)
INFER_320 = ("inference.v0.1.320x192.py", "d8db59a1a8b55e657655803b41d392e43a0a43a9065701299cf62e9c0f3842cd")


def teacher_file(size, proto2):
    t = proto2["teacher"]
    name = f"meiki.text.detect.v0.1.{size}.onnx"
    return name, t["files"][name], t["repo"], t["revision"]


def teacher_model(size, models_dir, proto2):
    """The pinned detector file of one size, sha256-checked: the 960x544 file is teacher/engines.py's (its PINS and
    model_path), the 320x192 file the copy fetch-teacher put in models_dir."""
    name, want, repo, rev = teacher_file(size, proto2)
    if size == "960x544":
        from teacher import engines as TE
        src, fname, pin = TE.PINS["meiki_det"]
        if (src[1], src[2], fname, pin) != (repo, rev, name, want):
            raise SystemExit("teacher/engines.py does not pin the protocol's 960x544 detector")
        path = TE.model_path("meiki_det")
    else:
        path = Path(models_dir) / name
    if not Path(path).exists():
        raise SystemExit(f"{path}: missing (textspot2.py fetch-teacher)")
    got = TS.sha256(path)
    if got != want:
        raise SystemExit(f"{path}: sha256 {got}, the protocol pins {want}")
    return Path(path)


def cmd_fetch_teacher(args):
    """The 320x192 detector and the repository's inference script for it, at the pinned revision (no frame is
    sent anywhere), sha256-checked."""
    from huggingface_hub import hf_hub_download
    proto2 = protocol2()
    d = Path(args.models)
    check_writable(d)
    d.mkdir(parents=True, exist_ok=True)
    name, want, repo, rev = teacher_file("320x192", proto2)
    for fname, sha in ((name, want), INFER_320):
        dest = d / fname
        if not (dest.exists() and TS.sha256(dest) == sha):
            shutil.copyfile(hf_hub_download(repo, fname, revision=rev), dest)
        got = TS.sha256(dest)
        if got != sha:
            raise SystemExit(f"{dest}: sha256 {got}, pinned {sha}")
        print(f"{dest}: sha256 {got} (as pinned)", flush=True)


def map_320(boxes, w, h):
    """inference.v0.1.320x192.py's way back to frame pixels: int(x / (320 / W)), int(y / (192 / H)); then clipped to
    the frame. boxes: the model's [[x0, y0, x1, y1]] at 320x192."""
    rw, rh = 320 / w, 192 / h
    out = [[int(x0 / rw), int(y0 / rh), int(x1 / rw), int(y1 / rh)] for x0, y0, x1, y1 in boxes]
    return np.clip(np.array(out, dtype=np.int64).reshape(-1, 4), 0, np.array([w, h, w, h]))


def run_320(sess, bgr):
    """meiki.text.detect v0.1 320x192 as the repository's inference.v0.1.320x192.py runs it: cv2.resize to 320x192
    (INTER_LINEAR, no aspect kept), / 255, CHW, sizes [[320, 192]] -> (boxes in frame pixels, scores)."""
    import cv2
    h, w = bgr.shape[:2]
    img = cv2.resize(bgr, (320, 192), interpolation=cv2.INTER_LINEAR)
    x = np.expand_dims(np.transpose(img.astype(np.float32) / 255.0, (2, 0, 1)), axis=0)
    names = [i.name for i in sess.get_inputs()]
    _, boxes, scores = sess.run(None, {names[0]: x, names[1]: np.array([[320, 192]], dtype=np.int64)})
    return map_320(boxes[0], w, h), np.asarray(scores[0])


class MeikiDet:
    """The teacher's detector on a BGR frame -> (boxes [[x0, y0, x1, y1]] int in frame pixels, scores): 960x544
    through teacher/engines.Meiki's detector (MeikiOCR's preprocessing, and Meiki.ocr's clip), 320x192 by run_320."""

    def __init__(self, size, path, threads=5):
        from teacher import engines as TE
        self.size = size
        if size == "960x544":
            self.meiki = TE.Meiki(det_thr=SCORE_MIN, n_threads=threads)
        else:
            self.sess = TE.session(path, threads)

    def __call__(self, bgr):
        h, w = bgr.shape[:2]
        if self.size != "960x544":
            return run_320(self.sess, bgr)
        m = self.meiki.m
        tensor, scale = m._preprocess_for_detection(bgr)
        _, boxes, scores = m._run_detection_inference(tensor, scale)
        return np.clip(boxes[0], 0, np.array([w, h, w, h])).astype(np.int32), np.asarray(scores[0])


def ceiling_boxes(raw, scores, width, height, score_min=SCORE_MIN):
    """One frame's ceiling boxes: every detection of score >= score_min with a positive size, vertical when taller
    than wide (MeikiOCR: h > w), padded by the spotter's pad rule and clipped (textspot_metrics.pad_box); empty
    boxes dropped. -> [(x, y, w, h, vertical, 0)] (conf 0: no confidence of its own, as the baselines)."""
    out = []
    for (x0, y0, x1, y1), sc in zip(raw, scores):
        bw, bh = int(x1) - int(x0), int(y1) - int(y0)
        if not float(sc) >= score_min or bw <= 0 or bh <= 0:
            continue
        v = int(bh > bw)
        x, y, w, h = M.pad_box([int(x0), int(y0), bw, bh], bool(v), width, height)
        if w > 0 and h > 0:
            out.append((x, y, w, h, v, 0))
    return out


def ceiling_arm(frame, boxes, kind, crefine=None):
    """One frame's arm result from its ceiling boxes, as eval/textspot.py makes a baseline's: dark flags by
    common.line_dark; for detref the horizontal boxes through refine_spot_boxes (vertical ones unchanged, the
    refinement also run in C when crefine is given); then textspot_twin.group_and_order."""
    import textspot_twin as S
    from common import line_dark
    g = frame.grey
    bx = [(*b[:6], int(line_dark(g, b[:4]))) for b in boxes]
    h_in = refine_c = None
    if kind == "detref":
        hz = [b for b in bx if not b[4]]
        bx, h_in = TS.refine_spot_boxes(bx, g)
        if crefine is not None:
            cb, cdk = crefine.refine(g, [b[:4] for b in hz], [b[6] for b in hz])
            refine_c = [(*b, d) for b, d in zip(cb, cdk)] == [(*b[:4], b[6]) for b in bx if not b[4]]
    ordered, blocks, pick = S.group_and_order(bx, frame.height)
    return {"boxes": [TS.ints(b) for b in ordered], "blocks": [TS.ints(b) for b in blocks], "pick": int(pick),
            "truncated": 0, "h_in": h_in, "c_equal": None, "refine_c_equal": refine_c}


class CeilingSpec(TS.Spec):
    """A teacher ceiling arm: the harness's Spec with its own meta. Its kind 'ceiling' is never a candidate
    (eval/textspot.py's eligible and select read det and detref arms only)."""

    def __init__(self, size, kind, model_sha256, teacher):
        name = f"meiki{size.split('x')[0]}_{kind}"
        super().__init__(name, "ceiling", text=name)
        self.size, self.ckind, self.model_sha256, self.teacher = size, kind, model_sha256, teacher

    @property
    def decode(self):
        return DECODE[self.ckind]

    def meta(self):
        return {"arm": self.name, "spec": self.text, "kind": "ceiling", "baseline": None, "decode": self.decode,
                "family": "ceiling", "run": None, "teacher": self.teacher, "size": self.size,
                "model_sha256": self.model_sha256, "score_min": SCORE_MIN, "vertical": "h > w (MeikiOCR)",
                "pad": "textspot_metrics.pad_box",
                "refine": "eval/textspot.py refine_spot_boxes" if self.ckind == "detref" else None}


def detect_all(size, path, pairs, out, threads=5, log=print):
    """The detector's raw boxes and scores (score >= RAW_MIN, exact floats) on every frame, kept in
    OUT/meiki_raw_<size>.json and reused for the same model file and frame pixels."""
    f = Path(out) / f"meiki_raw_{size}.json"
    sha = TS.sha256(path)
    if f.exists():
        doc = json.loads(f.read_text(encoding="utf-8"))
        if doc.get("model_sha256") == sha and all(doc["frames"].get(fr.id, {}).get("sha") == fr.sha for fr, _ in pairs):
            return doc
    det = MeikiDet(size, path, threads)
    frames = {}
    for i, (fr, rgb) in enumerate(pairs):
        boxes, scores = det(np.ascontiguousarray(rgb[:, :, ::-1]))
        keep = [k for k, s in enumerate(scores) if float(s) >= RAW_MIN]
        frames[fr.id] = {"sha": fr.sha, "boxes": [[int(v) for v in boxes[k]] for k in keep],
                         "scores": [float(scores[k]) for k in keep]}
        if (i + 1) % 50 == 0:
            log(f"meiki {size}: {i + 1}/{len(pairs)} frames")
    doc = {"kind": "textspot2-teacher-raw", "size": size, "model": TS.rel(path), "model_sha256": sha,
           "raw_min": RAW_MIN, "computed": TS.utc_now(), "frames": frames}
    TS.write_json(f, doc)
    return doc


def run_ceiling(pairs, manifest, out, sizes, models_dir, proto2, models, readers=None, cache=None, cdir=None,
                threads=5, workers=5, log=print, detections=None):
    """The ceiling's arms on the frames: detection (or the given {size: raw doc}), boxes.json per arm in the
    harness's format, the reading (eval/textspot.py read_all) and the units (score_arms). Returns the public
    record (also OUT/ceiling.json)."""
    out = Path(out)
    check_writable(out)
    frames = [f for f, _ in pairs]
    crefine = TS.CDrivers(cdir).crefine if cdir is not None else None
    t = proto2["teacher"]
    teacher = f"{t['repo']}@{t['revision']}"
    specs, per_frame, models_rec, rc = [], [{} for _ in frames], {}, {}
    for size in sizes:
        if detections is not None:
            raw = detections[size]
        else:
            raw = detect_all(size, teacher_model(size, models_dir, proto2), pairs, out, threads, log)
        models_rec[size] = {"model": raw.get("model"), "model_sha256": raw["model_sha256"]}
        for kind in KINDS:
            s = CeilingSpec(size, kind, raw["model_sha256"], teacher)
            fr = {}
            for i, f in enumerate(frames):
                r = raw["frames"][f.id]
                a = ceiling_arm(f, ceiling_boxes(r["boxes"], r["scores"], f.width, f.height), kind, crefine)
                per_frame[i][s.name] = a
                fr[f.id] = dict(a, part=f.part)
            vals = [a["refine_c_equal"] for a in fr.values()]
            rc[s.name] = {"equal": sum(1 for v in vals if v is True), "differ": sum(1 for v in vals if v is False),
                          "unchecked": sum(1 for v in vals if v is None)}
            doc = {"kind": "textspot-boxes", "arm": s.name, "spec_meta": s.meta(), "manifest": manifest,
                   "twin_sha256": TS.twin_identity(), "c_checked": False, "c_build": None, "computed": TS.utc_now(),
                   "frames": fr}
            TS.write_json(out / s.name / "boxes.json", doc)
            specs.append(s)
    if models:
        n = TS.read_all(frames, per_frame, models, readers, cache, workers, log)
        log(f"ceiling: read {n} new crops")
    summ = TS.score_arms(frames, specs, per_frame, manifest, out, models, readers, cache, log)
    rec = {"kind": "textspot2-ceiling", "computed": TS.utc_now(), "manifest": manifest, "teacher": teacher,
           "models": models_rec, "score_min": SCORE_MIN, "refine_c_equal": rc, "code_sha256": code_identity(),
           "note": "report only, host only, never a candidate (amendment 1)",
           "arms": {s.name: summ[s.name] for s in specs}}
    TS.assert_public(rec)
    TS.write_json(out / "ceiling.json", rec, indent=1)
    return rec


def cmd_ceiling(args):
    proto, proto2 = TS.protocol(), protocol2()
    check_recognizers(proto, proto2)
    out = Path(args.out)
    cache_dir = Path(args.cache) if args.cache else (DEFAULT["cache"] if args.set == "dev" else out / "cache")
    check_writable(out, cache_dir, args.rbuild, args.frames)
    sizes = [s for s in args.sizes.split(",") if s]
    if not sizes or set(sizes) - set(CEILING_SIZES):
        raise SystemExit(f"--sizes: of {CEILING_SIZES}")
    rows, manifest = TS.load_set(args, proto)
    check_dev_manifest(args.set, manifest, proto2)
    pairs = load_frames_rgb(rows, [Path(d) for d in args.shots] + list(TS.SHOT_DIRS))
    record_frames(args.frames or (DEFAULT["frames"] if args.set == "dev" else None), [f for f, _ in pairs], manifest)
    models = TS.parse_models(args.models)
    cdir = resolve_cbin(args.cbin, out, args.compiler)
    readers = cache = build = None
    if models:
        if args.set == "dev":
            seed_cache(cache_dir)
        build = Path(args.rbuild) / str(os.getpid())
        readers = readers_for(models, proto, build, args.compiler)
        cache = TS.Cache(cache_dir)
    try:
        rec = run_ceiling(pairs, manifest, out, sizes, args.models_dir, proto2, models, readers, cache, cdir,
                          args.threads, args.workers, log_line)
    finally:
        if cache is not None:
            cache.close()
        if build is not None:
            shutil.rmtree(build, ignore_errors=True)
    for name, by_m in rec["arms"].items():
        for m, s in by_m.items():
            d = s["detection"]
            print(f"{name} {m}: F1 {TS.fmt(d['f1'])} CER all {s.get('cer_all', {}).get('cer')} dialogue "
                  f"{s.get('cer_dialogue', {}).get('cer')} text-free FA {s['text_free']['share_with_any_box']}",
                  flush=True)


# ----------------------------------------------------------------------------- the secondary column

def rec_shas(units_doc):
    """(the horizontal recognizer's sha256, VOCR5-V's) of a units file."""
    rs = units_doc["meta"]["recognizers_sha256"]
    return rs[units_doc["meta"]["model"]], rs[TS.V_MODEL]


def dnc_of(boxes_doc, rows, shas, cache, rec_sha, units=None):
    """The secondary column's units of an arm (textspot2_rule.dnc_units: dialogue, name and choice references, no
    junk). eval/textspot.py's units hold no per-reference errors, so every frame's are recomputed from the arm's
    boxes and the cached readings with textspot_metrics.e2e_frame, as frame_unit scores them; with units given,
    each frame must give those units' all-text and dialogue errors exactly. rec_sha: rec_shas(units file)."""
    hsha, vsha = rec_sha
    ub = {u["unit"]: u for u in units} if units is not None else None
    frames = []
    for row in rows:
        fid = row["id"]
        dets = []
        for b in boxes_doc["frames"][fid]["boxes"]:
            t = cache.get(TS.Cache.key(shas[fid], b[:4], b[6], b[4], vsha if b[4] else hsha))
            if t is None:
                raise SystemExit(f"{boxes_doc['arm']} {fid}: a box without a cached reading")
            dets.append({"box": list(b[:4]), "vertical": bool(b[4]), "text": t})
        w, h = row["dims"]
        e = M.e2e_frame(dets, row.get("refs", []), w, h)
        if ub is not None:
            u = ub[fid]
            if [list(u["e2e"]["all"]), list(u["e2e"]["dialogue"])] != [list(e["all"]), list(e["dialogue"])]:
                raise SystemExit(f"{boxes_doc['arm']} {fid}: the recomputed errors are not its units'")
        frames.append({"unit": fid, "vn": row["vn"], "refs": row.get("refs", []), "per_ref": e["per_ref"]})
    return R.dnc_units(frames)


# ----------------------------------------------------------------------------- the arms select reads

def load_arm(d, model=TS.RULE_MODEL):
    """(boxes doc, units doc or None) of one arm directory (each with its file's sha256 as "_sha256"); the units must
    score those boxes."""
    d = Path(d)
    b = d / "boxes.json"
    doc = json.loads(b.read_text(encoding="utf-8"))
    doc["_sha256"] = TS.sha256(b)
    u = d / f"units_{model}.json"
    units = None
    if u.exists():
        units = json.loads(u.read_text(encoding="utf-8"))
        units["_sha256"] = TS.sha256(u)
        if units["meta"]["boxes_sha256"] != doc["_sha256"]:
            raise SystemExit(f"{u}: the units of other boxes than {b}")
    return doc, units


def seed_record(name, run, source, doc, units, d):
    s = units["summary"]
    return dict(parse_run(run), name=name, decode=doc["spec_meta"]["decode"], source=source, dir=Path(d), boxes=doc,
                units=units, f1=s["detection"]["f1"], cer_all=s["cer_all"]["cer"],
                cer_dialogue=s["cer_dialogue"]["cer"])


def t2_seeds(units_dirs, runs_dir=None):
    """Every T2_<run>_det / _detref arm of the unit directories (VOCR5-N units required); with runs_dir, each arm's
    blob must still be its run's final.vtsp (done.json)."""
    out = []
    for d in units_dirs:
        for b in sorted(Path(d).glob("T2_*/boxes.json")):
            doc, units = load_arm(b.parent)
            meta = doc["spec_meta"]
            if meta.get("kind") not in KINDS or meta.get("family") != "T2":
                continue
            if units is None:
                raise SystemExit(f"{b.parent}: no {TS.RULE_MODEL} units (textspot2.py run first)")
            done = Path(runs_dir) / meta["run"] / "done.json" if runs_dir else None
            if done is not None and done.exists():
                if json.loads(done.read_text(encoding="utf-8")).get("vtsp_sha256") != meta["blob_sha256"]:
                    raise SystemExit(f"{b.parent}: scored another blob than {meta['run']}'s final.vtsp")
            out.append(seed_record(doc["arm"], meta["run"], "textspot2", doc, units, b.parent))
    return out


def textspot_r0_seeds(t_dir=T, ts_dev=TS_DEV):
    """r0's seeds 1 and 2: textspot's runs, their VOCR5-N units under $T/eval/famB and famC, never re-run; each must
    be the units textspot's selection record (eval/textspot_dev_20261019.json) scored: its summary, exactly."""
    dev = json.loads(Path(ts_dev).read_text(encoding="utf-8"))
    out = []
    for arm_id, seeds in R0_TEXTSPOT.items():
        for seed, (fam, prefix) in sorted(seeds.items()):
            for kind in KINDS:
                name = f"{prefix}_{kind}"
                d = Path(t_dir) / "eval" / fam / name
                doc, units = load_arm(d)
                if units is None or dev["candidates"].get(name, {}).get("summary") != units["summary"]:
                    raise SystemExit(f"{d}: not the units textspot's selection record scored")
                out.append(seed_record(name, f"{arm_id}_seed{seed}", "textspot", doc, units, d))
    return out


def reference(proto2, key):
    """The protocol's ref or fast_ref: its units and boxes (textspot's), each checked against the protocol's sha256."""
    r = proto2["references"][key]
    up, bp = TS.resolve(r["units"]), TS.resolve(r["boxes"])
    for p, want in ((up, r["units_sha256"]), (bp, r["boxes_sha256"])):
        if TS.sha256(p) != want:
            raise SystemExit(f"{p}: not the {key} units / boxes the protocol pins")
    if Path(up).parent != Path(bp).parent:
        raise SystemExit(f"{key}: units and boxes in different directories")
    doc, units = load_arm(Path(up).parent)
    return {"name": r["name"], "role": key, "boxes": doc, "units": units, "arm": TS.rule_arm(units["units"])}


def baselines_of(base_doc, units_dir=None):
    """The frozen baseline's eight arms read by VOCR5-N: rule arms (eval/textspot.py baseline_rule_arms checks each
    units file's sha256), their boxes (checked against the record's boxes_sha256) and units."""
    arms = TS.baseline_rule_arms(base_doc, units_dir)
    out = {}
    for b in TS.BASELINES:
        e = base_doc["arms"][b][TS.RULE_MODEL]
        f = Path(units_dir) / b / f"units_{TS.RULE_MODEL}.json" if units_dir else TS.resolve(e["units"])
        doc, units = load_arm(Path(f).parent)
        if doc["_sha256"] != e["boxes_sha256"]:
            raise SystemExit(f"{b}: not the frozen baseline's boxes")
        out[b] = {"name": b, "boxes": doc, "units": units, "arm": arms[b]}
    return out


def ceiling_of(d):
    """The ceiling arms found under d (VOCR5-N units), {name: arm}."""
    out = {}
    for name in (f"meiki{s.split('x')[0]}_{k}" for s in CEILING_SIZES for k in KINDS):
        p = Path(d) / name
        if (p / "boxes.json").exists():
            doc, units = load_arm(p)
            if units is not None:
                out[name] = {"name": name, "boxes": doc, "units": units, "arm": TS.rule_arm(units["units"])}
    return out


# ----------------------------------------------------------------------------- eligibility and the seeds

def a9_rows_all(t2_doc=None, d2s_doc=None, ts_doc=None):
    """The A9 rows eligibility reads, first match wins: textspot's table (its rows for textspot's shapes), then
    textspot2's (eval/textspot2_a9_<date>.json), then the d2s table's."""
    ts_doc = ts_doc if ts_doc is not None else json.loads(TS_A9.read_text(encoding="utf-8"))
    rows = TS.a9_rows(ts_doc)
    for d in (t2_doc, d2s_doc):
        if d:
            rows += TS.a9_rows(d)
    return rows


def refine_arena2(t2_doc, cdir, proto2):
    """The refinement's arena at 960x544: the table's or the refinement driver's (eval/textspot.py refine_arena),
    which must be the protocol's 72,391 B; the protocol's figure when neither has it."""
    want = int(proto2["reused"]["a9"]["refine_arena_960x544"])
    got, src = TS.refine_arena(t2_doc or {}, cdir)
    if got is None:
        return want, "eval/textspot2_protocol.json reused.a9.refine_arena_960x544"
    if got != want:
        raise SystemExit(f"the refinement's arena {got} ({src}) is not the protocol's {want}")
    return got, src


def d2s_macs(shape, half_above):
    """MACs per 960x544 frame of a d2s shape at its network input (train/textspot2_export.costs: textdet_model's
    cost cannot read a kroa_x2 head)."""
    import textspot2_export as X
    return int(X.costs(shape, sizes=((960, 544),), half_above=half_above)["macs"]["960x544"])


def eligibility2(meta, boxes_doc, rows, ref_arena):
    """eval/textspot.py eligibility, with train/textspot2_shapes.register() called first (the protocol's new shapes
    resolve in train/textdet_model.SHAPES). For a d2s shape (head kroa_x2) the MACs it computes and checks against
    the A9 row are train/textspot2_export.costs' (d2s_macs), everything else unchanged."""
    S2.register()
    if meta.get("shape") not in S2.D2S:
        return TS.eligibility(meta, boxes_doc, rows, ref_arena)
    macs = d2s_macs(meta["shape"], meta["half_above"])
    saved = TS.macs_960
    TS.macs_960 = lambda shape, half_above: macs
    try:
        return dict(TS.eligibility(meta, boxes_doc, rows, ref_arena), macs_source="train/textspot2_export.costs")
    finally:
        TS.macs_960 = saved


def report_costs(e, rows):
    """The report's cost columns of an eligibility record: est. ms at 444 MHz (x 333/444) and the spotter's est. ms on
    the PSP's 480x272 frame at 333 MHz (the A9 row's, without the refinement)."""
    ms = e.get("est_ms_333_960")
    r = TS.a9_lookup(rows, e["shape"], e["half_above"])
    psp = (r or {}).get("row", {}).get("ms_333_480")
    return {"est_ms_444_960": round(ms * MHZ, 3) if ms is not None else None,
            "ms_333_480_spot": round(psp, 3) if psp is not None else None}


def representatives(seeds):
    """{(arm_id, decode): {"order", "index", "rep", "seeds"}}: each arm's and decode's representative by
    textspot2_rule.median_seed over its finished seeds."""
    groups = {}
    for s in seeds:
        groups.setdefault((s["arm_id"], s["decode"]), []).append(s)
    out = {}
    for key, ss in sorted(groups.items()):
        if len({x["seed"] for x in ss}) != len(ss):
            raise SystemExit(f"{key}: a seed appears twice")
        rep, info = R.median_seed(ss)
        out[key] = dict(info, rep=rep, seeds=sorted(ss, key=lambda x: x["seed"]))
    return out


def candidate_record(seed, e):
    """textspot2_rule's candidate record of a representative."""
    return {"name": seed["name"], "arm_id": seed["arm_id"], "seed": seed["seed"], "decode": seed["decode"],
            "blob_bytes": e["blob_bytes"], "arena_960": e["arena_960"], "est_ms_333_960": e["est_ms_333_960"],
            "c_equal": e["c_equal"], "arm": TS.rule_arm(seed["units"]["units"])}


# ----------------------------------------------------------------------------- the report's pieces

PARETO = (("est_ms_333_960", 1), ("blob_bytes", 1), ("arena_960", 1), ("f1", -1), ("cer_all", 1), ("cer_dialogue", 1))
STAGE1_ARMS = tuple(sorted({parse_run(n)["arm_id"] for n in STAGE1}))


def light(e):
    """The fields textspot2_rule.eligible reads, from an eligibility record."""
    return {"blob_bytes": e["blob_bytes"], "arena_960": e["arena_960"], "est_ms_333_960": e["est_ms_333_960"],
            "c_equal": e["c_equal"]}


def seed_row(s, rule):
    """One seed's row of the record: counts, rates, hashes and its eligibility inputs."""
    su, e = s["units"]["summary"], s["elig"]
    d = su["detection"]
    return {"name": s["name"], "run": s["run"], "arm_id": s["arm_id"], "recipe": s["recipe"], "shape": s["shape"],
            "seed": s["seed"], "decode": s["decode"], "source": s["source"], "units_sha256": s["units"]["_sha256"],
            "boxes_sha256": s["boxes"]["_sha256"], "blob_sha256": e["blob_sha256"], "blob_bytes": e["blob_bytes"],
            "macs_960": e["macs_960"], "arena_960": e["arena_960"], "est_ms_333_960": e["est_ms_333_960"],
            "est_ms_444_960": e.get("est_ms_444_960"), "ms_333_480_spot": e.get("ms_333_480_spot"),
            "median_h_boxes_vita": e.get("median_h_boxes_vita"), "c_equal": e["c_equal"],
            "c_unequal_frames": e["c_unequal_frames"], "c_unchecked_frames": e["c_unchecked_frames"],
            "eligible": R.eligible(light(e), rule),
            "eligible_fast": R.eligible(light(e), rule, rule["fast_ms_max_333_960"]),
            "precision": d["precision"], "recall": d["recall"], "f1": d["f1"], "cer_all": su["cer_all"],
            "cer_dialogue": su["cer_dialogue"], "dnc": M.cer_of(s["dnc"]) if s.get("dnc") is not None else None,
            "recall_by_class": {c: v["recall"] for c, v in su["recall_by_class"].items()},
            "text_free_share_with_any_box": su["text_free"]["share_with_any_box"],
            "pick_hit_share": su["pick"]["hit_share"], "mean_iou": su["tightness"]["mean_iou"]}


def pareto_front(points):
    """points: {name: {PARETO keys}} -> the names no other point dominates (no worse on every key, F1 counted
    higher-better, and better on one), sorted by name."""
    vec = {n: tuple(sign * p[k] for k, sign in PARETO) for n, p in points.items()}
    return sorted(n for n, v in vec.items() if not any(w != v and all(a <= b for a, b in zip(w, v))
                                                       for w in vec.values()))


def point_of(s, role):
    e, su = s["elig"], s["units"]["summary"]
    return {"role": role, "est_ms_333_960": e["est_ms_333_960"], "est_ms_444_960": e.get("est_ms_444_960"),
            "blob_bytes": e["blob_bytes"], "arena_960": e["arena_960"], "f1": su["detection"]["f1"],
            "cer_all": su["cer_all"]["cer"], "cer_dialogue": su["cer_dialogue"]["cer"]}


def best_of_shape(reps, shape, decode, exclude=("kd", "r0")):
    """The representative of the shape's best arm (lowest by textspot2_rule.seed_key) among the arms of the other
    recipes, for one decode."""
    pool = [g["rep"] for (aid, dec), g in reps.items()
            if dec == decode and g["rep"]["shape"] == shape and g["rep"]["recipe"] not in exclude]
    return min(pool, key=R.seed_key) if pool else None


def bought(reps, ref, ceil, result, ref_dnc):
    """The protocol's "what each arm bought": paired development differences of median seeds, per decode (lab - r0,
    data - lab, ema - data, cls - data, long - ema on s4_c24n4; each shape's data - data s4_c24n4; kd - the best
    non-kd arm of its shape; d2s - data s4_c24n4), then every pick and every ceiling arm against ref. Each row:
    textspot_metrics.compare and the secondary column's vn_diff."""
    def row(what, decode, a, b_name, b_arm, b_dnc, a_id=None, b_id=None):
        r = {"what": what, "decode": decode, "minuend": a["name"], "subtrahend": b_name, "minuend_arm": a_id,
             "subtrahend_arm": b_id, "comparison": M.compare(a["arm"], b_arm)}
        r["dnc"] = M.vn_diff(a["dnc"], b_dnc) if a.get("dnc") is not None and b_dnc is not None else None
        return r

    def as_arm(s):
        return {"name": s["name"], "arm": TS.rule_arm(s["units"]["units"]), "dnc": s.get("dnc")}

    ids = {aid for aid, _ in reps}
    pairs = [("lab - r0", "lab_h2_s4_c24n4", "r0_h2_s4_c24n4"), ("data - lab", "data_h2_s4_c24n4", "lab_h2_s4_c24n4"),
             ("ema - data", "ema_h2_s4_c24n4", "data_h2_s4_c24n4"),
             ("cls - data", "cls_h2_s4_c24n4", "data_h2_s4_c24n4"),
             ("long - ema", "long_h2_s4_c24n4", "ema_h2_s4_c24n4")]
    pairs += [(f"data {sh} - data s4_c24n4", f"data_h2_{sh}", "data_h2_s4_c24n4") for sh in S2.VTDT if sh != "s4_c24n4"]
    pairs += [(f"{aid} - best non-kd arm of its shape", aid, "best") for aid in sorted(ids) if aid.startswith("kd_")]
    pairs += [(f"{aid} - data s4_c24n4", aid, "data_h2_s4_c24n4") for aid in sorted(ids)
              if parse_run(f"{aid}_seed1")["shape"] in S2.D2S]
    out = []
    for what, a_id, b_id in pairs:
        for dec in ("det", "det+refine"):
            ga = reps.get((a_id, dec))
            if b_id == "best":
                b = best_of_shape(reps, parse_run(f"{a_id}_seed1")["shape"], dec) if ga else None
                b_id_x = b["arm_id"] if b else None
            else:
                gb = reps.get((b_id, dec))
                b, b_id_x = (gb["rep"] if gb else None), b_id
            if ga is None or b is None:
                out.append({"what": what, "decode": dec, "minuend_arm": a_id, "subtrahend_arm": b_id_x,
                            "missing": True})
                continue
            a, bb = as_arm(ga["rep"]), as_arm(b)
            out.append(row(what, dec, a, bb["name"], bb["arm"], bb["dnc"], a_id, b_id_x))
    by_name = {g["rep"]["name"]: g["rep"] for g in reps.values()}
    for role, w in (("main pick", result["main"]["winner"]), ("fast pick", result["fast"]["winner"])):
        if w:
            out.append(row(f"{role} - ref", by_name[w]["decode"], as_arm(by_name[w]), ref["name"], ref["arm"], ref_dnc))
    for name, c in sorted(ceil.items()):
        out.append(row(f"{name} - ref", c["boxes"]["spec_meta"]["decode"], c, ref["name"], ref["arm"], ref_dnc))
    return out


def stage2_inputs(reps, rule):
    """docs/textspot2.md's fixed stage-2 rules read off the stage-1 representatives (report; the conductor's
    amendment records the choices): the kept extras, the fast and main shapes, the kd teacher, the long arms."""
    def rep(aid, dec="det+refine"):
        g = reps.get((aid, dec))
        return g["rep"] if g else None

    d, e, c = rep("data_h2_s4_c24n4"), rep("ema_h2_s4_c24n4"), rep("cls_h2_s4_c24n4")
    out = {"ema_kept": None if not (d and e) else bool(e["cer_all"] < d["cer_all"]),
           "cls_kept": None if not (d and c) else bool(c["cer_all"] < d["cer_all"] and
                                                       c["cer_dialogue"] <= d["cer_dialogue"] + 0.5)}
    stage1 = [g["rep"] for (aid, _), g in sorted(reps.items()) if aid in STAGE1_ARMS]
    data = [s for s in stage1 if s["recipe"] == "data"]

    def best(pool, ms_max):
        el = [s for s in pool if R.eligible(light(s["elig"]), rule, ms_max)]
        if not el:
            return None
        b = min(el, key=R.seed_key)
        return {"shape": b["shape"], "by": b["name"], "run": b["run"], "decode": b["decode"], "cer_all": b["cer_all"],
                "est_ms_333_960": b["elig"]["est_ms_333_960"]}
    fast = best(data, rule["fast_ms_max_333_960"])
    out["fast_shape"] = fast or {"shape": "s4p_c8", "by": None, "note": "none eligible at <= 80 ms: s4p_c8"}
    out["main_shape"] = best(data, rule["ms_max_333_960"])
    out["kd_teacher"] = best([s for s in stage1 if s["recipe"] != "r0"], rule["ms_max_333_960"])
    out["long_on"] = sorted({x["shape"] for x in (out["main_shape"], out["fast_shape"]) if x}
                            - {"s4_c24n4", "s4p_c8"})
    return out


# ----------------------------------------------------------------------------- select

def gather(units_dirs, proto2, rows, shas, cache, a9_doc=None, a9_d2s=None, ts_a9=None, cdir=None, runs_dir=None,
           t_dir=T, ts_dev=TS_DEV, dnc=True):
    """Every seed (textspot2's arms and r0's textspot seeds) with its eligibility inputs and, when dnc, its secondary
    column; -> (seeds, A9 rows, (refine arena, its source))."""
    man = proto2["reused"]["dev_manifest"]["sha256"]
    rows_a9 = a9_rows_all(a9_doc, a9_d2s, ts_a9)
    ref_ar = refine_arena2(a9_doc, cdir, proto2)
    seeds = t2_seeds(units_dirs, runs_dir) + textspot_r0_seeds(t_dir, ts_dev)
    for s in seeds:
        if {s["boxes"]["manifest"]["sha256"], s["units"]["meta"]["manifest"]["sha256"]} != {man}:
            raise SystemExit(f"{s['name']}: scored on another manifest than the protocol's development set")
        s["elig"] = eligibility2(s["boxes"]["spec_meta"], s["boxes"], rows_a9, ref_ar[0])
        s["elig"].update(report_costs(s["elig"], rows_a9))
        s["dnc"] = dnc_of(s["boxes"], rows, shas, cache, rec_shas(s["units"]), s["units"]["units"]) if dnc else None
    return seeds, rows_a9, ref_ar


def select(units_dirs, a9_doc, proto2, base_doc, rows, shas, cache, a9_d2s=None, ts_a9=None, cdir=None,
           runs_dir=None, ceiling_dir=None, t_dir=T, ts_dev=TS_DEV, baseline_units=None, refs=None, log=print):
    """The selection record: every seed, the representatives, textspot2_rule.apply_rule2 once over the
    representatives of every arm except r0 (against the frozen baseline and the protocol's ref and fast_ref), every
    comparison, the Pareto front, the secondary column, what each arm bought, the ceiling and the stage-2 inputs."""
    rule = proto2["rule"]
    man = proto2["reused"]["dev_manifest"]["sha256"]
    if base_doc["manifest"]["sha256"] != man:
        raise SystemExit("the frozen baseline is of another manifest")
    seeds, rows_a9, (ref_ar, ref_src) = gather(units_dirs, proto2, rows, shas, cache, a9_doc, a9_d2s, ts_a9, cdir,
                                               runs_dir, t_dir, ts_dev)
    bases = baselines_of(base_doc, baseline_units)
    refs = refs if refs is not None else {k: reference(proto2, k) for k in ("ref", "fast_ref")}
    ceil = ceiling_of(ceiling_dir) if ceiling_dir else {}
    for x in list(refs.values()) + list(bases.values()) + list(ceil.values()):
        if {x["boxes"]["manifest"]["sha256"], x["units"]["meta"]["manifest"]["sha256"]} != {man}:
            raise SystemExit(f"{x['name']}: scored on another manifest than the protocol's development set")
        x["dnc"] = dnc_of(x["boxes"], rows, shas, cache, rec_shas(x["units"]), x["units"]["units"])
    reps = representatives(seeds)
    cands = [candidate_record(g["rep"], g["rep"]["elig"]) for g in reps.values() if g["rep"]["recipe"] != "r0"]
    log(f"{len(seeds)} seeds, {len(reps)} arms x decodes, {len(cands)} candidates")
    result = R.apply_rule2(cands, {b: v["arm"] for b, v in bases.items()}, refs["ref"], refs["fast_ref"], rule)
    ref_arm, ref_dnc = refs["ref"]["arm"], refs["ref"]["dnc"]

    def dnc_rec(x):
        return {"dnc": M.cer_of(x["dnc"]), "vs_ref": M.vn_diff(x["dnc"], ref_dnc)}

    extra = {}
    for g in reps.values():
        s = g["rep"]
        if s["name"] in result["comparisons"]:
            continue
        a = TS.rule_arm(s["units"]["units"])
        bars = {b: M.compare(a, v["arm"]) for b, v in sorted(bases.items())}
        vr = M.compare(a, ref_arm)
        extra[s["name"]] = {"baselines": bars, "bar": all(M.beats(x) for x in bars.values()), "vs_ref": vr,
                            "better_than_ref": R.better(vr)}
    arms, rep_out = {}, {}
    for (aid, dec), g in reps.items():
        s = g["rep"]
        arms[f"{aid}|{dec}"] = {"arm_id": aid, "decode": dec, "recipe": s["recipe"], "shape": s["shape"],
                                "role": "control" if s["recipe"] == "r0" else "candidate",
                                "seeds": [x["name"] for x in g["seeds"]], "order": g["order"], "index": g["index"],
                                "representative": s["name"], "finished": len(g["seeds"])}
        rep_out[s["name"]] = {"arm_id": aid, "decode": dec, "seed": s["seed"], "eligibility": s["elig"],
                              "summary": s["units"]["summary"], "secondary": dnc_rec(s)}
    pts = {g["rep"]["name"]: point_of(g["rep"], "candidate") for g in reps.values()
           if g["rep"]["recipe"] != "r0" and R.eligible(light(g["rep"]["elig"]), rule)}
    by_name = {s["name"]: s for s in seeds}
    for k in ("ref", "fast_ref"):
        s = by_name.get(refs[k]["name"])
        if s is not None and R.eligible(light(s["elig"]), rule):
            pts[s["name"]] = point_of(s, k)
    doc = {"kind": "textspot2-selection", "computed": TS.utc_now(), "rule": rule,
           "refine_arena": {"bytes_960x544": ref_ar, "source": ref_src}, "refine_ms_per_box": TS.REFINE_MS_PER_BOX,
           "a9_calibration": TS.A9_CAL, "manifest": base_doc["manifest"], "code_sha256": code_identity(),
           "references": {refs[k]["name"]: {"role": k, "units_sha256": refs[k]["units"]["_sha256"],
                                            "boxes_sha256": refs[k]["boxes"]["_sha256"],
                                            "summary": refs[k]["units"]["summary"],
                                            "secondary": {"dnc": M.cer_of(refs[k]["dnc"])}} for k in refs},
           "baselines": {b: {"units_sha256": v["units"]["_sha256"], "boxes_sha256": v["boxes"]["_sha256"],
                             "f1": v["units"]["summary"]["detection"]["f1"],
                             "cer_all": v["units"]["summary"]["cer_all"],
                             "cer_dialogue": v["units"]["summary"]["cer_dialogue"], "secondary": dnc_rec(v)}
                         for b, v in sorted(bases.items())},
           "seeds": [seed_row(s, rule) for s in sorted(seeds, key=lambda x: x["name"])],
           "arms": arms, "representatives": rep_out, "candidates": [c["name"] for c in cands], "result": result,
           "report_comparisons": extra,
           "pareto": {"keys": [k for k, _ in PARETO], "points": pts, "front": pareto_front(pts)},
           "bought": bought(reps, refs["ref"], ceil, result, ref_dnc),
           "ceiling": {n: {"decode": c["boxes"]["spec_meta"]["decode"], "spec_meta": c["boxes"]["spec_meta"],
                           "units_sha256": c["units"]["_sha256"], "boxes_sha256": c["boxes"]["_sha256"],
                           "summary": c["units"]["summary"], "secondary": dnc_rec(c),
                           "vs_ref": M.compare(c["arm"], ref_arm)} for n, c in sorted(ceil.items())},
           "stage2_inputs": stage2_inputs(reps, rule)}
    TS.assert_public(doc)
    return doc


def write_compact(path, doc, limit=MAX_RECORD):
    """The record as compact JSON (no indentation, no spaces), refused when over limit bytes."""
    text = json.dumps(doc, separators=(",", ":"), ensure_ascii=False) + "\n"
    n = len(text.encode("utf-8"))
    if n > limit:
        raise SystemExit(f"{path}: {n:,} bytes, over the {limit:,} B limit")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(p) + f".tmp{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, p)
    return n


def dev_inputs(proto2, frames_path):
    """(manifest rows, {frame id: pixel sha256}) of the development set, the rows checked against the protocol's
    sha256 and the frames file against the rows."""
    d = proto2["reused"]["dev_manifest"]
    path = ROOT / d["path"]
    if TS.sha256(path) != d["sha256"]:
        raise SystemExit(f"{path}: not the development manifest the protocol reuses")
    rows = TS.jsonl(path)
    fr = json.loads(Path(frames_path).read_text(encoding="utf-8"))
    if fr["manifest"]["sha256"] != d["sha256"] or not set(fr["frames"]) >= {r["id"] for r in rows}:
        raise SystemExit(f"{frames_path}: not every development frame's hash (textspot2.py run or ceiling writes it)")
    return rows, {k: v["sha"] for k, v in fr["frames"].items()}


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8")) if path else None


def cmd_select(args):
    proto2 = protocol2()
    base_doc = load_json(args.baseline)
    rows, shas = dev_inputs(proto2, args.frames)
    cache = TS.Cache(args.cache)
    doc = select(args.units, load_json(args.a9), proto2, base_doc, rows, shas, cache, load_json(args.a9_d2s),
                 load_json(args.a9_textspot), Path(args.cbin) if args.cbin else None, args.runs,
                 args.ceiling or None, log=log_line)
    doc["inputs"] = {"protocol2_sha256": TS.sha256(PROTOCOL2), "protocol_sha256": TS.sha256(TS.PROTOCOL),
                     "baseline_sha256": TS.sha256(args.baseline),
                     "a9": {k: {"path": TS.rel(p), "sha256": TS.sha256(p)} for k, p in
                            (("textspot2", args.a9), ("d2s", args.a9_d2s), ("textspot", args.a9_textspot)) if p},
                     "frames_sha256": TS.sha256(args.frames), "cache_entries": len(cache.data)}
    TS.assert_public(doc)
    n = write_compact(args.out, doc)
    copy = Path(args.indent_copy) if args.indent_copy else EVAL2 / f"{Path(args.out).stem}.indent1.json"
    check_writable(copy)
    TS.write_json(copy, doc, indent=1)
    res = doc["result"]
    print(f"{args.out}: {n:,} B (indented copy {copy}); {len(doc['seeds'])} seeds, {len(doc['candidates'])} "
          f"candidates, {len(res['eligible'])} eligible, {len(res['bar'])} clear the bar; main pick "
          f"{res['main']['winner']}; fast pick {res['fast']['winner']} (tier {res['fast']['tier']})", flush=True)


# ----------------------------------------------------------------------------- headline

def _f(v, nd=2):
    return "–" if v is None else f"{v:.{nd}f}"


def headline_md(seeds, rule, stage1=STAGE1):
    """Markdown: every seed's F1, CER all, dialogue, the secondary column and est. ms; the representatives (median
    seed) per arm and decode; which stage-1 runs are scored; the stage-2 inputs read off them."""
    reps = representatives(seeds)
    md = ["| seed | decode | F1 | CER all | dialogue | D+N+C | est. ms @333 | eligible (<= 80 ms) |",
          "|---|---|---:|---:|---:|---:|---:|---|"]
    for s in sorted(seeds, key=lambda x: (x["arm_id"], x["decode"], x["seed"])):
        r = seed_row(s, rule)
        md.append(f"| `{s['name']}` | {s['decode']} | {_f(r['f1'], 3)} | {_f(r['cer_all']['cer'])} | "
                  f"{_f(r['cer_dialogue']['cer'])} | {_f((r['dnc'] or {}).get('cer'))} | "
                  f"{_f(r['est_ms_333_960'], 1)} | {r['eligible']} ({r['eligible_fast']}) |")
    md += ["", "| arm | decode | seeds, best first | representative | F1 | CER all | dialogue | est. ms @333 | role |",
           "|---|---|---|---|---:|---:|---:|---:|---|"]
    for (aid, dec), g in reps.items():
        s = g["rep"]
        md.append(f"| `{aid}` | {dec} | {', '.join(map(str, g['order']))} | seed {s['seed']} | {_f(s['f1'], 3)} | "
                  f"{_f(s['cer_all'])} | {_f(s['cer_dialogue'])} | {_f(s['elig']['est_ms_333_960'], 1)} | "
                  f"{'control' if s['recipe'] == 'r0' else 'candidate'} |")
    done = {s["run"] for s in seeds}
    missing = [n for n in stage1 if n not in done]
    md += ["", f"Stage 1: {len(stage1) - len(missing)} of {len(stage1)} runs scored"
               + (f"; not yet: {', '.join(missing)}" if missing else "") + ".", "",
           "Stage-2 inputs (the protocol's fixed rules read off these representatives; the conductor's amendment "
           "decides): `" + json.dumps(stage2_inputs(reps, rule), ensure_ascii=False) + "`"]
    return "\n".join(md) + "\n"


def cmd_headline(args):
    proto2 = protocol2()
    rows = shas = cache = None
    dnc = bool(args.frames and Path(args.frames).exists())
    if dnc:
        rows, shas = dev_inputs(proto2, args.frames)
        cache = TS.Cache(args.cache)
    seeds, _, _ = gather(args.units, proto2, rows, shas, cache, load_json(args.a9), load_json(args.a9_d2s),
                         load_json(args.a9_textspot), Path(args.cbin) if args.cbin else None, args.runs, dnc=dnc)
    text = headline_md(seeds, proto2["rule"])
    if args.out:
        check_writable(args.out)
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)


# ----------------------------------------------------------------------------- tables (docs/textspot2.md)

def ci_txt(x, nd):
    if not x or x.get("delta") is None:
        return "–"
    c = x.get("ci95")
    return f"{x['delta']:+.{nd}f}" + (f" ({c[0]:+.{nd}f} to {c[1]:+.{nd}f})" if c else " (no interval)")


def _cer(c):
    return "–" if not c or c.get("cer") is None else f"{c['cer']:.2f}"


def _ms(ms):
    return "–" if ms is None else f"{ms:.1f} / {ms * MHZ:.1f}"


def rule_state(doc, name):
    c = doc["result"]["comparisons"].get(name) or doc["report_comparisons"].get(name) or {}
    return c.get("bar"), c.get("better_than_ref")


def selection_md(doc):
    res = doc["result"]
    md = [f"- Main pick: **{res['main']['winner'] or 'none'}** (comparator `{res['main']['comparator']}`; "
          f"qualifying: {', '.join(f'`{n}`' for n in res['main']['qualifying']) or 'none'})."]
    for st in res["main"]["steps"]:
        md.append(f"  - `{st['by']}` replaces `{st['replaces']}`: CER all "
                  f"{ci_txt(st['comparison']['cer_all'], 2)}, F1 {ci_txt(st['comparison']['f1'], 4)}")
    fq = ", ".join(f"`{n}`" for n in res["fast"]["qualifying"]) or "none"
    md.append(f"- Fast pick (<= {doc['rule']['fast_ms_max_333_960']} ms): **{res['fast']['winner'] or 'none'}** "
              f"(tier {res['fast']['tier']}, comparator `{res['fast']['comparator']}`; pool "
              f"{len(res['fast']['pool'])}, qualifying {fq}).")
    for st in res["fast"]["steps"]:
        md.append(f"  - `{st['by']}` replaces `{st['replaces']}`: CER all {ci_txt(st['comparison']['cer_all'], 2)}")
    md.append(f"- {len(doc['candidates'])} candidates (median seeds, r0 excluded), {len(res['eligible'])} eligible, "
              f"{len(res['bar'])} clear textspot's bar.")
    return "\n".join(md)


def candidates_md(doc):
    rows = []
    for name in doc["candidates"] + [a["representative"] for a in doc["arms"].values() if a["role"] == "control"]:
        r = doc["representatives"][name]
        e, s = r["eligibility"], r["summary"]
        rows.append((e["est_ms_333_960"] is None, e["est_ms_333_960"] or 0, e["blob_bytes"], e["arena_960"] or 0,
                     r["decode"] != "det", name, r, e, s))
    md = ["| arm (median seed) | decode | role | VTSP B | est. ms @333 / @444 | arena B | C = twin | F1 | CER all | "
          "dialogue | D+N+C | eligible | bar | better than ref |", "|---|---|---|---:|---:|---:|---|" + "---:|" * 4
          + "---|---|---|"]
    for *_, name, r, e, s in sorted(rows):
        bar, better = rule_state(doc, name)
        role = doc["arms"][f"{r['arm_id']}|{r['decode']}"]["role"]
        md.append(f"| `{name}` | {r['decode']} | {role} | {e['blob_bytes']:,} | {_ms(e['est_ms_333_960'])} | "
                  f"{'–' if e['arena_960'] is None else format(e['arena_960'], ',')} | {e['c_equal']} | "
                  f"{TS.fmt(s['detection']['f1'], 3)} | {_cer(s['cer_all'])} | {_cer(s['cer_dialogue'])} | "
                  f"{_cer(r['secondary']['dnc'])} | {name in doc['result']['eligible']} | {bar} | {better} |")
    return "\n".join(md)


def pareto_md(doc):
    p = doc["pareto"]
    md = ["| arm | role | VTSP B | est. ms @333 / @444 | arena B | F1 | CER all | dialogue |",
          "|---|---|---:|---:|---:|---:|---:|---:|"]
    for n in sorted(p["front"], key=lambda k: (p["points"][k]["est_ms_333_960"], p["points"][k]["blob_bytes"], k)):
        x = p["points"][n]
        md.append(f"| `{n}` | {x['role']} | {x['blob_bytes']:,} | {_ms(x['est_ms_333_960'])} | {x['arena_960']:,} | "
                  f"{x['f1']:.3f} | {x['cer_all']:.2f} | {x['cer_dialogue']:.2f} |")
    md += ["", f"{len(p['front'])} of {len(p['points'])} points (eligible candidates, ref and fast_ref) are on the "
               "front of est. ms, VTSP bytes, arena, F1, CER all and dialogue; est. ms are A9 cycle-model estimates."]
    return "\n".join(md)


def bought_md(doc):
    md = ["| what | decode | minuend − subtrahend | F1 Δ (95% CI) | CER all Δ, points | dialogue Δ | D+N+C Δ |",
          "|---|---|---|---:|---:|---:|---:|"]
    for b in doc["bought"]:
        if b.get("missing"):
            md.append(f"| {b['what']} | {b['decode']} | not run | – | – | – | – |")
            continue
        c = b["comparison"]
        md.append(f"| {b['what']} | {b['decode']} | `{b['minuend']}` − `{b['subtrahend']}` | {ci_txt(c['f1'], 4)} | "
                  f"{ci_txt(c['cer_all'], 2)} | {ci_txt(c['cer_dialogue'], 2)} | {ci_txt(b['dnc'], 2)} |")
    return "\n".join(md)


def ceiling_md(doc):
    md = ["| teacher arm (report only) | P | R | F1 | CER all | dialogue | D+N+C | text-free frames with a box | "
          "F1 − ref | CER all − ref | dialogue − ref |", "|---|" + "---:|" * 10]
    for n, c in doc["ceiling"].items():
        s, v = c["summary"], c["vs_ref"]
        d = s["detection"]
        md.append(f"| `{n}` | {TS.fmt(d['precision'], 3)} | {TS.fmt(d['recall'], 3)} | {TS.fmt(d['f1'], 3)} | "
                  f"{_cer(s['cer_all'])} | {_cer(s['cer_dialogue'])} | {_cer(c['secondary']['dnc'])} | "
                  f"{s['text_free']['with_any_box']} / {s['text_free']['frames']} | {ci_txt(v['f1'], 4)} | "
                  f"{ci_txt(v['cer_all'], 2)} | {ci_txt(v['cer_dialogue'], 2)} |")
    return "\n".join(md) if doc["ceiling"] else "The ceiling was not run."


def seeds_md(doc):
    md = ["| seed | decode | source | VTSP B | est. ms @333 | F1 | CER all | dialogue | D+N+C | eligible |",
          "|---|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for s in sorted(doc["seeds"], key=lambda x: (x["arm_id"], x["decode"], x["seed"])):
        md.append(f"| `{s['name']}` | {s['decode']} | {s['source']} | {s['blob_bytes']:,} | "
                  f"{_f(s['est_ms_333_960'], 1)} | {TS.fmt(s['f1'], 3)} | {_cer(s['cer_all'])} | "
                  f"{_cer(s['cer_dialogue'])} | {_cer(s['dnc'])} | {s['eligible']} |")
    return "\n".join(md)


def parts_md(doc, name=None):
    res = doc["result"]
    name = name or res["main"]["winner"] or res["fast"]["winner"]
    if name is None:
        el = [n for n in res["eligible"]]
        name = min(el, key=lambda n: doc["representatives"][n]["summary"]["cer_all"]["cer"]) if el else None
    if name is None:
        return "No eligible candidate: no per-part or per-class table."
    refn = next(n for n, r in doc["references"].items() if r["role"] == "ref")
    arms = [(name, doc["representatives"][name]["summary"]), (refn, doc["references"][refn]["summary"])]
    md = ["| part | frames | " + " | ".join(f"`{a}` F1 / CER all / dialogue" for a, _ in arms) + " |",
          "|---|---:|" + "---|" * len(arms)]
    for p in arms[0][1]["per_part"]:
        cells = []
        for _, s in arms:
            q = s["per_part"].get(p, {})
            cells.append(f"{TS.fmt((q.get('detection') or {}).get('f1'), 3)} / {_cer(q.get('cer_all'))} / "
                         f"{_cer(q.get('cer_dialogue'))}")
        md.append(f"| {p} | {arms[0][1]['per_part'][p]['frames']} | " + " | ".join(cells) + " |")
    allc = arms + [(n, c["summary"]) for n, c in doc["ceiling"].items()]
    cls = M.CLASSES
    md += ["", "| recall by class | " + " | ".join(f"{c} ({arms[0][1]['recall_by_class'][c]['refs']})" for c in cls)
           + " |", "|---|" + "---:|" * len(cls)]
    for a, s in allc:
        md.append(f"| `{a}` | " + " | ".join(TS.fmt(s["recall_by_class"][c]["recall"], 3) for c in cls) + " |")
    return "\n".join(md)


def report_sl(doc, units_dirs, names=None):
    """{arm: {model: summary}} of the report recognizers (VOCR5-S, VOCR5-L) of the picks (or names), from the units
    directories (textspot2.py run --names RUN --models S,L); each file must score the boxes the record scored."""
    res = doc["result"]
    names = names or [n for n in (res["main"]["winner"], res["fast"]["winner"]) if n]
    seeds = {s["name"]: s for s in doc["seeds"]}
    out = {}
    for n in dict.fromkeys(names):
        for d in units_dirs or []:
            for m in ("VOCR5-S", "VOCR5-L"):
                f = Path(d) / n / f"units_{m}.json"
                if f.exists():
                    u = json.loads(f.read_text(encoding="utf-8"))
                    if u["meta"]["boxes_sha256"] != seeds[n]["boxes_sha256"]:
                        raise SystemExit(f"{f}: the units of other boxes than the record's")
                    out.setdefault(n, {})[m] = u["summary"]
    return out


def sl_md(sl):
    if not sl:
        return "No VOCR5-S / VOCR5-L units of the picks yet (textspot2.py run --names RUN --models S,L)."
    md = ["| pick | recognizer | F1 | CER all | dialogue |", "|---|---|---:|---:|---:|"]
    for n, by in sl.items():
        for m, s in sorted(by.items()):
            md.append(f"| `{n}` | {m} | {TS.fmt(s['detection']['f1'], 3)} | {_cer(s['cer_all'])} | "
                      f"{_cer(s['cer_dialogue'])} |")
    return "\n".join(md)


def render(doc, part_arm=None, sl=None):
    return "\n".join(["### The selection", "", selection_md(doc), "", "### Candidates and controls (median seeds, cost "
                      "order)", "", candidates_md(doc), "", "### Pareto front", "", pareto_md(doc), "",
                      "### What each arm bought", "", bought_md(doc), "", "### The teacher ceiling (report only)", "",
                      ceiling_md(doc), "", "### Per part and per class", "", parts_md(doc, part_arm), "",
                      "### The picks read by VOCR5-S and VOCR5-L (report)", "", sl_md(sl), "",
                      "### Every seed", "", seeds_md(doc), ""])


def cmd_tables(args):
    doc = load_json(args.dev)
    text = render(doc, args.part_arm or None, report_sl(doc, args.units))
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)


# ----------------------------------------------------------------------------- main

def cmd_cbuild(args):
    rec = cbuild2(args.out, args.compiler)
    ref = T / "eval" / "cbuild" / "build.json"
    same = {}
    if ref.exists():
        old = json.loads(ref.read_text(encoding="utf-8"))["binaries"]
        same = {k: v == old.get(k) for k, v in rec["binaries"].items() if k in old}
    print(json.dumps({"binaries": rec["binaries"], "driver2": rec["driver2"], "same_as_textspot": same}, indent=1))


def cmd_seed_cache(args):
    print(json.dumps(seed_cache(args.cache), indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("cbuild")
    p.add_argument("--out", default=str(DEFAULT["cbuild"]))
    p.add_argument("--compiler", default="gcc")
    p = sub.add_parser("seed-cache")
    p.add_argument("--cache", default=str(DEFAULT["cache"]))
    p = sub.add_parser("fetch-teacher")
    p.add_argument("--models", default=str(DEFAULT["models"]))
    for name in ("run", "ceiling"):
        p = sub.add_parser(name)
        p.add_argument("--set", default="dev", choices=("dev", "pseudo"))
        p.add_argument("--manifest", default="")
        p.add_argument("--out", default=str(DEFAULT["units" if name == "run" else "ceiling"]))
        p.add_argument("--cbin", default=str(DEFAULT["cbuild"]) if name == "run" else "",
                       help="a cbuild directory (textspot2.py cbuild), 'auto', or '' for none")
        p.add_argument("--models", default="N", help="horizontal recognizers (N,S,L; none: detection only)")
        p.add_argument("--workers", type=int, default=5)
        p.add_argument("--shots", action="append", default=[], help="more source screenshot directories")
        p.add_argument("--cache", default="")
        p.add_argument("--rbuild", default=str(DEFAULT["rbuild"]))
        p.add_argument("--frames", default="", help="the frames' hashes file (dev default: T2/eval/frames.json)")
        p.add_argument("--compiler", default="gcc")
        p.add_argument("--limit", type=int, default=0)
        if name == "run":
            p.add_argument("--runs", default=str(DEFAULT["runs"]))
            p.add_argument("--names", nargs="*", default=None)
            p.add_argument("--force", action="store_true")
        else:
            p.add_argument("--sizes", default=",".join(CEILING_SIZES))
            p.add_argument("--models-dir", default=str(DEFAULT["models"]))
            p.add_argument("--threads", type=int, default=5)
    for name in ("headline", "select"):
        p = sub.add_parser(name)
        p.add_argument("--units", action="append", default=None)
        p.add_argument("--a9", required=name == "select", default="", help="eval/textspot2_a9_<date>.json")
        p.add_argument("--a9-d2s", default="", help="the d2s A9 table, when d2s arms ran")
        p.add_argument("--a9-textspot", default=str(TS_A9))
        p.add_argument("--cbin", default="", help="a cbuild directory (the refinement's arena from its driver)")
        p.add_argument("--runs", default=str(DEFAULT["runs"]))
        p.add_argument("--frames", default=str(DEFAULT["frames"]))
        p.add_argument("--cache", default=str(DEFAULT["cache"]))
        p.add_argument("--out", required=name == "select", default="")
        if name == "select":
            p.add_argument("--baseline", default=str(BASELINE))
            p.add_argument("--ceiling", default=str(DEFAULT["ceiling"]))
            p.add_argument("--indent-copy", default="")
    p = sub.add_parser("tables")
    p.add_argument("--dev", required=True)
    p.add_argument("--units", action="append", default=[], help="unit directories with the picks' S / L units")
    p.add_argument("--part-arm", default="")
    p.add_argument("--out", default="")
    args = ap.parse_args(argv)
    S2.register()                  # the protocol's shapes resolve in train/textdet_model.SHAPES (idempotent)
    if getattr(args, "units", "x") is None:
        args.units = [str(DEFAULT["units"])]
    {"cbuild": cmd_cbuild, "seed-cache": cmd_seed_cache, "fetch-teacher": cmd_fetch_teacher, "run": cmd_run,
     "ceiling": cmd_ceiling, "headline": cmd_headline, "select": cmd_select, "tables": cmd_tables}[args.cmd](args)


if __name__ == "__main__":
    main()
