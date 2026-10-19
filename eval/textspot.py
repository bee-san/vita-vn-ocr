#!/usr/bin/env python3
"""textspot: the evaluation harness of the full-frame text spotter (issue #104, docs/textspot.md). Every arm of
eval/textspot_protocol.json ("baselines", "candidates") on whole frames, read as its "reading" says and scored by
the frozen metrics (eval/textspot_metrics.py: match_frame, e2e_frame, arm_summary, apply_rule).

  textspot.py run --set dev --arms SPEC[,SPEC...] --out DIR [--cbin DIR|auto] [--models N,S,L|none] [--workers 5]
  textspot.py run --set pseudo --manifest JSONL --arms ... --out DIR [...]     training-side pseudo frames only
  textspot.py eligible --units DIR --a9 JSON [--arms SPEC,...]                 the candidates the rule can read
  textspot.py baseline --units DIR --out eval/textspot_baseline_<date>.json    the frozen baseline
  textspot.py select --baseline JSON --units DIR [--units DIR ...] --a9 JSON --out eval/textspot_dev_<date>.json
  textspot.py tables --baseline JSON [--dev JSON]                              markdown for docs/textspot.md
  textspot.py cbuild --out DIR                                                 the C drivers, built from this checkout
  textspot.py parity --set dev --arms ... --units DIR --scratch DIR --cbin DIR --out JSON
                                                   C parity of arms already run, as a separate record (--parity)
  textspot.py pseudo --out JSONL [--n 30]                  a training-side pseudo set (harness tests; never a score)

Frames. A row of eval/manifests/textspot-{dev,test}.jsonl (eval/textspot_sets.py writes them) names its source
screenshot by URL and sha256; the file is found as <vn>_<file> in the local screenshot directories (SHOT_DIRS,
--shots), checked against the row's sha256, decoded to RGB and scaled as the row's `scale` says (with
eval/textspot_sets.py's own function when it is importable). Every arm reads the frame's grey (synth/common.to_gray:
(77 R + 150 G + 29 B) >> 8), as the console does. The dev manifest must be pinned by a protocol amendment before
any arm reads it; the test manifest is read only by eval/textspot_protected.py.

Arms (SPEC):
  grad                  synth/common.find_lines_ex(frame, "grad"): the whole frame as one region
  grad_refine           grad, then synth/lines_refine.refine_lines with VOCR_REFINE_DEFAULT
  grad_refine_gate      grad_refine behind #101's p8_c8 gate in boxes mode (k 4, t -8): no box when it says no text
  s8c24_textdet         #97's s8_c24 through its own finder (synth/textdet_twin.find_lines): at most 16 lines,
                        horizontal and vertical
  s8c24_textdet_refine  eval/textdet.det_refine of those lines (#97's det+refine)
  s8c24_spot            s8_c24 through the textspot decode at full resolution (textspot_twin.pack(s8_c24, 0, 1))
  columns               synth/vertical_twin.find_columns(frame, 1) (VOCR_COLUMNS_VALLEYS): vertical boxes only
  grad_refine_columns   grad_refine's boxes plus every column of which less than half the area lies inside one
                        grad_refine box
  baselines             all eight
  det:NAME=VTSP         a candidate through the textspot decode (textspot_twin.spot)
  detref:NAME=VTSP      the same, then its horizontal boxes (in the decode's order) through refine_lines on the
                        frame; its vertical boxes unchanged
  A:RUN@HALF            family A: #97's run RUN (sweep/ or sweep_r3/ final.vtd) packed with half_above HALF and
                        conf_min 1, as det and det+refine (arms A_<RUN>_h<HALF>_det, A_<RUN>_h<HALF>_detref); a
                        suffix :det or :detref keeps one decode (also on famA, B:RUN, famB)
  famA                  every family A run at half_above 0 and 600
  B:RUN, famB           family B: $T/runs/RUN/final.vtsp (finished runs: done.json), det and det+refine
Every arm's boxes go through textspot_twin.group_and_order (blocks, reading order, pick); a box without a dark flag
of its own gets common.line_dark on the frame (every arm above has one). With --cbin the C drivers run on every
frame too (runtime/vocr_lines_test, vocr_lines_refine_test, vocr_textgate_test's finder mode, vocr_textdet_test,
vocr_vertical_test, vocr_refine_lines through ctypes for det+refine, runtime/vocr_textspot_test once it exists) and
each arm records c_equal per frame: its C boxes equal the twin's (None where no C driver exists yet).

Reading (the protocol's "reading"): every box cut from the frame's grey, read by the C runtime in W8I with greedy
decoding (eval/finder_tighten.Reader), VOCR_INVERT from its dark flag; a vertical box turned with
vertical_twin.rotate_ccw and read by VOCR5-V; horizontal boxes by VOCR5-N (the rule's) and VOCR5-S / VOCR5-L (report
arms). The four blobs are sha256-checked against the protocol. A cache keyed by (frame sha256, box, dark, vertical,
recognizer sha256) lets arms that return the same box share one reading: $T/eval/cache (dev), $P/cache (test).

Scoring: per frame textspot_metrics.match_frame and e2e_frame, plus the pick's report measures; one unit per frame;
DIR/<arm>/units_<recognizer>.json with the units and their summaries (pooled P / R / F1, recall per class,
tightness, CER all and dialogue, text-free false alarms, the pick, per part). DIR/<arm>/boxes.json holds the arm's
boxes per frame (no text). Units and boxes stay under $T; the committed records hold counts, hashes and metrics.
"""
import argparse
import ctypes as C
import datetime
import hashlib
import json
import os
import statistics
import struct
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "eval", ROOT / "synth", ROOT / "runtime", ROOT / "train"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import textspot_metrics as M  # noqa: E402
from metrics import lev, norm  # noqa: E402

HERD = Path(os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr"))
ART = HERD / "shared" / "artifacts"
T = ART / "textspot"
P = HERD / "shared" / "private" / "textspot"
PROTOCOL = ROOT / "eval" / "textspot_protocol.json"
MANIFESTS = {"dev": ROOT / "eval" / "manifests" / "textspot-dev.jsonl",
             "test": ROOT / "eval" / "manifests" / "textspot-test.jsonl"}
CACHES = {"dev": T / "eval" / "cache", "pseudo": T / "eval" / "pseudo" / "cache", "test": P / "cache"}
BLOBS = {"VOCR5-S": ART / "vocr5-train" / "release" / "v0.3.0" / "VOCR5_S_w8.vocr",
         "VOCR5-N": ART / "vocr5-train" / "release" / "v0.3.0" / "VOCR5_N_w8.vocr",
         "VOCR5-L": ART / "vocr5-train" / "release" / "v0.3.0" / "VOCR5_L_w8.vocr",
         "VOCR5-V": ART / "vertical" / "release" / "v0.3.2-vertical.1" / "VOCR5_V_w8.vocr"}
S8C24 = ART / "textdet" / "release" / "v0.3.1-textdet.1" / "textdet_s8_c24_w8.vtd"
GATE = ART / "textgate" / "release" / "v0.3.2-textgate.1" / "textgate_p8_c8_w8.vtd"
FAMILY_A = (ART / "textdet" / "sweep", ART / "textdet" / "sweep_r3")
FAMILY_B = T / "runs"
SHOT_DIRS = (ART / "realdata" / "shots" / "vita", ART / "realdata" / "shots" / "psp",
             ART / "realdata" / "shots" / "hires", ART / "devnative" / "shots" / "vita",
             ART / "devnative" / "shots" / "psp", ART / "realplus" / "shots" / "hires", T / "sets" / "dev" / "shots",
             T / "sets" / "shots")
BASELINES = ("grad", "grad_refine", "grad_refine_gate", "s8c24_textdet", "s8c24_textdet_refine", "s8c24_spot",
             "columns", "grad_refine_columns")
H_MODELS = ("VOCR5-N", "VOCR5-S", "VOCR5-L")
RULE_MODEL, V_MODEL = "VOCR5-N", "VOCR5-V"
SHORT = {"N": "VOCR5-N", "S": "VOCR5-S", "L": "VOCR5-L"}
HALF_ABOVE = (0, 600)
REFINE_MS_PER_BOX = 1.4          # the rule's cost of VOCR_REFINE_DEFAULT per horizontal box (333 MHz, estimate)
A9_CAL = 1.0007                  # the protocol's calibration of the cycle model's high bound at 333 MHz
REFINE_FLAGS = 0x1F
CODE = ("eval/textspot.py", "eval/textspot_metrics.py", "eval/metrics.py", "eval/textdet.py", "eval/textgate.py",
        "eval/finder_tighten.py", "synth/textspot_twin.py", "synth/textdet_twin.py", "synth/common.py",
        "synth/lines_refine.py", "synth/vertical_twin.py", "runtime/vocr.c", "runtime/vocr.h", "runtime/pyvocr.py")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b):
    return hashlib.sha256(bytes(b)).hexdigest()


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def protocol(path=PROTOCOL):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rel(path):
    """A path as the records name it: repo-relative, $HERD-relative or as given."""
    p = Path(path).resolve()
    for base, label in ((ROOT.resolve(), ""), (HERD.resolve(), "$HERD/")):
        try:
            return label + str(p.relative_to(base))
        except ValueError:
            pass
    return str(path)


def code_identity(files=CODE):
    return {f: sha256(ROOT / f) for f in files if (ROOT / f).exists()}


def write_json(path, doc, indent=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(doc, indent=indent, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


# ----------------------------------------------------------------------------- the protocol's pins

def pinned_sha(proto, relpath):
    """The sha256 an amendment pins for relpath (a key equal to the path whose value is a sha256, or an object
    {"path": relpath, "sha256": ...}), the latest amendment winning; None when no amendment pins it."""
    found = []

    def walk(v):
        if isinstance(v, dict):
            if v.get("path") == relpath and isinstance(v.get("sha256"), str):
                found.append(v["sha256"])
            for k, x in v.items():
                if k == relpath and isinstance(x, str) and len(x) == 64:
                    found.append(x)
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)
    for a in proto.get("amendments", []):
        walk(a)
    return found[-1] if found else None


def check_blob(path, want, what):
    got = sha256(path)
    if got != want:
        raise SystemExit(f"{what}: {path} has sha256 {got}, the protocol pins {want}")
    return Path(path)


def recognizer_paths(models, proto, paths=None):
    """{model: path} of the horizontal models asked for and VOCR5-V, each sha256-checked against the protocol."""
    paths = dict(BLOBS, **(paths or {}))
    out = {}
    for m in list(models) + [V_MODEL]:
        out[m] = check_blob(paths[m], proto["recognizers"][m]["sha256"], m)
    return out


# ----------------------------------------------------------------------------- frames

class Frame:
    """One manifest row with its frame's grey: id, vn, part, text_free, refs, width, height, grey, sha."""

    __slots__ = ("id", "vn", "part", "text_free", "refs", "width", "height", "grey", "sha", "row")

    def __init__(self, row, grey, sha):
        self.row = row
        self.id, self.vn, self.part = row["id"], row["vn"], row.get("part", "")
        self.text_free = bool(row.get("text_free"))
        self.refs = row.get("refs", [])
        self.grey = np.ascontiguousarray(grey, dtype=np.uint8)
        self.height, self.width = self.grey.shape
        self.sha = sha


def pixel_sha256(rgb):
    """sha256 of decoded pixels ('HxWxC' then the bytes; devnative/paths.pixel_sha256's form): the frame's id in
    the reading cache."""
    a = np.ascontiguousarray(rgb, dtype=np.uint8)
    h = hashlib.sha256(f"{a.shape[0]}x{a.shape[1]}x{a.shape[2] if a.ndim == 3 else 1}".encode())
    h.update(a.tobytes())
    return h.hexdigest()


def read_rgb(path):
    """HxWx3 uint8 RGB as OpenCV decodes it (devnative/paths.read_rgb: the decoder the labelling tools use)."""
    import cv2
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"{path}: not an image OpenCV can decode")
    return np.ascontiguousarray(bgr[:, :, ::-1])


def shot_file_name(vn, url):
    return f"{vn}_{url.rsplit('/', 1)[-1]}"


def source_path(row, dirs):
    """The row's source screenshot: its `path`, else <vn>_<file> in one of dirs; the first whose sha256 is the
    row's."""
    cands = [Path(row["path"])] if row.get("path") else []
    name = shot_file_name(row["vn"], row["url"])
    cands += [Path(d) / name for d in dirs]
    for p in cands:
        if p.is_file() and sha256(p) == row["sha256"]:
            return p
    raise FileNotFoundError(f"{row['id']}: no file {name} with sha256 {row['sha256'][:12]}... in {len(dirs)} dirs")


def _sets_build_frame():
    """eval/textspot_sets.build_frame(rgb, dims, scale) (the sets' own source-to-frame step), when importable."""
    try:
        import textspot_sets as TSS
    except ImportError:
        return None
    fn = getattr(TSS, "build_frame", None)
    return fn if callable(fn) else None


INTERP = {"INTER_LINEAR": "INTER_LINEAR", "LINEAR": "INTER_LINEAR", "BILINEAR": "INTER_LINEAR",
          "INTER_AREA": "INTER_AREA", "AREA": "INTER_AREA"}


def _resize_steps(rgb, steps, dims):
    import cv2
    out = rgb
    for st in steps:
        method = INTERP[str(st["method"]).upper().replace("CV2.", "")]
        w, h = st.get("dims") or dims
        out = cv2.resize(out, (int(w), int(h)), interpolation=getattr(cv2, method))
    return out


def scale_frame(row, rgb, build=None):
    """The frame of a row from its source's RGB. A row's scale is null (a native frame) or {"method":
    "cv2.INTER_LINEAR" | "cv2.INTER_AREA", "factor"} (resize to the row's dims): eval/textspot_sets.build_frame
    applies it when that module is importable (build: an explicit function, for tests), else the same cv2.resize
    here. A list of steps, each with its own "dims", is the pseudo set's form (resized in turn)."""
    steps = row.get("scale")
    if isinstance(steps, list):
        out = _resize_steps(rgb, steps, row["dims"])
    else:
        build = build if build is not None else _sets_build_frame()
        if build is not None:
            out = build(rgb, row["dims"], steps)
        elif not steps:
            out = rgb
        else:
            out = _resize_steps(rgb, [dict(steps, dims=row["dims"])], row["dims"])
    if [out.shape[1], out.shape[0]] != list(row["dims"]):
        raise ValueError(f"{row['id']}: frame {out.shape[1]}x{out.shape[0]} is not the row's dims {row['dims']}")
    return np.ascontiguousarray(out)


def load_frame(row, dirs):
    from common import to_gray
    src = read_rgb(source_path(row, dirs))
    if row.get("src_dims") and [src.shape[1], src.shape[0]] != list(row["src_dims"]):
        raise ValueError(f"{row['id']}: source is {src.shape[1]}x{src.shape[0]}, the row says {row['src_dims']}")
    rgb = scale_frame(row, src)
    return Frame(row, to_gray(rgb), pixel_sha256(rgb))


def load_frames(rows, dirs):
    return [load_frame(r, dirs) for r in rows]


# ----------------------------------------------------------------------------- arms

class Spec:
    """One arm: a baseline (base = its name) or a candidate (decode det / detref of a VTSP blob)."""

    def __init__(self, name, kind, base=None, blob=None, path=None, half_above=None, family=None, run=None,
                 text=None):
        self.name, self.kind, self.base = name, kind, base
        self.blob, self.path, self.half_above, self.family, self.run = blob, path, half_above, family, run
        self.text = text or name

    @property
    def decode(self):
        return {"det": "det", "detref": "det+refine"}.get(self.kind)

    @property
    def net_key(self):
        return None if self.blob is None else "spot:" + sha256_bytes(self.blob)

    def meta(self):
        import textspot_twin as S
        d = {"arm": self.name, "spec": self.text, "kind": self.kind, "baseline": self.base, "decode": self.decode,
             "family": self.family, "run": self.run}
        if self.blob is not None:
            m = S.Model(self.blob)
            d.update(blob_sha256=sha256_bytes(self.blob), blob_bytes=m.blob_bytes, shape=m.name,
                     stride=m.stride, half_above=m.half_above, conf_min=m.conf_min,
                     source=rel(self.path) if self.path else None)
        return d


def family_a_runs(dirs=FAMILY_A):
    out = {}
    for d in dirs:
        for p in sorted(Path(d).glob("*/final.vtd")):
            out.setdefault(p.parent.name, p)
    return out


def family_b_runs(directory=FAMILY_B):
    """Finished family B runs (done.json and final.vtsp present)."""
    return {p.parent.name: p for p in sorted(Path(directory).glob("*/final.vtsp")) if (p.parent / "done.json").exists()}


def candidate_specs(name, blob, path, family=None, run=None, half_above=None, kinds=("det", "detref"), text=None):
    return [Spec(f"{name}_{k}", k, blob=blob, path=path, half_above=half_above, family=family, run=run,
                 text=text) for k in kinds]


def _decode_suffix(tok):
    """'A:RUN@600:det' -> ('A:RUN@600', ('det',)); no suffix: both decodes."""
    for k in ("detref", "det"):
        if tok.endswith(":" + k):
            return tok[:-len(k) - 1], (k,)
    return tok, ("det", "detref")


def parse_arms(text, fam_a=None, fam_b=None):
    """The arm specs of a comma-separated list (see the module docstring), in order, without duplicates."""
    import textspot_twin as S
    out, seen = [], set()

    def add(specs):
        for s in specs:
            if s.name not in seen:
                seen.add(s.name)
                out.append(s)
    for tok in [t.strip() for t in text.split(",") if t.strip()]:
        if tok == "baselines":
            add(Spec(b, "baseline", base=b) for b in BASELINES)
        elif tok in BASELINES:
            add([Spec(tok, "baseline", base=tok)])
        elif tok.startswith(("det:", "detref:")):
            kind, _, rest = tok.partition(":")
            name, _, path = rest.partition("=")
            blob = Path(path).read_bytes()
            m = S.Model(blob)
            add([Spec(f"{name}_{kind}", kind, blob=blob, path=path, half_above=m.half_above, text=tok)])
        elif _decode_suffix(tok)[0] == "famA" or tok.startswith("A:"):
            runs = fam_a if fam_a is not None else family_a_runs()
            body, kinds = _decode_suffix(tok)
            if body == "famA":
                items = [(r, h) for r in runs for h in HALF_ABOVE]
            else:
                run, _, half = body[2:].partition("@")
                items = [(run, int(h)) for h in (half.split("+") if half else map(str, HALF_ABOVE))]
            for run, h in items:
                vtdt = Path(runs[run]).read_bytes()
                add(candidate_specs(f"A_{run}_h{h}", S.pack(vtdt, h, 1), runs[run], "A", run, h, kinds,
                                    text=f"A:{run}@{h}"))
        elif _decode_suffix(tok)[0] == "famB" or tok.startswith("B:"):
            runs = fam_b if fam_b is not None else family_b_runs()
            body, kinds = _decode_suffix(tok)
            names = list(runs) if body == "famB" else [body[2:]]
            for run in names:
                blob = Path(runs[run]).read_bytes()
                add(candidate_specs(f"B_{run}", blob, runs[run], "B", run, S.Model(blob).half_above, kinds,
                                    text=f"B:{run}"))
        else:
            raise SystemExit(f"unknown arm {tok!r}")
    return out


def load_nets(specs):
    """The twin models the specs need: {key: model}."""
    import textdet_twin as TW
    import textspot_twin as S
    nets = {}
    bases = {s.base for s in specs if s.kind == "baseline"}
    if bases & {"s8c24_textdet", "s8c24_textdet_refine"}:
        nets["s8c24"] = TW.Model(S8C24.read_bytes())
    if "s8c24_spot" in bases:
        nets["s8c24_spot"] = S.Model(S.pack(S8C24.read_bytes(), 0, 1))
    if "grad_refine_gate" in bases:
        nets["gate"] = TW.Model(GATE.read_bytes())
    for s in specs:
        if s.blob is not None:
            nets[s.net_key] = S.Model(s.blob)
    return nets


def check_baseline_blobs(proto):
    b = proto["baselines"]
    check_blob(S8C24, b["s8c24_textdet"]["blob"]["sha256"], "s8_c24")
    check_blob(GATE, b["grad_refine_gate"]["gate"]["sha256"], "p8_c8")


def gate_params(proto=None):
    g = (proto or protocol())["baselines"]["grad_refine_gate"]["gate"]
    if g["mode"] != "boxes":
        raise SystemExit("the gate baseline is registered in boxes mode")
    return int(g["k"]), int(g["t"])


def ints(t):
    return tuple(int(v) for v in t)


def union_columns(hboxes, columns):
    """grad_refine_columns: the horizontal boxes plus every column of which less than half the area lies inside
    one horizontal box. Boxes are (x, y, w, h, ...) tuples; the result keeps their order (columns after)."""
    out = list(hboxes)
    for c in columns:
        a = c[2] * c[3]
        if all(2 * M.inter(c[:4], h[:4]) < a for h in hboxes):
            out.append(c)
    return out


def refine_spot_boxes(boxes, grey):
    """det+refine: the decode's horizontal boxes, in its order, through refine_lines (VOCR_REFINE_DEFAULT) on the
    frame (a refined box keeps its source box's confidence); its vertical boxes unchanged. boxes: the decode's
    (x, y, w, h, vertical, conf, dark, ...) -> ([(x, y, w, h, vertical, conf, dark)], horizontal boxes in)."""
    import lines_refine as LR
    hz = [b for b in boxes if not b[4]]
    rb, rd, org = LR.refine_lines(grey, [b[:4] for b in hz], [b[6] for b in hz], LR.DEFAULT)
    out = [(*ints(bx), 0, int(hz[o][5]), int(dk)) for bx, dk, o in zip(rb, rd, org)]
    return out + [ints(b[:7]) for b in boxes if b[4]], len(hz)


class Twin:
    """The twins of every arm on one frame; GRAD, the refinement, the gate and each network's decode run once and
    are shared. comp holds each component's twin output (what the C drivers must reproduce)."""

    def __init__(self, frame, nets, gate=(4, -8)):
        self.g, self.H, self.nets, self.gate_kt = frame.grey, frame.height, nets, gate
        self.comp = {}

    def _memo(self, key, fn):
        if key not in self.comp:
            self.comp[key] = fn()
        return self.comp[key]

    def grad(self):
        def run():
            from common import find_lines_ex
            b, d, t = find_lines_ex(self.g, "grad")
            return [ints(x) for x in b], [int(v) for v in d], int(bool(t))
        return self._memo("grad", run)

    def refine(self):
        def run():
            import lines_refine as LR
            b, d, _ = self.grad()
            rb, rd, _ = LR.refine_lines(self.g, b, d, LR.DEFAULT)
            return [ints(x) for x in rb], [int(v) for v in rd]
        return self._memo("refine", run)

    def gate(self):
        def run():
            import textgate as G
            m = self.nets["gate"]
            k, t = self.gate_kt
            sc = int(G.score(G.kmap(m, self.g), "boxes", k, m.stride, self.grad()[0]))
            return sc, sc >= t
        return self._memo("gate", run)

    def textdet(self):
        def run():
            import textdet_twin as TW
            lines, tr = TW.find_lines(self.nets["s8c24"], self.g)
            return [ints(ln) for ln in lines], int(bool(tr))
        return self._memo("textdet", run)

    def textdet_refine(self):
        def run():
            import textdet as TD
            return [ints(ln) for ln in TD.det_refine(self.textdet()[0], self.g)]
        return self._memo("textdet_refine", run)

    def columns(self):
        def run():
            import vertical_twin as V
            b, d, t = V.find_columns(self.g, 1)
            return [ints(x) for x in b], [int(v) for v in d], int(bool(t))
        return self._memo("columns", run)

    def spot(self, key):
        def run():
            import textspot_twin as S
            r = S.spot(self.nets[key], self.g)
            return ([ints((b.x, b.y, b.w, b.h, b.vertical, b.conf, b.dark, b.block)) for b in r.boxes],
                    [ints((k.x, k.y, k.w, k.h, k.vertical, k.first, k.count)) for k in r.blocks],
                    int(r.pick), int(r.truncated))
        return self._memo(key, run)

    def spot_refine(self, key):
        def run():
            boxes = self.spot(key)[0]
            out, n = refine_spot_boxes(boxes, self.g)
            hz = [b for b in boxes if not b[4]]
            return out, n, [b[:4] for b in hz], [b[6] for b in hz]
        return self._memo(key + ":refine", run)

    # ---- the arms

    def raw(self, spec):
        """(boxes [(x, y, w, h, vertical, conf, dark)] before blocks, truncated, horizontal boxes refined (det+refine)
        or None, spot result or None)."""
        k = spec.base if spec.kind == "baseline" else spec.kind
        if k == "grad":
            b, d, t = self.grad()
            return [(*x, 0, 0, dk) for x, dk in zip(b, d)], t, None, None
        if k == "grad_refine":
            b, d = self.refine()
            return [(*x, 0, 0, dk) for x, dk in zip(b, d)], self.grad()[2], None, None
        if k == "grad_refine_gate":
            if not self.gate()[1]:
                return [], 0, None, None
            b, d = self.refine()
            return [(*x, 0, 0, dk) for x, dk in zip(b, d)], self.grad()[2], None, None
        if k == "s8c24_textdet":
            lines, t = self.textdet()
            return [(x, y, w, h, v, 0, dk) for x, y, w, h, dk, v in lines], t, None, None
        if k == "s8c24_textdet_refine":
            lines = self.textdet_refine()
            return [(x, y, w, h, v, 0, dk) for x, y, w, h, dk, v in lines], self.textdet()[1], None, None
        if k == "s8c24_spot":
            r = self.spot("s8c24_spot")
            return [b[:7] for b in r[0]], r[3], None, r
        if k == "columns":
            b, d, t = self.columns()
            return [(*x, 1, 0, dk) for x, dk in zip(b, d)], t, None, None
        if k == "grad_refine_columns":
            hb, hd = self.refine()
            cb, cd, ct = self.columns()
            boxes = union_columns([(*x, 0, 0, dk) for x, dk in zip(hb, hd)], [(*x, 1, 0, dk) for x, dk in zip(cb, cd)])
            return boxes, int(bool(self.grad()[2] or ct)), None, None
        if k == "det":
            r = self.spot(spec.net_key)
            return [b[:7] for b in r[0]], r[3], None, r
        if k == "detref":
            out, n, _, _ = self.spot_refine(spec.net_key)
            return out, self.spot(spec.net_key)[3], n, None
        raise ValueError(f"unknown arm kind {k}")

    def arm(self, spec):
        """{"boxes": [(x, y, w, h, vertical, conf, dark, block)] in reading order, "blocks", "pick", "truncated",
        "h_in"}: the arm's raw boxes through textspot_twin.group_and_order."""
        import textspot_twin as S
        from common import line_dark
        boxes, truncated, h_in, r = self.raw(spec)
        if r is not None:                                   # the decode's own blocks, order and pick
            return {"boxes": r[0], "blocks": r[1], "pick": r[2], "truncated": r[3], "h_in": h_in}
        boxes = [b if b[6] is not None else (*b[:6], int(line_dark(self.g, b[:4]))) for b in boxes]
        ordered, blocks, pick = S.group_and_order(boxes, self.H)
        return {"boxes": [ints(b) for b in ordered], "blocks": [ints(b) for b in blocks], "pick": int(pick),
                "truncated": int(truncated), "h_in": h_in}


# ---- detection over every frame (worker processes share the frames by fork)

_W = {}


def _detect_one(i):
    f = _W["frames"][i]
    tw = Twin(f, _W["nets"], _W["gate"])
    arms = {s.name: tw.arm(s) for s in _W["specs"]}
    return arms, tw.comp


def detect(frames, specs, nets, gate, workers=1):
    """[(arms {name: arm result}, twin components)] per frame."""
    _W.update(frames=frames, specs=specs, nets=nets, gate=gate)
    try:
        if workers <= 1 or len(frames) < 2:
            return [_detect_one(i) for i in range(len(frames))]
        import multiprocessing as mp
        with mp.get_context("fork").Pool(workers) as pool:
            return pool.map(_detect_one, range(len(frames)), chunksize=max(1, len(frames) // (4 * workers)))
    finally:
        _W.clear()


# ----------------------------------------------------------------------------- the C drivers

C_FLAGS = ("-std=c99", "-O2", "-ffp-contract=off", "-Wall", "-Wextra", "-Werror")
DRIVERS = {"vocr_lines_test": ("vocr_lines.c", "vocr_lines_test.c"),
           "vocr_lines_refine_test": ("vocr_lines.c", "vocr_lines_refine.c", "vocr_lines_refine_test.c"),
           "vocr_textgate_test": ("vocr_lines.c", "vocr_lines_rows.c", "vocr_textgate.c", "vocr_textgate_test.c"),
           "vocr_textdet_test": ("vocr_lines.c", "vocr_lines_rows.c", "vocr_lines_refine.c", "vocr_textdet.c",
                                 "vocr_textdet_test.c"),
           "vocr_vertical_test": ("vocr_lines.c", "vocr_lines_rows.c", "vocr_vertical.c", "vocr_vertical_test.c"),
           "vocr_textspot_test": ("vocr_lines.c", "vocr_lines_rows.c", "vocr_textspot.c", "vocr_textspot_test.c")}
LIBS = {"libvocr_refine.so": ("vocr_lines.c", "vocr_lines_refine.c")}


def cbuild(out, compiler="gcc"):
    """The C drivers (and the refinement's shared library for det+refine) from this checkout's runtime/: those whose
    sources exist. Returns the build record (sha256 of every binary and source)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    env = {"PATH": os.environ.get("PATH", os.defpath), "LC_ALL": "C", "SOURCE_DATE_EPOCH": "0"}
    rt = ROOT / "runtime"
    rec = {"compiler": subprocess.run([compiler, "--version"], capture_output=True, text=True).stdout.splitlines()[0],
           "flags": list(C_FLAGS), "binaries": {}, "sources_sha256": {}}
    for name, srcs in list(DRIVERS.items()) + list(LIBS.items()):
        if not all((rt / s).exists() for s in srcs):
            continue
        extra = ["-fPIC", "-shared"] if name.endswith(".so") else []
        subprocess.run([compiler, *C_FLAGS, *extra, "-o", str(out / name), *srcs], check=True, cwd=rt, env=env)
        rec["binaries"][name] = sha256(out / name)
        for s in srcs:
            rec["sources_sha256"][f"runtime/{s}"] = sha256(rt / s)
    write_json(out / "build.json", rec, indent=1)
    return rec


class _Img(C.Structure):
    _fields_ = [("pixels", C.c_void_p), ("width", C.c_uint), ("height", C.c_uint), ("stride", C.c_size_t),
                ("bpp", C.c_uint)]


class _Line(C.Structure):
    _fields_ = [("x", C.c_uint), ("y", C.c_uint), ("w", C.c_uint), ("h", C.c_uint), ("dark", C.c_uint)]


LINES_MAX = 16


class _Set(C.Structure):
    _fields_ = [("lines", _Line * LINES_MAX), ("count", C.c_uint), ("truncated", C.c_uint)]


class CRefine:
    """vocr_refine_lines (runtime/vocr_lines_refine.c) on any boxes of a grey frame, through ctypes.

    A VocrLineSet holds at most 16 lines. The twin (lines_refine.refine_lines) gives a box room to split into
    16 - (lines so far) - (boxes after it) lines, so with more than 16 boxes no box ever splits; a full set of 16
    in C never splits either (room <= 1), so more than 16 boxes go through as sets of exactly 16 (the last one
    padded in front with the boxes before it, whose results are dropped) and give the twin's boxes one for one."""

    def __init__(self, library):
        self.lib = C.CDLL(str(library))
        self.lib.vocr_refine_arena_bytes.argtypes = [C.c_uint, C.c_uint]
        self.lib.vocr_refine_arena_bytes.restype = C.c_size_t
        self.lib.vocr_refine_lines.argtypes = [C.POINTER(_Img), C.c_uint, C.c_void_p, C.c_size_t, C.POINTER(_Set),
                                               C.POINTER(C.c_ulong)]
        self.lib.vocr_refine_lines.restype = C.c_int

    def _call(self, g, boxes, dark):
        H, W = g.shape
        im = _Img(g.ctypes.data, W, H, W, 1)
        s = _Set()
        s.count, s.truncated = len(boxes), 0
        for i, ((x, y, w, h), d) in enumerate(zip(boxes, dark)):
            s.lines[i] = _Line(x, y, w, h, d)
        need = self.lib.vocr_refine_arena_bytes(W, H)
        arena = np.zeros(need + 16, np.uint8)
        rows = C.c_ulong()
        rc = self.lib.vocr_refine_lines(C.byref(im), REFINE_FLAGS, arena.ctypes.data, need + 16, C.byref(s),
                                        C.byref(rows))
        if rc:
            raise RuntimeError(f"vocr_refine_lines rc={rc}")
        ls = s.lines[:s.count]
        return [(ln.x, ln.y, ln.w, ln.h) for ln in ls], [ln.dark for ln in ls]

    def refine(self, grey, boxes, dark):
        g = np.ascontiguousarray(grey, dtype=np.uint8)
        boxes, dark = [ints(b) for b in boxes], [int(d) for d in dark]
        n = len(boxes)
        if n <= LINES_MAX:
            return self._call(g, boxes, dark) if n else ([], [])
        ob, od = [], []
        for s0 in range(0, n, LINES_MAX):
            chunk = list(range(s0, min(n, s0 + LINES_MAX)))
            idx = list(range(chunk[0] - (LINES_MAX - len(chunk)), chunk[0])) + chunk
            b, d = self._call(g, [boxes[i] for i in idx], [dark[i] for i in idx])
            if len(b) != LINES_MAX:
                raise RuntimeError("a full set of 16 split a box")
            ob += b[LINES_MAX - len(chunk):]
            od += d[LINES_MAX - len(chunk):]
        return ob, od


def _run(cmd, timeout=7200):
    p = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, timeout=timeout,
                       stdin=subprocess.DEVNULL)
    if p.returncode:
        raise RuntimeError(f"{cmd[0]}: rc {p.returncode}: {p.stderr[-2000:]}")
    return p.stdout.splitlines()


class CDrivers:
    """The C drivers of a cbuild directory (missing ones: None results)."""

    def __init__(self, directory):
        self.dir = Path(directory)
        rec = self.dir / "build.json"
        self.record = json.loads(rec.read_text()) if rec.exists() else {}
        lib = self.dir / "libvocr_refine.so"
        self.crefine = CRefine(lib) if lib.exists() else None

    def has(self, name):
        return (self.dir / name).exists()

    def _batch(self, name, args, records):
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            for r in records:
                f.write(r)
            path = f.name
        try:
            return _run([self.dir / name, *args, path])
        finally:
            os.unlink(path)

    def grad(self, greys):
        """vocr_find_lines (GRAD) per grey frame: [(boxes, dark, truncated)] (runtime/vocr_lines_test)."""
        recs = [struct.pack("<5I", g.shape[1], g.shape[0], 1, 0, g.shape[1]) + g.tobytes() for g in greys]
        out = []
        for line in self._batch("vocr_lines_test", [], recs):
            v = [int(t) for t in line.split()]
            if v[0]:
                raise RuntimeError(f"vocr_find_lines rc={v[0]}")
            n = v[1]
            out.append(([tuple(v[3 + 5 * i:7 + 5 * i]) for i in range(n)], [v[7 + 5 * i] for i in range(n)], v[2]))
        if len(out) != len(greys):
            raise RuntimeError("vocr_lines_test: one result per frame expected")
        return out

    def refine(self, greys):
        """GRAD, then vocr_refine_lines (VOCR_REFINE_DEFAULT): [(boxes, dark)] (runtime/vocr_lines_refine_test)."""
        import finder_tighten as FT
        return [(b, d) for _, b, d, _ in FT.c_refine(self.dir / "vocr_lines_refine_test", list(greys), REFINE_FLAGS)]

    def gated(self, greys, blob, k, t):
        """vocr_find_lines_gated (GRAD, then the gate in boxes mode): [(boxes, dark, truncated)]
        (runtime/vocr_textgate_test --batch, the finder mode)."""
        flags = 256 | 4 | (int(k) << 4)
        recs = [struct.pack("<4I", g.shape[1], g.shape[0], 1, flags) + g.tobytes() + struct.pack("<I", 0) +
                struct.pack("<2i", 0, int(t)) for g in greys]
        out = []
        for line in self._batch("vocr_textgate_test", ["--batch", blob], recs):
            v = [int(x) for x in line.split()]
            if v[0]:
                raise RuntimeError(f"vocr_find_lines_gated rc={v[0]}")
            n = v[1]
            out.append(([tuple(v[3 + 5 * i:7 + 5 * i]) for i in range(n)], [v[7 + 5 * i] for i in range(n)], v[2]))
        if len(out) != len(greys):
            raise RuntimeError("vocr_textgate_test: one result per frame expected")
        return out

    def textdet(self, greys, blob, flags):
        """[(lines (x, y, w, h, dark, vertical), truncated)] (runtime/vocr_textdet_test --batch; flags 2: det+refine,
        its refined horizontal lines only)."""
        import textdet as TD
        return [([ints(ln) for ln in lines], int(bool(tr)))
                for lines, tr in TD.c_detect(self.dir / "vocr_textdet_test", blob, list(greys), flags)]

    def columns(self, grey):
        """vocr_find_columns with VOCR_COLUMNS_VALLEYS: (boxes, dark, truncated) (runtime/vocr_vertical_test)."""
        with tempfile.TemporaryDirectory(prefix="textspot-col-") as td:
            p = Path(td) / "f.pgm"
            g = np.ascontiguousarray(grey, dtype=np.uint8)
            p.write_bytes(b"P5\n%d %d\n255\n" % (g.shape[1], g.shape[0]) + g.tobytes())
            lines = _run([self.dir / "vocr_vertical_test", p, "--flags", "1"])
        boxes, dark, tr = [], [], None
        for ln in lines:
            v = ln.split()
            if v[0] == "truncated":
                tr = int(v[1])
            else:
                boxes.append(tuple(int(t) for t in v[:4]))
                dark.append(int(v[4]))
        if tr is None:
            raise RuntimeError("vocr_vertical_test: no truncated line")
        return boxes, dark, tr

    def spot(self, greys, blob):
        """vocr_textspot_find per grey frame: [(boxes (x, y, w, h, vertical, conf, dark, block), blocks (x, y, w, h,
        vertical, first, count), pick, truncated)], or None while runtime/vocr_textspot_test does not exist.
        Batch records struct.pack("<4I", width, height, 1, 0) + pixels; one output line per frame: rc, count,
        block_count, pick, truncated, then 8 integers per box and 7 per block."""
        if not self.has("vocr_textspot_test"):
            return None
        with tempfile.NamedTemporaryFile(suffix=".vtsp", delete=False) as f:
            f.write(blob)
            bpath = f.name
        try:
            recs = [struct.pack("<4I", g.shape[1], g.shape[0], 1, 0) + g.tobytes() for g in greys]
            lines = self._batch("vocr_textspot_test", ["--batch", bpath], recs)
        finally:
            os.unlink(bpath)
        out = []
        for line in lines:
            v = [int(t) for t in line.split()]
            if v[0]:
                raise RuntimeError(f"vocr_textspot_find rc={v[0]}")
            n, nb, pick, tr = v[1:5]
            bx = [tuple(v[5 + 8 * i:13 + 8 * i]) for i in range(n)]
            o = 5 + 8 * n
            bk = [tuple(v[o + 7 * i:o + 7 + 7 * i]) for i in range(nb)]
            out.append((bx, bk, pick, tr))
        if len(out) != len(greys):
            raise RuntimeError("vocr_textspot_test: one result per frame expected")
        return out


def c_components(cd, frames, comps, nets_blobs, gate):
    """{component key: [equal per frame (bool, or None when no C driver)]}: every twin component the arms used,
    reproduced by its C driver."""
    keys = sorted({k for c in comps for k in c})
    greys = [f.grey for f in frames]
    eq = {}

    def col(k):
        return [c.get(k) for c in comps]

    with tempfile.TemporaryDirectory(prefix="textspot-c-") as td:
        blobs = {}
        for name, data in nets_blobs.items():
            p = Path(td) / f"{name.replace(':', '_')}.bin"
            p.write_bytes(data)
            blobs[name] = p
        for k in keys:
            tw = col(k)
            idx = [i for i, v in enumerate(tw) if v is not None]
            sub = [greys[i] for i in idx]
            res = [None] * len(frames)
            if k == "grad" and cd.has("vocr_lines_test"):
                for i, c in zip(idx, cd.grad(sub)):
                    res[i] = (list(c[0]), list(c[1]), int(c[2])) == (tw[i][0], tw[i][1], tw[i][2])
            elif k == "refine" and cd.has("vocr_lines_refine_test"):
                for i, c in zip(idx, cd.refine(sub)):
                    res[i] = (list(c[0]), list(c[1])) == (tw[i][0], tw[i][1])
            elif k == "gate" and cd.has("vocr_textgate_test"):
                for i, c in zip(idx, cd.gated(sub, blobs["gate"], *gate)):
                    gb, gd, gt = comps[i]["grad"] if tw[i][1] else ([], [], 0)
                    res[i] = (list(c[0]), list(c[1]), int(c[2])) == (gb, gd, gt)
            elif k == "textdet" and cd.has("vocr_textdet_test"):
                for i, c in zip(idx, cd.textdet(sub, blobs["s8c24"], 0)):
                    res[i] = c == (tw[i][0], tw[i][1])
            elif k == "textdet_refine" and cd.has("vocr_textdet_test"):
                for i, c in zip(idx, cd.textdet(sub, blobs["s8c24"], 2)):
                    res[i] = c == ([ln for ln in tw[i] if not ln[5]], comps[i]["textdet"][1])
            elif k == "columns" and cd.has("vocr_vertical_test"):
                for i in idx:
                    c = cd.columns(greys[i])
                    res[i] = (list(c[0]), list(c[1]), int(c[2])) == (tw[i][0], tw[i][1], tw[i][2])
            elif k.startswith("spot:") or k == "s8c24_spot":
                if k.endswith(":refine"):
                    if cd.crefine is not None:
                        for i in idx:
                            out, _, hb, hd = tw[i]
                            cb, cdk = cd.crefine.refine(greys[i], hb, hd)
                            res[i] = [(*b, dk) for b, dk in zip(cb, cdk)] == [(*b[:4], b[6]) for b in out if not b[4]]
                else:
                    got = cd.spot(sub, nets_blobs[k])
                    if got is not None:
                        for i, c in zip(idx, got):
                            res[i] = (list(c[0]), list(c[1]), int(c[2]), int(c[3])) == tuple(tw[i])
            eq[k] = res
    return eq


def arm_components(spec):
    k = spec.base if spec.kind == "baseline" else spec.kind
    return {"grad": ["grad"], "grad_refine": ["refine"], "grad_refine_gate": ["gate", "refine"],
            "s8c24_textdet": ["textdet"], "s8c24_textdet_refine": ["textdet", "textdet_refine"],
            "s8c24_spot": ["s8c24_spot"], "columns": ["columns"], "grad_refine_columns": ["refine", "columns"],
            "det": [spec.net_key], "detref": [spec.net_key, (spec.net_key or "") + ":refine"]}[k]


def arm_c_equal(spec, i, eq, comp):
    """c_equal of one arm on frame i: every component it used equals its C driver (None if one was not checked).
    The gate arm's refinement only counts where the gate passed."""
    vals = []
    for k in arm_components(spec):
        if k == "refine" and spec.base == "grad_refine_gate" and not comp["gate"][1]:
            continue
        if k not in eq:
            return None
        vals.append(eq[k][i])
    if any(v is False for v in vals):
        return False
    return None if any(v is None for v in vals) else True


# ----------------------------------------------------------------------------- reading

class Cache:
    """Readings keyed by (frame sha256, box, dark, vertical, recognizer sha256). Every process appends to its own
    shard file in the directory; all shards are read at start (a torn last line of a killed run is skipped)."""

    def __init__(self, directory):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.data = {}
        for p in sorted(self.dir.glob("*.jsonl")):
            with open(p, encoding="utf-8") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    self.data[r["k"]] = r["t"]
        self.path = self.dir / f"{datetime.datetime.now(datetime.timezone.utc):%Y%m%dT%H%M%S}-{os.getpid()}.jsonl"
        self.lock = threading.Lock()
        self.fh = None
        self.added = 0

    @staticmethod
    def key(frame_sha, box, dark, vertical, rec_sha):
        x, y, w, h = (int(v) for v in box[:4])
        return f"{frame_sha}|{x},{y},{w},{h}|{int(dark)}|{int(bool(vertical))}|{rec_sha}"

    def get(self, k):
        return self.data.get(k)

    def put(self, k, text):
        with self.lock:
            self.data[k] = text
            if self.fh is None:
                self.fh = open(self.path, "a", encoding="utf-8")
            self.fh.write(json.dumps({"k": k, "t": text}, ensure_ascii=False) + "\n")
            self.fh.flush()
            self.added += 1

    def close(self):
        with self.lock:
            if self.fh is not None:
                self.fh.close()
                self.fh = None


class Readers:
    """The recognizers, one set per worker thread (FT.Reader: the C runtime in W8I from a library built from this
    checkout). factory(path) -> an object with read(grey, invert) -> text (injected in tests)."""

    def __init__(self, paths, factory):
        self.paths = {m: Path(p) for m, p in paths.items()}
        self.sha = {m: sha256(p) for m, p in self.paths.items()}
        self.factory = factory
        self.local = threading.local()

    def get(self, model):
        d = self.local.__dict__.setdefault("r", {})
        if model not in d:
            d[model] = self.factory(self.paths[model])
        return d[model]


def c_readers(paths, build_dir, compiler="gcc"):
    import finder_tighten as FT
    ident = FT.build(Path(build_dir), compiler)
    rd = Readers(paths, lambda p: FT.Reader(p, ident["library"]))
    rd.identity = {k: ident[k] for k in ("library_sha256", "compiler", "scalar_flags")}
    return rd


def crop_of(grey, box, vertical):
    import vertical_twin as V
    x, y, w, h = (int(v) for v in box[:4])
    c = np.ascontiguousarray(grey[y:y + h, x:x + w])
    return V.rotate_ccw(c) if vertical else c


def read_all(frames, arms_per_frame, models, readers, cache, workers=5, log=print):
    """Reads every box of every arm that the cache does not hold: horizontal boxes by each of `models`, vertical
    ones by VOCR5-V. arms_per_frame: [{arm: arm result}] per frame."""
    jobs, seen = [], set()
    for f, arms in zip(frames, arms_per_frame):
        for a in arms.values():
            for b in a["boxes"]:
                for m in ([V_MODEL] if b[4] else models):
                    k = Cache.key(f.sha, b[:4], b[6], b[4], readers.sha[m])
                    if k in seen or cache.get(k) is not None:
                        continue
                    seen.add(k)
                    jobs.append((k, f, b, m))
    if not jobs:
        return 0
    lock, done = threading.Lock(), [0]

    def work(job):
        k, f, b, m = job
        text = readers.get(m).read(crop_of(f.grey, b, b[4]), bool(b[6]))
        cache.put(k, text)
        with lock:
            done[0] += 1
            if done[0] % 2000 == 0:
                log(f"read {done[0]}/{len(jobs)}")

    with ThreadPoolExecutor(max(1, workers)) as ex:
        list(ex.map(work, jobs))
    return len(jobs)


def texts_of(frame, boxes, model, readers, cache):
    out = []
    for b in boxes:
        m = V_MODEL if b[4] else model
        t = cache.get(Cache.key(frame.sha, b[:4], b[6], b[4], readers.sha[m]))
        if t is None:
            raise KeyError(f"{frame.id}: box {b[:4]} not read by {m}")
        out.append(t)
    return out


# ----------------------------------------------------------------------------- scoring

def ref_order(r):
    x, y, w, h = r["box"]
    return (1, -(x + w), y) if r.get("vertical") else (0, y, x)


def pick_measures(frame, arm, texts):
    """The pick (report only): on a frame with dialogue (classes dialogue and choice), the picked block's box
    against the union of the padded dialogue references (IoU >= 0.5), and the CER of the block's readings joined in
    its order against the dialogue references joined in reading order (top to bottom; columns right to left)."""
    dl = [r for r in frame.refs if not r.get("ignore") and r.get("cls") in M.DIALOGUE]
    if not dl:
        return {"dialogue": False}
    pads = [M.pad_box(r["box"], bool(r.get("vertical")), frame.width, frame.height) for r in dl]
    x0, y0 = min(p[0] for p in pads), min(p[1] for p in pads)
    x1, y1 = max(p[0] + p[2] for p in pads), max(p[1] + p[3] for p in pads)
    ref = "".join(norm(r["text"]) for r in sorted(dl, key=ref_order))
    out = {"dialogue": True, "chars": len(ref)}
    if arm["pick"] < 0:
        return dict(out, iou=0.0, hit=False, errors=len(ref) if texts is not None else None)
    bk = arm["blocks"][arm["pick"]]
    v = M.iou(list(bk[:4]), [x0, y0, x1 - x0, y1 - y0])
    out.update(iou=round(v, 6), hit=bool(v >= M.IOU_MIN))
    if texts is not None:
        out["errors"] = lev(ref, norm("".join(texts[bk[5]:bk[5] + bk[6]])))
    else:
        out["errors"] = None
    return out


def frame_unit(frame, arm, texts, c_equal=None):
    """One frame's unit: detection counts, the end-to-end errors (None without texts), junk, the pick."""
    dets = [{"box": list(b[:4]), "vertical": bool(b[4]), "text": (texts[i] if texts is not None else "")}
            for i, b in enumerate(arm["boxes"])]
    m = M.match_frame(dets, frame.refs, frame.width, frame.height)
    u = {"unit": frame.id, "vn": frame.vn, "part": frame.part, "text_free": frame.text_free,
         "boxes": len(dets), "vertical": sum(1 for d in dets if d["vertical"]), "truncated": int(arm["truncated"]),
         "left_out": len(m["left_out"]), "tp": m["tp"], "fp": m["fp"], "fn": m["fn"], "tp_cls": m["tp_cls"],
         "fn_cls": m["fn_cls"], "pairs": [[p[2], p[3]] for p in m["pairs"]],
         "e2e": None, "junk": None, "pick": pick_measures(frame, arm, texts), "c_equal": c_equal,
         "h_in": arm.get("h_in")}
    if texts is not None:
        e = M.e2e_frame(dets, frame.refs, frame.width, frame.height)
        u["e2e"] = {"all": list(e["all"]), "dialogue": list(e["dialogue"])}
        u["junk"] = e["junk"]
    return u


def rule_arm(units):
    """textspot_metrics.arm_summary of a list of units (the rule's comparisons read this)."""
    frames = [{"unit": u["unit"], "vn": u["vn"], "det": {"tp": u["tp"], "fp": u["fp"], "fn": u["fn"]},
               "e2e": {"all": tuple(u["e2e"]["all"]), "dialogue": tuple(u["e2e"]["dialogue"])}} for u in units]
    return M.arm_summary(frames)


def _share(a, b):
    return round(a / b, 6) if b else None


def summarize(units, parts=True):
    """The summaries of an arm's units: detection (pooled P / R / F1), recall per class, tightness, CER all and
    dialogue, junk, text-free false alarms, the pick, C parity; per part."""
    out = {"frames": len(units), "vns": len({u["vn"] for u in units}), "boxes": sum(u["boxes"] for u in units),
           "vertical_boxes": sum(u["vertical"] for u in units),
           "truncated_frames": sum(1 for u in units if u["truncated"]), "detection": M.pooled(units)}
    cls = {}
    for c in M.CLASSES:
        tp = sum(u["tp_cls"][c] for u in units)
        fn = sum(u["fn_cls"][c] for u in units)
        cls[c] = {"refs": tp + fn, "tp": tp, "recall": _share(tp, tp + fn)}
    out["recall_by_class"] = cls
    out["tightness"] = M.tightness([{"pairs": [(0, 0, p[0], p[1]) for p in u["pairs"]]} for u in units])
    if units and all(u["e2e"] is not None for u in units):
        out["cer_all"] = M.cer_of([{"errors": u["e2e"]["all"][0], "chars": u["e2e"]["all"][1]} for u in units])
        out["cer_dialogue"] = M.cer_of([{"errors": u["e2e"]["dialogue"][0], "chars": u["e2e"]["dialogue"][1]}
                                        for u in units])
        out["junk_chars"] = sum(u["junk"] for u in units)
    tf = [u for u in units if u["text_free"]]
    out["text_free"] = {"frames": len(tf), "with_any_box": sum(1 for u in tf if u["boxes"]),
                        "share_with_any_box": _share(sum(1 for u in tf if u["boxes"]), len(tf)),
                        "boxes_per_frame": _share(sum(u["boxes"] for u in tf), len(tf)),
                        "with_scored_box": sum(1 for u in tf if u["fp"]),
                        "share_with_scored_box": _share(sum(1 for u in tf if u["fp"]), len(tf)),
                        "scored_boxes_per_frame": _share(sum(u["fp"] for u in tf), len(tf))}
    pk = [u["pick"] for u in units if u["pick"]["dialogue"]]
    pick = {"frames_with_dialogue": len(pk), "hits": sum(1 for p in pk if p["hit"]),
            "hit_share": _share(sum(1 for p in pk if p["hit"]), len(pk))}
    if pk and all(p.get("errors") is not None for p in pk):
        pick["cer"] = M.cer_of([{"errors": p["errors"], "chars": p["chars"]} for p in pk])
    out["pick"] = pick
    ce = [u["c_equal"] for u in units]
    out["c_equal"] = None if not ce or any(v is None for v in ce) else all(ce)
    out["c_unequal_frames"] = sum(1 for v in ce if v is False)
    out["c_unchecked_frames"] = sum(1 for v in ce if v is None)
    if parts:
        names = {u["part"] for u in units}
        groups = {p: [u for u in units if u["part"] == p] for p in sorted(names)}
        if tf and not names & {"text-free", "text_free"}:
            groups["text_free"] = tf          # text-free frames filed under other parts: also a row of their own
        out["per_part"] = {p: summarize(us, False) for p, us in groups.items()}
    return out


# ----------------------------------------------------------------------------- run

def load_set(args, proto):
    """(rows, manifest record) of the set asked for. dev: eval/manifests/textspot-dev.jsonl, which an amendment must
    pin; pseudo: --manifest of training-side frames only; the test set is never read here."""
    from common import is_reserved_vn, test_set_vns
    if args.set == "test":
        raise SystemExit("the test set is protected: only eval/textspot_protected.py reads it, once")
    path = Path(args.manifest) if args.manifest else MANIFESTS.get(args.set)
    if path is None:
        raise SystemExit("--manifest is required for this set")
    if path.resolve() == MANIFESTS["test"].resolve():
        raise SystemExit("the test manifest is protected")
    digest = sha256(path)
    rows = jsonl(path)
    test_vns = set(test_set_vns())
    bad = sorted({r["vn"] for r in rows if r["vn"] in test_vns})
    if bad:
        raise SystemExit(f"{path}: rows of protected test VNs {bad[:5]}")
    if args.set == "dev":
        pin = pinned_sha(proto, "eval/manifests/textspot-dev.jsonl")
        if pin is None:
            raise SystemExit("the dev manifest is not pinned by a protocol amendment yet: no arm reads it before")
        if pin != digest:
            raise SystemExit(f"{path}: sha256 {digest} is not the pinned {pin}")
    elif args.set == "pseudo":
        bad = sorted({r["vn"] for r in rows if is_reserved_vn(r["vn"], dev=True)})
        if bad:
            raise SystemExit(f"pseudo frames must be training-side: reserved VNs {bad[:5]}")
    if args.limit:
        rows = rows[:args.limit]
    return rows, {"set": args.set, "path": rel(path), "sha256": digest, "frames": len(rows),
                  "vns": len({r["vn"] for r in rows})}


def reusable(path, spec_meta, manifest, need_c):
    if not path.exists():
        return None
    doc = json.loads(path.read_text(encoding="utf-8"))
    same = (doc.get("spec_meta") == spec_meta and doc.get("manifest", {}).get("sha256") == manifest["sha256"]
            and doc.get("twin_sha256") == twin_identity())
    if not same or (need_c and not doc.get("c_checked")):
        return None
    return doc


def twin_identity():
    return code_identity(("synth/textspot_twin.py", "synth/textdet_twin.py", "synth/common.py", "synth/lines_refine.py",
                          "synth/vertical_twin.py", "eval/textgate.py", "eval/textdet.py", "eval/textspot.py"))


def run_arms(frames, specs, manifest, out, models, readers=None, cache=None, cdir=None, workers=1, force=False,
             log=print, proto=None):
    """Detection (the twins, and the C drivers when cdir is given) and reading of every arm on every frame. Writes
    OUT/<arm>/boxes.json (reused when it matches); returns [{arm: arm result}] per frame."""
    proto = proto or protocol()
    gate = gate_params(proto)
    out = Path(out)
    per_frame = [{} for _ in frames]
    todo = []
    for s in specs:
        doc = None if force else reusable(out / s.name / "boxes.json", s.meta(), manifest, cdir is not None)
        if doc is not None and set(doc["frames"]) >= {f.id for f in frames}:
            for i, f in enumerate(frames):
                fr = doc["frames"][f.id]
                per_frame[i][s.name] = dict(fr, boxes=[tuple(b) for b in fr["boxes"]],
                                            blocks=[tuple(b) for b in fr["blocks"]])
            log(f"{s.name}: boxes reused")
        else:
            todo.append(s)
    if todo:
        nets = load_nets(todo)
        res = detect(frames, todo, nets, gate, workers)
        eq = {}
        cdr = None
        if cdir is not None:
            cdr = CDrivers(cdir)
            blobs = {s.net_key: s.blob for s in todo if s.blob is not None}
            if any(s.base in ("s8c24_textdet", "s8c24_textdet_refine") for s in todo):
                blobs["s8c24"] = S8C24.read_bytes()
            if any(s.base == "s8c24_spot" for s in todo):
                import textspot_twin as S
                blobs["s8c24_spot"] = S.pack(S8C24.read_bytes(), 0, 1)
            if any(s.base == "grad_refine_gate" for s in todo):
                blobs["gate"] = GATE.read_bytes()
            eq = c_components(cdr, frames, [c for _, c in res], blobs, gate)
            for k, v in sorted(eq.items()):
                name = k if not k.startswith("spot:") else "spot:" + k[5:17] + k[69:]
                log(f"C component {name}: equal {sum(1 for x in v if x is True)}, differ "
                    f"{sum(1 for x in v if x is False)}, unchecked {sum(1 for x in v if x is None)}")
        for s in todo:
            fr = {}
            for i, (f, (arms, comp)) in enumerate(zip(frames, res)):
                a = dict(arms[s.name])
                a["c_equal"] = arm_c_equal(s, i, eq, comp) if cdir is not None else None
                per_frame[i][s.name] = a
                fr[f.id] = {k: (list(map(list, v)) if k in ("boxes", "blocks") else v) for k, v in a.items()}
                fr[f.id]["part"] = f.part
            doc = {"kind": "textspot-boxes", "arm": s.name, "spec_meta": s.meta(), "manifest": manifest,
                   "twin_sha256": twin_identity(), "c_checked": cdir is not None,
                   "c_build": cdr.record if cdr is not None else None, "computed": utc_now(), "frames": fr}
            write_json(out / s.name / "boxes.json", doc)
            ce = [a["c_equal"] for a in fr.values()]
            log(f"{s.name}: {sum(len(a['boxes']) for a in fr.values())} boxes on {len(fr)} frames; C == twin: "
                f"{'n/a' if cdir is None else ('unchecked' if any(v is None for v in ce) else all(ce))}")
    if models:
        n = read_all(frames, per_frame, models, readers, cache, workers, log)
        log(f"read {n} new crops (cache {len(cache.data)})")
    return per_frame


def score_arms(frames, specs, per_frame, manifest, out, models, readers=None, cache=None, log=print):
    """Every arm's units per recognizer (units_detection.json without models) and their summaries:
    OUT/<arm>/units_<model>.json. Returns {arm: {model: summary}}."""
    out = Path(out)
    summaries = {}
    for s in specs:
        bpath = out / s.name / "boxes.json"
        for m in (models or [None]):
            units = []
            for f, arms in zip(frames, per_frame):
                a = arms[s.name]
                texts = texts_of(f, a["boxes"], m, readers, cache) if m else None
                units.append(frame_unit(f, a, texts, a.get("c_equal")))
            summ = summarize(units)
            meta = {"kind": "textspot-units", "arm": s.name, "spec_meta": s.meta(), "model": m,
                    "vertical_model": V_MODEL if m else None, "manifest": manifest,
                    "recognizers_sha256": {k: readers.sha[k] for k in (m, V_MODEL)} if m else None,
                    "reader": getattr(readers, "identity", None) if m else None,
                    "boxes_sha256": sha256(bpath), "code_sha256": code_identity(), "computed": utc_now()}
            write_json(out / s.name / f"units_{m or 'detection'}.json", {"meta": meta, "units": units,
                                                                       "summary": summ})
            summaries.setdefault(s.name, {})[m or "detection"] = summ
            d = summ["detection"]
            cer = summ.get("cer_all", {}).get("cer")
            dlg = summ.get("cer_dialogue", {}).get("cer")
            log(f"{s.name} {m or 'detection'}: P {fmt(d['precision'])} R {fmt(d['recall'])} F1 {fmt(d['f1'])} "
                f"CER all {cer} dialogue {dlg} text-free FA {summ['text_free']['share_with_any_box']} "
                f"pick {summ['pick']['hit_share']} C {summ['c_equal']}")
    return summaries


def evaluate(frames, specs, manifest, out, models, readers=None, cache=None, cdir=None, workers=1, force=False,
             log=print, proto=None):
    """run_arms then score_arms: returns {arm: {model: summary}}."""
    per_frame = run_arms(frames, specs, manifest, out, models, readers, cache, cdir, workers, force, log, proto)
    return score_arms(frames, specs, per_frame, manifest, out, models, readers, cache, log)


def fmt(v, nd=4):
    return "–" if v is None else f"{v:.{nd}f}"


def parse_models(text):
    if text in ("", "none"):
        return []
    out = []
    for t in text.split(","):
        m = SHORT.get(t.strip(), t.strip())
        if m not in H_MODELS:
            raise SystemExit(f"unknown recognizer {t!r}")
        out.append(m)
    return out


def cmd_run(args):
    proto = protocol()
    check_baseline_blobs(proto)
    rows, manifest = load_set(args, proto)
    dirs = [Path(d) for d in args.shots] + list(SHOT_DIRS)
    frames = load_frames(rows, dirs)
    specs = parse_arms(args.arms)
    models = parse_models(args.models)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cdir = None
    if args.cbin:
        cdir = out / "cbuild" if args.cbin == "auto" else Path(args.cbin)
        if args.cbin == "auto":
            cbuild(cdir, args.compiler)
    readers = cache = None
    if models:
        readers = c_readers(recognizer_paths(models, proto), out / f"rbuild-{os.getpid()}", args.compiler)
        cache = Cache(args.cache or CACHES.get(args.set, out / "cache"))
    print(f"{len(frames)} frames ({manifest['vns']} VNs), {len(specs)} arms, models {models or 'none'}", flush=True)
    try:
        evaluate(frames, specs, manifest, out, models, readers, cache, cdir, args.workers, args.force,
                 log=lambda s: print(s, flush=True), proto=proto)
    finally:
        if cache is not None:
            cache.close()


# ----------------------------------------------------------------------------- eligibility (the runtime's table)

def a9_rows(doc):
    """The runtime's A9 / arena table (eval/textspot_a9_<date>.json), normalised: [{"shape", "half_above",
    "size": (w, h), "ms_333", "arena", "macs", "vtsp_bytes"}]. Its "summary" holds the rule's columns (its
    "rule_reads"): ms_333_960 (the cycle model's high bound x 1.0007 at 333 MHz, the default call with the dark pass)
    and arena_up_to_960 (vocr_textspot_arena_bytes(m, 960, 544)); those rows come first. The "table" rows (per size:
    ms_per_frame["333"] or {"high", "calibrated"} or the cycles' high bound, and arena_bytes) are read too, for
    tables without a summary and for other sizes."""
    out = []
    for r in doc.get("summary") or []:
        ha = r.get("half_above", {"h2": 600, "f1": 0}.get(r.get("scale")))
        out.append({"shape": r["shape"], "half_above": ha, "size": (960, 544), "ms_333": r["ms_333_960"],
                    "arena": r["arena_up_to_960"], "macs": r.get("macs_960"), "vtsp_bytes": r.get("vtsp_bytes"),
                    "from": "summary", "row": r})
    for r in doc.get("table") or doc.get("rows") or []:
        ha = r.get("half_above")
        if ha is None and "scale" in r:
            ha = {"h2": 600, "f1": 0}[r["scale"]]
        size = tuple(r.get("size") or r.get("frame") or ())
        ms = (r.get("ms_per_frame") or r.get("ms_per_region") or {}).get("333")
        if isinstance(ms, dict):
            ms = ms.get("calibrated", ms["high"] * A9_CAL if ms.get("high") is not None else None)
        if ms is None and (r.get("cycles_per_frame") or {}).get("333"):
            ms = r["cycles_per_frame"]["333"][1] * A9_CAL / 333e3
        out.append({"shape": r["shape"], "half_above": ha, "size": size, "ms_333": ms,
                    "arena": r.get("arena_bytes", r.get("arena")), "macs": r.get("macs"),
                    "vtsp_bytes": r.get("blob_bytes"), "from": "table", "row": r})
    return out


def a9_lookup(rows, shape, half_above, size=(960, 544)):
    hit = [r for r in rows if r["shape"] == shape and r["half_above"] == half_above and tuple(r["size"]) == size]
    return hit[0] if hit else None


def refine_arena(doc, cdir=None):
    """(vocr_refine_arena_bytes(960, 544), its source): the table's record when it has one, else the refinement's
    driver (runtime/vocr_lines_refine_test --arena 960 544), else (None, None)."""
    for k in ("refine_arena_960x544", "refine_arena_bytes_960x544"):
        if k in doc:
            return int(doc[k]), f"a9:{k}"
    if isinstance(doc.get("refine"), dict) and "arena_960x544" in doc["refine"]:
        return int(doc["refine"]["arena_960x544"]), "a9:refine.arena_960x544"
    drv = Path(cdir) / "vocr_lines_refine_test" if cdir is not None else None
    if drv is not None and drv.exists():
        return (int(_run([drv, "--arena", 960, 544])[0].split()[0]),
                f"vocr_lines_refine_test --arena 960 544 (sha256 {sha256(drv)})")
    return None, None


def input_size(half_above, width=960, height=544):
    return (-(-width // 2), -(-height // 2)) if half_above and width > half_above else (width, height)


def macs_960(shape, half_above):
    import textdet_model as TM
    w, h = input_size(half_above)
    return int(TM.cost(TM.SHAPES[shape], w, h)["macs"])


def apply_parity(boxes_doc, parity):
    """boxes_doc with each frame's c_equal from a parity record (textspot.py parity) where the run did not check
    it; the record's boxes_sha256 must be the boxes'."""
    rec = (parity or {}).get("arms", {}).get(boxes_doc["arm"])
    if rec is None:
        return boxes_doc
    if rec["boxes_sha256"] != boxes_doc.get("_sha256"):
        raise SystemExit(f"{boxes_doc['arm']}: the parity record is of other boxes")
    frames = {k: dict(v, c_equal=v["c_equal"] if v.get("c_equal") is not None else rec["frames"].get(k))
              for k, v in boxes_doc["frames"].items()}
    return dict(boxes_doc, frames=frames)


def eligibility(meta, boxes_doc, rows, ref_arena):
    """The rule's eligibility inputs of one candidate arm: VTSP bytes, MACs per 960x544 frame (at the network's input
    size), the arena and the A9 high bound from the runtime's table, plus REFINE_MS_PER_BOX per horizontal box at the
    arm's median on the dev Vita frames for det+refine, and c_equal over every frame."""
    r = a9_lookup(rows, meta["shape"], meta["half_above"])
    macs = macs_960(meta["shape"], meta["half_above"])
    if r is not None and r.get("macs") is not None and int(r["macs"]) != macs:
        raise SystemExit(f"{boxes_doc['arm']}: the A9 table's MACs {r['macs']} are not train/textdet_model.cost's "
                         f"{macs} for {meta['shape']} at half_above {meta['half_above']}")
    if r is not None and r.get("vtsp_bytes") is not None and int(r["vtsp_bytes"]) != meta["blob_bytes"]:
        raise SystemExit(f"{boxes_doc['arm']}: VTSP bytes {meta['blob_bytes']} differ from the table's "
                         f"{r['vtsp_bytes']} for {meta['shape']}")
    fr = boxes_doc["frames"].values()
    ce = [f.get("c_equal") for f in fr]
    out = {"name": boxes_doc["arm"], "decode": meta["decode"], "family": meta.get("family"), "run": meta.get("run"),
           "shape": meta["shape"], "half_above": meta["half_above"], "blob_bytes": meta["blob_bytes"],
           "blob_sha256": meta["blob_sha256"], "macs_960": macs,
           "a9_row": r["from"] if r is not None else None, "arena_spot_960": r["arena"] if r else None,
           "ms_333_960_spot": round(r["ms_333"], 3) if r and r["ms_333"] is not None else None,
           "c_equal": bool(ce) and all(v is True for v in ce), "c_unchecked_frames": sum(1 for v in ce if v is None),
           "c_unequal_frames": sum(1 for v in ce if v is False)}
    if meta["decode"] == "det+refine":
        vita = [f["h_in"] for f in boxes_doc["frames"].values() if f.get("part") == "vita"]
        med = statistics.median(vita) if vita else None
        out["median_h_boxes_vita"] = med
        out["refine_ms"] = round(REFINE_MS_PER_BOX * med, 3) if med is not None else None
        out["arena_refine_960"] = ref_arena
        out["arena_960"] = (max(out["arena_spot_960"], ref_arena) if out["arena_spot_960"] is not None and
                            ref_arena is not None else None)
        out["est_ms_333_960"] = (round(out["ms_333_960_spot"] + out["refine_ms"], 3) if
                                 out["ms_333_960_spot"] is not None and med is not None else None)
    else:
        out["arena_960"] = out["arena_spot_960"]
        out["est_ms_333_960"] = out["ms_333_960_spot"]
    out["inputs_known"] = out["arena_960"] is not None and out["est_ms_333_960"] is not None
    return out


def rule_candidate(e, arm):
    """apply_rule's candidate record: an unknown cost (no table row) is never eligible."""
    return dict(e, arena_960=e["arena_960"] if e["arena_960"] is not None else 1 << 62,
                est_ms_333_960=e["est_ms_333_960"] if e["est_ms_333_960"] is not None else float("inf"), arm=arm)


def load_candidate_dirs(dirs):
    """{arm: (boxes doc, units doc of VOCR5-N or None)} of every candidate arm under the unit directories."""
    out = {}
    for d in dirs:
        for b in sorted(Path(d).glob("*/boxes.json")):
            doc = json.loads(b.read_text(encoding="utf-8"))
            if doc["spec_meta"]["kind"] not in ("det", "detref"):
                continue
            doc["_sha256"] = sha256(b)
            u = b.parent / f"units_{RULE_MODEL}.json"
            units = json.loads(u.read_text(encoding="utf-8")) if u.exists() else None
            det = b.parent / "units_detection.json"
            doc["_detection"] = (json.loads(det.read_text(encoding="utf-8"))["summary"] if det.exists() else None)
            out[doc["arm"]] = (doc, units, b.parent)
    return out


def cmd_eligible(args):
    proto = protocol()
    rule = proto["rule"]
    rows = a9_rows(json.loads(Path(args.a9).read_text(encoding="utf-8")))
    ref_ar, ref_src = refine_arena(json.loads(Path(args.a9).read_text(encoding="utf-8")), args.cbin or None)
    print(f"refinement arena at 960x544: {ref_ar} ({ref_src})")
    cands = load_candidate_dirs(args.units)
    parity = merge_parity(args.parity)
    names = []
    for arm, (doc, _, _) in sorted(cands.items()):
        e = eligibility(doc["spec_meta"], apply_parity(doc, parity), rows, ref_ar)
        ok = e["inputs_known"] and e["arena_960"] <= rule["arena_max"] and e["est_ms_333_960"] <= rule["ms_max_333_960"]
        why = "" if ok or e["inputs_known"] else (" (no table row)" if not e["a9_row"] else " (no Vita frame)")
        c = "unchecked" if e["c_unchecked_frames"] else e["c_equal"]
        print(f"{arm:40s} {e['blob_bytes']:6d} B  {e['macs_960'] / 1e6:6.1f} M MACs  arena {e['arena_960']}  "
              f"est {e['est_ms_333_960']} ms  C {c}  {'by cost: ELIGIBLE' if ok else 'not eligible'}{why}")
        if ok:
            names.append(f"{doc['spec_meta']['spec']}:{doc['spec_meta']['kind']}")
    print("eligible specs:", ",".join(sorted(set(names))))
    if args.out:
        Path(args.out).write_text(",".join(sorted(set(names))) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------- baseline

def load_units(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def cmd_baseline(args):
    d = Path(args.units)
    arms, man, ident = {}, None, None
    for arm in BASELINES:
        for m in H_MODELS:
            f = d / arm / f"units_{m}.json"
            if not f.exists():
                if m == RULE_MODEL:
                    raise SystemExit(f"{f} missing: every baseline is read by {RULE_MODEL}")
                continue
            doc = load_units(f)
            if man is None:
                man = doc["meta"]["manifest"]
            elif doc["meta"]["manifest"]["sha256"] != man["sha256"]:
                raise SystemExit(f"{f}: another manifest")
            ident = ident or doc["meta"].get("reader")
            arms.setdefault(arm, {})[m] = {"units": rel(f), "units_sha256": sha256(f),
                                           "boxes_sha256": doc["meta"]["boxes_sha256"],
                                           "recognizers_sha256": doc["meta"]["recognizers_sha256"],
                                           "summary": doc["summary"]}
    c_all = {a: v[RULE_MODEL]["summary"]["c_equal"] for a, v in arms.items()}
    doc = {"kind": "textspot-baseline", "computed": utc_now(), "protocol_sha256": sha256(PROTOCOL),
           "protocol_commit": git_head(), "manifest": man, "units_dir": rel(d), "code_sha256": code_identity(),
           "reader": ident, "c_equal": c_all,
           "note": "the frozen baseline (docs/textspot.md): every baseline arm on the dev frames, read by VOCR5-N "
                   "(the rule's) and VOCR5-S / VOCR5-L (report); vertical boxes by VOCR5-V. Counts, rates and "
                   "hashes only; the units stay under $T",
           "arms": arms}
    write_json(args.out, doc, indent=1)
    for arm in BASELINES:
        s = arms[arm][RULE_MODEL]["summary"]
        print(f"{arm:22s} F1 {fmt(s['detection']['f1'])}  CER all {s['cer_all']['cer']}  dialogue "
              f"{s['cer_dialogue']['cer']}  text-free FA {s['text_free']['share_with_any_box']}  "
              f"pick {s['pick']['hit_share']}  C {s['c_equal']}")


def git_head():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


# ----------------------------------------------------------------------------- select

def baseline_rule_arms(base_doc, units_dir=None):
    """{baseline: rule_arm(VOCR5-N units)} from the frozen baseline's units, each checked against its sha256."""
    out = {}
    for arm in BASELINES:
        e = base_doc["arms"][arm][RULE_MODEL]
        f = Path(units_dir) / arm / f"units_{RULE_MODEL}.json" if units_dir else resolve(e["units"])
        if sha256(f) != e["units_sha256"]:
            raise SystemExit(f"{f}: not the frozen baseline's units")
        out[arm] = rule_arm(load_units(f)["units"])
    return out


def resolve(path):
    s = str(path)
    if s.startswith("$HERD/"):
        return HERD / s[len("$HERD/"):]
    p = Path(s)
    return p if p.is_absolute() else ROOT / p


def select(base_doc, cand_dirs, a9_doc, rule, units_dir=None, cdir=None, parity=None):
    baselines = baseline_rule_arms(base_doc, units_dir)
    rows = a9_rows(a9_doc)
    ref_ar, ref_src = refine_arena(a9_doc, cdir)
    cands, report = [], {"_refine_arena": {"bytes_960x544": ref_ar, "source": ref_src}}
    for arm, (doc, units, _) in sorted(load_candidate_dirs(cand_dirs).items()):
        if units is not None and units["meta"]["manifest"]["sha256"] != base_doc["manifest"]["sha256"]:
            raise SystemExit(f"{arm}: scored on another manifest than the baseline")
        if doc["manifest"]["sha256"] != base_doc["manifest"]["sha256"]:
            raise SystemExit(f"{arm}: detected on another manifest than the baseline")
        doc = apply_parity(doc, parity)
        e = eligibility(doc["spec_meta"], doc, rows, ref_ar)
        rc = rule_candidate(e, None)
        el = M.eligible(rc, rule)
        if el and units is None:
            raise SystemExit(f"{arm} is eligible but has no {RULE_MODEL} units: read it first")
        rc["arm"] = rule_arm(units["units"]) if units is not None else None
        cands.append(rc)
        report[arm] = dict(e, eligible=el, summary=units["summary"] if units is not None else None,
                           detection_summary=doc["_detection"] if units is None else None)
    result = M.apply_rule(cands, baselines, rule)
    extra = {}
    for c in cands:              # ineligible ones (read anyway): their comparisons, for the report only
        if c["name"] not in result["comparisons"] and c["arm"] is not None:
            extra[c["name"]] = {b: M.compare(c["arm"], a) for b, a in sorted(baselines.items())}
    return result, report, extra


def cmd_select(args):
    proto = protocol()
    base_doc = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    a9 = json.loads(Path(args.a9).read_text(encoding="utf-8"))
    parity = merge_parity(args.parity)
    result, report, extra = select(base_doc, args.units, a9, proto["rule"], args.baseline_units, args.cbin or None,
                                   parity)
    refine_rec = report.pop("_refine_arena")
    doc = {"kind": "textspot-selection", "computed": utc_now(), "protocol_sha256": sha256(PROTOCOL),
           "baseline_sha256": sha256(args.baseline), "a9_sha256": sha256(args.a9), "a9": rel(args.a9),
           "rule": proto["rule"], "refine_ms_per_box": REFINE_MS_PER_BOX, "code_sha256": code_identity(),
           "parity_sha256": {rel(f): sha256(f) for f in args.parity}, "refine_arena": refine_rec,
           "manifest": base_doc["manifest"], "candidates": report, "result": result,
           "report_comparisons": extra,
           "baselines": {a: base_doc["arms"][a][RULE_MODEL]["summary"] for a in BASELINES}}
    write_json(args.out, doc, indent=1)
    for name in result["eligible"]:
        r = report[name]
        s = r["summary"]
        print(f"{name:40s} {r['blob_bytes']:6d} B  est {r['est_ms_333_960']} ms  F1 {fmt(s['detection']['f1'])}  "
              f"CER {s['cer_all']['cer']}  qualifies {name in result['qualifying']}")
    print(f"eligible {len(result['eligible'])} of {len(report)}; qualifying {result['qualifying']}; winner "
          f"{result['winner']}; report candidate {result['report_candidate']}")


# ----------------------------------------------------------------------------- tables

def _pct(v):
    return "–" if v is None else f"{100 * v:.1f}%"


def _cer(s, k):
    return "–" if not s.get(k) or s[k]["cer"] is None else f"{s[k]['cer']:.2f}"


def baseline_table(base_doc):
    md = ["| baseline | P | R | F1 | mean IoU | CER all (N) | dialogue (N) | dialogue S / L | text-free with a box "
          "| boxes per text-free frame | pick | C = twin |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for arm in BASELINES:
        e = base_doc["arms"][arm]
        s = e[RULE_MODEL]["summary"]
        d = s["detection"]
        sl = " / ".join(_cer(e[m]["summary"], "cer_dialogue") if m in e else "–" for m in ("VOCR5-S", "VOCR5-L"))
        tf = s["text_free"]
        md.append(f"| `{arm}` | {fmt(d['precision'], 3)} | {fmt(d['recall'], 3)} | {fmt(d['f1'], 3)} | "
                  f"{fmt(s['tightness']['mean_iou'], 3)} | {_cer(s, 'cer_all')} | {_cer(s, 'cer_dialogue')} | {sl} | "
                  f"{_pct(tf['share_with_any_box'])} | {fmt(tf['boxes_per_frame'], 2)} | "
                  f"{_pct(s['pick']['hit_share'])} | {s['c_equal']} |")
    return "\n".join(md)


def part_table(base_doc, arm="grad_refine"):
    s = base_doc["arms"][arm][RULE_MODEL]["summary"]
    md = [f"| part (`{arm}`, VOCR5-N) | frames | F1 | CER all | dialogue |", "|---|---:|---:|---:|---:|"]
    for p, ps in s.get("per_part", {}).items():
        md.append(f"| {p} | {ps['frames']} | {fmt(ps['detection']['f1'], 3)} | {_cer(ps, 'cer_all')} | "
                  f"{_cer(ps, 'cer_dialogue')} |")
    return "\n".join(md)


def class_table(base_doc):
    md = ["| baseline | " + " | ".join(M.CLASSES) + " |", "|---|" + "---:|" * len(M.CLASSES)]
    for arm in BASELINES:
        r = base_doc["arms"][arm][RULE_MODEL]["summary"]["recall_by_class"]
        md.append(f"| `{arm}` | " + " | ".join(fmt(r[c]["recall"], 3) for c in M.CLASSES) + " |")
    return "\n".join(md)


def candidate_table(dev_doc):
    res = dev_doc["result"]
    md = ["| candidate | decode | VTSP B | MACs / 960×544 | arena B | est. ms @333 | C = twin | F1 | CER all | "
          "dialogue | eligible | qualifies |", "|---|---|---:|---:|---:|---:|---|---:|---:|---:|---|---|"]
    order = sorted(dev_doc["candidates"].items(), key=lambda kv: (kv[1]["blob_bytes"], kv[1]["macs_960"],
                                                                 kv[1]["decode"] != "det", kv[0]))
    for name, c in order:
        s = c.get("summary") or {}
        d = s.get("detection") or {}
        ms = c["est_ms_333_960"]
        md.append(f"| `{name}` | {c['decode']} | {c['blob_bytes']:,} | {c['macs_960'] / 1e6:.1f} M | "
                  f"{'–' if c['arena_960'] is None else c['arena_960']} | "
                  f"{'–' if ms is None else ms} | {c['c_equal']} | {fmt(d.get('f1'), 3)} | "
                  f"{_cer(s, 'cer_all')} | {_cer(s, 'cer_dialogue')} | {c['eligible']} | "
                  f"{name in res['qualifying']} |")
    md.append("")
    md.append(f"Winner: {res['winner'] or 'none'}; report candidate: {res['report_candidate'] or 'none'}.")
    return "\n".join(md)


def cmd_tables(args):
    base_doc = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    parts = [baseline_table(base_doc), "", class_table(base_doc), "", part_table(base_doc)]
    if args.dev:
        parts += ["", candidate_table(json.loads(Path(args.dev).read_text(encoding="utf-8")))]
    text = "\n".join(parts) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)


# ----------------------------------------------------------------------------- C parity after the fact

def merge_parity(paths):
    out = {"arms": {}}
    for f in paths or []:
        out["arms"].update(json.loads(Path(f).read_text(encoding="utf-8"))["arms"])
    return out


def cmd_parity(args):
    """The C drivers on the frames of arms already run (OUT/<arm>/boxes.json, e.g. a frozen baseline detected
    before a driver existed): the twins run again into a scratch directory and must give the stored boxes; then
    each arm's c_equal per frame goes to a separate record (the stored boxes and units are not touched)."""
    proto = protocol()
    check_baseline_blobs(proto)
    rows, manifest = load_set(args, proto)
    frames = load_frames(rows, [Path(d) for d in args.shots] + list(SHOT_DIRS))
    specs = parse_arms(args.arms)
    cdir = Path(args.cbin) if args.cbin != "auto" else Path(args.scratch) / "cbuild"
    if args.cbin == "auto":
        cbuild(cdir, args.compiler)
    scratch = Path(args.scratch)
    per_frame = run_arms(frames, specs, manifest, scratch, [], cdir=cdir, workers=args.workers, force=True,
                         log=lambda x: print(x, flush=True), proto=proto)
    doc = {"kind": "textspot-parity", "computed": utc_now(), "manifest": manifest, "c_build": CDrivers(cdir).record,
           "code_sha256": code_identity(), "arms": {}}
    for s in specs:
        stored = Path(args.units) / s.name / "boxes.json"
        old = json.loads(stored.read_text(encoding="utf-8"))
        if old["manifest"]["sha256"] != manifest["sha256"]:
            raise SystemExit(f"{stored}: another manifest")
        fr = {}
        for f, arms in zip(frames, per_frame):
            a, o = arms[s.name], old["frames"][f.id]
            same = ([list(b) for b in a["boxes"]], [list(b) for b in a["blocks"]], a["pick"], a["truncated"]) == \
                (o["boxes"], o["blocks"], o["pick"], o["truncated"])
            if not same:
                raise SystemExit(f"{s.name} {f.id}: the twin no longer gives the stored boxes")
            fr[f.id] = a["c_equal"]
        doc["arms"][s.name] = {"boxes_sha256": sha256(stored), "frames": fr,
                               "c_equal": all(v is True for v in fr.values()),
                               "unchecked": sum(1 for v in fr.values() if v is None),
                               "differ": sum(1 for v in fr.values() if v is False)}
        print(f"{s.name}: C == twin {doc['arms'][s.name]['c_equal']} (unchecked {doc['arms'][s.name]['unchecked']}, "
              f"differ {doc['arms'][s.name]['differ']})", flush=True)
    write_json(args.out, doc, indent=1)


# ----------------------------------------------------------------------------- pseudo frames (harness tests only)

PSEUDO_PARTS = (("pc", 18), ("psp", 6), ("adrenaline", 4), ("text_free", 2))


def pseudo_rows(n=30, rp=None):
    """A pseudo set that exercises the harness on training-side data only: realplus screenshots (one per VN) of
    VNs that no reservation, eval set or eval/linefind manifest holds, with realplus's kept lines as pseudo
    references (their ink crops less the crop padding), at the protocol's parts: pc (scaled to fit 960x544), psp
    (scaled to fit 480x272), adrenaline (that, then 2x INTER_LINEAR; boxes x2) and text_free (no references).
    Nothing of it is a score."""
    from common import is_reserved_vn
    rp = Path(rp or ART / "realplus")
    shots = {f"{r['vn']}_{r['image']}": r for r in jsonl(rp / "manifests" / "shots.jsonl")}
    lines = {}
    for ln in jsonl(rp / "lines_full_v2.jsonl"):
        lines.setdefault(ln["shot"], []).append(ln)
    linefind = {r["vn"] for q in sorted((ROOT / "eval" / "linefind" / "manifests").glob("*.jsonl")) for r in jsonl(q)
                if "vn" in r}
    ok = [k for k, r in shots.items() if len(lines.get(k, [])) >= 2 and not is_reserved_vn(r["vn"], dev=True)
          and r["vn"] not in linefind and (rp / "shots" / "hires" / f"{k}.jpg").exists()]
    ok.sort(key=lambda k: hashlib.sha256(f"textspot-pseudo:{k}".encode()).hexdigest())
    picked, vns = [], set()
    for k in ok:
        if shots[k]["vn"] not in vns:
            vns.add(shots[k]["vn"])
            picked.append(k)
        if len(picked) == n:
            break
    parts = [p for p, c in PSEUDO_PARTS for _ in range(c)]
    parts = (parts * (n // len(parts) + 1))[:n]
    rows = []
    for k, part in zip(picked, parts):
        r = shots[k]
        w, h = r["dims"]

        def fit(tw, th):
            sc = min(tw / w, th / h)
            return [max(1, int(round(w * sc))), max(1, int(round(h * sc)))], ("INTER_AREA" if sc < 1 else
                                                                            "INTER_LINEAR")
        dv, mv = fit(960, 544)
        dp, mp = fit(480, 272)
        if part in ("pc", "text_free"):
            steps, dims, key, pad, f = [{"method": mv, "dims": dv}], dv, "vita_ink", 4, 1
        elif part == "psp":
            steps, dims, key, pad, f = [{"method": mp, "dims": dp}], dp, "psp_ink", 2, 1
        else:
            dims = [2 * dp[0], 2 * dp[1]]
            steps, key, pad, f = [{"method": mp, "dims": dp}, {"method": "INTER_LINEAR", "dims": dims}], "psp_ink", 2, 2
        refs = []
        for ln in ([] if part == "text_free" else lines[k]):
            if key not in ln["crops"]:
                continue
            x0, y0, x1, y1 = ln["crops"][key]
            bx = [f * (x0 + pad), f * (y0 + pad), f * (x1 - x0 - 2 * pad), f * (y1 - y0 - 2 * pad)]
            if bx[2] > 0 and bx[3] > 0:
                refs.append({"box": bx, "cls": "name" if ln.get("name") else "dialogue", "vertical": False,
                             "text": ln["label"], "ignore": False})
        rows.append({"id": f"pseudo-{part}-{k}", "vn": r["vn"], "part": part, "session": r.get("session"),
                     "url": r["url"], "sha256": r["sha256"], "src_dims": [w, h], "scale": steps, "dims": dims,
                     "text_free": part == "text_free", "refs": refs, "looks": 0,
                     "reviewer_kind": "pseudo: realplus kept lines"})
    return rows


def cmd_pseudo(args):
    rows = pseudo_rows(args.n)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} pseudo frames, {sum(len(r['refs']) for r in rows)} pseudo references -> {args.out}")


def cmd_cbuild(args):
    rec = cbuild(args.out, args.compiler)
    print(json.dumps(rec["binaries"], indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--set", required=True, choices=("dev", "pseudo", "test"))
    p.add_argument("--manifest", default="")
    p.add_argument("--arms", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--cbin", default="", help="a cbuild directory, or 'auto' to build one under OUT")
    p.add_argument("--models", default="N,S,L", help="horizontal recognizers (N,S,L or none: detection only)")
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--shots", action="append", default=[], help="more source screenshot directories")
    p.add_argument("--cache", default="")
    p.add_argument("--compiler", default="gcc")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--force", action="store_true", help="recompute boxes even where boxes.json matches")
    p = sub.add_parser("eligible")
    p.add_argument("--units", action="append", required=True)
    p.add_argument("--a9", required=True)
    p.add_argument("--cbin", default="")
    p.add_argument("--parity", action="append", default=[])
    p.add_argument("--out", default="", help="write the eligible arm specs (for run --arms)")
    p = sub.add_parser("parity")
    p.add_argument("--set", required=True, choices=("dev", "pseudo"))
    p.add_argument("--manifest", default="")
    p.add_argument("--arms", required=True)
    p.add_argument("--units", required=True, help="the run directory whose boxes are checked")
    p.add_argument("--scratch", required=True)
    p.add_argument("--cbin", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--shots", action="append", default=[])
    p.add_argument("--compiler", default="gcc")
    p.add_argument("--limit", type=int, default=0)
    p = sub.add_parser("baseline")
    p.add_argument("--units", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("select")
    p.add_argument("--baseline", required=True)
    p.add_argument("--baseline-units", default="")
    p.add_argument("--units", action="append", required=True)
    p.add_argument("--a9", required=True)
    p.add_argument("--cbin", default="")
    p.add_argument("--parity", action="append", default=[])
    p.add_argument("--out", required=True)
    p = sub.add_parser("tables")
    p.add_argument("--baseline", required=True)
    p.add_argument("--dev", default="")
    p.add_argument("--out", default="")
    p = sub.add_parser("cbuild")
    p.add_argument("--out", required=True)
    p.add_argument("--compiler", default="gcc")
    p = sub.add_parser("pseudo")
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=30)
    args = ap.parse_args(argv)
    {"run": cmd_run, "eligible": cmd_eligible, "baseline": cmd_baseline, "select": cmd_select,
     "tables": cmd_tables, "cbuild": cmd_cbuild, "pseudo": cmd_pseudo, "parity": cmd_parity}[args.cmd](args)


if __name__ == "__main__":
    main()
