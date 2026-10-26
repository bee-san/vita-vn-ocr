#!/usr/bin/env python3
"""Train one textspot6 run: a tagged full-frame text spotter (issue #128, docs/textspot6.md "Recipe";
eval/textspot6_protocol.json). Runs are named <recipe>_<shape>_seed<n>.

  textspot6_train.py --recipe base|nohook|long|kd --shape S --seed N --out RUN \
      --hook-manifest M --hook-shards DIR [--hook-splits train] \
      --manifest M ... --shards DIR ... [--teacher RUN] [--steps N] [--threads 5] [--probe N]

The network is train/textspot6_model.py's: textspot3's family T (a d2s trunk, VTSP version 4 "trunk") or family M
(multi-scale, VTSP version 4 "ms") with a head of 16 + 4 T channels, T = 4 tag maps (textspot6_twin.CLASSES:
dialogue, name, choice, ui, scene). The shape comes from train/textspot6_shapes.py and names its scale rule.

Data (a batch is 16 windows of 256 x 512 at the network's scale; each window's source is drawn by the mix):
  hook    the hooked page shards (synth/textspot6_pages.py; --hook-manifest pins them, --hook-splits chooses the
          splits read: train for screening, train,guard,hval for a final model); a title is drawn with
          p ~ pages^title_temp, then a page of it uniformly
  synth   textspot's synthetic whole frames (data/manifests/textspot_shards_v1.json, synth_frames)
  real    textspot2's real half (data/manifests/textspot2_shards_v1.json: meiki labels on VNDB screenshots outside
          the dev sample), its sources in textspot's proportions (realplus .175, teacher .175, owner .125, devnative
          .025, renormalised), textspot2's skip lists applied (textspot_train.FramePool)
  mix     base and long: hook 0.5, synth 0.25, real 0.25; nohook: synth 0.5, real 0.5 (what the hooked pages buy)

A window (make_window; the same for every source):
  1. adrenaline: a PSP hooked page (480 x 272) is, with p 0.5, first scaled 2x bilinear (cv2.INTER_LINEAR) to
     960 x 544, as Adrenaline shows a PSP game on the Vita
  2. the scale rule of the shape (textspot6_model.scales_of): f = 2 when the canvas is wider than half_above or
     taller than half_above_h, then textspot_twin.halve's exact 2 x 2 mean, as the runtime reads the frame
  3. zoom (p 0.3): a region of (256 / z) x (512 / z) network px, z log-uniform in [0.8, 1.25], resized to z times
     its size (INTER_AREA down, INTER_LINEAR up): other games' text sizes at the same screen size
  4. photometric, each independently: contrast and brightness (p 0.8: g' = a (g - m) + m + b, a in [0.7, 1.3], b in
     [-25, 25]), gamma (p 0.3, [0.7, 1.4]), inversion (p 0.1), Gaussian blur (p 0.15, sigma [0.3, 1.0]), noise
     (p 0.25, sigma [2, 8]), JPEG (p 0.15, quality [40, 90]), and a blend with a text-free synthetic frame's window
     (p 0.1: g' = alpha g + (1 - alpha) bg, alpha in [0.6, 0.9]): translucent boxes over busy scenes
  Boxes follow every geometric step (start floor, end ceil); a positive under 4 px across at the network's scale
  becomes an ignore area (textspot's rule); the window's valid area is its region.

Targets (targets): textspot3's round-1 kernel rule and weight map at stride 4 (textspot3_train.targets, line for
line), plus each kernel cell's class set as a 5-bit mask: dialogue {0}, name {1}, choice {2}, ui {3}, scene {4},
"notdlg" {1, 2, 3, 4} (hooked pages: text the hook does not cover), "text" (meiki's real half) every class (no tag
target). Loss: textspot2_train.loss_of (K BCE with the weight map, dice, 0.5 O BCE, A smooth L1) + the tag loss:
over the kernel cells with a class set that is not every class, -log of the softmax mass of the set over the five
logits (0, t1, t2, t3, t4), times the cell's line weight, averaged over those cells.

Schedule (base): textspot3's base recipe exactly: 30,000 steps, AdamW 4e-3 (textdet_train.param_groups' weight
decay), 500 warm-up steps, cosine to 1e-4, gradient clip 5; an EMA (0.999) of the parameters from step 1,000; at the
QAT start (80%) the average replaces the parameters, the BatchNorm statistics are re-estimated (32 batches), #97's
calibration (32 batches), the fold, the fake-quantised network; during QAT an average (0.999) of the QAT parameters
is the run's model. long: 60,000 steps, peak 2e-3, 1,000 warm-up steps. kd: base + the teacher's maps (a textspot6
run's final.pt in fake quantisation, frozen): textspot2_train.kd_loss on K, O, A1, A2 + the KL divergence of the
student's tag softmax from the teacher's on the cells where sigmoid(K_t) > 0.5.

Outputs (RUN must not exist or be empty; the last step is the run's model):
  RUN/run.json   the arguments, the recipe and every setting, the shape, the data (manifests' sha256, shards read,
                 the pool per source and title), the code's sha256, git state
  RUN/log.jsonl  every 200 steps: step, lr, sec, k_bce, dice, o, a, tag (and kd terms), loss
  RUN/final.pt   kind textspot6-qat (shape, tags, w, b, act_scale)
  RUN/final.vtsp the VTSP version-4 blob (synth/textspot6_twin.Model reads it back); RUN/final.vtn its network
  RUN/done.json  finished, steps, wall time, the blob's bytes and sha256
"""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "train", ROOT / "synth"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import textdet_train as TT  # noqa: E402
import textspot2_train as T2  # noqa: E402
import textspot3_train as T3T  # noqa: E402
import textspot6_model as M6  # noqa: E402
import textspot6_shapes as S6  # noqa: E402
import textspot6_twin as T6  # noqa: E402
import textspot_train as ST  # noqa: E402
import textspot_twin as TS  # noqa: E402

WIN_H, WIN_W = 256, 512
BATCH = 16
MAP_STRIDE = 4
MIN_ACROSS = 4
CLIP = 5.0
TAGS = M6.TAGS
N_CLASSES = TAGS + 1
ALL = (1 << N_CLASSES) - 1
CLASS_MASK = {"dialogue": 1, "name": 2, "choice": 4, "ui": 8, "scene": 16, "notdlg": 2 | 4 | 8 | 16, "text": ALL}
REAL_MIX = {"realplus": 0.175, "teacher": 0.175, "owner": 0.125, "devnative": 0.025}
MIXES = {"base": {"hook": 0.5, "synth": 0.25, "real": 0.25}, "nohook": {"synth": 0.5, "real": 0.5}}
RECIPES = {
    "base": {"steps": 30000, "lr": 4e-3, "warmup": 500, "mix": "base"},
    "nohook": {"steps": 30000, "lr": 4e-3, "warmup": 500, "mix": "nohook"},
    "long": {"steps": 60000, "lr": 2e-3, "warmup": 1000, "mix": "base"},
    "kd": {"steps": 30000, "lr": 4e-3, "warmup": 500, "mix": "base"},
}
AUG = {
    "adrenaline_p": 0.5,
    "zoom_p": 0.3, "zoom": [0.8, 1.25],
    "contrast_p": 0.8, "contrast": [0.7, 1.3], "brightness": [-25, 25],
    "gamma_p": 0.3, "gamma": [0.7, 1.4],
    "invert_p": 0.1,
    "blur_p": 0.15, "blur_sigma": [0.3, 1.0],
    "noise_p": 0.25, "noise_sigma": [2.0, 8.0],
    "jpeg_p": 0.15, "jpeg_quality": [40, 90],
    "blend_p": 0.1, "blend_alpha": [0.6, 0.9],
}
TITLE_TEMP = 0.5
TAG_WEIGHT = 1.0
KD_TAG_WEIGHT = 1.0
EMA_DECAY, EMA_START, BN_BATCHES = 0.999, 1000, 32
KIND = M6.KIND


# ----------------------------------------------------------------------------- the hooked pool

class HookPool:
    """The hooked page shards of a textspot6-pages manifest in the given split directories, memory-mapped,
    grouped by title. verify: each shard's npy and json sha256 against the manifest."""

    def __init__(self, manifest, root, splits, verify=False):
        doc = json.loads(Path(manifest).read_text(encoding="utf-8"))
        if doc.get("kind") != "textspot6-pages":
            raise SystemExit(f"{manifest}: not a textspot6-pages manifest")
        self.data, self.by_title, self.read = [], {}, []
        for s in doc["shards"]:
            if s["split"] not in splits:
                continue
            base = Path(root) / s["dir"] / s["name"]
            npy, js = base.with_suffix(".npy"), base.with_suffix(".json")
            if verify:
                for path, want in ((npy, s["npy_sha256"]), (js, s["json_sha256"])):
                    if TT.sha256(path) != want:
                        raise SystemExit(f"{path}: sha256 is not {manifest}'s")
            d = len(self.data)
            self.data.append(np.load(npy, mmap_mode="r"))
            samples = json.loads(js.read_text(encoding="utf-8"))["samples"]
            for smp in samples:
                if smp["split"] not in splits:
                    raise SystemExit(f"{js}: a {smp['split']} page in a {s['split']} shard")
                self.by_title.setdefault(smp["title"], []).append((d, smp))
            self.read.append({"dir": s["dir"], "name": s["name"], "samples": len(samples)})
        if not self.by_title:
            raise SystemExit(f"no hooked pages of splits {splits} in {manifest}")
        self.titles = sorted(self.by_title)

    def canvas(self, d, smp):
        return np.asarray(self.data[d][smp["off"]:smp["off"] + smp["h"] * smp["w"]]).reshape(smp["h"], smp["w"])

    def title_p(self, temp):
        n = np.array([len(self.by_title[t]) for t in self.titles], np.float64)
        p = n ** temp
        return p / p.sum()

    def summary(self, temp):
        p = self.title_p(temp)
        return {"titles": {t: {"pages": len(self.by_title[t]), "p": round(float(q), 5),
                               "lines": sum(len(s["lines"]) for _, s in self.by_title[t])}
                           for t, q in zip(self.titles, p)},
                "pages": sum(len(v) for v in self.by_title.values()), "shards_read": self.read}


class Sources:
    """Every source of the mix: (name, draw(rng) -> (canvas, sample)), with the mix's probabilities."""

    def __init__(self, mix, hook=None, frames=None, temp=TITLE_TEMP):
        self.names, self.p, self.draw = [], [], {}
        self.hook, self.frames = hook, frames
        if mix.get("hook"):
            if hook is None:
                raise SystemExit("the mix reads hooked pages: --hook-manifest and --hook-shards")
            tp = hook.title_p(temp)

            def draw_hook(rng):
                t = hook.titles[int(rng.choice(len(hook.titles), p=tp))]
                lst = hook.by_title[t]
                d, smp = lst[int(rng.integers(len(lst)))]
                return hook.canvas(d, smp), smp
            self._add("hook", mix["hook"], draw_hook)
        if mix.get("synth"):
            self._add_frames("synth", mix["synth"])
        if mix.get("real"):
            avail = {k: v for k, v in REAL_MIX.items() if frames is not None and frames.items.get(k)}
            if not avail:
                raise SystemExit("the mix reads the real half: pass textspot2's manifest and shard directories")
            tot = sum(avail.values())
            for k, v in avail.items():
                self._add_frames(k, mix["real"] * v / tot)
        p = np.array(self.p, np.float64)
        self.p = p / p.sum()
        # text-free synthetic frames for the blend
        self.empty = [] if frames is None else [(d, s) for d, s in frames.items.get("synth", []) if not s["lines"]]

    def _add(self, name, p, fn):
        self.names.append(name)
        self.p.append(float(p))
        self.draw[name] = fn

    def _add_frames(self, name, p):
        fr = self.frames
        if fr is None or not fr.items.get(name):
            raise SystemExit(f"no {name} frames: pass their manifest and shard directory")
        lst = fr.items[name]

        def draw(rng):
            d, smp = lst[int(rng.integers(len(lst)))]
            return fr.canvas(d, smp), smp
        self._add(name, p, draw)

    def pick(self, rng):
        name = self.names[int(rng.choice(len(self.names), p=self.p))]
        canvas, smp = self.draw[name](rng)
        return name, canvas, smp

    def empty_canvas(self, rng):
        if not self.empty:
            return None
        d, s = self.empty[int(rng.integers(len(self.empty)))]
        return self.frames.canvas(d, s)


# ----------------------------------------------------------------------------- windows

def scale_rule(shape):
    ha, hh = M6.scales_of(shape)

    def f_of(w, h):
        return 2 if (ha and w > ha) or (hh and h > hh) else 1
    return f_of


def _box_to(b, f, ox, oy, z):
    """A canvas box [x0, y0, x1, y1] at frame scale -> the window's (network px after the zoom), start floor, end
    ceil."""
    x0, y0, x1, y1 = (float(v) for v in b[:4])
    return [int(math.floor((x0 / f - ox) * z)), int(math.floor((y0 / f - oy) * z)),
            int(math.ceil((x1 / f - ox) * z)), int(math.ceil((y1 / f - oy) * z))]


def _photometric(img, rng, empty_win):
    import cv2
    g = img.astype(np.float32)
    a = AUG
    if rng.random() < a["contrast_p"]:
        m = float(g.mean())
        g = (g - m) * rng.uniform(*a["contrast"]) + m + rng.uniform(*a["brightness"])
    if rng.random() < a["gamma_p"]:
        g = 255.0 * np.power(np.clip(g, 0, 255) / 255.0, rng.uniform(*a["gamma"]))
    if empty_win is not None and rng.random() < a["blend_p"]:
        al = rng.uniform(*a["blend_alpha"])
        g = al * g + (1 - al) * empty_win.astype(np.float32)
    if rng.random() < a["invert_p"]:
        g = 255.0 - g
    if rng.random() < a["blur_p"]:
        g = cv2.GaussianBlur(g, (0, 0), rng.uniform(*a["blur_sigma"]))
    if rng.random() < a["noise_p"]:
        g = g + rng.normal(0.0, rng.uniform(*a["noise_sigma"]), g.shape)
    out = np.clip(g + 0.5, 0, 255).astype(np.uint8)
    if rng.random() < a["jpeg_p"]:
        ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(a["jpeg_quality"][0],
                                                                                         a["jpeg_quality"][1] + 1))])
        if ok:
            out = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
    return out


def make_window(canvas, smp, src, rng, f_of, sources=None, aug=True):
    """One window: (img uint8 h x w <= 256 x 512, positives [[x0, y0, x1, y1, vertical, weight, mask]], ignore
    areas) in window px at the network's scale."""
    import cv2
    lines, ignore = smp["lines"], smp["ignore"]
    up = 1
    if aug and src == "hook" and smp["kind"] == "psp" and rng.random() < AUG["adrenaline_p"]:
        canvas = cv2.resize(np.ascontiguousarray(canvas), (2 * canvas.shape[1], 2 * canvas.shape[0]),
                            interpolation=cv2.INTER_LINEAR)
        up = 2
    H, W = canvas.shape
    f = f_of(W, H)
    Hs, Ws = -(-H // f), -(-W // f)
    z = 1.0
    if aug and rng.random() < AUG["zoom_p"]:
        z = float(math.exp(rng.uniform(math.log(AUG["zoom"][0]), math.log(AUG["zoom"][1]))))
    rh, rw = min(Hs, int(round(WIN_H / z))), min(Ws, int(round(WIN_W / z)))
    ox, oy = int(rng.integers(0, Ws - rw + 1)), int(rng.integers(0, Hs - rh + 1))
    if f == 2:
        img = TS.halve(canvas[2 * oy:2 * (oy + rh), 2 * ox:2 * (ox + rw)])
    else:
        img = np.asarray(canvas[oy:oy + rh, ox:ox + rw])
    if z != 1.0:
        nh, nw = min(WIN_H, max(1, int(round(rh * z)))), min(WIN_W, max(1, int(round(rw * z))))
        img = cv2.resize(np.ascontiguousarray(img), (nw, nh), interpolation=cv2.INTER_AREA if z < 1 else
                         cv2.INTER_LINEAR)
        zx, zy = nw / rw, nh / rh
    else:
        zx = zy = 1.0
    h, w = img.shape
    empty_win = None
    if aug and sources is not None and AUG["blend_p"] > 0:
        e = sources.empty_canvas(rng)
        if e is not None:
            fe = f_of(e.shape[1], e.shape[0])
            es = TS.halve(e) if fe == 2 else e
            if es.shape[0] >= h and es.shape[1] >= w:
                ey, ex = int(rng.integers(0, es.shape[0] - h + 1)), int(rng.integers(0, es.shape[1] - w + 1))
                empty_win = es[ey:ey + h, ex:ex + w]
    if aug:
        img = _photometric(img, rng, empty_win)

    def to(b):
        x0, y0, x1, y1 = (float(v) * up for v in b[:4])
        return [int(math.floor((x0 / f - ox) * zx)), int(math.floor((y0 / f - oy) * zy)),
                int(math.ceil((x1 / f - ox) * zx)), int(math.ceil((y1 / f - oy) * zy))]

    ign = [to(b) for b in ignore]
    pos = []
    for ln in lines:
        b = to(ln)
        vertical = int(ln[4])
        if (b[2] - b[0] if vertical else b[3] - b[1]) < MIN_ACROSS:
            ign.append(b)
            continue
        cls = ln[5] if len(ln) > 5 else "text"
        if cls not in CLASS_MASK:
            raise ValueError(f"class {cls!r}")
        pos.append([*b, vertical, T2.line_weight(ln), CLASS_MASK[cls]])
    return np.ascontiguousarray(img), pos, ign


def targets(lines, ignore, w, h, s=MAP_STRIDE):
    """textspot3_train.targets on the WIN_H x WIN_W window (lines [[x0, y0, x1, y1, vertical, weight, mask]] in
    window px, the window's valid region w x h at the top left), line for line, plus the class-set mask of every
    kernel cell (0 where there is no tag target): (K, O, A1, A2, valid, kernel, W, M)."""
    ho, wo = WIN_H // s, WIN_W // s
    K = np.zeros((ho, wo), np.float32)
    O = np.zeros((ho, wo), np.float32)  # noqa: E741
    A1 = np.zeros((ho, wo), np.float32)
    A2 = np.zeros((ho, wo), np.float32)
    Wt = np.ones((ho, wo), np.float32)
    Mk = np.zeros((ho, wo), np.int64)
    owner = np.full((ho, wo), -1, np.int32)
    conflict = np.zeros((ho, wo), bool)
    valid = np.zeros((ho, wo), bool)
    vh, vw = (h + s - 1) // s, (w + s - 1) // s
    valid[:vh, :vw] = True
    ign = np.zeros((ho, wo), bool)

    def block_mark(m, b):
        x0, y0, x1, y1 = b
        if x1 <= x0 or y1 <= y0:
            return
        m[max(0, y0 // s):min(ho, (y1 - 1) // s + 1), max(0, x0 // s):min(wo, (x1 - 1) // s + 1)] = True

    for b in ignore:
        block_mark(ign, b)
    centres = np.arange(max(ho, wo)) * s + (s - 1) / 2
    for n, ln in enumerate(lines):
        x0, y0, x1, y1 = ln[:4]
        vert = bool(ln[4])
        a0, a1, b0, b1 = (x0, x1, y0, y1) if vert else (y0, y1, x0, x1)
        lim_a = w if vert else h
        lim_b = h if vert else w
        if a1 <= 0 or a0 >= lim_a or b1 <= 0 or b0 >= lim_b:
            continue
        if a0 < 0 or a1 > lim_a:
            block_mark(ign, (max(0, x0), max(0, y0), min(w, x1), min(h, y1)))
            continue
        b0, b1 = max(0, b0), min(lim_b, b1)
        across = a1 - a0
        c = (a0 + a1) / 2
        r = max(across / 4, s / 2)
        n_across = ho if not vert else wo
        dist = np.abs(centres[:n_across] - c)
        idx = np.flatnonzero(dist <= r)
        if not len(idx):
            idx = np.array([int(np.clip(round((c - (s - 1) / 2) / s), 0, n_across - 1))])
        along = np.arange(b0 // s, (b1 - 1) // s + 1)
        along = along[along < (wo if not vert else ho)]
        if not len(along):
            continue
        if vert:
            ii, jj = np.meshgrid(along, idx, indexing="ij")
        else:
            ii, jj = np.meshgrid(idx, along, indexing="ij")
        ii, jj = ii.ravel(), jj.ravel()
        cc = centres[jj] if vert else centres[ii]
        taken = owner[ii, jj] >= 0
        conflict[ii[taken], jj[taken]] = True
        owner[ii, jj] = n
        K[ii, jj] = 1
        O[ii, jj] = float(vert)
        A1[ii, jj] = cc - a0
        A2[ii, jj] = a1 - cc
        Wt[ii, jj] = float(ln[5])
        Mk[ii, jj] = int(ln[6])
    kern = (K > 0) & ~conflict & valid
    valid_k = valid & ~conflict & (~ign | kern)
    K[conflict] = 0
    Wt[~kern] = 1.0
    Mk[~kern] = 0
    Mk[Mk == ALL] = 0
    return K, O, A1, A2, valid_k, kern, Wt, Mk


def make_batch(sources, rng, f_of, aug=True):
    x = np.full((BATCH, 1, WIN_H, WIN_W), 128, np.uint8)
    valid_hw = []
    ho, wo = WIN_H // MAP_STRIDE, WIN_W // MAP_STRIDE
    tg = [np.zeros((BATCH, ho, wo), np.float32) for _ in range(5)]
    vm = np.zeros((BATCH, ho, wo), bool)
    km = np.zeros_like(vm)
    mk = np.zeros((BATCH, ho, wo), np.int64)
    srcs = []
    for b in range(BATCH):
        src, canvas, smp = sources.pick(rng)
        img, pos, ign = make_window(canvas, smp, src, rng, f_of, sources, aug)
        h, w = img.shape
        x[b, 0, :h, :w] = img
        valid_hw.append((h, w))
        K, O, A1, A2, v, k, Wt, M = targets(pos, ign, w, h)  # noqa: E741
        for t, a in zip(tg, (K, O, A1, A2, Wt)):
            t[b] = a
        vm[b], km[b], mk[b] = v, k, M
        srcs.append(src)
    K, O, A1, A2, Wt = (torch.from_numpy(t) for t in tg)  # noqa: E741
    return (torch.from_numpy(x), torch.tensor(valid_hw), K, O, A1, A2, torch.from_numpy(vm), torch.from_numpy(km),
            Wt, torch.from_numpy(mk))


class Batches(torch.utils.data.IterableDataset):
    def __init__(self, cfg, seed):
        super().__init__()
        self.cfg, self.seed = cfg, seed

    def __iter__(self):
        c = self.cfg
        hook = HookPool(c["hook_manifest"], c["hook_shards"], c["hook_splits"]) if c["mix"].get("hook") else None
        frames = ST.FramePool(c["manifests"], c["shards"], skip=c["skip"]) if c["manifests"] else None
        sources = Sources(c["mix"], hook, frames, c["title_temp"])
        rng = np.random.default_rng(self.seed)
        f_of = scale_rule(c["shape"])
        while True:
            yield make_batch(sources, rng, f_of, c.get("aug", True))


# ----------------------------------------------------------------------------- losses

def tag_logits(t):
    """(N, T, h, w) tag maps -> (N, T + 1, h, w) class logits (class 0's logit is 0)."""
    return torch.cat([torch.zeros_like(t[:, :1]), t], 1)


def tag_loss(t, Mk, Wt):
    """-log of the softmax mass of each tagged cell's class set, times its weight, averaged over the tagged
    cells."""
    logits = tag_logits(t)
    bits = (Mk.unsqueeze(1) >> torch.arange(N_CLASSES, device=Mk.device).view(1, -1, 1, 1)) & 1
    has = Mk > 0
    n = has.float().sum().clamp(min=1)
    lse_all = torch.logsumexp(logits, 1)
    masked = logits.masked_fill(bits == 0, float("-inf"))
    lse_set = torch.logsumexp(masked, 1)
    per = torch.where(has, lse_all - lse_set, torch.zeros_like(lse_all))
    loss = (per * Wt * has.float()).sum() / n
    with torch.no_grad():
        arg = logits.argmax(1, keepdim=True)
        right = bits.gather(1, arg).squeeze(1).bool() & has
        acc = float(right.float().sum() / n)
    return loss, {"tag": float(loss), "tag_acc": round(acc, 4), "tag_cells": int(has.sum())}


def kd_tag_loss(t, t_teacher, k_teacher):
    """KL(teacher || student) of the tag softmax over the cells where sigmoid(K_t) > 0.5."""
    m = (torch.sigmoid(k_teacher) > 0.5).float()
    n = m.sum().clamp(min=1)
    pt = F.softmax(tag_logits(t_teacher.detach()), 1)
    ls = F.log_softmax(tag_logits(t), 1)
    kl = (pt * (torch.log(pt.clamp(min=1e-12)) - ls)).sum(1)
    return (kl * m).sum() / n


# ----------------------------------------------------------------------------- the run

def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recipe", required=True, choices=list(RECIPES))
    ap.add_argument("--shape", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hook-manifest")
    ap.add_argument("--hook-shards")
    ap.add_argument("--hook-splits", default="train")
    ap.add_argument("--manifest", nargs="*", default=[])
    ap.add_argument("--shards", nargs="*", default=[])
    ap.add_argument("--teacher", help="kd: the teacher run's directory")
    ap.add_argument("--steps", type=int)
    ap.add_argument("--threads", type=int, default=5)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--probe", type=int, default=0)
    ap.add_argument("--cal-batches", type=int, default=TT.CAL_BATCHES)
    ap.add_argument("--bn-batches", type=int, default=BN_BATCHES)
    ap.add_argument("--ema-start", type=int, default=EMA_START)
    ap.add_argument("--shape-json", help="tests: the shape as a JSON object")
    ap.add_argument("--no-aug", action="store_true", help="tests: no augmentation")
    ap.add_argument("--mix-json", help="tests and probes: the mix as a JSON object (not a registered recipe's)")
    ap.add_argument("--verify", action="store_true", help="check every shard's sha256 first")
    args = ap.parse_args(argv)
    r = RECIPES[args.recipe]
    args.steps = r["steps"] if args.steps is None else args.steps
    if (args.recipe == "kd") != bool(args.teacher):
        ap.error("--teacher is required for kd and only for kd")
    if args.ema_start > int(TT.QAT_FRAC * args.steps):
        args.ema_start = max(1, int(TT.QAT_FRAC * args.steps) // 2)
    if args.shape_json:
        args.shape_spec = json.loads(args.shape_json)
        if args.shape_spec.get("name") != args.shape:
            ap.error("--shape-json names another shape")
    else:
        if args.shape not in S6.SHAPES:
            ap.error(f"{args.shape}: not a textspot6 shape ({sorted(S6.SHAPES)})")
        args.shape_spec = dict(S6.SHAPES[args.shape])
    args.name = f"{args.recipe}_{args.shape}_seed{args.seed}"
    return args


def code_sha256():
    files = ["train/textspot6_train.py", "train/textspot6_model.py", "train/textspot6_shapes.py",
             "synth/textspot6_twin.py", "train/textspot3_train.py", "train/textspot3_model.py",
             "train/textspot2_train.py", "train/textspot2_export.py", "train/textspot_train.py",
             "train/textdet_train.py", "train/textdet_model.py", "synth/textspot3_twin.py",
             "synth/textspot2_twin.py", "synth/textspot_twin.py", "synth/textdet_twin.py"]
    return {f: TT.sha256(ROOT / f) for f in files if (ROOT / f).exists()}


class Family:
    """The family's model, masks, maps and checkpoint (train/textspot6_model.py)."""

    def __init__(self, shape):
        self.shape = shape
        self.fam = M6.family_of(shape)

    def model(self):
        return M6.float_model(self.shape, TAGS)

    def masks(self, vhw):
        return M6.layer_masks(self.shape, vhw, WIN_H, WIN_W)

    def maps(self, y):
        return M6.to_maps(y, WIN_H, WIN_W, TAGS)

    def qmodel(self, folded, act_scale):
        return M6.q_model(self.shape, folded, act_scale, TAGS)


def load_teacher(run_dir):
    d = Path(run_dir)
    ck = torch.load(d / "final.pt", map_location="cpu", weights_only=False)
    if ck.get("kind") != KIND:
        raise SystemExit(f"{d}/final.pt: not a textspot6 checkpoint")
    fam = Family(ck["shape"])
    model = M6.q_model(ck["shape"], list(zip(ck["w"], ck["b"])), ck["act_scale"], int(ck["tags"]))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    rec = {"dir": str(d), "shape": ck["shape"]["name"], "pt_sha256": TT.sha256(d / "final.pt"),
           "vtsp_sha256": TT.sha256(d / "final.vtsp") if (d / "final.vtsp").exists() else None}
    return model, fam, rec


def main(argv=None):
    args = parse(argv)
    t_proc = time.time()
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty: a run starts from scratch in a fresh directory")
    r = RECIPES[args.recipe]
    shape = args.shape_spec
    fam = Family(shape)
    mix = json.loads(args.mix_json) if args.mix_json else MIXES[r["mix"]]
    skip_shots, skip_vns, lists = T2.skip_lists(args.manifest) if args.manifest else (frozenset(), frozenset(), [])
    cfg = {"mix": mix, "shape": shape, "hook_manifest": args.hook_manifest, "hook_shards": args.hook_shards,
           "hook_splits": tuple(args.hook_splits.split(",")), "manifests": list(args.manifest),
           "shards": list(args.shards), "skip": (skip_shots, skip_vns), "title_temp": TITLE_TEMP,
           "aug": not args.no_aug}
    # the pools once here: checks and the run record
    hook = HookPool(args.hook_manifest, args.hook_shards, cfg["hook_splits"], verify=args.verify) \
        if mix.get("hook") else None
    frames = ST.FramePool(args.manifest, args.shards, verify=args.verify, skip=(skip_shots, skip_vns)) \
        if args.manifest else None
    sources = Sources(mix, hook, frames, TITLE_TEMP)
    pool_rec = {"sources": dict(zip(sources.names, (round(float(q), 6) for q in sources.p))),
                "hook": hook.summary(TITLE_TEMP) if hook else None,
                "frames": frames.summary() if frames else None, "text_free_synthetic": len(sources.empty)}
    skip_rec = ST.skip_record(lists, frames.dropped) if lists and frames is not None else None
    del hook, frames, sources
    teacher = t_rec = t_fam = None
    if args.recipe == "kd":
        teacher, t_fam, t_rec = load_teacher(args.teacher)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    peak, warmup = r["lr"], r["warmup"]
    model = fam.model()
    opt = torch.optim.AdamW(TT.param_groups(list(model.parameters())), lr=peak)
    loader = torch.utils.data.DataLoader(Batches(cfg, args.seed), batch_size=None, num_workers=args.workers,
                                         persistent_workers=args.workers > 0,
                                         prefetch_factor=4 if args.workers > 0 else None)
    it = iter(loader)
    qat_start = int(TT.QAT_FRAC * args.steps)
    run = {"kind": "textspot6-run", "name": args.name, "started": ST.utc(), "recipe": args.recipe,
           "args": {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(args).items()},
           "shape": shape, "family": fam.fam, "tags": TAGS, "classes": list(T6.CLASSES), "seed": args.seed,
           "steps": args.steps, "qat_start": qat_start, "lr": {"peak": peak, "warmup": warmup, "min": TT.LR_MIN},
           "mix": mix, "real_mix": REAL_MIX, "title_temp": TITLE_TEMP, "aug": AUG if cfg["aug"] else None,
           "tag_weight": TAG_WEIGHT, "kd_tag_weight": KD_TAG_WEIGHT if teacher is not None else None,
           "class_mask": CLASS_MASK, "batch": BATCH, "window": [WIN_H, WIN_W], "min_across": MIN_ACROSS,
           "ema": {"decay": EMA_DECAY, "start": args.ema_start, "bn_batches": args.bn_batches},
           "teacher": t_rec, "pool": pool_rec,
           "manifests_sha256": {m: TT.sha256(m) for m in [*args.manifest, *([args.hook_manifest] if
                                                                              args.hook_manifest else [])]},
           "skip": skip_rec, "code_sha256": code_sha256(),
           "git": ST.git_state(), "torch": torch.__version__, "threads": args.threads}
    (out / "run.json").write_text(json.dumps(run, indent=1) + "\n", encoding="utf-8")
    log = open(out / "log.jsonl", "a", encoding="utf-8")

    def write(rec):
        log.write(json.dumps(rec) + "\n")
        log.flush()

    t0 = time.time()
    agg, n_agg = {}, 0
    act_scale = None
    ema = T2.EMA(EMA_DECAY)
    qema = None
    n_steps = min(args.steps, args.probe + 1) if args.probe else args.steps
    times = []
    for step in range(n_steps):
        if step == qat_start and not args.probe:
            ema.copy_to(model.parameters())
            n_bn = T3T.reestimate_bn(model, it, fam.masks, args.bn_batches)
            write({"step": step, "ema_swap": True, "ema_steps": ema.steps, "bn_layers": n_bn})
            act_scale = T3T.calibrate(model, it, fam.masks, args.cal_batches)
            model.eval()
            folded = M6.fold(model)
            model = fam.qmodel(folded, act_scale)
            opt = torch.optim.AdamW(TT.param_groups(list(model.parameters())),
                                    lr=T2.lr_at(step, args.steps, peak, warmup))
            qema = T2.EMA(EMA_DECAY).step(model.parameters())
            write({"step": step, "qat": True, "act_scale": act_scale})
        ts = time.perf_counter()
        lr = T2.lr_at(step, args.steps, peak, warmup)
        for g in opt.param_groups:
            g["lr"] = lr
        x, vhw, K, O, A1, A2, vm, km, Wt, Mk = next(it)  # noqa: E741
        td = time.perf_counter()
        masks = fam.masks(vhw.tolist())
        xin = TT.to_input(x)
        base, tags = fam.maps(model(xin, masks=masks))
        loss, parts = T2.loss_of(base, K, O, A1, A2, vm, km, Wt)
        lt, tparts = tag_loss(tags, Mk, Wt)
        loss = loss + TAG_WEIGHT * lt
        parts.update(tparts)
        if teacher is not None:
            with torch.no_grad():
                tb, tt = t_fam.maps(teacher(xin, masks=t_fam.masks(vhw.tolist())))
            kl, kparts = T2.kd_loss(base, tb, vm)
            kt = kd_tag_loss(tags, tt, tb[:, 0])
            loss = loss + kl + KD_TAG_WEIGHT * kt
            parts.update(kparts)
            parts["kd_tag"] = float(kt)
        parts["loss"] = float(loss)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        opt.step()
        if qema is not None:
            qema.step(model.parameters())
        elif step + 1 >= args.ema_start:
            ema.step(model.parameters())
        if args.probe:
            times.append((td - ts, time.perf_counter() - ts))
        for k, v in parts.items():
            agg[k] = agg.get(k, 0.0) + v
        n_agg += 1
        if (step + 1) % args.log_every == 0 or step + 1 == n_steps:
            rec = {"step": step + 1, "lr": round(lr, 6), "sec": round(time.time() - t0, 1),
                   **{k: round(v / n_agg, 5) for k, v in agg.items()}}
            write(rec)
            print(json.dumps(rec), flush=True)
            agg, n_agg = {}, 0
    if args.probe:
        timed = times[1:]
        ms = np.array([t for _, t in timed]) * 1000
        wait = np.array([t for t, _ in timed]) * 1000
        probe = {"finished": ST.utc(), "name": args.name, "steps_timed": len(timed),
                 "ms_per_step": round(float(ms.mean()), 2), "ms_per_step_median": round(float(np.median(ms)), 2),
                 "ms_waiting_for_data": round(float(wait.mean()), 2), "threads": args.threads,
                 "startup_s": round(t0 - t_proc, 1)}
        (out / "probe.json").write_text(json.dumps(probe, indent=1) + "\n", encoding="utf-8")
        log.close()
        print("TEXTSPOT6_PROBE_DONE", json.dumps(probe), flush=True)
        return probe
    if qema is not None:
        qema.copy_to(model.parameters())
        write({"step": n_steps, "qat_average": "the run's model", "qat_average_steps": qema.steps})
    ck = {"kind": KIND, "shape": shape, "tags": TAGS, "seed": args.seed, "steps": args.steps,
          "w": [w.detach().double() for w in model.w], "b": [b.detach().double() for b in model.b],
          "act_scale": act_scale, "run": run}
    torch.save(ck, out / "final.pt")
    net, vtsp = M6.export_checkpoint(out / "final.pt")
    m = T6.Model(vtsp)
    if m.version != T6.VERSION_TAG or m.tags != TAGS or m.name != shape["name"]:
        raise SystemExit("the VTSP version-4 blob does not read back")
    (out / "final.vtn").write_bytes(net)
    (out / "final.vtsp").write_bytes(vtsp)
    done = {"finished": ST.utc(), "name": args.name, "steps": args.steps, "seconds": round(time.time() - t0, 1),
            "wall_seconds": round(time.time() - t_proc, 1), "pt_sha256": TT.sha256(out / "final.pt"),
            "vtsp_bytes": len(vtsp), "vtsp_sha256": hashlib.sha256(vtsp).hexdigest(), "vtsp_version": m.version,
            "form": m.form}
    (out / "done.json").write_text(json.dumps(done, indent=1) + "\n", encoding="utf-8")
    log.close()
    print("TEXTSPOT6_TRAIN_DONE", json.dumps(done), flush=True)
    return done


if __name__ == "__main__":
    main()
