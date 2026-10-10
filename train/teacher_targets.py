#!/usr/bin/env python3
"""Teacher targets from PP-OCRv6 small rec for distilling the tiny recognizer.

  teacher_targets.py run --out DIR [--workers 5] [--limit N] SHARD.npz ...
  teacher_targets.py check DIR/NAME.v6s.npz --shard SHARD.npz     (agreement with labels)

For every line crop of a render.py-format shard (grey, any height) this writes, to
DIR/<shard name>.v6s.npz:

  text       the teacher's reading of the crop at its natural aspect, in the training-label
             form (teacher/vote.canon), preprocessed as rapidocr and the teacher labeller do
             (BGR, height 48, ceil(48 w / h) wide, (x/255 - 0.5)/0.5, zero-padded to 320 px):
             the reading PR #11 measured at 1.33% / 1.23% CER on vita-test / psp-test
  conf       per line: [min, mean] over the read characters of each character's mean
             frame probability (rapidocr's score)
  T          the student's frame count for the crop: ceil(W32 / 8), W32 = the width after
             train.py's resize to height 32
  soft_ok    whether the frame targets below are usable (see "alignment")
  kid, kp    frame targets on the student's frame grid: for frame t of line i (rows
             off[i] .. off[i] + T[i]) the K most probable classes of the STUDENT's vocab
             (class 0 = CTC blank, class j = data/vocab.txt line j) and their probabilities
             (float16); the rest of the mass is dropped.
  off        row offsets into kid / kp

Alignment. The teacher's stride is 8 px at height 48, the student's 8 px at height 32, so
at natural aspect the teacher has 1.5 frames per student frame. The aligned pass instead
reads the student's own input (the crop resized to 32 x W32, padded to 8 T columns by
repeating the last column, exactly train.py's make_batch) resized to 48 x 16 T. Teacher
frames 2t and 2t+1 then cover exactly student frame t's columns, and are pooled into one
distribution: blank = p_2t(blank) p_2t+1(blank), and every other class c gets
1 - (1 - p_2t(c)) (1 - p_2t+1(c)), renormalised to the non-blank mass. The aligned input is
1.33x wider than the teacher's natural aspect. A line's frame targets are used only when the
greedy reading of the pooled targets equals the natural-aspect reading (soft_ok), so lines
where the stretch changes the reading, or where pooling merges a repeated character, fall
back to the hard label.

Teacher classes map to the student's vocab through teacher/vote.canon (so ! -> ！, half-width
kana widen, full-width ASCII folds); the space class is blank, and the mass of classes with
no single-character image in the vocab is dropped before the renormalisation.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "synth"))
sys.path.insert(0, str(ROOT))

HERD = Path(os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr"))
V6S = Path(os.environ.get("VOCR_V6S", HERD / "shared/artifacts/teacher/models/PP-OCRv6_rec_small.onnx"))
V6S_SHA256 = "6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884"
K = 4


def canon(s):
    from teacher.vote import canon as _canon
    return _canon(s)


def student_input(g, height=32, stride=8, max_w=1024):
    """train.py's resize + make_batch padding (no augmentation): uint8 height x 8T, and T."""
    h, w = g.shape
    nw = max(8, min(max_w, int(round(w * height / h))))
    r = cv2.resize(g, (nw, height), interpolation=cv2.INTER_LINEAR)
    T = (nw + stride - 1) // stride
    x = np.empty((height, T * stride), np.uint8)
    x[:, :nw] = r
    x[:, nw:] = r[:, -1:]
    return x, T


class Teacher:
    def __init__(self, vocab, threads=1, path=V6S):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        self.s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.inp = self.s.get_inputs()[0].name
        self.set_classes(self.s.get_modelmeta().custom_metadata_map["character"].split("\n"), vocab)

    def set_classes(self, chars, vocab):
        """The teacher's dictionary (chars, without blank and space) mapped onto the student's
        vocab through canon(); split out of __init__ so tests can build it without the model."""
        self.chars = [""] + list(chars) + [" "]             # 0 = blank, last = space
        ours = {c: i + 1 for i, c in enumerate(vocab)}
        # teacher class -> student class (0 = blank), or -1 = no single-character image
        m = np.full(len(self.chars), -1, np.int64)
        m[0] = 0
        for i, c in enumerate(self.chars[1:], 1):
            k = canon(c)
            if k == "":
                m[i] = 0
            elif len(k) == 1 and k in ours:
                m[i] = ours[k]
        self.map = m
        self.n_student = len(vocab) + 1
        # teacher columns grouped by their student class, for one np.add.reduceat per line
        sel = np.flatnonzero(m >= 0)
        sel = sel[np.argsort(m[sel], kind="stable")]
        cls = m[sel]
        starts = np.flatnonzero(np.r_[True, cls[1:] != cls[:-1]])
        self.sel, self.starts, self.uniq = sel, starts, cls[starts]

    def _run(self, bgr48):
        """bgr48: 48 x W x 3 float32 in [0, 255]; zero-padded (normalised) to >= 320 px."""
        nw = bgr48.shape[1]
        x = (bgr48 / 255.0 - 0.5) / 0.5
        pad = np.zeros((48, max(nw, 320), 3), np.float32)
        pad[:, :nw] = x
        return self.s.run(None, {self.inp: pad.transpose(2, 0, 1)[None]})[0][0]   # frames x C

    def read_natural(self, g):
        """rapidocr's preprocessing of the crop (grey replicated to BGR): text, [min, mean]."""
        h, w = g.shape
        nw = min(4096, max(8, int(math.ceil(48 * w / max(h, 1)))))
        x = cv2.resize(g, (nw, 48), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        y = self._run(np.repeat(x[:, :, None], 3, axis=2))
        return self.decode(y)

    def decode(self, y):
        idx, prob = y.argmax(1), y.max(1)
        text, scores, last = [], [], 0
        for k, p in zip(idx, prob):
            if k and k != last:
                text.append(self.chars[k])
                scores.append([float(p)])
            elif k and k == last:
                scores[-1].append(float(p))
            last = k
        cs = [float(np.mean(s)) for s in scores]
        return "".join(text).strip(), (min(cs) if cs else 0.0, float(np.mean(cs)) if cs else 0.0)

    def aligned(self, g):
        """Frame targets on the student's grid: (T, K ids, K probs, greedy student-class seq)."""
        xs, T = student_input(g)
        x = cv2.resize(xs, (16 * T, 48), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        y = self._run(np.repeat(x[:, :, None], 3, axis=2))[:2 * T]       # 2T x C_teacher
        # teacher classes -> student classes (sum mass of classes that map to one class)
        ys = np.zeros((y.shape[0], self.n_student), np.float32)
        ys[:, self.uniq] = np.add.reduceat(y[:, self.sel], self.starts, axis=1)
        tot = ys.sum(1, keepdims=True)
        ys /= np.maximum(tot, 1e-12)
        a, b = ys[0::2], ys[1::2]
        blank = a[:, 0] * b[:, 0]
        nb = 1.0 - (1.0 - a[:, 1:]) * (1.0 - b[:, 1:])
        nb *= ((1.0 - blank) / np.maximum(nb.sum(1), 1e-12))[:, None]
        q = np.concatenate([blank[:, None], nb], axis=1)                  # T x C_student
        kid = np.argpartition(-q, K, axis=1)[:, :K]
        kp = np.take_along_axis(q, kid, 1)
        order = np.argsort(-kp, axis=1)
        kid, kp = np.take_along_axis(kid, order, 1), np.take_along_axis(kp, order, 1)
        best, seq, last = q.argmax(1), [], 0
        for t in best:
            if t and t != last:
                seq.append(int(t))
            last = t
        return T, kid.astype(np.int16), kp.astype(np.float16), seq


def load_vocab(path=ROOT / "data" / "vocab.txt"):
    return [c for c in Path(path).read_text(encoding="utf-8").split("\n") if c]


def shard_lines(path, limit=0):
    z = np.load(path)
    data, offs, shapes = z["data"], z["offsets"], z["shapes"]
    n = len(shapes) if not limit else min(limit, len(shapes))
    for i in range(n):
        h, w = (int(v) for v in shapes[i])
        yield i, data[offs[i]:offs[i + 1]].reshape(h, w)


def _work(args):
    path, lo, hi, vocab = args
    t = Teacher(vocab, threads=1)
    out = []
    z = np.load(path)
    data, offs, shapes = z["data"], z["offsets"], z["shapes"]
    for i in range(lo, hi):
        h, w = (int(v) for v in shapes[i])
        g = data[offs[i]:offs[i + 1]].reshape(h, w)
        text, conf = t.read_natural(g)
        T, kid, kp, seq = t.aligned(g)
        lab = canon(text)
        stext = "".join(vocab[j - 1] for j in seq)
        out.append((i, lab, conf, T, kid, kp, stext))
    return out


def run(args):
    import hashlib
    from multiprocessing import get_context
    got = hashlib.sha256(V6S.read_bytes()).hexdigest()
    if got != V6S_SHA256:
        raise SystemExit(f"{V6S}: sha256 {got} is not the pinned {V6S_SHA256}")
    vocab = load_vocab()
    args.out.mkdir(parents=True, exist_ok=True)
    ctx = get_context("spawn")
    for path in args.shards:
        path = Path(path)
        dst = args.out / (path.stem + ".v6s.npz")
        if dst.exists() and not args.force:
            print(f"{dst} exists, skipped", flush=True)
            continue
        n = len(np.load(path)["shapes"])
        if args.limit:
            n = min(n, args.limit)
        step = max(1, math.ceil(n / (args.workers * 8)))
        jobs = [(str(path), lo, min(n, lo + step), vocab) for lo in range(0, n, step)]
        t0 = time.time()
        rows = []
        with ctx.Pool(args.workers) as pool:
            for part in pool.imap_unordered(_work, jobs):
                rows += part
        rows.sort(key=lambda r: r[0])
        T = np.array([r[3] for r in rows], np.int32)
        off = np.concatenate([[0], np.cumsum(T)]).astype(np.int64)
        kid = np.concatenate([r[4] for r in rows]) if rows else np.zeros((0, K), np.int16)
        kp = np.concatenate([r[5] for r in rows]) if rows else np.zeros((0, K), np.float16)
        text = [r[1] for r in rows]
        stext = [r[6] for r in rows]
        soft_ok = np.array([a == b for a, b in zip(text, stext)], bool)
        conf = np.array([r[2] for r in rows], np.float32)
        tmp = dst.with_suffix(".tmp.npz")
        np.savez(tmp, text=json.dumps(text, ensure_ascii=False), stext=json.dumps(stext, ensure_ascii=False),
                 conf=conf, T=T, off=off, kid=kid, kp=kp, soft_ok=soft_ok,
                 info=json.dumps({"teacher": V6S.name, "sha256": got, "shard": str(path), "lines": len(rows),
                                  "K": K, "made": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))
        tmp.replace(dst)
        print(f"{path.name}: {len(rows)} lines in {time.time() - t0:.0f}s, soft_ok {soft_ok.mean():.3f}, "
              f"conf min p50 {np.median(conf[:, 0]) if len(conf) else 0:.3f} -> {dst}", flush=True)


def check(args):
    """Teacher readings against a labelled shard's labels (eval's scoring normalisation)."""
    sys.path.insert(0, str(ROOT / "eval"))
    from metrics import lev, norm
    z = np.load(args.targets)
    text = json.loads(str(z["text"]))
    stext = json.loads(str(z["stext"]))
    conf = z["conf"]
    labels = json.loads(str(np.load(args.shard)["labels"]))[:len(text)]
    rows = []
    for t, s, c, lab in zip(text, stext, conf, labels):
        if not lab:
            continue
        rows.append((lev(norm(t), norm(lab)), lev(norm(s), norm(lab)), len(norm(lab)), c[0]))
    e = np.array(rows, np.float64)
    print(f"lines {len(e)}  natural CER {e[:, 0].sum() / e[:, 2].sum():.4f} exact {(e[:, 0] == 0).mean():.4f}"
          f"  aligned CER {e[:, 1].sum() / e[:, 2].sum():.4f} exact {(e[:, 1] == 0).mean():.4f}")
    for th in (0.0, 0.3, 0.5, 0.7, 0.8, 0.9):
        m = e[:, 3] >= th
        if m.any():
            print(f"  conf_min >= {th}: keep {m.mean():.3f}  natural CER {e[m, 0].sum() / e[m, 2].sum():.4f}"
                  f"  exact {(e[m, 0] == 0).mean():.4f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("shards", nargs="+")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--workers", type=int, default=int(os.environ.get("OMP_NUM_THREADS") or 1))
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--force", action="store_true")
    c = sub.add_parser("check")
    c.add_argument("targets", type=Path)
    c.add_argument("--shard", type=Path, required=True)
    a = ap.parse_args()
    {"run": run, "check": check}[a.cmd](a)


if __name__ == "__main__":
    main()
