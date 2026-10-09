#!/usr/bin/env python3
"""A throwaway copy of the repo whose data/ holds only CI fixtures.

render.py and train.py find data/ and models/ beside their own source (common.ROOT), so CI
runs copies of the repo's files next to a data/ that contains just:

  data/vocab.txt and the other tracked data/ files
  data/fonts/mplus1p/   the pinned OFL font from ci/fonts.py, with its OFL.txt
  data/corpus/          ci/fixtures/corpus/*.txt as the text pools, plus char_freq.json

There are no screenshots, so render.py draws procedural backgrounds. The checkout's own
data/ (often symlinks into shared, read-only data) is never written to.

  sandbox.py --out DIR [--shards N] [--per-shard M] [--seed S] [--workers W]
      builds DIR/repo and renders DIR/repo/data/synth/synth_*.npz with synth/render.py
"""
import argparse
import collections
import json
import shutil
import subprocess
import sys
from pathlib import Path

import fonts

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "ci" / "fixtures" / "corpus"
POOLS = ("dialogue", "narration", "wiki")          # render.POOL_WEIGHT


def repo_files():
    """Files a commit of this checkout would contain: tracked plus untracked-but-not-ignored."""
    out = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                         cwd=ROOT, check=True, capture_output=True).stdout
    return [p for p in sorted(set(out.decode().split("\0"))) if p and (ROOT / p).is_file()]


def fixture_lines(corpus=FIXTURES):
    return {name: [s for s in (Path(corpus) / f"{name}.txt").read_text(encoding="utf-8").split("\n") if s]
            for name in POOLS}


def check_fixture(lines, vocab, font_path):
    """Every fixture character must be a vocab class and have a glyph in the font: render.py
    rejects a draw whose text the font can't show, and a fixture it can never draw would hang."""
    from fontTools.ttLib import TTFont
    cmap = TTFont(str(font_path), lazy=True).getBestCmap() or {}
    chars = set("".join(s for pool in lines.values() for s in pool))
    bad = sorted(c for c in chars if c not in vocab or ord(c) not in cmap)
    if bad:
        raise ValueError(f"fixture characters outside the vocab or the font: {''.join(bad)}")


def build(dest, font_path, corpus=FIXTURES):
    """Copy the repo to dest (which must not exist yet) and give it a fixture-only data/."""
    dest = Path(dest)
    dest.mkdir(parents=True)
    for rel in repo_files():
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, dest / rel)
    data = dest / "data"
    vocab = {c for c in (data / "vocab.txt").read_text(encoding="utf-8").split("\n") if c}
    font_path = Path(font_path)
    fam = data / "fonts" / fonts.FAMILY
    fam.mkdir(parents=True)
    for name in fonts.FILES:
        shutil.copy2(font_path.parent / name, fam / name)
    lines = fixture_lines(corpus)
    check_fixture(lines, vocab, font_path)
    (data / "corpus").mkdir()
    freq = collections.Counter()
    for name, pool in lines.items():
        (data / "corpus" / f"{name}.txt").write_text("\n".join(pool) + "\n", encoding="utf-8")
        freq.update("".join(pool))
    (data / "corpus" / "char_freq.json").write_text(
        json.dumps(dict(freq.most_common()), ensure_ascii=False), encoding="utf-8")
    return dest


def render(repo, shards=2, per_shard=160, seed=20261009, workers=2, timeout=900):
    """Run the sandbox's synth/render.py; returns the shard paths, sorted."""
    out = Path(repo) / "data" / "synth"
    cmd = [sys.executable, str(Path(repo) / "synth" / "render.py"), "shards", "--out", str(out),
           "--shards", str(shards), "--per-shard", str(per_shard), "--seed", str(seed),
           "--workers", str(workers)]
    subprocess.run(cmd, check=True, timeout=timeout, stdin=subprocess.DEVNULL)
    paths = sorted(out.glob("synth_*.npz"))
    if len(paths) != shards:
        raise RuntimeError(f"render.py wrote {len(paths)} shards, expected {shards}")
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--font-cache", type=Path, default=fonts.DEFAULT_CACHE)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--per-shard", type=int, default=48)
    ap.add_argument("--seed", type=int, default=20261009)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    repo = build(args.out / "repo", fonts.ensure_font(args.font_cache))
    for p in render(repo, args.shards, args.per_shard, args.seed, args.workers):
        print(p)


if __name__ == "__main__":
    main()
