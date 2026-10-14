#!/usr/bin/env python3
"""Training-side real labels without any reserved VN, for synth3's focus targets (version 3).

  synth3_real_labels.py --real A/teacher/teacher_train.npz+A/realdata/real_train.npz \
      --forbid S3/mining/exclude_dev_vns.txt data/manifests/synth3_exclude_fresh_native.json \
      data/manifests/synth3_exclude_clannad.json --out V3/text/real_labels_v3.npz
  synth3_text.py targets --pools V3/text/targets_v3 --real V3/text/real_labels_v3.npz ...

A row is dropped when its VN is one of eval's test games (test_games.json "all"), is_test_vn, or in
any --forbid list (a .txt of VN ids or an exclusion manifest's "vns"). The output holds only the
kept labels, in the shard format's "labels" field, which is all `synth3_text.py targets` reads.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from common import is_reserved_vn  # noqa: E402


def forbidden(paths):
    out = set(json.loads((ROOT / "eval" / "manifests" / "test_games.json").read_text())["all"])
    for p in map(Path, paths):
        if p.suffix == ".json":
            out |= set(json.loads(p.read_text(encoding="utf-8"))["vns"])
        else:
            out |= set(p.read_text().split())
    return out


def keep_labels(paths, forbid):
    labels, dropped = [], 0
    for p in paths:
        z = np.load(p)
        for lab, m in zip(json.loads(str(z["labels"])), json.loads(str(z["meta"]))):
            if not lab:
                continue
            if m.get("vn") in forbid or is_reserved_vn(m.get("vn", "")):
                dropped += 1
                continue
            labels.append(lab)
    return labels, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True, help="npz files joined by '+'")
    ap.add_argument("--forbid", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    labels, dropped = keep_labels(args.real.split("+"), forbidden(args.forbid))
    np.savez(args.out, labels=np.array(json.dumps(labels, ensure_ascii=False)))
    print(f"kept {len(labels)} labels, dropped {dropped} rows of reserved VNs -> {args.out}")


if __name__ == "__main__":
    main()
