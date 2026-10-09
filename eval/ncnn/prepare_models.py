#!/usr/bin/env python3
"""Set up the ncnn baseline models for eval/run.py --model ncnn:DIR.

  eval/ncnn/prepare_models.py [--out data/eval/models/ncnn] [--scratch DIR]

Copies two PP-OCRv5 mobile recognisers from the research campaign's scratch tree
($HERD/shared/research/scratch) and writes each one's dictionary (one class per
line, blank excluded) from the vocabulary array in its generated vjo_ppocr_data.h:

  ppocr-v5-mobile-shipped    the model vita-jp-overlay ships: nihui's ncnn conversion,
                             fp16-stored weights, 18,385 classes (8,242,276-byte bin)
  ppocr-v5-mobile-jis-int8   the research's recommendation: head pruned to JIS X 0208
                             (6,513 classes), squeeze-excite convs as InnerProduct,
                             ncnn2table KL calibration on Aozora lines + ncnn2int8
                             (2,749,540-byte bin; scratch/int8/sa250291)
Every copied file (param, bin, dict) must match the sha256 the baselines in
docs/results.md were measured with (PINNED below); otherwise the script exits non-zero.
The weights are Apache-2.0 PaddleOCR derivatives; keep them out of git anyway.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
HERD = Path(os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr"))
MODELS = {
    "ppocr-v5-mobile-shipped": ("eval/vita-jp-overlay/build/host/ocr-model/PP_OCRv5_mobile_rec.ncnn",
                                "eval/vita-jp-overlay/build/host/ocr-model/vjo_ppocr_data.h"),
    "ppocr-v5-mobile-jis-int8": ("int8/sa250291/models/jis_ip_aozora_int8",
                                 "acc/variants/jis/vjo_ppocr_data.h"),
}
# sha256 of the files the published baselines were measured with (the shipped bin's is the
# one vita-jp-overlay's tools/prepare_ocr.py pins). A copy that differs is refused.
PINNED = {
    "ppocr-v5-mobile-shipped": {
        "model.bin": "49d9907a55ba20fa6637f9f788f66ab00793bc8ce57a733a09dbe86b9a2e3db0",
        "model.param": "f52a6586ac3338d8c350db0c9f3c55ff2ecd3763327f9bc7f0efc091f8a63e74",
        "dict.txt": "e5f8ca61ba03d3a247d06b013119982fa6de2bd48a846018b67bca57ffc56de1",
    },
    "ppocr-v5-mobile-jis-int8": {
        "model.bin": "cd5e4a1f841f694ab238ecd39d9f114e639e77f0473c562a0d486a633a5205aa",
        "model.param": "55be70d826aded4b32dceeabe13048a738f99de38eb457ce6d70f6d810667f42",
        "dict.txt": "3ae7ab717737cdc75488235200c0d79922763a36b25c5e1a9812918fa26352d6",
    },
}
STR = re.compile(r'^"((?:[^"\\]|\\.)*)",\s*$')


def vocab_from_header(path):
    text = Path(path).read_text(encoding="utf-8")
    start = text.index("vjo_ppocr_vocab[] = {")
    out = []
    for line in text[start:].splitlines()[1:]:
        if line.startswith("}"):
            break
        m = STR.match(line.strip())
        if not m:
            raise ValueError(f"unexpected vocab line {line!r} in {path}")
        out.append(re.sub(r"\\(.)", r"\1", m.group(1)))
    return out


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "eval" / "models" / "ncnn")
    ap.add_argument("--scratch", type=Path, default=HERD / "shared" / "research" / "scratch")
    args = ap.parse_args()
    bad = []
    for name, (stem, header) in MODELS.items():
        d = args.out / name
        d.mkdir(parents=True, exist_ok=True)
        for ext in (".param", ".bin"):
            shutil.copyfile(args.scratch / (stem + ext), d / ("model" + ext))
        vocab = vocab_from_header(args.scratch / header)
        (d / "dict.txt").write_text("\n".join(vocab) + "\n", encoding="utf-8")
        info = {"source": str(args.scratch / stem), "bin_bytes": (d / "model.bin").stat().st_size,
                "bin_sha256": sha256(d / "model.bin"), "param_sha256": sha256(d / "model.param"),
                "dict_sha256": sha256(d / "dict.txt"), "classes": len(vocab) + 1}
        (d / "info.json").write_text(json.dumps(info, indent=1) + "\n", encoding="utf-8")
        print(name, info)
        bad += [f"{name}/{f}" for f, want in PINNED[name].items() if sha256(d / f) != want]
    if bad:
        raise SystemExit(f"sha256 differs from the measured baselines: {', '.join(bad)}")


if __name__ == "__main__":
    main()
