#!/usr/bin/env python3
"""Package textspot-desktop's pre-release (#173): the detector's ONNX and VTSD files, VOCR6.2-L's files byte for byte
from v0.5.1-onnx.2 / v0.5.0 (sha256 checked against those releases' SHA256SUMS), the Python pipeline from tools/ at
the release commit, the attribution files, the model card, a manifest and SHA256SUMS.
  package_release.py --run RUN_DIR --repo REPO --out DIR --tag TAG --card CARD.md [--size long:960]"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ART = Path("/local/home/skerraut/herd-vita-ocr/shared/artifacts")
ONNX2 = ART / "vocr-onnx" / "release"
V050 = ART / "vocr-onnx" / "v0.5.0"
HERE = Path(__file__).resolve().parent


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def sums(p):
    out = {}
    for line in Path(p).read_text(encoding="utf-8").splitlines():
        if line.strip():
            h, name = line.split(None, 1)
            out[name.strip().lstrip("*")] = h
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--card", required=True)
    ap.add_argument("--size", default="long:960")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    repo = Path(a.repo)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
    r = subprocess.run([str(repo / ".venv" / "bin" / "python"), str(repo / "train" / "tsdesk_det.py"), "export",
                        "--run", a.run, "--out", str(out), "--name", "textspot_desktop", "--size", a.size],
                       capture_output=True, text=True, check=True)
    export = json.loads(r.stdout.strip().splitlines()[-1])
    s_onnx2, s_v050 = sums(ONNX2 / "SHA256SUMS"), sums(V050 / "SHA256SUMS")
    copied = {}
    for name, src, ref in (("VOCR6.2_L_fast.onnx", ONNX2, s_onnx2), ("ONNX-ATTRIBUTION.md", ONNX2, s_onnx2),
                           ("VOCR6-ATTRIBUTION.md", ONNX2, s_onnx2), ("VOCR6.2_L_w8_2l.vocr", V050, s_v050)):
        h = sha(src / name)
        if ref.get(name) != h:
            raise SystemExit(f"{src / name}: sha256 {h} is not the release's {ref.get(name)}")
        shutil.copy2(src / name, out / name)
        copied[name] = {"from": "v0.5.1-onnx.2" if src == ONNX2 else "v0.5.0", "sha256": h}
    for f in ("tools/tsdesk.py", "tools/tsdesk_refine.py", "tools/tsdesk_c.py", "tools/vocr_onnx.py"):
        shutil.copy2(repo / f, out / Path(f).name)
    att = (HERE.parent / "release" / "TEXTSPOT-DESKTOP-ATTRIBUTION.md").read_text(encoding="utf-8").replace(
        "__TAG__", a.tag)
    (out / "TEXTSPOT-DESKTOP-ATTRIBUTION.md").write_text(att, encoding="utf-8")
    shutil.copy2(a.card, out / "MODEL_CARD_TEXTSPOT_DESKTOP.md")
    files = sorted(p.name for p in out.iterdir() if p.is_file())
    man = {"kind": "textspot-desktop-release", "tag": a.tag, "repo_commit": head, "issue": 173,
           "detector": export, "copied": copied,
           "files": {n: {"bytes": (out / n).stat().st_size, "sha256": sha(out / n)} for n in files}}
    (out / "tsdesk-manifest.json").write_text(json.dumps(man, indent=1) + "\n", encoding="utf-8")
    files = sorted(p.name for p in out.iterdir() if p.is_file())
    (out / "SHA256SUMS").write_text("".join(f"{sha(out / n)}  {n}\n" for n in files), encoding="utf-8")
    print(json.dumps({"out": str(out), "files": files, "onnx_sha256": export["onnx_sha256"],
                      "vtsd_sha256": export.get("vtsd_sha256")}))


if __name__ == "__main__":
    sys.exit(main())
