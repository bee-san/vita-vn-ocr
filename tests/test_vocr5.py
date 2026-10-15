"""VOCR5 (eval/vocr5_protocol.json): the protocol's structure, the tier shapes' exact sizes, the run script's names and
mixtures. No screenshots, checkpoints or network; the shape check builds random-init blobs (vita/model_tiers.py)."""
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
P = json.loads((ROOT / "eval" / "vocr5_protocol.json").read_text(encoding="utf-8"))
RUN = (ROOT / "eval" / "vocr5_run.sh").read_text(encoding="utf-8")
TIERS = {"Small": "S", "Normal": "N", "Large": "L"}
# the shapes of eval/vocr5_run.sh, as vita/model_tiers.py CANDIDATES entries
SHAPES = {"Small": (1.0, 96, 1, 2, 5, 4987, 32, 8, None),
          "Normal": (1.5, 96, 1, 2, 5, 5631, 32, 8, None),
          "Large": (2.0, 96, 1, 2, 5, 5631, 32, 8, (64, 128, 256, 480, 576))}


def test_tiers_limits_and_vocabs():
    for name, t in P["tiers"].items():
        e, lim = t["expected"], t["limits"]
        if "weights_lt" in lim:
            assert e["blob_bytes"] < lim["weights_lt"]
        else:
            assert e["blob_bytes"] <= lim["weights_le"]
        assert e["arena_w8i_bytes"] <= lim["arena_le"]
        v = ROOT / t["vocab"]["file"]
        assert hashlib.sha256(v.read_bytes()).hexdigest() == t["vocab"]["sha256"]
        lines = v.read_text(encoding="utf-8").splitlines()
        assert len(lines) + 1 == t["vocab"]["classes"]          # + the CTC blank
        assert lines[-1] == "〓"                                # the <unk> class of a reduced vocab
        assert t["incumbent"]["blob_bytes"] <= e["blob_bytes"]
    lim = {n: t["limits"] for n, t in P["tiers"].items()}
    assert lim["Small"]["weights_lt"] == 1_000_000
    assert lim["Normal"]["weights_le"] == 1_300_000           # conductor 10:07Z / 10:21Z: no test.3 before training
    assert lim["Large"]["weights_le"] == 2_500_000            # owner 18:09Z


def test_box_estimates_meet_the_tiers():
    tab = P["estimates"]["table"]
    for name, shape in (("Small", "s_w100v4987"), ("Normal", "n_w150v5631"), ("Large", "l_lw300v5631")):
        row, box = tab[shape], P["tiers"][name]["limits"]["box_s_le"]
        assert row["bytes"] == P["tiers"][name]["expected"]["blob_bytes"]
        assert row["macs_per_frame"] == P["tiers"][name]["expected"]["macs_per_frame"]
        for kind in ("main", "#85", "#85 no-PLD"):
            for mhz in ("333", "444"):
                ms, s = row[kind][mhz]
                assert abs(240 * ms / 1000 + 0.1 - s) < 0.002
                assert s <= box
    assert tab["n_w150v5631"]["#85 no-PLD"]["333"][1] <= 1.0   # Normal's about 1 s box at 333 MHz


def test_exported_shapes_match_the_protocol():
    pytest.importorskip("torch")
    sys.path[:0] = [str(ROOT / "vita"), str(ROOT / "train"), str(ROOT / "eval")]
    import budget_sweep as BS
    import model_tiers as MT
    for name, cfg in SHAPES.items():
        data, _ = MT.export_blob(cfg)
        info = BS.blob_info(data)
        e = P["tiers"][name]["expected"]
        assert len(data) == e["blob_bytes"], name
        assert info["arena"]["t4h16"]["w8i"] == e["arena_w8i_bytes"], name
        assert info["macs_per_frame"] == e["macs_per_frame"], name


def test_run_script_names_shapes_and_mixtures():
    assert subprocess.run(["bash", "-n", str(ROOT / "eval" / "vocr5_run.sh")]).returncode == 0
    assert "PRUN=pre5_${TIER}_seed${SEED}" in RUN
    assert "RUN=v5$(echo \"$STAGE\" | tr 'PF' 'pf')_${TIER}_seed${SEED}" in RUN
    assert "--chans 64,128,256,480,576" in RUN
    runs = P["runs"]
    for name, t in TIERS.items():
        seeds = P["tiers"][name]["seeds"]
        assert runs["pretrains"][name] == [f"pre5_{t}_seed{s}" for s in seeds]
        for arm in ("P", "F"):
            assert runs["fine_tunes"][name][arm] == [f"v5{arm.lower()}_{t}_seed{s}" for s in seeds]

    def weights(block):
        return [float(w) for w in re.findall(r'--src "[a-z0-9]+:([0-9.]+):', block)]
    pre = RUN.split('if [ "$RECIPE" = C ]; then', 1)[1].split("PRUN=", 1)[0]
    c_pre, m_pre = pre.split("else", 1)
    assert abs(sum(weights(c_pre)) - 1) < 1e-9 and abs(sum(weights(m_pre)) - 1) < 1e-9
    ft = RUN.split("# ---------------------------------------------------------------- fine-tunes", 1)[1]
    syn = ft.split('if [ "$RECIPE" = C ]; then', 1)[1].split('RUN=v5', 1)[0]
    c_ft, m_ft = syn.split("else", 1)
    assert abs(sum(weights(c_ft)) - 0.5) < 1e-9 and abs(sum(weights(m_ft)) - 0.5) < 1e-9


def test_decisions_pairs_and_cohorts():
    arms = set(P["locked_report"]["arms"])
    for a, b in P["locked_report"]["pairs"]:
        assert a in arms and b in arms and a != b
    for cand, inc in (("VOCR5_S", "VOCR4_FL10"), ("VOCR5_N", "VOCR4_H15"), ("VOCR5_L", "VOCR4_W20")):
        assert [cand, inc] in P["locked_report"]["pairs"]
    inc = {t["incumbent"]["sha256"][:8] for t in P["tiers"].values()}
    assert inc <= {v[:8] for v in P["locked_report"]["arms"].values()}
    c = P["cohorts"]
    assert len(c["tv"]["sha256"]) == 64 and len(c["dn"]["files"]) == 4 and len(c["wd"]["files"]) == 2
    assert "0.5 x (tv + dn) / 2 + 0.5 x wd" in P["score"]["dev_score"]
    assert P["amendments"] == [] or all("date" in a and "change" in a for a in P["amendments"])
