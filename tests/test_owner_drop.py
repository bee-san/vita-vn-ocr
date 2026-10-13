"""synth/owner_drop.py and realplus/shots.OWNER_ALLOWED: roles, folder names, the public table's guard and the
owner's planetarian exemption, on a tiny fake VNDB dump. Fast, offline, no frames."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for d in ("synth", "realplus", "tests"):
    if str(ROOT / d) not in sys.path:
        sys.path.insert(0, str(ROOT / d))

import shots as S  # noqa: E402

import owner_drop as OD  # noqa: E402
from common import is_test_vn  # noqa: E402
from test_realplus import train_side, write_dump  # noqa: E402


def test_decide_roles():
    assert OD.decide("v4", None, {})[0] == "test"                                    # CLANNAD: protected
    assert OD.decide("v9999999", "clannad-relation", {})[0] == "test"
    role, why = OD.decide("v34", None, {})
    assert role == "training" and "owner 2026-10-13" in why and "not reserved" in why
    role, why = OD.decide("v34", "shared-release", {"vita-test": 3})
    assert role == "blocked" and "vita-test" in why                                  # never over an eval set
    assert OD.decide("v57", "shared-release", {}) == ("test", "reserved: shared-release")
    assert OD.decide("v57", None, {"teacher_val": 5}) == ("test", "in teacher_val")
    assert OD.decide("v57", None, {}) == ("training", "not reserved, in no eval or development set")
    assert OD.decide("v57", None, {}, leak_frames=2)[0] == "blocked"
    assert OD.decide("v34", None, {}, leak_frames=1)[0] == "blocked"                 # the owner's too


def test_folder_names():
    assert OD.folder_name("psp", "planetarian", "v34", "training") == "psp-planetarian-v34"
    assert OD.folder_name("vita", "clannad", "v4", "test") == "vita-clannad-v4-TEST"
    have = ["psp-sharin-no-kuni-v57", "vita-other-v57-TEST", "psp-x-v3"]
    assert OD.folder_name("psp", "sharin", "v57", "training", have) == "psp-sharin-no-kuni-v57"   # reused
    assert OD.folder_name("psp", "sharin", "v57", "test", have) == "vita-other-v57-TEST"
    assert OD.folder_name("psp", "planetarian", "v34", "training", have) == "psp-planetarian-v34"  # not v3's


def plan(**over):
    g = {k: None for k in ("platform", "drop_title", "vn", "title", "release", "dims", "source_resolution",
                           "collections", "reserved_vns", "reserved_without_owner_allowed", "release_vns_reserved",
                           "sets", "why", "leak_max", "leak_frames", "folder")}
    g.update(frames=3, role="training", vn="v34", folder="psp-planetarian-v34")
    g.update(over)
    return {"games": [g, dict(g, vn="v4", role="test", frames=2, folder="vita-clannad-v4-TEST")]}


def test_public_table_counts_and_refuses_frame_names():
    out = OD.public(plan(), archive={"sha256": "0" * 64})
    assert out["frames"] == 5 and out["by_role"] == {"training": 3, "test": 2}
    assert [g["vn"] for g in out["games"]] == ["v34", "v4"] and "v34" in out["owner_decisions"]
    with pytest.raises(ValueError, match="names a frame"):
        OD.public(plan(drop_title="psp/Planetarian/0001.png"))
    with pytest.raises(KeyError):
        OD.public({"games": [{"vn": "v34", "frames": 1, "role": "training"}]})      # every public key is required


def test_owner_allowed_lifts_only_derived_reasons(tmp_path, monkeypatch):
    assert set(S.OWNER_ALLOWED) == {"v34"} and not is_test_vn("v34") and not S.val_bucket("v34")
    a, b = train_side(2, 700)
    vns = {v: "ja" for v in ("v4", "v34", a, b)}
    releases = {"rbox": (["win"], ["ja"], ["v4", "v34", a])}       # a compilation with CLANNAD (r3748's pattern)
    write_dump(tmp_path, vns, [], releases, relations=[("v34", b, "char")])
    d = S.load_dump(tmp_path)
    why = S.reserved_vns(d, test_all=set(), fresh=set(), dev=set())
    assert why["v4"] == "clannad" and why[a] == "shared-release" and "v34" not in why
    assert why.get(b) != "family"                                  # nothing derives from v34 either
    assert S.reserved_vns(d, test_all=set(), fresh=set(), dev={"v34"})["v34"] == "development"   # direct: kept
    monkeypatch.setattr(S, "OWNER_ALLOWED", {})
    assert S.reserved_vns(d, test_all=set(), fresh=set(), dev=set())["v34"] == "shared-release"


def test_owner_allowed_ignores_clannad_relations(tmp_path, monkeypatch):
    write_dump(tmp_path, {"v4": "ja", "v34": "ja"}, [], {}, relations=[("v4", "v34", "char")])
    d = S.load_dump(tmp_path)
    assert "v34" not in S.reserved_vns(d, test_all=set(), fresh=set(), dev=set())
    monkeypatch.setattr(S, "OWNER_ALLOWED", {})
    assert S.reserved_vns(d, test_all=set(), fresh=set(), dev=set())["v34"] == "clannad-relation"
