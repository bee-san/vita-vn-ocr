"""Generated, offline evidence for the prospective generic VN selector."""

import copy
import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
import group_metrics as GM  # noqa: E402

import general_vn_select as GS  # noqa: E402

REAL_ENVIRONMENT_IDENTITY = GS.environment_identity


def fake_compiler(directory):
    compiler = directory / "generated-compiler"
    log = directory / "compiler-invocations.jsonl"
    variation = directory / "compiler-output-variation"
    compiler.write_text(
        "\n".join(
            [
                f"#!{sys.executable}",
                "import json, os, sys",
                "from pathlib import Path",
                "if '--version' in sys.argv:",
                "    print('generated compiler 1.0')",
                "    raise SystemExit(0)",
                f"log = Path({str(log)!r})",
                "with log.open('a') as stream:",
                "    stream.write(json.dumps({'argv': sys.argv[1:], 'env': dict(os.environ)}) + '\\n')",
                "output = Path(sys.argv[sys.argv.index('-o') + 1])",
                "source = next(Path(arg) for arg in sys.argv[1:] if arg.endswith('.c'))",
                f"variation = Path({str(variation)!r})",
                "output.write_bytes(b'generated scalar library\\n' + source.read_bytes() +",
                "                   (variation.read_bytes() if variation.exists() else b''))",
                "",
            ]
        )
    )
    compiler.chmod(0o755)
    return compiler, log, variation


def runtime_sources(workspace):
    runtime = workspace / "runtime"
    runtime.mkdir()
    (runtime / "vocr.c").write_text("int generated_runtime(void) { return 1; }\n")
    (runtime / "vocr.h").write_text("int generated_runtime(void);\n")


def row(platform="psp", vn="v1", frame=0, **changes):
    identity = f"{platform}-{vn}-{frame}"
    return {
        "id": identity,
        "platform": platform,
        "vn": vn,
        "dims": [480, 272] if platform == "psp" else [960, 544],
        "split": "dev",
        "hand_checked": True,
        "session": f"generated-{platform}-{vn}",
        "release": f"generated-release-{platform}-{vn}",
        "sha256": hashlib.sha256(identity.encode()).hexdigest(),
        "styles": ["outlined"],
        "text": "あ" * 40 + "漢" * 60,
        "pred": "あ" * 40 + "漢" * 60,
        "box": [8, 8, 32, 16],
        **changes,
    }


def quota_rows():
    return [row(platform, f"v{vn}", frame) for platform in ("psp", "vita") for vn in (1, 2, 3) for frame in range(5)]


def scores(errors=10, chars=100):
    groups = {}
    for platform in ("psp", "vita"):
        for vn in ("v1", "v2"):
            group = GM.group_score([row(platform, vn)])[f"{platform}:{vn}"]
            for part in ("all", "body"):
                for script in ("full", "hira"):
                    group[part][script] = {"chars": chars, "errors": errors}
            groups[f"{platform}:{vn}"] = group
    return {
        "groups": groups,
        "synthetic": {
            "book-dev": {"full": {"chars": chars, "errors": errors}, "hira": {"chars": chars, "errors": errors}}
        },
    }


def candidate(name, base, gain=1, step=500):
    result = {"checkpoint": name, "step": step, "scores": copy.deepcopy(base)}
    for group in result["scores"]["groups"].values():
        for part in ("all", "body"):
            for script in ("full", "hira"):
                group[part][script]["errors"] -= gain
    return result


@pytest.mark.parametrize("part,script", [("all", "full"), ("all", "hira"), ("body", "full"), ("body", "hira")])
def test_each_platform_vn_guard_blocks_one_regression_despite_other_gains(part, script):
    base = scores()
    proposed = candidate("generated", base, gain=3)
    proposed["scores"]["groups"]["psp:v1"][part][script]["errors"] = 11
    reasons = GS.qualify(base, proposed["scores"])
    assert any(f"psp:v1/{part}/{script}" in reason for reason in reasons)
    baseline = {"scores": base}
    assert GS.choose(baseline, [proposed]) == (baseline, False)
    assert not proposed["qualified"] and not proposed["eligible"]


@pytest.mark.parametrize("script", ["full", "hira"])
def test_synthetic_guards_cannot_be_bought_with_real_vn_gains(script):
    base = scores()
    proposed = candidate("generated", base, gain=3)
    proposed["scores"]["synthetic"]["book-dev"][script]["errors"] = 11
    assert any(f"book-dev/{script}" in reason for reason in GS.qualify(base, proposed["scores"]))


@pytest.mark.parametrize("change", ["group", "vn", "provenance", "reference", "lines", "chars", "negative", "bool"])
def test_qualification_binds_group_reference_and_counter_identity(change):
    base = scores()
    got = copy.deepcopy(base)
    group = got["groups"]["psp:v1"]
    if change == "group":
        got["groups"].pop("vita:v2")
    elif change == "vn":
        group["vn"] = "v9"
    elif change == "provenance":
        group["provenance"][0]["session"] = "unbound-session"
    elif change == "reference":
        group["reference_sha256"] = "0" * 64
    elif change == "lines":
        group["body"]["lines"] += 1
    elif change == "chars":
        group["all"]["hira"]["chars"] += 1
    elif change == "negative":
        group["all"]["hira"]["errors"] = -1
    else:
        group["all"]["hira"]["errors"] = False
    assert GS.qualify(base, got)


def test_balanced_rate_weights_consoles_and_vns_equally_not_characters():
    groups = {}
    for platform, vn, errors, chars in [("psp", "v1", 1, 10), ("psp", "v2", 3, 10), ("vita", "v3", 1, 100000)]:
        groups[f"{platform}:{vn}"] = {
            "platform": platform,
            "all": {"hira": {"errors": errors, "chars": chars}, "full": {"errors": errors, "chars": chars}},
        }
    expected = (Fraction(1, 10) + Fraction(3, 10)) / 4 + Fraction(1, 100000) / 2
    assert GM.balanced_rate(groups) == expected
    assert GM.balanced_rate(groups, "full") == expected
    assert expected != Fraction(5, 100020)
    with pytest.raises(ValueError, match="both consoles"):
        GM.balanced_rate({key: value for key, value in groups.items() if value["platform"] == "psp"})


def test_exact_fraction_rank_defeats_display_float_ties_and_is_deterministic():
    chars = 2**60
    base = scores(errors=chars // 2, chars=chars)
    worse = candidate("a-earlier", base, gain=1, step=1)
    better = candidate("z-later", base, gain=2, step=999)
    assert float(GM.balanced_rate(worse["scores"]["groups"])) == float(GM.balanced_rate(better["scores"]["groups"]))
    baseline = {"scores": base}
    assert GS.choose(baseline, [worse, better]) == (better, True)
    assert GS.choose(baseline, [better, worse]) == (better, True)
    equal_a = candidate("a-path", base, gain=2, step=500)
    equal_z = candidate("z-path", base, gain=2, step=500)
    assert GS.choose(baseline, [equal_z, equal_a]) == (equal_a, True)
    assert GS.choose(baseline, [candidate("same", base, gain=0)]) == (baseline, False)


def test_strict_gain_needs_two_distinct_vns_not_two_versions_of_one_game():
    base = scores()
    proposed = candidate("one-game", base, gain=0)
    for platform in ("psp", "vita"):
        proposed["scores"]["groups"][f"{platform}:v1"]["all"]["hira"]["errors"] -= 1
    baseline = {"scores": base}
    assert GS.choose(baseline, [proposed]) == (baseline, False)
    assert proposed["qualified"] and proposed["gain_vns"] == ["v1"] and not proposed["eligible"]
    proposed["scores"]["groups"]["vita:v2"]["all"]["hira"]["errors"] -= 1
    assert GS.choose(baseline, [proposed]) == (proposed, True)
    assert proposed["gain_vns"] == ["v1", "v2"]


def test_console_balanced_full_rate_breaks_hiragana_ties_before_total_errors():
    base = scores()
    for part in ("all", "body"):
        base["groups"]["psp:v1"][part]["full"]["chars"] = 10
        base["groups"]["psp:v2"][part]["full"]["chars"] = 10000
    macro_winner = candidate("z-macro", base, step=999)
    total_winner = candidate("a-total", base, step=1)
    for part in ("all", "body"):
        macro_winner["scores"]["groups"]["psp:v1"][part]["full"]["errors"] = 8
        total_winner["scores"]["groups"]["psp:v2"][part]["full"]["errors"] = 0
    macro = macro_winner["scores"]["groups"]
    total = total_winner["scores"]["groups"]
    assert GM.balanced_rate(macro) == GM.balanced_rate(total)
    assert GM.balanced_rate(macro, "full") < GM.balanced_rate(total, "full")
    assert sum(group["all"]["full"]["errors"] for group in macro.values()) > sum(
        group["all"]["full"]["errors"] for group in total.values()
    )
    assert GS.choose({"scores": base}, [total_winner, macro_winner]) == (macro_winner, True)


def test_group_scoring_separates_body_and_keeps_small_kana_errors():
    rows = [
        row(text="っが！", pred="つか!"),
        row(frame=1, text="あい", pred="", styles=["name"]),
        row(frame=2, text="う", pred="", content_kind="name"),
        row(frame=3, text="お", pred="", uncertain=True),
    ]
    group = GM.group_score(rows)["psp:v1"]
    assert group["all"]["full"] == {"chars": 6, "errors": 5}
    assert group["body"]["full"] == {"chars": 3, "errors": 2}
    assert group["body"]["hira"] == {"chars": 2, "errors": 2}
    assert group["all"]["lines"] == 3 and group["body"]["lines"] == 1
    assert [entry["id"] for entry in group["provenance"]] == [entry["id"] for entry in rows[:3]]
    changed = GM.group_score([dict(rows[0], text="つが！")])["psp:v1"]
    assert changed["reference_sha256"] != GM.group_score(rows[:1])["psp:v1"]["reference_sha256"]


@pytest.mark.parametrize(
    "platform,dims",
    [
        ("PSP", [480, 272]),
        ("vita", [480, 272]),
        (None, [480, 272]),
        ("psp", [480.0, 272]),
        ("psp", "480,272"),
        ("psp", [480, 272, 1]),
    ],
)
def test_explicit_platform_typos_and_dimension_conflicts_are_rejected(platform, dims):
    with pytest.raises(ValueError, match="platform|dimensions"):
        GM.normalize_row(row(platform=platform, dims=dims))
    normalized = GM.normalize_row(row(platform="psv", dims=[960, 544]))
    assert normalized["platform"] == "vita"


@pytest.mark.parametrize("deficit", ["vns", "frames", "body", "hiragana"])
def test_readiness_enforces_every_per_console_and_per_vn_quota(deficit):
    rows = quota_rows()
    ready = GM.readiness(rows)
    assert ready["ready"] and ready["vns_per_console"] == {"psp": 3, "vita": 3}
    assert ready["distinct_vns"] == 3
    target = [entry for entry in rows if entry["platform"] == "psp" and entry["vn"] == "v1"]
    if deficit == "vns":
        rows = [entry for entry in rows if not (entry["platform"] == "psp" and entry["vn"] == "v3")]
    elif deficit == "frames":
        target[-1]["sha256"] = target[0]["sha256"]
    elif deficit == "body":
        target[-1]["text"] = "あ" * 40 + "漢" * 59
    else:
        target[-1]["text"] = "あ" * 39 + "漢" * 61
    result = GM.readiness(rows)
    assert not result["ready"] and any("psp" in reason for reason in result["reasons"])
    if deficit != "vns":
        assert any("psp:v1" in reason for reason in result["reasons"])


def test_name_rows_uncertainty_and_repeated_lines_cannot_rescue_body_readiness():
    rows = quota_rows()
    target = [entry for entry in rows if entry["platform"] == "psp" and entry["vn"] == "v1"]
    target[0]["styles"] = ["name"]
    rows.extend([row(frame=100, styles=["name"], text="あ" * 1000), row(frame=101, uncertain=True, text="あ" * 1000)])
    result = GM.readiness(rows)
    assert not result["ready"]
    assert result["groups"]["psp:v1"]["body_chars"] == 400
    assert result["groups"]["psp:v1"]["body_hiragana"] == 160
    with pytest.raises(ValueError, match="duplicate"):
        GM.readiness(rows + [dict(rows[0])])
    lowered = dict(GM.MINIMUMS, body_chars_per_vn=499)
    with pytest.raises(ValueError, match="cannot be lowered"):
        GM.readiness(quota_rows(), lowered)


@pytest.mark.parametrize("split", ["train", "test", None])
def test_manifest_role_rejected_before_any_image_bytes(tmp_path, monkeypatch, split):
    manifest = tmp_path / "generated.jsonl"
    manifest.write_text(json.dumps(row(split=split, image="missing.png")) + "\n")
    monkeypatch.setattr(GS, "sha256", lambda path: pytest.fail("non-dev manifest reached source bytes"))
    with pytest.raises(ValueError, match="development-only"):
        GS.read_rows([manifest])


@pytest.mark.parametrize("change", [{"hand_checked": False}, {"session": None}])
def test_unreviewed_or_sessionless_manifest_rejected_before_image_bytes(tmp_path, monkeypatch, change):
    manifest = tmp_path / "generated.jsonl"
    manifest.write_text(json.dumps(row(image="missing.png", **change)) + "\n")
    monkeypatch.setattr(GS, "sha256", lambda path: pytest.fail("unreviewed manifest reached source bytes"))
    with pytest.raises(ValueError, match="reviewed labels and sessions"):
        GS.read_rows([manifest])


@pytest.mark.parametrize(
    "box",
    [
        None,
        "8,8,32,16",
        [],
        [8, 8, 32],
        [8, 8, 32, 16, 1],
        [8.0, 8, 32, 16],
        [True, 8, 32, 16],
        [-1, 8, 32, 16],
        [8, -1, 32, 16],
        [8, 8, 0, 16],
        [8, 8, 32, -1],
        [479, 8, 2, 16],
        [8, 271, 32, 2],
    ],
)
def test_malformed_or_out_of_bounds_ink_box_rejected_before_image_bytes(tmp_path, monkeypatch, box):
    manifest = tmp_path / "generated.jsonl"
    manifest.write_text(json.dumps(row(image="must-not-read.png", box=box)) + "\n")

    def hash_manifest_only(path):
        assert Path(path).resolve() == manifest.resolve(), "invalid ink box reached source-image bytes"
        return hashlib.sha256(manifest.read_bytes()).hexdigest()

    monkeypatch.setattr(GS, "sha256", hash_manifest_only)
    with pytest.raises(ValueError, match="integer ink boxes|bounds"):
        GS.read_rows([manifest])


@pytest.fixture
def package(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    for name in ("generated-code.py", "requirements.lock", "requirements-ci.lock"):
        (workspace / name).write_text("generated immutable identity\n")
    runtime_sources(workspace)
    compiler, compiler_log, variation = fake_compiler(tmp_path)
    monkeypatch.setattr(GS, "ROOT", workspace)
    monkeypatch.setattr(GS, "CODE", ["generated-code.py"])
    monkeypatch.setattr(GS.export, "read_blob", lambda path: {})
    monkeypatch.setattr(GS, "validate_blob", lambda blob: None)
    monkeypatch.setattr(GS, "NativeVocr", lambda *args: pytest.fail("generated identity tests reached OCR"))
    environment = {
        "python": {"executable": "generated-python", "version": "generated"},
        "packages": {"torch": "generated-torch", "numpy": "generated-numpy", "Pillow": "generated-pillow"},
    }
    monkeypatch.setattr(GS, "environment_identity", lambda: copy.deepcopy(environment))
    baseline = tmp_path / "generated-baseline.vocr"
    baseline.write_bytes(b"generated baseline identity; never a model")
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps(GS.PROTOCOL))
    records = quota_rows()
    for index, record in enumerate(records):
        path = tmp_path / f"generated-{index}.png"
        width, height = record["dims"]
        Image.fromarray(np.full((height, width, 3), index + 10, np.uint8)).save(path)
        record.update(image=path.name, sha256=GS.sha256(path))
        record.pop("pred")
    manifest = tmp_path / "dev.jsonl"
    manifest.write_text("".join(json.dumps(record) + "\n" for record in records))
    synthetic = tmp_path / "synthetic.npz"
    np.savez(
        synthetic,
        labels=json.dumps(["あ"]),
        meta=json.dumps([{"split": "dev", "book_split_sha256": "a" * 64}]),
        data=np.zeros(6, np.uint8),
        offsets=[0, 6],
        shapes=[[2, 3]],
    )
    return {
        "root": tmp_path,
        "workspace": workspace,
        "baseline": baseline,
        "protocol": protocol,
        "manifest": manifest,
        "synthetic": synthetic,
        "out": tmp_path / "prepared.json",
        "rows": records,
        "compiler": compiler,
        "compiler_log": compiler_log,
        "variation": variation,
    }


def prepare(package):
    return GS.prepare(
        package["protocol"],
        package["baseline"],
        [package["manifest"]],
        {"book-dev": package["synthetic"]},
        package["out"],
        compiler=str(package["compiler"]),
    )


def test_prepare_freezes_development_identities_and_refuses_semantic_rewrite(package):
    identity = prepare(package)
    raw = package["out"].read_bytes()
    assert identity["readiness"]["ready"] and not identity["test_or_google_evaluated"]
    assert not identity["candidates_evaluated"] and identity["protocol"] == GS.PROTOCOL
    assert identity["baseline_sha256"] == GS.sha256(package["baseline"])
    assert prepare(package) == identity and package["out"].read_bytes() == raw
    assert GS.verify_lock(package["out"])[0] == identity
    with pytest.raises(ValueError, match="frozen"):
        GS.frozen_file(package["out"], dict(identity, baseline_sha256="0" * 64))
    assert package["out"].read_bytes() == raw


@pytest.mark.parametrize("changed", ["baseline", "protocol", "manifest", "image", "synthetic", "code", "dependency"])
def test_prepared_identity_rejects_mutated_bytes_before_selection(package, monkeypatch, changed):
    prepare(package)
    path = {
        "baseline": package["baseline"],
        "protocol": package["protocol"],
        "manifest": package["manifest"],
        "image": package["root"] / package["rows"][0]["image"],
        "synthetic": package["synthetic"],
        "code": package["workspace"] / "generated-code.py",
        "dependency": package["workspace"] / "requirements.lock",
    }[changed]
    path.write_bytes(path.read_bytes() + b"\ngenerated change")
    monkeypatch.setattr(GS, "load_pixels", lambda *args: pytest.fail("changed input reached pixels"))
    monkeypatch.setattr(GS, "score_blob", lambda *args: pytest.fail("changed input reached scoring"))
    with pytest.raises(ValueError, match="changed"):
        GS.select(package["out"], [], package["root"] / "selection.json")


@pytest.mark.parametrize("swapped", ["manifest", "synthetic"])
def test_lock_cannot_redirect_to_an_unbound_same_readiness_input(package, swapped):
    lock = prepare(package)
    original = package[swapped]
    alternate = original.with_name("unbound-" + original.name)
    alternate.write_bytes(original.read_bytes())
    if swapped == "manifest":
        lock["manifests"] = [str(alternate.resolve())]
    else:
        lock["synthetic"]["book-dev"] = str(alternate.resolve())
    package["out"].write_text(json.dumps(lock))
    with pytest.raises(ValueError, match="bound|input|identity|frozen"):
        GS.verify_lock(package["out"])


@pytest.mark.parametrize("changed", ["code_sha256", "dependency_sha256", "candidates_evaluated"])
def test_lock_requires_complete_identity_namespaces_and_prospective_role(package, changed):
    lock = prepare(package)
    if changed == "candidates_evaluated":
        lock[changed] = True
    else:
        lock[changed].pop(next(iter(lock[changed])))
    package["out"].write_text(json.dumps(lock))
    with pytest.raises(ValueError, match="lock|identity|namespace|code|dependenc|prospective"):
        GS.verify_lock(package["out"])


def test_insufficient_readiness_is_reported_and_blocks_selection_before_pixels(package, monkeypatch):
    package["manifest"].write_text(json.dumps(package["rows"][0]) + "\n")
    identity = prepare(package)
    assert not identity["readiness"]["ready"] and identity["readiness"]["reasons"]
    monkeypatch.setattr(GS, "load_pixels", lambda *args: pytest.fail("not-ready cohort reached pixels"))
    monkeypatch.setattr(GS, "score_blob", lambda *args: pytest.fail("not-ready cohort reached scoring"))
    with pytest.raises(ValueError, match="not ready"):
        GS.select(package["out"], [], package["root"] / "selection.json")


def test_synthetic_metadata_rejects_nonhex_book_identity(tmp_path):
    path = tmp_path / "synthetic.npz"
    np.savez(path, labels=json.dumps(["あ"]), meta=json.dumps([{"split": "dev", "book_split_sha256": "z" * 64}]))
    with pytest.raises(ValueError, match="cohort|provenance"):
        GS.synthetic_metadata(path)


def test_completed_selection_rejected_before_reopening_any_data(tmp_path, monkeypatch):
    out = tmp_path / "selection.json"
    out.write_text("generated completed selection\n")
    monkeypatch.setattr(GS, "verify_lock", lambda *args: pytest.fail("completed selection reopened data"))
    with pytest.raises(FileExistsError, match="completed"):
        GS.select(tmp_path / "absent.json", [], out)


def test_selection_publishes_frozen_model_checkpoint_and_cohort_identities(package, monkeypatch):
    lock = prepare(package)
    checkpoints = [package["root"] / "b-checkpoint.pt", package["root"] / "a-checkpoint.pt"]
    for index, path in enumerate(checkpoints):
        path.write_bytes(f"generated checkpoint {index}".encode())
    base = scores()
    proposed = candidate("unused", base)["scores"]
    calls = []

    def score(path, rows, crops, synthetic, runtime):
        assert runtime == lock["runtime"]
        calls.append(str(path))
        return copy.deepcopy(base if str(path) == str(package["baseline"]) else proposed)

    def export(path, reference, directory):
        path = Path(path)
        blob = directory / (path.stem + ".vocr")
        blob.write_bytes(b"generated blob " + path.read_bytes())
        return {
            "checkpoint": str(path),
            "checkpoint_sha256": GS.sha256(path),
            "step": 500,
            "blob": str(blob),
            "blob_sha256": GS.sha256(blob),
        }

    monkeypatch.setattr(GS, "score_blob", score)
    monkeypatch.setattr(GS, "export_checkpoint", export)
    monkeypatch.setattr(GS, "load_pixels", lambda rows, protocol: [np.zeros((2, 3), np.uint8)] * len(rows))
    monkeypatch.setattr(GS, "load_synthetic_development", lambda path: ([], {}))
    out = Path(lock["selection_out"])
    result = GS.select(package["out"], checkpoints, out)
    assert calls[0] == str(package["baseline"]) and len(calls) == 3
    assert result["strict_general_dev_improvement"] and not result["runtime_promotion"]
    assert not result["test_or_google_evaluated"]
    assert result["runtime"] == lock["runtime"] and result["environment"] == lock["environment"]
    assert result["protocol_lock_sha256"] == GS.sha256(package["out"])
    assert result["chosen"]["checkpoint"] == str(checkpoints[1])
    assert result["chosen"]["checkpoint_sha256"] == GS.sha256(checkpoints[1])
    assert result["chosen"]["blob_sha256"] == GS.sha256(result["chosen"]["blob"])
    inventory_path = package["out"].with_name(package["out"].stem + "-candidates.json")
    inventory = json.loads(inventory_path.read_text())
    assert inventory["checkpoints"] == {str(path): GS.sha256(path) for path in checkpoints}
    assert result["candidate_inventory_sha256"] == GS.sha256(inventory_path)
    assert inventory["selection_out"] == str(out) and json.loads(out.read_text()) == result
    assert (out.with_name(out.stem + "_exports") / ".gitignore").read_text() == "*\n"


@pytest.mark.parametrize("changed", ["bytes", "inventory", "destination"])
def test_interrupted_selection_cannot_change_frozen_candidates_or_destination(package, monkeypatch, changed):
    lock = prepare(package)
    checkpoint = package["root"] / "generated.pt"
    checkpoint.write_bytes(b"generated original checkpoint")
    out = Path(lock["selection_out"])

    def interrupted(*args):
        raise RuntimeError("generated interruption before scoring")

    monkeypatch.setattr(GS, "load_pixels", interrupted)
    monkeypatch.setattr(GS, "score_blob", lambda *args: pytest.fail("inventory mismatch reached scoring"))
    with pytest.raises(RuntimeError, match="interruption"):
        GS.select(package["out"], [checkpoint], out)
    inventory_path = package["out"].with_name(package["out"].stem + "-candidates.json")
    original_inventory = inventory_path.read_bytes()
    paths = [checkpoint]
    if changed == "bytes":
        checkpoint.write_bytes(b"generated altered checkpoint")
    elif changed == "inventory":
        extra = package["root"] / "unregistered.pt"
        extra.write_bytes(b"generated additional checkpoint")
        paths.append(extra)
    else:
        out = out.with_name("unregistered-output.json")
    with pytest.raises(ValueError, match="frozen|destination"):
        GS.select(package["out"], paths, out)
    assert inventory_path.read_bytes() == original_inventory
    assert not out.exists()


def test_inventory_json_mutated_during_scoring_rejected_before_publication(package, monkeypatch):
    lock = prepare(package)
    checkpoint = package["root"] / "generated.pt"
    checkpoint.write_bytes(b"generated checkpoint identity")
    checkpoint_sha = GS.sha256(checkpoint)
    lock_sha = GS.sha256(package["out"])
    inventory_path = package["out"].with_name(package["out"].stem + "-candidates.json")
    calls = []

    def score(*args):
        calls.append(args[0])
        if len(calls) == 1:
            inventory = json.loads(inventory_path.read_text())
            inventory["checkpoints"] = {}
            inventory_path.write_text(json.dumps(inventory))
        return scores()

    monkeypatch.setattr(GS, "score_blob", score)
    monkeypatch.setattr(GS, "load_pixels", lambda rows, protocol: [np.zeros((2, 3), np.uint8)] * len(rows))
    monkeypatch.setattr(GS, "load_synthetic_development", lambda path: ([], {}))
    monkeypatch.setattr(
        GS,
        "export_checkpoint",
        lambda path, reference, directory: {
            "checkpoint": str(path),
            "checkpoint_sha256": checkpoint_sha,
            "step": 500,
            "blob": str(package["baseline"]),
            "blob_sha256": GS.sha256(package["baseline"]),
        },
    )
    out = Path(lock["selection_out"])
    with pytest.raises(ValueError, match="checkpoint inventory changed"):
        GS.select(package["out"], [checkpoint], out)
    assert len(calls) == 2 and not out.exists()
    assert GS.sha256(checkpoint) == checkpoint_sha and GS.sha256(package["out"]) == lock_sha
    assert json.loads(inventory_path.read_text())["checkpoints"] == {}


@pytest.fixture
def compiler_workspace(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runtime_sources(workspace)
    compiler, log, variation = fake_compiler(tmp_path)
    monkeypatch.setattr(GS, "ROOT", workspace)
    return {
        "workspace": workspace,
        "compiler": compiler,
        "log": log,
        "variation": variation,
        "directory": tmp_path / "scalar-runtime",
    }


def test_scalar_runtime_rebuilds_stale_library_and_cache_with_fixed_scalar_flags(compiler_workspace, monkeypatch):
    fixture = compiler_workspace
    directory = fixture["directory"]
    directory.mkdir()
    library = directory / "libvocr_scalar.so"
    library.write_bytes(b"stale generated cache")
    monkeypatch.setenv("CFLAGS", "untrusted inherited compiler flags")
    first = GS.build_scalar_runtime(directory, str(fixture["compiler"]))
    second = GS.build_scalar_runtime(directory, str(fixture["compiler"]), expected=first)
    assert first == second
    assert library.read_bytes() != b"stale generated cache"
    assert first["library_sha256"] == GS.sha256(library)
    assert first["compiler"]["sha256"] == GS.sha256(fixture["compiler"])
    assert first["compiler"]["version"] == "generated compiler 1.0"
    invocations = [json.loads(line) for line in fixture["log"].read_text().splitlines()]
    assert len(invocations) == 2
    for invocation in invocations:
        assert invocation["argv"] == first["argv"][1:]
        assert "-DVOCR_NO_NEON" in invocation["argv"] and "-ffp-contract=off" in invocation["argv"]
        assert invocation["env"] == GS.BUILD_ENV and "CFLAGS" not in invocation["env"]
    assert not (directory / "libvocr_scalar.building.so").exists()


def test_mismatched_rebuild_preserves_original_library_and_cleans_staging(compiler_workspace):
    fixture = compiler_workspace
    identity = GS.build_scalar_runtime(fixture["directory"], str(fixture["compiler"]))
    library = Path(identity["library"])
    original = library.read_bytes()
    fixture["variation"].write_bytes(b"unexpected generated compiler output")
    with pytest.raises(ValueError, match="rebuilt scalar runtime differs"):
        GS.build_scalar_runtime(fixture["directory"], str(fixture["compiler"]), expected=identity)
    assert library.read_bytes() == original and GS.sha256(library) == identity["library_sha256"]
    assert not (fixture["directory"] / "libvocr_scalar.building.so").exists()
    assert len(fixture["log"].read_text().splitlines()) == 2


@pytest.mark.parametrize("phase", ["prepared", "resume"])
@pytest.mark.parametrize("drift", ["torch", "numpy", "Pillow", "python", "compiler", "library"])
def test_execution_drift_rejected_before_source_pixels_and_scoring(package, monkeypatch, phase, drift):
    lock = prepare(package)
    out = Path(lock["selection_out"])
    checkpoint = package["root"] / "generated.pt"
    checkpoint.write_bytes(b"generated checkpoint")
    inventory_path = package["out"].with_name(package["out"].stem + "-candidates.json")
    if phase == "resume":

        def interrupted(*args):
            raise RuntimeError("generated pre-pixel interruption")

        monkeypatch.setattr(GS, "load_pixels", interrupted)
        with pytest.raises(RuntimeError, match="interruption"):
            GS.select(package["out"], [checkpoint], out)
        assert inventory_path.exists()
    if drift in ("torch", "numpy", "Pillow", "python"):
        changed = copy.deepcopy(lock["environment"])
        if drift == "python":
            changed["python"]["version"] = "generated changed Python"
        else:
            changed["packages"][drift] = "generated changed package"
        monkeypatch.setattr(GS, "environment_identity", lambda: copy.deepcopy(changed))
    elif drift == "compiler":
        package["compiler"].write_text(package["compiler"].read_text() + "# generated changed compiler bytes\n")
    else:
        library = Path(lock["runtime"]["library"])
        library.write_bytes(library.read_bytes() + b"generated corruption")
    image_paths = {(package["root"] / record["image"]).resolve() for record in package["rows"]}
    original_sha = GS.sha256

    def no_image_bytes(path):
        assert Path(path).resolve() not in image_paths, "execution drift reached source-image bytes"
        return original_sha(path)

    monkeypatch.setattr(GS, "sha256", no_image_bytes)
    monkeypatch.setattr(GS, "load_pixels", lambda *args: pytest.fail("execution drift reached pixels"))
    monkeypatch.setattr(GS, "score_blob", lambda *args: pytest.fail("execution drift reached scoring"))
    with pytest.raises(ValueError, match="environment|compiler/library"):
        GS.select(package["out"], [checkpoint], out)
    assert not out.exists()
    assert inventory_path.exists() == (phase == "resume")


def test_reprepare_cannot_rebuild_away_evidence_of_original_library_corruption(package):
    lock = prepare(package)
    original_lock = package["out"].read_bytes()
    library = Path(lock["runtime"]["library"])
    library.write_bytes(b"generated corrupt original library")
    compile_log = package["compiler_log"].read_bytes()
    with pytest.raises(ValueError, match="compiler/library"):
        prepare(package)
    assert library.read_bytes() == b"generated corrupt original library"
    assert package["out"].read_bytes() == original_lock
    assert package["compiler_log"].read_bytes() == compile_log


def test_environment_identity_binds_executable_and_actual_installed_module_versions(tmp_path, monkeypatch):
    python = tmp_path / "generated-python"
    python.write_bytes(b"generated Python executable identity")
    monkeypatch.setattr(GS.sys, "executable", str(python))
    monkeypatch.setattr(GS.torch, "__version__", "generated-torch-version")
    monkeypatch.setattr(GS.np, "__version__", "generated-numpy-version")
    monkeypatch.setattr(GS.Image, "__version__", "generated-pillow-version")
    identity = REAL_ENVIRONMENT_IDENTITY()
    assert identity["python"]["executable"] == str(python)
    assert identity["python"]["sha256"] == GS.sha256(python)
    assert identity["packages"] == {
        "torch": "generated-torch-version",
        "numpy": "generated-numpy-version",
        "Pillow": "generated-pillow-version",
    }


@pytest.mark.parametrize("corrupt", [False, True])
def test_explicit_reader_binds_selected_library_and_never_default_cache(tmp_path, monkeypatch, corrupt):
    sys.path.insert(0, str(ROOT / "runtime"))
    import pyvocr

    library = tmp_path / "generated-scalar.so"
    library.write_bytes(b"generated scalar identity")
    runtime = {"library": str(library), "library_sha256": GS.sha256(library)}
    selected_library = object()
    calls = []

    def load_library(path):
        assert path == str(library)
        calls.append(("library", path))
        return selected_library

    def reader(path, **kwargs):
        assert kwargs == {"mode": "w8i", "max_width": 1024, "lib": selected_library}
        calls.append(("reader", path))
        return type("GeneratedReader", (), {"blob_bytes": GS.MODEL_BYTES, "arena_bytes": GS.ARENA_BYTES})()

    monkeypatch.setattr(pyvocr, "load_lib", load_library)
    monkeypatch.setattr(pyvocr, "Vocr", reader)
    blob = tmp_path / "nonexistent-generated-model.vocr"
    if corrupt:
        library.write_bytes(b"generated changed scalar identity")
        with pytest.raises(ValueError, match="library changed"):
            GS.NativeVocr(blob, runtime)
        assert not calls
    else:
        native = GS.NativeVocr(blob, runtime)
        assert calls == [("library", str(library)), ("reader", blob)]
        assert native.info == {"blob_bytes": GS.MODEL_BYTES, "arena_bytes": GS.ARENA_BYTES}
