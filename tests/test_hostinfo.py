"""train/hostinfo.py and the writers of result files: a generic host label, never the machine's network name."""
import hashlib
import json
import platform
import re
import socket
import sys
from pathlib import Path

import numpy as np
import pytest

import hostinfo

ROOT = Path(__file__).resolve().parents[1]
SECRET = "private-node.corp.example"  # what platform.node(), gethostname() and getfqdn() return below
# This machine's real names too (when long enough to be unambiguous): no writer may record them either.
REAL = {n for n in (platform.node(), socket.gethostname(), socket.getfqdn())
        if len(n) >= 6 and "localhost" not in n and n not in hostinfo.host_label()}


@pytest.fixture
def private_name(monkeypatch):
    """The machine's network name, as the standard library would report it."""
    monkeypatch.setattr(platform, "node", lambda: SECRET)
    monkeypatch.setattr(socket, "gethostname", lambda: SECRET)
    monkeypatch.setattr(socket, "getfqdn", lambda *a: SECRET)
    return SECRET


def names_no_machine(record):
    text = json.dumps(record, ensure_ascii=False)
    return SECRET not in text and not any(n in text for n in REAL)


def test_label_is_the_cpu_count_and_model(tmp_path, monkeypatch):
    info = tmp_path / "cpuinfo"
    info.write_text("processor\t: 0\nvendor_id\t: AuthenticAMD\nmodel name\t: AMD EPYC  9R14\n\n"
                    "processor\t: 1\nmodel name\t: AMD EPYC 9R14\n")
    monkeypatch.setattr(hostinfo.os, "cpu_count", lambda: 64)
    assert hostinfo.cpu_model(info) == "AMD EPYC 9R14"
    assert hostinfo.host_label(info) == "64-core AMD EPYC 9R14 host"


def test_label_without_a_cpu_model_is_build_host(tmp_path, monkeypatch):
    info = tmp_path / "cpuinfo"
    info.write_text("processor\t: 0\nFeatures\t: fp asimd\n")    # some ARM kernels give no model name
    assert hostinfo.host_label(info) == hostinfo.FALLBACK == "build-host"
    assert hostinfo.host_label(tmp_path / "missing") == "build-host"
    info.write_text("model name\t: AMD EPYC 9R14\n")
    monkeypatch.setattr(hostinfo.os, "cpu_count", lambda: None)
    assert hostinfo.host_label(info) == "build-host"


def test_label_on_this_machine(private_name):
    label = hostinfo.host_label()
    assert re.fullmatch(r"\d+-core \S.* host|build-host", label) and names_no_machine(label)


def test_eval_run_records_the_label(private_name, tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / "eval"))
    import run as eval_run
    import sets
    from PIL import Image
    monkeypatch.setattr(sets, "SETS", dict(sets.SETS))
    monkeypatch.setattr(sets, "LOCAL_MANIFESTS", {})
    image = tmp_path / "capture.png"
    Image.fromarray(np.full((64, 96, 3), 30, np.uint8)).save(image)
    row = {"id": "capture-1", "vn": "v4", "image": image.name, "box": [8, 8, 40, 16], "region": [0, 0, 96, 64],
           "text": "あいう", "split": "dev", "session": "A", "hand_checked": True,
           "sha256": hashlib.sha256(image.read_bytes()).hexdigest()}
    manifest = tmp_path / "local.jsonl"
    manifest.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    res = eval_run.main(["--model", "cmd:sed s/.*/あ/", "--set", "private-dev", "--manifest", str(manifest),
                         "--data", str(tmp_path)])
    assert res["sets"]["private-dev"]["all"]["lines"] == 1
    assert res["env"]["host"] == hostinfo.host_label() and names_no_machine(res)


def test_vocr3_report_environment_records_the_label(private_name):
    sys.path.insert(0, str(ROOT / "eval"))
    import vocr3_report as R
    env = R.environment()
    assert env["host"] == hostinfo.host_label() and names_no_machine(env)
    assert "host" not in R.stable_environment()


def test_hiragana_fresh_environment_records_the_label(private_name):
    import hiragana_fresh as HF
    env = HF.host_environment()
    assert env["host"] == hostinfo.host_label() and names_no_machine(env)


def test_train2_execution_identity_records_the_label(private_name, tmp_path):
    import train2 as T2
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("あ\nい\n", encoding="utf-8")
    identity = T2.execution_identity(vocab, ["あ", "い"])
    assert identity["host"]["node"] == hostinfo.host_label() and names_no_machine(identity)
