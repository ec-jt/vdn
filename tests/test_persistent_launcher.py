import importlib.util
import json
from pathlib import Path
import sys

import pytest


@pytest.fixture
def launcher(monkeypatch):
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    spec = importlib.util.spec_from_file_location("persistent_launcher_test", "scripts/benchmark_rtx5090_persistent.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_persistent_dry_run_selects_new_worker_without_changing_preset(launcher, capsys):
    assert launcher.main(["persistent-test-dry", "plain", "--requests", "3", "--dry-run"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert "src/inference/infer_persistent.py" in plan["command"]
    assert "src/inference/infer_ulysses.py" not in plan["command"]
    assert plan["environment"]["VDN_PERSISTENT_REQUESTS"] == "3"
    assert plan["selected"]["render"]["num_frames"] == 124
    assert plan["selected"]["render"]["num_steps"] == 8
    assert plan["selected"]["memory"]["weight_prefetch"]


@pytest.mark.parametrize("count", ["0", "9"])
def test_persistent_batch_is_bounded(launcher, count):
    with pytest.raises(SystemExit):
        launcher.main(["persistent-test-dry", "plain", "--requests", count, "--dry-run"])


def test_persistent_rejects_partial_stack(launcher):
    with pytest.raises(SystemExit):
        launcher.main(["persistent-test-dry", "plain", "--disable", "mxfp8", "--dry-run"])


def test_tp_explicit_only_and_fixed_workload(launcher,capsys):
    launcher.main(["tp-test-dry","plain","--tensor-parallel","--dry-run"])
    plan = json.loads(capsys.readouterr().out)
    assert plan["environment"]["VDN_PERMANENT_TP"] == "1"
    assert plan["selected"]["render"]["num_frames"] == 124
    assert plan["selected"]["render"]["num_steps"] == 8
    launcher.main(["old-test-dry","plain","--dry-run"])
    assert json.loads(capsys.readouterr().out)["environment"]["VDN_PERMANENT_TP"] == "0"
