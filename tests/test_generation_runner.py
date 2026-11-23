import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from src.evaluation import generation_runner as runner
from src.evaluation.generation_protocol import REPO, json_sha


def test_plan_has_exactly_nine_generation_jobs():
    plan = runner.job_plan(8)
    jobs = [job for job in plan if job["action"] == "generate"]
    assert [(j["seed"], j["mode"]) for j in jobs] == [
        (s, m) for s in (1234, 2345, 3456) for m in (0, 1, 2)
    ]
    assert len(jobs) * 5000 == 45000
    assert plan[-2]["action"] == "features" and plan[-1]["action"] == "score"


def test_gpu_selection_rejects_duplicate_or_incomplete_devices():
    inventory = [dict(index=0, uuid="GPU-a"), dict(index=1, uuid="GPU-b")]
    assert runner.select_gpus(inventory, "1,0", 2) == inventory[::-1]
    assert runner.select_gpus(inventory, "GPU-a", 1) == inventory[:1]
    for selection in ("0,0", "0", "0,2", ""):
        with pytest.raises(ValueError):
            runner.select_gpus(inventory, selection, 2)


def test_changed_source_or_protocol_cannot_resume(monkeypatch):
    protocol = dict(source_sha256={"file": "old"})
    experiment = dict(protocol=protocol, fingerprint=json_sha(protocol))
    monkeypatch.setattr(runner, "source_fingerprints", lambda: {"file": "new"})
    with pytest.raises(RuntimeError, match="source changed"):
        runner.verify_experiment(experiment)
    experiment["protocol"]["extra"] = True
    with pytest.raises(ValueError, match="fingerprint"):
        runner.verify_experiment(experiment)


@pytest.mark.parametrize("contaminate", [False, True])
def test_lifecycle_finishes_all_stages_or_stops_own_workers(
    tmp_path, monkeypatch, contaminate
):
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "prepared.json").write_text("{}")
    output = tmp_path / "output"
    args = argparse.Namespace(
        checkpoint=str(checkpoint),
        inputs=str(inputs),
        output=str(output),
        dry_run=False,
        workers=2,
        gpus=None,
        label="test",
        recurrence="vanilla",
        expected_step=22000,
    )
    inventory = [
        dict(index=i, uuid=f"GPU-{i}", utilization=0, memory_mib=0) for i in range(2)
    ]
    monkeypatch.setattr(runner, "gpu_inventory", lambda: inventory)
    monkeypatch.setattr(
        runner,
        "build_experiment",
        lambda *a: dict(fingerprint="fixed", protocol=dict(world_size=2)),
    )
    monkeypatch.setattr(runner, "verify_experiment", lambda *a: None)
    monkeypatch.setattr(runner, "load_records", lambda *a: range(5000))
    active = {"action": None}
    launched, stopped = [], []
    monkeypatch.setattr(
        runner,
        "gpu_processes",
        lambda uuids: (
            [dict(pid=999999, gpu_uuid="GPU-0")]
            if contaminate and active["action"] == "generate"
            else []
        ),
    )
    monkeypatch.setattr(runner, "is_owned", lambda pid, roots: pid in roots)
    monkeypatch.setattr(
        runner,
        "stop_children",
        lambda children: stopped.extend(child.pid for child in children),
    )

    class Child:
        pid = 12345
        returncode = 0

        def poll(self):
            return 0

    def launch(command, **kwargs):
        action = command[command.index("--action") + 1]
        active["action"] = action
        launched.append(action)
        if action == "score":
            (output / "summary.json").write_text(
                json.dumps(dict(per_seed=[dict(n=5000)] * 9))
            )
        return Child()

    monkeypatch.setattr(runner.subprocess, "Popen", launch)
    if contaminate:
        with pytest.raises(RuntimeError, match="Foreign GPU"):
            runner.run(args)
        assert json.loads((output / "state.json").read_text())["status"] == "failed"
        assert (output / "timing_invalidated.json").exists()
        assert "score" not in launched
        assert set(stopped) == {12345}
    else:
        runner.run(args)
        state = json.loads((output / "state.json").read_text())
        assert state["status"] == "complete" and state["generation_records"] == 45000
        assert launched.count("generate") == 18
        assert launched[-1] == "score"
        assert len(state["completed_jobs"]) == 13


@pytest.mark.parametrize(
    "run_key,run_name,recurrence",
    [
        ("grt_dmo", "fineweb_grt135m_loop8_mixture_22k_seed42", "grt"),
        ("grt_soft_entropy", "fineweb_grt135m_loop8_soft_entropy_22k_seed42", "grt"),
        ("vanilla_dmo", "fineweb_naive135m_loop8_mixture_22k_seed42", "vanilla"),
        (
            "vanilla_soft_entropy",
            "fineweb_naive135m_loop8_soft_entropy_22k_seed42",
            "vanilla",
        ),
    ],
)
def test_four_one_command_entrypoints(tmp_path, run_key, run_name, recurrence):
    checkpoint = tmp_path / "runs" / run_name / "checkpoints/step_0022000.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "prepared.json").write_text("{}")
    env = dict(
        os.environ,
        PYTHON=sys.executable,
        RUN_ROOT=str(tmp_path / "runs"),
        GENERATION_DATA=str(inputs),
        GENERATION_ROOT=str(tmp_path / "results"),
    )
    result = subprocess.run(
        ["bash", "scripts/generate_135m.sh", run_key, "--dry-run"],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    plan = json.loads(result.stdout)
    assert plan["checkpoint"] == str(checkpoint)
    assert plan["recurrence"] == recurrence
    assert plan["generations"] == 45000
    assert plan["output"] == str(tmp_path / "results" / run_key)
    assert not (tmp_path / "results").exists()
