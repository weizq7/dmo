import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.evaluation.generation_data import read_selection, digest_text
from src.evaluation.generation_metrics import generation_metrics, mean_sd
from src.evaluation.generation_protocol import (
    ASSETS,
    file_sha,
    json_sha,
    load_checkpoint,
    load_completed,
    load_records,
    worker_protocol,
)
from src.evaluation.generation_scoring import (
    paired_check,
    validate_feature_shard,
    summarize,
    write_report,
)
from tests.test_generation_reference import tiny


def test_fixed_cohort_is_complete_and_unique():
    rows = read_selection()
    assert len(rows) == 5000
    assert len({r["document_sha256"] for r in rows}) == 5000
    assert len({(r["shard"], r["row_index"]) for r in rows}) == 5000
    assert rows[0]["shard"] == "shard_000058.parquet"
    assert rows[0]["row_index"] == 9140
    assert digest_text("a \n b") == digest_text("a b")
    meta = json.loads((ASSETS / "cohort.json").read_text())
    assert file_sha(ASSETS / "selection.csv") == meta["selection_sha256"]


@pytest.mark.parametrize("grt", [False, True])
def test_clean_and_compatible_checkpoint_loading(tmp_path, grt):
    model = tiny(grt)
    kwargs = {"use_grt": True, "grt_use_anchor": True} if grt else {}
    checkpoint = dict(
        model_config=model.config.__dict__,
        model_kwargs=kwargs,
        model_state_dict=model.state_dict(),
        step=22000,
    )
    path = tmp_path / "model.pt"
    torch.save(checkpoint, path)
    loaded, metadata = load_checkpoint(
        path,
        "cpu",
        expected_recurrence="grt" if grt else "vanilla",
        expected_step=22000,
    )
    assert metadata["recurrence"] == ("grt" if grt else "vanilla")
    for name, value in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], value, atol=0, rtol=0)
    with pytest.raises(ValueError, match="step"):
        load_checkpoint(path, "cpu", expected_step=56000)
    checkpoint["model_kwargs"]["exit_head_use_final_norm"] = True
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="raw hidden"):
        load_checkpoint(path, "cpu")


def test_resume_only_repairs_the_torn_last_record(tmp_path):
    path = tmp_path / "shard.jsonl"
    row = dict(sample_id=0, experiment_fingerprint="fixed")
    path.write_bytes((json.dumps(row) + '\n{"sample_id":').encode())
    with pytest.raises(ValueError, match="Corrupt"):
        load_completed(path, "fixed")
    assert load_completed(path, "fixed", repair_tail=True) == {0: row}
    assert path.read_text() == json.dumps(row) + "\n"
    path.write_text(json.dumps(row))
    assert load_completed(path, "fixed", repair_tail=True) == {0: row}
    assert path.read_bytes().endswith(b"\n")
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="Duplicate"):
        load_completed(path, "fixed", repair_tail=True)
    path.write_text("{bad}\n" + json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="Corrupt"):
        load_completed(path, "fixed", repair_tail=True)
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="mixed-protocol"):
        load_completed(path, "different")


def test_scoring_rejects_incomplete_shards(tmp_path):
    experiment = dict(fingerprint="fixed", protocol=dict(world_size=2))
    protocol = worker_protocol(experiment, 0, 1234)
    fingerprint = json_sha(protocol)
    directory = tmp_path / "generations"
    directory.mkdir()
    for rank in range(2):
        stem = directory / f"mode0_seed1234_rank{rank}"
        stem.with_suffix(".config.json").write_text(
            json.dumps(dict(protocol=protocol, fingerprint=fingerprint))
        )
        stem.with_suffix(".jsonl").write_text(
            "\n".join(
                json.dumps(dict(sample_id=i, experiment_fingerprint=fingerprint))
                for i in range(rank, 5000, 2)
            )
            + "\n"
        )
    assert len(load_records(tmp_path, experiment, 0, 1234)) == 5000
    (directory / "mode0_seed1234_rank1.jsonl").write_text("")
    with pytest.raises(ValueError, match="Incomplete"):
        load_records(tmp_path, experiment, 0, 1234)


def record(tokens=3, eos=False):
    intervals = [0.01] * (tokens - 1)
    return dict(
        sample_id=0,
        token_ids=list(range(tokens)),
        exit_depths=[2] * tokens,
        generated_tokens=tokens,
        eos=eos,
        inter_token_seconds=intervals,
        prefill_seconds=0.1,
        ttft_seconds=0.11,
        emission_seconds=0.11 + sum(intervals),
        decode_visible_seconds=sum(intervals),
        service_seconds=0.12 + sum(intervals),
        drain_seconds=0.01,
        decode_matmul_flops=200,
        work=dict(
            decode_loops=8 * (tokens - 1),
            foreground_decode_loops=2 * (tokens - 1),
            background_decode_loops=6 * (tokens - 1),
            matmul_flops=1000,
            kv_copy_bytes=0,
        ),
    )


def test_latency_denominators_eos_and_work():
    metrics = generation_metrics([record(), record(1, True)])
    assert metrics["decode_input_tokens"] == 2
    assert metrics["decode_ms_per_token"] == pytest.approx(10)
    assert metrics["executed_loops_per_decode_input"] == 8
    assert metrics["foreground_loops_per_decode_input"] == 2
    assert metrics["background_loops_per_decode_input"] == 6
    assert metrics["sampled_mean_depth"] == 2
    assert metrics["mean_service_seconds"] > metrics["mean_emission_seconds"]
    eos = generation_metrics([record(1, True)])
    assert eos["decode_ms_per_token"] is None and eos["iti_p95_ms"] is None
    assert mean_sd([None, None]) == dict(mean=None, std=None, n_defined=0)
    assert mean_sd([1.0, 2.0, 3.0])["std"] == 1


def test_paired_mode_check_detects_both_token_and_depth_changes():
    a = record()
    assert paired_check([a], [dict(a)])["checked_samples"] == 1
    for key in ("token_ids", "exit_depths"):
        with pytest.raises(ValueError, match="equivalence"):
            paired_check([a], [dict(a, **{key: [8, 8, 8]})])


def test_feature_fingerprints_and_shape():
    shard = dict(
        ids=np.array([0, 2]),
        features=np.zeros((2, 1280)),
        token_lengths=np.array([2, 5]),
        text_sha256=np.array("text"),
        experiment_fingerprint=np.array("experiment"),
    )
    validate_feature_shard(shard, np.array([0, 2]), "text", "experiment")
    for change in (
        dict(ids=np.array([0, 1])),
        dict(features=np.zeros((2, 4))),
        dict(text_sha256="different"),
    ):
        with pytest.raises(ValueError):
            validate_feature_shard(
                {**shard, **change}, np.array([0, 2]), "text", "experiment"
            )


def test_report_contains_three_modes_nine_scores_and_paired_speedups(tmp_path):
    scores = []
    for seed in (1234, 2345, 3456):
        for mode in (0, 1, 2):
            scores.append(
                dict(
                    mode=mode,
                    seed=seed,
                    n=5000,
                    mauve=0.5,
                    **generation_metrics([record()]),
                )
            )
    experiment = dict(
        fingerprint="example",
        protocol=dict(label="Vanilla + DMO", recurrence="vanilla"),
    )
    summary = summarize(scores, [], experiment)
    assert all(row["decode_speedup_vs_mode0"] == 1 for row in summary["per_seed"])
    assert len(summary["per_seed"]) == 9
    write_report(tmp_path, summary)
    assert set(
        json.loads((tmp_path / "summary.json").read_text())["three_seed_summary"]
    ) == {"0", "1", "2"}
    assert "Executed loops/decode input" in (tmp_path / "RESULTS.md").read_text()
    assert "equivalence: not verified" in (tmp_path / "RESULTS.md").read_text()
    summary["coupled_mode0_mode1_checks"] = [
        dict(seed=seed, coupled_sequence_mismatches=0, checked_samples=5000)
        for seed in (1234, 2345, 3456)
    ]
    write_report(tmp_path, summary)
    assert "passed for 15,000 paired prompts" in (tmp_path / "RESULTS.md").read_text()
