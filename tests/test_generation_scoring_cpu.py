import copy
import json

import numpy as np
import pytest

from src.evaluation import generation_scoring as scoring
from src.evaluation.generation_metrics import generation_metrics
from src.evaluation.generation_protocol import source_fingerprints
from tests.test_generation_protocol import record


def test_parallel_aggregation_keeps_each_modes_timing_and_compute(tmp_path, monkeypatch):
    faiss = pytest.importorskip("faiss")
    mauve = pytest.importorskip("mauve")
    from threadpoolctl import threadpool_limits

    rng = np.random.default_rng(5)
    reference, a, b = [rng.normal(size=(80, 12)).astype(np.float32) for _ in range(3)]
    features = {"reference": reference}
    lengths = np.full(80, 576)
    rows_by_mode = {}
    for mode in scoring.MODES:
        rows = []
        for sample_id in range(80):
            row = record()
            row.update(sample_id=sample_id, full_text=f"sample {sample_id}")
            row["decode_visible_seconds"] /= mode + 1
            row["inter_token_seconds"] = [x / (mode + 1) for x in row["inter_token_seconds"]]
            if mode == 0:
                row["work"].update(foreground_decode_loops=16, background_decode_loops=0)
            if mode == 2:
                row["work"].update(decode_loops=4, background_decode_loops=0,
                                   matmul_flops=400, kv_copy_bytes=64)
                row["decode_matmul_flops"] = 100
            rows.append(row)
        rows_by_mode[mode] = rows
    for seed in scoring.SEEDS:
        for mode in scoring.MODES:
            features[f"mode{mode}_seed{seed}"] = (a if mode != 2 else b).copy()
    options = dict(num_buckets=4, pca_max_data=-1, kmeans_num_redo=2,
                   kmeans_max_iter=8, seed=0, mauve_scaling_factor=5)
    protocol = {**scoring.SCORER_PROTOCOL, "mauve": options}
    monkeypatch.setattr(scoring, "SCORER_PROTOCOL", protocol)
    monkeypatch.setattr(scoring, "load_manifest", lambda path: [
        dict(human_text=f"reference {i}") for i in range(80)
    ])
    monkeypatch.setattr(scoring, "load_records", lambda output, experiment, mode, seed:
                        copy.deepcopy(rows_by_mode[mode]))
    monkeypatch.setattr(scoring, "read_features", lambda output, experiment, name, texts:
                        (features[name], lengths))
    monkeypatch.setenv("MAUVE_WORKERS", "2")
    monkeypatch.setenv("MAUVE_THREADS", "2")
    monkeypatch.delenv("MAUVE_DISTANCE_BACKEND", raising=False)
    experiment = dict(fingerprint="fixture", paths=dict(manifest="fixture"),
                      protocol=dict(label="Test model", recurrence="vanilla"))
    old_threads = faiss.omp_get_max_threads()
    try:
        faiss.omp_set_num_threads(1)
        with threadpool_limits(limits=1, user_api="blas"):
            expected = [mauve.compute_mauve(p_features=reference, q_features=q, **options)
                        for q in (a, b)]
        scoring.aggregate_job(experiment, tmp_path)
    finally:
        faiss.omp_set_num_threads(old_threads)
    summary = json.loads((tmp_path / "summary.json").read_text())
    runtime = json.loads((tmp_path / "aggregation_runtime.json").read_text())
    assert len(summary["per_seed"]) == runtime["jobs"] == 9
    assert runtime["unique_jobs"] == 2 and len(runtime["reused_jobs"]) == 7
    assert summary["scorer"] == protocol
    for row in summary["per_seed"]:
        mode = row["mode"]
        result = expected[0 if mode != 2 else 1]
        assert row["mauve"] == float(result.mauve)
        assert row["mauve_star"] == float(result.mauve_star)
        for key, value in generation_metrics(rows_by_mode[mode]).items():
            assert row[key] == value
        assert row["decode_speedup_vs_mode0"] == pytest.approx(mode + 1)
        assert row["decode_compute_ratio_vs_mode0"] == (0.5 if mode == 2 else 1)
        assert json.loads((tmp_path / f"mode{mode}_seed{row['seed']}_metrics.json").read_text()) == row
    assert "passed for 240 paired prompts" in (tmp_path / "RESULTS.md").read_text()


def test_scorer_is_included_in_experiment_fingerprint():
    assert "src/evaluation/mauve_cpu.py" in source_fingerprints()
