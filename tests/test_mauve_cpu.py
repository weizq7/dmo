from pathlib import Path

import numpy as np
import pytest

from src.evaluation import mauve_cpu as cpu


def test_reuse_requires_equal_values_shape_and_dtype(monkeypatch):
    base = np.arange(24, dtype=np.float32).reshape(6, 4)
    changed = base.copy()
    changed[0, 0] += 1e-5
    jobs = [
        ("original", base),
        ("copy", np.asfortranarray(base)),
        ("changed", changed),
        ("dtype", base.astype(np.float64)),
        ("shape", base.reshape(3, 8)),
    ]
    expected = dict(original="original", copy="original", changed="changed",
                    dtype="dtype", shape="shape")
    unique, aliases = cpu.unique_jobs(jobs)
    assert len(unique) == 4 and aliases == expected
    # A digest collision alone must never cause score reuse.
    monkeypatch.setattr(cpu, "feature_key", lambda array: "collision")
    unique, aliases = cpu.unique_jobs(jobs)
    assert len(unique) == 4 and aliases == expected


@pytest.mark.parametrize(
    "array",
    [np.zeros((0, 2)), np.zeros((2, 0)), np.zeros(2),
     np.array([[np.nan]]), np.array([[np.inf]]), np.array([["text"]])],
)
def test_invalid_features_are_rejected(array):
    with pytest.raises(ValueError, match="features"):
        cpu.feature_key(array)


@pytest.mark.parametrize("quota,expected", [("max 100000", 12), ("250000 100000", 2)])
def test_cpu_budget_respects_affinity_and_quota(monkeypatch, quota, expected):
    monkeypatch.setattr(cpu.os, "sched_getaffinity", lambda pid: set(range(12)))
    monkeypatch.setattr(Path, "read_text", lambda path: quota)
    assert cpu.available_cpus() == expected


def test_parallel_plan_caps_cpu_use_and_honors_overrides(monkeypatch):
    monkeypatch.setattr(cpu, "available_cpus", lambda: 12)
    monkeypatch.delenv("MAUVE_WORKERS", raising=False)
    monkeypatch.delenv("MAUVE_THREADS", raising=False)
    assert cpu.execution_plan(9, workers=9, threads=8)["workers"] == 1
    assert cpu.execution_plan(9, workers=9, threads=4)["workers"] == 3
    assert cpu.execution_plan(1, threads=40)["threads_per_worker"] == 12
    monkeypatch.setenv("MAUVE_WORKERS", "2")
    monkeypatch.setenv("MAUVE_THREADS", "3")
    plan = cpu.execution_plan(9)
    assert plan["workers"] == 2 and plan["threads_per_worker"] == 3
    for workers, threads in [(0, 8), (1, 0), (-1, 8)]:
        with pytest.raises(ValueError):
            cpu.execution_plan(9, workers, threads)
    with pytest.raises(ValueError):
        cpu.execution_plan(0)


def test_invalid_jobs_fail_before_launch(monkeypatch):
    features = np.ones((2, 2))
    with pytest.raises(ValueError, match="Duplicate"):
        cpu.unique_jobs([("a", features), ("a", features)])
    with pytest.raises(ValueError, match="dimensions"):
        cpu.score_feature_sets(features, [("a", np.ones((2, 3)))], {})
    monkeypatch.setenv("MAUVE_DISTANCE_BACKEND", "typo")
    with pytest.raises(ValueError, match="BACKEND"):
        cpu.score_feature_sets(features, [("a", features)], {})


@pytest.mark.parametrize("backend", ["blas", None], ids=["original_index", "default_replicas"])
def test_parallel_scores_match_official_complete_datasets(monkeypatch, backend):
    faiss = pytest.importorskip("faiss")
    mauve = pytest.importorskip("mauve")
    from threadpoolctl import threadpool_limits

    rng = np.random.default_rng(42)
    reference = rng.normal(size=(160, 12)).astype(np.float32)
    a = rng.normal(size=(160, 12)).astype(np.float32)
    b = (rng.normal(size=(160, 12)) + 1).astype(np.float32)
    options = dict(num_buckets=4, pca_max_data=-1, kmeans_num_redo=2,
                   kmeans_max_iter=8, seed=0, mauve_scaling_factor=5)
    if backend is None:
        monkeypatch.delenv("MAUVE_DISTANCE_BACKEND", raising=False)
    else:
        monkeypatch.setenv("MAUVE_DISTANCE_BACKEND", backend)
    old_kmeans = faiss.Kmeans
    old_threshold = faiss.cvar.distance_compute_blas_threshold
    old_threads = faiss.omp_get_max_threads()
    try:
        faiss.omp_set_num_threads(2)
        with threadpool_limits(limits=1, user_api="blas"):
            expected = {}
            for name, features in (("a", a), ("b", b)):
                result = mauve.compute_mauve(
                    p_features=reference, q_features=features, **options
                )
                expected[name] = dict(mauve=float(result.mauve),
                                      mauve_star=float(result.mauve_star))
        scores, runtime = cpu.score_feature_sets(
            reference, [("a", a), ("copy", a.copy()), ("b", b)], options,
            workers=2, threads=2,
        )
        assert scores == {"a": expected["a"], "copy": expected["a"], "b": expected["b"]}
        assert runtime["distance_backend"] == (backend or "replicas")
        assert runtime["unique_jobs"] == 2
        assert runtime["reused_jobs"] == {"copy": "a"}
        assert set(runtime["score_seconds"]) == {"a", "b"}
        assert scores["a"] is not scores["copy"]
        # FAISS changes are confined to spawned scoring processes.
        assert faiss.Kmeans is old_kmeans
        assert faiss.cvar.distance_compute_blas_threshold == old_threshold
        assert faiss.omp_get_max_threads() == 2
    finally:
        faiss.omp_set_num_threads(old_threads)
