"""Official MAUVE scoring with bounded processes and parallel FAISS queries."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import math
import multiprocessing
import os
from pathlib import Path
import time

import numpy as np

_REFERENCE = None
_OPTIONS = None


def available_cpus():
    """Respect process affinity and a container CPU quota, when present."""
    count = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else (os.cpu_count() or 1)
    )
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            count = min(count, max(1, math.floor(int(quota) / int(period))))
    except (OSError, ValueError):
        pass
    return max(1, count)


def execution_plan(job_count, workers=None, threads=None):
    if job_count < 1:
        raise ValueError("At least one MAUVE job is required")
    cpus = available_cpus()
    threads = int(
        threads if threads is not None else os.environ.get("MAUVE_THREADS", "8")
    )
    requested = workers if workers is not None else os.environ.get("MAUVE_WORKERS")
    if threads < 1 or (requested is not None and int(requested) < 1):
        raise ValueError("MAUVE_WORKERS and MAUVE_THREADS must be positive")
    threads = min(threads, cpus)
    workers = min(
        job_count, max(1, cpus // threads), int(requested) if requested is not None else 6
    )
    return dict(
        workers=workers,
        threads_per_worker=threads,
        blas_threads=1,
        available_cpus=cpus,
    )


def feature_key(array):
    array = np.asarray(array)
    if (
        array.ndim != 2
        or not all(array.shape)
        or array.dtype.kind not in "fiu"
        or not np.isfinite(array).all()
    ):
        raise ValueError("MAUVE features must be nonempty finite numeric matrices")
    array = np.ascontiguousarray(array)
    return (array.shape, array.dtype.str, hashlib.sha256(array.view(np.uint8)).hexdigest())


def unique_jobs(jobs):
    """Reuse only identical matrices; digests merely accelerate candidate lookup."""
    unique, aliases, by_key = {}, {}, {}
    for name, array in jobs:
        if name in aliases:
            raise ValueError(f"Duplicate MAUVE job name: {name}")
        array = np.asarray(array)
        key = feature_key(array)
        owner = next(
            (
                other
                for other in by_key.get(key, [])
                if unique[other].shape == array.shape
                and unique[other].dtype == array.dtype
                and np.array_equal(unique[other], array)
            ),
            None,
        )
        if owner is None:
            owner = name
            unique[name] = array
            by_key.setdefault(key, []).append(name)
        aliases[name] = owner
    return unique, aliases


def _initialize(reference, options, threads, backend):
    global _REFERENCE, _OPTIONS
    omp_threads = 1 if backend == "replicas" else threads
    os.environ.update(
        OMP_NUM_THREADS=str(omp_threads),
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        OMP_WAIT_POLICY="PASSIVE",
    )
    import faiss
    import mauve  # Load numerical libraries before limiting their thread pools.
    from threadpoolctl import threadpool_limits

    threadpool_limits(limits=1)
    faiss.omp_set_num_threads(omp_threads)
    if backend == "replicas":
        original_kmeans = faiss.Kmeans

        class QueryParallelKmeans(original_kmeans):
            def set_index(self):
                super().set_index()
                if (
                    self.gpu
                    or self.cp.spherical
                    or self.cp.__class__ != faiss.ClusteringParameters
                ):
                    raise ValueError("Query parallelism requires ordinary CPU L2 clustering")
                # Every index holds ALL centroids. Only query rows are split;
                # clustering still operates on the complete feature dataset.
                self._replicas = [faiss.IndexFlatL2(self.d) for _ in range(threads)]
                self.index = faiss.IndexReplicas(self.d, True)
                for replica in self._replicas:
                    self.index.addIndex(replica)

        # Private to this spawned process; official MAUVE source, PCA, clustering
        # iterations, initializations and best-restart selection are unchanged.
        faiss.Kmeans = QueryParallelKmeans
    _REFERENCE, _OPTIONS = reference, options


def _score(name, features):
    import mauve

    started = time.monotonic()
    print(f"MAUVE_STARTED {name}", flush=True)
    result = mauve.compute_mauve(
        p_features=_REFERENCE, q_features=features, verbose=False, **_OPTIONS
    )
    seconds = time.monotonic() - started
    print(f"MAUVE_COMPLETE {name} seconds={seconds:.3f}", flush=True)
    return (
        name,
        dict(mauve=float(result.mauve), mauve_star=float(result.mauve_star)),
        seconds,
    )


def score_feature_sets(reference, jobs, options, *, workers=None, threads=None):
    """Score whole datasets concurrently, reusing only exactly identical inputs."""
    started = time.monotonic()
    reference = np.asarray(reference)
    feature_key(reference)
    unique, aliases = unique_jobs(jobs)
    if any(array.shape[1] != reference.shape[1] for array in unique.values()):
        raise ValueError("Reference and generated feature dimensions must match")
    plan = execution_plan(len(unique), workers, threads)
    backend = os.environ.get("MAUVE_DISTANCE_BACKEND", "replicas")
    if backend not in ("blas", "replicas"):
        raise ValueError("MAUVE_DISTANCE_BACKEND must be blas or replicas")
    print(
        f"MAUVE_PLAN jobs={len(aliases)} unique={len(unique)} "
        f"workers={plan['workers']} threads={plan['threads_per_worker']} backend={backend}",
        flush=True,
    )
    results, timings = {}, {}
    # Spawn avoids forking a previously initialized Torch/OpenMP runtime.
    with ProcessPoolExecutor(
        max_workers=plan["workers"],
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_initialize,
        initargs=(reference, options, plan["threads_per_worker"], backend),
    ) as pool:
        futures = [pool.submit(_score, name, features) for name, features in unique.items()]
        for future in as_completed(futures):
            name, score, seconds = future.result()
            results[name], timings[name] = score, seconds
    expanded = {name: dict(results[owner]) for name, owner in aliases.items()}
    metadata = dict(
        **plan,
        jobs=len(aliases),
        unique_jobs=len(unique),
        distance_backend=backend,
        reused_jobs={name: owner for name, owner in aliases.items() if name != owner},
        score_seconds=timings,
        wall_seconds=time.monotonic() - started,
    )
    return expanded, metadata
