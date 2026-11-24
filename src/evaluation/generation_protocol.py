"""Fixed cohort, model inputs, fingerprints, and resumable record validation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ASSETS = REPO / "assets/generation"
SEEDS = (1234, 2345, 3456)
MODES = (0, 1, 2)
SAMPLE_COUNT = 5000
PROMPT_TOKENS = 64
MAX_NEW_TOKENS = 512
TOKENIZER_ID = "HuggingFaceTB/SmolLM2-135M"
TOKENIZER_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
SCORER_ID = "openai-community/gpt2-large"
SCORER_REVISION = "32b71b12589c2f8d625668d2335a01cac3249519"
VERSIONS = {
    "torch": "2.8.0+cu128",
    "transformers": "5.17.0",
    "numpy": "2.5.4",
    "scikit-learn": "1.9.1",
    "mauve-text": "0.4.0",
    "faiss-cpu": "1.12.0",
    "threadpoolctl": "3.7.0",
}
MAUVE = dict(
    num_buckets=500,
    pca_max_data=-1,
    kmeans_num_redo=5,
    kmeans_max_iter=500,
    seed=0,
    mauve_scaling_factor=5,
)
SCORER_PROTOCOL = dict(
    model_id=SCORER_ID,
    revision=SCORER_REVISION,
    hidden_size=1280,
    max_text_length=1024,
    feature="last_layer_last_nonpadding_token",
    precision="float32",
    text="prompt_plus_continuation",
    mauve=MAUVE,
)
MAUVE_SOURCES = {
    "compute_mauve.py": "5d78227f6f232ea0a87283750a22e556f013fc3337a4df3e8606af434af3c0cc",
    "utils.py": "1cc2e9f0c383b4e82949649a350b2395c112018ffd3b211742a212034a5d20f1",
}


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_manifest(path):
    cohort = json.loads((ASSETS / "cohort.json").read_text())
    if file_sha(path) != cohort["manifest_sha256"]:
        raise ValueError(
            "Manifest does not match the fixed 5,000-prompt cohort; run prepare_generation.py"
        )
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    if [r["sample_id"] for r in rows] != list(range(SAMPLE_COUNT)) or any(
        len(r["prompt_ids"]) != PROMPT_TOKENS
        or len(r["reference_ids"]) != MAX_NEW_TOKENS
        for r in rows
    ):
        raise ValueError("Invalid prompt/reference token lengths or sample IDs")
    return rows


def check_versions():
    actual = {}
    for name, expected in VERSIONS.items():
        try:
            actual[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                f"Missing {name}; install requirements-generation.txt"
            ) from exc
        if actual[name] != expected:
            raise RuntimeError(f"{name}=={expected} is required, found {actual[name]}")
    spec = importlib.util.find_spec("mauve")
    root = Path(spec.origin).parent
    for name, expected in MAUVE_SOURCES.items():
        if file_sha(root / name) != expected:
            raise RuntimeError(
                f"MAUVE source differs from the pinned official implementation: {name}"
            )
    return actual


def source_fingerprints():
    files = set((REPO / "src/evaluation").glob("*.py"))
    files.update((REPO / "src/model").glob("*.py"))
    files.update((REPO / "src/inference").glob("generation*.py"))
    files.add(REPO / "src/inference/cuda_generation.py")
    files.update((REPO / "src/inference/csrc").glob("*.cu"))
    for name in (
        "src/__init__.py",
        "src/inference/__init__.py",
        "scripts/evaluate_generation.py",
        "scripts/prepare_generation.py",
        "scripts/generate_135m.sh",
    ):
        files.add(REPO / name)
    return {str(p.relative_to(REPO)): file_sha(p) for p in sorted(files)}


def runtime_env():
    return {
        **os.environ,
        "CUDA_DEVICE_MAX_CONNECTIONS": "32",
        "CUDA_DEVICE_MAX_COPY_CONNECTIONS": "32",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONUNBUFFERED": "1",
    }


def load_checkpoint(path, device, *, expected_recurrence=None, expected_step=None):
    import torch
    from src.model.config import LoopLMConfig
    from src.model.looplm import LoopLM, model_kwargs_from_checkpoint

    checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    if expected_step is not None and checkpoint.get("step") != expected_step:
        raise ValueError(
            f"Expected final step {expected_step}, got {checkpoint.get('step')}"
        )
    cfg = LoopLMConfig(**checkpoint["model_config"])
    kwargs = model_kwargs_from_checkpoint(checkpoint)
    # Older checkpoint metadata explicitly spelled out these fixed defaults.
    if kwargs.pop("exit_head_use_final_norm", False):
        raise ValueError("The exit head must read the raw hidden state")
    if not kwargs.pop("grt_use_anchor", True):
        raise ValueError("GRT requires the fixed embedding anchor")
    supported = set(inspect.signature(LoopLM).parameters) - {"config"}
    unknown = set(kwargs) - supported
    if unknown:
        raise ValueError(f"Unsupported checkpoint options: {sorted(unknown)}")
    if any(
        kwargs.get(k, False)
        for k in ("use_postnorm", "use_q_act", "use_moe_recurrence")
    ):
        raise ValueError(
            "Generation supports standard sandwich-normalized Vanilla and GRT models"
        )
    recurrence = "grt" if kwargs.get("use_grt") else "vanilla"
    stored_recurrence = checkpoint.get("trainer_config", {}).get(
        "recurrence", recurrence
    )
    if stored_recurrence == "naive":
        stored_recurrence = "vanilla"
    if stored_recurrence != recurrence or expected_recurrence not in (None, recurrence):
        raise ValueError("Checkpoint recurrence does not match the requested model")
    if cfg.max_recurrent_steps != 8:
        raise ValueError("This evaluation protocol requires eight recurrent depths")
    model = LoopLM(cfg, **kwargs)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    metadata = dict(
        step=checkpoint.get("step"),
        recurrence=recurrence,
        model_config=cfg.__dict__,
        model_kwargs=kwargs,
    )
    del checkpoint
    return model.eval().to(device=device, dtype=torch.float32), metadata


def worker_protocol(experiment, mode, seed):
    if mode not in MODES or seed not in SEEDS:
        raise ValueError("Unknown mode or sampling seed")
    return dict(experiment_sha256=experiment["fingerprint"], mode=mode, seed=seed)


def load_completed(path, fingerprint, *, repair_tail=False):
    """Reject mixed/duplicate records; only a worker may repair its torn last line."""
    path = Path(path)
    if not path.exists():
        return {}
    done = {}
    with path.open("r+b" if repair_tail else "rb") as stream:
        while True:
            start = stream.tell()
            line = stream.readline()
            if not line:
                break
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                if not repair_tail or stream.read(1):
                    raise ValueError(f"Corrupt generation record: {path}")
                stream.seek(start)
                stream.truncate()
                stream.flush()
                os.fsync(stream.fileno())
                break
            if row["experiment_fingerprint"] != fingerprint:
                raise ValueError(f"Refusing mixed-protocol resume: {path}")
            if row["sample_id"] in done:
                raise ValueError(f"Duplicate sample in {path}")
            done[row["sample_id"]] = row
            if not line.endswith(b"\n"):
                if not repair_tail:
                    raise ValueError(f"Unterminated final record: {path}")
                stream.write(b"\n")
                stream.flush()
                os.fsync(stream.fileno())
    return done


def load_records(output, experiment, mode, seed):
    rows = []
    fingerprint = json_sha(worker_protocol(experiment, mode, seed))
    for rank in range(experiment["protocol"]["world_size"]):
        stem = Path(output) / "generations" / f"mode{mode}_seed{seed}_rank{rank}"
        config = json.loads(stem.with_suffix(".config.json").read_text())
        if config["fingerprint"] != fingerprint or config[
            "protocol"
        ] != worker_protocol(experiment, mode, seed):
            raise ValueError("Generation shard protocol mismatch")
        shard = load_completed(stem.with_suffix(".jsonl"), fingerprint)
        if set(shard) != set(
            range(rank, SAMPLE_COUNT, experiment["protocol"]["world_size"])
        ):
            raise ValueError(f"Incomplete/incorrect shard: {stem}")
        rows.extend(shard.values())
    rows.sort(key=lambda r: r["sample_id"])
    if [r["sample_id"] for r in rows] != list(range(SAMPLE_COUNT)):
        raise ValueError("Expected exactly 5,000 unique samples")
    return rows
