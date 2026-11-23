"""Pinned GPT-2-large features and official MAUVE aggregation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .generation_metrics import text_hash, generation_metrics, mean_sd, display_mean_sd
from .generation_protocol import (
    SEEDS,
    MODES,
    SAMPLE_COUNT,
    SCORER_PROTOCOL,
    atomic_json,
    load_manifest,
    load_records,
)


def scorer_model(experiment):
    import torch
    from transformers import AutoModel, AutoTokenizer

    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    path = experiment["paths"]["scorer"]
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    model = (
        AutoModel.from_pretrained(path, local_files_only=True, dtype=torch.float32)
        .eval()
        .to("cuda")
    )
    return tokenizer, model


def encode_batch(tokenizer, model, texts):
    import torch

    encoded = tokenizer(texts, add_special_tokens=False, truncation=False)["input_ids"]
    lengths = [len(ids) for ids in encoded]
    if not all(lengths):
        raise ValueError("Cannot featurize an empty prompt-plus-continuation")
    clipped = [ids[: SCORER_PROTOCOL["max_text_length"]] for ids in encoded]
    inputs = tokenizer.pad(
        {"input_ids": clipped}, padding=True, return_tensors="pt"
    ).to("cuda")
    with torch.inference_mode():
        states = model(**inputs, use_cache=False, return_dict=True).last_hidden_state
        vectors = states[
            torch.arange(len(texts), device="cuda"), inputs.attention_mask.sum(1) - 1
        ]
        vectors = vectors.float().cpu().numpy()
    if vectors.shape != (len(texts), 1280) or not np.isfinite(vectors).all():
        raise RuntimeError("Invalid GPT-2-large feature vectors")
    return vectors, lengths


def check_scorer(experiment, output):
    import torch

    tokenizer, model = scorer_model(experiment)
    texts = [
        "A short test.",
        "This is a synthetic feature extraction compatibility check.",
    ]
    together, _ = encode_batch(tokenizer, model, texts)
    differences = []
    for index, text in enumerate(texts):
        single, _ = encode_batch(tokenizer, model, [text])
        torch.testing.assert_close(
            torch.from_numpy(together[index]),
            torch.from_numpy(single[0]),
            atol=5e-4,
            rtol=1e-4,
        )
        differences.append(float(np.max(np.abs(together[index] - single[0]))))
    atomic_json(
        Path(output) / "scorer_check.json",
        dict(
            passed=True,
            padding_max_abs_differences=differences,
            experiment_fingerprint=experiment["fingerprint"],
        ),
    )


def feature_job(experiment, output, rank):
    output = Path(output)
    world_size = experiment["protocol"]["world_size"]
    tokenizer, model = scorer_model(experiment)
    manifest = load_manifest(experiment["paths"]["manifest"])
    features = output / "features"
    features.mkdir(exist_ok=True)
    jobs = [("reference", None, None)] + [
        (f"mode{m}_seed{s}", m, s) for s in SEEDS for m in MODES
    ]
    for name, mode, seed in jobs:
        texts = (
            [r["human_text"] for r in manifest]
            if mode is None
            else [r["full_text"] for r in load_records(output, experiment, mode, seed)]
        )
        ids = np.arange(rank, SAMPLE_COUNT, world_size, dtype=np.int64)
        shard_texts = [texts[i] for i in ids]
        digest = text_hash(shard_texts)
        path = features / f"{name}_rank{rank}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as data:
                validate_feature_shard(data, ids, digest, experiment["fingerprint"])
            print(f"FEATURES_ALREADY_COMPLETE {name} rank={rank}", flush=True)
            continue
        vectors, lengths = [], []
        batch_size = experiment["protocol"]["feature_batch_size"]
        for start in range(0, len(shard_texts), batch_size):
            batch_vectors, batch_lengths = encode_batch(
                tokenizer, model, shard_texts[start : start + batch_size]
            )
            vectors.append(batch_vectors)
            lengths.extend(batch_lengths)
            if start % 80 == 0:
                print(
                    f"FEATURES {name} rank={rank} {start + len(batch_lengths)}/{len(shard_texts)}",
                    flush=True,
                )
        temporary = path.with_suffix(".tmp.npz")
        np.savez(
            temporary,
            ids=ids,
            features=np.concatenate(vectors),
            token_lengths=np.asarray(lengths),
            text_sha256=digest,
            experiment_fingerprint=experiment["fingerprint"],
        )
        temporary.replace(path)


def validate_feature_shard(data, ids, digest, fingerprint):
    if (
        str(data["text_sha256"]) != digest
        or str(data["experiment_fingerprint"]) != fingerprint
        or not np.array_equal(data["ids"], ids)
        or data["features"].shape != (len(ids), 1280)
        or data["token_lengths"].shape != (len(ids),)
        or not np.isfinite(data["features"]).all()
        or np.any(data["token_lengths"] <= 0)
    ):
        raise ValueError(
            "Feature shard is incomplete or belongs to different texts/protocol"
        )


def read_features(output, experiment, name, texts):
    vectors, lengths, all_ids = [], [], []
    world_size = experiment["protocol"]["world_size"]
    for rank in range(world_size):
        ids = np.arange(rank, SAMPLE_COUNT, world_size, dtype=np.int64)
        with np.load(
            Path(output) / "features" / f"{name}_rank{rank}.npz", allow_pickle=False
        ) as data:
            validate_feature_shard(
                data, ids, text_hash([texts[i] for i in ids]), experiment["fingerprint"]
            )
            vectors.append(data["features"])
            lengths.append(data["token_lengths"])
            all_ids.append(ids)
    order = np.argsort(np.concatenate(all_ids))
    return np.concatenate(vectors)[order], np.concatenate(lengths)[order]


def paired_check(baseline, exact):
    if len(baseline) != len(exact):
        raise ValueError("Mode 0/1 sample counts differ")
    mismatches = [
        a["sample_id"]
        for a, b in zip(baseline, exact)
        if any(a[k] != b[k] for k in ("sample_id", "token_ids", "exit_depths"))
    ]
    if mismatches:
        raise ValueError(
            f"Coupled mode 0/1 equivalence failed at sample IDs {mismatches[:20]}"
        )
    return dict(coupled_sequence_mismatches=0, checked_samples=len(baseline))


def summarize(scores, equivalence, experiment):
    fields = [
        "mauve",
        "repeat_4gram",
        "mean_generated_tokens",
        "sampled_mean_depth",
        "sampled_decode_mean_depth",
        "decode_ms_per_token",
        "decode_speedup_vs_mode0",
        "decode_compute_ratio_vs_mode0",
        "mean_service_seconds",
        "matmul_tflops_per_sample",
        "decode_matmul_gflops_per_generated_token",
        "executed_loops_per_decode_input",
        "foreground_loops_per_decode_input",
        "background_loops_per_decode_input",
    ]
    for seed in SEEDS:
        baseline = next(r for r in scores if r["seed"] == seed and r["mode"] == 0)
        for row in [r for r in scores if r["seed"] == seed]:
            latency = row["decode_ms_per_token"]
            row["decode_speedup_vs_mode0"] = (
                baseline["decode_ms_per_token"] / latency
                if latency and baseline["decode_ms_per_token"] is not None
                else None
            )
            compute = baseline["decode_matmul_gflops_per_generated_token"]
            row["decode_compute_ratio_vs_mode0"] = (
                row["decode_matmul_gflops_per_generated_token"] / compute
                if compute
                else None
            )
    grouped = {
        str(mode): {
            key: mean_sd([r[key] for r in scores if r["mode"] == mode])
            for key in fields
        }
        for mode in MODES
    }
    return dict(
        per_seed=scores,
        three_seed_summary=grouped,
        coupled_mode0_mode1_checks=equivalence,
        experiment_fingerprint=experiment["fingerprint"],
        model=experiment["protocol"]["label"],
        recurrence=experiment["protocol"]["recurrence"],
        scorer=SCORER_PROTOCOL,
        flops_definition="Executed dense GEMM-equivalent FLOPs: QKVO, SwiGLU, QK/AV, vocabulary and exit heads; GRT adds 10*hidden_size^2 per loop-token. Excludes elementwise operations, memory movement, and scheduling.",
        notes=[
            "Decode latency = 1000 * total time between first and last emitted token / total (generated_tokens - 1).",
            "Prompt prefill, TTFT, and final background drain are reported separately. Service time includes drain.",
            "Mode 0 and mode 1 both complete all eight loops; actual loop counters include background work.",
            "Mean exit depth includes the first emitted token. Sampled decode depth and actual loops exclude prefill.",
            "Decode GEMM FLOPs per generated token include the first-token readout and divide by generated_tokens.",
            "Means and sample standard deviations are across three sampling seeds, not prompt-level confidence intervals.",
            "Speedups use this checkpoint's own mode 0. Do not compare absolute latency across different machines.",
        ],
    )


def write_report(output, summary):
    output = Path(output)
    atomic_json(output / "summary.json", summary)
    checks = summary["coupled_mode0_mode1_checks"]
    checked = sum(row["checked_samples"] for row in checks)
    equivalence = (
        f"passed for {checked:,} paired prompts"
        if checks and all(row["coupled_sequence_mismatches"] == 0 for row in checks)
        else "not verified"
    )
    lines = [
        f"# {summary['model']}: text-continuation comparison",
        "",
        "Mean ± sample SD over 3 sampling seeds; 5,000 identical prompts per seed and mode.",
        "",
        "| Mode | MAUVE ↑ | Decode ms/token ↓ | Speedup vs mode 0 | Mean exit depth | Executed loops/decode input | Decode GEMM GFLOPs/token ↓ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    names = {
        0: "0 — Full depth",
        1: "1 — Exact early emission",
        2: "2 — Early exit + KV reuse",
    }
    fields = (
        "mauve",
        "decode_ms_per_token",
        "decode_speedup_vs_mode0",
        "sampled_mean_depth",
        "executed_loops_per_decode_input",
        "decode_matmul_gflops_per_generated_token",
    )
    for mode in MODES:
        group = summary["three_seed_summary"][str(mode)]
        lines.append(
            f"| {names[mode]} | "
            + " | ".join(display_mean_sd(group[k]) for k in fields)
            + " |"
        )
    lines += [
        "",
        f"Mode 0/1 token and exit-depth equivalence: {equivalence}.",
        "",
        "## Per-seed results",
        "",
        "| Mode | Seed | MAUVE | Decode ms/token | Mean exit depth | Executed loops/decode input |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["per_seed"]:
        cells = [
            "N/A" if row[k] is None else f"{row[k]:.6f}"
            for k in (
                "mauve",
                "decode_ms_per_token",
                "sampled_mean_depth",
                "executed_loops_per_decode_input",
            )
        ]
        lines.append(f"| {row['mode']} | {row['seed']} | " + " | ".join(cells) + " |")
    lines += [
        "",
        *["- " + note for note in summary["notes"]],
        "",
        summary["flops_definition"],
        "",
        "See `summary.json` for per-seed lengths, repetition, TTFT, drain, service time, and feature truncation rates.",
    ]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")


def aggregate_job(experiment, output):
    import faiss
    import mauve

    faiss.omp_set_num_threads(8)
    output = Path(output)
    manifest = load_manifest(experiment["paths"]["manifest"])
    p, p_lengths = read_features(
        output, experiment, "reference", [r["human_text"] for r in manifest]
    )
    scores, equivalence = [], []
    for seed in SEEDS:
        baseline = load_records(output, experiment, 0, seed)
        exact = load_records(output, experiment, 1, seed)
        equivalence.append(dict(seed=seed, **paired_check(baseline, exact)))
        for mode in MODES:
            rows = (
                baseline
                if mode == 0
                else (
                    exact if mode == 1 else load_records(output, experiment, mode, seed)
                )
            )
            name = f"mode{mode}_seed{seed}"
            q, q_lengths = read_features(
                output, experiment, name, [r["full_text"] for r in rows]
            )
            print(f"SCORING {name}", flush=True)
            result = mauve.compute_mauve(
                p_features=p, q_features=q, verbose=False, **SCORER_PROTOCOL["mauve"]
            )
            row = dict(
                mode=mode,
                seed=seed,
                n=len(rows),
                mauve=float(result.mauve),
                mauve_star=float(result.mauve_star),
                **generation_metrics(rows),
                reference_feature_truncation_rate=float(np.mean(p_lengths > 1024)),
                generated_feature_truncation_rate=float(np.mean(q_lengths > 1024)),
            )
            scores.append(row)
    summary = summarize(scores, equivalence, experiment)
    for row in scores:
        atomic_json(output / f"mode{row['mode']}_seed{row['seed']}_metrics.json", row)
    write_report(output, summary)
    print("SCORING_COMPLETE", flush=True)
