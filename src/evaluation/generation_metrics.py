"""Timing, work, and text diagnostics with explicit EOS-aware denominators."""

import hashlib
import numpy as np


def text_hash(texts):
    h = hashlib.sha256()
    for s in texts:
        b = s.encode()
        h.update(len(b).to_bytes(8, "little"))
        h.update(b)
    return h.hexdigest()


def repeat_rate(ids, n):
    grams = [tuple(ids[i : i + n]) for i in range(max(0, len(ids) - n + 1))]
    return 0.0 if not grams else 1 - len(set(grams)) / len(grams)


def generation_metrics(rows):
    """Keep immediate-EOS samples; undefined decode timings are JSON null."""
    if not rows:
        raise ValueError("Cannot summarize an empty generation set")
    tokens = sum(r["generated_tokens"] for r in rows)
    if tokens < len(rows):
        raise ValueError("Every prompt must emit at least one token, including EOS")
    decode_inputs = tokens - len(rows)
    itis = np.asarray(
        [v for r in rows for v in r["inter_token_seconds"]], dtype=np.float64
    )
    if len(itis) != decode_inputs:
        raise ValueError("Inter-token timing count does not match generated lengths")
    loop_count = sum(r["work"]["decode_loops"] for r in rows)
    return {
        "generated_tokens": tokens,
        "decode_input_tokens": decode_inputs,
        "mean_generated_tokens": tokens / len(rows),
        "eos_rate": float(np.mean([r["eos"] for r in rows])),
        "sampled_mean_depth": sum(sum(r["exit_depths"]) for r in rows) / tokens,
        "executed_decode_loops": loop_count,
        "foreground_loops_per_decode_input": (
            sum(r["work"]["foreground_decode_loops"] for r in rows) / decode_inputs
            if decode_inputs
            else None
        ),
        "background_loops_per_decode_input": (
            sum(r["work"]["background_decode_loops"] for r in rows) / decode_inputs
            if decode_inputs
            else None
        ),
        "sampled_decode_mean_depth": (
            sum(sum(r["exit_depths"][1:]) for r in rows) / decode_inputs
            if decode_inputs
            else None
        ),
        "executed_loops_per_decode_input": (
            loop_count / decode_inputs if decode_inputs else None
        ),
        "decode_ms_per_token": (
            1000 * sum(r["decode_visible_seconds"] for r in rows) / decode_inputs
            if decode_inputs
            else None
        ),
        "mean_prefill_ms": 1000 * float(np.mean([r["prefill_seconds"] for r in rows])),
        "mean_ttft_ms": 1000 * float(np.mean([r["ttft_seconds"] for r in rows])),
        "mean_emission_seconds": float(np.mean([r["emission_seconds"] for r in rows])),
        "mean_service_seconds": float(np.mean([r["service_seconds"] for r in rows])),
        "mean_drain_seconds": float(np.mean([r["drain_seconds"] for r in rows])),
        "iti_p50_ms": float(np.quantile(itis, 0.5) * 1000) if itis.size else None,
        "iti_p95_ms": float(np.quantile(itis, 0.95) * 1000) if itis.size else None,
        "matmul_tflops_per_sample": sum(r["work"]["matmul_flops"] for r in rows)
        / len(rows)
        / 1e12,
        "decode_matmul_gflops_per_generated_token": sum(
            r["decode_matmul_flops"] for r in rows
        )
        / tokens
        / 1e9,
        "copied_kv_bytes_per_generated_token": sum(
            r["work"]["kv_copy_bytes"] for r in rows
        )
        / tokens,
        **{
            f"repeat_{n}gram": float(
                np.mean([repeat_rate(r["token_ids"], n) for r in rows])
            )
            for n in (2, 3, 4)
        },
    }


def mean_sd(values):
    defined = [v for v in values if v is not None]
    return {
        "mean": float(np.mean(defined)) if defined else None,
        "std": float(np.std(defined, ddof=1)) if len(defined) > 1 else None,
        "n_defined": len(defined),
    }


def display_mean_sd(value):
    if value["mean"] is None:
        return "N/A"
    spread = "N/A" if value["std"] is None else f"{value['std']:.4f}"
    return f"{value['mean']:.4f} ± {spread}"
