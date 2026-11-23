"""One independent GPU worker; each timed request has batch size one."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time

from .generation_protocol import (
    SEEDS,
    MODES,
    MAX_NEW_TOKENS,
    atomic_json,
    json_sha,
    load_checkpoint,
    load_completed,
    load_manifest,
    worker_protocol,
)


def initialize(experiment, max_new_tokens):
    import torch
    from transformers import AutoTokenizer
    from src.inference.cuda_generation import CudaDecoder

    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    if torch.cuda.get_device_capability(0) != (8, 0):
        raise RuntimeError(
            "This exact benchmark implementation is validated for SM80/A100 GPUs"
        )
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    protocol = experiment["protocol"]
    tokenizer = AutoTokenizer.from_pretrained(
        experiment["paths"]["tokenizer"], local_files_only=True
    )
    model, metadata = load_checkpoint(
        experiment["paths"]["checkpoint"],
        torch.device("cuda"),
        expected_recurrence=protocol["recurrence"],
        expected_step=protocol["checkpoint_step"],
    )
    decoder = CudaDecoder(model, max_new_tokens=max_new_tokens, capacity=576)
    decoder.prepare(64)
    return tokenizer, decoder, metadata


def check_generation(experiment, output):
    """Synthetic correctness checks, outside the 45,000 measured generations."""
    import torch
    from src.inference.generation_kernels import FusedRMSNorm, exact_rope
    from src.model.rope import apply_rope

    _, decoder, metadata = initialize(experiment, 32)
    checks = []
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            norm = decoder.model.layers[0].pre_attn_norm
            fused = FusedRMSNorm(norm)
            for dtype in (torch.float32, torch.bfloat16):
                x = torch.randn(
                    1, 1, decoder.model.config.hidden_size, device="cuda", dtype=dtype
                )
                torch.testing.assert_close(fused(x), norm(x), atol=0, rtol=0)
                q = x.view(1, 1, decoder.H, decoder.D).transpose(1, 2)
                cos, sin = decoder.cos[:1], decoder.sin[:1]
                torch.testing.assert_close(
                    exact_rope(q, cos, sin), apply_rope(q, cos, sin), atol=0, rtol=0
                )
        for seed in SEEDS:
            baseline = None
            for mode in MODES:
                result = decoder.generate(
                    list(range(10, 74)),
                    mode=mode,
                    seed=seed,
                    sample_id=-1,
                    eos_token_id=-1,
                )
                if mode == 0:
                    baseline = result
                    keys, values = (
                        decoder.key_storage.clone(),
                        decoder.value_storage.clone(),
                    )
                elif mode == 1:
                    for key in ("token_ids", "exit_depths"):
                        if result[key] != baseline[key]:
                            raise RuntimeError(f"Mode 0/1 synthetic mismatch: {key}")
                    torch.testing.assert_close(
                        keys, decoder.key_storage, atol=0, rtol=0
                    )
                    torch.testing.assert_close(
                        values, decoder.value_storage, atol=0, rtol=0
                    )
                checks.append(
                    dict(
                        mode=mode,
                        seed=seed,
                        tokens=result["generated_tokens"],
                        passed=True,
                    )
                )
        atomic_json(
            Path(output) / "generation_check.json",
            dict(
                passed=True,
                checks=checks,
                model=metadata,
                shared_weight_and_fused_kernel_checks=True,
                experiment_fingerprint=experiment["fingerprint"],
            ),
        )
    finally:
        decoder.close()


def generation_job(experiment, output, mode, seed, rank):
    output = Path(output)
    world_size = experiment["protocol"]["world_size"]
    if not 0 <= rank < world_size:
        raise ValueError("Invalid GPU shard rank")
    manifest = load_manifest(experiment["paths"]["manifest"])
    assigned = manifest[rank::world_size]
    wanted = {row["sample_id"] for row in assigned}
    protocol = worker_protocol(experiment, mode, seed)
    fingerprint = json_sha(protocol)
    directory = output / "generations"
    directory.mkdir(exist_ok=True)
    stem = directory / f"mode{mode}_seed{seed}_rank{rank}"
    path = stem.with_suffix(".jsonl")
    done = load_completed(path, fingerprint, repair_tail=True)
    if not set(done).issubset(wanted):
        raise ValueError("GPU shard assignment changed")
    config_path = stem.with_suffix(".config.json")
    if done:
        config = json.loads(config_path.read_text())
        if config["protocol"] != protocol or config["fingerprint"] != fingerprint:
            raise ValueError("Existing shard metadata differs from this experiment")
    if set(done) == wanted:
        print(f"ALREADY_COMPLETE {path}", flush=True)
        return
    baseline = {}
    if mode == 1:
        base_protocol = worker_protocol(experiment, 0, seed)
        baseline = load_completed(
            directory / f"mode0_seed{seed}_rank{rank}.jsonl", json_sha(base_protocol)
        )
        if set(baseline) != wanted:
            raise ValueError("Mode 0 must finish before coupled mode 1")
    tokenizer, decoder, metadata = initialize(experiment, 16)
    try:
        for i in range(3):
            decoder.generate(
                list(range(10, 74)),
                mode=mode,
                seed=1234,
                sample_id=-10 - i,
                eos_token_id=-1,
            )
        decoder.max_new_tokens = MAX_NEW_TOKENS
        import torch

        atomic_json(
            config_path,
            dict(
                protocol=protocol,
                fingerprint=fingerprint,
                model=metadata,
                device=str(torch.cuda.get_device_properties(0)),
                depth_stream_count=len(decoder.owned_streams),
            ),
        )
        started = time.monotonic()
        with path.open("a", encoding="utf-8", buffering=1) as stream:
            for row in assigned:
                sample_id = row["sample_id"]
                if sample_id in done:
                    continue
                result = decoder.generate(
                    row["prompt_ids"],
                    mode=mode,
                    seed=seed,
                    sample_id=sample_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                if mode == 1 and any(
                    result[key] != baseline[sample_id][key]
                    for key in ("token_ids", "exit_depths")
                ):
                    atomic_json(
                        output / f"equivalence_failure_seed{seed}_rank{rank}.json",
                        dict(
                            sample_id=sample_id, mode0=baseline[sample_id], mode1=result
                        ),
                    )
                    raise RuntimeError(
                        f"Coupled mode 0/1 mismatch at sample {sample_id}"
                    )
                result.update(
                    sample_id=sample_id,
                    mode=mode,
                    seed=seed,
                    experiment_fingerprint=fingerprint,
                    continuation=tokenizer.decode(
                        result["token_ids"],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    ),
                    full_text=tokenizer.decode(
                        row["prompt_ids"] + result["token_ids"],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    ),
                )
                stream.write(
                    json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n"
                )
                stream.flush()
                done[sample_id] = True
                if len(done) % 10 == 0 or len(done) == len(assigned):
                    os.fsync(stream.fileno())
                    print(
                        f"mode={mode} seed={seed} rank={rank} completed={len(done)}/{len(assigned)} "
                        f"elapsed={time.monotonic()-started:.1f}s",
                        flush=True,
                    )
        print(f"GENERATION_COMPLETE {path}", flush=True)
    finally:
        decoder.close()
