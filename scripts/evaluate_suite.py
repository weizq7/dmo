#!/usr/bin/env python3
"""Run the Parcae-style language-modeling and downstream evaluation suite.

The suite reports:

* FineWeb-Edu held-out validation PPL;
* WikiText-103 validation PPL;
* LAMBADA PPL from lm-evaluation-harness;
* DataComp-LM Core and Core-Extended aggregate scores.

The default ``mixture`` prediction mode uses the exact exit-weighted marginal
distribution, matching LoopLM's intended latent-depth inference procedure.
``final`` remains available for strict fixed-depth comparisons with Parcae.
The PPL reports always include both final-step and mixture values for the two
corpus validation sets.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.distributed as dist
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.model.config import LoopLMConfig  # noqa: E402
from src.model.looplm import (  # noqa: E402
    LoopLM,
    compute_exit_distribution,
    model_kwargs_from_checkpoint,
)
from src.training.data import make_dataloader, make_parquet_dataloader  # noqa: E402


def _rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def _world_size() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def print0(*args: Any, **kwargs: Any) -> None:
    if _rank() == 0:
        print(*args, **kwargs, flush=True)


def init_distributed(device_arg: str) -> torch.device:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed evaluation requires CUDA")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        # Full FineWeb validation can leave one rank behind for a long time
        # while it is reading/tokenizing parquet shards.  The training entry
        # point keeps the normal short NCCL timeout; evaluation gets a long,
        # explicit timeout so a temporary rank skew is not misdiagnosed as a
        # communication failure.  Slurm's job time limit remains the ultimate
        # bound if a rank really hangs.
        timeout_hours = float(os.environ.get("LOOPLM_EVAL_NCCL_TIMEOUT_HOURS", "24"))
        if timeout_hours <= 0:
            raise ValueError("LOOPLM_EVAL_NCCL_TIMEOUT_HOURS must be positive")
        dist.init_process_group(
            backend="nccl",
            timeout=datetime.timedelta(hours=timeout_hours),
        )
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}")

    if device_arg == "auto":
        device_arg = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast(device.type, enabled=False)


def load_model_and_tokenizer(
    checkpoint_path: Path,
    device: torch.device,
    num_steps_override: int | None,
) -> tuple[LoopLM, Any, LoopLMConfig, dict[str, Any], int, str]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    config = LoopLMConfig(**checkpoint["model_config"])
    model_kwargs = model_kwargs_from_checkpoint(checkpoint)
    model = LoopLM(config, **model_kwargs)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    trainer_config = checkpoint.get("trainer_config", {})
    tokenizer_id = trainer_config.get(
        "tokenizer_id", checkpoint.get("tokenizer_id", "HuggingFaceTB/SmolLM2-135M")
    )
    del checkpoint

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    model.eval().to(device=device, dtype=torch.float32)
    num_steps = int(num_steps_override or config.max_recurrent_steps)
    recurrence = str(trainer_config.get("recurrence", "grt" if model_kwargs.get("use_grt") else "naive"))
    return model, tokenizer, config, model_kwargs, num_steps, recurrence


def _per_token_metrics(
    output: Any,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    per_depth_ce = torch.stack(
        [
            F.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                ignore_index=-100,
                reduction="none",
            ).view_as(targets)
            for logits in output.logits
        ]
    )
    exit_probs = compute_exit_distribution(
        [value.float() for value in output.exit_lambdas]
    ).clamp_min(1e-30)
    final_nll = per_depth_ce[-1]
    mixture_nll = -torch.logsumexp(
        exit_probs.log() - per_depth_ce,
        dim=0,
    )
    soft_nll = (exit_probs * per_depth_ce).sum(dim=0)
    return final_nll, mixture_nll, soft_nll


def _reduce_totals(total: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
    return total


@torch.inference_mode()
def evaluate_ppl_loader(
    model: LoopLM,
    loader: Iterable[torch.Tensor],
    device: torch.device,
    num_steps: int,
    recurrence: str,
    max_batches: int | None,
    label: str,
) -> dict[str, Any]:
    totals = torch.zeros(5, dtype=torch.float64, device=device)
    # final NLL, exact mixture NLL, soft expected NLL, valid-token count, batches
    for batch_index, token_ids in enumerate(loader, start=1):
        if max_batches is not None and batch_index > max_batches:
            break
        token_ids = token_ids.to(device=device, non_blocking=True)
        inputs, targets = token_ids[:, :-1], token_ids[:, 1:]
        with autocast_context(device):
            output = model(
                inputs,
                num_steps=num_steps,
                recurrence=recurrence,
            )
            final_nll, mixture_nll, soft_nll = _per_token_metrics(output, targets)
        valid = targets.ne(-100)
        totals[0] += final_nll[valid].double().sum()
        totals[1] += mixture_nll[valid].double().sum()
        totals[2] += soft_nll[valid].double().sum()
        totals[3] += valid.sum().double()
        totals[4] += 1.0
        if batch_index % 25 == 0:
            print0(f"{label}: batch {batch_index}; tokens={int(totals[3].item()):,}")

    totals = _reduce_totals(totals)
    count = int(totals[3].item())
    if count == 0:
        raise RuntimeError(f"{label}: no valid tokens were evaluated")
    final_nll = float((totals[0] / totals[3]).item())
    mixture_nll = float((totals[1] / totals[3]).item())
    soft_nll = float((totals[2] / totals[3]).item())
    return {
        "final_nll": final_nll,
        "final_ppl": math.exp(min(final_nll, 80.0)),
        "mixture_nll": mixture_nll,
        "mixture_ppl": math.exp(min(mixture_nll, 80.0)),
        "soft_expected_nll": soft_nll,
        "soft_expected_ppl": math.exp(min(soft_nll, 80.0)),
        "valid_tokens": count,
        "batches": int(totals[4].item()),
    }


class LoopLMCoreAdapter:
    """Expose LoopLM as the callable interface expected by Parcae's Core code."""

    def __init__(
        self,
        model: LoopLM,
        device: torch.device,
        num_steps: int,
        recurrence: str,
        prediction_mode: str,
    ) -> None:
        self.model = model
        self.device = device
        self.num_steps = num_steps
        self.recurrence = recurrence
        self.prediction_mode = prediction_mode

    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor, **_: Any) -> dict[str, torch.Tensor]:
        input_ids = input_ids.to(self.device)
        with autocast_context(self.device):
            output = self.model(
                input_ids,
                num_steps=self.num_steps,
                recurrence=self.recurrence,
            )
        if self.prediction_mode == "final":
            logits = output.logits[-1].float()
        else:
            step_logits = torch.stack(output.logits, dim=0).float()
            step_log_probs = F.log_softmax(step_logits, dim=-1)
            exit_probs = compute_exit_distribution(
                [value.float() for value in output.exit_lambdas]
            ).clamp_min(1e-30)
            logits = torch.logsumexp(
                exit_probs.log().unsqueeze(-1) + step_log_probs,
                dim=0,
            )
        return {"logits": logits}

    def __call__(self, input_ids: torch.Tensor, **kwargs: Any) -> dict[str, torch.Tensor]:
        return self.forward(input_ids, **kwargs)


def run_core_suite(
    model: LoopLM,
    tokenizer: Any,
    device: torch.device,
    config: LoopLMConfig,
    num_steps: int,
    recurrence: str,
    prediction_mode: str,
    core_cache: Path,
    parcae_repo: Path,
    max_per_task: int,
    seeds: list[int],
    run_core: bool = True,
    run_core_extended: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    os.environ["PARCAE_CACHE"] = str(core_cache)
    sys.path.insert(0, str(parcae_repo))
    from receval.tasks.core_eval import run_core_eval
    from receval.tasks.core_extended_eval import run_core_extended_eval

    # Several Core tasks use 10-shot prompts.  The Parcae evaluator samples
    # those demonstrations from the same task data, so a tiny smoke-test cap
    # must still leave room for the demonstrations plus one scored example.
    effective_max_per_task = max_per_task
    if 0 < effective_max_per_task < 16:
        print0(
            f"Core cap {effective_max_per_task} is too small for 10-shot tasks; "
            "using 16 examples per task instead."
        )
        effective_max_per_task = 16

    adapter = LoopLMCoreAdapter(
        model, device, num_steps, recurrence, prediction_mode
    )
    core = {}
    extended = {}
    if run_core:
        core = run_core_eval(
            adapter,
            tokenizer,
            device,
            max_seq_len=config.max_seq_len,
            max_per_task=effective_max_per_task,
            seeds=seeds,
        )
    if run_core_extended:
        extended = run_core_extended_eval(
            adapter,
            tokenizer,
            device,
            max_seq_len=config.max_seq_len,
            max_per_task=effective_max_per_task,
            seeds=seeds,
        )
    return core, extended


@torch.inference_mode()
def run_lambada_local(
    model: LoopLM,
    tokenizer: Any,
    device: torch.device,
    num_steps: int,
    recurrence: str,
    prediction_mode: str,
    data_path: Path,
    limit: int | None,
) -> dict[str, Any]:
    """Evaluate the cached LAMBADA test Arrow file without Hub access.

    lm-eval's task definition is used semantically (last whitespace-delimited
    word as target), but reading the already-cached Arrow file directly makes
    the Slurm job genuinely offline and reproducible.
    """
    from datasets import Dataset

    dataset = Dataset.from_file(str(data_path))
    indices = list(range(len(dataset)))
    if limit is not None:
        indices = indices[:limit]
    if _world_size() > 1:
        indices = indices[_rank() :: _world_size()]

    # lm-evaluation-harness's ``perplexity`` aggregation averages the
    # complete continuation log-likelihood per example before exponentiating.
    # Keep token totals too, since the token-weighted value is useful as a
    # diagnostic but is not the Parcae table's LAMBADA convention.
    total_log_prob = torch.zeros(4, dtype=torch.float64, device=device)
    for local_index, dataset_index in enumerate(indices, start=1):
        text = str(dataset[dataset_index]["text"])
        words = text.split()
        if len(words) < 2:
            continue
        context = " ".join(words[:-1])
        continuation = " " + words[-1]
        ctx_tokens = tokenizer.encode(context, add_special_tokens=False)
        full_tokens = tokenizer.encode(context + continuation, add_special_tokens=False)
        if full_tokens[: len(ctx_tokens)] != ctx_tokens:
            context = context.rstrip()
            ctx_tokens = tokenizer.encode(context, add_special_tokens=False)
            full_tokens = tokenizer.encode(context + continuation, add_special_tokens=False)
        cont_tokens = full_tokens[len(ctx_tokens) :]
        if not cont_tokens or len(full_tokens) > model.config.max_seq_len:
            continue
        input_ids = torch.tensor([full_tokens], dtype=torch.long, device=device)
        with autocast_context(device):
            output = model(input_ids, num_steps=num_steps, recurrence=recurrence)
            if prediction_mode == "final":
                logits = output.logits[-1].float()
            else:
                step_logits = torch.stack(output.logits, dim=0).float()
                step_log_probs = F.log_softmax(step_logits, dim=-1)
                exit_probs = compute_exit_distribution(
                    [value.float() for value in output.exit_lambdas]
                ).clamp_min(1e-30)
                logits = torch.logsumexp(
                    exit_probs.log().unsqueeze(-1) + step_log_probs,
                    dim=0,
                )
            log_probs = F.log_softmax(logits, dim=-1)[0]
        cont_ids = torch.tensor(cont_tokens, dtype=torch.long, device=device)
        start = len(ctx_tokens) - 1
        example_log_prob = log_probs[start : start + len(cont_tokens), :].gather(
            1, cont_ids.unsqueeze(1)
        ).sum().double()
        total_log_prob[0] += example_log_prob
        total_log_prob[1] += len(cont_tokens)
        total_log_prob[2] += 1.0
        total_log_prob[3] += example_log_prob
        if local_index % 500 == 0:
            print0(f"LAMBADA: processed {local_index}/{len(indices)} examples")

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(total_log_prob, op=dist.ReduceOp.SUM)
    token_count = int(total_log_prob[1].item())
    example_count = int(total_log_prob[2].item())
    if token_count == 0 or example_count == 0:
        raise RuntimeError("LAMBADA evaluation produced no valid target tokens")
    token_nll = float((-total_log_prob[0] / total_log_prob[1]).item())
    example_nll = float((-total_log_prob[3] / total_log_prob[2]).item())
    return {
        # ``ppl`` is the lm-eval-compatible, example-weighted metric.
        "ppl": math.exp(min(example_nll, 80.0)),
        "nll": example_nll,
        "example_ppl": math.exp(min(example_nll, 80.0)),
        "example_nll": example_nll,
        "token_ppl": math.exp(min(token_nll, 80.0)),
        "token_nll": token_nll,
        "target_tokens": token_count,
        "valid_examples": example_count,
        "examples": len(dataset) if limit is None else min(limit, len(dataset)),
        "data_path": str(data_path),
    }


def find_lambada_arrow() -> Path:
    """Find the cached Arrow file produced by ``datasets.load_dataset``.

    The public evaluation wrapper keeps the LAMBADA path optional so that a
    clone can use either its repository-local Hugging Face cache or an
    existing user-level cache.  The cache must be populated before running in
    offline environments.
    """
    roots: list[Path] = []
    datasets_cache = os.environ.get("HF_DATASETS_CACHE")
    if datasets_cache:
        roots.append(Path(datasets_cache))
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        roots.append(Path(hf_home) / "datasets")
    roots.extend(
        [
            REPO / "hf_cache" / "datasets",
            Path.home() / ".cache" / "huggingface" / "datasets",
        ]
    )

    candidates: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        root = root.expanduser()
        if root in seen or not root.is_dir():
            continue
        seen.add(root)
        candidates.extend(root.glob("EleutherAI___lambada_openai/**/lambada_openai-test.arrow"))
    if not candidates:
        searched = ", ".join(str(root) for root in roots)
        raise FileNotFoundError(
            "LAMBADA cache not found. Run "
            "load_dataset('EleutherAI/lambada_openai', split='test') "
            "once, or pass --lambada-data explicitly. Searched: "
            f"{searched}"
        )
    return sorted(set(candidates))[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--tasks",
        default="validation,wikitext,lambada,core,core_extended",
        help="Comma-separated tasks, or 'all'.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--fineweb-val-dir",
        default=None,
        help="FineWeb-Edu validation parquet directory (required for validation).",
    )
    parser.add_argument("--fineweb-max-batches", type=int, default=0)
    parser.add_argument("--wikitext-dataset", default="Salesforce/wikitext")
    parser.add_argument("--wikitext-config", default="wikitext-103-v1")
    parser.add_argument("--wikitext-split", default="validation")
    parser.add_argument("--wikitext-max-batches", type=int, default=0)
    parser.add_argument("--tokenizer-id", default=None)
    parser.add_argument(
        "--prediction-mode",
        choices=("final", "mixture"),
        default="mixture",
        help=(
            "Mode for LAMBADA/Core (default: exact exit-weighted mixture; "
            "use final for fixed-depth Parcae comparison). Corpus PPL always "
            "reports both."
        ),
    )
    parser.add_argument("--lambada-limit", type=int, default=None)
    parser.add_argument(
        "--lambada-data",
        type=Path,
        default=None,
        help="Path to lm-eval's cached lambada_openai-test.arrow file.",
    )
    parser.add_argument("--core-max-per-task", type=int, default=-1)
    parser.add_argument("--core-seeds", default="1234,2345,3456")
    parser.add_argument(
        "--core-cache",
        type=Path,
        default=Path("eval_cache"),
        help="Cache directory for DataComp-LM Core evaluation.",
    )
    parser.add_argument(
        "--parcae-repo",
        type=Path,
        default=None,
        help="Local checkout of the Parcae evaluation package (required for Core tasks).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested = {value.strip() for value in args.tasks.split(",") if value.strip()}
    if "all" in requested:
        requested = {"validation", "wikitext", "lambada", "core", "core_extended"}
    valid_tasks = {"validation", "wikitext", "lambada", "core", "core_extended"}
    unknown = requested - valid_tasks
    if unknown:
        raise ValueError(f"Unknown tasks: {sorted(unknown)}")
    if "core" in requested and "core_extended" in requested:
        # Core-Extended evaluates the complete Core task subset as well and
        # reports both metrics.  Avoid evaluating those Core tasks twice when
        # callers use the historical all-tasks list.
        print0(
            "Core-Extended includes the Core subset; "
            "skipping the standalone Core pass."
        )
        requested.remove("core")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if "validation" in requested and not args.fineweb_val_dir:
        raise ValueError("--fineweb-val-dir is required when validation is requested")
    if "lambada" in requested and args.lambada_data is None:
        args.lambada_data = find_lambada_arrow()
    if ({"core", "core_extended"} & requested) and args.parcae_repo is None:
        raise ValueError("--parcae-repo is required for Core evaluation")
    if args.fineweb_val_dir is not None and not Path(args.fineweb_val_dir).is_dir():
        raise FileNotFoundError(args.fineweb_val_dir)
    if args.lambada_data is not None and not args.lambada_data.is_file():
        raise FileNotFoundError(args.lambada_data)
    if args.parcae_repo is not None and not args.parcae_repo.is_dir():
        raise FileNotFoundError(args.parcae_repo)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = init_distributed(args.device)
    model, tokenizer, config, model_kwargs, num_steps, recurrence = load_model_and_tokenizer(
        args.checkpoint, device, args.num_steps
    )
    if args.tokenizer_id is not None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id)
    seq_len = int(args.seq_len or config.max_seq_len)
    seeds = [int(value) for value in args.core_seeds.split(",") if value.strip()]
    max_fineweb_batches = args.fineweb_max_batches or None
    max_wikitext_batches = args.wikitext_max_batches or None

    print0(f"Checkpoint: {args.checkpoint}")
    print0(f"Model: {config.num_parameters() / 1e6:.2f}M parameters")
    print0(f"Recurrence: {recurrence}, steps={num_steps}, prediction_mode={args.prediction_mode}")
    print0(f"Tokenizer: {args.tokenizer_id or tokenizer.name_or_path}")
    print0(f"Tasks: {sorted(requested)}")
    if _world_size() > 1:
        print0(
            "Distributed evaluation NCCL timeout: "
            f"{os.environ.get('LOOPLM_EVAL_NCCL_TIMEOUT_HOURS', '24')} hours"
        )

    summary: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "saved_step": None,
        "model_parameters": config.num_parameters(),
        "recurrence": recurrence,
        "num_steps": num_steps,
        "prediction_mode": args.prediction_mode,
        "tokenizer": args.tokenizer_id or getattr(tokenizer, "name_or_path", None),
        "seq_len": seq_len,
        "tasks": sorted(requested),
    }
    # The checkpoint step is read separately to avoid retaining optimizer state.
    checkpoint_meta = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    summary["saved_step"] = checkpoint_meta.get("step")
    del checkpoint_meta

    if "validation" in requested:
        print0("\n=== FineWeb-Edu validation ===")
        loader = make_parquet_dataloader(
            data_dir=args.fineweb_val_dir,
            tokenizer_id=args.tokenizer_id or tokenizer.name_or_path,
            seq_len=seq_len,
            batch_size=args.batch_size,
            seed=42,
            pack_buffer_size=1000,
            drop_last=False,
        )
        summary["fineweb_validation"] = evaluate_ppl_loader(
            model, loader, device, num_steps, recurrence,
            max_fineweb_batches, "FineWeb validation",
        )
        print0(json.dumps(summary["fineweb_validation"], indent=2))

    if "wikitext" in requested:
        print0("\n=== WikiText-103 validation ===")
        loader = make_dataloader(
            dataset_name=args.wikitext_dataset,
            dataset_config=args.wikitext_config,
            split=args.wikitext_split,
            tokenizer_id=args.tokenizer_id or tokenizer.name_or_path,
            seq_len=seq_len,
            batch_size=args.batch_size,
            shuffle=False,
            seed=42,
            drop_last=False,
            use_distributed_sampler=_world_size() > 1,
        )
        summary["wikitext_validation"] = evaluate_ppl_loader(
            model, loader, device, num_steps, recurrence,
            max_wikitext_batches, "WikiText validation",
        )
        print0(json.dumps(summary["wikitext_validation"], indent=2))

    if "lambada" in requested:
        barrier()
        print0("\n=== LAMBADA ===")
        lambada_data = args.lambada_data
        lambada = run_lambada_local(
            model, tokenizer, device, num_steps, recurrence,
            args.prediction_mode, lambada_data, args.lambada_limit,
        )
        if _rank() == 0:
            summary["lambada"] = lambada
            print0(json.dumps(summary["lambada"], indent=2))
        barrier()

    if "core" in requested or "core_extended" in requested:
        print0("\n=== Core / Core-Extended ===")
        core, extended = run_core_suite(
            model, tokenizer, device, config, num_steps, recurrence,
            args.prediction_mode, args.core_cache, args.parcae_repo,
            args.core_max_per_task, seeds,
            run_core="core" in requested,
            run_core_extended="core_extended" in requested,
        )
        if _rank() == 0:
            if "core" in requested:
                summary["core"] = core
            if "core_extended" in requested:
                summary["core_extended"] = extended
            print0(json.dumps({"core": core, "core_extended": extended}, indent=2))

    barrier()
    if _rank() == 0:
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, default=str) + "\n"
        )
        print0(f"\nWrote {args.output_dir / 'summary.json'}")
    barrier()
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
