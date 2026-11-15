#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <run-name>" >&2
  echo "Run names: grt_dmo grt_soft grt_soft_entropy grt_uniform" >&2
  echo "           vanilla_dmo vanilla_soft vanilla_dmo_entropy vanilla_soft_entropy" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/runs}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-$REPO_ROOT/analysis/135m}"
FINEWEB_VAL="${FINEWEB_VAL:-$REPO_ROOT/data/fineweb-edu-shuffled/data/val/train}"
CORE_CACHE="${CORE_CACHE:-$REPO_ROOT/eval_cache}"
PARCAE_REPO="${PARCAE_REPO:-$REPO_ROOT/../parcae}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-0022000}"

case "$1" in
  grt_dmo)
    RUN_NAME="fineweb_grt135m_loop8_mixture_22k_seed42"
    PREDICTION_MODE="mixture"
    ;;
  grt_soft)
    RUN_NAME="fineweb_grt135m_loop8_soft_no_entropy_22k_seed42"
    PREDICTION_MODE="mixture"
    ;;
  grt_soft_entropy)
    RUN_NAME="fineweb_grt135m_loop8_soft_entropy_22k_seed42"
    PREDICTION_MODE="mixture"
    ;;
  grt_uniform)
    RUN_NAME="fineweb_grt135m_uniform15_22k_seed42"
    PREDICTION_MODE="final"
    ;;
  vanilla_dmo)
    RUN_NAME="fineweb_naive135m_loop8_mixture_22k_seed42"
    PREDICTION_MODE="mixture"
    ;;
  vanilla_soft)
    RUN_NAME="fineweb_naive135m_loop8_soft_no_entropy_22k_seed42"
    PREDICTION_MODE="mixture"
    ;;
  vanilla_dmo_entropy)
    RUN_NAME="fineweb_naive135m_loop8_mixture_entropy_22k_seed42"
    PREDICTION_MODE="mixture"
    ;;
  vanilla_soft_entropy)
    RUN_NAME="fineweb_naive135m_loop8_soft_entropy_22k_seed42"
    PREDICTION_MODE="mixture"
    ;;
  *)
    echo "Unknown run name: $1" >&2
    echo "Run names: grt_dmo grt_soft grt_soft_entropy grt_uniform" >&2
    echo "           vanilla_dmo vanilla_soft vanilla_dmo_entropy vanilla_soft_entropy" >&2
    exit 2
    ;;
esac

CHECKPOINT="$RUN_ROOT/$RUN_NAME/checkpoints/step_${CHECKPOINT_STEP}.pt"
OUTPUT_DIR="$ANALYSIS_ROOT/$1"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Expected final checkpoint is missing: $CHECKPOINT" >&2
  echo "Set RUN_ROOT to the directory containing the reproduced runs if needed." >&2
  exit 2
fi
if [[ ! -d "$FINEWEB_VAL" ]]; then
  echo "FineWeb-Edu validation directory is missing: $FINEWEB_VAL" >&2
  echo "Set FINEWEB_VAL to the downloaded validation parquet directory if needed." >&2
  exit 2
fi
if [[ ! -d "$PARCAE_REPO" ]]; then
  echo "Parcae checkout is missing: $PARCAE_REPO" >&2
  echo "Clone it as ../parcae or set PARCAE_REPO explicitly." >&2
  exit 2
fi

export HF_HOME="${HF_HOME:-$REPO_ROOT/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PARCAE_CACHE="$CORE_CACHE"

LAMBADA_ARGS=()
if [[ -n "${LAMBADA_DATA:-}" ]]; then
  LAMBADA_ARGS=(--lambada-data "$LAMBADA_DATA")
fi

mkdir -p "$OUTPUT_DIR"
cd "$REPO_ROOT"

torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" \
  scripts/evaluate_suite.py \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --tasks validation,wikitext,lambada,core_extended \
  --fineweb-val-dir "$FINEWEB_VAL" \
  --fineweb-max-batches 0 \
  --core-cache "$CORE_CACHE" \
  --parcae-repo "$PARCAE_REPO" \
  --core-seeds 1234,2345,3456 \
  --prediction-mode "$PREDICTION_MODE" \
  --batch-size 2 \
  "${LAMBADA_ARGS[@]}"
