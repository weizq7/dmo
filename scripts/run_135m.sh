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

TRAIN_DATA="${TRAIN_DATA:-$REPO_ROOT/data/fineweb-edu-shuffled/data/sample-100BT/train}"
VAL_DATA="${VAL_DATA:-$REPO_ROOT/data/fineweb-edu-shuffled/data/val/train}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/runs}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

case "$1" in
  grt_dmo)
    RUN_NAME="fineweb_grt135m_loop8_mixture_22k_seed42"
    RECURRENCE="grt"
    METHOD="dmo"
    BETA_KL="0.0"
    ;;
  grt_soft)
    RUN_NAME="fineweb_grt135m_loop8_soft_no_entropy_22k_seed42"
    RECURRENCE="grt"
    METHOD="soft"
    BETA_KL="0.0"
    ;;
  grt_soft_entropy)
    RUN_NAME="fineweb_grt135m_loop8_soft_entropy_22k_seed42"
    RECURRENCE="grt"
    METHOD="soft_entropy"
    BETA_KL="0.1"
    ;;
  grt_uniform)
    RUN_NAME="fineweb_grt135m_uniform15_22k_seed42"
    RECURRENCE="grt"
    METHOD="uniform"
    BETA_KL="0.0"
    ;;
  vanilla_dmo)
    RUN_NAME="fineweb_naive135m_loop8_mixture_22k_seed42"
    RECURRENCE="naive"
    METHOD="dmo"
    BETA_KL="0.0"
    ;;
  vanilla_soft)
    RUN_NAME="fineweb_naive135m_loop8_soft_no_entropy_22k_seed42"
    RECURRENCE="naive"
    METHOD="soft"
    BETA_KL="0.0"
    ;;
  vanilla_dmo_entropy)
    RUN_NAME="fineweb_naive135m_loop8_mixture_entropy_22k_seed42"
    RECURRENCE="naive"
    METHOD="dmo_entropy"
    BETA_KL="0.1"
    ;;
  vanilla_soft_entropy)
    RUN_NAME="fineweb_naive135m_loop8_soft_entropy_22k_seed42"
    RECURRENCE="naive"
    METHOD="soft_entropy"
    BETA_KL="0.1"
    ;;
  *)
    echo "Unknown run name: $1" >&2
    echo "Run names: grt_dmo grt_soft grt_soft_entropy grt_uniform" >&2
    echo "           vanilla_dmo vanilla_soft vanilla_dmo_entropy vanilla_soft_entropy" >&2
    exit 2
    ;;
esac

COMMON_ARGS=(
  --model-config scale_135m
  --seq-len 2048
  --num-recurrent-steps 8
  --seed 42
  --max-steps 22000
  --batch-size 8
  --grad-accum-steps 4
  --lr 6e-4
  --lr-schedule warmup_cosine
  --warmup-steps 2000
  --lr-total-steps 22000
  --min-lr 6e-5
  --parquet-data-dir "$TRAIN_DATA"
  --parquet-val-dir "$VAL_DATA"
  --pack-buffer-size 1000
  --log-every 50
  --save-every 1000
  --val-every 2000
  --val-batch-size 2
  --val-max-batches 1000
  --gradient-checkpointing
)

RECURRENCE_ARGS=()
if [[ "$RECURRENCE" == "grt" ]]; then
  RECURRENCE_ARGS=(
    --grt-state-noise 0.1
    --grt-gate-noise 0.1
    --grt-gate-temperature 1.0
  )
fi

cd "$REPO_ROOT"

torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" scripts/train.py \
  "${COMMON_ARGS[@]}" \
  --recurrence "$RECURRENCE" \
  "${RECURRENCE_ARGS[@]}" \
  --training-method "$METHOD" \
  --beta-kl "$BETA_KL" \
  --output-dir "$RUN_ROOT/$RUN_NAME"
