#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <grt_dmo|grt_soft_entropy|vanilla_dmo|vanilla_soft_entropy> [--dry-run]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/runs}"
GENERATION_DATA="${GENERATION_DATA:-$REPO_ROOT/data/generation}"
GENERATION_ROOT="${GENERATION_ROOT:-$REPO_ROOT/analysis/generation}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
RUN_KEY="$1"
shift

case "$RUN_KEY" in
  grt_dmo)
    RUN_NAME="fineweb_grt135m_loop8_mixture_22k_seed42"
    RECURRENCE="grt"
    LABEL="GRT + DMO"
    ;;
  grt_soft_entropy)
    RUN_NAME="fineweb_grt135m_loop8_soft_entropy_22k_seed42"
    RECURRENCE="grt"
    LABEL="GRT + soft loss + exit-distribution entropy regularization"
    ;;
  vanilla_dmo)
    RUN_NAME="fineweb_naive135m_loop8_mixture_22k_seed42"
    RECURRENCE="vanilla"
    LABEL="Vanilla + DMO"
    ;;
  vanilla_soft_entropy)
    RUN_NAME="fineweb_naive135m_loop8_soft_entropy_22k_seed42"
    RECURRENCE="vanilla"
    LABEL="Vanilla + soft loss + exit-distribution entropy regularization"
    ;;
  *) echo "Unsupported generation comparison: $RUN_KEY" >&2; exit 2 ;;
esac

export HF_HOME="${HF_HOME:-$REPO_ROOT/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export CUDA_DEVICE_MAX_CONNECTIONS=32 CUDA_DEVICE_MAX_COPY_CONNECTIONS=32
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1

cd "$REPO_ROOT"
exec "${PYTHON:-python}" scripts/evaluate_generation.py \
  --checkpoint "$RUN_ROOT/$RUN_NAME/checkpoints/step_0022000.pt" \
  --recurrence "$RECURRENCE" --label "$LABEL" --expected-step 22000 \
  --inputs "$GENERATION_DATA" --output "$GENERATION_ROOT/$RUN_KEY" \
  --workers "$NPROC_PER_NODE" "$@"
