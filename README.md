# Direct Mixture Optimization (DMO)

Code for Direct Mixture Optimization, our proposed method for pre-training Ouro-style looped language models whose next-token distribution is a finite mixture over recurrent depths.

Project blog: https://weizq7.notion.site/dmo

Paper coming soon! 🚀

## Environment

The reference environment uses Python 3.12 and PyTorch 2.8.0 with CUDA 12.8.
Create and activate the environment with:

```bash
conda create -n dmo python=3.12 -y
conda activate dmo
```

Check that the host driver supports the intended CUDA build:

```bash
nvidia-smi
```

Install the CUDA-enabled PyTorch build and the pinned dependencies:

```bash
python -m pip install \
  torch==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128

python -m pip install \
  "transformers==5.17.0" \
  "datasets==4.7.0" \
  "huggingface-hub==1.33.0" \
  "tokenizers==0.23.3" \
  "pyarrow==23.0.1" \
  "wandb==0.25.1" \
  "lm-eval==0.4.13" \
  "pytest==9.2.0" \
  "pyyaml==6.0.3"

python -m pip install -e . --no-deps
```

The `--no-deps` flag keeps the pinned versions above from being replaced by
unpinned dependencies declared in `pyproject.toml`.

## Dataset Preparation

The FineWeb-Edu experiments use the `sample-100BT` training split and the
separate `val` split from
[`SandyResearch/fineweb-edu-shuffled`](https://huggingface.co/datasets/SandyResearch/fineweb-edu-shuffled).
The published Parcae downloader creates the directory layout expected by the
training and evaluation scripts:

```bash
conda activate dmo

git clone https://github.com/SandyResearch/parcae.git ../parcae

export DATA_DIR="$PWD/data"
export HF_CACHE_DIR="$PWD/hf_cache"
export HF_HOME="$HF_CACHE_DIR"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_CACHE="$HF_HOME/hub"

python ../parcae/scripts/download_data.py fineweb-100bt
```

After downloading, the two data roots passed to the DMO scripts are:

```text
data/fineweb-edu-shuffled/data/sample-100BT/train
data/fineweb-edu-shuffled/data/val/train
```

Cache the tokenizer used by the released training and evaluation entry points
in the same Hugging Face cache:

```bash
python - <<'PY'
from transformers import AutoTokenizer

AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
PY
```

Cache the LAMBADA test set used by the full evaluation suite in the same
Hugging Face cache:

```bash
python - <<'PY'
from datasets import load_dataset

load_dataset("EleutherAI/lambada_openai", split="test")
PY
```

Cache the WikiText-103 validation split used by the full evaluation suite:

```bash
python - <<'PY'
from datasets import load_dataset

load_dataset("Salesforce/wikitext", "wikitext-103-v1", split="validation")
PY
```

The Core and Core-Extended evaluations use the `receval` implementation from
the Parcae checkout above; they are not tasks supplied by `lm-eval`. The
evaluation wrapper imports `receval` directly from `../parcae`, so no separate
Parcae installation is needed, but that checkout must remain available. On
the first Core evaluation, the required evaluation bundle is downloaded to
`eval_cache` (or to `CORE_CACHE` if that variable is set). To download it
ahead of time, run:

```bash
export PARCAE_REPO="$PWD/../parcae"
export CORE_CACHE="$PWD/eval_cache"
PARCAE_CACHE="$CORE_CACHE" python - <<'PY'
import os
import sys

sys.path.insert(0, os.environ["PARCAE_REPO"])
from receval.tasks.core_eval import download_eval_bundle

download_eval_bundle()
PY
```

## Reproducing the 135M Training Runs

The commands below reproduce the eight FineWeb-Edu experiments in the 135M
comparison. Run them from the repository root on a node with eight GPUs. Each
run uses the same 22,000-step schedule, sequence length, optimizer settings,
periodic validation, and gradient-checkpointing configuration as the reported
experiments.

The helper script `scripts/run_135m.sh` stores the shared configuration and
maps a short run name to each reported experiment. It locates the repository,
dataset, and output directories automatically. To use a different location,
override `TRAIN_DATA`, `VAL_DATA`, or `RUN_ROOT` before running it.

The eight commands below reproduce the reported 135M runs. Run the commands
one at a time, or submit them separately according to your cluster workflow.

```bash
# GRT + DMO
bash scripts/run_135m.sh grt_dmo

# GRT + soft loss
bash scripts/run_135m.sh grt_soft

# GRT + soft loss + exit-distribution entropy regularization
bash scripts/run_135m.sh grt_soft_entropy

# GRT + uniform depth sampling
bash scripts/run_135m.sh grt_uniform

# Vanilla + DMO
bash scripts/run_135m.sh vanilla_dmo

# Vanilla + soft loss
bash scripts/run_135m.sh vanilla_soft

# Vanilla + DMO + exit-distribution entropy regularization
bash scripts/run_135m.sh vanilla_dmo_entropy

# Vanilla + soft loss + exit-distribution entropy regularization
bash scripts/run_135m.sh vanilla_soft_entropy
```

For the uniform baseline, `--training-method uniform` samples a depth
uniformly from `1` through `2 * num_recurrent_steps - 1`. With
`num_recurrent_steps=8`, this means depths 1--15 and an average sampled depth
of 8.

## Evaluating the Final 135M Checkpoints

The full evaluation suite reports complete FineWeb-Edu validation perplexity,
WikiText-103 perplexity, LAMBADA perplexity, and DataComp-LM Core-Extended
scores. Core-Extended includes the Core tasks and uses the three reported
evaluation seeds (`1234,2345,3456`). Each command below evaluates the final
22,000-step checkpoint for one training run and writes its results to
`analysis/135m/<run-name>`.

The evaluator discovers the cached LAMBADA Arrow file under `HF_HOME`; the
dataset-preparation command above only needs to be run once.

Installing `lm-eval` alone is therefore not sufficient for the complete
evaluation suite: LAMBADA also needs its cached dataset, and Core/Core-
Extended need a local clone of Parcae, which provides the `receval`
evaluation code and evaluation bundle.

Run the commands one at a time on a node with eight GPUs:

```bash
# GRT + DMO
bash scripts/evaluate_135m.sh grt_dmo

# GRT + soft loss
bash scripts/evaluate_135m.sh grt_soft

# GRT + soft loss + exit-distribution entropy regularization
bash scripts/evaluate_135m.sh grt_soft_entropy

# GRT + uniform depth sampling
bash scripts/evaluate_135m.sh grt_uniform

# Vanilla + DMO
bash scripts/evaluate_135m.sh vanilla_dmo

# Vanilla + soft loss
bash scripts/evaluate_135m.sh vanilla_soft

# Vanilla + DMO + exit-distribution entropy regularization
bash scripts/evaluate_135m.sh vanilla_dmo_entropy

# Vanilla + soft loss + exit-distribution entropy regularization
bash scripts/evaluate_135m.sh vanilla_soft_entropy
```

## Acknowledgements

This repository builds on the [LoopLM](https://github.com/rkstgr/LoopLM)
codebase. We also thank the authors of [Parcae](https://github.com/sandyresearch/parcae)
for their evaluation tools and resources.
