#!/usr/bin/env python3
"""Run a complete three-mode text-continuation comparison."""
import os
from pathlib import Path
import sys

# These must be set before importing torch or initializing any CUDA context.
for name in ("CUDA_DEVICE_MAX_CONNECTIONS", "CUDA_DEVICE_MAX_COPY_CONNECTIONS"):
    os.environ.setdefault(name, "32")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.generation_runner import main

if __name__ == "__main__":
    main()
