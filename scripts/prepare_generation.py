#!/usr/bin/env python3
"""Prepare the fixed text-continuation benchmark inputs."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.generation_data import main

if __name__ == "__main__":
    main()
