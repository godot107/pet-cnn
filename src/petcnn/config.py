"""Paths and config.yaml, loaded once. Env vars relocate data/models inside a SageMaker container."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = Path(os.environ.get("PETCNN_PROCESSED", DATA / "processed"))
MODELS = Path(os.environ.get("PETCNN_MODELS", ROOT / "models"))
RESULTS = ROOT / "results"
REPORTS = ROOT / "reports"


@lru_cache
def load() -> dict:
    return yaml.safe_load((ROOT / "config.yaml").read_text())
