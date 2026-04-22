"""YAML config loader."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | os.PathLike) -> Dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dataset_config(name: str) -> Dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    return load_config(root / "configs" / f"{name}.yaml")


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]
