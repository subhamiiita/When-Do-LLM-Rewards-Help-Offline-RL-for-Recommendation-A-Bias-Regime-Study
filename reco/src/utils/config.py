from pathlib import Path
from typing import Any, Dict
import copy
import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(base_path: str | Path, *overrides: str | Path) -> Dict[str, Any]:
    cfg = load_yaml(base_path)
    for o in overrides:
        cfg = deep_update(cfg, load_yaml(o))
    return cfg
