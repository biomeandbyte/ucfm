from __future__ import annotations

from pathlib import Path

import yaml

from ucfm.config import REGISTRY_PATH


def load_registry(path: Path | None = None) -> list[dict]:
    p = path or REGISTRY_PATH
    if not p.exists():
        return []
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    datasets = raw.get("datasets") or []
    if not isinstance(datasets, list):
        raise ValueError(f"{p}: datasets must be a list")
    return datasets


def registry_ids(path: Path | None = None) -> list[str]:
    return [str(d["id"]) for d in load_registry(path) if d.get("id")]
