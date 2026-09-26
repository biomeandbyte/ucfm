from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = ROOT / "datasets" / "registry.yaml"


class StorageCfg(BaseModel):
    backend: str = "local"
    bucket: str = "biomeandbyte-uav-continual-learning"
    local_root: str = "data"
    prefix: str = ""


class TrainCfg(BaseModel):
    dinov2_model: str = "facebook/dinov2-base"
    image_size: int = 224
    pretrain_steps: int = 200
    batch_size: int = 16
    lr: float = 1e-5
    probe_epochs: int = 15
    probe_batch_size: int = 32
    probe_lr: float = 1e-3


class EvalCfg(BaseModel):
    ws_dir: str = "eval/ws"
    gs_dir: str = "eval/gs"


class FileConfig(BaseModel):
    storage: StorageCfg = Field(default_factory=StorageCfg)
    train: TrainCfg = Field(default_factory=TrainCfg)
    eval: EvalCfg = Field(default_factory=EvalCfg)


class EnvSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT / ".env"),
        extra="ignore",
        populate_by_name=True,
    )

    ucfm_s3_bucket: str = ""


def _load_dotenv() -> None:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    load_dotenv(Path.cwd() / ".env")


def env() -> EnvSettings:
    _load_dotenv()
    return EnvSettings()


def load_config(path: Path | None = None) -> FileConfig:
    cfg_path = path or (ROOT / "config.yaml")
    raw: dict[str, Any] = {}
    if cfg_path.exists():
        loaded = yaml.safe_load(cfg_path.read_text()) or {}
        if isinstance(loaded, dict):
            raw = loaded
    cfg = FileConfig.model_validate(raw)
    if env().ucfm_s3_bucket:
        cfg.storage.backend = "s3"
        cfg.storage.bucket = env().ucfm_s3_bucket
    return cfg
