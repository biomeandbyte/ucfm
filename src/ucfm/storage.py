from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from ucfm.config import FileConfig, ROOT

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def _local_root(cfg: FileConfig) -> Path:
    root = Path(cfg.storage.local_root)
    if not root.is_absolute():
        root = ROOT / root
    root.mkdir(parents=True, exist_ok=True)
    return root


@dataclass
class Store:
    cfg: FileConfig

    @property
    def root(self) -> Path:
        return _local_root(self.cfg)

    def local_path(self, key: str) -> Path:
        p = self.root / key
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def put_file(self, key: str, src: Path) -> str:
        dest = self.local_path(key)
        if src.resolve() != dest.resolve():
            dest.write_bytes(src.read_bytes())
        if self.cfg.storage.backend == "s3":
            self._s3().upload_file(str(dest), self.cfg.storage.bucket, self._s3_key(key))
        return str(dest)

    def exists(self, key: str) -> bool:
        return self.local_path(key).exists()

    def append_jsonl(self, key: str, row: dict) -> None:
        path = self.local_path(key)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if self.cfg.storage.backend == "s3":
            self._s3().upload_file(str(path), self.cfg.storage.bucket, self._s3_key(key))

    def read_json(self, key: str, default):
        path = self.local_path(key)
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))

    def write_json(self, key: str, obj) -> None:
        path = self.local_path(key)
        path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if self.cfg.storage.backend == "s3":
            self._s3().upload_file(str(path), self.cfg.storage.bucket, self._s3_key(key))

    def rglob(self, pattern: str) -> Iterator[Path]:
        yield from self.root.rglob(pattern)

    def _s3_key(self, key: str) -> str:
        prefix = self.cfg.storage.prefix.strip("/")
        return f"{prefix}/{key}" if prefix else key

    def _s3(self):
        import boto3

        return boto3.client("s3")


def _is_pool_image(path: Path, pool: Path) -> bool:
    if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
        return False
    rel = path.relative_to(pool)
    if "raw" in rel.parts[:-1]:
        return False
    return True


def list_pool_images(store: Store) -> list[Path]:
    pool = store.local_path("pool")
    if not pool.exists():
        return []
    return sorted(p for p in pool.rglob("*") if _is_pool_image(p, pool))


def count_pool_images(store: Store) -> int:
    return len(list_pool_images(store))
