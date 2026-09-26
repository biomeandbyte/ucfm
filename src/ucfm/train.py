"""DINOv2 pretrain on pooled UAV images + linear probes on ws / gs."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from ucfm.storage import IMAGE_EXTS, Store, count_pool_images, list_pool_images

CKPT_NAME = "checkpoints/dinov2_continual.pt"
_IMAGENET_NORM = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))


class RandomSquareCrop:
    """Crop a square of side min(W, H) at a random offset along the long axis."""

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        side = min(w, h)
        left = 0 if w == side else random.randint(0, w - side)
        top = 0 if h == side else random.randint(0, h - side)
        return img.crop((left, top, left + side, top + side))


class CenterSquareCrop:
    """Crop a square of side min(W, H) from the center (val / test)."""

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        return img.crop((left, top, left + side, top + side))


def _train_tfm(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            RandomSquareCrop(),
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            _IMAGENET_NORM,
        ]
    )


def _eval_tfm(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            CenterSquareCrop(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            _IMAGENET_NORM,
        ]
    )


class ImageFolderTask(Dataset):
    """eval/{ws|gs}/{train|val|test}/{class}/*"""

    def __init__(self, root: Path, split: str, image_size: int, train: bool):
        self.samples: list[tuple[Path, int]] = []
        self.classes: list[str] = []
        split_dir = root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"missing eval split: {split_dir}")
        class_dirs = sorted(p for p in split_dir.iterdir() if p.is_dir())
        self.classes = [p.name for p in class_dirs]
        class_to_idx = {c: i for i, c in enumerate(self.classes)}
        for cdir in class_dirs:
            for img in sorted(cdir.rglob("*")):
                if img.is_file() and img.suffix.lower() in IMAGE_EXTS:
                    self.samples.append((img, class_to_idx[cdir.name]))
        if not self.samples:
            raise FileNotFoundError(f"no images under {split_dir}")
        self.tfm = _train_tfm(image_size) if train else _eval_tfm(image_size)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        path, y = self.samples[i]
        img = Image.open(path).convert("RGB")
        return self.tfm(img), y


class PoolPairs(Dataset):
    def __init__(self, paths: list[Path], image_size: int):
        self.paths = paths
        self.tfm = _train_tfm(image_size)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.tfm(img), self.tfm(img)


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_dinov2(store: Store, pretrained: bool = True):
    from transformers import Dinov2Model

    name = store.cfg.train.dinov2_model
    model = Dinov2Model.from_pretrained(name) if pretrained else Dinov2Model.from_pretrained(name)
    ckpt = store.local_path(CKPT_NAME)
    if ckpt.exists():
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"], strict=False)
    return model


def save_dinov2(store: Store, model: nn.Module, extra: dict | None = None) -> str:
    path = store.local_path(CKPT_NAME)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {"model": model.state_dict(), "dinov2_model": store.cfg.train.dinov2_model}
    if extra:
        blob.update(extra)
    torch.save(blob, path)
    if store.cfg.storage.backend == "s3":
        store.put_file(CKPT_NAME, path)
    return str(path)


@torch.no_grad()
def _embed(model: nn.Module, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    xs, ys = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        out = model(xb).last_hidden_state[:, 0]
        xs.append(out.cpu().numpy())
        ys.append(yb.numpy())
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def linear_probe(store: Store, task: str, use_continual_ckpt: bool) -> float:
    cfg = store.cfg
    root = store.local_path(cfg.eval.ws_dir if task == "ws" else cfg.eval.gs_dir)
    device = _device()
    from transformers import Dinov2Model

    if use_continual_ckpt and store.local_path(CKPT_NAME).exists():
        model = load_dinov2(store, pretrained=True)
    else:
        model = Dinov2Model.from_pretrained(cfg.train.dinov2_model)
    model = model.to(device)
    train_ds = ImageFolderTask(root, "train", cfg.train.image_size, train=True)
    val_ds = ImageFolderTask(root, "val", cfg.train.image_size, train=False)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.probe_batch_size, shuffle=False, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.train.probe_batch_size, shuffle=False, num_workers=0
    )
    Xtr, ytr = _embed(model, train_loader, device)
    Xva, yva = _embed(model, val_loader, device)
    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000)),
        ]
    )
    clf.fit(Xtr, ytr)
    return float(clf.score(Xva, yva))


def run_baseline(store: Store) -> dict:
    score_ws = linear_probe(store, "ws", use_continual_ckpt=False)
    score_gs = linear_probe(store, "gs", use_continual_ckpt=False)
    return {"score_ws": score_ws, "score_gs": score_gs, "backbone": store.cfg.train.dinov2_model}


def pretrain_dinov2(store: Store) -> dict:
    paths = list_pool_images(store)
    if len(paths) < 2:
        raise RuntimeError(f"need >=2 pool images to pretrain; found {len(paths)}")
    cfg = store.cfg
    device = _device()
    model = load_dinov2(store, pretrained=True).to(device)
    model.train()
    ds = PoolPairs(paths, cfg.train.image_size)
    dl = DataLoader(
        ds,
        batch_size=min(cfg.train.batch_size, len(ds)),
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)
    loss_fn = nn.CosineEmbeddingLoss()
    steps = 0
    target = cfg.train.pretrain_steps
    last_loss = None
    while steps < target:
        for v1, v2 in dl:
            v1, v2 = v1.to(device), v2.to(device)
            opt.zero_grad()
            f1 = model(v1).last_hidden_state[:, 0]
            f2 = model(v2).last_hidden_state[:, 0]
            ones = torch.ones(v1.size(0), device=device)
            loss = loss_fn(f1, f2, ones)
            loss.backward()
            opt.step()
            last_loss = float(loss.item())
            steps += 1
            if steps >= target:
                break
    path = save_dinov2(store, model, extra={"steps": steps, "n_pool": len(paths)})
    return {
        "backbone": cfg.train.dinov2_model,
        "steps": steps,
        "n_pool": len(paths),
        "loss": last_loss,
        "device": device,
        "ckpt": path,
    }


def eval_ws_gs(store: Store) -> dict:
    return {
        "n_images": count_pool_images(store),
        "score_ws": linear_probe(store, "ws", use_continual_ckpt=True),
        "score_gs": linear_probe(store, "gs", use_continual_ckpt=True),
        "backbone": store.cfg.train.dinov2_model,
    }
