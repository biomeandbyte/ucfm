"""Baseline + train/eval orchestration."""
from __future__ import annotations

from ucfm.metrics import append_metrics, load_baseline, save_baseline, update_readme
from ucfm.storage import Store, count_pool_images
from ucfm.train import eval_ws_gs, pretrain_dinov2, run_baseline

LAST_TRAINED_KEY = "state/last_trained_n_images.json"


def ensure_eval_present(store: Store) -> None:
    ws = store.local_path(store.cfg.eval.ws_dir)
    gs = store.local_path(store.cfg.eval.gs_dir)
    if not (ws / "train").exists() or not (gs / "train").exists():
        raise FileNotFoundError(
            f"Frozen eval missing. Expected {ws}/train and {gs}/train. "
            "Sync from S3: `ucfm sync-eval` or copy splits into data/eval/."
        )


def run_baseline_phase(store: Store, *, force: bool = False) -> dict:
    ensure_eval_present(store)
    if not force and load_baseline(store):
        return load_baseline(store)  # type: ignore
    scores = run_baseline(store)
    blob = save_baseline(store, scores["score_ws"], scores["score_gs"])
    append_metrics(
        store,
        {
            "n_images": 0,
            "score_ws": scores["score_ws"],
            "score_gs": scores["score_gs"],
            "baseline_ws": scores["score_ws"],
            "baseline_gs": scores["score_gs"],
            "note": "baseline (n_images=0, no pool pretrain)",
        },
    )
    update_readme(store)
    return blob


def train_and_eval(store: Store) -> dict:
    ensure_eval_present(store)
    n = count_pool_images(store)
    if n < 2:
        raise RuntimeError(
            f"pool has {n} images — upload data under pool/<id>/ (any subdirs) before training"
        )
    baseline = load_baseline(store) or run_baseline_phase(store)
    pre = pretrain_dinov2(store)
    scores = eval_ws_gs(store)
    row = append_metrics(
        store,
        {
            **scores,
            "baseline_ws": baseline.get("score_ws"),
            "baseline_gs": baseline.get("score_gs"),
            "note": f"after pretrain epochs={pre.get('epochs')} steps={pre.get('steps')}",
        },
    )
    update_readme(store)
    store.write_json(LAST_TRAINED_KEY, {"n_images": n, "ts": row["ts"]})
    return {"pretrain": pre, "eval": scores, "logged": row}
