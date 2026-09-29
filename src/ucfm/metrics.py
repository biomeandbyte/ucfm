from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from ucfm.config import ROOT
from ucfm.storage import Store, count_pool_images

METRICS_KEY = "logs/metrics.jsonl"
BASELINE_KEY = "logs/baseline.json"
README_PATH = ROOT / "README.md"
CHART_PATH = ROOT / "assets" / "performance.png"
METRICS_START = "<!-- ucfm:metrics:start -->"
METRICS_END = "<!-- ucfm:metrics:end -->"


def append_metrics(store: Store, scores: dict) -> dict:
    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "n_images": scores.get("n_images", count_pool_images(store)),
        "score_ws": scores.get("score_ws"),
        "score_gs": scores.get("score_gs"),
        "baseline_ws": scores.get("baseline_ws"),
        "baseline_gs": scores.get("baseline_gs"),
        "backbone": scores.get("backbone", "dinov2"),
        "note": scores.get("note", ""),
    }
    store.append_jsonl(METRICS_KEY, row)
    return row


def read_metrics(store: Store) -> list[dict]:
    path = store.local_path(METRICS_KEY)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def save_baseline(store: Store, score_ws: float, score_gs: float) -> dict:
    blob = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "score_ws": score_ws,
        "score_gs": score_gs,
        "note": "DINOv2 ImageNet init + linear probe (no pool pretrain)",
    }
    store.write_json(BASELINE_KEY, blob)
    return blob


def load_baseline(store: Store) -> dict | None:
    path = store.local_path(BASELINE_KEY)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def render_chart(metrics: list[dict], baseline: dict, dest: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from matplotlib.ticker import FuncFormatter

    dest.parent.mkdir(parents=True, exist_ok=True)
    xs = [m.get("n_images", 0) for m in metrics if m.get("score_ws") is not None]
    ws = [m.get("score_ws") for m in metrics if m.get("score_ws") is not None]
    gs = [m.get("score_gs") for m in metrics if m.get("score_gs") is not None]
    b_ws = baseline.get("score_ws")
    b_gs = baseline.get("score_gs")

    paper, ink, grid = "#f4f6f3", "#1c2820", "#d3dbd4"
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), dpi=160, facecolor=paper)
    for ax, ys, title, color, bline in (
        (axes[0], ws, "Water stress", "#9c4221", b_ws),
        (axes[1], gs, "Growth stage", "#1f4d3a", b_gs),
    ):
        ax.set_facecolor(paper)
        if bline is not None and xs:
            ax.axhline(
                float(bline), color="#8a8378", linestyle=(0, (4, 3)), linewidth=1.1,
                label="Baseline (no pool pretraining)", zorder=2,
            )
        if xs and ys:
            ax.plot(
                xs, ys, "o-", color=color, linewidth=1.8, markersize=5.5,
                markeredgecolor=paper, markeredgewidth=1.0, label="Pool-pretrained", zorder=3,
            )
        ax.set_title(title, loc="left", fontsize=11, color=ink, pad=8)
        ax.set_xlabel("Images in pretraining pool", fontsize=8.5, color="#3e4a40")
        ax.set_ylabel("Validation accuracy (linear probe)", fontsize=8.5, color="#3e4a40")
        ax.set_ylim(0, 1)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:,.0f}"))
        ax.tick_params(labelsize=8, colors="#3e4a40", length=0, pad=4)
        ax.grid(True, axis="y", color=grid, linewidth=0.6, zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(grid)
        if xs:
            ax.legend(loc="lower right", fontsize=7.5, frameon=False)
    fig.tight_layout(w_pad=3)
    fig.savefig(dest, bbox_inches="tight", facecolor=paper)
    plt.close(fig)
    return dest


def build_readme_section(metrics: list[dict], baseline: dict) -> str:
    updated = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return "\n".join(
        [
            METRICS_START,
            "![Validation accuracy on water stress and growth stage vs images in the "
            "pretraining pool](assets/performance.png)",
            "",
            "_Validation accuracy of a linear probe on frozen features, by number of images "
            "in the pretraining pool. Dashed line: DINOv2 before any pool pretraining. "
            f"Updated {updated}._",
            METRICS_END,
        ]
    )


def update_readme(store: Store, readme: Path | None = None, chart: Path | None = None) -> Path:
    readme_path = readme or README_PATH
    chart_path = chart or CHART_PATH
    baseline = load_baseline(store) or {}
    metrics = read_metrics(store)
    render_chart(metrics, baseline, chart_path)
    section = build_readme_section(metrics, baseline)
    text = readme_path.read_text(encoding="utf-8")
    pattern = re.compile(
        re.escape(METRICS_START) + r".*?" + re.escape(METRICS_END),
        flags=re.DOTALL,
    )
    if not pattern.search(text):
        raise RuntimeError(
            f"README missing {METRICS_START} … {METRICS_END} markers — cannot update."
        )
    readme_path.write_text(pattern.sub(section, text, count=1), encoding="utf-8")
    return readme_path
