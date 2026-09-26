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


def _fmt_score(value) -> str:
    if value is None:
        return "—"
    return f"{float(value):.3f}"


def _metrics_table(metrics: list[dict], baseline: dict) -> str:
    lines = [
        "| # pool images | ws (val acc) | gs (val acc) | note |",
        "|--------------:|-------------:|-------------:|------|",
    ]
    if baseline:
        lines.append(
            f"| 0 (baseline) | {_fmt_score(baseline.get('score_ws'))} | "
            f"{_fmt_score(baseline.get('score_gs'))} | no pool pretrain |"
        )
    for row in metrics:
        if row.get("n_images") == 0 and row.get("note", "").startswith("baseline"):
            continue
        lines.append(
            f"| {row.get('n_images', '—')} | {_fmt_score(row.get('score_ws'))} | "
            f"{_fmt_score(row.get('score_gs'))} | {row.get('note', '')} |"
        )
    if len(lines) == 2:
        lines.append("| — | — | — | no runs yet |")
    return "\n".join(lines)


def render_chart(metrics: list[dict], baseline: dict, dest: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dest.parent.mkdir(parents=True, exist_ok=True)
    xs = [m.get("n_images", 0) for m in metrics if m.get("score_ws") is not None]
    ws = [m.get("score_ws") for m in metrics if m.get("score_ws") is not None]
    gs = [m.get("score_gs") for m in metrics if m.get("score_gs") is not None]
    b_ws = baseline.get("score_ws")
    b_gs = baseline.get("score_gs")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=120)
    for ax, ys, title, color, bline in (
        (axes[0], ws, "Water stress (ws)", "#2a9d6f", b_ws),
        (axes[1], gs, "Growth stage (gs)", "#3d8fd1", b_gs),
    ):
        if xs and ys:
            ax.plot(xs, ys, "o-", color=color, label=title)
        if bline is not None and xs:
            ax.axhline(float(bline), color="#c9a227", linestyle="--", label="baseline")
        ax.set_title(title)
        ax.set_xlabel("# pretrained images")
        ax.set_ylabel("val accuracy")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        if xs:
            ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(dest, bbox_inches="tight")
    plt.close(fig)
    return dest


def build_readme_section(metrics: list[dict], baseline: dict) -> str:
    updated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return "\n".join(
        [
            METRICS_START,
            f"_Updated {updated}. Dashed gold = baseline (DINOv2, no pool pretrain)._",
            "",
            _metrics_table(metrics, baseline),
            "",
            "![ws/gs vs pool size](assets/performance.png)",
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
