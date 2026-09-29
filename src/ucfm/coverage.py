"""Draw the public field-site map from datasets/registry.yaml."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from ucfm.config import ROOT

COVERAGE_START = "<!-- ucfm:coverage:start -->"
COVERAGE_END = "<!-- ucfm:coverage:end -->"
MAP_PATH = ROOT / "assets" / "coverage.png"
LAND_PATH = ROOT / "assets" / "basemap" / "ne_110m_land.geojson"
STATES_PATH = ROOT / "assets" / "basemap" / "ne_110m_us_states.geojson"
PUBLIC_README = ROOT / "README.md"

WATER = "#c5d6e0"
LAND = "#e4efe3"
COAST = "#7d9480"
STATE_EDGE = "#6d8a74"

# One clay, then a short set of muted inks. Same order every run.
SITE_COLORS = ("#9c4221", "#1f4d3a", "#2c5f8a", "#6b5428", "#5c4d7a", "#3d6b5a")


def iter_sites(datasets: list[dict]) -> list[dict]:
    """One row per mapped field, west to east.

    A dataset may list several `sites`, or a single `lat` / `lon` pair.
    Entries without coordinates are skipped — they cannot be drawn.
    """
    rows: list[dict] = []
    for ds in datasets:
        sites = list(ds.get("sites") or [])
        if not sites and ds.get("lat") is not None and ds.get("lon") is not None:
            sites = [
                {
                    "name": ds.get("place") or ds.get("title") or ds.get("id") or "site",
                    "lat": ds.get("lat"),
                    "lon": ds.get("lon"),
                }
            ]
        for site in sites:
            lat, lon = site.get("lat"), site.get("lon")
            if lat is None or lon is None:
                continue
            rows.append(
                {
                    "dataset_id": str(ds.get("id") or ""),
                    "title": str(ds.get("title") or ds.get("id") or ""),
                    "crop": str(ds.get("crop") or ""),
                    "year": ds.get("year"),
                    "source": str(ds.get("source") or ""),
                    "license": str(ds.get("license") or ""),
                    "note": str(ds.get("note") or ""),
                    "name": str(site.get("name") or ds.get("id") or "site"),
                    "lat": float(lat),
                    "lon": float(lon),
                }
            )
    rows.sort(key=lambda row: (row["lon"], row["lat"], row["name"]))
    return rows


def view_window(lats: list[float], lons: list[float]) -> tuple[float, float, float, float, bool]:
    """Return min_lon, max_lon, min_lat, max_lat, and whether the map is zoomed in.

    A tight cluster (one region) is framed with padding. Sites spread across
    continents fall back to a world view.
    """
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)
    span_lat = max(max_lat - min_lat, 0.0)
    span_lon = max(max_lon - min_lon, 0.0)
    if span_lat > 50 or span_lon > 100:
        return (-170.0, 190.0, -56.0, 78.0, False)
    lat_pad = max(1.5, span_lat * 0.35, (10.0 - span_lat) / 2.0)
    lon_pad = max(2.0, span_lon * 0.22, (24.0 - span_lon) / 2.0)
    return (
        min_lon - lon_pad,
        max_lon + lon_pad,
        min_lat - lat_pad,
        max_lat + lat_pad,
        True,
    )


def _outer_rings(path: Path) -> list[list[tuple[float, float]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rings: list[list[tuple[float, float]]] = []
    for feature in data.get("features") or []:
        geom = feature.get("geometry") or {}
        gtype = geom.get("type")
        coords = geom.get("coordinates") or []
        polygons = [coords] if gtype == "Polygon" else coords if gtype == "MultiPolygon" else []
        for polygon in polygons:
            if not polygon:
                continue
            rings.append([(float(x), float(y)) for x, y in polygon[0]])
    return rings


def build_coverage_section(datasets: list[dict]) -> str:
    caption = (
        "_Each dot is a field site listed in `datasets/registry.yaml`. The frame follows the sites._"
        if iter_sites(datasets)
        else "_No field sites with coordinates yet._"
    )
    return "\n".join(
        [
            COVERAGE_START,
            "![Field sites in the training pool](assets/coverage.png)",
            "",
            caption,
            COVERAGE_END,
        ]
    )


def _label_offsets(sites: list[dict]) -> list[tuple[float, float]]:
    """Nudge overlapping names above or below the dot."""
    offsets: list[tuple[float, float]] = []
    for site in sites:
        dy = 7.0
        for other in sites:
            if other is site:
                continue
            if abs(site["lon"] - other["lon"]) < 1.6 and abs(site["lat"] - other["lat"]) < 1.3:
                dy = 8.0 if site["lat"] >= other["lat"] else -12.0
                break
        offsets.append((0.0, dy))
    return offsets


def render_coverage_map(
    datasets: list[dict],
    dest: Path | None = None,
    land_path: Path | None = None,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import FuncFormatter

    dest = dest or MAP_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    sites = iter_sites(datasets)
    rings = _outer_rings(land_path or LAND_PATH)
    states = _outer_rings(STATES_PATH) if STATES_PATH.exists() else []

    if sites:
        min_lon, max_lon, min_lat, max_lat, zoomed = view_window(
            [s["lat"] for s in sites],
            [s["lon"] for s in sites],
        )
    else:
        min_lon, max_lon, min_lat, max_lat, zoomed = (-170.0, 190.0, -56.0, 78.0, False)

    fig = plt.figure(figsize=(11.2, 6.15), dpi=160, facecolor="#f4f6f3")
    ax = fig.add_axes([0.055, 0.08, 0.92, 0.86])
    ax.set_facecolor(WATER)
    _draw_land(ax, rings, facecolor=LAND, edgecolor=COAST, linewidth=0.45)
    _draw_land(ax, states, facecolor="none", edgecolor=STATE_EDGE, linewidth=0.55)
    _draw_sites(ax, sites)

    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)
    mid_lat = (min_lat + max_lat) / 2.0
    ax.set_aspect(1.0 / max(math.cos(math.radians(mid_lat)), 0.2), adjustable="box")
    ax.set_xlabel("Longitude", fontsize=8, color="#3e4a40")
    ax.set_ylabel("Latitude", fontsize=8, color="#3e4a40")
    ax.tick_params(labelsize=7.5, colors="#3e4a40", length=0, pad=2)
    ax.xaxis.set_major_formatter(FuncFormatter(_geo_tick))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:.0f}°"))
    ax.grid(True, color="#c5d1c8", linewidth=0.45, zorder=0)
    for spine in ax.spines.values():
        spine.set_color("#c5d1c8")
        spine.set_linewidth(0.6)

    if zoomed:
        inset = ax.inset_axes([0.012, 0.63, 0.30, 0.34])
        inset.set_facecolor(WATER)
        _draw_land(inset, rings, facecolor=LAND, edgecolor=COAST, linewidth=0.15)
        _draw_sites(inset, sites, size=8)
        inset.set_xlim(-168, 192)
        inset.set_ylim(-56, 78)
        inset.set_xticks([])
        inset.set_yticks([])
        inset.add_patch(
            Rectangle(
                (min_lon, min_lat),
                max_lon - min_lon,
                max_lat - min_lat,
                fill=False,
                edgecolor="#9c4221",
                linewidth=0.7,
                zorder=4,
            )
        )
        for spine in inset.spines.values():
            spine.set_color("#8fa096")
            spine.set_linewidth(0.6)

    fig.savefig(dest, dpi=160)
    plt.close(fig)
    return dest


def _draw_land(
    ax,
    rings: list[list[tuple[float, float]]],
    facecolor: str,
    edgecolor: str,
    linewidth: float,
) -> None:
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Polygon as MplPolygon

    if not rings:
        return
    patches = [MplPolygon(ring, closed=True) for ring in rings if len(ring) >= 3]
    ax.add_collection(
        PatchCollection(
            patches,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            zorder=1,
        )
    )


def _draw_sites(ax, sites: list[dict], size: float = 42) -> None:
    from matplotlib.patheffects import withStroke

    if not sites:
        return
    dataset_ids = list(dict.fromkeys(s["dataset_id"] for s in sites))
    color_for = {
        dataset_id: SITE_COLORS[i % len(SITE_COLORS)] for i, dataset_id in enumerate(dataset_ids)
    }
    offsets = _label_offsets(sites) if size >= 20 else [(0.0, 0.0)] * len(sites)
    for site, (dx, dy) in zip(sites, offsets):
        color = color_for[site["dataset_id"]]
        ax.scatter(
            [site["lon"]],
            [site["lat"]],
            s=size,
            c=color,
            edgecolors="#f3f5f2",
            linewidths=0.8 if size >= 20 else 0.3,
            zorder=4,
        )
        if size < 20:
            continue
        label = site["name"].split(",")[0].strip()
        text = ax.annotate(
            label,
            (site["lon"], site["lat"]),
            textcoords="offset points",
            xytext=(dx, dy),
            ha="center",
            va="bottom" if dy >= 0 else "top",
            fontsize=7.5,
            color="#1c2820",
            zorder=5,
        )
        text.set_path_effects([withStroke(linewidth=2.6, foreground="#f3f5f2")])
    if len(dataset_ids) > 1 and size >= 20:
        handles = []
        import matplotlib.pyplot as plt

        for dataset_id in dataset_ids:
            title = next(s["title"] for s in sites if s["dataset_id"] == dataset_id)
            handles.append(
                plt.Line2D(
                    [],
                    [],
                    marker="o",
                    color="none",
                    markerfacecolor=color_for[dataset_id],
                    markeredgecolor="#f3f5f2",
                    markersize=6,
                    label=title,
                )
            )
        ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=7.5)


def _geo_tick(value, _pos) -> str:
    value = float(value)
    if value < -0.5:
        return f"{abs(value):.0f}°W"
    if value > 0.5:
        return f"{value:.0f}°E"
    return "0°"


def update_coverage_readme(
    datasets: list[dict],
    readme: Path | None = None,
    chart: Path | None = None,
) -> Path:
    readme_path = readme or PUBLIC_README
    render_coverage_map(datasets, chart)
    section = build_coverage_section(datasets)
    text = readme_path.read_text(encoding="utf-8")
    pattern = re.compile(
        re.escape(COVERAGE_START) + r".*?" + re.escape(COVERAGE_END),
        flags=re.DOTALL,
    )
    if not pattern.search(text):
        raise RuntimeError(
            f"{readme_path} is missing {COVERAGE_START} … {COVERAGE_END} markers."
        )
    readme_path.write_text(pattern.sub(lambda _m: section, text, count=1), encoding="utf-8")
    return readme_path
