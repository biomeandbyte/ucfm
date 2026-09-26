from pathlib import Path

from PIL import Image

from ucfm.metrics import METRICS_END, METRICS_START, build_readme_section, update_readme
from ucfm.storage import Store
from ucfm.config import FileConfig
from ucfm.train import CenterSquareCrop, RandomSquareCrop


def test_build_readme_section():
    section = build_readme_section(
        [{"n_images": 100, "score_ws": 0.5, "score_gs": 0.6, "note": "test"}],
        {"score_ws": 0.4, "score_gs": 0.55},
    )
    assert METRICS_START in section
    assert METRICS_END in section
    assert "100" in section


def test_update_readme(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(f"# test\n\n{METRICS_START}\nPLACEHOLDER\n{METRICS_END}\n", encoding="utf-8")
    chart = tmp_path / "assets" / "performance.png"
    store = Store(FileConfig())
    update_readme(store, readme=readme, chart=chart)
    assert chart.exists()
    text = readme.read_text(encoding="utf-8")
    assert "PLACEHOLDER" not in text
    assert "assets/performance.png" in text


def test_square_crops_on_rectangles():
    wide = Image.new("RGB", (300, 100), color=(10, 20, 30))
    tall = Image.new("RGB", (80, 240), color=(1, 2, 3))
    square = Image.new("RGB", (50, 50), color=(4, 5, 6))
    assert RandomSquareCrop()(wide).size == (100, 100)
    assert CenterSquareCrop()(wide).size == (100, 100)
    assert CenterSquareCrop()(tall).size == (80, 80)
    assert CenterSquareCrop()(square).size == (50, 50)
