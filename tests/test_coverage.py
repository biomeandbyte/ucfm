from ucfm.coverage import (
    COVERAGE_END,
    COVERAGE_START,
    build_coverage_section,
    iter_sites,
    update_coverage_readme,
    view_window,
)
from ucfm.registry import load_registry


def test_registry_sites_are_the_five_uav_fields():
    sites = iter_sites(load_registry())
    assert [s["name"] for s in sites] == [
        "Scottsbluff, NE",
        "Lincoln, NE",
        "Missouri Valley, IA",
        "Ames, IA",
        "Crawfordsville, IA",
    ]
    window = view_window([s["lat"] for s in sites], [s["lon"] for s in sites])
    min_lon, max_lon, min_lat, max_lat, zoomed = window
    assert zoomed
    assert min_lon < -103.70 < max_lon
    assert min_lat < 40.86 < max_lat


def test_single_lat_lon_and_missing_coordinates():
    sites = iter_sites(
        [
            {"id": "no-coords", "title": "Skipped"},
            {"id": "one-field", "title": "One field", "place": "Nairobi", "lat": -1.29, "lon": 36.82},
        ]
    )
    assert len(sites) == 1
    assert sites[0]["name"] == "Nairobi"
    assert view_window([s["lat"] for s in sites], [s["lon"] for s in sites])[4] is True


def test_spread_sites_use_a_world_view():
    _min_lon, _max_lon, _min_lat, _max_lat, zoomed = view_window(
        [-1.3, 41.8],
        [36.8, -103.7],
    )
    assert zoomed is False


def test_coverage_section_is_the_map_only():
    section = build_coverage_section(load_registry())
    assert COVERAGE_START in section
    assert "assets/coverage.png" in section
    assert "doi:" not in section
    assert "|" not in section


def test_update_coverage_readme(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(f"# t\n\n{COVERAGE_START}\nPLACEHOLDER\n{COVERAGE_END}\n", encoding="utf-8")
    chart = tmp_path / "coverage.png"
    update_coverage_readme(
        [{"id": "demo", "title": "Demo", "lat": 41.85, "lon": -103.7, "source": "https://example.com/data"}],
        readme=readme,
        chart=chart,
    )
    text = readme.read_text(encoding="utf-8")
    assert "PLACEHOLDER" not in text
    assert "assets/coverage.png" in text
    assert chart.exists()
    assert chart.stat().st_size > 1000
