from ucfm.registry import load_registry, registry_ids


def test_missing_registry(tmp_path):
    missing = tmp_path / "registry.yaml"
    assert load_registry(missing) == []
    assert registry_ids(missing) == []


def test_registry_ids():
    assert "shrestha_2025_uav" in registry_ids()
