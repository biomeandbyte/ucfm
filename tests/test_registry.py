from ucfm.registry import load_registry, registry_ids


def test_empty_registry():
    assert load_registry() == []
    assert registry_ids() == []
