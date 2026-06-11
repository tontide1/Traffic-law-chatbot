import pytest

from scripts.run_regression_check import _available_categories, _filter_items


def test_available_categories_are_derived_from_items():
    items = [
        {"category": "graph_strength"},
        {"category": "exactness"},
        {"category": "graph_strength"},
        {"category": ""},
        {},
    ]

    assert _available_categories(items) == ["exactness", "graph_strength"]


def test_filter_items_accepts_fixture_defined_category_not_hardcoded():
    items = [
        {"category": "custom_category", "question": "custom"},
        {"category": "exactness", "question": "exact"},
    ]

    assert _filter_items(items, index=None, category="custom_category") == [items[0]]


def test_filter_items_rejects_unknown_category_with_available_categories():
    items = [{"category": "exactness"}, {"category": "graph_strength"}]

    with pytest.raises(ValueError, match="available categories: exactness, graph_strength"):
        _filter_items(items, index=None, category="missing")
