import pytest

from app.rag.metadata import (
    build_metadata_filter,
    document_type_filter,
    normalize_document_type,
)


@pytest.mark.parametrize("canonical", ["10-K", "10-Q", "8-K", "20-F", "40-F"])
def test_normalize_preserves_canonical_document_types(canonical: str) -> None:
    assert normalize_document_type(canonical) == canonical


@pytest.mark.parametrize(
    "legacy,canonical",
    [
        ("10K", "10-K"),
        ("10Q", "10-Q"),
        ("8K", "8-K"),
        ("20F", "20-F"),
        ("40F", "40-F"),
    ],
)
def test_normalize_maps_compact_legacy_aliases(legacy: str, canonical: str) -> None:
    assert normalize_document_type(legacy) == canonical


@pytest.mark.parametrize("value,canonical", [(" 10k ", "10-K"), ("10-q", "10-Q")])
def test_normalize_accepts_case_and_surrounding_whitespace(
    value: str, canonical: str
) -> None:
    assert normalize_document_type(value) == canonical


def test_normalize_accepts_none() -> None:
    assert normalize_document_type(None) is None


@pytest.mark.parametrize("value", ["", "  ", "filing", "annual report", "13F"])
def test_normalize_rejects_blank_or_unsupported_values(value: str) -> None:
    with pytest.raises(ValueError, match="Unsupported document type"):
        normalize_document_type(value)


@pytest.mark.parametrize(
    "canonical,legacy",
    [
        ("10-K", "10K"),
        ("10-Q", "10Q"),
        ("8-K", "8K"),
        ("20-F", "20F"),
        ("40-F", "40F"),
    ],
)
def test_document_type_filter_includes_canonical_and_legacy(
    canonical: str, legacy: str
) -> None:
    expected = {"document_type": {"$in": [canonical, legacy]}}
    assert document_type_filter(canonical) == expected
    assert document_type_filter(legacy) == expected


def test_build_metadata_filter_returns_none_without_conditions() -> None:
    assert build_metadata_filter() is None


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({"company": "Apple"}, {"company": {"$eq": "Apple"}}),
        ({"ticker": "AAPL"}, {"ticker": {"$eq": "AAPL"}}),
        ({"fiscal_year": 2025}, {"fiscal_year": {"$eq": 2025}}),
        (
            {"document_type": "10-K"},
            {"document_type": {"$in": ["10-K", "10K"]}},
        ),
        (
            {"document_type": "10K"},
            {"document_type": {"$in": ["10-K", "10K"]}},
        ),
    ],
)
def test_build_metadata_filter_returns_exact_single_condition(
    kwargs: dict, expected: dict
) -> None:
    assert build_metadata_filter(**kwargs) == expected


def test_build_metadata_filter_preserves_condition_order() -> None:
    assert build_metadata_filter(
        company="Apple", ticker="AAPL", fiscal_year=2025, document_type="10-K"
    ) == {
        "$and": [
            {"company": {"$eq": "Apple"}},
            {"ticker": {"$eq": "AAPL"}},
            {"fiscal_year": {"$eq": 2025}},
            {"document_type": {"$in": ["10-K", "10K"]}},
        ]
    }


def test_build_metadata_filter_rejects_unsupported_document_type() -> None:
    with pytest.raises(ValueError, match="Unsupported document type"):
        build_metadata_filter(company="Apple", document_type="filing")
