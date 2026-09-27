"""Canonical document metadata and legacy-compatible query filters."""

from typing import Any, Literal, TypeAlias


CanonicalDocumentType: TypeAlias = Literal[
    "10-K", "10-Q", "8-K", "20-F", "40-F"
]

_CANONICAL_BY_VALUE: dict[str, CanonicalDocumentType] = {
    "10-K": "10-K",
    "10K": "10-K",
    "10-Q": "10-Q",
    "10Q": "10-Q",
    "8-K": "8-K",
    "8K": "8-K",
    "20-F": "20-F",
    "20F": "20-F",
    "40-F": "40-F",
    "40F": "40-F",
}


def normalize_document_type(value: str | None) -> CanonicalDocumentType | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("Unsupported document type.")
    canonical = _CANONICAL_BY_VALUE.get(value.strip().upper())
    if canonical is None:
        raise ValueError(f"Unsupported document type: {value!r}.")
    return canonical


def document_type_filter(value: str) -> dict[str, Any]:
    canonical = normalize_document_type(value)
    if canonical is None:
        raise ValueError("Document type filter requires a filing form.")
    return {"document_type": {"$in": [canonical, canonical.replace("-", "")]}}


def build_metadata_filter(
    company: str | None = None,
    ticker: str | None = None,
    fiscal_year: int | None = None,
    document_type: str | None = None,
) -> dict[str, Any] | None:
    conditions: list[dict[str, Any]] = []
    if company is not None:
        conditions.append({"company": {"$eq": company}})
    if ticker is not None:
        conditions.append({"ticker": {"$eq": ticker}})
    if fiscal_year is not None:
        conditions.append({"fiscal_year": {"$eq": fiscal_year}})
    if document_type is not None:
        conditions.append(document_type_filter(document_type))

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}
