"""Strict structured report draft model contracts."""

import pytest
from pydantic import ValidationError

from app.report.synthesis_models import ReportClaim, ReportDraft, ReportSection


def _claim(**overrides: object) -> ReportClaim:
    values = {"task_id": "apple_risks", "text": "Apple described a risk.", "evidence_ids": ("E1",)}
    values.update(overrides)
    return ReportClaim(**values)


def _section(**overrides: object) -> ReportSection:
    values = {"heading": "Risks", "claims": (_claim(),)}
    values.update(overrides)
    return ReportSection(**values)


def _draft(**overrides: object) -> ReportDraft:
    values = {"title": "Apple research", "sections": (_section(),)}
    values.update(overrides)
    return ReportDraft(**values)


def test_report_claim_valid_and_frozen() -> None:
    claim = _claim(evidence_ids=("E1", "E12"))

    assert claim.task_id == "apple_risks"
    assert claim.evidence_ids == ("E1", "E12")
    with pytest.raises(ValidationError):
        claim.text = "Changed"


@pytest.mark.parametrize(
    "overrides",
    [
        {"extra": "forbidden"},
        {"text": " \t\n"},
        {"text": "x" * 1501},
        {"task_id": "Bad-ID"},
        {"task_id": "x" * 65},
        {"task_id": 42},
        {"evidence_ids": ()},
        {"evidence_ids": ("E0",)},
        {"evidence_ids": ("E01",)},
        {"evidence_ids": ("E1", "E1")},
        {"evidence_ids": tuple(f"E{index}" for index in range(1, 14))},
        {"evidence_ids": ["E1"]},
    ],
)
def test_report_claim_rejects_invalid_fields(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        _claim(**overrides)


def test_report_section_valid_and_frozen() -> None:
    section = _section()

    assert section.claims[0].evidence_ids == ("E1",)
    with pytest.raises(ValidationError):
        section.heading = "Changed"


@pytest.mark.parametrize(
    "overrides",
    [
        {"extra": "forbidden"},
        {"heading": " \t"},
        {"heading": "x" * 121},
        {"claims": ()},
        {"claims": tuple(_claim() for _ in range(21))},
        {"claims": [_claim()]},
    ],
)
def test_report_section_rejects_invalid_fields(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        _section(**overrides)


def test_report_draft_valid_and_frozen() -> None:
    draft = _draft()

    assert draft.title == "Apple research"
    with pytest.raises(ValidationError):
        draft.title = "Changed"


@pytest.mark.parametrize(
    "overrides",
    [
        {"extra": "forbidden"},
        {"title": " \t"},
        {"title": "x" * 201},
        {"sections": ()},
        {"sections": tuple(_section() for _ in range(13))},
        {"sections": [_section()]},
    ],
)
def test_report_draft_rejects_invalid_fields(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        _draft(**overrides)
