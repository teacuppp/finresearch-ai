import pytest

from app.rag.validation import (
    INSUFFICIENT_INFORMATION_RESPONSE,
    has_source_citation,
    validate_answer,
    AnswerValidationError
)


def test_has_source_citation():
    answer = (
        "Apple reported total net sales of "
        "$416.161 billion. [Source 1]"
    )

    assert has_source_citation(answer)


def test_missing_source_citation():
    answer = (
        "Apple reported total net sales of "
        "$416.161 billion."
    )

    assert not has_source_citation(answer)


def test_validate_answer_with_citation():
    validate_answer(
        "Apple reported $416.161 billion. [Source 1]"
    )


def test_validate_insufficient_information():
    validate_answer(
        INSUFFICIENT_INFORMATION_RESPONSE
    )


def test_validate_answer_without_citation():
    with pytest.raises(AnswerValidationError):
        validate_answer(
            "Apple reported $416.161 billion."
        )