import re


INSUFFICIENT_INFORMATION_RESPONSE = (
    "I could not find enough information in the provided documents."
)


class AnswerValidationError(Exception):
    pass


def has_source_citation(
    answer: str,
) -> bool:
    return bool(
        re.search(
            r"\[Source\s+\d+\]",
            answer,
        )
    )


def validate_answer(
    answer: str,
) -> None:
    if answer == INSUFFICIENT_INFORMATION_RESPONSE:
        return

    if not has_source_citation(answer):
        raise AnswerValidationError(
            "Generated answer does not contain "
            "a valid source citation."
        )