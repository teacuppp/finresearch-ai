def hit_at_k(
    relevant_ranks: list[int],
    k: int,
) -> float:
    return float(
        any(
            rank <= k
            for rank in relevant_ranks
        )
    )


def reciprocal_rank(
    relevant_ranks: list[int],
) -> float:
    if not relevant_ranks:
        return 0.0

    first_relevant_rank = min(
        relevant_ranks
    )

    return (
        1.0
        / first_relevant_rank
    )


def mean(
    values: list[float],
) -> float:
    if not values:
        return 0.0

    return (
        sum(values)
        / len(values)
    )


def _normalize_text(
    text: str,
) -> str:
    return " ".join(
        text.casefold().split()
    )


def _line_contains_all_terms(
    line: str,
    terms: list[str],
) -> bool:
    normalized_line = _normalize_text(
        line
    )

    return all(
        _normalize_text(term)
        in normalized_line
        for term in terms
    )


def _text_contains_all_terms(
    text: str,
    terms: list[str],
) -> bool:
    normalized_text = _normalize_text(
        text
    )

    return all(
        _normalize_text(term)
        in normalized_text
        for term in terms
    )


def _matches_structured_evidence(
    text: str,
    row_terms: list[str],
    preceding_terms: list[str],
    preceding_line_window: int,
) -> bool:
    lines = [
        line
        for line in text.splitlines()
        if line.strip()
    ]

    for row_index, line in enumerate(
        lines
    ):
        if not _line_contains_all_terms(
            line,
            row_terms,
        ):
            continue

        if not preceding_terms:
            return True

        start_index = max(
            0,
            row_index
            - preceding_line_window,
        )

        preceding_lines = lines[
            start_index:row_index
        ]

        preceding_text = "\n".join(
            preceding_lines
        )

        if _text_contains_all_terms(
            preceding_text,
            preceding_terms,
        ):
            return True

    return False


def _matches_relevant_source(
    result,
    source,
) -> bool:
    if (
        result.document
        != source.document
    ):
        return False

    if result.page != source.page:
        return False

    return _matches_structured_evidence(
        text=result.text,
        row_terms=source.row_terms,
        preceding_terms=(
            source.preceding_terms
        ),
        preceding_line_window=(
            source.preceding_line_window
        ),
    )


def find_relevant_ranks(
    results,
    relevant_sources,
) -> list[int]:
    ranks = []

    for rank, result in enumerate(
        results,
        start=1,
    ):
        if any(
            _matches_relevant_source(
                result,
                source,
            )
            for source
            in relevant_sources
        ):
            ranks.append(rank)

    return ranks