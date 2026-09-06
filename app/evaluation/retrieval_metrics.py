def hit_at_k(
    relevant_ranks: list[int],
    k: int,
) -> float:
    return float(
        any(rank <= k for rank in relevant_ranks)
    )


def reciprocal_rank(
    relevant_ranks: list[int],
) -> float:
    if not relevant_ranks:
        return 0.0

    first_relevant_rank = min(relevant_ranks)

    return 1.0 / first_relevant_rank


def mean(values: list[float]) -> float:
    if not values:
        return 0.0

    return sum(values) / len(values)

def find_relevant_ranks(
    results,
    relevant_sources,
) -> list[int]:
    relevant_keys = {
        (
            source.document,
            source.page,
            source.chunk_index,
        )
        for source in relevant_sources
    }

    ranks = []

    for rank, result in enumerate(
        results,
        start=1,
    ):
        result_key = (
            result.document,
            result.page,
            result.chunk_index,
        )

        if result_key in relevant_keys:
            ranks.append(rank)

    return ranks