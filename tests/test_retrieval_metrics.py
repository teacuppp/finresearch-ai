from app.evaluation.retrieval_metrics import (
    hit_at_k,
    mean,
    reciprocal_rank,
    find_relevant_ranks
)


from app.evaluation.models import RelevantSource
from app.rag.models import RetrievedChunk



def test_hit_at_k():
    assert hit_at_k([1], 1) == 1.0
    assert hit_at_k([2], 1) == 0.0
    assert hit_at_k([2], 3) == 1.0
    assert hit_at_k([], 5) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank([1]) == 1.0
    assert reciprocal_rank([2]) == 0.5
    assert reciprocal_rank([3]) == 1 / 3
    assert reciprocal_rank([]) == 0.0


def test_reciprocal_rank_uses_first_relevant_result():
    assert reciprocal_rank([4, 2, 5]) == 0.5


def test_mean():
    assert mean([1.0, 0.5, 0.0]) == 0.5
    assert mean([]) == 0.0


# 补2个 find_relevant_ranks() 单测

def test_find_relevant_ranks():
    results = [
        RetrievedChunk(
            text="irrelevant",
            document="apple.pdf",
            page=10,
            chunk_index=0,
            distance=0.1,
        ),
        RetrievedChunk(
            text="relevant",
            document="apple.pdf",
            page=35,
            chunk_index=0,
            distance=0.2,
        ),
        RetrievedChunk(
            text="also relevant",
            document="apple.pdf",
            page=35,
            chunk_index=1,
            distance=0.3,
        ),
    ]

    relevant_sources = [
        RelevantSource(
            document="apple.pdf",
            page=35,
            chunk_index=0,
        ),
        RelevantSource(
            document="apple.pdf",
            page=35,
            chunk_index=1,
        ),
    ]

    ranks = find_relevant_ranks(
        results,
        relevant_sources,
    )

    assert ranks == [2, 3]




def test_find_relevant_ranks_returns_empty_when_no_match():
    results = [
        RetrievedChunk(
            text="irrelevant",
            document="apple.pdf",
            page=10,
            chunk_index=0,
            distance=0.1,
        )
    ]

    relevant_sources = [
        RelevantSource(
            document="apple.pdf",
            page=35,
            chunk_index=0,
        )
    ]

    assert find_relevant_ranks(
        results,
        relevant_sources,
    ) == []