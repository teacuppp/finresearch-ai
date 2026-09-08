from app.evaluation.models import RelevantSource
from app.evaluation.retrieval_metrics import (
    find_relevant_ranks,
    hit_at_k,
    mean,
    reciprocal_rank,
)
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
    assert reciprocal_rank(
        [4, 2, 5]
    ) == 0.5


def test_mean():
    assert mean(
        [1.0, 0.5, 0.0]
    ) == 0.5

    assert mean([]) == 0.0


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
            text=(
                "2025 2024 2023\n"
                "Services 109,158 96,169 85,200"
            ),
            document="apple.pdf",
            page=35,
            chunk_index=8,
            distance=0.2,
        ),
        RetrievedChunk(
            text=(
                "2025 2024 2023\n"
                "Services 109,158 96,169 85,200"
            ),
            document="apple.pdf",
            page=35,
            chunk_index=99,
            distance=0.3,
        ),
    ]

    relevant_sources = [
        RelevantSource(
            document="apple.pdf",
            page=35,
            row_terms=[
                "Services",
                "109,158",
            ],
            preceding_terms=[
                "2025",
            ],
            preceding_line_window=5,
        )
    ]

    assert find_relevant_ranks(
        results,
        relevant_sources,
    ) == [2, 3]


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
            row_terms=[
                "Total net sales",
                "416,161",
            ],
            preceding_terms=[
                "2025",
            ],
            preceding_line_window=5,
        )
    ]

    assert find_relevant_ranks(
        results,
        relevant_sources,
    ) == []


def test_structured_evidence_matches_header_before_row():
    results = [
        RetrievedChunk(
            text=(
                "SUMMARY RESULTS OF OPERATIONS\n"
                "2025 2024 Change\n"
                "Revenue $ 281,724 $ 245,122 15%"
            ),
            document="microsoft.pdf",
            page=40,
            chunk_index=1,
            distance=0.1,
        )
    ]

    relevant_sources = [
        RelevantSource(
            document="microsoft.pdf",
            page=40,
            row_terms=[
                "Revenue",
                "281,724",
            ],
            preceding_terms=[
                "2025",
            ],
            preceding_line_window=5,
        )
    ]

    assert find_relevant_ranks(
        results,
        relevant_sources,
    ) == [1]


def test_structured_evidence_rejects_year_after_row():
    results = [
        RetrievedChunk(
            text=(
                "Services 109,158 96,169 85,200\n"
                "Some unrelated text\n"
                "iPhone net sales increased during "
                "2025 compared to 2024"
            ),
            document="apple.pdf",
            page=29,
            chunk_index=1,
            distance=0.1,
        )
    ]

    relevant_sources = [
        RelevantSource(
            document="apple.pdf",
            page=29,
            row_terms=[
                "Services",
                "109,158",
            ],
            preceding_terms=[
                "2025",
            ],
            preceding_line_window=5,
        )
    ]

    assert find_relevant_ranks(
        results,
        relevant_sources,
    ) == []


def test_structured_evidence_requires_terms_on_same_row():
    results = [
        RetrievedChunk(
            text=(
                "$ 281,724 $ 245,122 15%\n"
                "Gross margin 193,893 171,008\n"
                "Fiscal Year 2025 Compared "
                "with Fiscal Year 2024\n"
                "Revenue increased $36.6 billion"
            ),
            document="microsoft.pdf",
            page=40,
            chunk_index=2,
            distance=0.1,
        )
    ]

    relevant_sources = [
        RelevantSource(
            document="microsoft.pdf",
            page=40,
            row_terms=[
                "Revenue",
                "281,724",
            ],
            preceding_terms=[
                "2025",
            ],
            preceding_line_window=5,
        )
    ]

    assert find_relevant_ranks(
        results,
        relevant_sources,
    ) == []


def test_structured_evidence_requires_matching_page():
    results = [
        RetrievedChunk(
            text=(
                "2025 2024 Change\n"
                "Revenue $ 281,724 $ 245,122"
            ),
            document="microsoft.pdf",
            page=99,
            chunk_index=0,
            distance=0.1,
        )
    ]

    relevant_sources = [
        RelevantSource(
            document="microsoft.pdf",
            page=40,
            row_terms=[
                "Revenue",
                "281,724",
            ],
            preceding_terms=[
                "2025",
            ],
            preceding_line_window=5,
        )
    ]

    assert find_relevant_ranks(
        results,
        relevant_sources,
    ) == []