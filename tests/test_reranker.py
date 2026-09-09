from app.rag.models import RetrievedChunk
from app.rag.reranker import Reranker


class FakeCrossEncoder:
    def predict(
        self,
        pairs,
    ):
        return [
            0.1,
            0.9,
            0.4,
        ]


def test_reranker_orders_results_by_score():
    reranker = Reranker.__new__(
        Reranker
    )

    reranker.model = (
        FakeCrossEncoder()
    )

    results = [
        RetrievedChunk(
            text="first",
            document="test.pdf",
            page=1,
            chunk_index=0,
            distance=0.1,
        ),
        RetrievedChunk(
            text="second",
            document="test.pdf",
            page=2,
            chunk_index=0,
            distance=0.2,
        ),
        RetrievedChunk(
            text="third",
            document="test.pdf",
            page=3,
            chunk_index=0,
            distance=0.3,
        ),
    ]

    reranked = reranker.rerank(
        query="test query",
        results=results,
    )

    assert [
        result.text
        for result in reranked
    ] == [
        "second",
        "third",
        "first",
    ]