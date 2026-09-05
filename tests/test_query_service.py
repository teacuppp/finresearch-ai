import pytest

from app.rag.models import RetrievedChunk
from app.rag.pipeline import RAGResult
from app.services.query_service import (
    AmbiguousQueryError,
    QueryService,
)


class FakeVectorStore:
    def __init__(self, tickers):
        self.tickers = tickers

    def list_tickers(self):
        return self.tickers


class FakePipeline:
    def __init__(self):
        self.called = False

    def ask(
        self,
        question,
        top_k=5,
        where=None,
    ):
        self.called = True

        return RAGResult(
            answer="Answer [Source 1]",
            sources=[
                RetrievedChunk(
                    text="Example",
                    document="example.pdf",
                    page=1,
                    chunk_index=0,
                    distance=0.1,
                )
            ],
        )


def test_rejects_ambiguous_query():
    pipeline = FakePipeline()

    service = QueryService(
        rag_pipeline=pipeline,
        vector_store=FakeVectorStore(
            {"AAPL", "MSFT"}
        ),
    )

    with pytest.raises(AmbiguousQueryError):
        service.ask(
            question="What was total revenue in 2025?"
        )

    assert pipeline.called is False


def test_allows_query_with_ticker():
    pipeline = FakePipeline()

    service = QueryService(
        rag_pipeline=pipeline,
        vector_store=FakeVectorStore(
            {"AAPL", "MSFT"}
        ),
    )

    result = service.ask(
        question="What was total revenue in 2025?",
        ticker="MSFT",
        where={
            "ticker": {
                "$eq": "MSFT"
            }
        },
    )

    assert pipeline.called is True
    assert result.answer == "Answer [Source 1]"


def test_allows_unfiltered_query_for_single_ticker():
    pipeline = FakePipeline()

    service = QueryService(
        rag_pipeline=pipeline,
        vector_store=FakeVectorStore(
            {"MSFT"}
        ),
    )

    service.ask(
        question="What was total revenue in 2025?"
    )

    assert pipeline.called is True