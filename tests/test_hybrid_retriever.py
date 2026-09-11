import pytest

from app.rag.hybrid_retriever import (
    HybridRetriever,
)
from app.rag.models import RetrievedChunk


def _chunk(
    document: str,
    page: int,
    chunk_index: int,
) -> RetrievedChunk:
    return RetrievedChunk(
        text=(
            f"{document}-"
            f"{page}-"
            f"{chunk_index}"
        ),
        document=document,
        page=page,
        chunk_index=chunk_index,
        distance=0.0,
    )


class FakeRetriever:
    def __init__(
        self,
        results: list[RetrievedChunk],
    ):
        self.results = results

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        where=None,
    ) -> list[RetrievedChunk]:
        return self.results[:top_k]


def test_hybrid_retriever_rewards_overlap():
    shared = _chunk(
        document="shared.pdf",
        page=1,
        chunk_index=0,
    )

    dense_only = _chunk(
        document="dense.pdf",
        page=1,
        chunk_index=0,
    )

    lexical_only = _chunk(
        document="lexical.pdf",
        page=1,
        chunk_index=0,
    )

    dense = FakeRetriever(
        [
            dense_only,
            shared,
        ]
    )

    lexical = FakeRetriever(
        [
            lexical_only,
            shared,
        ]
    )

    retriever = HybridRetriever(
        dense_retriever=dense,
        lexical_retriever=lexical,
    )

    results = retriever.retrieve(
        query="test",
        top_k=3,
        candidate_k=2,
    )

    assert len(results) == 3

    assert (
        results[0].document
        == "shared.pdf"
    )


def test_hybrid_retriever_respects_top_k():
    dense = FakeRetriever(
        [
            _chunk(
                "dense-1.pdf",
                1,
                0,
            ),
            _chunk(
                "dense-2.pdf",
                1,
                0,
            ),
            _chunk(
                "dense-3.pdf",
                1,
                0,
            ),
        ]
    )

    lexical = FakeRetriever(
        [
            _chunk(
                "lexical-1.pdf",
                1,
                0,
            ),
            _chunk(
                "lexical-2.pdf",
                1,
                0,
            ),
            _chunk(
                "lexical-3.pdf",
                1,
                0,
            ),
        ]
    )

    retriever = HybridRetriever(
        dense_retriever=dense,
        lexical_retriever=lexical,
    )

    results = retriever.retrieve(
        query="test query",
        top_k=2,
        candidate_k=3,
    )

    assert len(results) == 2


def test_hybrid_retriever_handles_empty_lists():
    dense = FakeRetriever(
        []
    )

    lexical = FakeRetriever(
        []
    )

    retriever = HybridRetriever(
        dense_retriever=dense,
        lexical_retriever=lexical,
    )

    results = retriever.retrieve(
        query="test query",
        top_k=5,
        candidate_k=20,
    )

    assert results == []


def test_hybrid_retriever_handles_one_empty_retriever():
    dense_chunk = _chunk(
        document="dense.pdf",
        page=1,
        chunk_index=0,
    )

    dense = FakeRetriever(
        [
            dense_chunk,
        ]
    )

    lexical = FakeRetriever(
        []
    )

    retriever = HybridRetriever(
        dense_retriever=dense,
        lexical_retriever=lexical,
    )

    results = retriever.retrieve(
        query="test query",
        top_k=5,
        candidate_k=20,
    )

    assert len(results) == 1

    assert (
        results[0].document
        == "dense.pdf"
    )


def test_hybrid_retriever_rejects_nonpositive_rrf_k():
    dense = FakeRetriever(
        []
    )

    lexical = FakeRetriever(
        []
    )

    with pytest.raises(
        ValueError,
        match="rrf_k must be positive",
    ):
        HybridRetriever(
            dense_retriever=dense,
            lexical_retriever=lexical,
            rrf_k=0,
        )


def test_hybrid_retriever_rejects_negative_rrf_k():
    dense = FakeRetriever(
        []
    )

    lexical = FakeRetriever(
        []
    )

    with pytest.raises(
        ValueError,
        match="rrf_k must be positive",
    ):
        HybridRetriever(
            dense_retriever=dense,
            lexical_retriever=lexical,
            rrf_k=-1,
        )


def test_hybrid_retriever_deduplicates_same_chunk():
    shared_dense = _chunk(
        document="shared.pdf",
        page=1,
        chunk_index=0,
    )

    shared_lexical = _chunk(
        document="shared.pdf",
        page=1,
        chunk_index=0,
    )

    dense = FakeRetriever(
        [
            shared_dense,
        ]
    )

    lexical = FakeRetriever(
        [
            shared_lexical,
        ]
    )

    retriever = HybridRetriever(
        dense_retriever=dense,
        lexical_retriever=lexical,
    )

    results = retriever.retrieve(
        query="test query",
        top_k=5,
        candidate_k=20,
    )

    assert len(results) == 1

    assert (
        results[0].document
        == "shared.pdf"
    )

    assert results[0].page == 1

    assert (
        results[0].chunk_index
        == 0
    )


def test_hybrid_retriever_returns_empty_for_nonpositive_top_k():
    dense = FakeRetriever(
        [
            _chunk(
                "dense.pdf",
                1,
                0,
            )
        ]
    )

    lexical = FakeRetriever(
        [
            _chunk(
                "lexical.pdf",
                1,
                0,
            )
        ]
    )

    retriever = HybridRetriever(
        dense_retriever=dense,
        lexical_retriever=lexical,
    )

    assert (
        retriever.retrieve(
            query="test",
            top_k=0,
        )
        == []
    )


def test_hybrid_retriever_returns_empty_for_nonpositive_candidate_k():
    dense = FakeRetriever(
        [
            _chunk(
                "dense.pdf",
                1,
                0,
            )
        ]
    )

    lexical = FakeRetriever(
        [
            _chunk(
                "lexical.pdf",
                1,
                0,
            )
        ]
    )

    retriever = HybridRetriever(
        dense_retriever=dense,
        lexical_retriever=lexical,
    )

    assert (
        retriever.retrieve(
            query="test",
            top_k=5,
            candidate_k=0,
        )
        == []
    )