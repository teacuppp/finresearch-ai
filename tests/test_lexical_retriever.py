from app.rag.lexical_retriever import (
    LexicalRetriever,
)
from app.rag.models import DocumentChunk


class FakeVectorStore:
    def __init__(
        self,
        chunks: list[DocumentChunk],
    ):
        self.chunks = chunks

    def get_chunks(
        self,
        where=None,
    ):
        return self.chunks


def _chunk(
    text: str,
    document: str,
    page: int,
    chunk_index: int,
) -> DocumentChunk:
    return DocumentChunk(
        text=text,
        document=document,
        page=page,
        chunk_index=chunk_index,
        ticker="AAPL",
        fiscal_year=2025,
    )


def test_lexical_retriever_prefers_exact_terms():
    vector_store = FakeVectorStore(
        [
            _chunk(
                text=(
                    "Operating income "
                    "was 133,050"
                ),
                document="a.pdf",
                page=1,
                chunk_index=0,
            ),
            _chunk(
                text=(
                    "Operating activities "
                    "generated cash"
                ),
                document="a.pdf",
                page=2,
                chunk_index=0,
            ),
        ]
    )

    retriever = LexicalRetriever(
        vector_store=vector_store,
    )

    results = retriever.retrieve(
        query="operating income",
        top_k=2,
    )

    assert len(results) == 2
    assert results[0].page == 1
    assert (
        "Operating income"
        in results[0].text
    )


def test_lexical_retriever_returns_empty_results():
    vector_store = FakeVectorStore(
        []
    )

    retriever = LexicalRetriever(
        vector_store=vector_store,
    )

    results = retriever.retrieve(
        query="operating income",
        top_k=5,
    )

    assert results == []


def test_lexical_retriever_respects_top_k():
    vector_store = FakeVectorStore(
        [
            _chunk(
                text=(
                    "Operating income "
                    "was 133,050"
                ),
                document="a.pdf",
                page=1,
                chunk_index=0,
            ),
            _chunk(
                text=(
                    "Operating income "
                    "increased during 2025"
                ),
                document="a.pdf",
                page=2,
                chunk_index=0,
            ),
            _chunk(
                text=(
                    "Net income "
                    "was 112,010"
                ),
                document="a.pdf",
                page=3,
                chunk_index=0,
            ),
        ]
    )

    retriever = LexicalRetriever(
        vector_store=vector_store,
    )

    results = retriever.retrieve(
        query="operating income",
        top_k=2,
    )

    assert len(results) == 2