from app.rag.lexical_retriever import (
    LexicalRetriever,
)
from app.rag.models import DocumentChunk


class FakeVectorStore:
    def get_chunks(
        self,
        where=None,
    ):
        return [
            DocumentChunk(
                text=(
                    "Operating income "
                    "was 133,050"
                ),
                document="a.pdf",
                page=1,
                chunk_index=0,
                ticker="AAPL",
                fiscal_year=2025,
            ),
            DocumentChunk(
                text=(
                    "Operating activities "
                    "generated cash"
                ),
                document="a.pdf",
                page=2,
                chunk_index=0,
                ticker="AAPL",
                fiscal_year=2025,
            ),
        ]


def test_lexical_retriever_prefers_exact_terms():
    retriever = LexicalRetriever(
        vector_store=FakeVectorStore()
    )

    results = retriever.retrieve(
        query="operating income",
        top_k=2,
    )

    assert results[0].page == 1


# def test_lexical_retriever_returns_empty_results():
#     retriever = LexicalRetriever(
#         vector_store=FakeVectorStore()
#     )

#     results = retriever.retrieve(
#         query="nonexistent_term",
#         top_k=5,
#     )

#     assert results == []


def test_lexical_retriever_respects_top_k():
    retriever = LexicalRetriever(
        vector_store=FakeVectorStore()
    )

    # "operating" 会匹配两个块（page 1 和 page 2）
    results = retriever.retrieve(
        query="operating",
        top_k=1,
    )

    assert len(results) == 1
