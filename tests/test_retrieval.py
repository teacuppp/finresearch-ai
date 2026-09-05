from app.rag.embeddings import EmbeddingModel
from app.rag.models import DocumentChunk
from app.rag.retriever import Retriever
from app.rag.vector_store import VectorStore


def test_semantic_retrieval(tmp_path):

    embedding_model = EmbeddingModel()

    vector_store = VectorStore(
        path=str(tmp_path / "chroma"),
        collection_name="retrieval_test",
    )

    chunks = [
        DocumentChunk(
            text=(
                "Tencent is a technology company that operates "
                "social networks, video games, digital content "
                "and financial technology services."
            ),
            document="tencent.pdf",
            page=10,
            chunk_index=0,
        ),
        DocumentChunk(
            text=(
                "Tesla designs and manufactures electric vehicles, "
                "energy storage systems and solar energy products."
            ),
            document="tesla.pdf",
            page=20,
            chunk_index=0,
        ),
        DocumentChunk(
            text=(
                "Apple designs smartphones, personal computers, "
                "tablets and other consumer electronic devices."
            ),
            document="apple.pdf",
            page=30,
            chunk_index=0,
        ),
    ]

    document_embeddings = embedding_model.embed_documents(
        [chunk.text for chunk in chunks]
    )

    vector_store.add_chunks(
        chunks=chunks,
        embeddings=document_embeddings,
    )

    retriever = Retriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
    )

    results = retriever.retrieve(
        query="Which company operates video games and social networks?",
        top_k=1,
    )

    assert len(results) == 1

    first_result = results[0]

    assert first_result.document == "tencent.pdf"
    assert first_result.page == 10
    assert first_result.chunk_index == 0
    assert first_result.distance >= 0





#让 Microsoft 和 Apple 都在 collection 里. 20s
def test_retrieval_with_metadata_filter(tmp_path):

    embedding_model = EmbeddingModel()
    
    # vector_store = VectorStore(
    #     path="data/chroma",
    #     collection_name="financial_documents",
    # )

    vector_store = VectorStore(
            path=str(tmp_path / "chroma"),
            collection_name="retrieval_test",
        )

    chunks = [
            DocumentChunk(
                text=(
                    "Tencent is a technology company that operates "
                    "social networks, video games, digital content "
                    "and financial technology services."
                ),
                document="tencent.pdf",
                page=10,
                chunk_index=0,
                company = None,
                ticker = "MSFT",
                fiscal_year = None,
                document_type =None,
            ),
            DocumentChunk(
                text=(
                    "Tesla designs and manufactures electric vehicles, "
                    "energy storage systems and solar energy products."
                ),
                document="tesla.pdf",
                page=20,
                chunk_index=0,
                company = None,
                ticker = None,
                fiscal_year = None,
                document_type =None,
            ),
            DocumentChunk(
                text=(
                    "Apple designs smartphones, personal computers, "
                    "tablets and other consumer electronic devices."
                ),
                document="apple.pdf",
                page=30,
                chunk_index=0,
                company = None,
                ticker = None,
                fiscal_year = None,
                document_type =None,
            ),
             ]

    vector_store.add_chunks(
        chunks,
        embedding_model.embed_documents(
        [chunk.text for chunk in chunks]
    )
    )

    
    retriever = Retriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
    )

    results = retriever.retrieve(
        query="What was revenue?",
        top_k=5,
        where={
            "ticker": {
                "$eq": "MSFT"
            }
        },
    )

    assert len(results) > 0

    for result in results:
        assert result.ticker == "MSFT"


#.   pytest tests/test_retrieval.py -v