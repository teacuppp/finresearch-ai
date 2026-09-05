from app.rag.models import DocumentChunk
from app.rag.vector_store import VectorStore


def test_add_and_search_chunks(tmp_path):
    store = VectorStore(
        path=str(tmp_path / "chroma"),
        collection_name="test_collection",
    )

    chunks = [
        DocumentChunk(
            text="Tencent reported strong revenue growth.",
            document="tencent.pdf",
            page=1,
            chunk_index=0,
            company="Tencent",
            ticker="00700",
            fiscal_year=2025,
            document_type="10-K",
        ),
        DocumentChunk(
            text="Alibaba expanded its cloud computing business.",
            document="alibaba.pdf",
            page=2,
            chunk_index=0,
            company="Alibaba",
            ticker="09988",
            fiscal_year=2025,
            document_type="10-K",
        ),
    ]

    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]

    store.add_chunks(
        chunks=chunks,
        embeddings=embeddings,
    )

    results = store.search(
        query_embedding=[1.0, 0.0, 0.0],
        top_k=1,
    )

    assert len(results["documents"][0]) == 1

    assert (
        results["documents"][0][0]
        == "Tencent reported strong revenue growth."
    )

    assert (
        results["metadatas"][0][0]["document"]
        == "tencent.pdf"
    )

    assert results["metadatas"][0][0]["page"] == 1


    metadata = results["metadatas"][0][0]

    assert metadata["company"] == "Tencent"
    assert metadata["ticker"] == "00700"
    assert metadata["fiscal_year"] == 2025
    assert metadata["document_type"] == "10-K"



import pytest


def test_chunk_embedding_count_mismatch(tmp_path):
    store = VectorStore(
        path=str(tmp_path / "chroma"),
        collection_name="test_collection",
    )

    chunks = [
        DocumentChunk(
            text="Tencent revenue",
            document="tencent.pdf",
            page=1,
            chunk_index=0,
        ),
    ]

    embeddings = []

    with pytest.raises(ValueError):
        store.add_chunks(
            chunks=chunks,
            embeddings=embeddings,
        )

#证明：metadata filtering 发生在候选集合约束层，而不是“搜索完再随便筛”。
def test_search_with_metadata_filter(tmp_path):
    store = VectorStore(
        path=str(tmp_path / "chroma"),
        collection_name="filter_test",
    )

    chunks = [
        DocumentChunk(
            text="Microsoft reported revenue growth.",
            document="microsoft.pdf",
            page=1,
            chunk_index=0,
            company="Microsoft",
            ticker="MSFT",
            fiscal_year=2025,
            document_type="10-K",
        ),
        DocumentChunk(
            text="Apple reported revenue growth.",
            document="apple.pdf",
            page=1,
            chunk_index=0,
            company="Apple",
            ticker="AAPL",
            fiscal_year=2025,
            document_type="10-K",
        ),
    ]

    embeddings = [
        [1.0, 0.0, 0.0],
        [0.9, 0.1, 0.0],
    ]

    store.add_chunks(
        chunks=chunks,
        embeddings=embeddings,
    )

    results = store.search(
        query_embedding=[1.0, 0.0, 0.0],
        top_k=5,
        where={
            "ticker": {
                "$eq": "AAPL"
            }
        },
    )

    assert len(
        results["documents"][0]
    ) == 1

    assert (
        results["metadatas"][0][0]["ticker"]
        == "AAPL"
    )

#测试删除
def test_delete_document(tmp_path):
    store = VectorStore(
        path=str(tmp_path / "chroma"),
        collection_name="delete_test",
    )

    chunks = [
        DocumentChunk(
            text="Microsoft revenue",
            document="microsoft.pdf",
            page=1,
            chunk_index=0,
            ticker="MSFT",
        ),
        DocumentChunk(
            text="Apple revenue",
            document="apple.pdf",
            page=1,
            chunk_index=0,
            ticker="AAPL",
        ),
    ]

    embeddings = [
        [1.0, 0.0, 0.0],
        [0.9, 0.1, 0.0],
    ]

    store.add_chunks(
        chunks=chunks,
        embeddings=embeddings,
    )

    store.delete_document(
        "microsoft.pdf"
    )

    results = store.search(
        query_embedding=[1.0, 0.0, 0.0],
        top_k=5,
    )

    returned_documents = results["documents"][0]

    assert "Microsoft revenue" not in returned_documents
    assert "Apple revenue" in returned_documents

# #给 DocumentService 测 replacement 行为
def test_reindex_replaces_old_document(tmp_path):
    store = VectorStore(
        path=str(tmp_path / "chroma"),
        collection_name="replace_test",
    )

    old_chunks = [
        DocumentChunk(
            text="Old chunk 1",
            document="microsoft.pdf",
            page=1,
            chunk_index=0,
        ),
        DocumentChunk(
            text="Old chunk 2",
            document="microsoft.pdf",
            page=1,
            chunk_index=1,
        ),
    ]

    store.add_chunks(
        chunks=old_chunks,
        embeddings=[
            [1.0, 0.0],
            [0.9, 0.1],
        ],
    )

    store.delete_document(
        "microsoft.pdf"
    )

    new_chunks = [
        DocumentChunk(
            text="New chunk",
            document="microsoft.pdf",
            page=1,
            chunk_index=0,
        ),
    ]

    store.add_chunks(
        chunks=new_chunks,
        embeddings=[
            [1.0, 0.0],
        ],
    )

    assert store.count() == 1



#   pytest tests/test_vector_store.py -v