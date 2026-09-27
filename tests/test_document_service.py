from pathlib import Path

import pytest

from app.rag.models import DocumentChunk
from app.services import document_service
from app.services.document_service import DocumentService


class FakeEmbeddingModel:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[0.1, 0.2] for _ in texts]


class FakeVectorStore:
    def __init__(self) -> None:
        self.deleted: list[str] = []
        self.added: list[dict] = []

    def delete_document(self, document: str) -> None:
        self.deleted.append(document)

    def add_chunks(
        self, chunks: list[DocumentChunk], embeddings: list[list[float]]
    ) -> None:
        self.added.append({"chunks": chunks, "embeddings": embeddings})


@pytest.mark.parametrize("supplied", ["10K", "10-K"])
def test_document_service_indexes_canonical_document_type(
    supplied: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    process_calls: list[dict] = []

    def fake_process_pdf(**kwargs) -> list[DocumentChunk]:
        process_calls.append(kwargs)
        return [
            DocumentChunk(
                text="Revenue disclosure",
                document=kwargs["file_path"].name,
                page=1,
                chunk_index=0,
                document_type=kwargs["document_type"],
            )
        ]

    monkeypatch.setattr(document_service, "process_pdf", fake_process_pdf)
    embeddings = FakeEmbeddingModel()
    store = FakeVectorStore()
    service = DocumentService(embedding_model=embeddings, vector_store=store)
    file_path = tmp_path / "apple.pdf"

    count = service.index_pdf(file_path, document_type=supplied)

    assert count == 1
    assert process_calls == [{
        "file_path": file_path,
        "company": None,
        "ticker": None,
        "fiscal_year": None,
        "document_type": "10-K",
    }]
    assert embeddings.calls == [["Revenue disclosure"]]
    assert store.deleted == ["apple.pdf"]
    assert len(store.added) == 1
    assert store.added[0]["chunks"][0].document_type == "10-K"


@pytest.mark.parametrize("supplied", ["filing", "", "13F"])
def test_invalid_document_type_fails_before_any_index_mutation(
    supplied: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    process_calls: list[dict] = []

    def fake_process_pdf(**kwargs) -> list[DocumentChunk]:
        process_calls.append(kwargs)
        return []

    monkeypatch.setattr(document_service, "process_pdf", fake_process_pdf)
    embeddings = FakeEmbeddingModel()
    store = FakeVectorStore()
    service = DocumentService(embedding_model=embeddings, vector_store=store)

    with pytest.raises(ValueError, match="Unsupported document type"):
        service.index_pdf(tmp_path / "apple.pdf", document_type=supplied)

    assert process_calls == []
    assert embeddings.calls == []
    assert store.deleted == []
    assert store.added == []
