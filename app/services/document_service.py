from pathlib import Path

from app.rag.embeddings import EmbeddingModel
from app.rag.ingestion import process_pdf
from app.rag.vector_store import VectorStore


class DocumentService:
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    def index_pdf(
        self,
        file_path: Path,
        company: str | None = None,
        ticker: str | None = None,
        fiscal_year: int | None = None,
        document_type: str | None = None,
    ) -> int:
        chunks = process_pdf(
            file_path=file_path,
            company=company,
            ticker=ticker,
            fiscal_year=fiscal_year,
            document_type=document_type,
        )
        texts = [
            chunk.text
            for chunk in chunks
        ]

        embeddings = self.embedding_model.embed_documents(
            texts
        )

        self.vector_store.delete_document(
            file_path.name
        )

        self.vector_store.add_chunks(
            chunks=chunks,
            embeddings=embeddings,
        )

        return len(chunks)