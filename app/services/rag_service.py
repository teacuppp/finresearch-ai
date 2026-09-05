from dataclasses import dataclass

from app.rag.embeddings import EmbeddingModel
from app.rag.generator import AnswerGenerator
from app.rag.pipeline import RAGPipeline
from app.rag.retriever import Retriever
from app.rag.vector_store import VectorStore
from app.services.document_service import DocumentService


@dataclass
class ApplicationServices:
    rag_pipeline: RAGPipeline
    document_service: DocumentService


def create_application_services() -> ApplicationServices:
    embedding_model = EmbeddingModel()

    vector_store = VectorStore(
        path="data/chroma",
        collection_name="financial_documents",
    )

    retriever = Retriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
    )

    generator = AnswerGenerator(
        model="qwen3:4b",
    )

    rag_pipeline = RAGPipeline(
        retriever=retriever,
        generator=generator,
    )

    document_service = DocumentService(
        embedding_model=embedding_model,
        vector_store=vector_store,
    )

    return ApplicationServices(
        rag_pipeline=rag_pipeline,
        document_service=document_service,
    )