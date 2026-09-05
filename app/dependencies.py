from fastapi import Request

from app.rag.pipeline import RAGPipeline
from app.services.document_service import DocumentService


def get_rag_pipeline(
    request: Request,
) -> RAGPipeline:
    return request.app.state.rag_pipeline


def get_document_service(
    request: Request,
) -> DocumentService:
    return request.app.state.document_service