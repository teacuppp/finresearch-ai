from fastapi import Request

from app.rag.pipeline import RAGPipeline
from app.services.document_service import DocumentService
from app.services.query_service import QueryService


def get_rag_pipeline(
    request: Request,
) -> RAGPipeline:
    return request.app.state.rag_pipeline


def get_document_service(
    request: Request,
) -> DocumentService:
    return request.app.state.document_service



def get_query_service(
    request: Request,
) -> QueryService:
    return request.app.state.query_service