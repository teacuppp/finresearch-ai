from fastapi import Request

from app.rag.pipeline import RAGPipeline
from app.report.bundle import ReportBundleRenderer
from app.report.generation import ReportGenerationService
from app.services.agent_service import AgentService
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


def get_agent_service(
    request: Request,
) -> AgentService:
    return request.app.state.agent_service


def get_report_generation_service(
    request: Request,
) -> ReportGenerationService:
    return request.app.state.report_generation_service


def get_report_bundle_renderer(
    request: Request,
) -> ReportBundleRenderer:
    return request.app.state.report_bundle_renderer
