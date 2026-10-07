"""Report services are composed once and exposed through application state."""

from unittest.mock import Mock

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

import app.main as main_module
from app.dependencies import (
    get_agent_service,
    get_document_service,
    get_query_service,
    get_rag_pipeline,
    get_report_bundle_renderer,
    get_report_generation_service,
)
from app.report.bundle import ReportBundleRenderer
from app.report.chart_attachments import ReportChartAttachmentBuilder
from app.report.executor import ReportResearchExecutor
from app.report.generation import ReportGenerationService
from app.report.markdown_renderer import ReportMarkdownRenderer
from app.report.planner import ReportPlanner
from app.report.service import ReportService
from app.report.synthesizer import ReportSynthesizer
from app.services import rag_service
from app.services.agent_service import AgentService
from app.services.query_service import QueryService


def test_application_service_graph_reuses_agent_and_wires_report_models(
    monkeypatch,
) -> None:
    # Construct real lightweight service wrappers while replacing model and store
    # constructors, so the factory performs no downloads or database work.
    factories = {}
    for name in (
        "EmbeddingModel",
        "VectorStore",
        "Retriever",
        "Reranker",
        "AnswerGenerator",
        "RAGPipeline",
        "DocumentService",
        "LLMQuestionRouter",
        "SQLGenerator",
        "SQLExecutor",
        "SQLAnalysisClassifier",
        "AnalysisRetrievalPlanner",
        "AnalysisSQLBuilder",
        "AnalysisPlanner",
        "FinancialAnalyzer",
        "ChartDataBuilder",
        "ChartIntentClassifier",
        "ChartDecisionPolicy",
        "ChartPlanner",
        "ChartRenderer",
        "build_agent_graph",
    ):
        factory = Mock(return_value=object())
        monkeypatch.setattr(rag_service, name, factory)
        factories[name] = factory

    services = rag_service.create_application_services()

    assert services.rag_pipeline is factories["RAGPipeline"].return_value
    assert services.document_service is factories["DocumentService"].return_value
    assert isinstance(services.query_service, QueryService)
    assert services.query_service.rag_pipeline is services.rag_pipeline
    assert services.query_service.vector_store is factories["VectorStore"].return_value
    assert isinstance(services.agent_service, AgentService)
    assert services.agent_service.graph is factories["build_agent_graph"].return_value
    factories["build_agent_graph"].assert_called_once()

    generation = services.report_generation_service
    assert isinstance(generation, ReportGenerationService)
    assert isinstance(generation.research_service, ReportService)
    assert isinstance(generation.research_service.planner, ReportPlanner)
    assert generation.research_service.planner.model == "qwen3:4b-instruct"
    executor = generation.research_service.executor
    assert isinstance(executor, ReportResearchExecutor)
    assert executor.agent_service is services.agent_service
    assert isinstance(generation.synthesizer, ReportSynthesizer)
    assert generation.synthesizer.model == "qwen3:4b-instruct"

    bundle = services.report_bundle_renderer
    assert isinstance(bundle, ReportBundleRenderer)
    assert isinstance(bundle.markdown_renderer, ReportMarkdownRenderer)
    assert isinstance(bundle.attachment_builder, ReportChartAttachmentBuilder)


def test_report_dependency_getters_return_exact_state_objects() -> None:
    app = FastAPI()
    generation = object()
    bundle = object()
    app.state.report_generation_service = generation
    app.state.report_bundle_renderer = bundle
    request = Request({"type": "http", "app": app})

    assert get_report_generation_service(request) is generation
    assert get_report_bundle_renderer(request) is bundle


def test_lifespan_exposes_and_clears_all_services_without_running_report(
    monkeypatch,
) -> None:
    existing = [object() for _ in range(4)]
    generation = object()
    bundle = object()
    services = rag_service.ApplicationServices(
        rag_pipeline=existing[0],
        document_service=existing[1],
        query_service=existing[2],
        agent_service=existing[3],
        report_generation_service=generation,
        report_bundle_renderer=bundle,
    )
    factory = Mock(return_value=services)
    monkeypatch.setattr(main_module, "create_application_services", factory)
    app = main_module.app

    with TestClient(app) as client:
        factory.assert_called_once_with()
        request = Request({"type": "http", "app": app})
        assert app.state.rag_pipeline is existing[0]
        assert app.state.document_service is existing[1]
        assert app.state.query_service is existing[2]
        assert app.state.agent_service is existing[3]
        assert app.state.report_generation_service is generation
        assert app.state.report_bundle_renderer is bundle
        assert get_rag_pipeline(request) is existing[0]
        assert get_document_service(request) is existing[1]
        assert get_query_service(request) is existing[2]
        assert get_agent_service(request) is existing[3]
        assert get_report_generation_service(request) is generation
        assert get_report_bundle_renderer(request) is bundle
        assert client.get("/health").json() == {"status": "ok"}
        assert not any(path.startswith("/report") for path in app.openapi()["paths"])

    assert app.state.rag_pipeline is None
    assert app.state.document_service is None
    assert app.state.query_service is None
    assert app.state.agent_service is None
    assert app.state.report_generation_service is None
    assert app.state.report_bundle_renderer is None
