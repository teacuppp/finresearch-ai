from dataclasses import dataclass
from pathlib import Path

from app.agent.graph import build_agent_graph
from app.agent.router import LLMQuestionRouter
from app.agent.sql_executor import SQLExecutor
from app.agent.sql_generator import SQLGenerator
from app.analysis.chart_data import ChartDataBuilder
from app.analysis.chart_decision import ChartDecisionPolicy, ChartIntentClassifier
from app.analysis.chart_planner import ChartPlanner
from app.analysis.chart_renderer import ChartRenderer
from app.analysis.financial_analyzer import FinancialAnalyzer
from app.analysis.intent import SQLAnalysisClassifier
from app.analysis.planner import AnalysisPlanner
from app.analysis.retrieval import AnalysisRetrievalPlanner
from app.analysis.sql_builder import AnalysisSQLBuilder
from app.rag.embeddings import EmbeddingModel
from app.rag.generator import AnswerGenerator
from app.rag.pipeline import RAGPipeline
from app.rag.reranker import Reranker
from app.rag.retriever import Retriever
from app.rag.vector_store import VectorStore
from app.report.bundle import ReportBundleRenderer
from app.report.chart_attachments import ReportChartAttachmentBuilder
from app.report.executor import ReportResearchExecutor
from app.report.generation import ReportGenerationService
from app.report.markdown_renderer import ReportMarkdownRenderer
from app.report.planner import ReportPlanner
from app.report.service import ReportService
from app.report.synthesizer import ReportSynthesizer
from app.services.agent_service import AgentService
from app.services.document_service import (
    DocumentService,
)
from app.services.query_service import (
    QueryService,
)


FINANCIAL_DATABASE_PATH = Path(
    "data/financial_demo.db"
)


@dataclass
class ApplicationServices:
    rag_pipeline: RAGPipeline
    document_service: DocumentService
    query_service: QueryService
    agent_service: AgentService
    report_generation_service: ReportGenerationService
    report_bundle_renderer: ReportBundleRenderer


def create_application_services() -> (
    ApplicationServices
):
    embedding_model = EmbeddingModel()

    vector_store = VectorStore(
        path="data/chroma",
        collection_name=(
            "financial_documents"
        ),
    )

    retriever = Retriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
    )

    reranker = Reranker()

    generator = AnswerGenerator(
        model="qwen3:4b",
    )

    rag_pipeline = RAGPipeline(
        retriever=retriever,
        reranker=reranker,
        generator=generator,
        retrieval_depth=20,
    )

    document_service = DocumentService(
        embedding_model=embedding_model,
        vector_store=vector_store,
    )

    query_service = QueryService(
        rag_pipeline=rag_pipeline,
        vector_store=vector_store,
    )

    question_router = LLMQuestionRouter(model="qwen3:4b-instruct")

    sql_generator = SQLGenerator(model="qwen3:4b-instruct")
    sql_repair_generator = SQLGenerator(model="qwen3:4b")

    sql_executor = SQLExecutor(
        database_path=(
            FINANCIAL_DATABASE_PATH
        ),
    )

    sql_analysis_classifier = SQLAnalysisClassifier(model="qwen3:4b-instruct")
    analysis_retrieval_planner = AnalysisRetrievalPlanner(model="qwen3:4b")
    analysis_sql_builder = AnalysisSQLBuilder()
    analysis_planner = AnalysisPlanner(model="qwen3:4b")
    financial_analyzer = FinancialAnalyzer()
    chart_data_builder = ChartDataBuilder()
    chart_intent_classifier = ChartIntentClassifier(model="qwen3:4b-instruct")
    chart_decision_policy = ChartDecisionPolicy()
    direct_chart_planner = ChartPlanner(model="qwen3:4b-instruct")
    analysis_chart_planner = ChartPlanner(model="qwen3:4b")
    chart_renderer = ChartRenderer()

    agent_graph = build_agent_graph(
        router=question_router,
        query_service=query_service,
        sql_generator=sql_generator,
        sql_repair_generator=sql_repair_generator,
        sql_executor=sql_executor,
        sql_analysis_classifier=sql_analysis_classifier,
        analysis_retrieval_planner=analysis_retrieval_planner,
        analysis_sql_builder=analysis_sql_builder,
        analysis_planner=analysis_planner,
        financial_analyzer=financial_analyzer,
        chart_data_builder=chart_data_builder,
        chart_intent_classifier=chart_intent_classifier,
        chart_decision_policy=chart_decision_policy,
        direct_chart_planner=direct_chart_planner,
        analysis_chart_planner=analysis_chart_planner,
        chart_renderer=chart_renderer,
    )

    agent_service = AgentService(
        graph=agent_graph
    )

    report_planner = ReportPlanner(model="qwen3:4b-instruct")
    report_executor = ReportResearchExecutor(agent_service=agent_service)
    report_service = ReportService(planner=report_planner, executor=report_executor)
    report_synthesizer = ReportSynthesizer(model="qwen3:4b-instruct")
    report_generation_service = ReportGenerationService(
        research_service=report_service,
        synthesizer=report_synthesizer,
    )

    report_markdown_renderer = ReportMarkdownRenderer()
    report_attachment_builder = ReportChartAttachmentBuilder()
    report_bundle_renderer = ReportBundleRenderer(
        markdown_renderer=report_markdown_renderer,
        attachment_builder=report_attachment_builder,
    )

    return ApplicationServices(
        rag_pipeline=rag_pipeline,
        document_service=(
            document_service
        ),
        query_service=query_service,
        agent_service=agent_service,
        report_generation_service=report_generation_service,
        report_bundle_renderer=report_bundle_renderer,
    )
