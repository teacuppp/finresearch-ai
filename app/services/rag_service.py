from dataclasses import dataclass
from pathlib import Path

from app.agent.graph import build_agent_graph
from app.agent.router import LLMQuestionRouter
from app.agent.sql_executor import SQLExecutor
from app.agent.sql_generator import SQLGenerator
from app.rag.embeddings import EmbeddingModel
from app.rag.generator import AnswerGenerator
from app.rag.pipeline import RAGPipeline
from app.rag.reranker import Reranker
from app.rag.retriever import Retriever
from app.rag.vector_store import VectorStore
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

    question_router = LLMQuestionRouter()

    sql_generator = SQLGenerator()

    sql_executor = SQLExecutor(
        database_path=(
            FINANCIAL_DATABASE_PATH
        ),
    )

    agent_graph = build_agent_graph(
        router=question_router,
        query_service=query_service,
        sql_generator=sql_generator,
        sql_executor=sql_executor,
    )

    agent_service = AgentService(
        graph=agent_graph
    )

    return ApplicationServices(
        rag_pipeline=rag_pipeline,
        document_service=(
            document_service
        ),
        query_service=query_service,
        agent_service=agent_service,
    )
