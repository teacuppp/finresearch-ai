from unittest.mock import Mock

from app.services import rag_service
from app.services.agent_service import AgentService
from app.services.query_service import QueryService


def test_composition_exposes_agent_and_reuses_query_service(
    monkeypatch,
):
    embedding_model = object()
    vector_store = object()
    retriever = object()
    reranker = object()
    generator = object()
    rag_pipeline = object()
    document_service = object()
    question_router = object()
    sql_generator = object()
    sql_executor = object()
    graph = object()

    embedding_factory = Mock(
        return_value=embedding_model
    )
    vector_store_factory = Mock(
        return_value=vector_store
    )
    retriever_factory = Mock(
        return_value=retriever
    )
    reranker_factory = Mock(
        return_value=reranker
    )
    generator_factory = Mock(
        return_value=generator
    )
    pipeline_factory = Mock(
        return_value=rag_pipeline
    )
    document_service_factory = Mock(
        return_value=document_service
    )
    router_factory = Mock(
        return_value=question_router
    )
    sql_generator_factory = Mock(
        return_value=sql_generator
    )
    sql_executor_factory = Mock(
        return_value=sql_executor
    )
    graph_builder = Mock(
        return_value=graph
    )

    monkeypatch.setattr(
        rag_service,
        "EmbeddingModel",
        embedding_factory,
    )
    monkeypatch.setattr(
        rag_service,
        "VectorStore",
        vector_store_factory,
    )
    monkeypatch.setattr(
        rag_service,
        "Retriever",
        retriever_factory,
    )
    monkeypatch.setattr(
        rag_service,
        "Reranker",
        reranker_factory,
    )
    monkeypatch.setattr(
        rag_service,
        "AnswerGenerator",
        generator_factory,
    )
    monkeypatch.setattr(
        rag_service,
        "RAGPipeline",
        pipeline_factory,
    )
    monkeypatch.setattr(
        rag_service,
        "DocumentService",
        document_service_factory,
    )
    monkeypatch.setattr(
        rag_service,
        "LLMQuestionRouter",
        router_factory,
    )
    monkeypatch.setattr(
        rag_service,
        "SQLGenerator",
        sql_generator_factory,
    )
    monkeypatch.setattr(
        rag_service,
        "SQLExecutor",
        sql_executor_factory,
    )
    monkeypatch.setattr(
        rag_service,
        "build_agent_graph",
        graph_builder,
    )

    services = (
        rag_service.create_application_services()
    )

    assert isinstance(
        services.query_service,
        QueryService,
    )
    assert isinstance(
        services.agent_service,
        AgentService,
    )
    assert services.agent_service.graph is graph
    assert services.rag_pipeline is rag_pipeline
    assert services.document_service is document_service
    assert services.query_service.rag_pipeline is rag_pipeline
    assert services.query_service.vector_store is vector_store

    graph_builder.assert_called_once_with(
        router=question_router,
        query_service=services.query_service,
        sql_generator=sql_generator,
        sql_executor=sql_executor,
    )
    sql_executor_factory.assert_called_once_with(
        database_path=(
            rag_service.FINANCIAL_DATABASE_PATH
        ),
    )
    assert (
        rag_service.FINANCIAL_DATABASE_PATH
        == rag_service.Path(
            "data/financial_demo.db"
        )
    )
