import pytest

from app.agent.graph import (
    UnsupportedRouteError,
    build_agent_graph,
)
from app.agent.schema import FINANCIAL_SCHEMA
from app.agent.sql_executor import SQLQueryResult
from app.rag.models import RetrievedChunk
from app.rag.pipeline import RAGResult


class FakeRouter:
    def __init__(self, route: str):
        self.selected_route = route
        self.calls = []

    def route(self, question: str) -> str:
        self.calls.append(question)
        return self.selected_route


class FakeQueryService:
    def __init__(self):
        self.calls = []

    def ask(
        self,
        question: str,
        top_k: int = 5,
        where: dict | None = None,
        company: str | None = None,
        ticker: str | None = None,
    ) -> RAGResult:
        self.calls.append({
            "question": question,
            "top_k": top_k,
            "where": where,
            "company": company,
            "ticker": ticker,
        })

        return RAGResult(
            answer=f"RAG answer for {question} [Source 1]",
            sources=[
                RetrievedChunk(
                    text="Financial filing text",
                    document="filing.pdf",
                    page=1,
                    chunk_index=0,
                    distance=0.1,
                )
            ],
        )


class FakeSQLGenerator:
    def __init__(self):
        self.calls = []

    def generate(self, question: str, schema: str) -> str:
        self.calls.append({
            "question": question,
            "schema": schema,
        })

        return "SELECT revenue_musd FROM financial_metrics"


class FakeSQLExecutor:
    def __init__(self):
        self.calls = []

    def execute(
        self,
        sql: str,
        parameters: tuple[object, ...] = (),
    ) -> SQLQueryResult:
        self.calls.append({
            "sql": sql,
            "parameters": parameters,
        })

        return SQLQueryResult(
            columns=["revenue_musd"],
            rows=[{"revenue_musd": 416161.0}],
            row_count=1,
        )


def _graph_with_fakes(route: str):
    router = FakeRouter(route)
    query_service = FakeQueryService()
    sql_generator = FakeSQLGenerator()
    sql_executor = FakeSQLExecutor()

    graph = build_agent_graph(
        router=router,
        query_service=query_service,
        sql_generator=sql_generator,
        sql_executor=sql_executor,
    )

    return (
        graph,
        router,
        query_service,
        sql_generator,
        sql_executor,
    )


def test_rag_route_delegates_to_query_service():
    (
        graph,
        router,
        query_service,
        sql_generator,
        sql_executor,
    ) = _graph_with_fakes("rag")

    question = "What risks did Apple disclose?"
    result = graph.invoke({"question": question})

    assert router.calls == [question]
    assert query_service.calls == [
        {
            "question": question,
            "top_k": 5,
            "where": None,
            "company": None,
            "ticker": None,
        }
    ]
    assert sql_generator.calls == []
    assert sql_executor.calls == []
    assert result["route"] == "rag"
    assert result["answer"] == f"RAG answer for {question} [Source 1]"
    assert len(result["sources"]) == 1


def test_rag_route_forwards_query_options_unchanged():
    (
        graph,
        _,
        query_service,
        sql_generator,
        sql_executor,
    ) = _graph_with_fakes("rag")
    where = {
        "$and": [
            {"ticker": {"$eq": "MSFT"}},
            {"fiscal_year": {"$eq": 2025}},
        ]
    }

    graph.invoke({
        "question": "What risks did Microsoft disclose?",
        "top_k": 3,
        "where": where,
        "company": "Microsoft",
        "ticker": "MSFT",
    })

    assert query_service.calls == [{
        "question": "What risks did Microsoft disclose?",
        "top_k": 3,
        "where": where,
        "company": "Microsoft",
        "ticker": "MSFT",
    }]
    assert query_service.calls[0]["where"] is where
    assert sql_generator.calls == []
    assert sql_executor.calls == []


def test_sql_route_delegates_to_generator_and_executor():
    (
        graph,
        router,
        query_service,
        sql_generator,
        sql_executor,
    ) = _graph_with_fakes("sql")

    question = "What was Apple's revenue?"
    result = graph.invoke({"question": question})

    generated_sql = "SELECT revenue_musd FROM financial_metrics"

    assert router.calls == [question]
    assert query_service.calls == []
    assert sql_generator.calls == [
        {
            "question": question,
            "schema": FINANCIAL_SCHEMA,
        }
    ]
    assert sql_executor.calls == [
        {
            "sql": generated_sql,
            "parameters": (),
        }
    ]
    assert result["route"] == "sql"
    assert result["generated_sql"] == generated_sql
    assert result["sql_result"] == SQLQueryResult(
        columns=["revenue_musd"],
        rows=[{"revenue_musd": 416161.0}],
        row_count=1,
    )


def test_sql_route_ignores_rag_query_options():
    (
        graph,
        router,
        query_service,
        sql_generator,
        sql_executor,
    ) = _graph_with_fakes("sql")
    question = "What was Apple's revenue?"

    result = graph.invoke({
        "question": question,
        "top_k": 2,
        "where": {"ticker": {"$eq": "AAPL"}},
        "company": "Apple",
        "ticker": "AAPL",
    })

    assert router.calls == [question]
    assert query_service.calls == []
    assert sql_generator.calls == [{
        "question": question,
        "schema": FINANCIAL_SCHEMA,
    }]
    assert len(sql_executor.calls) == 1
    assert result["route"] == "sql"


def test_invalid_route_fails_explicitly():
    (
        graph,
        _,
        query_service,
        sql_generator,
        sql_executor,
    ) = _graph_with_fakes("python")

    with pytest.raises(
        UnsupportedRouteError, match="Unsupported route: 'python'"
    ):
        graph.invoke({"question": "Analyze revenue"})

    assert query_service.calls == []
    assert sql_generator.calls == []
    assert sql_executor.calls == []


def test_graph_invocations_do_not_share_results():
    (
        graph,
        _,
        query_service,
        _,
        _,
    ) = _graph_with_fakes("rag")

    where = {"ticker": {"$eq": "AAPL"}}
    first_result = graph.invoke({
        "question": "First question",
        "top_k": 2,
        "where": where,
        "company": "Apple",
        "ticker": "AAPL",
    })
    second_result = graph.invoke({"question": "Second question"})

    assert query_service.calls == [
        {
            "question": "First question",
            "top_k": 2,
            "where": where,
            "company": "Apple",
            "ticker": "AAPL",
        },
        {
            "question": "Second question",
            "top_k": 5,
            "where": None,
            "company": None,
            "ticker": None,
        },
    ]
    assert first_result["answer"] == (
        "RAG answer for First question [Source 1]"
    )
    assert second_result["answer"] == (
        "RAG answer for Second question [Source 1]"
    )
    assert first_result is not second_result
    assert first_result["sources"] is not second_result["sources"]


def test_graph_runs_with_only_fake_dependencies():
    (
        graph,
        _,
        query_service,
        sql_generator,
        sql_executor,
    ) = _graph_with_fakes("sql")

    result = graph.invoke({"question": "Show revenue"})

    assert query_service.calls == []
    assert len(sql_generator.calls) == 1
    assert len(sql_executor.calls) == 1
    assert result["sql_result"].row_count == 1
