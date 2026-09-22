import pytest

from app.agent.sql_executor import SQLQueryResult
from app.rag.models import RetrievedChunk
from app.services.agent_service import (
    AgentService,
    InvalidAgentResultError,
)


class FakeGraph:
    def __init__(self, results):
        self.results = iter(results)
        self.calls = []

    def invoke(self, input):
        self.calls.append(input)
        return next(self.results)


def test_delegates_one_invocation_with_initial_question():
    question = "What risks did Apple disclose?"
    graph = FakeGraph([
        {
            "route": "rag",
            "answer": "Supply chain risks [Source 1]",
            "sources": [],
        }
    ])
    service = AgentService(graph=graph)

    service.ask(question)

    assert graph.calls == [
        {
            "question": question,
            "top_k": 5,
            "where": None,
            "company": None,
            "ticker": None,
        }
    ]


def test_initial_state_contains_supplied_query_options():
    graph = FakeGraph([{
        "route": "rag",
        "answer": "Answer [Source 1]",
        "sources": [],
    }])
    service = AgentService(graph=graph)
    where = {"document_type": {"$eq": "10-K"}}

    service.ask(
        "What risks did Apple disclose?",
        top_k=3,
        where=where,
        company="Apple",
        ticker="AAPL",
    )

    assert graph.calls == [{
        "question": "What risks did Apple disclose?",
        "top_k": 3,
        "where": where,
        "company": "Apple",
        "ticker": "AAPL",
    }]
    assert graph.calls[0]["where"] is where


def test_maps_rag_graph_result():
    source = RetrievedChunk(
        text="Risk disclosure",
        document="apple.pdf",
        page=12,
        chunk_index=3,
        distance=0.2,
    )
    service = AgentService(
        graph=FakeGraph([
            {
                "question": "ignored output state",
                "route": "rag",
                "answer": "Supply chain risks [Source 1]",
                "sources": [source],
            }
        ])
    )

    result = service.ask(
        "What risks did Apple disclose?"
    )

    assert result.route == "rag"
    assert result.answer == (
        "Supply chain risks [Source 1]"
    )
    assert result.sources == [source]
    assert result.generated_sql is None
    assert result.sql_result is None


def test_maps_sql_graph_result():
    sql_result = SQLQueryResult(
        columns=["revenue_musd"],
        rows=[{"revenue_musd": 416161.0}],
        row_count=1,
    )
    service = AgentService(
        graph=FakeGraph([
            {
                "question": "ignored output state",
                "route": "sql",
                "generated_sql": (
                    "SELECT revenue_musd "
                    "FROM financial_metrics"
                ),
                "sql_result": sql_result,
            }
        ])
    )

    result = service.ask(
        "What was Apple's revenue?"
    )

    assert result.route == "sql"
    assert result.generated_sql == (
        "SELECT revenue_musd "
        "FROM financial_metrics"
    )
    assert result.sql_result == sql_result
    assert result.answer is None
    assert result.sources == []


def test_service_state_does_not_leak_between_calls():
    first_source = RetrievedChunk(
        text="First",
        document="first.pdf",
        page=1,
        chunk_index=0,
        distance=0.1,
    )
    graph = FakeGraph([
        {
            "route": "rag",
            "answer": "First answer [Source 1]",
            "sources": [first_source],
        },
        {
            "route": "sql",
            "generated_sql": "SELECT 2",
            "sql_result": SQLQueryResult(
                columns=["2"],
                rows=[{"2": 2}],
                row_count=1,
            ),
        },
    ])
    service = AgentService(graph=graph)

    where = {"ticker": {"$eq": "AAPL"}}
    first_result = service.ask(
        "First question",
        top_k=2,
        where=where,
        company="Apple",
        ticker="AAPL",
    )
    second_result = service.ask("Second question")

    assert graph.calls == [
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
    assert graph.calls[0] is not graph.calls[1]
    assert first_result.route == "rag"
    assert first_result.sources == [first_source]
    assert second_result.route == "sql"
    assert second_result.answer is None
    assert second_result.sources == []


def test_rag_result_without_answer_raises_invalid_result_error():
    graph = FakeGraph([{
        "route": "rag",
        "sources": [],
    }])
    service = AgentService(graph=graph)

    with pytest.raises(
        InvalidAgentResultError,
        match="RAG graph result is missing an answer",
    ):
        service.ask("What risks did Apple disclose?")


def test_rag_result_without_sources_raises_invalid_result_error():
    graph = FakeGraph([{
        "route": "rag",
        "answer": "Supply chain risks [Source 1]",
    }])
    service = AgentService(graph=graph)

    with pytest.raises(
        InvalidAgentResultError,
        match="RAG graph result is missing sources",
    ):
        service.ask("What risks did Apple disclose?")


def test_sql_result_without_generated_sql_raises_invalid_result_error():
    graph = FakeGraph([{
        "route": "sql",
        "sql_result": SQLQueryResult(
            columns=[],
            rows=[],
            row_count=0,
        ),
    }])
    service = AgentService(graph=graph)

    with pytest.raises(
        InvalidAgentResultError,
        match="SQL graph result is missing generated SQL",
    ):
        service.ask("What was Apple's revenue?")


def test_sql_result_without_query_result_raises_invalid_result_error():
    graph = FakeGraph([{
        "route": "sql",
        "generated_sql": "SELECT revenue_musd FROM financial_metrics",
    }])
    service = AgentService(graph=graph)

    with pytest.raises(
        InvalidAgentResultError,
        match="SQL graph result is missing a query result",
    ):
        service.ask("What was Apple's revenue?")


def test_unsupported_graph_route_raises_invalid_result_error():
    graph = FakeGraph([{"route": "unsupported"}])
    service = AgentService(graph=graph)

    with pytest.raises(
        InvalidAgentResultError,
        match="Agent graph returned an unsupported route",
    ):
        service.ask("Analyze revenue")


def test_blank_question_raises_value_error_without_invoking_graph():
    graph = FakeGraph([])
    service = AgentService(graph=graph)

    with pytest.raises(
        ValueError,
        match="question must not be empty",
    ):
        service.ask(" \t ")

    assert graph.calls == []
