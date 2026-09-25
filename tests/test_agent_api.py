from types import MappingProxyType
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from app.agent.graph import UnsupportedRouteError, build_agent_graph
from app.agent.schema import FINANCIAL_SCHEMA
from app.agent.router import QuestionRoutingError
from app.agent.sql_executor import SQLExecutionError, SQLQueryResult, SQLValidationError
from app.agent.sql_generator import SQLGenerationError
from app.analysis.financial_analyzer import (
    AnalysisError, AnalysisOperation, AnalysisResult, FinancialAnalyzer,
)
from app.analysis.intent import AnalysisSQLTask, SQLAnalysisClassificationError
from app.analysis.planner import AnalysisPlanningError, PercentageChangePlan
from app.dependencies import get_agent_service
from app.main import app
from app.rag.models import RetrievedChunk
from app.rag.validation import AnswerValidationError
from app.services.agent_service import (
    AgentResult, AgentService, InvalidAgentResultError,
)
from app.services.query_service import AmbiguousQueryError


class FakeAgentService:
    def __init__(self, result: AgentResult | None = None, error: Exception | None = None):
        self.result = result
        self.error = error
        self.calls: list[dict] = []

    def ask(
        self,
        question: str,
        top_k: int = 5,
        where: dict | None = None,
        company: str | None = None,
        ticker: str | None = None,
    ) -> AgentResult:
        self.calls.append({
            "question": question,
            "top_k": top_k,
            "where": where,
            "company": company,
            "ticker": ticker,
        })
        if self.error is not None:
            raise self.error
        assert self.result is not None
        return self.result


@pytest.fixture
def client_with_service():
    def make_client(service: FakeAgentService) -> TestClient:
        app.dependency_overrides[get_agent_service] = lambda: service
        return TestClient(app)

    yield make_client
    app.dependency_overrides.pop(get_agent_service, None)


def test_rag_response_serialization_and_exact_scoping(client_with_service):
    source = RetrievedChunk(
        text="Document body is excluded",
        document="apple.pdf",
        page=12,
        chunk_index=3,
        distance=0.2,
        company="Apple",
        ticker="AAPL",
        fiscal_year=2025,
        document_type="10-K",
    )
    service = FakeAgentService(AgentResult(
        route="rag",
        answer="Risk disclosure [Source 1]",
        sources=[source],
    ))
    client = client_with_service(service)
    question = "  What risks did Apple disclose?  "

    response = client.post("/agent/ask", json={
        "question": question,
        "top_k": 3,
        "company": "Apple",
        "ticker": "AAPL",
        "fiscal_year": 2025,
        "document_type": "10-K",
    })

    assert response.status_code == 200
    assert response.json() == {
        "route": "rag",
        "answer": "Risk disclosure [Source 1]",
        "sources": [{
            "document": "apple.pdf",
            "page": 12,
            "chunk_index": 3,
            "distance": 0.2,
            "company": "Apple",
            "ticker": "AAPL",
            "fiscal_year": 2025,
            "document_type": "10-K",
        }],
    }
    assert service.calls == [{
        "question": question,
        "top_k": 3,
        "where": {"$and": [
            {"company": {"$eq": "Apple"}},
            {"ticker": {"$eq": "AAPL"}},
            {"fiscal_year": {"$eq": 2025}},
            {"document_type": {"$eq": "10-K"}},
        ]},
        "company": "Apple",
        "ticker": "AAPL",
    }]


def test_sql_response_serialization_and_default_options(client_with_service):
    service = FakeAgentService(AgentResult(
        route="sql",
        generated_sql="SELECT revenue FROM financial_metrics",
        sql_result=SQLQueryResult(
            columns=["revenue"],
            rows=[{"revenue": 416161.0}],
            row_count=1,
        ),
    ))
    client = client_with_service(service)

    response = client.post("/agent/ask", json={
        "question": "What was Apple's revenue?",
    })

    assert response.status_code == 200
    assert response.json() == {
        "route": "sql",
        "generated_sql": "SELECT revenue FROM financial_metrics",
        "sql_result": {
            "columns": ["revenue"],
            "rows": [{"revenue": 416161.0}],
            "row_count": 1,
        },
    }
    assert service.calls == [{
        "question": "What was Apple's revenue?",
        "top_k": 5,
        "where": None,
        "company": None,
        "ticker": None,
    }]


def test_sql_empty_result_serialization(client_with_service):
    service = FakeAgentService(AgentResult(
        route="sql",
        generated_sql="SELECT revenue FROM financial_metrics WHERE 1 = 0",
        sql_result=SQLQueryResult(
            columns=["revenue"],
            rows=[],
            row_count=0,
        ),
    ))
    client = client_with_service(service)

    response = client.post("/agent/ask", json={"question": "Missing revenue?"})

    assert response.status_code == 200
    assert response.json() == {
        "route": "sql",
        "generated_sql": "SELECT revenue FROM financial_metrics WHERE 1 = 0",
        "sql_result": {
            "columns": ["revenue"],
            "rows": [],
            "row_count": 0,
        },
    }


@pytest.mark.parametrize("body", [
    {"question": ""},
    {"question": " \t\n "},
    {"question": "Valid?", "top_k": 0},
    {"question": "Valid?", "top_k": 21},
    {"question": 123},
    {},
])
def test_invalid_requests_are_rejected_without_calling_service(
    client_with_service, body
):
    service = FakeAgentService()
    client = client_with_service(service)

    response = client.post("/agent/ask", json=body)

    assert response.status_code == 422
    assert service.calls == []


@pytest.mark.parametrize(("error", "status", "detail"), [
    (AmbiguousQueryError("Specify a company."), 400, "Specify a company."),
    (AnswerValidationError("private detail"), 502,
     "The language model produced an invalid source-grounded response."),
    (QuestionRoutingError("private detail"), 502,
     "The agent could not complete the request."),
    (UnsupportedRouteError("private detail"), 502,
     "The agent could not complete the request."),
    (SQLGenerationError("private detail"), 502,
     "The agent could not complete the request."),
    (SQLExecutionError("private detail"), 502,
     "The agent could not complete the request."),
    (SQLValidationError("private detail"), 502,
     "The agent could not complete the request."),
    (InvalidAgentResultError("private detail"), 502,
     "The agent could not complete the request."),
    (SQLAnalysisClassificationError("private model detail"), 502,
     "The agent could not complete the request."),
    (AnalysisPlanningError("private model detail"), 502,
     "The agent could not complete the request."),
    (AnalysisError("private financial data"), 502,
     "The agent could not complete the request."),
])
def test_expected_application_errors_have_stable_http_mapping(
    client_with_service, error, status, detail
):
    service = FakeAgentService(error=error)
    client = client_with_service(service)

    response = client.post("/agent/ask", json={"question": "A question"})

    assert response.status_code == status
    assert response.json() == {"detail": detail}
    assert len(service.calls) == 1


def test_endpoint_reads_agent_service_from_app_state():
    service = FakeAgentService(AgentResult(route="rag", answer="Answer", sources=[]))
    previous_service = getattr(app.state, "agent_service", None)
    had_service = hasattr(app.state, "agent_service")
    app.state.agent_service = service
    try:
        response = TestClient(app).post("/agent/ask", json={"question": "Question"})
        assert response.status_code == 200
        assert len(service.calls) == 1
    finally:
        if had_service:
            app.state.agent_service = previous_service
        else:
            del app.state.agent_service


def test_response_schema_is_discriminated_by_route():
    schema = app.openapi()["paths"]["/agent/ask"]["post"]["responses"]["200"]
    response_schema = schema["content"]["application/json"]["schema"]

    assert response_schema["discriminator"]["propertyName"] == "route"
    assert len(response_schema["oneOf"]) == 2


@pytest.mark.parametrize("analysis, expected", [
    (
        AnalysisResult(AnalysisOperation.PERCENTAGE_CHANGE, value=25.0),
        {"operation": "percentage_change", "value": 25.0, "ranked_rows": []},
    ),
    (
        AnalysisResult(AnalysisOperation.RANKING, ranked_rows=(
            MappingProxyType({"ticker": "MSFT", "revenue_musd": 125}),
            MappingProxyType({"ticker": "AAPL", "revenue_musd": 100}),
        )),
        {"operation": "ranking", "value": None, "ranked_rows": [
            {"ticker": "MSFT", "revenue_musd": 125},
            {"ticker": "AAPL", "revenue_musd": 100},
        ]},
    ),
])
def test_analysis_response_serializes_domain_result(
    client_with_service, analysis, expected,
):
    rows = [
        {"ticker": "AAPL", "revenue_musd": 100},
        {"ticker": "MSFT", "revenue_musd": 125},
    ]
    service = FakeAgentService(AgentResult(
        route="sql", generated_sql="SELECT ticker, revenue_musd FROM financial_metrics",
        sql_result=SQLQueryResult(
            columns=["ticker", "revenue_musd"], rows=rows, row_count=2,
        ),
        analysis_result=analysis,
    ))

    response = client_with_service(service).post(
        "/agent/ask", json={"question": "Analyze revenue"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "route": "sql",
        "generated_sql": "SELECT ticker, revenue_musd FROM financial_metrics",
        "sql_result": {
            "columns": ["ticker", "revenue_musd"], "rows": rows, "row_count": 2,
        },
        "analysis_result": expected,
    }
    if analysis.ranked_rows:
        assert isinstance(analysis.ranked_rows[0], MappingProxyType)
        with pytest.raises(TypeError):
            analysis.ranked_rows[0]["revenue_musd"] = 0


def test_rag_optional_source_fields_still_serialize_as_null(client_with_service):
    service = FakeAgentService(AgentResult(
        route="rag", answer="Answer", sources=[RetrievedChunk(
            text="text", document="filing.pdf", page=1, chunk_index=0, distance=0.1,
        )],
    ))

    response = client_with_service(service).post(
        "/agent/ask", json={"question": "Risks?"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "route": "rag", "answer": "Answer", "sources": [{
            "document": "filing.pdf", "page": 1, "chunk_index": 0, "distance": 0.1,
            "company": None, "ticker": None, "fiscal_year": None, "document_type": None,
        }],
    }


def test_analysis_repair_graph_service_and_http_work_together(client_with_service):
    raw = SQLQueryResult(
        columns=["fiscal_year", "revenue_musd"],
        rows=[{"fiscal_year": 2024, "revenue_musd": 100},
              {"fiscal_year": 2025, "revenue_musd": 125}], row_count=2,
    )
    operation = AnalysisOperation.PERCENTAGE_CHANGE
    generator = Mock()
    generator.generate_analysis_data.return_value = "SELECT fiscal_year, revenue"
    final_sql = "SELECT fiscal_year, revenue_musd FROM financial_metrics"
    generator.repair_analysis_data.return_value = final_sql
    executor = Mock(execute=Mock(side_effect=[
        SQLExecutionError("no such column"), raw,
    ]))
    planner = Mock(plan=Mock(return_value=PercentageChangePlan(
        operation=operation, old_row=0, old_column="revenue_musd",
        new_row=1, new_column="revenue_musd",
    )))
    graph = build_agent_graph(
        router=Mock(route=Mock(return_value="sql")), query_service=Mock(),
        sql_generator=generator, sql_executor=executor,
        sql_analysis_classifier=Mock(classify=Mock(return_value=AnalysisSQLTask(
            mode="analysis", operation=operation,
        ))),
        analysis_planner=planner, financial_analyzer=FinancialAnalyzer(),
    )
    question = "By what percentage did revenue change?"

    response = client_with_service(AgentService(graph)).post(
        "/agent/ask", json={"question": question},
    )

    assert response.status_code == 200
    assert response.json() == {
        "route": "sql", "generated_sql": final_sql,
        "sql_result": {"columns": raw.columns, "rows": raw.rows, "row_count": 2},
        "analysis_result": {"operation": "percentage_change", "value": 25.0,
                            "ranked_rows": []},
    }
    generator.generate_analysis_data.assert_called_once_with(
        question=question, schema=FINANCIAL_SCHEMA, operation=operation,
    )
    generator.repair_analysis_data.assert_called_once_with(
        question=question, schema=FINANCIAL_SCHEMA, operation=operation,
        previous_sql="SELECT fiscal_year, revenue", error_message="no such column",
    )
    generator.generate.assert_not_called()
    generator.repair.assert_not_called()
    planner.plan.assert_called_once_with(question=question, sql_result=raw)
