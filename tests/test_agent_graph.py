from unittest.mock import Mock

import pytest

from app.agent.graph import (
    UnsupportedRouteError,
    build_agent_graph,
)
from app.agent.schema import FINANCIAL_SCHEMA
from app.agent.sql_executor import (
    SQLExecutionError,
    SQLQueryResult,
    SQLValidationError,
)
from app.rag.models import RetrievedChunk
from app.rag.pipeline import RAGResult
from app.services.agent_service import AgentService
from app.analysis.chart_data import ChartDataBuilder, ChartDataError
from app.analysis.chart_decision import (
    ChartDecisionError, ChartDecisionPolicy, ChartIntentClassifier, ChartIntentResponse,
)
from app.analysis.chart_planner import ChartPlanner, ChartPlanningError
from app.analysis.chart_renderer import (
    ChartArtifact, ChartRenderer, ChartRenderingError, ChartSpec, ChartType,
)
from app.analysis.financial_analyzer import (
    AnalysisError, AnalysisOperation, FinancialAnalyzer, RankingDirection,
)
from app.analysis.intent import (
    AnalysisSQLTask, DirectSQLTask, SQLAnalysisClassificationError,
    SQLAnalysisClassifier,
)
from app.analysis.planner import (
    AbsoluteChangePlan, AnalysisPlanner, AnalysisPlanningError,
    DifferencePlan, PercentageChangePlan, RankingPlan,
)


INITIAL_SQL = "SELECT revenue_musd FROM financial_metrics"
FIRST_REPAIR = "SELECT revenue_musd FROM financial_metrics WHERE ticker = 'AAPL'"
SECOND_REPAIR = (
    "SELECT revenue_musd FROM financial_metrics "
    "WHERE ticker = 'AAPL' AND fiscal_year = 2025"
)


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
    def __init__(self, repaired_sqls: list[str] | None = None):
        self.calls = []
        self.repair_calls = []
        self.repaired_sqls = iter(repaired_sqls or [])
        self.analysis_calls = []
        self.analysis_repair_calls = []

    def generate(self, question: str, schema: str, *, result_mode="answer") -> str:
        self.calls.append({
            "question": question,
            "schema": schema,
            "result_mode": result_mode,
        })

        return INITIAL_SQL

    def repair(
        self,
        question: str,
        schema: str,
        previous_sql: str,
        error_message: str,
        *,
        result_mode="answer",
    ) -> str:
        self.repair_calls.append({
            "question": question,
            "schema": schema,
            "previous_sql": previous_sql,
            "error_message": error_message,
            "result_mode": result_mode,
        })
        return next(self.repaired_sqls)

    def generate_analysis_data(self, question, schema, operation):
        self.analysis_calls.append({
            "question": question, "schema": schema, "operation": operation,
        })
        return RAW_SQL

    def repair_analysis_data(
        self, question, schema, operation, previous_sql, error_message
    ):
        self.analysis_repair_calls.append({
            "question": question, "schema": schema, "operation": operation,
            "previous_sql": previous_sql, "error_message": error_message,
        })
        return next(self.repaired_sqls)


class FakeSQLExecutor:
    def __init__(
        self,
        outcomes: list[SQLQueryResult | Exception] | None = None,
    ):
        self.calls = []
        self.outcomes = iter(outcomes) if outcomes is not None else None

    def execute(
        self,
        sql: str,
        parameters: tuple[object, ...] = (),
    ) -> SQLQueryResult:
        self.calls.append({
            "sql": sql,
            "parameters": parameters,
        })

        if self.outcomes is not None:
            outcome = next(self.outcomes)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        return SQLQueryResult(
            columns=["revenue_musd"],
            rows=[{"revenue_musd": 416161.0}],
            row_count=1,
        )


def _graph_with_fakes(
    route: str,
    max_sql_retries: int = 2,
    repaired_sqls: list[str] | None = None,
    execution_outcomes: list[SQLQueryResult | Exception] | None = None,
    *,
    classifier=None,
    planner=None,
    analyzer=None,
    chart_dependencies=None,
):
    router = FakeRouter(route)
    query_service = FakeQueryService()
    sql_generator = FakeSQLGenerator(repaired_sqls)
    sql_executor = FakeSQLExecutor(execution_outcomes)

    graph = build_agent_graph(
        router=router,
        query_service=query_service,
        sql_generator=sql_generator,
        sql_executor=sql_executor,
        max_sql_retries=max_sql_retries,
        sql_analysis_classifier=(
            classifier if classifier is not None
            else Mock(spec=SQLAnalysisClassifier,
                      classify=Mock(return_value=DirectSQLTask(mode="direct")))
        ),
        analysis_planner=(
            planner if planner is not None
            else Mock(spec=AnalysisPlanner,
                      plan=Mock(side_effect=AssertionError("Unexpected planning")))
        ),
        financial_analyzer=(
            analyzer if analyzer is not None
            else Mock(spec=FinancialAnalyzer)
        ),
        **(_chart_dependencies() if chart_dependencies is None else chart_dependencies),
    )

    return (
        graph,
        router,
        query_service,
        sql_generator,
        sql_executor,
    )


def _chart_dependencies(intent="lookup"):
    return {
        "chart_data_builder": Mock(spec=ChartDataBuilder, wraps=ChartDataBuilder()),
        "chart_intent_classifier": Mock(spec=ChartIntentClassifier, classify=Mock(
            return_value=ChartIntentResponse(intent=intent)
        )),
        "chart_decision_policy": Mock(spec=ChartDecisionPolicy, wraps=ChartDecisionPolicy()),
        "chart_planner": Mock(spec=ChartPlanner, plan=Mock(return_value=ChartSpec(
            chart_type=ChartType.BAR, x_column="ticker", y_column="revenue_musd",
            title="Revenue",
        ))),
        "chart_renderer": Mock(spec=ChartRenderer, render=Mock(return_value=ChartArtifact(
            media_type="image/png", content=b"fake png",
        ))),
    }


def test_rag_route_delegates_to_query_service():
    charts = _chart_dependencies()
    (
        graph,
        router,
        query_service,
        sql_generator,
        sql_executor,
    ) = _graph_with_fakes("rag", chart_dependencies=charts)

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
    assert sql_generator.repair_calls == []
    assert sql_executor.calls == []
    assert result["route"] == "rag"
    assert result["answer"] == f"RAG answer for {question} [Source 1]"
    assert len(result["sources"]) == 1
    assert all(component.mock_calls == [] for component in charts.values())
    assert not any(key.startswith("chart_") for key in result)


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
    assert sql_generator.repair_calls == []
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
            "result_mode": "answer",
        }
    ]
    assert sql_executor.calls == [
        {
            "sql": generated_sql,
            "parameters": (),
        }
    ]
    assert sql_generator.repair_calls == []
    assert result["route"] == "sql"
    assert result["generated_sql"] == generated_sql
    assert result["sql_retry_count"] == 0
    assert result["sql_error"] is None
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
        "result_mode": "answer",
    }]
    assert len(sql_executor.calls) == 1
    assert result["route"] == "sql"


def test_first_sql_repair_succeeds_with_final_sql_and_result():
    question = "What was Apple's 2025 revenue?"
    expected_result = SQLQueryResult(
        columns=["revenue_musd"],
        rows=[{"revenue_musd": 416161.0}],
        row_count=1,
    )
    graph, _, query_service, generator, executor = _graph_with_fakes(
        "sql",
        repaired_sqls=[FIRST_REPAIR],
        execution_outcomes=[
            SQLExecutionError("no such column: revenue"),
            expected_result,
        ],
    )

    result = graph.invoke({"question": question})

    assert query_service.calls == []
    assert generator.calls == [{
        "question": question,
        "schema": FINANCIAL_SCHEMA,
        "result_mode": "answer",
    }]
    assert generator.repair_calls == [{
        "question": question,
        "schema": FINANCIAL_SCHEMA,
        "previous_sql": INITIAL_SQL,
        "error_message": "no such column: revenue",
        "result_mode": "answer",
    }]
    assert [call["sql"] for call in executor.calls] == [
        INITIAL_SQL,
        FIRST_REPAIR,
    ]
    assert result["generated_sql"] == FIRST_REPAIR
    assert result["sql_result"] == expected_result
    assert result["sql_retry_count"] == 1
    assert result["sql_error"] is None


def test_two_sql_repairs_succeed_within_retry_budget():
    expected_result = SQLQueryResult(
        columns=["revenue_musd"],
        rows=[{"revenue_musd": 416161.0}],
        row_count=1,
    )
    graph, _, _, generator, executor = _graph_with_fakes(
        "sql",
        max_sql_retries=2,
        repaired_sqls=[FIRST_REPAIR, SECOND_REPAIR],
        execution_outcomes=[
            SQLExecutionError("first failure"),
            SQLExecutionError("second failure"),
            expected_result,
        ],
    )

    result = graph.invoke({"question": "Revenue in 2025?"})

    assert len(generator.calls) == 1
    assert len(generator.repair_calls) == 2
    assert generator.repair_calls[1]["previous_sql"] == FIRST_REPAIR
    assert generator.repair_calls[1]["error_message"] == "second failure"
    assert [call["sql"] for call in executor.calls] == [
        INITIAL_SQL,
        FIRST_REPAIR,
        SECOND_REPAIR,
    ]
    assert result["generated_sql"] == SECOND_REPAIR
    assert result["sql_result"] == expected_result
    assert result["sql_retry_count"] == 2
    assert result["sql_error"] is None


def test_sql_retry_exhaustion_raises_final_execution_error():
    graph, _, _, generator, executor = _graph_with_fakes(
        "sql",
        max_sql_retries=2,
        repaired_sqls=[FIRST_REPAIR, SECOND_REPAIR],
        execution_outcomes=[
            SQLExecutionError("first failure"),
            SQLExecutionError("second failure"),
            SQLExecutionError("final failure"),
        ],
    )

    with pytest.raises(SQLExecutionError, match="final failure") as exc_info:
        graph.invoke({"question": "Revenue in 2025?"})

    assert str(exc_info.value).startswith("final failure")
    assert len(generator.calls) == 1
    assert len(generator.repair_calls) == 2
    assert [call["sql"] for call in executor.calls] == [
        INITIAL_SQL,
        FIRST_REPAIR,
        SECOND_REPAIR,
    ]


def test_zero_sql_retries_fails_after_initial_execution():
    graph, _, _, generator, executor = _graph_with_fakes(
        "sql",
        max_sql_retries=0,
        execution_outcomes=[SQLExecutionError("initial failure")],
    )

    with pytest.raises(SQLExecutionError, match="initial failure") as exc_info:
        graph.invoke({"question": "Revenue in 2025?"})

    assert str(exc_info.value).startswith("initial failure")
    assert len(generator.calls) == 1
    assert generator.repair_calls == []
    assert [call["sql"] for call in executor.calls] == [INITIAL_SQL]


def test_sql_validation_failure_is_not_repaired():
    graph, _, _, generator, executor = _graph_with_fakes(
        "sql",
        repaired_sqls=[FIRST_REPAIR],
        execution_outcomes=[SQLValidationError("Only SELECT queries are allowed.")],
    )

    with pytest.raises(SQLValidationError, match="Only SELECT queries are allowed"):
        graph.invoke({"question": "Revenue in 2025?"})

    assert generator.repair_calls == []
    assert [call["sql"] for call in executor.calls] == [INITIAL_SQL]


def test_negative_sql_retry_budget_is_rejected():
    with pytest.raises(ValueError, match="max_sql_retries must be non-negative"):
        _graph_with_fakes("sql", max_sql_retries=-1)


def test_retry_budget_above_default_graph_recursion_limit_can_succeed():
    repaired_sqls = [f"SELECT {index}" for index in range(12)]
    expected_result = SQLQueryResult(
        columns=["value"], rows=[{"value": 1}], row_count=1
    )
    graph, _, _, generator, executor = _graph_with_fakes(
        "sql",
        max_sql_retries=12,
        repaired_sqls=repaired_sqls,
        execution_outcomes=[
            *[SQLExecutionError("retry") for _ in range(12)],
            expected_result,
        ],
    )

    result = graph.invoke({"question": "Revenue?"})

    assert len(generator.calls) == 1
    assert len(generator.repair_calls) == 12
    assert len(executor.calls) == 13
    assert result["generated_sql"] == repaired_sqls[-1]
    assert result["sql_retry_count"] == 12
    assert result["sql_result"] == expected_result


def test_sql_retry_state_does_not_leak_between_graph_invocations():
    first_result = SQLQueryResult(columns=["first"], rows=[{"first": 1}], row_count=1)
    second_result = SQLQueryResult(
        columns=["second"], rows=[{"second": 2}], row_count=1
    )
    graph, router, _, generator, executor = _graph_with_fakes(
        "sql",
        repaired_sqls=[FIRST_REPAIR],
        execution_outcomes=[
            SQLExecutionError("first failure"),
            first_result,
            second_result,
        ],
    )

    first = graph.invoke({"question": "First question"})
    second = graph.invoke({"question": "Second question"})

    assert router.calls == ["First question", "Second question"]
    assert len(generator.calls) == 2
    assert len(generator.repair_calls) == 1
    assert [call["sql"] for call in executor.calls] == [
        INITIAL_SQL,
        FIRST_REPAIR,
        INITIAL_SQL,
    ]
    assert first["sql_retry_count"] == 1
    assert first["sql_result"] == first_result
    assert second["sql_retry_count"] == 0
    assert second["sql_error"] is None
    assert second["generated_sql"] == INITIAL_SQL
    assert second["sql_result"] == second_result


def test_agent_service_returns_successful_repaired_sql():
    expected_result = SQLQueryResult(
        columns=["value"], rows=[{"value": 1}], row_count=1
    )
    graph, _, _, generator, executor = _graph_with_fakes(
        "sql",
        repaired_sqls=[FIRST_REPAIR],
        execution_outcomes=[SQLExecutionError("first failure"), expected_result],
    )

    result = AgentService(graph=graph).ask("Revenue?")

    assert result.route == "sql"
    assert result.generated_sql == FIRST_REPAIR
    assert result.sql_result == expected_result
    assert result.answer is None
    assert len(generator.calls) == 1
    assert len(generator.repair_calls) == 1
    assert [call["sql"] for call in executor.calls] == [INITIAL_SQL, FIRST_REPAIR]


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
    assert sql_generator.repair_calls == []
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


RAW_SQL = "SELECT ticker, fiscal_year, revenue_musd FROM financial_metrics"
REPAIRED_RAW_SQL = RAW_SQL + " ORDER BY fiscal_year"


def _raw_result():
    return SQLQueryResult(
        columns=["ticker", "fiscal_year", "revenue_musd"],
        rows=[
            {"ticker": "AAPL", "fiscal_year": 2024, "revenue_musd": 100},
            {"ticker": "AAPL", "fiscal_year": 2025, "revenue_musd": 125},
        ],
        row_count=2,
    )


def _analysis_dependencies(operation=AnalysisOperation.PERCENTAGE_CHANGE):
    if operation == AnalysisOperation.PERCENTAGE_CHANGE:
        plan = PercentageChangePlan(
            operation=operation, old_row=0, old_column="revenue_musd",
            new_row=1, new_column="revenue_musd",
        )
    elif operation == AnalysisOperation.ABSOLUTE_CHANGE:
        plan = AbsoluteChangePlan(
            operation=operation, old_row=0, old_column="revenue_musd",
            new_row=1, new_column="revenue_musd",
        )
    elif operation == AnalysisOperation.DIFFERENCE:
        plan = DifferencePlan(
            operation=operation, left_row=1, left_column="revenue_musd",
            right_row=0, right_column="revenue_musd",
        )
    else:
        plan = RankingPlan(
            operation=operation, column="revenue_musd",
            direction=RankingDirection.DESCENDING,
        )
    return {
        "classifier": Mock(spec=SQLAnalysisClassifier, classify=Mock(
            return_value=AnalysisSQLTask(mode="analysis", operation=operation)
        )),
        "planner": Mock(spec=AnalysisPlanner, plan=Mock(return_value=plan)),
        "analyzer": Mock(spec=FinancialAnalyzer, wraps=FinancialAnalyzer()),
    }


@pytest.mark.parametrize("operation", list(AnalysisOperation))
def test_analysis_success_dispatches_to_the_existing_analyzer(operation):
    dependencies = _analysis_dependencies(operation)
    charts = _chart_dependencies()
    raw = _raw_result()
    graph, _, query, generator, executor = _graph_with_fakes(
        "sql", execution_outcomes=[raw], chart_dependencies=charts, **dependencies
    )
    question = "Analyze these financial values"

    result = graph.invoke({"question": question})

    dependencies["classifier"].classify.assert_called_once_with(question)
    assert generator.analysis_calls == [{
        "question": question, "schema": FINANCIAL_SCHEMA, "operation": operation,
    }]
    assert generator.calls == generator.repair_calls == []
    assert executor.calls == [{"sql": RAW_SQL, "parameters": ()}]
    dependencies["planner"].plan.assert_called_once_with(
        question=question, sql_result=raw,
    )
    assert len(dependencies["analyzer"].mock_calls) == 1
    name, args, kwargs = dependencies["analyzer"].mock_calls[0]
    assert name == operation.value
    assert args == (raw,)
    expected_arguments = dependencies["planner"].plan.return_value.model_dump(
        exclude={"operation"}
    )
    assert kwargs == expected_arguments
    assert result["sql_task_mode"] == "analysis"
    assert result["analysis_operation"] is operation
    assert result["analysis_plan"] is dependencies["planner"].plan.return_value
    assert result["analysis_result"].operation is operation
    if operation == AnalysisOperation.RANKING:
        assert [
            row["revenue_musd"] for row in result["analysis_result"].ranked_rows
        ] == [125, 100]
    else:
        assert result["analysis_result"].value == 25
    assert result["sql_result"] is raw
    assert result["generated_sql"] == RAW_SQL
    assert result["sql_retry_count"] == 0
    assert result["sql_error"] is None
    assert query.calls == []
    charts["chart_intent_classifier"].classify.assert_not_called()
    charts["chart_data_builder"].build.assert_called_once_with(
        raw, analysis_result=result["analysis_result"],
    )
    if operation is not AnalysisOperation.RANKING:
        assert result["chart_data"] is None
        assert "chart_intent" not in result
        assert "chart_artifact" not in result
        for name in ("chart_decision_policy", "chart_planner", "chart_renderer"):
            assert charts[name].mock_calls == []


def test_analysis_graph_has_explicit_generation_planning_and_execution_nodes():
    dependencies = _analysis_dependencies()
    graph, *_ = _graph_with_fakes(
        "sql", execution_outcomes=[_raw_result()], **dependencies,
    )

    updates = list(graph.stream({"question": "Analyze revenue"}))

    assert [name for update in updates for name in update] == [
        "route_question", "classify_sql_task", "generate_analysis_data",
        "execute_sql", "plan_analysis", "execute_analysis",
        "prepare_chart_data",
    ]


@pytest.mark.parametrize("route", ["rag", "sql"])
def test_rag_and_direct_sql_skip_analysis_components(route):
    dependencies = _analysis_dependencies()
    dependencies["classifier"].classify.return_value = DirectSQLTask(mode="direct")
    graph, _, _, generator, _ = _graph_with_fakes(route, **dependencies)

    result = graph.invoke({"question": "A question"})

    assert dependencies["classifier"].classify.call_count == (route == "sql")
    assert dependencies["planner"].mock_calls == []
    assert dependencies["analyzer"].mock_calls == []
    assert generator.analysis_calls == generator.analysis_repair_calls == []
    assert not any(key.startswith("analysis_") for key in result)
    assert len(generator.calls) == (route == "sql")


def test_analysis_repair_uses_the_raw_data_repair_method_and_finishes_analysis():
    dependencies = _analysis_dependencies()
    raw = _raw_result()
    graph, _, _, generator, executor = _graph_with_fakes(
        "sql", repaired_sqls=[REPAIRED_RAW_SQL],
        execution_outcomes=[SQLExecutionError("no such column: revenue"), raw],
        **dependencies,
    )

    result = AgentService(graph).ask("Revenue growth?")

    assert generator.repair_calls == generator.calls == []
    assert generator.analysis_repair_calls == [{
        "question": "Revenue growth?", "schema": FINANCIAL_SCHEMA,
        "operation": AnalysisOperation.PERCENTAGE_CHANGE,
        "previous_sql": RAW_SQL, "error_message": "no such column: revenue",
    }]
    assert [call["sql"] for call in executor.calls] == [RAW_SQL, REPAIRED_RAW_SQL]
    dependencies["planner"].plan.assert_called_once_with(
        question="Revenue growth?", sql_result=raw,
    )
    assert result.generated_sql == REPAIRED_RAW_SQL
    assert result.sql_result is raw
    assert result.analysis_result.value == 25


def test_direct_repair_does_not_use_analysis_repair():
    graph, _, _, generator, _ = _graph_with_fakes(
        "sql", repaired_sqls=[FIRST_REPAIR],
        execution_outcomes=[SQLExecutionError("failure"), _raw_result()],
    )

    result = graph.invoke({"question": "Revenue?"})

    assert len(generator.repair_calls) == 1
    assert generator.analysis_repair_calls == []
    assert result["sql_task_mode"] == "direct"


@pytest.mark.parametrize("budget", [0, 2, 12])
def test_analysis_retry_budget_can_succeed_on_last_allowed_execution(budget):
    dependencies = _analysis_dependencies()
    graph, _, _, generator, executor = _graph_with_fakes(
        "sql", max_sql_retries=budget,
        repaired_sqls=[REPAIRED_RAW_SQL] * budget,
        execution_outcomes=[*[SQLExecutionError("failure")] * budget, _raw_result()],
        **dependencies,
    )

    result = graph.invoke({"question": "Growth?"})

    assert len(generator.analysis_calls) == 1
    assert len(generator.analysis_repair_calls) == budget
    assert len(executor.calls) == budget + 1
    assert result["sql_retry_count"] == budget
    assert result["sql_error"] is None
    assert result["analysis_result"].value == 25


@pytest.mark.parametrize("budget", [0, 2])
def test_analysis_retry_exhaustion_never_runs_planner_or_analyzer(budget):
    dependencies = _analysis_dependencies()
    graph, _, _, generator, executor = _graph_with_fakes(
        "sql", max_sql_retries=budget,
        repaired_sqls=[REPAIRED_RAW_SQL] * budget,
        execution_outcomes=[SQLExecutionError("final failure")] * (budget + 1),
        **dependencies,
    )

    with pytest.raises(SQLExecutionError, match="final failure"):
        graph.invoke({"question": "Growth?"})

    assert len(generator.analysis_repair_calls) == budget
    assert len(executor.calls) == budget + 1
    assert generator.repair_calls == []
    assert dependencies["planner"].mock_calls == []
    assert dependencies["analyzer"].mock_calls == []


def test_analysis_sql_validation_failure_is_not_repaired():
    dependencies = _analysis_dependencies()
    graph, _, _, generator, executor = _graph_with_fakes(
        "sql", execution_outcomes=[SQLValidationError("unsafe")], **dependencies
    )

    with pytest.raises(SQLValidationError, match="unsafe"):
        graph.invoke({"question": "Growth?"})

    assert generator.repair_calls == generator.analysis_repair_calls == []
    assert len(executor.calls) == 1
    assert dependencies["planner"].mock_calls == []
    assert dependencies["analyzer"].mock_calls == []


def test_direct_analysis_direct_invocations_keep_state_isolated():
    dependencies = _analysis_dependencies()
    dependencies["classifier"].classify.side_effect = [
        DirectSQLTask(mode="direct"),
        AnalysisSQLTask(mode="analysis", operation=AnalysisOperation.PERCENTAGE_CHANGE),
        DirectSQLTask(mode="direct"),
    ]
    graph, _, _, generator, _ = _graph_with_fakes(
        "sql", repaired_sqls=[FIRST_REPAIR, REPAIRED_RAW_SQL],
        execution_outcomes=[
            SQLExecutionError("direct failure"), _raw_result(),
            SQLExecutionError("analysis failure"), _raw_result(), _raw_result(),
        ], **dependencies,
    )

    first = graph.invoke({"question": "First"})
    second = graph.invoke({"question": "Second"})
    third = graph.invoke({"question": "Third"})

    assert [r["sql_retry_count"] for r in (first, second, third)] == [1, 1, 0]
    assert all(r["sql_error"] is None for r in (first, second, third))
    assert "analysis_result" in second
    for result in (first, third):
        assert not any(key.startswith("analysis_") for key in result)
    assert len(generator.repair_calls) == len(generator.analysis_repair_calls) == 1
    assert dependencies["planner"].plan.call_count == 1


@pytest.mark.parametrize("stage", ["classify", "plan", "calculate"])
def test_analysis_component_failure_propagates_without_direct_fallback(stage):
    dependencies = _analysis_dependencies()
    errors = {
        "classify": SQLAnalysisClassificationError("classification failed"),
        "plan": AnalysisPlanningError("planning failed"),
        "calculate": AnalysisError("calculation failed"),
    }
    method = {
        "classify": dependencies["classifier"].classify,
        "plan": dependencies["planner"].plan,
        "calculate": dependencies["analyzer"].percentage_change,
    }[stage]
    method.side_effect = errors[stage]
    graph, _, _, generator, _ = _graph_with_fakes(
        "sql", execution_outcomes=[_raw_result()], **dependencies,
    )

    with pytest.raises(type(errors[stage])) as exc:
        graph.invoke({"question": "Growth?"})

    assert exc.value is errors[stage]
    assert generator.calls == generator.repair_calls == []


@pytest.mark.parametrize("plan", [
    object(),
    RankingPlan(operation=AnalysisOperation.RANKING, column="revenue_musd",
                direction=RankingDirection.DESCENDING),
])
def test_unexpected_or_mismatched_analysis_plan_fails_explicitly(plan):
    dependencies = _analysis_dependencies()
    dependencies["planner"].plan.return_value = plan
    graph, *_ = _graph_with_fakes(
        "sql", execution_outcomes=[_raw_result()], **dependencies,
    )

    with pytest.raises(AnalysisPlanningError):
        graph.invoke({"question": "Growth?"})

    assert dependencies["analyzer"].mock_calls == []


def test_unexpected_sql_classifier_decision_fails_without_generating_sql():
    dependencies = _analysis_dependencies()
    dependencies["classifier"].classify.return_value = {"mode": "unsupported"}
    graph, _, _, generator, executor = _graph_with_fakes("sql", **dependencies)

    with pytest.raises(SQLAnalysisClassificationError, match="Unexpected SQL task"):
        graph.invoke({"question": "Growth?"})

    assert generator.calls == generator.analysis_calls == executor.calls == []


@pytest.mark.parametrize("question,intent,rows,mode,decision", [
    ("What was Apple's 2025 revenue?", "lookup", 1, "answer", "none"),
    ("List Apple and Microsoft revenue rows.", "list", 2, "answer", "none"),
    ("Plot Apple's 2025 revenue.", "lookup", 1, "chart_ready", "chart"),
    ("Compare Apple and Microsoft revenue.", "comparison", 2, "chart_ready", "chart"),
    ("Show Apple's revenue trend.", "trend", 2, "chart_ready", "chart"),
    ("Rank the revenue rows.", "ranking", 2, "chart_ready", "chart"),
    ("Plot Apple's revenue trend.", "trend", 1, "chart_ready", "none"),
    ("Plot Apple's 2025 revenue.", "lookup", 0, "chart_ready", "none"),
])
def test_direct_chart_intent_controls_sql_mode_and_post_execution_chart(
    question, intent, rows, mode, decision
):
    charts = _chart_dependencies(intent)
    raw = _raw_result()
    raw.rows = raw.rows[:rows]
    raw.row_count = len(raw.rows)
    graph, _, _, generator, _ = _graph_with_fakes(
        "sql", execution_outcomes=[raw], chart_dependencies=charts,
    )

    result = graph.invoke({"question": question})

    charts["chart_intent_classifier"].classify.assert_called_once_with(question)
    assert result["sql_result_mode"] == mode
    assert generator.calls == [{
        "question": question, "schema": FINANCIAL_SCHEMA, "result_mode": mode,
    }]
    assert result["explicit_visualization"] == question.startswith("Plot")
    assert result["chart_data"] is raw
    assert result["sql_result"] is raw
    charts["chart_data_builder"].build.assert_called_once_with(raw, analysis_result=None)
    charts["chart_decision_policy"].decide.assert_called_once_with(
        result["chart_intent"], raw,
        explicit_visualization=result["explicit_visualization"],
    )
    assert result["chart_decision"] == decision
    if decision == "chart":
        spec = charts["chart_planner"].plan.return_value
        charts["chart_planner"].plan.assert_called_once_with(question=question, sql_result=raw)
        charts["chart_renderer"].render.assert_called_once_with(raw, spec)
        assert result["chart_spec"] is spec
        assert result["chart_artifact"] is charts["chart_renderer"].render.return_value
    else:
        charts["chart_planner"].plan.assert_not_called()
        charts["chart_renderer"].render.assert_not_called()
        assert "chart_spec" not in result
        assert "chart_artifact" not in result


def test_chart_ready_direct_repair_keeps_mode_and_continues_chart_pipeline():
    charts = _chart_dependencies("lookup")
    raw = _raw_result()
    graph, _, _, generator, executor = _graph_with_fakes(
        "sql", repaired_sqls=[FIRST_REPAIR],
        execution_outcomes=[SQLExecutionError("no such column: revenue"), raw],
        chart_dependencies=charts,
    )

    result = AgentService(graph).ask("Plot Apple's revenue.")

    assert generator.calls[0]["result_mode"] == "chart_ready"
    assert generator.repair_calls == [{
        "question": "Plot Apple's revenue.", "schema": FINANCIAL_SCHEMA,
        "previous_sql": INITIAL_SQL, "error_message": "no such column: revenue",
        "result_mode": "chart_ready",
    }]
    assert generator.analysis_repair_calls == []
    assert [call["sql"] for call in executor.calls] == [INITIAL_SQL, FIRST_REPAIR]
    assert result.generated_sql == FIRST_REPAIR
    assert result.sql_result is raw
    assert result.chart_spec is charts["chart_planner"].plan.return_value
    assert result.chart_artifact is charts["chart_renderer"].render.return_value


@pytest.mark.parametrize("rows", [1, 2])
def test_ranking_charts_ranked_rows_and_service_preserves_raw_rows(rows):
    dependencies = _analysis_dependencies(AnalysisOperation.RANKING)
    charts = _chart_dependencies()
    raw = SQLQueryResult(
        columns=["company", "revenue_musd"],
        rows=[
            {"company": "Microsoft", "revenue_musd": 281724},
            {"company": "Apple", "revenue_musd": 416161},
        ][:rows],
        row_count=rows,
    )
    graph, *_ = _graph_with_fakes(
        "sql", execution_outcomes=[raw], chart_dependencies=charts, **dependencies,
    )
    question = "Rank Apple and Microsoft by revenue."

    result = AgentService(graph).ask(question)

    assert result.sql_result is raw
    assert result.sql_result.rows[0]["company"] == "Microsoft"
    assert result.analysis_result.operation is AnalysisOperation.RANKING
    charts["chart_intent_classifier"].classify.assert_not_called()
    charts["chart_data_builder"].build.assert_called_once_with(
        raw, analysis_result=result.analysis_result,
    )
    decision_call = charts["chart_decision_policy"].decide.call_args
    assert decision_call.args[0] == ChartIntentResponse(intent="ranking")
    assert decision_call.kwargs == {"explicit_visualization": False}
    chart_data = decision_call.args[1]
    assert chart_data is not raw
    assert chart_data.rows == [dict(row) for row in result.analysis_result.ranked_rows]
    if rows == 2:
        assert [row["company"] for row in chart_data.rows] == ["Apple", "Microsoft"]
        charts["chart_planner"].plan.assert_called_once_with(
            question=question, sql_result=chart_data,
        )
        assert charts["chart_planner"].plan.call_args.kwargs["sql_result"] is chart_data
        charts["chart_renderer"].render.assert_called_once_with(chart_data, result.chart_spec)
        assert charts["chart_renderer"].render.call_args.args[0] is chart_data
        assert result.chart_artifact is charts["chart_renderer"].render.return_value
    else:
        charts["chart_planner"].plan.assert_not_called()
        charts["chart_renderer"].render.assert_not_called()
        assert result.chart_spec is result.chart_artifact is None


def test_chart_then_non_chart_invocations_do_not_share_fields():
    charts = _chart_dependencies()
    graph, *_ = _graph_with_fakes("sql", chart_dependencies=charts)

    first = graph.invoke({"question": "Plot Apple's revenue."})
    second = graph.invoke({"question": "What was Apple's revenue?"})

    assert first["chart_artifact"] is charts["chart_renderer"].render.return_value
    assert first["sql_result_mode"] == "chart_ready"
    assert second["sql_result_mode"] == "answer"
    assert second["explicit_visualization"] is False
    assert second["chart_decision"] == "none"
    assert "chart_spec" not in second
    assert "chart_artifact" not in second
    assert charts["chart_renderer"].render.call_count == 1


def test_ranking_chart_then_direct_invocations_keep_chart_and_analysis_state_isolated():
    dependencies = _analysis_dependencies(AnalysisOperation.RANKING)
    dependencies["classifier"].classify.side_effect = [
        AnalysisSQLTask(mode="analysis", operation=AnalysisOperation.RANKING),
        DirectSQLTask(mode="direct"),
    ]
    charts = _chart_dependencies()
    graph, *_ = _graph_with_fakes(
        "sql", execution_outcomes=[_raw_result(), _raw_result()],
        chart_dependencies=charts, **dependencies,
    )

    first = graph.invoke({"question": "Rank revenue."})
    second = graph.invoke({"question": "What was revenue?"})

    assert first["chart_intent"] == ChartIntentResponse(intent="ranking")
    assert "chart_artifact" in first
    assert second["chart_intent"] == ChartIntentResponse(intent="lookup")
    assert second["chart_decision"] == "none"
    assert "chart_artifact" not in second
    assert "chart_spec" not in second
    assert not any(key.startswith("analysis_") for key in second)
    charts["chart_intent_classifier"].classify.assert_called_once_with("What was revenue?")


@pytest.mark.parametrize("stage,error", [
    ("chart_intent_classifier", ChartDecisionError("intent failed")),
    ("chart_data_builder", ChartDataError("data failed")),
    ("chart_decision_policy", ChartDecisionError("decision failed")),
    ("chart_planner", ChartPlanningError("planning failed")),
    ("chart_renderer", ChartRenderingError("rendering failed")),
])
def test_chart_errors_propagate_without_fallback_or_retry(stage, error):
    charts = _chart_dependencies()
    methods = [
        ("chart_intent_classifier", "classify"), ("chart_data_builder", "build"),
        ("chart_decision_policy", "decide"), ("chart_planner", "plan"),
        ("chart_renderer", "render"),
    ]
    stage_index = [name for name, _ in methods].index(stage)
    getattr(charts[stage], methods[stage_index][1]).side_effect = error
    graph, _, _, generator, _ = _graph_with_fakes("sql", chart_dependencies=charts)

    with pytest.raises(type(error)) as exc:
        AgentService(graph).ask("Plot Apple's revenue.")

    assert exc.value is error
    assert getattr(charts[stage], methods[stage_index][1]).call_count == 1
    for name, _ in methods[stage_index + 1:]:
        assert charts[name].mock_calls == []
    assert generator.repair_calls == generator.analysis_repair_calls == []


@pytest.mark.parametrize("analysis", [False, True])
def test_high_sql_retry_budget_reaches_complete_chart_path(analysis):
    budget = 12
    charts = _chart_dependencies("comparison")
    dependencies = _analysis_dependencies(AnalysisOperation.RANKING) if analysis else {}
    graph, _, _, generator, executor = _graph_with_fakes(
        "sql", max_sql_retries=budget, repaired_sqls=[REPAIRED_RAW_SQL] * budget,
        execution_outcomes=[SQLExecutionError("retry")] * budget + [_raw_result()],
        chart_dependencies=charts, **dependencies,
    )

    result = graph.invoke({"question": "Compare company revenues."})

    assert result["sql_retry_count"] == budget
    assert len(executor.calls) == budget + 1
    assert result["chart_artifact"] is charts["chart_renderer"].render.return_value
    assert result["generated_sql"] == REPAIRED_RAW_SQL
    if analysis:
        assert len(generator.analysis_repair_calls) == budget
        assert generator.repair_calls == []
        charts["chart_intent_classifier"].classify.assert_not_called()
    else:
        assert len(generator.repair_calls) == budget
        assert all(call["result_mode"] == "chart_ready" for call in generator.repair_calls)
        assert generator.analysis_repair_calls == []


def test_chart_graph_topology_has_explicit_preparation_decision_plan_and_render_nodes():
    charts = _chart_dependencies("comparison")
    graph, *_ = _graph_with_fakes(
        "sql", execution_outcomes=[_raw_result()], chart_dependencies=charts,
    )

    updates = list(graph.stream({"question": "Compare revenue."}))

    assert [name for update in updates for name in update] == [
        "route_question", "classify_sql_task", "classify_chart_intent",
        "generate_sql", "execute_sql", "prepare_chart_data", "decide_chart",
        "plan_chart", "render_chart",
    ]
