import pytest

from app.agent.sql_executor import SQLQueryResult
from app.analysis.chart_renderer import ChartArtifact, ChartSpec, ChartType
from app.analysis.financial_analyzer import AnalysisOperation, AnalysisResult
from app.report.evidence import ReportResearchResult
from app.report.executor import ReportResearchExecutionError, ReportResearchExecutor
from app.report.models import ReportPlan, ReportRAGScope, ReportTask
from app.services.agent_service import AgentResult


class FakeAgent:
    def __init__(self, responses: list[AgentResult | Exception]) -> None:
        self.responses = iter(responses)
        self.calls: list[dict] = []

    def ask(self, **kwargs) -> AgentResult:
        self.calls.append(kwargs)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def rag_task(scope: ReportRAGScope, task_id: str = "risks") -> ReportTask:
    return ReportTask(
        task_id=task_id,
        question=f"What risks did {scope.company or scope.ticker} disclose?",
        expected_route="rag",
        rag_scope=scope,
    )


def sql_task(
    task_id: str = "revenue", question: str = "Retrieve Apple revenue."
) -> ReportTask:
    return ReportTask(
        task_id=task_id,
        question=question,
        expected_route="sql",
    )


def plan(*tasks: ReportTask) -> ReportPlan:
    return ReportPlan(title="Apple research", tasks=tasks)


def sql_result() -> SQLQueryResult:
    return SQLQueryResult(
        columns=["revenue_musd"],
        rows=[{"revenue_musd": 416161}],
        row_count=1,
    )


def test_default_top_k_is_passed_to_rag_call() -> None:
    task = rag_task(ReportRAGScope(company="Apple"))
    result = AgentResult(route="rag", answer="Risks", sources=[])
    agent = FakeAgent([result])
    executor = ReportResearchExecutor(agent)
    original_plan = plan(task)

    research = executor.execute(original_plan)

    assert executor.top_k == 5
    assert agent.calls == [{
        "question": task.question,
        "top_k": 5,
        "where": {"company": {"$eq": "Apple"}},
        "company": "Apple",
        "ticker": None,
    }]
    assert research.evidence[0].task is original_plan.tasks[0]
    assert research.evidence[0].result is result


@pytest.mark.parametrize("top_k", [0, 21, True, False, 2.5, "5", None])
def test_invalid_top_k_is_rejected(top_k: object) -> None:
    with pytest.raises(ValueError, match="top_k"):
        ReportResearchExecutor(FakeAgent([]), top_k=top_k)


def test_custom_top_k_is_passed_only_to_rag_call() -> None:
    rag = rag_task(ReportRAGScope(company="Apple"))
    sql = sql_task()
    agent = FakeAgent([AgentResult(route="rag"), AgentResult(route="sql")])

    ReportResearchExecutor(agent, top_k=12).execute(plan(rag, sql))

    assert agent.calls[0]["top_k"] == 12
    assert agent.calls[1] == {"question": sql.question}


@pytest.mark.parametrize("top_k", [1, 20])
def test_top_k_boundaries_are_accepted(top_k: int) -> None:
    assert ReportResearchExecutor(FakeAgent([]), top_k=top_k).top_k == top_k


@pytest.mark.parametrize(
    "scope,expected_where",
    [
        (ReportRAGScope(company="Apple"), {"company": {"$eq": "Apple"}}),
        (ReportRAGScope(ticker="AAPL"), {"ticker": {"$eq": "AAPL"}}),
        (
            ReportRAGScope(company="Apple", fiscal_year=2025),
            {"$and": [
                {"company": {"$eq": "Apple"}},
                {"fiscal_year": {"$eq": 2025}},
            ]},
        ),
        (
            ReportRAGScope(
                company="Apple", ticker="AAPL", fiscal_year=2025,
                document_type="10-K",
            ),
            {"$and": [
                {"company": {"$eq": "Apple"}},
                {"ticker": {"$eq": "AAPL"}},
                {"fiscal_year": {"$eq": 2025}},
                {"document_type": {"$eq": "10-K"}},
            ]},
        ),
    ],
)
def test_rag_scope_propagates_exact_filter_and_structured_identity(
    scope: ReportRAGScope, expected_where: dict
) -> None:
    task = rag_task(scope)
    agent = FakeAgent([AgentResult(route="rag")])

    ReportResearchExecutor(agent).execute(plan(task))

    assert agent.calls == [{
        "question": task.question,
        "top_k": 5,
        "where": expected_where,
        "company": scope.company,
        "ticker": scope.ticker,
    }]


def test_sql_delegates_question_only_and_retains_direct_result() -> None:
    task = sql_task()
    data = sql_result()
    result = AgentResult(route="sql", generated_sql="SELECT revenue_musd", sql_result=data)
    agent = FakeAgent([result])
    original_plan = plan(task)

    research = ReportResearchExecutor(agent).execute(original_plan)

    assert agent.calls == [{"question": task.question}]
    assert research.evidence[0].task is original_plan.tasks[0]
    assert research.evidence[0].result is result
    assert research.evidence[0].result.sql_result is data


def test_sql_analysis_and_chart_objects_are_retained_unchanged() -> None:
    task = sql_task()
    data = sql_result()
    analysis = AnalysisResult(AnalysisOperation.PERCENTAGE_CHANGE, value=6.5)
    spec = ChartSpec(ChartType.BAR, "company", "revenue_musd", "Revenue")
    artifact = ChartArtifact(media_type="image/png", content=b"chart bytes")
    result = AgentResult(
        route="sql", generated_sql="SELECT revenue_musd", sql_result=data,
        analysis_result=analysis, chart_spec=spec, chart_artifact=artifact,
    )

    research = ReportResearchExecutor(FakeAgent([result])).execute(plan(task))

    assert research.evidence[0].result is result
    assert result.analysis_result is analysis
    assert result.chart_spec is spec
    assert result.chart_artifact is artifact
    assert result.chart_artifact.content == b"chart bytes"


def test_mixed_tasks_execute_sequentially_and_preserve_evidence_order() -> None:
    first = rag_task(ReportRAGScope(company="Apple"), "first")
    second = sql_task("second")
    third = rag_task(ReportRAGScope(ticker="AAPL"), "third")
    tasks = (first, second, third)
    results = (
        AgentResult(route="rag"),
        AgentResult(route="sql"),
        AgentResult(route="rag"),
    )
    original_plan = plan(*tasks)
    agent = FakeAgent(list(results))

    research = ReportResearchExecutor(agent).execute(original_plan)

    assert isinstance(research, ReportResearchResult)
    assert research.plan is original_plan
    assert [call["question"] for call in agent.calls] == [task.question for task in tasks]
    assert isinstance(research.evidence, tuple)
    assert len(research.evidence) == 3
    assert all(
        item.task is task for item, task in zip(research.evidence, original_plan.tasks)
    )
    assert all(item.result is result for item, result in zip(research.evidence, results))


@pytest.mark.parametrize("expected,actual", [("rag", "sql"), ("sql", "rag")])
def test_route_mismatch_fails_once_with_task_context(
    expected: str, actual: str
) -> None:
    task = (
        rag_task(ReportRAGScope(company="Apple"))
        if expected == "rag" else sql_task()
    )
    agent = FakeAgent([AgentResult(route=actual), AgentResult(route=expected)])

    with pytest.raises(ReportResearchExecutionError) as exc_info:
        ReportResearchExecutor(agent).execute(plan(task))

    message = str(exc_info.value)
    assert task.task_id in message
    assert expected in message
    assert actual in message
    assert len(agent.calls) == 1


def test_agent_failure_preserves_cause_and_stops_later_tasks() -> None:
    first = sql_task("first")
    second = rag_task(ReportRAGScope(company="Apple"), "second")
    third = sql_task("third", "Retrieve Apple net income.")
    original = RuntimeError("agent unavailable")
    agent = FakeAgent([AgentResult(route="sql"), original, AgentResult(route="sql")])

    with pytest.raises(ReportResearchExecutionError) as exc_info:
        ReportResearchExecutor(agent).execute(plan(first, second, third))

    assert exc_info.value.__cause__ is original
    assert "second" in str(exc_info.value)
    assert [call["question"] for call in agent.calls] == [first.question, second.question]


def test_non_report_plan_is_rejected_before_agent_call() -> None:
    agent = FakeAgent([])
    with pytest.raises(ReportResearchExecutionError, match="ReportPlan"):
        ReportResearchExecutor(agent).execute({"title": "Not a plan"})
    assert agent.calls == []


def test_forged_report_plan_is_rejected_before_agent_call() -> None:
    valid = plan(sql_task())
    malformed_task = valid.tasks[0].model_copy(update={"task_id": "Bad-ID"})
    malformed = valid.model_copy(update={"tasks": (malformed_task,)})
    agent = FakeAgent([])

    with pytest.raises(ReportResearchExecutionError, match="Invalid report plan") as exc_info:
        ReportResearchExecutor(agent).execute(malformed)

    assert exc_info.value.__cause__ is not None
    assert agent.calls == []


def test_forged_plan_with_list_tasks_is_rejected_before_agent_call() -> None:
    valid = plan(sql_task())
    malformed = valid.model_copy(update={"tasks": list(valid.tasks)})
    agent = FakeAgent([])

    with pytest.raises(ReportResearchExecutionError, match="Invalid report plan"):
        ReportResearchExecutor(agent).execute(malformed)

    assert agent.calls == []
