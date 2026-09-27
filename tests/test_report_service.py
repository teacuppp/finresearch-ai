"""Report service orchestration tests using fake dependencies."""

import pytest

from app.report.evidence import ReportResearchResult
from app.report.executor import ReportResearchExecutionError
from app.report.models import ReportPlan, ReportRAGScope, ReportTask
from app.report.planner import ReportPlanningError
from app.report.service import ReportService


class FakePlanner:
    def __init__(
        self,
        plan: ReportPlan | None = None,
        error: Exception | None = None,
    ) -> None:
        self.planned = plan
        self.error = error
        self.calls: list[object] = []

    def plan(self, request: str) -> ReportPlan:
        self.calls.append(request)
        if self.error is not None:
            raise self.error
        assert self.planned is not None
        return self.planned


class FakeExecutor:
    def __init__(
        self,
        result: ReportResearchResult | None = None,
        error: Exception | None = None,
    ) -> None:
        self.result = result
        self.error = error
        self.calls: list[ReportPlan] = []

    def execute(self, plan: ReportPlan) -> ReportResearchResult:
        self.calls.append(plan)
        if self.error is not None:
            raise self.error
        assert self.result is not None
        return self.result


def _plan(routes: tuple[str, ...]) -> ReportPlan:
    tasks = tuple(
        ReportTask(
            task_id=f"task_{index}",
            question=f"Question {index}?",
            expected_route=route,
            rag_scope=ReportRAGScope(company="Apple") if route == "rag" else None,
        )
        for index, route in enumerate(routes, start=1)
    )
    return ReportPlan(title="Research plan", tasks=tasks)


@pytest.mark.parametrize("routes", [("rag", "sql"), ("rag",), ("sql",)])
def test_research_passes_plan_and_result_by_identity(routes: tuple[str, ...]) -> None:
    request = "Research Apple"
    plan = _plan(routes)
    tasks = plan.tasks
    result = ReportResearchResult(plan=plan, evidence=())
    planner = FakePlanner(plan=plan)
    executor = FakeExecutor(result=result)

    actual = ReportService(planner, executor).research(request)

    assert len(planner.calls) == 1
    assert planner.calls[0] is request
    assert len(executor.calls) == 1
    assert executor.calls[0] is plan
    assert executor.calls[0].tasks is tasks
    assert actual is result
    assert actual.plan is plan


@pytest.mark.parametrize(
    "error", [ReportPlanningError("planning failed"), RuntimeError("unexpected")]
)
def test_research_propagates_planner_error_without_execution(error: Exception) -> None:
    planner = FakePlanner(error=error)
    executor = FakeExecutor()

    with pytest.raises(type(error)) as caught:
        ReportService(planner, executor).research("Research Apple")

    assert caught.value is error
    assert len(planner.calls) == 1
    assert executor.calls == []


@pytest.mark.parametrize(
    "error",
    [ReportResearchExecutionError("execution failed"), RuntimeError("unexpected")],
)
def test_research_propagates_executor_error_without_retry(error: Exception) -> None:
    plan = _plan(("sql",))
    planner = FakePlanner(plan=plan)
    executor = FakeExecutor(error=error)

    with pytest.raises(type(error)) as caught:
        ReportService(planner, executor).research("Research Apple")

    assert caught.value is error
    assert len(planner.calls) == 1
    assert len(executor.calls) == 1
    assert executor.calls[0] is plan


@pytest.mark.parametrize("candidate", ["", 42])
def test_research_delegates_request_without_validation(candidate: object) -> None:
    error = ReportPlanningError("invalid request")
    planner = FakePlanner(error=error)
    executor = FakeExecutor()

    with pytest.raises(ReportPlanningError) as caught:
        ReportService(planner, executor).research(candidate)  # type: ignore[arg-type]

    assert caught.value is error
    assert len(planner.calls) == 1
    assert planner.calls[0] is candidate
    assert executor.calls == []
