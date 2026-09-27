import pytest
from pydantic import ValidationError

from app.report.models import ReportPlan, ReportRAGScope, ReportTask


def task(task_id: str = "revenue", question: str = "What was Apple revenue?", route: str = "sql") -> ReportTask:
    scope = ReportRAGScope(company="Apple") if route == "rag" else None
    return ReportTask(
        task_id=task_id, question=question, expected_route=route, rag_scope=scope
    )


@pytest.mark.parametrize(
    "values",
    [
        {"company": "Apple"},
        {"ticker": "AAPL"},
        {"company": "Apple", "ticker": "AAPL"},
        {"company": "Apple", "fiscal_year": 2025},
        {"ticker": "AAPL", "document_type": "10-K"},
    ],
)
def test_report_rag_scope_accepts_supported_fields(values: dict) -> None:
    scope = ReportRAGScope(**values)
    for field, value in values.items():
        assert getattr(scope, field) == value


def test_report_rag_scope_preserves_original_text() -> None:
    scope = ReportRAGScope(company="  Apple  ", ticker=" AAPL ", document_type="10-K")
    assert scope.company == "  Apple  "
    assert scope.ticker == " AAPL "
    assert scope.document_type == "10-K"


@pytest.mark.parametrize("filing_form", ["10-K", "10-Q", "8-K", "20-F", "40-F"])
def test_report_rag_scope_accepts_concrete_filing_forms(filing_form: str) -> None:
    assert ReportRAGScope(company="Apple", document_type=filing_form).document_type == filing_form


def test_report_rag_scope_accepts_no_document_type() -> None:
    assert ReportRAGScope(company="Apple").document_type is None


@pytest.mark.parametrize(
    "document_type",
    ["filing", "annual filing", "annual report", "forward-looking statements", "memo"],
)
def test_report_rag_scope_rejects_generic_or_arbitrary_document_type(
    document_type: str,
) -> None:
    with pytest.raises(ValidationError):
        ReportRAGScope(company="Apple", document_type=document_type)


def test_report_rag_scope_is_frozen() -> None:
    with pytest.raises(ValidationError):
        ReportRAGScope(company="Apple").company = "Microsoft"


def test_report_rag_scope_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        ReportRAGScope(company="Apple", database_table="financials")


@pytest.mark.parametrize("field", ["company", "ticker", "document_type"])
@pytest.mark.parametrize("value", ["", " \t\n"])
def test_report_rag_scope_rejects_blank_strings(field: str, value: str) -> None:
    values = {"company": "Apple", field: value}
    with pytest.raises(ValidationError):
        ReportRAGScope(**values)


def test_report_rag_scope_requires_company_or_ticker() -> None:
    with pytest.raises(ValidationError):
        ReportRAGScope(fiscal_year=2025, document_type="10-K")


@pytest.mark.parametrize("year", [True, False, 1899, 2101])
def test_report_rag_scope_rejects_invalid_year(year: object) -> None:
    with pytest.raises(ValidationError):
        ReportRAGScope(company="Apple", fiscal_year=year)


@pytest.mark.parametrize("route", ["rag", "sql"])
def test_valid_report_task_routes(route: str) -> None:
    result = task(route=route)
    assert result.expected_route == route
    assert result.question == "What was Apple revenue?"
    assert (result.rag_scope is not None) == (route == "rag")


def test_report_task_rejects_rag_without_scope() -> None:
    with pytest.raises(ValidationError, match="rag_scope"):
        ReportTask(task_id="risks", question="What were Apple's risks?", expected_route="rag")


def test_report_task_rejects_sql_with_scope() -> None:
    with pytest.raises(ValidationError, match="rag_scope"):
        ReportTask(
            task_id="revenue",
            question="What was Apple's revenue?",
            expected_route="sql",
            rag_scope=ReportRAGScope(company="Apple"),
        )


def test_report_task_preserves_question_whitespace() -> None:
    assert task(question="  What was Apple revenue?  ").question == "  What was Apple revenue?  "


def test_report_task_is_frozen() -> None:
    with pytest.raises(ValidationError):
        task().question = "Changed"


def test_report_task_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        ReportTask(task_id="revenue", question="Revenue?", expected_route="sql", answer="42")


@pytest.mark.parametrize(
    "task_id",
    ["", "  ", "Revenue", "1revenue", "revenue-growth", "a" * 65],
)
def test_report_task_rejects_invalid_id(task_id: str) -> None:
    with pytest.raises(ValidationError):
        task(task_id=task_id)


@pytest.mark.parametrize("question", ["", " \t\n", "a" * 1001])
def test_report_task_rejects_invalid_question(question: str) -> None:
    with pytest.raises(ValidationError):
        task(question=question)


def test_report_task_rejects_invalid_route() -> None:
    with pytest.raises(ValidationError):
        task(route="chart")


@pytest.mark.parametrize("routes", [("rag", "sql"), ("rag",), ("sql",)])
def test_report_plan_accepts_route_combinations(routes: tuple[str, ...]) -> None:
    tasks = tuple(task(f"task_{index}", f"Question {index}?", route) for index, route in enumerate(routes))
    plan = ReportPlan(title="Apple research", tasks=tasks)
    assert plan.tasks == tasks
    assert isinstance(plan.tasks, tuple)


def test_report_plan_is_frozen() -> None:
    plan = ReportPlan(title="Apple research", tasks=(task(),))
    with pytest.raises(ValidationError):
        plan.title = "Changed"


def test_report_plan_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        ReportPlan(title="Apple research", tasks=(task(),), status="ready")


@pytest.mark.parametrize("title", ["", " \t\n", "a" * 201])
def test_report_plan_rejects_invalid_title(title: str) -> None:
    with pytest.raises(ValidationError):
        ReportPlan(title=title, tasks=(task(),))


@pytest.mark.parametrize("count", [0, 13])
def test_report_plan_rejects_invalid_task_count(count: int) -> None:
    tasks = tuple(task(f"task_{index}", f"Question {index}?") for index in range(count))
    with pytest.raises(ValidationError):
        ReportPlan(title="Apple research", tasks=tasks)


def test_report_plan_rejects_duplicate_task_ids() -> None:
    with pytest.raises(ValidationError, match="task_id"):
        ReportPlan(title="Apple research", tasks=(task("same", "One?"), task("same", "Two?")))


@pytest.mark.parametrize("second", ["WHAT WAS APPLE REVENUE?", "  what was apple revenue?  "])
def test_report_plan_rejects_duplicate_questions(second: str) -> None:
    with pytest.raises(ValidationError, match="questions must be unique"):
        ReportPlan(title="Apple research", tasks=(task(), task("other", second)))
