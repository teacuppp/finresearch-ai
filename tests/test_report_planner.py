from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.report.models import ReportPlan, ReportRAGScope, ReportTask
from app.report.planner import REPORT_PLANNER_SYSTEM_PROMPT, ReportPlanner, ReportPlanningError


def response(parsed: object, content: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed, content=content))])


class FakeCompletions:
    def __init__(self, returned: object = None, error: Exception | None = None) -> None:
        self.returned = returned
        self.error = error
        self.calls: list[dict] = []

    def parse(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.returned


class FakeClient:
    def __init__(self, returned: object = None, error: Exception | None = None) -> None:
        self.completions = FakeCompletions(returned, error)
        self.chat = SimpleNamespace(completions=self.completions)


def plan(*routes: str) -> ReportPlan:
    return ReportPlan(
        title="Apple research",
        tasks=tuple(
            ReportTask(
                task_id=f"task_{index}",
                question=f"Question {index}?",
                expected_route=route,
                rag_scope=ReportRAGScope(company="Apple") if route == "rag" else None,
            )
            for index, route in enumerate(routes)
        ),
    )


def scoped_plan(
    *,
    title: str = "Apple research",
    question: str = "What risks did Apple describe?",
    company: str | None = "Apple",
    ticker: str | None = None,
    fiscal_year: int | None = None,
    document_type: str | None = None,
) -> ReportPlan:
    return ReportPlan(
        title=title,
        tasks=(
            ReportTask(
                task_id="evidence",
                question=question,
                expected_route="rag",
                rag_scope=ReportRAGScope(
                    company=company,
                    ticker=ticker,
                    fiscal_year=fiscal_year,
                    document_type=document_type,
                ),
            ),
        ),
    )


def sql_plan(*, title: str, question: str) -> ReportPlan:
    return ReportPlan(
        title=title,
        tasks=(ReportTask(task_id="values", question=question, expected_route="sql"),),
    )


def assert_accepted(request: str, parsed: ReportPlan) -> None:
    client = FakeClient(response(parsed))
    assert ReportPlanner(client=client).plan(request) is parsed
    assert len(client.completions.calls) == 1


def assert_rejected(request: str, parsed: ReportPlan) -> None:
    client = FakeClient(response(parsed, content="A fallback answer"))
    with pytest.raises(ReportPlanningError) as exc_info:
        ReportPlanner(client=client).plan(request)
    assert isinstance(exc_info.value.__cause__, ValueError)
    assert len(client.completions.calls) == 1
    assert client.completions.returned.choices[0].message.parsed is parsed


@pytest.mark.parametrize("routes", [("rag", "sql"), ("rag",), ("sql",)])
def test_valid_parsed_plan_returned_unchanged(routes: tuple[str, ...]) -> None:
    parsed = plan(*routes)
    client = FakeClient(response(parsed))
    assert ReportPlanner(client=client).plan("Research Apple.") is parsed


@pytest.mark.parametrize("request_value", ["", "  \t\n", 42, None, "a" * 2001])
def test_invalid_request_rejected_before_client_call(request_value: object) -> None:
    client = FakeClient(response(plan("rag")))
    with pytest.raises(ReportPlanningError):
        ReportPlanner(client=client).plan(request_value)
    assert client.completions.calls == []


def test_model_request_contract_and_exact_user_message() -> None:
    client = FakeClient(response(plan("rag")))
    request = "  Compare Apple's 2025 revenue and risks.\n"
    ReportPlanner(client=client).plan(request)
    assert len(client.completions.calls) == 1
    call = client.completions.calls[0]
    assert call == {
        "model": "qwen3:4b-instruct",
        "messages": [
            {"role": "system", "content": REPORT_PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": request},
        ],
        "response_format": ReportPlan,
        "temperature": 0,
    }
    assert "reasoning_effort" not in call


def test_custom_model_is_passed_to_client() -> None:
    client = FakeClient(response(plan("sql")))
    ReportPlanner(model="custom", client=client).plan("Revenue?")
    assert client.completions.calls[0]["model"] == "custom"


@pytest.mark.parametrize("returned", [response(None), response(None, "The answer is 42."), response({"title": "wrong"}), SimpleNamespace(choices=[])])
def test_missing_or_wrong_parsed_output_fails_without_fallback(returned: object) -> None:
    client = FakeClient(returned)
    with pytest.raises(ReportPlanningError) as exc_info:
        ReportPlanner(client=client).plan("Research Apple.")
    assert exc_info.value.__cause__ is not None


def test_client_exception_is_translated_with_cause() -> None:
    original = RuntimeError("model unavailable")
    client = FakeClient(error=original)
    with pytest.raises(ReportPlanningError) as exc_info:
        ReportPlanner(client=client).plan("Research Apple.")
    assert exc_info.value.__cause__ is original


def test_malformed_pydantic_output_fails_without_repair() -> None:
    malformed = plan("rag").model_copy(update={"title": "  "})
    client = FakeClient(response(malformed))
    with pytest.raises(ReportPlanningError) as exc_info:
        ReportPlanner(client=client).plan("Research Apple.")
    assert exc_info.value.__cause__ is not None


def test_malformed_nested_task_fails_without_repair() -> None:
    malformed_task = ReportTask.model_construct(
        task_id="Bad-ID",
        question="Question?",
        expected_route="rag",
        rag_scope=ReportRAGScope(company="Apple"),
    )
    malformed = plan("rag").model_copy(update={"tasks": (malformed_task,)})
    client = FakeClient(response(malformed))
    with pytest.raises(ReportPlanningError) as exc_info:
        ReportPlanner(client=client).plan("Research Apple.")
    assert exc_info.value.__cause__ is not None


def test_requested_2025_year_is_accepted_in_title_question_and_scope() -> None:
    parsed = scoped_plan(
        title="Apple 2025 risks",
        question="What risks did Apple describe in 2025?",
        fiscal_year=2025,
    )
    assert_accepted("Summarize Apple 2025 risks.", parsed)


def test_both_requested_years_are_accepted() -> None:
    parsed = sql_plan(
        title="Apple revenue 2024 to 2025",
        question="Retrieve Apple revenue values for 2024 and 2025.",
    )
    assert_accepted("Show Apple revenue from 2024 to 2025.", parsed)


@pytest.mark.parametrize(
    "parsed",
    [
        sql_plan(title="Apple 2025 revenue", question="Retrieve Apple 2026 revenue."),
        sql_plan(title="Apple 2026 revenue", question="Retrieve Apple 2025 revenue."),
        scoped_plan(question="What risks did Apple describe in 2025?", fiscal_year=2026),
    ],
)
def test_unrequested_year_in_question_title_or_rag_scope_is_rejected(
    parsed: ReportPlan,
) -> None:
    assert_rejected("Research Apple in 2025.", parsed)


def test_rag_company_from_request_is_accepted_without_normalizing_scope() -> None:
    parsed = scoped_plan(company="  Apple  ")
    assert_accepted("Research APPLE risks.", parsed)
    assert parsed.tasks[0].rag_scope.company == "  Apple  "


@pytest.mark.parametrize(
    "request_text,company",
    [("Research Apple risks.", "Microsoft"), ("Research Pineapple risks.", "Apple")],
)
def test_unrequested_rag_company_is_rejected(
    request_text: str, company: str
) -> None:
    assert_rejected(
        request_text,
        scoped_plan(company=company, question=f"What risks did {company} describe?"),
    )


def test_explicit_ticker_is_accepted() -> None:
    parsed = scoped_plan(company=None, ticker="AAPL", question="What risks did AAPL describe?")
    assert_accepted("Research AAPL risks.", parsed)


def test_inferred_ticker_is_removed_with_one_model_call() -> None:
    parsed = scoped_plan(
        title="Apple 2025 risks",
        question="What risks did Apple describe in 2025?",
        company="Apple",
        ticker="AAPL",
        fiscal_year=2025,
    )
    client = FakeClient(response(parsed, content="A fallback answer"))
    result = ReportPlanner(client=client).plan("Research Apple 2025 risks.")

    assert result is not parsed
    assert result.title == parsed.title
    assert result.tasks[0].question == parsed.tasks[0].question
    assert result.tasks[0].expected_route == "rag"
    assert result.tasks[0].rag_scope.company == "Apple"
    assert result.tasks[0].rag_scope.ticker is None
    assert result.tasks[0].rag_scope.fiscal_year == 2025
    assert parsed.tasks[0].rag_scope.ticker == "AAPL"
    assert ReportPlan.model_validate(result) == result
    with pytest.raises(ValidationError):
        result.tasks[0].rag_scope.ticker = "AAPL"
    assert len(client.completions.calls) == 1


@pytest.mark.parametrize("request_text", ["Research Apple risks.", "Research XAAPL risks."])
def test_unrequested_ticker_is_rejected(request_text: str) -> None:
    parsed = scoped_plan(company=None, ticker="AAPL", question="What risks did Apple describe?")
    assert_rejected(request_text, parsed)


@pytest.mark.parametrize("filing_form", ["10-K", "10-Q", "8-K", "20-F", "40-F"])
def test_explicit_concrete_filing_form_is_accepted(filing_form: str) -> None:
    parsed = scoped_plan(document_type=filing_form)
    assert_accepted(f"Research Apple's {filing_form} risks.", parsed)


def test_generic_filing_request_accepts_no_document_type_scope() -> None:
    parsed = scoped_plan(
        title="Apple 2025 risks",
        question="What risks did Apple describe in its 2025 filing?",
        fiscal_year=2025,
        document_type=None,
    )
    assert_accepted("Research Apple's 2025 filing risks.", parsed)


def test_inferred_document_form_is_removed() -> None:
    parsed = scoped_plan(
        title="Apple 2025 risks",
        question="What risks did Apple describe in its 2025 filing?",
        fiscal_year=2025,
        document_type="10-K",
    )
    client = FakeClient(response(parsed, content="A fallback answer"))
    result = ReportPlanner(client=client).plan("Research Apple's 2025 filing risks.")

    assert result is not parsed
    assert result.tasks[0].rag_scope.document_type is None
    assert result.tasks[0].rag_scope.company == "Apple"
    assert result.tasks[0].rag_scope.fiscal_year == 2025
    assert parsed.tasks[0].rag_scope.document_type == "10-K"
    assert ReportPlan.model_validate(result) == result
    assert len(client.completions.calls) == 1


@pytest.mark.parametrize(
    "document_type",
    ["filing", "annual filing", "report", "annual document", "forward-looking statements"],
)
def test_document_type_without_explicit_matching_form_is_rejected(
    document_type: str,
) -> None:
    valid = scoped_plan()
    forged_scope = valid.tasks[0].rag_scope.model_copy(
        update={"document_type": document_type}
    )
    forged_task = valid.tasks[0].model_copy(update={"rag_scope": forged_scope})
    forged_plan = valid.model_copy(update={"tasks": (forged_task,)})
    client = FakeClient(response(forged_plan, content="A fallback answer"))
    with pytest.raises(ReportPlanningError) as exc_info:
        ReportPlanner(client=client).plan("Research Apple's annual filing risks.")
    assert isinstance(exc_info.value.__cause__, ValidationError)
    assert len(client.completions.calls) == 1


def test_requested_sql_metrics_are_accepted() -> None:
    parsed = sql_plan(
        title="Apple 2025 financial values",
        question="Retrieve Apple 2025 revenue and net income values.",
    )
    assert_accepted("Research Apple 2025 revenue and net income.", parsed)


@pytest.mark.parametrize("invented_metric", ["gross margin", "operating income"])
def test_unrequested_sql_metric_is_rejected(invented_metric: str) -> None:
    parsed = sql_plan(
        title="Apple 2025 financial values",
        question=f"Retrieve Apple 2025 revenue, net income, and {invented_metric} values.",
    )
    assert_rejected("Research Apple 2025 revenue and net income.", parsed)


def test_sql_metric_cannot_be_invented_when_request_has_no_recognized_metric() -> None:
    parsed = sql_plan(title="Apple 2025 values", question="Retrieve Apple 2025 revenue.")
    assert_rejected("Research Apple in 2025.", parsed)


def test_metric_restriction_does_not_apply_to_rag_narrative_question() -> None:
    parsed = scoped_plan(question="What gross margin drivers did Apple discuss?")
    assert_accepted("Research Apple's financial drivers.", parsed)


@pytest.mark.parametrize(
    "request_text,parsed",
    [
        (
            "Research projected Apple revenue for 2026.",
            scoped_plan(
                title="Apple 2026 forecast",
                question="What revenue did Apple project for 2026?",
            ),
        ),
        (
            "Research Apple's revenue forecast for 2026.",
            scoped_plan(
                title="Apple 2026 projected revenue",
                question="What revenue did Apple forecast for 2026?",
            ),
        ),
    ],
)
def test_explicit_forecast_semantics_are_accepted(
    request_text: str, parsed: ReportPlan
) -> None:
    assert_accepted(request_text, parsed)


@pytest.mark.parametrize(
    "parsed",
    [
        sql_plan(title="Apple projected revenue 2025", question="Retrieve Apple 2025 revenue."),
        sql_plan(title="Apple revenue 2025", question="Retrieve forecasted Apple 2025 revenue."),
    ],
)
def test_historical_request_rejects_invented_forecast_semantics(
    parsed: ReportPlan,
) -> None:
    assert_rejected("Retrieve Apple 2025 revenue.", parsed)


def test_prompt_contract() -> None:
    prompt = " ".join(REPORT_PLANNER_SYSTEM_PROMPT.lower().split())
    for phrase in (
        "evidence-gathering",
        "independently answerable",
        "do not answer",
        "do not generate sql",
        "calculate financial values",
        "report prose",
        "expected_route is an expectation",
        "not a command that bypasses",
        "both rag and sql tasks",
        "never create one task per entity × metric cell",
        "prefer one comparative sql task",
        "same comparison question",
        "ordinary direct metric lookups and comparisons",
        "combine entities and metrics when they share the same context",
        "absolute_change and percentage_change, use exactly one entity",
        "two fiscal years, and exactly one metric per task",
        "multi-company change requests, create one task per entity",
        "for difference, use exactly two entities, exactly one fiscal year",
        "for ranking, use at least two named entities (or all entities), exactly one fiscal year",
        "at most one supported deterministic analysis operation",
        "ranking with an unrelated metric comparison, split",
        "one company/ticker",
        "create one rag task per company",
        "every rag task must contain rag_scope",
        "never introduce forecast, forecasted, projected, estimated, or expected",
        "preserve whether the request is historical, current, or forecast-oriented",
        "do not alter requested fiscal years",
        "do not invent entities, years, metrics",
        "preserve metric names and company names",
        "every company or ticker named in any generated task must come from the original request",
        "never infer comparison peers",
        "never infer a ticker from a company name",
        "retrieve apple and microsoft 2025 operating income values",
        "one direct sql evidence task",
        "prefer one combined rag evidence task",
        "source document fiscal year, not the target forecast year",
        "if no source filing year is specified, leave rag_scope.fiscal_year=none",
        "generic \"filing\", \"annual filing\"",
        "explicit deterministic operation precedence",
        "preserve that operation in its evidence task",
        "ordinary retrieve-underlying-values rule applies only",
        "difference request: \"what is the difference between apple and microsoft 2025 revenue?\"",
        "ranking request: \"rank apple and microsoft by 2025 revenue.\"",
        "do not add a redundant direct sql raw-value task",
        "same entity, year, and metric already consumed",
        "ranking task already provides raw revenue evidence",
        "rag_scope.document_type is either 10-k, 10-q, 8-k, 20-f, or 40-f",
        "document_type=none. do not copy \"filing\" into document_type",
        "use exactly the structured metrics explicitly requested by the user",
        "never add another financial metric for completeness",
        "merely because it is available in the database",
        "because it appears in an example elsewhere in this system prompt",
        "do not expand revenue + net income into revenue + net income + gross margin",
        "retrieve apple's fy2025 revenue and net income values",
        "retrieve apple's fy2025 revenue, net income, and gross margin values",
    ):
        assert phrase in prompt
