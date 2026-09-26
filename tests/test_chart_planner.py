import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import app.analysis.chart_planner as chart_planner
from app.agent.sql_executor import SQLQueryResult
from app.analysis.chart_planner import (
    CHART_SYSTEM_PROMPT,
    ChartPlanner,
    ChartPlanningError,
    ChartPlanResponse,
)
from app.analysis.chart_renderer import ChartSpec, ChartType


def sql_result(
    *, with_company: bool = True, with_year: bool = False
) -> SQLQueryResult:
    rows = [
        {"ticker": "MSFT", "revenue_musd": 281724},
        {"ticker": "AAPL", "revenue_musd": 416161},
    ]
    columns = ["ticker", "revenue_musd"]
    if with_company:
        columns.insert(0, "company")
        rows[0]["company"] = "Microsoft"
        rows[1]["company"] = "Apple"
    if with_year:
        columns.insert(1, "fiscal_year")
        rows[0]["fiscal_year"] = 2024
        rows[1]["fiscal_year"] = 2025
    return SQLQueryResult(columns=columns, rows=rows, row_count=len(rows))


def parsed_chart(**changes: object) -> ChartPlanResponse:
    fields = {
        "chart_type": ChartType.BAR,
        "x_column": "company",
        "y_column": "revenue_musd",
        "title": "Revenue by Company",
        "x_label": None,
        "y_label": None,
    }
    fields.update(changes)
    return ChartPlanResponse.model_validate(fields)


def response(parsed: object, content: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed, content=content))]
    )


class FakeCompletions:
    def __init__(self, returned: object = None, error: Exception | None = None):
        self.returned = returned
        self.error = error
        self.calls: list[dict] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.returned


class FakeClient:
    def __init__(self, returned: object = None, error: Exception | None = None):
        self.completions = FakeCompletions(returned, error)
        self.chat = SimpleNamespace(completions=self.completions)


def planner_with_chart(**changes: object) -> tuple[ChartPlanner, FakeClient]:
    client = FakeClient(response(parsed_chart(**changes)))
    return ChartPlanner(client=client), client


def test_company_comparison_selects_bar_and_company_axis() -> None:
    planner, _ = planner_with_chart()

    chart = planner.plan("Compare Apple and Microsoft revenue.", sql_result())

    assert type(chart) is ChartSpec
    assert chart.chart_type is ChartType.BAR
    assert chart.x_column == "company"
    assert chart.y_column == "revenue_musd"


def test_ticker_comparison_selects_bar_when_company_is_unavailable() -> None:
    planner, _ = planner_with_chart(x_column="ticker")

    chart = planner.plan("Compare AAPL and MSFT revenue.", sql_result(with_company=False))

    assert chart.chart_type is ChartType.BAR
    assert chart.x_column == "ticker"


def test_fiscal_year_trend_selects_line() -> None:
    planner, _ = planner_with_chart(
        chart_type=ChartType.LINE,
        x_column="fiscal_year",
        title="Apple Revenue by Fiscal Year",
    )

    chart = planner.plan("Show Apple's revenue trend from 2024 to 2025.", sql_result(with_year=True))

    assert chart.chart_type is ChartType.LINE
    assert chart.x_column == "fiscal_year"


def test_ranking_style_question_selects_bar() -> None:
    planner, _ = planner_with_chart()

    chart = planner.plan("Rank companies by 2025 revenue.", sql_result())

    assert chart.chart_type is ChartType.BAR


def test_response_parses_existing_chart_type_and_returns_existing_chart_spec() -> None:
    parsed = ChartPlanResponse.model_validate(
        {
            "chart_type": "line",
            "x_column": "fiscal_year",
            "y_column": "revenue_musd",
            "title": "Revenue Trend",
            "x_label": "Fiscal year",
            "y_label": "Revenue (USD millions)",
        }
    )
    client = FakeClient(response(parsed))

    chart = ChartPlanner(client=client).plan("Show revenue trend.", sql_result(with_year=True))

    assert parsed.chart_type is ChartType.LINE
    assert isinstance(chart, ChartSpec)
    assert chart.chart_type is ChartType.LINE
    assert chart.x_label == "Fiscal year"
    assert chart.y_label == "Revenue (USD millions)"


def test_question_and_original_numbered_rows_are_sent_unchanged() -> None:
    planner, client = planner_with_chart()
    question = "  Rank the companies by revenue.  "
    data = sql_result()

    planner.plan(question, data)

    user_prompt = client.completions.calls[0]["messages"][1]["content"]
    assert f"Original question:\n{question}\n" in user_prompt
    serialized = user_prompt.split("SQL result (original row order):\n", 1)[1]
    assert json.loads(serialized) == {
        "columns": data.columns,
        "row_count": 2,
        "numbered_rows": [
            {"row_index": 0, "values": data.rows[0]},
            {"row_index": 1, "values": data.rows[1]},
        ],
    }
    assert data.rows[0]["ticker"] == "MSFT"
    assert data.rows[1]["ticker"] == "AAPL"


def test_model_call_disables_thinking_and_uses_structured_zero_temperature_output() -> None:
    planner, client = planner_with_chart()

    planner.plan("Compare revenue.", sql_result())

    call = client.completions.calls[0]
    assert call["model"] == "qwen3:4b"
    assert call["temperature"] == 0
    assert call["reasoning_effort"] == "none"
    assert call["response_format"] is ChartPlanResponse
    assert call["messages"][0] == {"role": "system", "content": CHART_SYSTEM_PROMPT}


def test_configured_model_is_used() -> None:
    client = FakeClient(response(parsed_chart()))

    ChartPlanner(model="custom", client=client).plan("Compare revenue.", sql_result())

    assert client.completions.calls[0]["model"] == "custom"


def test_injected_client_is_used_without_constructing_another_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient(response(parsed_chart()))

    def unexpected_client(**kwargs):
        raise AssertionError("OpenAI constructor must not be called")

    monkeypatch.setattr(chart_planner, "OpenAI", unexpected_client)

    planner = ChartPlanner(client=client)

    assert planner.client is client
    assert planner.plan("Compare revenue.", sql_result()).chart_type is ChartType.BAR


def test_default_client_uses_configured_connection_and_finite_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient(response(parsed_chart()))
    construction_calls: list[dict] = []

    def make_client(**kwargs):
        construction_calls.append(kwargs)
        return client

    monkeypatch.setattr(chart_planner, "OpenAI", make_client)

    planner = ChartPlanner(base_url="http://example.test/v1/", api_key="test-key")

    assert planner.client is client
    assert construction_calls == [
        {
            "base_url": "http://example.test/v1/",
            "api_key": "test-key",
            "timeout": 30.0,
        }
    ]


def test_prompt_describes_selection_and_forbids_analysis_or_code() -> None:
    for phrase in (
        "Use bar to compare or rank discrete",
        "Use line for a change or trend",
        "fiscal_year",
        "financial numeric metric",
        "Do not calculate",
        "Do not sort",
        "reorder",
        "Python code",
        "numeric answer",
        "Do not generate or execute Python code or render a chart",
        "do not invent columns",
        "Title and axis labels are plain text values",
        "include no JSON syntax, braces",
        "surrounding quotation marks",
    ):
        assert phrase in CHART_SYSTEM_PROMPT


def test_structured_response_has_only_six_fields_and_forbids_extras() -> None:
    assert set(ChartPlanResponse.model_fields) == {
        "chart_type", "x_column", "y_column", "title", "x_label", "y_label"
    }
    with pytest.raises(ValidationError):
        ChartPlanResponse.model_validate(
            {**parsed_chart().model_dump(), "reasoning": "unsupported"}
        )


@pytest.mark.parametrize("question", ["", "  ", "\n\t", None])
def test_blank_question_is_rejected_before_client_call(question: object) -> None:
    planner, client = planner_with_chart()

    with pytest.raises(ChartPlanningError, match="question must not be empty"):
        planner.plan(question, sql_result())  # type: ignore[arg-type]
    assert client.completions.calls == []


def test_empty_result_is_rejected_before_client_call() -> None:
    planner, client = planner_with_chart()

    with pytest.raises(ChartPlanningError, match="no rows"):
        planner.plan("Compare revenue.", SQLQueryResult([], [], 0))
    assert client.completions.calls == []


def test_unserializable_result_is_rejected_before_client_call() -> None:
    planner, client = planner_with_chart()
    data = sql_result()
    data.rows[0]["revenue_musd"] = object()

    with pytest.raises(ChartPlanningError, match="cannot be serialized") as error:
        planner.plan("Compare revenue.", data)
    assert isinstance(error.value.__cause__, TypeError)
    assert client.completions.calls == []


def test_missing_parsed_output_is_rejected_without_content_fallback() -> None:
    client = FakeClient(response(None, content="bar chart: company versus revenue"))

    with pytest.raises(ChartPlanningError, match="no parsed chart"):
        ChartPlanner(client=client).plan("Compare revenue.", sql_result())


@pytest.mark.parametrize("returned", [None, object(), SimpleNamespace(choices=[])])
def test_malformed_response_shape_is_translated(returned: object) -> None:
    client = FakeClient(returned)

    with pytest.raises(ChartPlanningError, match="invalid structured chart") as error:
        ChartPlanner(client=client).plan("Compare revenue.", sql_result())
    assert error.value.__cause__ is not None


def test_unexpected_parsed_wrapper_type_is_rejected() -> None:
    client = FakeClient(response({"chart_type": "bar"}))

    with pytest.raises(ChartPlanningError, match="unexpected chart response type"):
        ChartPlanner(client=client).plan("Compare revenue.", sql_result())


@pytest.mark.parametrize("field", ["x_column", "y_column"])
def test_missing_declared_column_is_rejected(field: str) -> None:
    planner, _ = planner_with_chart(**{field: "invented_column"})

    with pytest.raises(ChartPlanningError, match="does not exist"):
        planner.plan("Compare revenue.", sql_result())


@pytest.mark.parametrize("field", ["x_column", "y_column"])
def test_missing_row_cell_is_rejected(field: str) -> None:
    planner, _ = planner_with_chart()
    data = sql_result()
    del data.rows[1]["company" if field == "x_column" else "revenue_musd"]

    with pytest.raises(ChartPlanningError, match="missing from row 1"):
        planner.plan("Compare revenue.", data)


def test_same_x_and_y_column_is_rejected() -> None:
    planner, _ = planner_with_chart(y_column="company")

    with pytest.raises(ChartPlanningError, match="different columns"):
        planner.plan("Compare revenue.", sql_result())


def test_context_y_column_is_rejected_when_numeric_metric_exists() -> None:
    planner, _ = planner_with_chart(
        x_column="company", y_column="fiscal_year", title="Year by Company"
    )

    with pytest.raises(ChartPlanningError, match="financial metric"):
        planner.plan("Compare company revenue.", sql_result(with_year=True))


def test_id_y_column_is_rejected_when_revenue_metric_exists() -> None:
    planner, _ = planner_with_chart(y_column="id")
    data = sql_result()
    data.columns.insert(0, "id")
    data.rows[0]["id"] = 1
    data.rows[1]["id"] = 2

    with pytest.raises(ChartPlanningError, match="financial metric"):
        planner.plan("Compare company revenue.", data)


def test_invalid_chart_type_is_rejected() -> None:
    forged = ChartPlanResponse.model_construct(
        chart_type="pie",
        x_column="company",
        y_column="revenue_musd",
        title="Revenue",
    )
    client = FakeClient(response(forged))

    with pytest.raises(ChartPlanningError, match="invalid chart specification"):
        ChartPlanner(client=client).plan("Compare revenue.", sql_result())


@pytest.mark.parametrize("field", ["title", "x_label", "y_label"])
def test_blank_structured_text_is_rejected(field: str) -> None:
    planner, _ = planner_with_chart(**{field: "  "})

    with pytest.raises(ChartPlanningError, match=field):
        planner.plan("Compare revenue.", sql_result())


@pytest.mark.parametrize(
    "title",
    ["Apple {Revenue}", "Apple Revenue}", "Apple Revenue (2024-2025)'}"],
)
def test_title_with_structured_output_braces_is_rejected(title: str) -> None:
    planner, _ = planner_with_chart(title=title)

    with pytest.raises(ChartPlanningError, match="title contains structured-output artifacts"):
        planner.plan("Compare revenue.", sql_result())


@pytest.mark.parametrize("title", ['"Apple Revenue', 'Apple Revenue"', '  Apple Revenue"  '])
def test_title_with_outer_double_quote_artifact_is_rejected(title: str) -> None:
    planner, _ = planner_with_chart(title=title)

    with pytest.raises(ChartPlanningError, match="title contains structured-output artifacts"):
        planner.plan("Compare revenue.", sql_result())


@pytest.mark.parametrize("field", ["x_label", "y_label"])
@pytest.mark.parametrize("value", ["Revenue {USD}", "Revenue }USD"])
def test_label_with_braces_is_rejected_at_runtime(field: str, value: str) -> None:
    planner, _ = planner_with_chart(**{field: value})

    with pytest.raises(ChartPlanningError, match=f"{field} contains structured-output artifacts"):
        planner.plan("Compare revenue.", sql_result())


@pytest.mark.parametrize("field", ["x_label", "y_label"])
def test_label_with_quote_artifact_is_rejected_at_runtime(field: str) -> None:
    planner, _ = planner_with_chart(**{field: 'Revenue"'})

    with pytest.raises(ChartPlanningError, match=f"{field} contains structured-output artifacts"):
        planner.plan("Compare revenue.", sql_result())


@pytest.mark.parametrize("field", ["title", "x_label", "y_label"])
def test_runtime_guard_still_rejects_braces_in_forged_structured_objects(field: str) -> None:
    fields = parsed_chart().model_dump()
    fields[field] = "Revenue {USD}"
    forged = ChartPlanResponse.model_construct(**fields)

    with pytest.raises(ChartPlanningError, match=f"{field} contains structured-output artifacts"):
        ChartPlanner._validate_references(forged, sql_result())


def test_normal_apostrophes_and_punctuation_remain_valid() -> None:
    planner, _ = planner_with_chart(
        title="Apple's Revenue (2024-2025)",
        x_label="Company's ticker: AAPL",
        y_label="Revenue: USD (%)",
    )

    chart = planner.plan("Compare revenue.", sql_result())

    assert chart.title == "Apple's Revenue (2024-2025)"
    assert chart.x_label == "Company's ticker: AAPL"
    assert chart.y_label == "Revenue: USD (%)"


def test_client_exception_preserves_cause() -> None:
    original = RuntimeError("client unavailable")
    client = FakeClient(error=original)

    with pytest.raises(ChartPlanningError, match="request failed") as error:
        ChartPlanner(client=client).plan("Compare revenue.", sql_result())
    assert error.value.__cause__ is original


def test_invalid_model_fields_are_rejected_by_structured_schema() -> None:
    for changes in (
        {"chart_type": "pie"},
        {"x_column": ""},
        {"y_column": ""},
        {"title": ""},
    ):
        with pytest.raises((ValidationError, ValueError)):
            ChartPlanResponse.model_validate({**parsed_chart().model_dump(), **changes})


def test_planner_does_not_mutate_or_reorder_or_calculate_rows() -> None:
    planner, _ = planner_with_chart()
    data = sql_result()
    original = deepcopy(data)

    chart = planner.plan("Rank companies by revenue.", data)

    assert data == original
    assert [row["ticker"] for row in data.rows] == ["MSFT", "AAPL"]
    assert set(vars(chart)) == {
        "chart_type", "x_column", "y_column", "title", "x_label", "y_label"
    }
