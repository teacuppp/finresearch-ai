import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.agent.sql_executor import SQLQueryResult
from app.analysis.financial_analyzer import AnalysisOperation, RankingDirection
from app.analysis.planner import (
    PLAN_SYSTEM_PROMPT,
    AbsoluteChangePlan,
    AnalysisPlanner,
    AnalysisPlanningError,
    AnalysisPlanResponse,
    DifferencePlan,
    PercentageChangePlan,
    RankingPlan,
)


def _sql_result() -> SQLQueryResult:
    return SQLQueryResult(
        columns=["ticker", "fiscal_year", "revenue_musd"],
        rows=[
            {"ticker": "AAPL", "fiscal_year": 2024, "revenue_musd": 391035},
            {"ticker": "AAPL", "fiscal_year": 2025, "revenue_musd": 416161},
        ],
        row_count=2,
    )


def _response(parsed: object, content: str | None = None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(parsed=parsed, content=content)
            )
        ]
    )


class FakeCompletions:
    def __init__(
        self,
        response: object | None = None,
        error: Exception | None = None,
    ):
        self.response = response
        self.error = error
        self.calls: list[dict] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class FakeClient:
    def __init__(
        self,
        response: object | None = None,
        error: Exception | None = None,
    ):
        self.completions = FakeCompletions(response=response, error=error)
        self.chat = SimpleNamespace(completions=self.completions)


def _planner_with_plan(plan: object) -> tuple[AnalysisPlanner, FakeClient]:
    client = FakeClient(response=_response(AnalysisPlanResponse(plan=plan)))
    return AnalysisPlanner(client=client), client


def test_percentage_change_plan_parses_to_specific_schema():
    plan = PercentageChangePlan(
        operation=AnalysisOperation.PERCENTAGE_CHANGE,
        old_row=0,
        old_column="revenue_musd",
        new_row=1,
        new_column="revenue_musd",
    )
    planner, _ = _planner_with_plan(plan)

    result = planner.plan("What was the revenue growth rate?", _sql_result())

    assert result is plan
    assert result.old_row == 0
    assert result.new_row == 1


def test_absolute_change_plan_parses_to_specific_schema():
    plan = AbsoluteChangePlan(
        operation=AnalysisOperation.ABSOLUTE_CHANGE,
        old_row=0,
        old_column="revenue_musd",
        new_row=1,
        new_column="revenue_musd",
    )
    planner, _ = _planner_with_plan(plan)

    assert planner.plan("How much did revenue increase?", _sql_result()) is plan


def test_difference_plan_parses_to_specific_schema():
    plan = DifferencePlan(
        operation=AnalysisOperation.DIFFERENCE,
        left_row=1,
        left_column="revenue_musd",
        right_row=0,
        right_column="revenue_musd",
    )
    planner, _ = _planner_with_plan(plan)

    assert planner.plan("What is the difference?", _sql_result()) is plan


@pytest.mark.parametrize(
    "direction",
    [RankingDirection.ASCENDING, RankingDirection.DESCENDING],
)
def test_ranking_plan_preserves_requested_direction(direction):
    plan = RankingPlan(
        operation=AnalysisOperation.RANKING,
        column="revenue_musd",
        direction=direction,
    )
    planner, _ = _planner_with_plan(plan)

    result = planner.plan(f"Rank revenue {direction.value}.", _sql_result())

    assert result is plan
    assert result.direction is direction


def test_prompt_contains_original_question_columns_and_numbered_rows():
    plan = RankingPlan(
        operation=AnalysisOperation.RANKING,
        column="revenue_musd",
        direction=RankingDirection.DESCENDING,
    )
    planner, client = _planner_with_plan(plan)
    question = "  Rank Apple revenue descending.  "
    sql_result = _sql_result()

    planner.plan(question, sql_result)

    user_content = client.completions.calls[0]["messages"][1]["content"]
    assert f"Original question:\n{question}\n" in user_content
    serialized = user_content.split("SQL result (original row order):\n", 1)[1]
    assert json.loads(serialized) == {
        "columns": sql_result.columns,
        "row_count": 2,
        "numbered_rows": [
            {"row_index": 0, "values": sql_result.rows[0]},
            {"row_index": 1, "values": sql_result.rows[1]},
        ],
    }
    assert sql_result.rows[0]["fiscal_year"] == 2024
    assert sql_result.rows[1]["fiscal_year"] == 2025


def test_model_call_uses_structured_response_and_zero_temperature():
    plan = RankingPlan(
        operation=AnalysisOperation.RANKING,
        column="revenue_musd",
        direction=RankingDirection.ASCENDING,
    )
    planner, client = _planner_with_plan(plan)

    planner.plan("Rank revenue ascending.", _sql_result())

    assert len(client.completions.calls) == 1
    call = client.completions.calls[0]
    assert call["model"] == "qwen3:4b"
    assert call["temperature"] == 0
    assert call["reasoning_effort"] == "none"
    assert call["response_format"] is AnalysisPlanResponse
    assert call["messages"][0] == {
        "role": "system",
        "content": PLAN_SYSTEM_PROMPT,
    }


def test_prompt_limits_operations_and_forbids_calculation():
    prompt = PLAN_SYSTEM_PROMPT
    for phrase in (
        "absolute_change",
        "percentage_change",
        "difference",
        "ranking",
        "new - old",
        "(new - old) / old * 100",
        "left - right",
        "zero-based",
        "ascending or descending",
        "Do not calculate",
    ):
        assert phrase in prompt


def test_response_schema_uses_discriminated_union_and_forbids_extra_fields():
    plan_schema = AnalysisPlanResponse.model_json_schema()["properties"]["plan"]
    assert plan_schema["discriminator"]["propertyName"] == "operation"
    assert len(plan_schema["oneOf"]) == 4

    with pytest.raises(ValidationError):
        AnalysisPlanResponse.model_validate({
            "plan": {
                "operation": "ranking",
                "column": "revenue_musd",
                "direction": "descending",
                "reasoning": "Unneeded private reasoning",
            }
        })


def test_structured_output_parses_percentage_change_variant():
    response = AnalysisPlanResponse.model_validate({
        "plan": {
            "operation": "percentage_change",
            "old_row": 0,
            "old_column": "revenue_musd",
            "new_row": 1,
            "new_column": "revenue_musd",
        }
    })

    assert isinstance(response.plan, PercentageChangePlan)
    assert response.plan.operation is AnalysisOperation.PERCENTAGE_CHANGE


def test_blank_question_is_rejected_before_model_call():
    client = FakeClient()
    planner = AnalysisPlanner(client=client)

    with pytest.raises(AnalysisPlanningError, match="question must not be empty"):
        planner.plan(" \t ", _sql_result())

    assert client.completions.calls == []


def test_empty_sql_result_is_rejected_before_model_call():
    client = FakeClient()
    planner = AnalysisPlanner(client=client)
    sql_result = SQLQueryResult(columns=["value"], rows=[], row_count=0)

    with pytest.raises(AnalysisPlanningError, match="SQL result has no rows"):
        planner.plan("What changed?", sql_result)

    assert client.completions.calls == []


def test_missing_parsed_output_is_rejected():
    client = FakeClient(response=_response(None))

    with pytest.raises(AnalysisPlanningError, match="no parsed analysis plan"):
        AnalysisPlanner(client=client).plan("What changed?", _sql_result())


def test_unexpected_parsed_wrapper_type_is_rejected():
    plan = RankingPlan(
        operation=AnalysisOperation.RANKING,
        column="revenue_musd",
        direction=RankingDirection.ASCENDING,
    )
    client = FakeClient(response=_response(plan))

    with pytest.raises(AnalysisPlanningError, match="unexpected plan type"):
        AnalysisPlanner(client=client).plan("Rank revenue.", _sql_result())


def test_unexpected_plan_variant_is_rejected():
    response = AnalysisPlanResponse.model_construct(plan="unsupported")
    client = FakeClient(response=_response(response))

    with pytest.raises(AnalysisPlanningError, match="unexpected plan type"):
        AnalysisPlanner(client=client).plan("What changed?", _sql_result())


def test_invalid_row_index_is_rejected_after_parsing():
    plan = AbsoluteChangePlan(
        operation=AnalysisOperation.ABSOLUTE_CHANGE,
        old_row=0,
        old_column="revenue_musd",
        new_row=2,
        new_column="revenue_musd",
    )
    planner, _ = _planner_with_plan(plan)

    with pytest.raises(AnalysisPlanningError, match="row index 2 is out of range"):
        planner.plan("What changed?", _sql_result())


def test_invalid_column_is_rejected_after_parsing():
    plan = PercentageChangePlan(
        operation=AnalysisOperation.PERCENTAGE_CHANGE,
        old_row=0,
        old_column="revenue",
        new_row=1,
        new_column="revenue_musd",
    )
    planner, _ = _planner_with_plan(plan)

    with pytest.raises(AnalysisPlanningError, match="column 'revenue' does not exist"):
        planner.plan("What changed?", _sql_result())


def test_invalid_ranking_column_is_rejected_after_parsing():
    plan = RankingPlan(
        operation=AnalysisOperation.RANKING,
        column="revenue",
        direction=RankingDirection.DESCENDING,
    )
    planner, _ = _planner_with_plan(plan)

    with pytest.raises(AnalysisPlanningError, match="column 'revenue' does not exist"):
        planner.plan("Rank revenue.", _sql_result())


def test_plan_column_must_be_present_in_selected_row():
    sql_result = _sql_result()
    del sql_result.rows[1]["revenue_musd"]
    plan = DifferencePlan(
        operation=AnalysisOperation.DIFFERENCE,
        left_row=1,
        left_column="revenue_musd",
        right_row=0,
        right_column="revenue_musd",
    )
    planner, _ = _planner_with_plan(plan)

    with pytest.raises(AnalysisPlanningError, match="missing from row 1"):
        planner.plan("What is the difference?", sql_result)


def test_ranking_column_must_be_present_in_every_row():
    sql_result = _sql_result()
    del sql_result.rows[1]["revenue_musd"]
    plan = RankingPlan(
        operation=AnalysisOperation.RANKING,
        column="revenue_musd",
        direction=RankingDirection.DESCENDING,
    )
    planner, _ = _planner_with_plan(plan)

    with pytest.raises(AnalysisPlanningError, match="missing from a result row"):
        planner.plan("Rank revenue.", sql_result)


def test_invalid_row_field_is_rejected_even_if_model_validation_was_bypassed():
    plan = AbsoluteChangePlan.model_construct(
        operation=AnalysisOperation.ABSOLUTE_CHANGE,
        old_row="0",
        old_column="revenue_musd",
        new_row=1,
        new_column="revenue_musd",
    )
    response = AnalysisPlanResponse.model_construct(plan=plan)
    client = FakeClient(response=_response(response))

    with pytest.raises(AnalysisPlanningError, match="row index '0' is out of range"):
        AnalysisPlanner(client=client).plan("What changed?", _sql_result())


def test_mismatched_operation_is_rejected_even_if_model_validation_was_bypassed():
    plan = AbsoluteChangePlan.model_construct(
        operation=AnalysisOperation.DIFFERENCE,
        old_row=0,
        old_column="revenue_musd",
        new_row=1,
        new_column="revenue_musd",
    )
    response = AnalysisPlanResponse.model_construct(plan=plan)
    client = FakeClient(response=_response(response))

    with pytest.raises(AnalysisPlanningError, match="does not match its schema"):
        AnalysisPlanner(client=client).plan("What changed?", _sql_result())


def test_client_error_is_translated_with_original_cause():
    client_error = RuntimeError("Ollama is unavailable")
    client = FakeClient(error=client_error)

    with pytest.raises(AnalysisPlanningError, match="planning request failed") as exc:
        AnalysisPlanner(client=client).plan("What changed?", _sql_result())

    assert exc.value.__cause__ is client_error


def test_parse_error_is_translated_with_original_cause():
    parse_error = ValueError("invalid JSON")
    client = FakeClient(error=parse_error)

    with pytest.raises(AnalysisPlanningError, match="planning request failed") as exc:
        AnalysisPlanner(client=client).plan("What changed?", _sql_result())

    assert exc.value.__cause__ is parse_error


def test_malformed_response_shape_is_translated_with_original_cause():
    client = FakeClient(response=SimpleNamespace(choices=[]))

    with pytest.raises(AnalysisPlanningError, match="invalid structured") as exc:
        AnalysisPlanner(client=client).plan("What changed?", _sql_result())

    assert isinstance(exc.value.__cause__, IndexError)


def test_natural_language_response_is_not_used_as_fallback():
    client = FakeClient(response=_response(None, content="Use percentage change."))

    with pytest.raises(AnalysisPlanningError, match="no parsed analysis plan"):
        AnalysisPlanner(client=client).plan("What changed?", _sql_result())


def test_planner_does_not_validate_numeric_values_or_calculate():
    sql_result = SQLQueryResult(
        columns=["value"],
        rows=[{"value": "100"}, {"value": "125"}],
        row_count=2,
    )
    plan = PercentageChangePlan(
        operation=AnalysisOperation.PERCENTAGE_CHANGE,
        old_row=0,
        old_column="value",
        new_row=1,
        new_column="value",
    )
    planner, _ = _planner_with_plan(plan)

    assert planner.plan("What was the percentage change?", sql_result) is plan
