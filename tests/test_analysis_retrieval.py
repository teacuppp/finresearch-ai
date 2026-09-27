from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.analysis.financial_analyzer import AnalysisOperation
from app.analysis.retrieval import (
    ANALYSIS_RETRIEVAL_SYSTEM_PROMPT,
    AnalysisRetrievalPlan,
    AnalysisRetrievalPlanner,
    AnalysisRetrievalPlanningError,
)


class FakeCompletions:
    def __init__(self, parsed=None, error=None):
        self.parsed = parsed
        self.error = error
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(parsed=self.parsed, content="not a fallback")
        )])


class FakeClient:
    def __init__(self, parsed=None, error=None):
        self.completions = FakeCompletions(parsed, error)
        self.chat = SimpleNamespace(completions=self.completions)


def make_plan(entity_column="company", entities=("Apple",),
              fiscal_years=(2024, 2025), metric="revenue_musd"):
    return AnalysisRetrievalPlan(
        entity_column=entity_column, entities=entities,
        fiscal_years=fiscal_years, metric=metric,
    )


@pytest.mark.parametrize("operation,plan", [
    (AnalysisOperation.ABSOLUTE_CHANGE, make_plan()),
    (AnalysisOperation.PERCENTAGE_CHANGE,
     make_plan("ticker", ("AAPL",), (2024, 2025), "net_income_musd")),
    (AnalysisOperation.DIFFERENCE,
     make_plan("company", ("Apple", "Microsoft"), (2025,), "revenue_musd")),
    (AnalysisOperation.DIFFERENCE,
     make_plan("ticker", ("AAPL", "MSFT"), (2025,), "gross_margin")),
    (AnalysisOperation.RANKING,
     make_plan("company", ("Apple", "Microsoft"), (2025,), "operating_income_musd")),
    (AnalysisOperation.RANKING,
     make_plan("ticker", (), (2025,), "revenue_musd")),
])
def test_valid_extraction_for_operations_entities_and_metrics(operation, plan):
    client = FakeClient(plan)
    assert AnalysisRetrievalPlanner(client=client).plan("Original question", operation) == plan
    assert len(client.completions.calls) == 1


def test_model_call_preserves_question_operation_and_structured_settings():
    client = FakeClient(make_plan())
    question = "  Apple's revenue from 2024 to 2025?\n"
    AnalysisRetrievalPlanner(client=client).plan(question, AnalysisOperation.ABSOLUTE_CHANGE)

    assert client.completions.calls == [{
        "model": "qwen3:4b",
        "messages": [
            {"role": "system", "content": ANALYSIS_RETRIEVAL_SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Original question:\n{question}\n\n"
                "Requested analysis operation: absolute_change"
            )},
        ],
        "response_format": AnalysisRetrievalPlan,
        "temperature": 0,
        "reasoning_effort": "none",
    }]


def test_configured_model_and_injected_client_are_used():
    client = FakeClient(make_plan())
    planner = AnalysisRetrievalPlanner(model="custom", client=client)
    assert planner.client is client
    planner.plan("question", AnalysisOperation.ABSOLUTE_CHANGE)
    assert client.completions.calls[0]["model"] == "custom"


def test_prompt_limits_retrieval_and_forbids_sql_calculation_and_ranking():
    prompt = ANALYSIS_RETRIEVAL_SYSTEM_PROMPT
    for phrase in ("company", "ticker", "exact entities", "fiscal years",
                   "Do not generate SQL", "calculate", "rank, sort", "reasoning"):
        assert phrase in prompt
    for metric in ("revenue_musd", "net_income_musd",
                   "operating_income_musd", "gross_margin"):
        assert metric in prompt


@pytest.mark.parametrize("field,value", [
    ("entity_column", "id"), ("metric", "invented"),
])
def test_schema_rejects_unsupported_identifier_or_metric(field, value):
    data = make_plan().model_dump()
    data[field] = value
    with pytest.raises(ValidationError):
        AnalysisRetrievalPlan.model_validate(data)


def test_schema_is_frozen_and_rejects_extra_fields():
    with pytest.raises(ValidationError):
        AnalysisRetrievalPlan.model_validate({**make_plan().model_dump(), "sql": "SELECT 1"})
    with pytest.raises(ValidationError):
        make_plan().metric = "gross_margin"


def test_fiscal_year_schema_accepts_range_boundaries():
    plan = make_plan(fiscal_years=(1900, 2100))
    assert plan.fiscal_years == (1900, 2100)


@pytest.mark.parametrize("year", [1899, 2101, 2])
def test_fiscal_year_schema_rejects_out_of_range_values(year):
    data = make_plan().model_dump()
    data["fiscal_years"] = (2024, year)
    with pytest.raises(ValidationError):
        AnalysisRetrievalPlan.model_validate(data)


@pytest.mark.parametrize("year", [True, False, "2025"])
def test_fiscal_year_schema_rejects_non_integer_values(year):
    data = make_plan().model_dump()
    data["fiscal_years"] = (2024, year)
    with pytest.raises(ValidationError):
        AnalysisRetrievalPlan.model_validate(data)


def test_out_of_range_year_in_model_output_is_translated_with_cause():
    malformed = AnalysisRetrievalPlan.model_construct(
        entity_column="company",
        entities=("Microsoft",),
        fiscal_years=(2024, 2),
        metric="revenue_musd",
    )
    client = FakeClient(malformed)
    question = "By what percentage did Microsoft's revenue change from 2024 to 2025?"

    with pytest.raises(AnalysisRetrievalPlanningError, match="invalid retrieval plan") as exc:
        AnalysisRetrievalPlanner(client=client).plan(
            question, AnalysisOperation.PERCENTAGE_CHANGE
        )
    assert isinstance(exc.value.__cause__, ValidationError)


@pytest.mark.parametrize("question,operation", [
    ("", AnalysisOperation.RANKING), ("  \n", AnalysisOperation.RANKING),
    ("question", "ranking"), ("question", object()),
])
def test_invalid_input_is_rejected_before_model_call(question, operation):
    client = FakeClient()
    with pytest.raises(AnalysisRetrievalPlanningError):
        AnalysisRetrievalPlanner(client=client).plan(question, operation)
    assert client.completions.calls == []


@pytest.mark.parametrize("operation,plan", [
    (AnalysisOperation.ABSOLUTE_CHANGE, make_plan(entities=())),
    (AnalysisOperation.PERCENTAGE_CHANGE, make_plan(fiscal_years=(2025,))),
    (AnalysisOperation.DIFFERENCE, make_plan()),
    (AnalysisOperation.RANKING, make_plan(entities=("Apple",), fiscal_years=(2025,))),
    (AnalysisOperation.RANKING, make_plan(entities=(), fiscal_years=(2024, 2025))),
    (AnalysisOperation.DIFFERENCE, make_plan(entities=("Apple", "Microsoft"), fiscal_years=())),
    (AnalysisOperation.ABSOLUTE_CHANGE, make_plan(entities=(" ",))),
    (AnalysisOperation.DIFFERENCE,
     make_plan(entities=("Apple", " apple "), fiscal_years=(2025,))),
    (AnalysisOperation.ABSOLUTE_CHANGE, make_plan(fiscal_years=(2025, 2025))),
])
def test_malformed_cardinality_or_values_are_rejected(operation, plan):
    client = FakeClient(plan)
    with pytest.raises(AnalysisRetrievalPlanningError, match="invalid retrieval plan"):
        AnalysisRetrievalPlanner(client=client).plan("question", operation)


@pytest.mark.parametrize("parsed", [None, "natural language", {"metric": "revenue_musd"}])
def test_missing_or_wrong_parsed_output_has_no_content_fallback(parsed):
    client = FakeClient(parsed)
    with pytest.raises(AnalysisRetrievalPlanningError):
        AnalysisRetrievalPlanner(client=client).plan("question", AnalysisOperation.RANKING)


def test_malformed_response_wrapper_preserves_cause():
    client = FakeClient(make_plan())
    client.completions.parse = lambda **kwargs: SimpleNamespace(choices=[])
    with pytest.raises(AnalysisRetrievalPlanningError) as exc:
        AnalysisRetrievalPlanner(client=client).plan("question", AnalysisOperation.RANKING)
    assert isinstance(exc.value.__cause__, IndexError)


def test_model_failure_is_translated_with_cause():
    failure = RuntimeError("private model error")
    client = FakeClient(error=failure)
    with pytest.raises(AnalysisRetrievalPlanningError) as exc:
        AnalysisRetrievalPlanner(client=client).plan("question", AnalysisOperation.RANKING)
    assert exc.value.__cause__ is failure


def test_forged_model_is_revalidated():
    forged = AnalysisRetrievalPlan.model_construct(
        entity_column="id", entities=("Apple",), fiscal_years=(2024, 2025),
        metric="revenue_musd",
    )
    with pytest.raises(AnalysisRetrievalPlanningError):
        AnalysisRetrievalPlanner(client=FakeClient(forged)).plan(
            "question", AnalysisOperation.ABSOLUTE_CHANGE
        )
