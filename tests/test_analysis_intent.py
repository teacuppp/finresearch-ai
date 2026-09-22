from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.analysis.financial_analyzer import AnalysisOperation
from app.analysis.intent import (
    SQL_ANALYSIS_SYSTEM_PROMPT,
    AnalysisSQLTask,
    DirectSQLTask,
    SQLAnalysisClassificationError,
    SQLAnalysisClassifier,
    SQLTaskResponse,
)


def _response(parsed: object, content: str | None = None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(message=SimpleNamespace(parsed=parsed, content=content))
        ]
    )


class FakeCompletions:
    def __init__(self, response: object = None, error: Exception | None = None):
        self.response = response
        self.error = error
        self.calls: list[dict] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class FakeClient:
    def __init__(self, response: object = None, error: Exception | None = None):
        self.completions = FakeCompletions(response=response, error=error)
        self.chat = SimpleNamespace(completions=self.completions)


@pytest.mark.parametrize(
    "question",
    [
        "What was Apple's revenue in 2025?",
        "Show Microsoft's 2025 net income.",
        "What was the average revenue of Apple and Microsoft in 2025?",
        "How many financial metric rows exist?",
        "What is the sum of Apple and Microsoft revenue in 2025?",
        "What is the minimum revenue in 2025?",
        "What is the maximum revenue in 2025?",
        "What is Apple's ratio of net income to revenue in 2025?",
    ],
)
def test_direct_structured_response_returns_direct_decision(question):
    parsed = SQLTaskResponse.model_validate({"decision": {"mode": "direct"}})
    client = FakeClient(response=_response(parsed))

    decision = SQLAnalysisClassifier(client=client).classify(question)

    assert isinstance(decision, DirectSQLTask)
    assert decision.model_dump() == {"mode": "direct"}


@pytest.mark.parametrize(
    "question, operation",
    [
        (
            "By what percentage did Apple's revenue change from 2024 to 2025?",
            AnalysisOperation.PERCENTAGE_CHANGE,
        ),
        (
            "How much did Apple's revenue increase from 2024 to 2025?",
            AnalysisOperation.ABSOLUTE_CHANGE,
        ),
        (
            "What is the difference between Apple's and Microsoft's 2025 revenue?",
            AnalysisOperation.DIFFERENCE,
        ),
        (
            "Rank Apple and Microsoft by 2025 revenue from highest to lowest.",
            AnalysisOperation.RANKING,
        ),
    ],
)
def test_analysis_structured_response_returns_supported_operation(question, operation):
    parsed = SQLTaskResponse.model_validate({
        "decision": {"mode": "analysis", "operation": operation.value}
    })
    client = FakeClient(response=_response(parsed))

    decision = SQLAnalysisClassifier(client=client).classify(question)

    assert isinstance(decision, AnalysisSQLTask)
    assert decision.mode == "analysis"
    assert decision.operation is operation


def test_model_call_preserves_question_and_uses_structured_zero_temperature_output():
    parsed = SQLTaskResponse(decision=DirectSQLTask(mode="direct"))
    client = FakeClient(response=_response(parsed))
    question = "  What was Apple's revenue in 2025?\n"

    SQLAnalysisClassifier(client=client).classify(question)

    assert client.completions.calls == [{
        "model": "qwen3:4b",
        "messages": [
            {"role": "system", "content": SQL_ANALYSIS_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "response_format": SQLTaskResponse,
        "temperature": 0,
    }]


def test_configured_model_is_used():
    parsed = SQLTaskResponse(decision=DirectSQLTask(mode="direct"))
    client = FakeClient(response=_response(parsed))

    SQLAnalysisClassifier(model="custom-model", client=client).classify("Show revenue.")

    assert client.completions.calls[0]["model"] == "custom-model"


def test_prompt_encodes_supported_operations_and_direct_sql_policy():
    prompt = SQL_ANALYSIS_SYSTEM_PROMPT
    assert "top-level router has selected SQL" in prompt
    assert "Python FinancialAnalyzer" in prompt
    assert "existing SQL generator should answer" in prompt
    for operation in AnalysisOperation:
        assert operation.value in prompt
    for unsupported in ("average / mean", "sum", "count", "min/max", "ratios"):
        assert unsupported in prompt
    assert "Use direct for metric lookups and for unsupported operations" in prompt
    assert "Choose analysis ONLY" in prompt
    assert "Do not answer the financial question, generate SQL, or perform" in prompt


def test_schema_discriminates_on_mode_and_reuses_analysis_operation():
    schema = SQLTaskResponse.model_json_schema()
    assert schema["properties"]["decision"]["discriminator"]["propertyName"] == "mode"
    assert len(schema["properties"]["decision"]["oneOf"]) == 2
    assert AnalysisSQLTask.model_fields["operation"].annotation is AnalysisOperation
    assert set(DirectSQLTask.model_fields) == {"mode"}
    assert set(AnalysisSQLTask.model_fields) == {"mode", "operation"}


@pytest.mark.parametrize("question", ["", " ", "\n\t"])
def test_blank_question_is_rejected_before_model_call(question):
    client = FakeClient()

    with pytest.raises(SQLAnalysisClassificationError, match="question must not be empty"):
        SQLAnalysisClassifier(client=client).classify(question)

    assert client.completions.calls == []


def test_missing_parsed_output_is_rejected():
    client = FakeClient(response=_response(None))

    with pytest.raises(SQLAnalysisClassificationError, match="no parsed SQL task"):
        SQLAnalysisClassifier(client=client).classify("Show revenue.")


@pytest.mark.parametrize(
    "response",
    [SimpleNamespace(choices=[]), SimpleNamespace(), _response(None).choices[0]],
)
def test_malformed_response_is_rejected_with_cause(response):
    client = FakeClient(response=response)

    with pytest.raises(SQLAnalysisClassificationError, match="invalid structured") as exc:
        SQLAnalysisClassifier(client=client).classify("Show revenue.")

    assert isinstance(exc.value.__cause__, (AttributeError, IndexError))


@pytest.mark.parametrize("parsed", [{"decision": {"mode": "direct"}}, "direct"])
def test_unexpected_structured_output_type_is_rejected(parsed):
    client = FakeClient(response=_response(parsed))

    with pytest.raises(SQLAnalysisClassificationError, match="unexpected structured"):
        SQLAnalysisClassifier(client=client).classify("Show revenue.")


@pytest.mark.parametrize(
    "parsed",
    [
        SQLTaskResponse.model_construct(),
        SQLTaskResponse.model_construct(decision="direct"),
        SQLTaskResponse.model_construct(
            decision=DirectSQLTask.model_construct(mode="rag")
        ),
        SQLTaskResponse.model_construct(
            decision=AnalysisSQLTask.model_construct(mode="analysis")
        ),
        SQLTaskResponse.model_construct(
            decision=AnalysisSQLTask.model_construct(mode="analysis", operation="mean")
        ),
    ],
)
def test_malformed_model_instances_are_rejected_with_cause(parsed):
    client = FakeClient(response=_response(parsed))

    with pytest.raises(SQLAnalysisClassificationError, match="invalid structured") as exc:
        SQLAnalysisClassifier(client=client).classify("Show revenue.")

    assert isinstance(exc.value.__cause__, ValidationError)


@pytest.mark.parametrize("error", [RuntimeError("client unavailable"), ValueError("JSON")])
def test_client_or_parsing_error_preserves_cause(error):
    client = FakeClient(error=error)

    with pytest.raises(SQLAnalysisClassificationError, match="request failed") as exc:
        SQLAnalysisClassifier(client=client).classify("Show revenue.")

    assert exc.value.__cause__ is error
    assert len(client.completions.calls) == 1


def test_natural_language_content_is_not_used_as_fallback():
    client = FakeClient(response=_response(None, content="Use direct SQL."))

    with pytest.raises(SQLAnalysisClassificationError, match="no parsed SQL task"):
        SQLAnalysisClassifier(client=client).classify("Show revenue.")


@pytest.mark.parametrize(
    "payload",
    [
        {"decision": {"mode": "direct", "operation": "ranking"}},
        {"decision": {"mode": "direct", "reasoning": "A metric lookup."}},
        {"decision": {"mode": "analysis", "operation": "ranking", "sql": "SELECT 1"}},
        {"decision": {"mode": "direct"}, "confidence": 1.0},
    ],
)
def test_extra_structured_fields_are_rejected(payload):
    with pytest.raises(ValidationError):
        SQLTaskResponse.model_validate(payload)


@pytest.mark.parametrize("operation", ["average", "sum", "count", "min", "max", "ratio"])
def test_unsupported_analysis_operation_is_rejected_by_schema(operation):
    with pytest.raises(ValidationError):
        SQLTaskResponse.model_validate({
            "decision": {"mode": "analysis", "operation": operation}
        })
