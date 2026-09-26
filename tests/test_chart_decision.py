from copy import deepcopy
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import app.analysis.chart_decision as chart_decision
from app.agent.sql_executor import SQLQueryResult
from app.analysis.chart_decision import (
    CHART_INTENT_SYSTEM_PROMPT,
    ChartDecisionError,
    ChartDecisionPolicy,
    ChartIntentClassifier,
    ChartIntentResponse,
    has_explicit_visualization_request,
)


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


def classifier_with_intent(intent: str) -> tuple[ChartIntentClassifier, FakeClient]:
    parsed = ChartIntentResponse.model_validate({"intent": intent})
    client = FakeClient(response(parsed))
    return ChartIntentClassifier(client=client), client


def sql_result(row_count: int, *, multiple_metrics: bool = False) -> SQLQueryResult:
    rows = [
        {"company": "Apple", "revenue_musd": 416161},
        {"company": "Microsoft", "revenue_musd": 281724},
    ][:row_count]
    columns = ["company", "revenue_musd"]
    if multiple_metrics:
        columns.append("net_income_musd")
        for row in rows:
            row["net_income_musd"] = 100000
    return SQLQueryResult(columns=columns, rows=rows, row_count=999)


@pytest.mark.parametrize(
    ("question", "intent"),
    [
        ("Show Apple's revenue trend.", "trend"),
        ("Compare Apple and Microsoft 2025 revenue.", "comparison"),
        ("Rank Apple and Microsoft by 2025 revenue.", "ranking"),
        ("What was Apple's 2025 revenue?", "lookup"),
        ("List Apple and Microsoft revenue rows.", "list"),
        ("Explain the market context.", "other"),
        ("Plot Apple's 2025 revenue.", "lookup"),
        ("Chart Apple's 2025 revenue.", "lookup"),
        ("Show Apple's 2025 revenue.", "lookup"),
        ("Plot Apple's revenue trend.", "trend"),
        ("Graph Apple versus Microsoft 2025 revenue.", "comparison"),
        ("Visualize the ranking of Apple and Microsoft by 2025 revenue.", "ranking"),
        ("List revenue and net income for Apple and Microsoft.", "list"),
    ],
)
def test_fake_classifier_extracts_only_semantic_intent(
    question: str, intent: str
) -> None:
    classifier, client = classifier_with_intent(intent)

    assert classifier.classify(question) == ChartIntentResponse(intent=intent)
    assert client.completions.calls[0]["messages"][1] == {
        "role": "user",
        "content": question,
    }


def test_prompt_defines_intents_examples_and_extraction_boundaries() -> None:
    for phrase in (
        "trend: trend, change, or progression",
        "comparison: explicit comparison",
        "ranking: explicit ranking",
        "lookup: factual metric or value lookup",
        "list: list, display, return, or table",
        "other: none of the above",
        "IGNORE presentation or visualization wording",
        '"plot", "chart", "graph", and "visualize"',
        '"Plot Apple\'s 2025 revenue." -> lookup',
        '"Chart Apple\'s 2025 revenue." -> lookup',
        '"Plot Apple\'s revenue trend." -> trend',
        '"Graph Apple versus Microsoft 2025 revenue." -> comparison',
        '"Visualize the ranking of Apple and Microsoft by 2025 revenue." -> ranking',
        "Do not determine explicit visualization, inspect rows or row counts",
        "Do not answer",
        "choose chart type",
        "select axes",
        "calculate",
        "generate",
        "SQL or Python",
        "render",
        "reasoning, confidence, or report fields",
    ):
        assert phrase in CHART_INTENT_SYSTEM_PROMPT


def test_original_question_is_sent_unchanged_without_sql_data() -> None:
    classifier, client = classifier_with_intent("comparison")
    question = "  Compare Apple and Microsoft revenue.  "

    classifier.classify(question)

    call = client.completions.calls[0]
    assert call["messages"] == [
        {"role": "system", "content": CHART_INTENT_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    assert "columns" not in str(call["messages"])
    assert "row_count" not in str(call["messages"])
    assert "rows" not in call["messages"][1]["content"]


def test_call_uses_structured_schema_and_runtime_settings() -> None:
    classifier, client = classifier_with_intent("trend")

    classifier.classify("Show Apple's revenue trend.")

    call = client.completions.calls[0]
    assert call["model"] == "qwen3:4b"
    assert call["temperature"] == 0
    assert call["reasoning_effort"] == "none"
    assert call["response_format"] is ChartIntentResponse


def test_response_schema_is_closed_frozen_and_has_only_intent_fields() -> None:
    schema = ChartIntentResponse.model_json_schema()
    assert set(ChartIntentResponse.model_fields) == {"intent"}
    assert schema["additionalProperties"] is False
    assert set(schema["properties"]) == {"intent"}
    assert schema["properties"]["intent"]["enum"] == [
        "trend", "comparison", "ranking", "lookup", "list", "other"
    ]
    with pytest.raises(ValidationError):
        ChartIntentResponse.model_validate(
            {"intent": "lookup", "explicit_visualization": False}
        )
    with pytest.raises(ValidationError):
        ChartIntentResponse(intent="lookup").intent = "list"


def test_configured_model_is_respected() -> None:
    client = FakeClient(response(ChartIntentResponse(intent="lookup")))

    ChartIntentClassifier(model="custom", client=client).classify("What was revenue?")

    assert client.completions.calls[0]["model"] == "custom"


def test_injected_client_is_used_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    classifier, client = classifier_with_intent("lookup")

    def unexpected_client(**kwargs):
        raise AssertionError("OpenAI constructor must not be called")

    monkeypatch.setattr(chart_decision, "OpenAI", unexpected_client)
    assert classifier.client is client
    assert classifier.classify("What was revenue?").intent == "lookup"


def test_default_client_uses_configured_connection_and_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient()
    calls: list[dict] = []

    def make_client(**kwargs):
        calls.append(kwargs)
        return client

    monkeypatch.setattr(chart_decision, "OpenAI", make_client)

    classifier = ChartIntentClassifier(
        base_url="http://example.test/v1/", api_key="test-key"
    )

    assert classifier.client is client
    assert calls == [
        {"base_url": "http://example.test/v1/", "api_key": "test-key", "timeout": 30.0}
    ]


@pytest.mark.parametrize("question", ["", "  ", "\n\t", None, 42])
def test_invalid_question_is_rejected_before_client_call(question: object) -> None:
    classifier, client = classifier_with_intent("lookup")

    with pytest.raises(ChartDecisionError, match="question must not be empty"):
        classifier.classify(question)  # type: ignore[arg-type]
    assert client.completions.calls == []


def test_missing_parsed_output_does_not_use_natural_language_fallback() -> None:
    client = FakeClient(response(None, content="trend"))

    with pytest.raises(ChartDecisionError, match="no parsed chart intent"):
        ChartIntentClassifier(client=client).classify("Plot revenue trend.")


@pytest.mark.parametrize("returned", [None, object(), SimpleNamespace(choices=[])])
def test_malformed_response_shape_is_rejected(returned: object) -> None:
    client = FakeClient(returned)

    with pytest.raises(ChartDecisionError, match="invalid structured") as error:
        ChartIntentClassifier(client=client).classify("Plot revenue trend.")
    assert error.value.__cause__ is not None


def test_unexpected_parsed_wrapper_is_rejected() -> None:
    client = FakeClient(response({"intent": "trend"}))

    with pytest.raises(ChartDecisionError, match="unexpected chart intent type"):
        ChartIntentClassifier(client=client).classify("Plot revenue trend.")


def test_schema_validation_failure_is_translated_with_cause() -> None:
    forged = ChartIntentResponse.model_construct(intent="chart")
    client = FakeClient(response(forged))

    with pytest.raises(ChartDecisionError, match="invalid chart intent") as error:
        ChartIntentClassifier(client=client).classify("Plot revenue trend.")
    assert isinstance(error.value.__cause__, ValidationError)


def test_client_failure_preserves_original_cause() -> None:
    original = RuntimeError("client unavailable")
    client = FakeClient(error=original)

    with pytest.raises(ChartDecisionError, match="request failed") as error:
        ChartIntentClassifier(client=client).classify("Plot revenue trend.")
    assert error.value.__cause__ is original


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("plot revenue", True),
        ("make a chart", True),
        ("graph revenue", True),
        ("visualize revenue", True),
        ("PLOT revenue", True),
        ("Make a ChArT", True),
        ("GRAPH revenue", True),
        ("VISUALIZE revenue", True),
        ("show revenue", False),
        ("display revenue", False),
        ("list revenue", False),
        ("return revenue", False),
        ("subplot revenue", False),
        ("flowchart revenue", False),
        ("infographic revenue", False),
        ("visualized revenue", False),
        ("plotting revenue", False),
    ],
)
def test_explicit_visualization_detector_uses_only_whole_words(
    question: str, expected: bool
) -> None:
    assert has_explicit_visualization_request(question) is expected


@pytest.mark.parametrize(
    ("intent", "explicit", "rows", "expected"),
    [
        ("trend", False, 0, "none"),
        ("trend", False, 1, "none"),
        ("trend", False, 2, "chart"),
        ("trend", True, 1, "none"),
        ("trend", True, 2, "chart"),
        ("comparison", False, 1, "none"),
        ("comparison", False, 2, "chart"),
        ("ranking", False, 1, "none"),
        ("ranking", False, 2, "chart"),
        ("lookup", True, 1, "chart"),
        ("lookup", False, 1, "none"),
        ("list", False, 2, "none"),
        ("list", True, 2, "chart"),
        ("other", False, 2, "none"),
        ("other", True, 2, "chart"),
        ("lookup", True, 0, "none"),
        ("list", True, 0, "none"),
        ("other", True, 0, "none"),
    ],
)
def test_policy_applies_ordered_chart_eligibility(
    intent: str, explicit: bool, rows: int, expected: str
) -> None:
    decision = ChartDecisionPolicy().decide(
        ChartIntentResponse(intent=intent),
        sql_result(rows),
        explicit_visualization=explicit,
    )

    assert decision == expected


def test_multiple_numeric_columns_do_not_force_list_chart() -> None:
    intent = ChartIntentResponse(intent="list")
    assert ChartDecisionPolicy().decide(
        intent, sql_result(2, multiple_metrics=True), explicit_visualization=False
    ) == "none"


def test_policy_uses_actual_rows_not_declared_row_count_and_mutates_nothing() -> None:
    intent = ChartIntentResponse(intent="trend")
    data = sql_result(1)
    before_data = deepcopy(data)
    before_intent = intent.model_copy(deep=True)

    assert ChartDecisionPolicy().decide(intent, data, explicit_visualization=True) == "none"
    assert data == before_data
    assert intent == before_intent


def test_policy_does_not_call_model(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_model(*args, **kwargs):
        raise AssertionError("policy must not construct an LLM client")

    monkeypatch.setattr(chart_decision, "OpenAI", unexpected_model)
    assert ChartDecisionPolicy().decide(
        ChartIntentResponse(intent="comparison"),
        sql_result(2),
        explicit_visualization=False,
    ) == "chart"


@pytest.mark.parametrize(
    ("question", "intent", "rows", "expected"),
    [
        ("Plot Apple's revenue trend.", "trend", 1, "none"),
        ("Plot Apple's 2025 revenue.", "lookup", 1, "chart"),
        ("Show Apple's 2025 revenue.", "lookup", 1, "none"),
        ("List Apple and Microsoft revenue rows.", "list", 2, "none"),
    ],
)
def test_detector_and_policy_compose_without_model_decision(
    question: str, intent: str, rows: int, expected: str
) -> None:
    assert ChartDecisionPolicy().decide(
        ChartIntentResponse(intent=intent),
        sql_result(rows),
        explicit_visualization=has_explicit_visualization_request(question),
    ) == expected
