from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.agent.router import (
    LLMQuestionRouter,
    QuestionRoutingError,
    ROUTER_SYSTEM_PROMPT,
    RouteDecision,
)


def _response(
    parsed: object,
    content: str | None = None,
):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    parsed=parsed,
                    content=content,
                )
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
        self.calls = []

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
        self.completions = FakeCompletions(
            response=response,
            error=error,
        )
        self.chat = SimpleNamespace(
            completions=self.completions
        )


def _router_with_response(
    parsed: object,
    content: str | None = None,
) -> tuple[LLMQuestionRouter, FakeClient]:
    client = FakeClient(
        response=_response(
            parsed=parsed,
            content=content,
        )
    )
    router = LLMQuestionRouter(client=client)

    return router, client


def test_structured_rag_response_returns_rag():
    router, _ = _router_with_response(
        RouteDecision(route="rag")
    )

    assert router.route("What risks did Apple disclose?") == "rag"


def test_structured_sql_response_returns_sql():
    router, _ = _router_with_response(
        RouteDecision(route="sql")
    )

    assert router.route("What was Apple's revenue in 2025?") == "sql"


def test_original_question_is_sent_to_model():
    router, client = _router_with_response(
        RouteDecision(route="sql")
    )
    question = "  Compare Apple and Microsoft net income in 2025.  "

    router.route(question)

    call = client.completions.calls[0]
    messages = call["messages"]
    assert messages[0] == {
        "role": "system",
        "content": ROUTER_SYSTEM_PROMPT,
    }
    assert messages[1] == {
        "role": "user",
        "content": question,
    }
    assert call["model"] == "qwen3:4b"
    assert call["temperature"] == 0
    assert call["reasoning_effort"] == "none"
    assert call["response_format"] is RouteDecision


def test_routing_instructions_encode_sql_and_rag_boundary():
    prompt = ROUTER_SYSTEM_PROMPT.lower()

    for field in (
        "company",
        "ticker",
        "fiscal_year",
        "revenue_musd",
        "operating_income_musd",
        "net_income_musd",
        "gross_margin",
    ):
        assert field in prompt

    assert "fully answered" in prompt
    assert "comparisons" in prompt
    assert "filters" in prompt
    assert "ordering" in prompt
    assert "aggregations" in prompt
    assert "risk disclosures" in prompt
    assert "strategy" in prompt
    assert "management commentary" in prompt
    assert "explanations or drivers" in prompt
    assert 'choose "sql" whenever' in prompt
    assert "do not answer" in prompt


def test_route_decision_rejects_extra_fields():
    with pytest.raises(ValidationError):
        RouteDecision.model_validate(
            {
                "route": "sql",
                "reasoning": "Revenue is a structured field.",
            }
        )


def test_missing_parsed_output_raises_router_error():
    router, _ = _router_with_response(None)

    with pytest.raises(
        QuestionRoutingError,
        match="no parsed route decision",
    ):
        router.route("What risks did Apple disclose?")


def test_client_failure_is_translated_to_router_error():
    client_error = RuntimeError("Ollama is unavailable")
    client = FakeClient(error=client_error)
    router = LLMQuestionRouter(client=client)

    with pytest.raises(
        QuestionRoutingError,
        match="routing request failed",
    ) as exc_info:
        router.route("What was Apple's revenue in 2025?")

    assert exc_info.value.__cause__ is client_error


def test_unparseable_structured_output_raises_router_error():
    parse_error = ValueError("invalid JSON")
    client = FakeClient(error=parse_error)
    router = LLMQuestionRouter(client=client)

    with pytest.raises(
        QuestionRoutingError,
        match="routing request failed",
    ) as exc_info:
        router.route("What risks did Apple disclose?")

    assert exc_info.value.__cause__ is parse_error


def test_natural_language_content_is_not_used_as_fallback():
    router, _ = _router_with_response(
        parsed=None,
        content="The correct route is sql.",
    )

    with pytest.raises(
        QuestionRoutingError,
        match="no parsed route decision",
    ):
        router.route("What was Apple's revenue in 2025?")


def test_unstructured_parsed_value_is_rejected():
    router, _ = _router_with_response(
        parsed="The correct route is rag.",
    )

    with pytest.raises(
        QuestionRoutingError,
        match="invalid structured route decision",
    ):
        router.route("What risks did Apple disclose?")


def test_malformed_response_shape_raises_router_error():
    client = FakeClient(
        response=SimpleNamespace(choices=[])
    )
    router = LLMQuestionRouter(client=client)

    with pytest.raises(
        QuestionRoutingError,
        match="invalid structured routing response",
    ):
        router.route("What risks did Apple disclose?")


def test_empty_question_is_rejected_without_calling_model():
    router, client = _router_with_response(
        RouteDecision(route="rag")
    )

    with pytest.raises(
        ValueError,
        match="question must not be empty",
    ):
        router.route("   ")

    assert client.completions.calls == []
