
# 我们可以容忍**单个纯 SQL code fence**，但随后仍然走同一个 read-only validator。

# ---

# ## 3. `tests/test_sql_generator.py`

# 这里仍然不调用真实 Ollama。

# Unit tests 必须 deterministic，所以 mock OpenAI client。

# 完整写成：

# ```python
from types import SimpleNamespace

import pytest

from app.agent.sql_generator import (
    SQLGenerationError,
    SQLGenerator,
    extract_sql,
)
from app.agent.schema import (
    FINANCIAL_SCHEMA,
)


def _response(
    content: str | None,
):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=content
                )
            )
        ]
    )


class FakeCompletions:
    def __init__(
        self,
        response_content: str | None,
    ):
        self.response_content = (
            response_content
        )

        self.calls = []

    def create(
        self,
        **kwargs,
    ):
        self.calls.append(
            kwargs
        )

        return _response(
            self.response_content
        )


class FakeClient:
    def __init__(
        self,
        response_content: str | None,
    ):
        self.completions = (
            FakeCompletions(
                response_content
            )
        )

        self.chat = SimpleNamespace(
            completions=self.completions
        )


def _generator_with_response(
    response_content: str | None,
) -> tuple[
    SQLGenerator,
    FakeClient,
]:
    generator = SQLGenerator()

    fake_client = FakeClient(
        response_content
    )

    generator.client = fake_client

    return (
        generator,
        fake_client,
    )


def test_extract_sql_accepts_select():
    result = extract_sql(
        """
        SELECT net_income_musd
        FROM financial_metrics
        WHERE ticker = 'MSFT'
          AND fiscal_year = 2025
        """
    )

    assert result.startswith(
        "SELECT"
    )

    assert (
        "net_income_musd"
        in result
    )


def test_extract_sql_removes_sql_fence():
    result = extract_sql(
        """
        ```sql
        SELECT revenue_musd
        FROM financial_metrics
        WHERE ticker = 'AAPL'
        ```
        """
    )

    assert result == (
        "SELECT revenue_musd\n"
        "        FROM financial_metrics\n"
        "        WHERE ticker = 'AAPL'"
    )


def test_extract_sql_rejects_empty_response():
    with pytest.raises(
        SQLGenerationError,
        match="empty SQL response",
    ):
        extract_sql(
            "   "
        )


def test_extract_sql_rejects_write_sql():
    with pytest.raises(
        SQLGenerationError,
        match=(
            "failed validation"
        ),
    ):
        extract_sql(
            """
            DELETE FROM financial_metrics
            """
        )


def test_extract_sql_rejects_multiple_statements():
    with pytest.raises(
        SQLGenerationError,
        match=(
            "failed validation"
        ),
    ):
        extract_sql(
            """
            SELECT 1;
            SELECT 2;
            """
        )


def test_generator_returns_validated_sql():
    generator, fake_client = (
        _generator_with_response(
            """
            SELECT net_income_musd
            FROM financial_metrics
            WHERE ticker = 'MSFT'
              AND fiscal_year = 2025
            """
        )
    )

    result = generator.generate(
        question=(
            "What was Microsoft's "
            "net income in 2025?"
        ),
        schema=FINANCIAL_SCHEMA,
    )

    assert (
        "net_income_musd"
        in result
    )

    assert (
        "MSFT"
        in result
    )

    assert (
        "2025"
        in result
    )

    assert len(
        fake_client
        .completions
        .calls
    ) == 1


def test_generator_uses_zero_temperature():
    generator, fake_client = (
        _generator_with_response(
            """
            SELECT revenue_musd
            FROM financial_metrics
            """
        )
    )

    generator.generate(
        question=(
            "Show revenue."
        ),
        schema=FINANCIAL_SCHEMA,
    )

    call = (
        fake_client
        .completions
        .calls[0]
    )

    assert (
        call["temperature"]
        == 0
    )


def test_generator_includes_schema_and_question():
    generator, fake_client = (
        _generator_with_response(
            """
            SELECT revenue_musd
            FROM financial_metrics
            """
        )
    )

    question = (
        "What was Apple's revenue "
        "in 2025?"
    )

    generator.generate(
        question=question,
        schema=FINANCIAL_SCHEMA,
    )

    call = (
        fake_client
        .completions
        .calls[0]
    )

    messages = call[
        "messages"
    ]

    assert len(messages) == 2

    assert (
        "financial_metrics"
        in messages[1]["content"]
    )

    assert (
        question
        in messages[1]["content"]
    )


def test_generator_rejects_empty_question():
    generator, _ = (
        _generator_with_response(
            "SELECT 1"
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "question must not be empty"
        ),
    ):
        generator.generate(
            question="   ",
            schema=FINANCIAL_SCHEMA,
        )


def test_generator_rejects_empty_schema():
    generator, _ = (
        _generator_with_response(
            "SELECT 1"
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "schema must not be empty"
        ),
    ):
        generator.generate(
            question="Show revenue",
            schema="   ",
        )


def test_generator_rejects_none_content():
    generator, _ = (
        _generator_with_response(
            None
        )
    )

    with pytest.raises(
        SQLGenerationError,
        match=(
            "empty SQL response"
        ),
    ):
        generator.generate(
            question="Show revenue",
            schema=FINANCIAL_SCHEMA,
        )


def test_generator_rejects_invalid_model_sql():
    generator, _ = (
        _generator_with_response(
            """
            DROP TABLE financial_metrics
            """
        )
    )

    with pytest.raises(
        SQLGenerationError,
        match=(
            "failed validation"
        ),
    ):
        generator.generate(
            question=(
                "Delete the table"
            ),
            schema=FINANCIAL_SCHEMA,
        )


def test_repair_returns_corrected_sql():
    generator, fake_client = (
        _generator_with_response(
            """
            SELECT net_income_musd
            FROM financial_metrics
            WHERE ticker = 'MSFT'
              AND fiscal_year = 2025
            """
        )
    )

    result = generator.repair(
        question=(
            "What was Microsoft's "
            "net income in 2025?"
        ),
        schema=FINANCIAL_SCHEMA,
        previous_sql=(
            """
            SELECT net_income
            FROM financial_metrics
            WHERE ticker = 'MSFT'
            """
        ),
        error_message=(
            "no such column: "
            "net_income"
        ),
    )

    assert (
        "net_income_musd"
        in result
    )

    call = (
        fake_client
        .completions
        .calls[0]
    )

    user_prompt = (
        call["messages"][1][
            "content"
        ]
    )

    assert (
        "net_income"
        in user_prompt
    )

    assert (
        "no such column"
        in user_prompt
    )


@pytest.mark.parametrize(
    (
        "field",
        "value",
        "message",
    ),
    [
        (
            "question",
            "   ",
            "question must not be empty",
        ),
        (
            "schema",
            "   ",
            "schema must not be empty",
        ),
        (
            "previous_sql",
            "   ",
            (
                "previous_sql must not "
                "be empty"
            ),
        ),
        (
            "error_message",
            "   ",
            (
                "error_message must not "
                "be empty"
            ),
        ),
    ],
)
def test_repair_rejects_empty_inputs(
    field,
    value,
    message,
):
    generator, _ = (
        _generator_with_response(
            "SELECT 1"
        )
    )

    arguments = {
        "question": (
            "Show Microsoft revenue"
        ),
        "schema": FINANCIAL_SCHEMA,
        "previous_sql": (
            "SELECT revenue_musd "
            "FROM financial_metrics"
        ),
        "error_message": (
            "test error"
        ),
    }

    arguments[field] = value

    with pytest.raises(
        ValueError,
        match=message,
    ):
        generator.repair(
            **arguments
        )