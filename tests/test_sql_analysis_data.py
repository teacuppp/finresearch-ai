from types import SimpleNamespace

import pytest

from app.agent.schema import FINANCIAL_SCHEMA
from app.agent.sql_executor import SQLValidationError, validate_read_only_sql
from app.agent.sql_generator import (
    ANALYSIS_DATA_SQL_SYSTEM_PROMPT,
    SQL_SYSTEM_PROMPT,
    SQLGenerationError,
    SQLGenerator,
)
from app.analysis.financial_analyzer import AnalysisOperation


class FakeCompletions:
    def __init__(self, content: str | None):
        self.response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )
        self.error: Exception | None = None
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class FakeClient:
    def __init__(self, content: str | None):
        self.completions = FakeCompletions(content)
        self.chat = SimpleNamespace(completions=self.completions)


def _generator_with_response(content: str | None) -> tuple[SQLGenerator, FakeClient]:
    generator = SQLGenerator()
    client = FakeClient(content)
    generator.client = client
    return generator, client


@pytest.mark.parametrize(
    "question, operation, sql, required_fields",
    [
        pytest.param(
            "By what percentage did Apple's revenue change from 2024 to 2025?",
            AnalysisOperation.PERCENTAGE_CHANGE,
            "SELECT ticker, fiscal_year, revenue_musd "
            "FROM financial_metrics WHERE ticker = 'AAPL' "
            "AND fiscal_year IN (2024, 2025) ORDER BY fiscal_year",
            ("ticker", "fiscal_year", "revenue_musd"),
            id="percentage_change",
        ),
        pytest.param(
            "How much did Apple's revenue increase from 2024 to 2025?",
            AnalysisOperation.ABSOLUTE_CHANGE,
            "SELECT ticker, fiscal_year, revenue_musd "
            "FROM financial_metrics WHERE ticker = 'AAPL' "
            "AND fiscal_year IN (2024, 2025) ORDER BY fiscal_year",
            ("ticker", "fiscal_year", "revenue_musd"),
            id="absolute_change",
        ),
        pytest.param(
            "What is the difference between Apple's and Microsoft's 2025 revenue?",
            AnalysisOperation.DIFFERENCE,
            "SELECT ticker, fiscal_year, revenue_musd "
            "FROM financial_metrics WHERE ticker IN ('AAPL', 'MSFT') "
            "AND fiscal_year = 2025 ORDER BY ticker",
            ("ticker", "fiscal_year", "revenue_musd"),
            id="difference",
        ),
        pytest.param(
            "Rank Apple and Microsoft by 2025 revenue from highest to lowest.",
            AnalysisOperation.RANKING,
            "SELECT ticker, fiscal_year, revenue_musd "
            "FROM financial_metrics WHERE ticker IN ('AAPL', 'MSFT') "
            "AND fiscal_year = 2025",
            ("ticker", "fiscal_year", "revenue_musd"),
            id="ranking",
        ),
    ],
)
def test_analysis_data_generation_returns_raw_values_for_each_operation(
    question, operation, sql, required_fields
):
    generator, client = _generator_with_response(sql)

    result = generator.generate_analysis_data(question, FINANCIAL_SCHEMA, operation)

    assert result == sql
    for field in required_fields:
        assert field in result
    call = client.completions.calls[0]
    assert call["messages"][0]["content"] == ANALYSIS_DATA_SQL_SYSTEM_PROMPT
    assert question in call["messages"][1]["content"]
    assert FINANCIAL_SCHEMA in call["messages"][1]["content"]
    assert f"Requested analysis operation: {operation.value}" in (
        call["messages"][1]["content"]
    )


def test_analysis_data_prompt_forbids_derived_results_and_preserves_row_context():
    prompt = ANALYSIS_DATA_SQL_SYSTEM_PROMPT
    assert "exactly one read-only SELECT or WITH ... SELECT statement" in prompt
    assert "Do not compute the requested derived result in SQL" in prompt
    assert "Do not use SQL arithmetic" in prompt
    assert "Do not use ORDER BY merely to perform the requested ranking" in prompt
    assert "fiscal_year" in prompt
    assert "company/ticker" in prompt
    assert "Do not rely on implicit database row order" in prompt
    assert "Use only the supplied database schema" in prompt
    assert "Do not invent tables or columns" in prompt
    assert "Return SQL only" in prompt


def test_analysis_data_uses_existing_operation_enum_and_zero_temperature():
    assert (
        SQLGenerator.generate_analysis_data.__annotations__["operation"]
        is AnalysisOperation
    )
    generator, client = _generator_with_response("SELECT ticker FROM financial_metrics")
    generator.generate_analysis_data(
        "Rank companies by revenue.", FINANCIAL_SCHEMA, AnalysisOperation.RANKING
    )

    call = client.completions.calls[0]
    assert call["model"] == "qwen3:4b"
    assert call["temperature"] == 0
    assert len(client.completions.calls) == 1


def test_analysis_data_uses_existing_read_only_validator(monkeypatch):
    sql = "SELECT ticker, fiscal_year, revenue_musd FROM financial_metrics"
    generator, _ = _generator_with_response(sql)
    validated: list[str] = []

    def validating_spy(statement: str) -> str:
        validated.append(statement)
        return validate_read_only_sql(statement)

    monkeypatch.setattr(
        "app.agent.sql_generator.validate_read_only_sql", validating_spy
    )

    assert generator.generate_analysis_data(
        "Show values to rank.", FINANCIAL_SCHEMA, AnalysisOperation.RANKING
    ) == sql
    assert validated == [sql]


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM financial_metrics",
        "WITH updated AS (DELETE FROM financial_metrics) SELECT 1",
        "SELECT ticker FROM financial_metrics; "
        "SELECT revenue_musd FROM financial_metrics",
    ],
)
def test_analysis_data_rejects_write_or_multiple_statements(sql):
    generator, _ = _generator_with_response(sql)

    with pytest.raises(SQLGenerationError, match="failed validation") as exc:
        generator.generate_analysis_data(
            "Rank companies.", FINANCIAL_SCHEMA, AnalysisOperation.RANKING
        )

    assert isinstance(exc.value.__cause__, SQLValidationError)


def test_analysis_data_handles_sql_fence_like_direct_generation():
    generator, _ = _generator_with_response(
        "```sql\nSELECT fiscal_year, revenue_musd FROM financial_metrics\n```"
    )

    assert generator.generate_analysis_data(
        "How much did revenue rise?",
        FINANCIAL_SCHEMA,
        AnalysisOperation.ABSOLUTE_CHANGE,
    ) == "SELECT fiscal_year, revenue_musd FROM financial_metrics"


@pytest.mark.parametrize("content", [None, "", " \n "])
def test_analysis_data_rejects_empty_model_content(content):
    generator, _ = _generator_with_response(content)

    with pytest.raises(SQLGenerationError, match="empty SQL response"):
        generator.generate_analysis_data(
            "Show revenue values.",
            FINANCIAL_SCHEMA,
            AnalysisOperation.PERCENTAGE_CHANGE,
        )


@pytest.mark.parametrize(
    "question, schema, operation, message",
    [
        pytest.param(
            " ", FINANCIAL_SCHEMA, AnalysisOperation.RANKING, "question",
            id="blank_question",
        ),
        pytest.param(
            "Rank revenue.", " ", AnalysisOperation.RANKING, "schema",
            id="blank_schema",
        ),
        pytest.param(
            "Rank revenue.", FINANCIAL_SCHEMA, "ranking", "operation",
            id="invalid_operation",
        ),
    ],
)
def test_analysis_data_rejects_invalid_inputs_before_model_call(
    question, schema, operation, message
):
    generator, client = _generator_with_response("SELECT 1")

    with pytest.raises(ValueError, match=message):
        generator.generate_analysis_data(question, schema, operation)

    assert client.completions.calls == []


def test_analysis_data_api_failure_matches_direct_generation_behavior():
    generator, client = _generator_with_response(None)
    api_error = RuntimeError("Ollama unavailable")
    client.completions.error = api_error

    with pytest.raises(RuntimeError) as exc:
        generator.generate_analysis_data(
            "Show revenue.", FINANCIAL_SCHEMA, AnalysisOperation.RANKING
        )

    assert exc.value is api_error


def test_analysis_data_malformed_response_matches_direct_generation_behavior():
    generator, client = _generator_with_response(None)
    client.completions.response = SimpleNamespace(choices=[])

    with pytest.raises(IndexError):
        generator.generate_analysis_data(
            "Show revenue.", FINANCIAL_SCHEMA, AnalysisOperation.RANKING
        )


def test_direct_sql_generation_keeps_its_existing_prompt():
    generator, client = _generator_with_response(
        "SELECT revenue_musd FROM financial_metrics"
    )

    assert generator.generate("What is revenue?", FINANCIAL_SCHEMA) == (
        "SELECT revenue_musd FROM financial_metrics"
    )
    assert client.completions.calls[0]["messages"][0]["content"] == (
        SQL_SYSTEM_PROMPT
    )
