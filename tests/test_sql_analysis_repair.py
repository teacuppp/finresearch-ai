from types import SimpleNamespace

import pytest

from app.agent.schema import FINANCIAL_SCHEMA
from app.agent.sql_executor import SQLValidationError, validate_read_only_sql
from app.agent.sql_generator import (
    ANALYSIS_DATA_SQL_REPAIR_SYSTEM_PROMPT,
    ANALYSIS_DATA_SQL_SYSTEM_PROMPT,
    SQL_SYSTEM_PROMPT,
    SQLGenerationError,
    SQLGenerator,
)
from app.analysis.financial_analyzer import AnalysisOperation


QUESTION = "By what percentage did Apple's revenue change from 2024 to 2025?"
PREVIOUS_SQL = (
    "SELECT ticker, fiscal_year, revenue FROM financial_metrics "
    "WHERE ticker = 'AAPL' AND fiscal_year IN (2024, 2025) "
    "ORDER BY fiscal_year"
)
ERROR = "no such column: revenue"
RAW_SQL = (
    "SELECT ticker, fiscal_year, revenue_musd FROM financial_metrics "
    "WHERE ticker = 'AAPL' AND fiscal_year IN (2024, 2025) "
    "ORDER BY fiscal_year"
)


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


def _repair(
    generator: SQLGenerator,
    operation: AnalysisOperation = AnalysisOperation.PERCENTAGE_CHANGE,
) -> str:
    return generator.repair_analysis_data(
        question=QUESTION,
        schema=FINANCIAL_SCHEMA,
        operation=operation,
        previous_sql=PREVIOUS_SQL,
        error_message=ERROR,
    )


@pytest.mark.parametrize(
    "operation, question, previous_sql, corrected_sql",
    [
        pytest.param(
            AnalysisOperation.PERCENTAGE_CHANGE,
            QUESTION,
            PREVIOUS_SQL,
            RAW_SQL,
            id="percentage_change",
        ),
        pytest.param(
            AnalysisOperation.ABSOLUTE_CHANGE,
            "How much did Apple's revenue increase from 2024 to 2025?",
            PREVIOUS_SQL,
            RAW_SQL,
            id="absolute_change",
        ),
        pytest.param(
            AnalysisOperation.DIFFERENCE,
            "What is the difference between Apple's and Microsoft's 2025 revenue?",
            "SELECT ticker, revenue FROM financial_metrics "
            "WHERE ticker IN ('AAPL', 'MSFT') AND fiscal_year = 2025",
            "SELECT ticker, fiscal_year, revenue_musd FROM financial_metrics "
            "WHERE ticker IN ('AAPL', 'MSFT') AND fiscal_year = 2025 "
            "ORDER BY ticker",
            id="difference",
        ),
        pytest.param(
            AnalysisOperation.RANKING,
            "Rank Apple and Microsoft by 2025 revenue from highest to lowest.",
            "SELECT ticker, revenue FROM financial_metrics "
            "WHERE ticker IN ('AAPL', 'MSFT') AND fiscal_year = 2025",
            "SELECT ticker, fiscal_year, revenue_musd FROM financial_metrics "
            "WHERE ticker IN ('AAPL', 'MSFT') AND fiscal_year = 2025",
            id="ranking",
        ),
    ],
)
def test_repair_preserves_raw_rows_for_each_operation(
    operation, question, previous_sql, corrected_sql
):
    generator, client = _generator_with_response(corrected_sql)

    result = generator.repair_analysis_data(
        question, FINANCIAL_SCHEMA, operation, previous_sql, ERROR
    )

    assert result == corrected_sql
    assert "ticker, fiscal_year, revenue_musd" in result
    assert "revenue_musd -" not in result
    call = client.completions.calls[0]
    assert call["model"] == "qwen3:4b"
    assert call["temperature"] == 0
    assert call["messages"][0]["content"] == (
        ANALYSIS_DATA_SQL_REPAIR_SYSTEM_PROMPT
    )
    user_prompt = call["messages"][1]["content"]
    assert question in user_prompt
    assert FINANCIAL_SCHEMA in user_prompt
    assert f"Requested analysis operation: {operation.value}" in user_prompt
    assert previous_sql in user_prompt
    assert ERROR in user_prompt


def test_repair_prompt_preserves_analysis_data_semantics():
    prompt = ANALYSIS_DATA_SQL_REPAIR_SYSTEM_PROMPT
    assert prompt.startswith(ANALYSIS_DATA_SQL_SYSTEM_PROMPT)
    assert "exactly one read-only SELECT or WITH ... SELECT statement" in prompt
    assert "Do not compute the requested derived result in SQL" in prompt
    assert "Do not use SQL arithmetic" in prompt
    assert "Do not use ORDER BY merely to perform the requested ranking" in prompt
    assert "fiscal_year" in prompt
    assert "company/ticker" in prompt
    assert "Use only the supplied database schema" in prompt
    assert "Change only" in prompt
    assert "raw source values and identifier/context columns" in prompt
    assert "never\nreplace it with SQL that calculates the final answer" in prompt


def test_repaired_sql_uses_existing_read_only_validator(monkeypatch):
    generator, _ = _generator_with_response(RAW_SQL)
    validated: list[str] = []

    def validating_spy(statement: str) -> str:
        validated.append(statement)
        return validate_read_only_sql(statement)

    monkeypatch.setattr(
        "app.agent.sql_generator.validate_read_only_sql", validating_spy
    )

    assert _repair(generator) == RAW_SQL
    assert validated == [RAW_SQL]


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM financial_metrics",
        "CREATE TABLE bogus (id INTEGER)",
        "PRAGMA writable_schema = ON",
        "WITH changed AS (DELETE FROM financial_metrics) SELECT 1",
        "SELECT ticker FROM financial_metrics; "
        "SELECT revenue_musd FROM financial_metrics",
        "SELECT ticker FROM financial_metrics -- comment",
        "not SQL",
    ],
)
def test_repaired_sql_rejects_unsafe_or_malformed_statements(sql):
    generator, _ = _generator_with_response(sql)

    with pytest.raises(SQLGenerationError, match="failed validation") as exc:
        _repair(generator)

    assert isinstance(exc.value.__cause__, SQLValidationError)


def test_repaired_sql_handles_single_sql_fence():
    generator, _ = _generator_with_response(f"```sql\n{RAW_SQL}\n```")

    assert _repair(generator) == RAW_SQL


@pytest.mark.parametrize("content", [None, "", " \n "])
def test_repair_rejects_empty_model_content(content):
    generator, _ = _generator_with_response(content)

    with pytest.raises(SQLGenerationError, match="empty SQL"):
        _repair(generator)


@pytest.mark.parametrize(
    "field, value, message",
    [
        pytest.param("question", " ", "question", id="question"),
        pytest.param("schema", " ", "schema", id="schema"),
        pytest.param("operation", "ranking", "operation", id="operation"),
        pytest.param("previous_sql", " ", "previous_sql", id="previous_sql"),
        pytest.param("error_message", " ", "error_message", id="error_message"),
    ],
)
def test_invalid_inputs_fail_before_model_call(field, value, message):
    generator, client = _generator_with_response(RAW_SQL)
    arguments = {
        "question": QUESTION,
        "schema": FINANCIAL_SCHEMA,
        "operation": AnalysisOperation.PERCENTAGE_CHANGE,
        "previous_sql": PREVIOUS_SQL,
        "error_message": ERROR,
    }
    arguments[field] = value

    with pytest.raises(ValueError, match=message):
        generator.repair_analysis_data(**arguments)

    assert client.completions.calls == []


def test_client_failure_matches_existing_generator_behavior():
    generator, client = _generator_with_response(None)
    failure = RuntimeError("Ollama unavailable")
    client.completions.error = failure

    with pytest.raises(RuntimeError) as exc:
        _repair(generator)

    assert exc.value is failure


def test_malformed_response_matches_existing_generator_behavior():
    generator, client = _generator_with_response(None)
    client.completions.response = SimpleNamespace(choices=[])

    with pytest.raises(IndexError):
        _repair(generator)


def test_direct_repair_and_analysis_generation_keep_their_prompts():
    generator, client = _generator_with_response(RAW_SQL)

    assert generator.repair(QUESTION, FINANCIAL_SCHEMA, PREVIOUS_SQL, ERROR) == RAW_SQL
    assert client.completions.calls[0]["messages"][0]["content"] == SQL_SYSTEM_PROMPT

    assert generator.generate_analysis_data(
        QUESTION, FINANCIAL_SCHEMA, AnalysisOperation.PERCENTAGE_CHANGE
    ) == RAW_SQL
    assert client.completions.calls[1]["messages"][0]["content"] == (
        ANALYSIS_DATA_SQL_SYSTEM_PROMPT
    )
