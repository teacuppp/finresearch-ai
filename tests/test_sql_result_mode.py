from types import SimpleNamespace

import pytest

import app.agent.sql_generator as sql_generator
from app.agent.schema import FINANCIAL_SCHEMA
from app.agent.sql_executor import SQLValidationError, validate_read_only_sql
from app.agent.sql_generator import (
    ANALYSIS_DATA_SQL_REPAIR_SYSTEM_PROMPT,
    ANALYSIS_DATA_SQL_SYSTEM_PROMPT,
    CHART_READY_SQL_REPAIR_SYSTEM_PROMPT,
    CHART_READY_SQL_SYSTEM_PROMPT,
    SQL_SYSTEM_PROMPT,
    SQLGenerationError,
    SQLGenerator,
    SQLResultMode,
)
from app.analysis.financial_analyzer import AnalysisOperation


class FakeCompletions:
    def __init__(self, content: str | None):
        self.content = content
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))]
        )


def generator_with_sql(content: str | None) -> tuple[SQLGenerator, FakeCompletions]:
    generator = SQLGenerator()
    completions = FakeCompletions(content)
    generator.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return generator, completions


def system_prompt(completions: FakeCompletions) -> str:
    return completions.calls[0]["messages"][0]["content"]


def test_result_mode_type_is_closed_and_keyword_only() -> None:
    assert SQLResultMode.__args__ == ("answer", "chart_ready")
    generator, completions = generator_with_sql("SELECT revenue_musd FROM financial_metrics")

    with pytest.raises(TypeError):
        generator.generate("What was revenue?", FINANCIAL_SCHEMA, "chart_ready")
    with pytest.raises(TypeError):
        generator.repair("What was revenue?", FINANCIAL_SCHEMA, "SELECT 1", "error", "chart_ready")
    assert completions.calls == []


@pytest.mark.parametrize("result_mode", [None, "chart", "analysis", 1])
def test_invalid_generation_mode_is_rejected_before_client_call(result_mode: object) -> None:
    generator, completions = generator_with_sql("SELECT 1")

    with pytest.raises(ValueError, match="unsupported result_mode"):
        generator.generate("What was revenue?", FINANCIAL_SCHEMA, result_mode=result_mode)
    assert completions.calls == []


@pytest.mark.parametrize("result_mode", [None, "chart", "analysis", 1])
def test_invalid_repair_mode_is_rejected_before_client_call(result_mode: object) -> None:
    generator, completions = generator_with_sql("SELECT 1")

    with pytest.raises(ValueError, match="unsupported result_mode"):
        generator.repair(
            "What was revenue?", FINANCIAL_SCHEMA, "SELECT 1", "no such column",
            result_mode=result_mode,
        )
    assert completions.calls == []


def test_default_and_explicit_answer_generation_keep_existing_prompt_and_sql() -> None:
    sql = "SELECT revenue_musd FROM financial_metrics WHERE ticker = 'AAPL'"
    generator, completions = generator_with_sql(sql)

    assert generator.generate("What was Apple's revenue?", FINANCIAL_SCHEMA) == sql
    assert generator.generate(
        "What was Apple's revenue?", FINANCIAL_SCHEMA, result_mode="answer"
    ) == sql
    assert [call["messages"][0]["content"] for call in completions.calls] == [
        SQL_SYSTEM_PROMPT, SQL_SYSTEM_PROMPT
    ]
    assert "chart-ready" not in SQL_SYSTEM_PROMPT.lower()
    assert "category/context column" not in SQL_SYSTEM_PROMPT
    assert all("reasoning_effort" not in call for call in completions.calls)


def test_chart_ready_generation_uses_dedicated_shape_contract() -> None:
    sql = "SELECT company, revenue_musd FROM financial_metrics"
    generator, completions = generator_with_sql(sql)

    assert generator.generate("Compare revenue.", FINANCIAL_SCHEMA, result_mode="chart_ready") == sql
    assert system_prompt(completions) == CHART_READY_SQL_SYSTEM_PROMPT
    prompt = CHART_READY_SQL_SYSTEM_PROMPT
    for phrase in (
        "exactly one read-only SELECT or WITH ... SELECT",
        "using only",
        "Do not invent columns",
        "comparisons between entities or categories",
        "category/context column",
        "preferably company",
        "otherwise ticker",
        "requested numeric metric",
        "multi-company comparison",
        "explicit visualization of one entity and value",
        "even when the query",
        "returns just one row",
        "fiscal_year",
        "ORDER BY fiscal_year",
        "Do not SELECT *",
        "Do not reorder categories",
        "no useful chart dimension",
        "no comments, explanation, or Markdown fences",
        "choose bar or line",
        "render",
        "generate Python",
    ):
        assert phrase in prompt
    assert "reasoning_effort" not in completions.calls[0]


def test_chart_ready_generation_passes_existing_extract_and_validation(monkeypatch) -> None:
    sql = "SELECT company, revenue_musd FROM financial_metrics"
    generator, _ = generator_with_sql(f"```sql\n{sql}\n```")
    validated: list[str] = []

    def validator(statement: str) -> str:
        validated.append(statement)
        return validate_read_only_sql(statement)

    monkeypatch.setattr(sql_generator, "validate_read_only_sql", validator)
    assert generator.generate("Plot revenue.", FINANCIAL_SCHEMA, result_mode="chart_ready") == sql
    assert validated == [sql]


@pytest.mark.parametrize("sql", ["DELETE FROM financial_metrics", "SELECT 1; SELECT 2"])
def test_chart_ready_generation_still_rejects_unsafe_sql(sql: str) -> None:
    generator, _ = generator_with_sql(sql)
    with pytest.raises(SQLGenerationError, match="failed validation") as exc:
        generator.generate("Plot revenue.", FINANCIAL_SCHEMA, result_mode="chart_ready")
    assert isinstance(exc.value.__cause__, SQLValidationError)


def test_default_and_explicit_answer_repair_keep_existing_prompt_and_sql() -> None:
    sql = "SELECT revenue_musd FROM financial_metrics"
    generator, completions = generator_with_sql(sql)
    args = ("What was revenue?", FINANCIAL_SCHEMA, "SELECT revenue FROM financial_metrics", "no such column")

    assert generator.repair(*args) == sql
    assert generator.repair(*args, result_mode="answer") == sql
    assert [call["messages"][0]["content"] for call in completions.calls] == [
        SQL_SYSTEM_PROMPT, SQL_SYSTEM_PROMPT
    ]
    assert "Rewrite the SQL so that it correctly answers" in completions.calls[0]["messages"][1]["content"]


def test_chart_ready_repair_preserves_dimensions_and_error_context() -> None:
    sql = "SELECT company, revenue_musd FROM financial_metrics"
    generator, completions = generator_with_sql(sql)

    assert generator.repair(
        "Compare Apple and Microsoft revenue.",
        FINANCIAL_SCHEMA,
        "SELECT company, revenue FROM financial_metrics",
        "no such column: revenue",
        result_mode="chart_ready",
    ) == sql
    assert system_prompt(completions) == CHART_READY_SQL_REPAIR_SYSTEM_PROMPT
    prompt = CHART_READY_SQL_REPAIR_SYSTEM_PROMPT
    assert CHART_READY_SQL_SYSTEM_PROMPT in prompt
    assert "database error" in prompt
    assert "preserving the chart-ready" in prompt
    assert "Do not silently drop" in prompt
    assert "category/context column" in prompt
    assert "fiscal_year" in prompt
    assert "requested metric" in prompt
    user_prompt = completions.calls[0]["messages"][1]["content"]
    for text in ("Compare Apple and Microsoft revenue.", FINANCIAL_SCHEMA,
                 "SELECT company, revenue FROM financial_metrics", "no such column: revenue"):
        assert text in user_prompt
    assert "reasoning_effort" not in completions.calls[0]


def test_chart_ready_repair_passes_existing_extract_and_validation(monkeypatch) -> None:
    sql = "SELECT company, revenue_musd FROM financial_metrics"
    generator, _ = generator_with_sql(f"```sql\n{sql}\n```")
    validated: list[str] = []

    def validator(statement: str) -> str:
        validated.append(statement)
        return validate_read_only_sql(statement)

    monkeypatch.setattr(sql_generator, "validate_read_only_sql", validator)
    assert generator.repair(
        "Plot revenue.", FINANCIAL_SCHEMA, "SELECT company, revenue FROM financial_metrics",
        "no such column: revenue", result_mode="chart_ready"
    ) == sql
    assert validated == [sql]


@pytest.mark.parametrize("sql", ["UPDATE financial_metrics SET revenue_musd = 0", "SELECT 1; SELECT 2"])
def test_chart_ready_repair_still_rejects_unsafe_sql(sql: str) -> None:
    generator, _ = generator_with_sql(sql)
    with pytest.raises(SQLGenerationError, match="failed validation") as exc:
        generator.repair(
            "Plot revenue.", FINANCIAL_SCHEMA, "SELECT revenue FROM financial_metrics",
            "no such column", result_mode="chart_ready"
        )
    assert isinstance(exc.value.__cause__, SQLValidationError)


def test_analysis_data_generation_and_repair_prompts_remain_separate() -> None:
    generator, completions = generator_with_sql("SELECT ticker, fiscal_year, revenue_musd FROM financial_metrics")
    generator.generate_analysis_data(
        "How much did revenue change?", FINANCIAL_SCHEMA, AnalysisOperation.ABSOLUTE_CHANGE
    )
    generator.repair_analysis_data(
        "How much did revenue change?", FINANCIAL_SCHEMA, AnalysisOperation.ABSOLUTE_CHANGE,
        "SELECT ticker, fiscal_year, revenue FROM financial_metrics", "no such column"
    )
    assert [call["messages"][0]["content"] for call in completions.calls] == [
        ANALYSIS_DATA_SQL_SYSTEM_PROMPT, ANALYSIS_DATA_SQL_REPAIR_SYSTEM_PROMPT
    ]
