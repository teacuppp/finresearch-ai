"""Structured chart selection over an existing SQL result."""

import json

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

from app.agent.sql_executor import SQLQueryResult
from app.analysis.chart_renderer import ChartSpec, ChartType


CHART_SYSTEM_PROMPT = """
Choose one chart for the supplied question and chart-ready SQL rows.
Return only the structured chart specification using existing column names.

Use bar to compare or rank discrete companies, tickers, or other categories.
Prefer a human-readable company column for the x axis when available; ticker
is also valid. Use line for a change or trend over an ordered numeric dimension
such as fiscal_year, and prefer fiscal_year for that x axis when appropriate.
Do not choose line just because there are two numeric columns, or bar just
because there are two rows. Consider the question and available columns.

Choose a financial numeric metric for the y axis. Do not use company, ticker,
or fiscal_year as the y metric when a financial metric is available. Use only
columns present in the supplied SQL result; do not invent columns.

Rows are numbered in their current order. Preserve that order. Do not sort,
rank, reorder, filter, aggregate, or transform rows. Do not calculate a
percentage change, difference, ranking, or any other numeric answer.
Do not generate or execute Python code or render a chart.

Write a concise, human-readable title without a numeric answer or unsupported
factual claim. Axis labels may be null if the source column names are clear.
Title and axis labels are plain text values: include no JSON syntax, braces,
or surrounding quotation marks inside those strings.
Return no reasoning, confidence, colors, themes, formulas, output paths,
Matplotlib arguments, or report metadata.
""".strip()


class ChartPlanningError(RuntimeError):
    """A valid chart specification could not be produced for the SQL result."""


class ChartPlanResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")

    chart_type: ChartType
    x_column: str = Field(min_length=1, strict=True)
    y_column: str = Field(min_length=1, strict=True)
    title: str = Field(min_length=1, strict=True)
    x_label: str | None = None
    y_label: str | None = None


_CONTEXT_COLUMNS = frozenset(
    {"company", "company_name", "ticker", "symbol", "fiscal_year", "year", "quarter", "id"}
)


class ChartPlanner:
    def __init__(
        self,
        model: str = "qwen3:4b",
        base_url: str = "http://localhost:11434/v1/",
        api_key: str = "ollama",
        client: OpenAI | None = None,
    ) -> None:
        self.model = model
        self.client = (
            client
            if client is not None
            else OpenAI(base_url=base_url, api_key=api_key, timeout=30.0)
        )

    def plan(self, question: str, sql_result: SQLQueryResult) -> ChartSpec:
        if not isinstance(question, str) or not question.strip():
            raise ChartPlanningError("question must not be empty")
        if not sql_result.rows:
            raise ChartPlanningError("SQL result has no rows")

        try:
            result_json = json.dumps(
                {
                    "columns": sql_result.columns,
                    "row_count": sql_result.row_count,
                    "numbered_rows": [
                        {"row_index": index, "values": row}
                        for index, row in enumerate(sql_result.rows)
                    ],
                }
            )
        except (TypeError, ValueError) as exc:
            raise ChartPlanningError(
                "SQL result cannot be serialized for chart planning."
            ) from exc

        user_prompt = (
            f"Original question:\n{question}\n\n"
            f"SQL result (original row order):\n{result_json}"
        )
        try:
            response = self.client.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": CHART_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=ChartPlanResponse,
                temperature=0,
                reasoning_effort="none",
            )
        except Exception as exc:
            raise ChartPlanningError("Chart planning request failed.") from exc

        try:
            parsed = response.choices[0].message.parsed
        except (AttributeError, IndexError, KeyError, TypeError) as exc:
            raise ChartPlanningError(
                "Model returned an invalid structured chart response."
            ) from exc

        if parsed is None:
            raise ChartPlanningError("Model returned no parsed chart specification.")
        if not isinstance(parsed, ChartPlanResponse):
            raise ChartPlanningError("Model returned an unexpected chart response type.")

        try:
            validated = ChartPlanResponse.model_validate(parsed)
        except Exception as exc:
            raise ChartPlanningError("Model returned an invalid chart specification.") from exc

        self._validate_references(validated, sql_result)
        return ChartSpec(
            chart_type=validated.chart_type,
            x_column=validated.x_column,
            y_column=validated.y_column,
            title=validated.title,
            x_label=validated.x_label,
            y_label=validated.y_label,
        )

    @staticmethod
    def _validate_references(
        response: ChartPlanResponse, sql_result: SQLQueryResult
    ) -> None:
        if not isinstance(response.chart_type, ChartType):
            raise ChartPlanningError("Chart type must be bar or line.")
        for name in ("x_column", "y_column", "title", "x_label", "y_label"):
            value = getattr(response, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ChartPlanningError(f"{name} must not be blank")
        for name in ("title", "x_label", "y_label"):
            value = getattr(response, name)
            if value is not None:
                ChartPlanner._validate_human_text(name, value)
        if response.x_column == response.y_column:
            raise ChartPlanningError("Chart axes must use different columns.")
        for column in (response.x_column, response.y_column):
            if column not in sql_result.columns:
                raise ChartPlanningError(f"Chart column {column!r} does not exist.")
            for index, row in enumerate(sql_result.rows):
                if column not in row:
                    raise ChartPlanningError(
                        f"Chart column {column!r} is missing from row {index}."
                    )

        if response.y_column.lower() in _CONTEXT_COLUMNS:
            has_metric = any(
                column.lower() not in _CONTEXT_COLUMNS
                and all(
                    column in row
                    and isinstance(row[column], (int, float))
                    and not isinstance(row[column], bool)
                    for row in sql_result.rows
                )
                for column in sql_result.columns
            )
            if has_metric:
                raise ChartPlanningError(
                    "Chart y_column must select a financial metric."
                )

    @staticmethod
    def _validate_human_text(name: str, value: str) -> None:
        text = value.strip()
        if "{" in text or "}" in text or text.startswith('"') or text.endswith('"'):
            raise ChartPlanningError(f"{name} contains structured-output artifacts.")
