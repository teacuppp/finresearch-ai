import re
from typing import Literal, TypeAlias

from openai import OpenAI

from app.agent.sql_executor import (
    SQLValidationError,
    validate_read_only_sql,
)
from app.analysis.financial_analyzer import AnalysisOperation


SQL_SYSTEM_PROMPT = """
You are a financial SQL generation assistant.

Your task is to convert a user's financial question into exactly
one valid SQLite read-only query.

Rules:

1. Use only tables and columns explicitly listed in the schema.

2. Generate only SELECT queries or WITH ... SELECT queries.

3. Never generate INSERT, UPDATE, DELETE, DROP, ALTER, CREATE,
   REPLACE, PRAGMA, ATTACH, DETACH, or any other write or
   administrative statement.

4. Return SQL only.

5. Do not use Markdown code fences.

6. Do not explain the query.

7. Do not include SQL comments.

8. Generate exactly one SQL statement.

9. Use SQLite syntax.

10. Financial columns ending in `_musd` are measured in millions
    of US dollars.

11. `gross_margin` is stored as a decimal ratio.

12. Prefer ticker identifiers when the user's company can be
    clearly mapped to a ticker provided by the schema or question.

13. Do not invent tables or columns.

Example user question:
What was Microsoft's net income in 2025?

Example output:
SELECT net_income_musd
FROM financial_metrics
WHERE ticker = 'MSFT'
  AND fiscal_year = 2025
""".strip()


SQLResultMode: TypeAlias = Literal["answer", "chart_ready"]


CHART_READY_SQL_SYSTEM_PROMPT = """
You generate chart-ready SQLite queries that answer the user's financial
question while preserving the relevant dimensions in the result rows.
Generate exactly one read-only SELECT or WITH ... SELECT statement using only
tables and columns in the supplied schema. Do not invent columns.

For comparisons between entities or categories, select a human-readable
category/context column, preferably company when available, otherwise ticker
or symbol when appropriate, together with the requested numeric metric. Do
not return only the metric for a multi-company comparison.

For an explicit visualization of one entity and value, include a relevant
category/context column and the requested numeric metric even when the query
returns just one row. For example, plotting Apple's 2025 revenue should
return company or ticker together with revenue_musd, not only revenue_musd.

For a trend across fiscal years, include fiscal_year and the requested numeric
metric. Use ORDER BY fiscal_year when appropriate to make the temporal order
deterministic. Do not reorder categories merely to make a prettier chart.

Select only context and dimension columns relevant to the visualization.
Do not SELECT *. Preserve the user's financial semantics. If the schema has
no useful chart dimension, generate the best valid read-only SQL without
inventing one; later layers decide whether a chart is possible.

Use SQLite syntax. Never generate writes, schema changes, or administrative
statements. Return SQL only: no comments, explanation, or Markdown fences.
Do not compute chart metadata, choose bar or line, render, or generate Python.
""".strip()


CHART_READY_SQL_REPAIR_SYSTEM_PROMPT = f"""
{CHART_READY_SQL_SYSTEM_PROMPT}

Repair the failed query using the database error. Fix what is needed for it
to execute against the supplied schema while preserving the chart-ready
result shape. Do not silently drop the relevant category/context column or
the ordered trend dimension, such as fiscal_year, or the requested metric.
""".strip()


ANALYSIS_DATA_SQL_SYSTEM_PROMPT = """
You generate SQLite queries that supply raw rows to a deterministic financial
analyzer. Generate exactly one read-only SELECT or WITH ... SELECT statement.
Use only the supplied database schema. Do not invent tables or columns.

Retrieve the original numeric values needed for the requested analysis
operation. Do not compute the requested derived result in SQL:
- For percentage_change and absolute_change, retrieve both the old and new
  source values. Include fiscal_year and any needed company/ticker identifiers
  so each value's meaning is clear. Do not use SQL arithmetic to calculate
  the percentage or absolute change.
- For difference, retrieve each original entity value and identifying
  company/ticker columns. Do not use SQL arithmetic to calculate the difference.
- For ranking, retrieve the entity identifiers and numeric values to rank.
  Do not use ORDER BY merely to perform the requested ranking; the analyzer
  performs the final sort.

Do not rely on implicit database row order. If row order is useful for
interpreting source values, use ORDER BY on identifying fields such as
fiscal_year or ticker, not on a calculated result or the ranking metric.

Never generate INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, REPLACE, PRAGMA,
ATTACH, DETACH, or any other write or administrative statement. Use SQLite
syntax. Return SQL only, without Markdown fences, explanation, or comments.
""".strip()


ANALYSIS_DATA_SQL_REPAIR_SYSTEM_PROMPT = f"""
{ANALYSIS_DATA_SQL_SYSTEM_PROMPT}

Repair the failed analysis-data query using the database error. Change only
what is needed to make it execute against the supplied schema. Preserve the
requested operation's raw source values and identifier/context columns.
The repaired query must still supply rows to the financial analyzer; never
replace it with SQL that calculates the final answer.
""".strip()


class SQLGenerationError(RuntimeError):
    pass


def extract_sql(
    response_text: str,
) -> str:
    text = response_text.strip()

    if not text:
        raise SQLGenerationError(
            "LLM returned an empty SQL response."
        )

    fenced_match = re.fullmatch(
        r"```(?:sql)?\s*(.*?)\s*```",
        text,
        flags=(
            re.IGNORECASE
            | re.DOTALL
        ),
    )

    if fenced_match is not None:
        text = (
            fenced_match
            .group(1)
            .strip()
        )

    try:
        return validate_read_only_sql(
            text
        )
    except SQLValidationError as exc:
        raise SQLGenerationError(
            "Generated SQL failed validation: "
            f"{exc}"
        ) from exc


class SQLGenerator:
    def __init__(
        self,
        model: str = "qwen3:4b",
        base_url: str = (
            "http://localhost:11434/v1/"
        ),
        api_key: str = "ollama",
    ):
        self.model = model

        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )

    def generate(
        self,
        question: str,
        schema: str,
        *,
        result_mode: SQLResultMode = "answer",
    ) -> str:
        question = question.strip()
        schema = schema.strip()

        if not question:
            raise ValueError(
                "question must not be empty"
            )

        if not schema:
            raise ValueError(
                "schema must not be empty"
            )

        if result_mode not in ("answer", "chart_ready"):
            raise ValueError("unsupported result_mode")

        user_prompt = f"""
Database schema:

{schema}

User question:

{question}

Generate the SQLite query.
Return SQL only.
""".strip()

        response = (
            self.client
            .chat
            .completions
            .create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            SQL_SYSTEM_PROMPT
                            if result_mode == "answer"
                            else CHART_READY_SQL_SYSTEM_PROMPT
                        ),
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                temperature=0,
            )
        )

        content = (
            response
            .choices[0]
            .message
            .content
        )

        if content is None:
            raise SQLGenerationError(
                "LLM returned an empty "
                "SQL response."
            )

        return extract_sql(
            content
        )

    def generate_analysis_data(
        self,
        question: str,
        schema: str,
        operation: AnalysisOperation,
    ) -> str:
        question = question.strip()
        schema = schema.strip()

        if not question:
            raise ValueError("question must not be empty")
        if not schema:
            raise ValueError("schema must not be empty")
        if not isinstance(operation, AnalysisOperation):
            raise ValueError("operation must be an AnalysisOperation")

        user_prompt = f"""
Database schema:

{schema}

User question:

{question}

Requested analysis operation: {operation.value}

Generate the SQLite query that retrieves the raw input rows and values.
Return SQL only.
""".strip()

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": ANALYSIS_DATA_SQL_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )

        content = response.choices[0].message.content
        if content is None:
            raise SQLGenerationError("LLM returned an empty SQL response.")
        return extract_sql(content)

    def repair(
        self,
        question: str,
        schema: str,
        previous_sql: str,
        error_message: str,
        *,
        result_mode: SQLResultMode = "answer",
    ) -> str:
        question = question.strip()
        schema = schema.strip()
        previous_sql = (
            previous_sql.strip()
        )
        error_message = (
            error_message.strip()
        )

        if not question:
            raise ValueError(
                "question must not be empty"
            )

        if not schema:
            raise ValueError(
                "schema must not be empty"
            )

        if not previous_sql:
            raise ValueError(
                "previous_sql must not be empty"
            )

        if not error_message:
            raise ValueError(
                "error_message must not be empty"
            )

        if result_mode not in ("answer", "chart_ready"):
            raise ValueError("unsupported result_mode")

        user_prompt = f"""
Database schema:

{schema}

Original user question:

{question}

Previous SQL:

{previous_sql}

Database error:

{error_message}

Rewrite the SQL so that it correctly answers the original question
and executes successfully against the provided schema.

Return SQL only.
""".strip()

        response = (
            self.client
            .chat
            .completions
            .create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            SQL_SYSTEM_PROMPT
                            if result_mode == "answer"
                            else CHART_READY_SQL_REPAIR_SYSTEM_PROMPT
                        ),
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                temperature=0,
            )
        )

        content = (
            response
            .choices[0]
            .message
            .content
        )

        if content is None:
            raise SQLGenerationError(
                "LLM returned an empty "
                "SQL repair response."
            )

        return extract_sql(
            content
        )

    def repair_analysis_data(
        self,
        question: str,
        schema: str,
        operation: AnalysisOperation,
        previous_sql: str,
        error_message: str,
    ) -> str:
        question = question.strip()
        schema = schema.strip()
        previous_sql = previous_sql.strip()
        error_message = error_message.strip()

        if not question:
            raise ValueError("question must not be empty")
        if not schema:
            raise ValueError("schema must not be empty")
        if not isinstance(operation, AnalysisOperation):
            raise ValueError("operation must be an AnalysisOperation")
        if not previous_sql:
            raise ValueError("previous_sql must not be empty")
        if not error_message:
            raise ValueError("error_message must not be empty")

        user_prompt = f"""
Database schema:

{schema}

Original user question:

{question}

Requested analysis operation: {operation.value}

Previous SQL:

{previous_sql}

Database error:

{error_message}

Fix the execution error while preserving raw input rows and values for the
requested analysis operation. Return SQL only.
""".strip()

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": ANALYSIS_DATA_SQL_REPAIR_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )

        content = response.choices[0].message.content
        if content is None:
            raise SQLGenerationError("LLM returned an empty SQL repair response.")
        return extract_sql(content)
