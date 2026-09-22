import re

from openai import OpenAI

from app.agent.sql_executor import (
    SQLValidationError,
    validate_read_only_sql,
)


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

    def repair(
        self,
        question: str,
        schema: str,
        previous_sql: str,
        error_message: str,
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