from typing import Annotated, Literal, TypeAlias

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.analysis.financial_analyzer import AnalysisOperation


SQL_ANALYSIS_SYSTEM_PROMPT = """
Classify a question after the existing top-level router has selected SQL.
Choose only the SQL task mode: "direct" or "analysis".

"analysis" means the Python FinancialAnalyzer should perform exactly one of
these four supported deterministic operations:
- absolute_change: the amount of change from an old value to a new value.
- percentage_change: the percentage change from an old value to a new value.
- difference: the difference between a left value and a right value.
- ranking: sort rows by one numeric metric in ascending or descending order.

"direct" means the existing SQL generator should answer the question itself.
Use direct for metric lookups and for unsupported operations, including
average / mean, sum, count, min/max aggregation, and ratios other than the
supported percentage_change. Arithmetic alone does not imply analysis.
Choose analysis ONLY for a request for one of the four supported operations.

Do not answer the financial question, generate SQL, or perform calculations.
Return only the structured decision. Do not include reasoning, confidence,
formulas, Python code, or chart/report fields.

Examples (decision inside the response wrapper):
- "What was Apple's revenue in 2025?" -> {"mode": "direct"}
- "Show Microsoft's 2025 net income." -> {"mode": "direct"}
- "By what percentage did Apple's revenue change from 2024 to 2025?"
  -> {"mode": "analysis", "operation": "percentage_change"}
- "How much did Apple's revenue increase from 2024 to 2025?"
  -> {"mode": "analysis", "operation": "absolute_change"}
- "What is the difference between Apple's and Microsoft's 2025 revenue?"
  -> {"mode": "analysis", "operation": "difference"}
- "Rank Apple and Microsoft by 2025 revenue from highest to lowest."
  -> {"mode": "analysis", "operation": "ranking"}
- "What was the average revenue of Apple and Microsoft in 2025?"
  -> {"mode": "direct"}
- "How many financial metric rows exist?" -> {"mode": "direct"}
""".strip()


class SQLAnalysisClassificationError(RuntimeError):
    """A valid SQL analysis-intent decision could not be produced."""


class _SQLTaskModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, revalidate_instances="always"
    )


class DirectSQLTask(_SQLTaskModel):
    mode: Literal["direct"]


class AnalysisSQLTask(_SQLTaskModel):
    mode: Literal["analysis"]
    operation: AnalysisOperation


SQLTaskDecision: TypeAlias = Annotated[
    DirectSQLTask | AnalysisSQLTask,
    Field(discriminator="mode"),
]


class SQLTaskResponse(_SQLTaskModel):
    decision: SQLTaskDecision


class SQLAnalysisClassifier:
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
            else OpenAI(base_url=base_url, api_key=api_key)
        )

    def classify(self, question: str) -> SQLTaskDecision:
        if not isinstance(question, str) or not question.strip():
            raise SQLAnalysisClassificationError("question must not be empty")

        try:
            response = self.client.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": SQL_ANALYSIS_SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
                response_format=SQLTaskResponse,
                temperature=0,
            )
        except Exception as exc:
            raise SQLAnalysisClassificationError(
                "SQL analysis classification request failed."
            ) from exc

        try:
            parsed = response.choices[0].message.parsed
        except (AttributeError, IndexError, KeyError, TypeError) as exc:
            raise SQLAnalysisClassificationError(
                "Model returned an invalid structured classification response."
            ) from exc

        if parsed is None:
            raise SQLAnalysisClassificationError(
                "Model returned no parsed SQL task decision."
            )
        if not isinstance(parsed, SQLTaskResponse):
            raise SQLAnalysisClassificationError(
                "Model returned an unexpected structured classification output."
            )

        try:
            validated = SQLTaskResponse.model_validate(parsed)
        except ValidationError as exc:
            raise SQLAnalysisClassificationError(
                "Model returned an invalid structured SQL task decision."
            ) from exc
        return validated.decision
