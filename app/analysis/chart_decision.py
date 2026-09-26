"""Semantic chart intent extraction and deterministic chart eligibility."""

import re
from typing import Literal, TypeAlias

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, ValidationError

from app.agent.sql_executor import SQLQueryResult


CHART_INTENT_SYSTEM_PROMPT = """
Extract the original question's underlying financial semantic intent.
Return only the structured intent field.

When assigning intent, IGNORE presentation or visualization wording such as
"plot", "chart", "graph", and "visualize". Those words do not themselves
imply trend, comparison, ranking, lookup, list, or other.

intent meanings:
- trend: trend, change, or progression across an ordered dimension.
- comparison: explicit comparison or versus between categories or entities.
- ranking: explicit ranking, order, top, or bottom request.
- lookup: factual metric or value lookup.
- list: list, display, return, or table of rows or metrics without comparison,
  ranking, or trend.
- other: none of the above.

Examples (question -> intent):
"What was Apple's 2025 revenue?" -> lookup
"Show Apple's 2025 revenue." -> lookup
"Plot Apple's 2025 revenue." -> lookup
"Chart Apple's 2025 revenue." -> lookup
"Show Apple's revenue trend." -> trend
"Plot Apple's revenue trend." -> trend
"Compare Apple and Microsoft 2025 revenue." -> comparison
"Graph Apple versus Microsoft 2025 revenue." -> comparison
"Rank Apple and Microsoft by 2025 revenue." -> ranking
"Visualize the ranking of Apple and Microsoft by 2025 revenue." -> ranking
"List Apple and Microsoft revenue rows." -> list

Do not determine explicit visualization, inspect rows or row counts, or return
chart/none. Do not answer the financial question, choose chart type, select axes,
calculate, generate
SQL or Python, or render. Do not include reasoning, confidence, or report fields.
""".strip()


_EXPLICIT_VISUALIZATION_PATTERN = re.compile(
    r"\b(?:plot|chart|graph|visualize)\b", re.IGNORECASE
)


def has_explicit_visualization_request(question: str) -> bool:
    """Detect only the four explicit visualization terms as whole words."""
    return _EXPLICIT_VISUALIZATION_PATTERN.search(question) is not None


class ChartDecisionError(RuntimeError):
    """A valid structured chart intent could not be produced."""


ChartDecision: TypeAlias = Literal["chart", "none"]
ChartSemanticIntent: TypeAlias = Literal[
    "trend", "comparison", "ranking", "lookup", "list", "other"
]


class ChartIntentResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")

    intent: ChartSemanticIntent


class ChartIntentClassifier:
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

    def classify(self, question: str) -> ChartIntentResponse:
        if not isinstance(question, str) or not question.strip():
            raise ChartDecisionError("question must not be empty")

        try:
            response = self.client.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": CHART_INTENT_SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
                response_format=ChartIntentResponse,
                temperature=0,
                reasoning_effort="none",
            )
        except Exception as exc:
            raise ChartDecisionError("Chart intent request failed.") from exc

        try:
            parsed = response.choices[0].message.parsed
        except (AttributeError, IndexError, KeyError, TypeError) as exc:
            raise ChartDecisionError(
                "Model returned an invalid structured chart intent."
            ) from exc

        if parsed is None:
            raise ChartDecisionError("Model returned no parsed chart intent.")
        if not isinstance(parsed, ChartIntentResponse):
            raise ChartDecisionError("Model returned an unexpected chart intent type.")

        try:
            return ChartIntentResponse.model_validate(parsed)
        except ValidationError as exc:
            raise ChartDecisionError("Model returned an invalid chart intent.") from exc


class ChartDecisionPolicy:
    """Apply chart eligibility rules to structured intent and actual SQL rows."""

    def decide(
        self,
        intent: ChartIntentResponse,
        sql_result: SQLQueryResult,
        explicit_visualization: bool,
    ) -> ChartDecision:
        row_count = len(sql_result.rows)

        if intent.intent == "trend":
            return "chart" if row_count >= 2 else "none"
        if intent.intent in ("comparison", "ranking"):
            return "chart" if row_count >= 2 else "none"
        if explicit_visualization and row_count > 0:
            return "chart"
        return "none"
