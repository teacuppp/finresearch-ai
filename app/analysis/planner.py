import json
from typing import Annotated, Literal, TypeAlias

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

from app.agent.sql_executor import SQLQueryResult
from app.analysis.financial_analyzer import AnalysisOperation, RankingDirection


PLAN_SYSTEM_PROMPT = """
You plan one deterministic financial analysis operation. Choose exactly one:
absolute_change, percentage_change, difference, or ranking.

Return only the structured plan. Do not calculate a numeric result, explain
your choice, include reasoning, write Python code, or invent a formula.

Use only row indices and column names present in the supplied SQL result.
Rows are in their original order and row indices are zero-based.

absolute_change uses new - old.
percentage_change uses (new - old) / old * 100.
difference uses left - right.
ranking sorts all rows by one numeric column; preserve the direction
requested by the user (ascending or descending).
""".strip()


class AnalysisPlanningError(RuntimeError):
    """A valid analysis plan could not be produced for the SQL result."""


class _PlanModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AbsoluteChangePlan(_PlanModel):
    operation: Literal[AnalysisOperation.ABSOLUTE_CHANGE]
    old_row: int = Field(ge=0, strict=True)
    old_column: str = Field(min_length=1)
    new_row: int = Field(ge=0, strict=True)
    new_column: str = Field(min_length=1)


class PercentageChangePlan(_PlanModel):
    operation: Literal[AnalysisOperation.PERCENTAGE_CHANGE]
    old_row: int = Field(ge=0, strict=True)
    old_column: str = Field(min_length=1)
    new_row: int = Field(ge=0, strict=True)
    new_column: str = Field(min_length=1)


class DifferencePlan(_PlanModel):
    operation: Literal[AnalysisOperation.DIFFERENCE]
    left_row: int = Field(ge=0, strict=True)
    left_column: str = Field(min_length=1)
    right_row: int = Field(ge=0, strict=True)
    right_column: str = Field(min_length=1)


class RankingPlan(_PlanModel):
    operation: Literal[AnalysisOperation.RANKING]
    column: str = Field(min_length=1)
    direction: RankingDirection


AnalysisPlan: TypeAlias = Annotated[
    AbsoluteChangePlan | PercentageChangePlan | DifferencePlan | RankingPlan,
    Field(discriminator="operation"),
]


class AnalysisPlanResponse(_PlanModel):
    plan: AnalysisPlan


class AnalysisPlanner:
    def __init__(
        self,
        model: str = "qwen3:4b",
        base_url: str = "http://localhost:11434/v1/",
        api_key: str = "ollama",
        client: OpenAI | None = None,
    ):
        self.model = model
        self.client = (
            client
            if client is not None
            else OpenAI(base_url=base_url, api_key=api_key)
        )

    def plan(
        self,
        question: str,
        sql_result: SQLQueryResult,
    ) -> AnalysisPlan:
        if not isinstance(question, str) or not question.strip():
            raise AnalysisPlanningError("question must not be empty")
        if not sql_result.rows:
            raise AnalysisPlanningError("SQL result has no rows")

        try:
            result_json = json.dumps(
                {
                    "columns": sql_result.columns,
                    "row_count": sql_result.row_count,
                    "numbered_rows": [
                        {"row_index": index, "values": row}
                        for index, row in enumerate(sql_result.rows)
                    ],
                },
            )
        except (TypeError, ValueError) as exc:
            raise AnalysisPlanningError(
                "SQL result cannot be serialized for planning."
            ) from exc

        user_prompt = (
            f"Original question:\n{question}\n\n"
            f"SQL result (original row order):\n{result_json}"
        )
        try:
            response = self.client.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": PLAN_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=AnalysisPlanResponse,
                temperature=0,
            )
        except Exception as exc:
            raise AnalysisPlanningError(
                "Analysis planning request failed."
            ) from exc

        try:
            parsed = response.choices[0].message.parsed
        except (AttributeError, IndexError, KeyError, TypeError) as exc:
            raise AnalysisPlanningError(
                "Model returned an invalid structured planning response."
            ) from exc

        if parsed is None:
            raise AnalysisPlanningError("Model returned no parsed analysis plan.")
        if not isinstance(parsed, AnalysisPlanResponse):
            raise AnalysisPlanningError("Model returned an unexpected plan type.")

        try:
            plan = parsed.plan
        except AttributeError as exc:
            raise AnalysisPlanningError("Model returned an incomplete plan.") from exc

        self._validate_references(plan, sql_result)
        return plan

    @classmethod
    def _validate_references(
        cls,
        plan: object,
        sql_result: SQLQueryResult,
    ) -> None:
        if isinstance(plan, AbsoluteChangePlan):
            cls._check_operation(plan, AnalysisOperation.ABSOLUTE_CHANGE)
            cls._check_cell(sql_result, plan.old_row, plan.old_column)
            cls._check_cell(sql_result, plan.new_row, plan.new_column)
        elif isinstance(plan, PercentageChangePlan):
            cls._check_operation(plan, AnalysisOperation.PERCENTAGE_CHANGE)
            cls._check_cell(sql_result, plan.old_row, plan.old_column)
            cls._check_cell(sql_result, plan.new_row, plan.new_column)
        elif isinstance(plan, DifferencePlan):
            cls._check_operation(plan, AnalysisOperation.DIFFERENCE)
            cls._check_cell(sql_result, plan.left_row, plan.left_column)
            cls._check_cell(sql_result, plan.right_row, plan.right_column)
        elif isinstance(plan, RankingPlan):
            cls._check_operation(plan, AnalysisOperation.RANKING)
            if not isinstance(plan.direction, RankingDirection):
                raise AnalysisPlanningError("Plan has an invalid ranking direction.")
            cls._check_column(sql_result, plan.column)
            for row in sql_result.rows:
                if plan.column not in row:
                    raise AnalysisPlanningError(
                        f"Plan column {plan.column!r} is missing from a result row."
                    )
        else:
            raise AnalysisPlanningError("Model returned an unexpected plan type.")

    @staticmethod
    def _check_operation(plan: object, expected: AnalysisOperation) -> None:
        if getattr(plan, "operation", None) != expected:
            raise AnalysisPlanningError("Plan operation does not match its schema.")

    @classmethod
    def _check_cell(
        cls,
        sql_result: SQLQueryResult,
        row_index: int,
        column: str,
    ) -> None:
        if (
            isinstance(row_index, bool)
            or not isinstance(row_index, int)
            or not 0 <= row_index < len(sql_result.rows)
        ):
            raise AnalysisPlanningError(
                f"Plan row index {row_index!r} is out of range."
            )
        cls._check_column(sql_result, column)
        if column not in sql_result.rows[row_index]:
            raise AnalysisPlanningError(
                f"Plan column {column!r} is missing from row {row_index}."
            )

    @staticmethod
    def _check_column(sql_result: SQLQueryResult, column: str) -> None:
        if not isinstance(column, str) or column not in sql_result.columns:
            raise AnalysisPlanningError(f"Plan column {column!r} does not exist.")
