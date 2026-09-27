from typing import Annotated, Literal

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.analysis.financial_analyzer import AnalysisOperation


MetricColumn = Literal[
    "revenue_musd",
    "net_income_musd",
    "operating_income_musd",
    "gross_margin",
]
EntityColumn = Literal["company", "ticker"]
FiscalYear = Annotated[int, Field(strict=True, ge=1900, le=2100)]


ANALYSIS_RETRIEVAL_SYSTEM_PROMPT = """
Extract only the structured raw-data retrieval parameters for the requested
financial analysis operation. Identify whether the user names companies or
tickers, the exact entities requested, the fiscal years needed to retrieve the
original values, and one metric from revenue_musd, net_income_musd,
operating_income_musd, or gross_margin. Use company for company names and ticker
for ticker symbols. Do not invent entities, years, or columns.

For absolute_change and percentage_change, return one entity and two fiscal
years. For difference, return two entities and one fiscal year. For ranking,
return one fiscal year and either at least two named entities or no entities
when the user requests all companies. Fiscal years are raw-data retrieval years.

Do not generate SQL, calculate, rank, sort, or provide reasoning. Return only
the structured retrieval plan. Do not include confidence or chart metadata.
""".strip()


class AnalysisRetrievalPlanningError(RuntimeError):
    """A valid raw-data retrieval plan could not be produced."""


class AnalysisRetrievalPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")

    entity_column: EntityColumn
    entities: tuple[str, ...]
    fiscal_years: tuple[FiscalYear, ...]
    metric: MetricColumn


def _validated_plan(
    plan: AnalysisRetrievalPlan, operation: AnalysisOperation
) -> AnalysisRetrievalPlan:
    if not isinstance(operation, AnalysisOperation):
        raise ValueError("Unsupported analysis operation.")
    if not isinstance(plan, AnalysisRetrievalPlan):
        raise ValueError("Unexpected retrieval plan type.")

    validated = AnalysisRetrievalPlan.model_validate(plan)
    if any(not isinstance(entity, str) or not entity.strip() for entity in validated.entities):
        raise ValueError("Entities must be nonblank strings.")
    if len({entity.strip().casefold() for entity in validated.entities}) != len(
        validated.entities
    ):
        raise ValueError("Duplicate entities are not allowed.")
    if any(type(year) is not int for year in validated.fiscal_years):
        raise ValueError("Fiscal years must be integers.")
    if len(set(validated.fiscal_years)) != len(validated.fiscal_years):
        raise ValueError("Duplicate fiscal years are not allowed.")

    entity_count = len(validated.entities)
    year_count = len(validated.fiscal_years)
    if operation in (
        AnalysisOperation.ABSOLUTE_CHANGE,
        AnalysisOperation.PERCENTAGE_CHANGE,
    ):
        valid = entity_count == 1 and year_count == 2
    elif operation is AnalysisOperation.DIFFERENCE:
        valid = entity_count == 2 and year_count == 1
    else:
        valid = entity_count != 1 and year_count == 1
    if not valid:
        raise ValueError(f"Invalid retrieval cardinality for {operation.value}.")
    return validated


class AnalysisRetrievalPlanner:
    def __init__(
        self,
        model: str = "qwen3:4b",
        base_url: str = "http://localhost:11434/v1/",
        api_key: str = "ollama",
        client: OpenAI | None = None,
    ) -> None:
        self.model = model
        self.client = (
            client if client is not None else OpenAI(base_url=base_url, api_key=api_key)
        )

    def plan(
        self, question: str, operation: AnalysisOperation
    ) -> AnalysisRetrievalPlan:
        if not isinstance(question, str) or not question.strip():
            raise AnalysisRetrievalPlanningError("question must not be empty")
        if not isinstance(operation, AnalysisOperation):
            raise AnalysisRetrievalPlanningError("Unsupported analysis operation.")

        try:
            response = self.client.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": ANALYSIS_RETRIEVAL_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Original question:\n{question}\n\n"
                            f"Requested analysis operation: {operation.value}"
                        ),
                    },
                ],
                response_format=AnalysisRetrievalPlan,
                temperature=0,
                reasoning_effort="none",
            )
        except Exception as exc:
            raise AnalysisRetrievalPlanningError(
                "Analysis retrieval planning request failed."
            ) from exc

        try:
            parsed = response.choices[0].message.parsed
        except (AttributeError, IndexError, KeyError, TypeError) as exc:
            raise AnalysisRetrievalPlanningError(
                "Model returned an invalid structured retrieval response."
            ) from exc
        if parsed is None:
            raise AnalysisRetrievalPlanningError("Model returned no parsed retrieval plan.")
        try:
            return _validated_plan(parsed, operation)
        except (ValueError, ValidationError) as exc:
            raise AnalysisRetrievalPlanningError("Model returned an invalid retrieval plan.") from exc
