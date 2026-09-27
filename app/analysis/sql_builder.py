from app.agent.sql_executor import SQLValidationError, validate_read_only_sql
from app.analysis.financial_analyzer import AnalysisOperation
from app.analysis.retrieval import AnalysisRetrievalPlan, _validated_plan


class AnalysisSQLBuildError(ValueError):
    """A raw-data SQL query cannot be constructed from the retrieval plan."""


def _quote_entity(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


class AnalysisSQLBuilder:
    def build(
        self, plan: AnalysisRetrievalPlan, operation: AnalysisOperation
    ) -> str:
        try:
            validated = _validated_plan(plan, operation)
        except ValueError as exc:
            raise AnalysisSQLBuildError("Invalid analysis retrieval plan.") from exc

        filters: list[str] = []
        if len(validated.entities) == 1:
            filters.append(
                f"{validated.entity_column} = {_quote_entity(validated.entities[0])}"
            )
        elif validated.entities:
            values = ", ".join(_quote_entity(entity) for entity in validated.entities)
            filters.append(f"{validated.entity_column} IN ({values})")

        years = sorted(validated.fiscal_years)
        if len(years) == 1:
            filters.append(f"fiscal_year = {years[0]}")
        else:
            filters.append("fiscal_year IN (" + ", ".join(map(str, years)) + ")")

        order_column = (
            "fiscal_year"
            if operation in (
                AnalysisOperation.ABSOLUTE_CHANGE,
                AnalysisOperation.PERCENTAGE_CHANGE,
            )
            else validated.entity_column
        )
        statement = (
            f"SELECT company, ticker, fiscal_year, {validated.metric}\n"
            "FROM financial_metrics\n"
            "WHERE " + " AND ".join(filters) + f"\nORDER BY {order_column}"
        )
        try:
            return validate_read_only_sql(statement)
        except SQLValidationError as exc:
            raise AnalysisSQLBuildError("Constructed SQL failed read-only validation.") from exc
