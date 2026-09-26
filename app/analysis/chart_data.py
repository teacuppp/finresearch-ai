"""Prepare chart-ready rows without changing financial analysis semantics."""

from collections.abc import Mapping
from typing import Any

from app.agent.sql_executor import SQLQueryResult
from app.analysis.financial_analyzer import AnalysisOperation, AnalysisResult


class ChartDataError(ValueError):
    """A ranking result cannot be represented as chart-ready SQL rows."""


class ChartDataBuilder:
    def build(
        self,
        sql_result: SQLQueryResult,
        analysis_result: AnalysisResult | None = None,
    ) -> SQLQueryResult | None:
        if analysis_result is None:
            return sql_result

        if analysis_result.operation in (
            AnalysisOperation.ABSOLUTE_CHANGE,
            AnalysisOperation.PERCENTAGE_CHANGE,
            AnalysisOperation.DIFFERENCE,
        ):
            return None

        if analysis_result.operation is not AnalysisOperation.RANKING:
            raise ChartDataError("Unsupported analysis operation for chart data.")
        if not analysis_result.ranked_rows:
            raise ChartDataError("Ranking result has no ranked rows.")

        rows: list[dict[str, Any]] = []
        for index, ranked_row in enumerate(analysis_result.ranked_rows):
            if not isinstance(ranked_row, Mapping):
                raise ChartDataError(f"Ranked row {index} is not a mapping.")
            for column in sql_result.columns:
                if column not in ranked_row:
                    raise ChartDataError(
                        f"Ranked row {index} is missing column {column!r}."
                    )
            rows.append({column: ranked_row[column] for column in sql_result.columns})

        return SQLQueryResult(
            columns=list(sql_result.columns),
            rows=rows,
            row_count=len(rows),
        )
