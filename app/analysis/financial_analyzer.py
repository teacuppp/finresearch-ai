from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from types import MappingProxyType
from typing import Any, Mapping

from app.agent.sql_executor import SQLQueryResult


class AnalysisError(ValueError):
    """The SQL result or requested analysis inputs are invalid."""


class AnalysisOperation(StrEnum):
    ABSOLUTE_CHANGE = "absolute_change"
    PERCENTAGE_CHANGE = "percentage_change"
    DIFFERENCE = "difference"
    RANKING = "ranking"

    @classmethod
    def _missing_(cls, value: object) -> None:
        raise AnalysisError(f"unsupported analysis operation: {value!r}")


class RankingDirection(StrEnum):
    ASCENDING = "ascending"
    DESCENDING = "descending"

    @classmethod
    def _missing_(cls, value: object) -> None:
        raise AnalysisError(f"unsupported ranking direction: {value!r}")


@dataclass(frozen=True)
class AnalysisResult:
    operation: AnalysisOperation
    value: int | float | None = None
    ranked_rows: tuple[Mapping[str, Any], ...] = ()


class FinancialAnalyzer:
    def absolute_change(
        self,
        sql_result: SQLQueryResult,
        *,
        old_row: int,
        old_column: str,
        new_row: int,
        new_column: str,
    ) -> AnalysisResult:
        old = self._value(sql_result, old_row, old_column)
        new = self._value(sql_result, new_row, new_column)
        value = self._subtract(new, old)
        return AnalysisResult(AnalysisOperation.ABSOLUTE_CHANGE, value=value)

    def percentage_change(
        self,
        sql_result: SQLQueryResult,
        *,
        old_row: int,
        old_column: str,
        new_row: int,
        new_column: str,
    ) -> AnalysisResult:
        old = self._value(sql_result, old_row, old_column)
        new = self._value(sql_result, new_row, new_column)
        if old == 0:
            raise AnalysisError("percentage_change requires a nonzero old value")
        try:
            value = ((new - old) / old) * 100
        except OverflowError as exc:
            raise AnalysisError("analysis result must be finite") from exc
        return AnalysisResult(
            AnalysisOperation.PERCENTAGE_CHANGE,
            value=self._finite_result(value),
        )

    def difference(
        self,
        sql_result: SQLQueryResult,
        *,
        left_row: int,
        left_column: str,
        right_row: int,
        right_column: str,
    ) -> AnalysisResult:
        left = self._value(sql_result, left_row, left_column)
        right = self._value(sql_result, right_row, right_column)
        value = self._subtract(left, right)
        return AnalysisResult(AnalysisOperation.DIFFERENCE, value=value)

    def ranking(
        self,
        sql_result: SQLQueryResult,
        *,
        column: str,
        direction: RankingDirection,
    ) -> AnalysisResult:
        if not isinstance(direction, RankingDirection):
            raise AnalysisError("ranking direction must be ascending or descending")
        self._require_rows(sql_result)
        ranked = sorted(
            sql_result.rows,
            key=lambda row: self._row_value(sql_result, row, column),
            reverse=direction is RankingDirection.DESCENDING,
        )
        return AnalysisResult(
            AnalysisOperation.RANKING,
            ranked_rows=tuple(MappingProxyType(dict(row)) for row in ranked),
        )

    @staticmethod
    def _require_rows(sql_result: SQLQueryResult) -> None:
        if not sql_result.rows:
            raise AnalysisError("SQL result has no rows")

    @classmethod
    def _value(
        cls,
        sql_result: SQLQueryResult,
        row_index: int,
        column: str,
    ) -> int | float:
        cls._require_rows(sql_result)
        if (
            isinstance(row_index, bool)
            or not isinstance(row_index, int)
            or not 0 <= row_index < len(sql_result.rows)
        ):
            raise AnalysisError(f"row index {row_index!r} is out of range")
        return cls._row_value(sql_result, sql_result.rows[row_index], column)

    @staticmethod
    def _row_value(
        sql_result: SQLQueryResult,
        row: Mapping[str, Any],
        column: str,
    ) -> int | float:
        if column not in sql_result.columns or column not in row:
            raise AnalysisError(f"column {column!r} is missing")
        value = row[column]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise AnalysisError(f"column {column!r} must contain numeric values")
        if isinstance(value, float) and not isfinite(value):
            raise AnalysisError(f"column {column!r} must contain finite values")
        return value

    @staticmethod
    def _finite_result(value: int | float) -> int | float:
        if isinstance(value, float) and not isfinite(value):
            raise AnalysisError("analysis result must be finite")
        return value

    @classmethod
    def _subtract(cls, left: int | float, right: int | float) -> int | float:
        try:
            value = left - right
        except OverflowError as exc:
            raise AnalysisError("analysis result must be finite") from exc
        return cls._finite_result(value)
