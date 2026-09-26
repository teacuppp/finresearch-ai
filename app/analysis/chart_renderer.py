"""Deterministic, in-memory charts for structured SQL results."""

from dataclasses import dataclass
from enum import StrEnum
from io import BytesIO
from math import isfinite
from typing import Literal

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from app.agent.sql_executor import SQLQueryResult


class ChartRenderingError(ValueError):
    """The chart specification or SQL result cannot be rendered."""


class ChartType(StrEnum):
    BAR = "bar"
    LINE = "line"

    @classmethod
    def _missing_(cls, value: object) -> None:
        raise ChartRenderingError(f"Unsupported chart type: {value!r}")


@dataclass(frozen=True)
class ChartSpec:
    chart_type: ChartType
    x_column: str
    y_column: str
    title: str
    x_label: str | None = None
    y_label: str | None = None


@dataclass(frozen=True)
class ChartArtifact:
    media_type: Literal["image/png"]
    content: bytes


class ChartRenderer:
    def render(self, sql_result: SQLQueryResult, spec: ChartSpec) -> ChartArtifact:
        self._validate_spec(spec)
        x_values, y_values = self._values(sql_result, spec)

        figure = Figure()
        canvas = FigureCanvasAgg(figure)
        try:
            axes = figure.add_subplot(111)
            if spec.chart_type is ChartType.BAR:
                # Positions preserve SQL row order and avoid categorical overlap.
                positions = list(range(len(x_values)))
                axes.bar(positions, y_values)
                axes.set_xticks(positions, [str(value) for value in x_values])
            elif all(isinstance(value, (int, float)) for value in x_values):
                axes.plot(x_values, y_values)
            else:
                positions = list(range(len(x_values)))
                axes.plot(positions, y_values)
                axes.set_xticks(positions, x_values)

            axes.set_title(spec.title)
            axes.set_xlabel(spec.x_label if spec.x_label is not None else spec.x_column)
            axes.set_ylabel(spec.y_label if spec.y_label is not None else spec.y_column)

            with BytesIO() as output:
                canvas.print_png(output)
                content = output.getvalue()
        except (OverflowError, TypeError, ValueError) as exc:
            raise ChartRenderingError("Chart could not be rendered") from exc

        return ChartArtifact(media_type="image/png", content=content)

    @staticmethod
    def _validate_spec(spec: ChartSpec) -> None:
        if not isinstance(spec, ChartSpec):
            raise ChartRenderingError("A ChartSpec is required")
        if not isinstance(spec.chart_type, ChartType):
            raise ChartRenderingError(f"Unsupported chart type: {spec.chart_type!r}")
        for name, value in (
            ("x_column", spec.x_column),
            ("y_column", spec.y_column),
            ("title", spec.title),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ChartRenderingError(f"{name} must not be blank")
        for name, value in (("x_label", spec.x_label), ("y_label", spec.y_label)):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ChartRenderingError(f"{name} must not be blank")

    @classmethod
    def _values(
        cls, sql_result: SQLQueryResult, spec: ChartSpec
    ) -> tuple[list[str | int | float], list[int | float]]:
        if not sql_result.rows:
            raise ChartRenderingError("SQL result has no rows")
        for column in (spec.x_column, spec.y_column):
            if column not in sql_result.columns:
                raise ChartRenderingError(f"Column {column!r} is missing")
        if spec.chart_type is ChartType.LINE and len(sql_result.rows) < 2:
            raise ChartRenderingError("A line chart requires at least two rows")

        x_values: list[str | int | float] = []
        y_values: list[int | float] = []
        for row in sql_result.rows:
            for column in (spec.x_column, spec.y_column):
                if column not in row:
                    raise ChartRenderingError(f"Column {column!r} is missing from a row")
            x_values.append(cls._x_value(row[spec.x_column]))
            y_values.append(cls._number(row[spec.y_column], spec.y_column))

        if any(isinstance(value, str) for value in x_values) and not all(
            isinstance(value, str) for value in x_values
        ):
            raise ChartRenderingError("X values must be all text or all numeric")
        if len(set(x_values)) != len(x_values):
            raise ChartRenderingError("Duplicate x values are ambiguous")
        return x_values, y_values

    @staticmethod
    def _x_value(value: object) -> str | int | float:
        if isinstance(value, str):
            if value.strip():
                return value
            raise ChartRenderingError("X values must not be blank")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ChartRenderingError("X values must be text or finite numbers")
        ChartRenderer._require_finite(value, "X values")
        return value

    @staticmethod
    def _number(value: object, column: str) -> int | float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ChartRenderingError(f"Column {column!r} must contain numeric values")
        ChartRenderer._require_finite(value, f"Column {column!r}")
        return value

    @staticmethod
    def _require_finite(value: int | float, label: str) -> None:
        try:
            finite = isfinite(value)
        except (OverflowError, TypeError, ValueError):
            finite = False
        if not finite:
            raise ChartRenderingError(f"{label} must contain finite values")
