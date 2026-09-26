"""Deterministic chart rendering from SQLQueryResult without external services."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
from struct import unpack

import pytest
from matplotlib._pylab_helpers import Gcf
from matplotlib.figure import Figure

from app.agent.sql_executor import SQLQueryResult
from app.analysis import chart_renderer
from app.analysis.chart_renderer import (
    ChartArtifact,
    ChartRenderer,
    ChartRenderingError,
    ChartSpec,
    ChartType,
)


def result(rows: list[dict]) -> SQLQueryResult:
    return SQLQueryResult(
        columns=["ticker", "revenue_musd"],
        rows=rows,
        row_count=len(rows),
    )


def spec(chart_type: ChartType = ChartType.BAR) -> ChartSpec:
    return ChartSpec(
        chart_type=chart_type,
        x_column="ticker",
        y_column="revenue_musd",
        title="Revenue",
    )


def two_rows() -> SQLQueryResult:
    return result(
        [
            {"ticker": "MSFT", "revenue_musd": 281724},
            {"ticker": "AAPL", "revenue_musd": 416161},
        ]
    )


@pytest.fixture
def captured_figures(monkeypatch: pytest.MonkeyPatch) -> list[Figure]:
    figures: list[Figure] = []

    def make_figure() -> Figure:
        figure = Figure()
        figures.append(figure)
        return figure

    monkeypatch.setattr(chart_renderer, "Figure", make_figure)
    return figures


@pytest.mark.parametrize("chart_type", [ChartType.BAR, ChartType.LINE])
def test_render_returns_in_memory_png(chart_type: ChartType) -> None:
    artifact = ChartRenderer().render(two_rows(), spec(chart_type))

    assert artifact.media_type == "image/png"
    assert isinstance(artifact.content, bytes)
    assert len(artifact.content) > 100
    assert artifact.content[:8] == b"\x89PNG\r\n\x1a\n"
    assert artifact.content[12:16] == b"IHDR"
    width, height = unpack(">II", artifact.content[16:24])
    assert width > 0 and height > 0


def test_bar_preserves_sql_row_order(captured_figures: list[Figure]) -> None:
    ChartRenderer().render(two_rows(), spec())

    axes = captured_figures[0].axes[0]
    assert [patch.get_height() for patch in axes.patches] == [281724, 416161]
    assert [label.get_text() for label in axes.get_xticklabels()] == ["MSFT", "AAPL"]


def test_line_preserves_numeric_input_order(captured_figures: list[Figure]) -> None:
    sql_result = SQLQueryResult(
        columns=["fiscal_year", "revenue_musd"],
        rows=[
            {"fiscal_year": 2025, "revenue_musd": 416161},
            {"fiscal_year": 2024, "revenue_musd": 391035},
        ],
        row_count=2,
    )
    chart_spec = ChartSpec(ChartType.LINE, "fiscal_year", "revenue_musd", "Revenue")

    ChartRenderer().render(sql_result, chart_spec)

    assert captured_figures[0].axes[0].lines[0].get_xydata().tolist() == [
        [2025.0, 416161.0],
        [2024.0, 391035.0],
    ]


def test_line_uses_categorical_labels_in_input_order(
    captured_figures: list[Figure],
) -> None:
    ChartRenderer().render(two_rows(), spec(ChartType.LINE))

    axes = captured_figures[0].axes[0]
    assert axes.lines[0].get_xydata().tolist() == [
        [0.0, 281724.0],
        [1.0, 416161.0],
    ]
    assert [label.get_text() for label in axes.get_xticklabels()] == ["MSFT", "AAPL"]


def test_custom_title_and_labels_are_applied(captured_figures: list[Figure]) -> None:
    chart_spec = ChartSpec(
        ChartType.BAR,
        "ticker",
        "revenue_musd",
        "2025 revenue",
        x_label="Company",
        y_label="USD millions",
    )
    ChartRenderer().render(two_rows(), chart_spec)

    axes = captured_figures[0].axes[0]
    assert axes.get_title() == "2025 revenue"
    assert axes.get_xlabel() == "Company"
    assert axes.get_ylabel() == "USD millions"


def test_default_axis_labels_use_source_columns(captured_figures: list[Figure]) -> None:
    ChartRenderer().render(two_rows(), spec())

    axes = captured_figures[0].axes[0]
    assert axes.get_xlabel() == "ticker"
    assert axes.get_ylabel() == "revenue_musd"


def test_sql_result_and_rows_are_not_mutated() -> None:
    sql_result = two_rows()
    original = deepcopy(sql_result)

    ChartRenderer().render(sql_result, spec())

    assert sql_result == original


def test_chart_spec_and_artifact_are_immutable() -> None:
    chart_spec = spec()
    artifact = ChartRenderer().render(two_rows(), chart_spec)

    with pytest.raises(FrozenInstanceError):
        chart_spec.title = "Changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        artifact.content = b""  # type: ignore[misc]
    assert isinstance(artifact, ChartArtifact)


def test_empty_result_is_rejected() -> None:
    with pytest.raises(ChartRenderingError, match="no rows"):
        ChartRenderer().render(result([]), spec())


@pytest.mark.parametrize("missing_column", ["ticker", "revenue_musd"])
def test_missing_declared_column_is_rejected(missing_column: str) -> None:
    sql_result = two_rows()
    sql_result.columns.remove(missing_column)

    with pytest.raises(ChartRenderingError, match="is missing"):
        ChartRenderer().render(sql_result, spec())


@pytest.mark.parametrize("missing_column", ["ticker", "revenue_musd"])
def test_missing_row_cell_is_rejected(missing_column: str) -> None:
    sql_result = two_rows()
    del sql_result.rows[1][missing_column]

    with pytest.raises(ChartRenderingError, match="missing from a row"):
        ChartRenderer().render(sql_result, spec())


@pytest.mark.parametrize("value", ["416161", True, float("nan"), float("inf"), -float("inf")])
def test_invalid_y_value_is_rejected(value: object) -> None:
    sql_result = two_rows()
    sql_result.rows[0]["revenue_musd"] = value

    with pytest.raises(ChartRenderingError, match="numeric|finite"):
        ChartRenderer().render(sql_result, spec())


@pytest.mark.parametrize("value", [None, True, "  ", float("nan"), float("inf"), -float("inf")])
def test_invalid_x_value_is_rejected(value: object) -> None:
    sql_result = two_rows()
    sql_result.rows[0]["ticker"] = value

    with pytest.raises(ChartRenderingError, match="X values"):
        ChartRenderer().render(sql_result, spec())


def test_mixed_text_and_numeric_x_values_are_rejected() -> None:
    sql_result = two_rows()
    sql_result.rows[0]["ticker"] = 2025

    with pytest.raises(ChartRenderingError, match="all text or all numeric"):
        ChartRenderer().render(sql_result, spec())


def test_line_requires_at_least_two_rows() -> None:
    with pytest.raises(ChartRenderingError, match="at least two rows"):
        ChartRenderer().render(result([two_rows().rows[0]]), spec(ChartType.LINE))


@pytest.mark.parametrize("field", ["title", "x_column", "y_column", "x_label", "y_label"])
def test_blank_spec_fields_are_rejected(field: str) -> None:
    values = {
        "chart_type": ChartType.BAR,
        "x_column": "ticker",
        "y_column": "revenue_musd",
        "title": "Revenue",
    }
    values[field] = "  "

    with pytest.raises(ChartRenderingError, match=field):
        ChartRenderer().render(two_rows(), ChartSpec(**values))


def test_unsupported_chart_type_is_rejected() -> None:
    chart_spec = ChartSpec("pie", "ticker", "revenue_musd", "Revenue")  # type: ignore[arg-type]

    with pytest.raises(ChartRenderingError, match="Unsupported chart type"):
        ChartRenderer().render(two_rows(), chart_spec)
    with pytest.raises(ChartRenderingError, match="Unsupported chart type"):
        ChartType("pie")


@pytest.mark.parametrize("chart_type", [ChartType.BAR, ChartType.LINE])
def test_duplicate_categorical_x_is_rejected(chart_type: ChartType) -> None:
    sql_result = two_rows()
    sql_result.rows[1]["ticker"] = "MSFT"

    with pytest.raises(ChartRenderingError, match="Duplicate x values"):
        ChartRenderer().render(sql_result, spec(chart_type))


def test_duplicate_numeric_x_is_rejected() -> None:
    sql_result = two_rows()
    sql_result.rows[0]["ticker"] = 2025
    sql_result.rows[1]["ticker"] = 2025.0

    with pytest.raises(ChartRenderingError, match="Duplicate x values"):
        ChartRenderer().render(sql_result, spec(ChartType.LINE))


def test_repeated_renders_use_isolated_figures_without_pyplot_managers(
    captured_figures: list[Figure],
) -> None:
    original_managers = tuple(Gcf.get_all_fig_managers())
    renderer = ChartRenderer()

    renderer.render(two_rows(), spec())
    renderer.render(two_rows(), spec(ChartType.LINE))

    assert len(captured_figures) == 2
    assert captured_figures[0] is not captured_figures[1]
    assert captured_figures[0].axes[0] is not captured_figures[1].axes[0]
    assert len(captured_figures[0].axes[0].patches) == 2
    assert len(captured_figures[0].axes[0].lines) == 0
    assert len(captured_figures[1].axes[0].patches) == 0
    assert len(captured_figures[1].axes[0].lines) == 1
    assert tuple(Gcf.get_all_fig_managers()) == original_managers


def test_render_creates_no_file_in_working_directory(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    ChartRenderer().render(two_rows(), spec())

    assert list(tmp_path.iterdir()) == []
