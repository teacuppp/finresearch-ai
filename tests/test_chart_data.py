from copy import deepcopy
from types import MappingProxyType

import pytest

from app.agent.sql_executor import SQLQueryResult
from app.analysis.chart_data import ChartDataBuilder, ChartDataError
from app.analysis.financial_analyzer import AnalysisOperation, AnalysisResult


def raw_result() -> SQLQueryResult:
    return SQLQueryResult(
        columns=["ticker", "revenue_musd"],
        rows=[
            {"ticker": "MSFT", "revenue_musd": 281724},
            {"ticker": "AAPL", "revenue_musd": 416161},
        ],
        row_count=2,
    )


def ranking_result() -> AnalysisResult:
    return AnalysisResult(
        operation=AnalysisOperation.RANKING,
        ranked_rows=(
            MappingProxyType({"ticker": "AAPL", "revenue_musd": 416161}),
            MappingProxyType({"ticker": "MSFT", "revenue_musd": 281724}),
        ),
    )


def test_direct_sql_returns_exact_existing_result_without_changing_rows() -> None:
    raw = raw_result()
    original = deepcopy(raw)

    built = ChartDataBuilder().build(raw)

    assert built is raw
    assert raw == original
    assert [row["ticker"] for row in built.rows] == ["MSFT", "AAPL"]


@pytest.mark.parametrize(
    "operation",
    [
        AnalysisOperation.ABSOLUTE_CHANGE,
        AnalysisOperation.PERCENTAGE_CHANGE,
        AnalysisOperation.DIFFERENCE,
    ],
)
def test_scalar_analysis_returns_none_even_when_raw_result_has_two_rows(
    operation: AnalysisOperation,
) -> None:
    raw = raw_result()
    original = deepcopy(raw)
    analysis = AnalysisResult(operation=operation, value=6.42)

    assert ChartDataBuilder().build(raw, analysis) is None
    assert raw == original
    assert analysis.value == 6.42


def test_ranking_builds_new_result_in_analysis_order_with_original_columns() -> None:
    raw = raw_result()
    analysis = ranking_result()

    built = ChartDataBuilder().build(raw, analysis)

    assert isinstance(built, SQLQueryResult)
    assert built is not raw
    assert built.columns == raw.columns
    assert built.columns is not raw.columns
    assert built.row_count == len(analysis.ranked_rows) == 2
    assert [row["ticker"] for row in built.rows] == ["AAPL", "MSFT"]
    assert [row["ticker"] for row in raw.rows] == ["MSFT", "AAPL"]


def test_ranking_rows_are_independent_plain_dict_copies() -> None:
    raw = raw_result()
    analysis = ranking_result()
    raw_before = deepcopy(raw)
    ranked_before = tuple(dict(row) for row in analysis.ranked_rows)

    built = ChartDataBuilder().build(raw, analysis)

    assert built is not None
    assert all(type(row) is dict for row in built.rows)
    assert all(
        built_row is not ranked_row
        for built_row, ranked_row in zip(built.rows, analysis.ranked_rows)
    )
    assert all(
        built_row is not raw_row
        for built_row in built.rows
        for raw_row in raw.rows
    )
    built.rows[0]["revenue_musd"] = 0
    built.columns.append("new_column")
    assert raw == raw_before
    assert tuple(dict(row) for row in analysis.ranked_rows) == ranked_before
    assert analysis.ranked_rows[0]["revenue_musd"] == 416161


def test_ranking_projects_only_declared_columns_and_discards_extra_keys() -> None:
    raw = raw_result()
    raw_before = deepcopy(raw)
    analysis = AnalysisResult(
        operation=AnalysisOperation.RANKING,
        ranked_rows=(
            MappingProxyType(
                {"ticker": "AAPL", "revenue_musd": 416161, "internal_note": "extra"}
            ),
            MappingProxyType(
                {"ticker": "MSFT", "revenue_musd": 281724, "internal_note": "extra"}
            ),
        ),
    )
    ranked_before = tuple(dict(row) for row in analysis.ranked_rows)

    built = ChartDataBuilder().build(raw, analysis)

    assert built is not None
    assert built.columns == raw.columns
    assert built.rows == [
        {"ticker": "AAPL", "revenue_musd": 416161},
        {"ticker": "MSFT", "revenue_musd": 281724},
    ]
    assert all(type(row) is dict for row in built.rows)
    assert raw == raw_before
    assert tuple(dict(row) for row in analysis.ranked_rows) == ranked_before


def test_ranking_row_count_matches_ranked_rows_not_raw_row_count() -> None:
    raw = raw_result()
    analysis = AnalysisResult(
        operation=AnalysisOperation.RANKING,
        ranked_rows=(MappingProxyType({"ticker": "AAPL", "revenue_musd": 416161}),),
    )

    built = ChartDataBuilder().build(raw, analysis)

    assert built is not None
    assert built.row_count == 1
    assert len(built.rows) == 1


def test_empty_ranking_rows_are_rejected_without_raw_fallback() -> None:
    raw = raw_result()
    analysis = AnalysisResult(operation=AnalysisOperation.RANKING)

    with pytest.raises(ChartDataError, match="no ranked rows"):
        ChartDataBuilder().build(raw, analysis)


def test_ranked_row_missing_original_column_is_rejected() -> None:
    raw = raw_result()
    analysis = AnalysisResult(
        operation=AnalysisOperation.RANKING,
        ranked_rows=(MappingProxyType({"ticker": "AAPL"}),),
    )

    with pytest.raises(ChartDataError, match="missing column 'revenue_musd'"):
        ChartDataBuilder().build(raw, analysis)


def test_non_mapping_ranked_row_is_rejected() -> None:
    raw = raw_result()
    analysis = AnalysisResult(
        operation=AnalysisOperation.RANKING,
        ranked_rows=(object(),),  # type: ignore[arg-type]
    )

    with pytest.raises(ChartDataError, match="not a mapping"):
        ChartDataBuilder().build(raw, analysis)


def test_unsupported_analysis_operation_is_rejected() -> None:
    analysis = AnalysisResult(operation="forecast")  # type: ignore[arg-type]

    with pytest.raises(ChartDataError, match="Unsupported analysis operation"):
        ChartDataBuilder().build(raw_result(), analysis)
