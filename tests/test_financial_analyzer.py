from dataclasses import FrozenInstanceError

import pytest

from app.agent.sql_executor import SQLQueryResult
from app.analysis.financial_analyzer import (
    AnalysisError,
    AnalysisOperation,
    FinancialAnalyzer,
    RankingDirection,
)


def result(rows: list[dict], columns: list[str]) -> SQLQueryResult:
    return SQLQueryResult(columns=columns, rows=rows, row_count=len(rows))


def test_absolute_change_subtracts_old_from_new():
    sql_result = result(
        [{"year": 2024, "revenue": 391035}, {"year": 2025, "revenue": 416161}],
        ["year", "revenue"],
    )

    analysis = FinancialAnalyzer().absolute_change(
        sql_result,
        old_row=0,
        old_column="revenue",
        new_row=1,
        new_column="revenue",
    )

    assert analysis.operation is AnalysisOperation.ABSOLUTE_CHANGE
    assert analysis.value == 25126
    assert analysis.ranked_rows == ()


def test_positive_percentage_change_uses_old_as_baseline():
    sql_result = result([{"value": 100}, {"value": 125}], ["value"])

    analysis = FinancialAnalyzer().percentage_change(
        sql_result,
        old_row=0,
        old_column="value",
        new_row=1,
        new_column="value",
    )

    assert analysis.operation is AnalysisOperation.PERCENTAGE_CHANGE
    assert analysis.value == pytest.approx(25.0)


def test_negative_percentage_change():
    sql_result = result([{"value": 200}, {"value": 150}], ["value"])

    analysis = FinancialAnalyzer().percentage_change(
        sql_result,
        old_row=0,
        old_column="value",
        new_row=1,
        new_column="value",
    )

    assert analysis.value == pytest.approx(-25.0)


def test_difference_subtracts_right_from_left_using_explicit_columns():
    sql_result = result(
        [{"net_income": 30, "revenue": 100}],
        ["net_income", "revenue"],
    )

    analysis = FinancialAnalyzer().difference(
        sql_result,
        left_row=0,
        left_column="revenue",
        right_row=0,
        right_column="net_income",
    )

    assert analysis.operation is AnalysisOperation.DIFFERENCE
    assert analysis.value == 70


@pytest.mark.parametrize(
    ("direction", "expected_tickers"),
    [
        (RankingDirection.DESCENDING, ["AAPL", "MSFT", "GOOG"]),
        (RankingDirection.ASCENDING, ["GOOG", "MSFT", "AAPL"]),
    ],
)
def test_ranking_preserves_complete_rows(direction, expected_tickers):
    sql_result = result(
        [
            {"ticker": "MSFT", "revenue": 281724, "year": 2025},
            {"ticker": "GOOG", "revenue": 200000, "year": 2025},
            {"ticker": "AAPL", "revenue": 416161, "year": 2025},
        ],
        ["ticker", "revenue", "year"],
    )

    analysis = FinancialAnalyzer().ranking(
        sql_result,
        column="revenue",
        direction=direction,
    )

    assert analysis.operation is AnalysisOperation.RANKING
    assert analysis.value is None
    assert [row["ticker"] for row in analysis.ranked_rows] == expected_tickers
    assert all(
        set(row) == {"ticker", "revenue", "year"}
        for row in analysis.ranked_rows
    )


def test_analysis_does_not_mutate_sql_result_or_expose_mutable_ranked_rows():
    rows = [
        {"ticker": "MSFT", "revenue": 281724},
        {"ticker": "AAPL", "revenue": 416161},
    ]
    sql_result = result(rows, ["ticker", "revenue"])

    analysis = FinancialAnalyzer().ranking(
        sql_result,
        column="revenue",
        direction=RankingDirection.DESCENDING,
    )

    assert sql_result.rows == rows
    assert [row["ticker"] for row in sql_result.rows] == ["MSFT", "AAPL"]
    with pytest.raises(TypeError):
        analysis.ranked_rows[0]["revenue"] = 0
    with pytest.raises(FrozenInstanceError):
        analysis.value = 0
    rows[1]["revenue"] = 0
    assert analysis.ranked_rows[0]["revenue"] == 416161


@pytest.mark.parametrize("operation", ["absolute_change", "ranking"])
def test_empty_sql_result_is_rejected(operation):
    sql_result = result([], ["value"])
    analyzer = FinancialAnalyzer()

    with pytest.raises(AnalysisError, match="no rows"):
        if operation == "ranking":
            analyzer.ranking(
                sql_result,
                column="value",
                direction=RankingDirection.ASCENDING,
            )
        else:
            analyzer.absolute_change(
                sql_result,
                old_row=0,
                old_column="value",
                new_row=1,
                new_column="value",
            )


def test_missing_requested_column_is_rejected():
    sql_result = result([{"value": 1}, {"value": 2}], ["value"])

    with pytest.raises(AnalysisError, match="column 'missing' is missing"):
        FinancialAnalyzer().absolute_change(
            sql_result,
            old_row=0,
            old_column="value",
            new_row=1,
            new_column="missing",
        )


def test_missing_row_value_is_rejected_even_when_column_is_listed():
    sql_result = result([{"value": 1}, {"other": 2}], ["value"])

    with pytest.raises(AnalysisError, match="column 'value' is missing"):
        FinancialAnalyzer().ranking(
            sql_result,
            column="value",
            direction=RankingDirection.ASCENDING,
        )


@pytest.mark.parametrize(
    "invalid_value",
    ["416161", None, True, float("nan"), float("inf"), -float("inf")],
)
def test_invalid_numeric_values_are_rejected(invalid_value):
    sql_result = result([{"value": 1}, {"value": invalid_value}], ["value"])

    with pytest.raises(AnalysisError, match="(numeric|finite) values"):
        FinancialAnalyzer().percentage_change(
            sql_result,
            old_row=0,
            old_column="value",
            new_row=1,
            new_column="value",
        )


@pytest.mark.parametrize("invalid_value", ["416161", True, float("nan"), float("inf")])
def test_ranking_rejects_invalid_numeric_values(invalid_value):
    sql_result = result([{"value": 1}, {"value": invalid_value}], ["value"])

    with pytest.raises(AnalysisError, match="(numeric|finite) values"):
        FinancialAnalyzer().ranking(
            sql_result,
            column="value",
            direction=RankingDirection.DESCENDING,
        )


def test_zero_percentage_baseline_is_rejected():
    sql_result = result([{"value": 0}, {"value": 25}], ["value"])

    with pytest.raises(AnalysisError, match="nonzero old value"):
        FinancialAnalyzer().percentage_change(
            sql_result,
            old_row=0,
            old_column="value",
            new_row=1,
            new_column="value",
        )


def test_insufficient_rows_for_requested_positions_are_rejected():
    sql_result = result([{"value": 25}], ["value"])

    with pytest.raises(AnalysisError, match="row index 1 is out of range"):
        FinancialAnalyzer().absolute_change(
            sql_result,
            old_row=0,
            old_column="value",
            new_row=1,
            new_column="value",
        )


def test_negative_row_index_is_not_used_as_python_reverse_index():
    sql_result = result([{"value": 25}, {"value": 30}], ["value"])

    with pytest.raises(AnalysisError, match="row index -1 is out of range"):
        FinancialAnalyzer().difference(
            sql_result,
            left_row=-1,
            left_column="value",
            right_row=1,
            right_column="value",
        )


def test_unsupported_ranking_direction_is_rejected():
    sql_result = result([{"value": 25}], ["value"])

    with pytest.raises(AnalysisError, match="ranking direction"):
        FinancialAnalyzer().ranking(
            sql_result,
            column="value",
            direction="sideways",  # type: ignore[arg-type]
        )


def test_operation_names_are_closed():
    with pytest.raises(AnalysisError, match="unsupported analysis operation: 'mean'"):
        AnalysisOperation("mean")


def test_nonfinite_calculated_result_is_rejected():
    sql_result = result([{"value": -1e308}, {"value": 1e308}], ["value"])

    with pytest.raises(AnalysisError, match="result must be finite"):
        FinancialAnalyzer().absolute_change(
            sql_result,
            old_row=0,
            old_column="value",
            new_row=1,
            new_column="value",
        )


def test_numeric_overflow_is_reported_as_analysis_error():
    sql_result = result([{"value": 10**400}, {"value": 1.0}], ["value"])

    with pytest.raises(AnalysisError, match="result must be finite"):
        FinancialAnalyzer().absolute_change(
            sql_result,
            old_row=0,
            old_column="value",
            new_row=1,
            new_column="value",
        )
