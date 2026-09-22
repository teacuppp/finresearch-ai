import sqlite3

import pytest

from app.agent.sql_executor import (
    SQLExecutionError,
    SQLExecutor,
    SQLValidationError,
    validate_read_only_sql,
)


def _create_test_database(
    tmp_path,
):
    database_path = (
        tmp_path
        / "financial_test.db"
    )

    with sqlite3.connect(
        database_path
    ) as connection:
        connection.execute(
            """
            CREATE TABLE financial_metrics (
                ticker TEXT NOT NULL,
                fiscal_year INTEGER NOT NULL,
                revenue_musd REAL NOT NULL,
                net_income_musd REAL NOT NULL
            )
            """
        )

        connection.executemany(
            """
            INSERT INTO financial_metrics (
                ticker,
                fiscal_year,
                revenue_musd,
                net_income_musd
            )
            VALUES (?, ?, ?, ?)
            """,
            [
                (
                    "AAPL",
                    2024,
                    391035.0,
                    93736.0,
                ),
                (
                    "AAPL",
                    2025,
                    416161.0,
                    112010.0,
                ),
                (
                    "MSFT",
                    2025,
                    281724.0,
                    101832.0,
                ),
            ],
        )

        connection.commit()

    return database_path


def test_validate_allows_select():
    sql = validate_read_only_sql(
        """
        SELECT ticker, revenue_musd
        FROM financial_metrics
        """
    )

    assert sql.startswith(
        "SELECT"
    )


def test_validate_allows_cte():
    sql = validate_read_only_sql(
        """
        WITH latest AS (
            SELECT *
            FROM financial_metrics
            WHERE fiscal_year = 2025
        )
        SELECT *
        FROM latest
        """
    )

    assert sql.startswith(
        "WITH"
    )


@pytest.mark.parametrize(
    "sql",
    [
        (
            "INSERT INTO financial_metrics "
            "VALUES ('TEST', 2025, 1, 1)"
        ),
        (
            "UPDATE financial_metrics "
            "SET revenue_musd = 0"
        ),
        (
            "DELETE FROM financial_metrics"
        ),
        (
            "DROP TABLE financial_metrics"
        ),
        (
            "CREATE TABLE test (id INTEGER)"
        ),
        (
            "ALTER TABLE financial_metrics "
            "ADD COLUMN test TEXT"
        ),
        (
            "PRAGMA table_info("
            "financial_metrics)"
        ),
        (
            "ATTACH DATABASE "
            "'other.db' AS other"
        ),
    ],
)
def test_validate_rejects_write_or_admin_sql(
    sql,
):
    with pytest.raises(
        SQLValidationError
    ):
        validate_read_only_sql(
            sql
        )


def test_validate_rejects_multiple_statements():
    with pytest.raises(
        SQLValidationError,
        match=(
            "Only one SQL statement"
        ),
    ):
        validate_read_only_sql(
            "SELECT 1; SELECT 2"
        )


def test_validate_rejects_empty_sql():
    with pytest.raises(
        SQLValidationError,
        match=(
            "must not be empty"
        ),
    ):
        validate_read_only_sql(
            "   "
        )


def test_validate_rejects_sql_comments():
    with pytest.raises(
        SQLValidationError,
        match=(
            "comments are not allowed"
        ),
    ):
        validate_read_only_sql(
            """
            SELECT *
            FROM financial_metrics
            -- comment
            """
        )


def test_executor_returns_structured_rows(
    tmp_path,
):
    database_path = (
        _create_test_database(
            tmp_path
        )
    )

    executor = SQLExecutor(
        database_path=database_path
    )

    result = executor.execute(
        """
        SELECT
            ticker,
            fiscal_year,
            net_income_musd
        FROM financial_metrics
        WHERE ticker = ?
          AND fiscal_year = ?
        """,
        (
            "MSFT",
            2025,
        ),
    )

    assert result.columns == [
        "ticker",
        "fiscal_year",
        "net_income_musd",
    ]

    assert result.row_count == 1

    assert result.rows == [
        {
            "ticker": "MSFT",
            "fiscal_year": 2025,
            "net_income_musd": (
                101832.0
            ),
        }
    ]


def test_executor_supports_cte(
    tmp_path,
):
    database_path = (
        _create_test_database(
            tmp_path
        )
    )

    executor = SQLExecutor(
        database_path=database_path
    )

    result = executor.execute(
        """
        WITH apple AS (
            SELECT *
            FROM financial_metrics
            WHERE ticker = ?
        )
        SELECT
            fiscal_year,
            revenue_musd
        FROM apple
        ORDER BY fiscal_year
        """,
        (
            "AAPL",
        ),
    )

    assert result.row_count == 2

    assert result.rows[0] == {
        "fiscal_year": 2024,
        "revenue_musd": 391035.0,
    }

    assert result.rows[1] == {
        "fiscal_year": 2025,
        "revenue_musd": 416161.0,
    }


def test_executor_rejects_write_query(
    tmp_path,
):
    database_path = (
        _create_test_database(
            tmp_path
        )
    )

    executor = SQLExecutor(
        database_path=database_path
    )

    with pytest.raises(
        SQLValidationError
    ):
        executor.execute(
            """
            DELETE FROM financial_metrics
            """
        )


def test_executor_raises_for_invalid_sql(
    tmp_path,
):
    database_path = (
        _create_test_database(
            tmp_path
        )
    )

    executor = SQLExecutor(
        database_path=database_path
    )

    with pytest.raises(
        SQLExecutionError
    ):
        executor.execute(
            """
            SELECT missing_column
            FROM financial_metrics
            """
        )


def test_executor_raises_when_database_missing(
    tmp_path,
):
    database_path = (
        tmp_path
        / "missing.db"
    )

    executor = SQLExecutor(
        database_path=database_path
    )

    with pytest.raises(
        SQLExecutionError,
        match=(
            "Database does not exist"
        ),
    ):
        executor.execute(
            "SELECT 1"
        )


def test_executor_enforces_max_rows(
    tmp_path,
):
    database_path = (
        _create_test_database(
            tmp_path
        )
    )

    executor = SQLExecutor(
        database_path=database_path,
        max_rows=2,
    )

    with pytest.raises(
        SQLExecutionError,
        match=(
            "more than 2 rows"
        ),
    ):
        executor.execute(
            """
            SELECT *
            FROM financial_metrics
            """
        )


def test_executor_rejects_nonpositive_max_rows(
    tmp_path,
):
    database_path = (
        tmp_path
        / "test.db"
    )

    with pytest.raises(
        ValueError,
        match=(
            "max_rows must be positive"
        ),
    ):
        SQLExecutor(
            database_path=database_path,
            max_rows=0,
        )