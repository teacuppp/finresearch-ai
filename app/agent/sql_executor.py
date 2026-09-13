from dataclasses import dataclass
from pathlib import Path
import re
import sqlite3
from typing import Any


class SQLValidationError(ValueError):
    pass


class SQLExecutionError(RuntimeError):
    pass


@dataclass
class SQLQueryResult:
    columns: list[str]
    rows: list[dict[str, Any]]
    row_count: int


_FORBIDDEN_SQL_PATTERN = re.compile(
    r"\b("
    r"INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|"
    r"REPLACE|ATTACH|DETACH|VACUUM|PRAGMA|"
    r"REINDEX|ANALYZE|BEGIN|COMMIT|ROLLBACK|"
    r"SAVEPOINT|RELEASE"
    r")\b",
    re.IGNORECASE,
)


def validate_read_only_sql(
    sql: str,
) -> str:
    statement = sql.strip()

    if not statement:
        raise SQLValidationError(
            "SQL query must not be empty."
        )

    if "--" in statement:
        raise SQLValidationError(
            "SQL comments are not allowed."
        )

    if "/*" in statement or "*/" in statement:
        raise SQLValidationError(
            "SQL comments are not allowed."
        )

    if statement.endswith(";"):
        statement = statement[:-1].rstrip()

    if ";" in statement:
        raise SQLValidationError(
            "Only one SQL statement is allowed."
        )

    first_token_match = re.match(
        r"^\s*([A-Za-z]+)",
        statement,
    )

    if first_token_match is None:
        raise SQLValidationError(
            "Unable to determine SQL statement type."
        )

    first_token = (
        first_token_match.group(1).upper()
    )

    if first_token not in {
        "SELECT",
        "WITH",
    }:
        raise SQLValidationError(
            "Only SELECT queries are allowed."
        )

    forbidden_match = (
        _FORBIDDEN_SQL_PATTERN.search(
            statement
        )
    )

    if forbidden_match is not None:
        keyword = (
            forbidden_match.group(1).upper()
        )

        raise SQLValidationError(
            f"Forbidden SQL keyword: {keyword}"
        )

    return statement


class SQLExecutor:
    def __init__(
        self,
        database_path: str | Path,
        max_rows: int = 1000,
    ):
        if max_rows <= 0:
            raise ValueError(
                "max_rows must be positive"
            )

        self.database_path = Path(
            database_path
        )

        self.max_rows = max_rows

    def execute(
        self,
        sql: str,
        parameters: tuple[Any, ...] = (),
    ) -> SQLQueryResult:
        statement = validate_read_only_sql(
            sql
        )

        if not self.database_path.exists():
            raise SQLExecutionError(
                "Database does not exist: "
                f"{self.database_path}"
            )

        database_uri = (
            f"file:{self.database_path.resolve()}"
            "?mode=ro"
        )

        try:
            with sqlite3.connect(
                database_uri,
                uri=True,
            ) as connection:
                connection.row_factory = (
                    sqlite3.Row
                )

                connection.execute(
                    "PRAGMA query_only = ON"
                )

                cursor = connection.execute(
                    statement,
                    parameters,
                )

                columns = [
                    description[0]
                    for description
                    in cursor.description or []
                ]

                fetched_rows = cursor.fetchmany(
                    self.max_rows + 1
                )

        except sqlite3.Error as exc:
            raise SQLExecutionError(
                str(exc)
            ) from exc

        if len(fetched_rows) > self.max_rows:
            raise SQLExecutionError(
                "Query returned more than "
                f"{self.max_rows} rows."
            )

        rows = [
            {
                column: row[column]
                for column in columns
            }
            for row in fetched_rows
        ]

        return SQLQueryResult(
            columns=columns,
            rows=rows,
            row_count=len(rows),
        )