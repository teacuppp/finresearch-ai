from pathlib import Path
import sqlite3


DATABASE_PATH = Path(
    "data/financial_demo.db"
)


FINANCIAL_ROWS = [
    (
        "Apple",
        "AAPL",
        2024,
        391035.0,
        93736.0,
        123216.0,
        0.462,
    ),
    (
        "Apple",
        "AAPL",
        2025,
        416161.0,
        112010.0,
        133050.0,
        0.469,
    ),
    (
        "Microsoft",
        "MSFT",
        2024,
        245122.0,
        88136.0,
        109433.0,
        0.697,
    ),
    (
        "Microsoft",
        "MSFT",
        2025,
        281724.0,
        101832.0,
        128528.0,
        0.688,
    ),
]


def create_database(
    database_path: Path,
) -> None:
    database_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if database_path.exists():
        database_path.unlink()

    with sqlite3.connect(
        database_path
    ) as connection:
        connection.execute(
            """
            CREATE TABLE financial_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                company TEXT NOT NULL,
                ticker TEXT NOT NULL,
                fiscal_year INTEGER NOT NULL,
                revenue_musd REAL NOT NULL,
                net_income_musd REAL NOT NULL,
                operating_income_musd REAL NOT NULL,
                gross_margin REAL NOT NULL,
                UNIQUE (
                    ticker,
                    fiscal_year
                )
            )
            """
        )

        connection.executemany(
            """
            INSERT INTO financial_metrics (
                company,
                ticker,
                fiscal_year,
                revenue_musd,
                net_income_musd,
                operating_income_musd,
                gross_margin
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            FINANCIAL_ROWS,
        )

        connection.commit()


def main() -> None:
    create_database(
        DATABASE_PATH
    )

    print(
        "Created financial demo database:"
    )

    print(
        DATABASE_PATH.resolve()
    )


if __name__ == "__main__":
    main()