FINANCIAL_SCHEMA = """
Table: financial_metrics

Columns:
- id INTEGER
  Primary key.

- company TEXT
  Company name.
  Examples: Apple, Microsoft.

- ticker TEXT
  Stock ticker.
  Examples: AAPL, MSFT.

- fiscal_year INTEGER
  Fiscal year.
  Examples: 2024, 2025.

- revenue_musd REAL
  Total annual revenue in millions of US dollars.

- net_income_musd REAL
  Annual net income in millions of US dollars.

- operating_income_musd REAL
  Annual operating income in millions of US dollars.

- gross_margin REAL
  Gross margin stored as a decimal ratio.
  Example: 0.469 means 46.9%.

Constraints:
- One row represents one company and one fiscal year.
- ticker and fiscal_year uniquely identify a financial period.
""".strip()