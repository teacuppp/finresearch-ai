"""Run one live direct-SQL evaluation against the local financial database.

Run: python -m scripts.evaluate_direct_sql_model
"""

from dataclasses import dataclass
from math import isclose
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Literal, cast

from app.agent.schema import FINANCIAL_SCHEMA
from app.agent.sql_executor import SQLExecutor, SQLQueryResult
from app.agent.sql_generator import SQLGenerator, SQLResultMode


DATABASE_PATH = Path(__file__).resolve().parents[1] / "data/financial_demo.db"
CANDIDATE_MODEL = "qwen3:4b-instruct"
CONTEXT_COLUMNS = ("company", "ticker", "fiscal_year")
ENTITY_COLUMNS = ("company", "ticker")
CheckKind = Literal["lookup", "scalar", "single_chart", "comparison", "trend"]


@dataclass(frozen=True)
class EvaluationCase:
    name: str
    mode: SQLResultMode
    question: str
    reference_sql: str
    kind: CheckKind
    metric: str | None = None


CASES = (
    EvaluationCase(
        "lookup_company", "answer", "What was Apple's 2025 revenue?",
        "SELECT revenue_musd FROM financial_metrics "
        "WHERE company = 'Apple' AND fiscal_year = 2025",
        "lookup", "revenue_musd",
    ),
    EvaluationCase(
        "lookup_ticker", "answer", "What was MSFT's 2025 net income?",
        "SELECT net_income_musd FROM financial_metrics "
        "WHERE ticker = 'MSFT' AND fiscal_year = 2025",
        "lookup", "net_income_musd",
    ),
    EvaluationCase(
        "average", "answer",
        "What was the average 2025 revenue of Apple and Microsoft?",
        "SELECT AVG(revenue_musd) AS expected_value FROM financial_metrics "
        "WHERE company IN ('Apple', 'Microsoft') AND fiscal_year = 2025",
        "scalar",
    ),
    EvaluationCase(
        "count", "answer", "How many financial metric rows exist?",
        "SELECT COUNT(*) AS expected_value FROM financial_metrics", "scalar",
    ),
    EvaluationCase(
        "ratio", "answer",
        "What was Apple's ratio of net income to revenue in 2025?",
        "SELECT net_income_musd * 1.0 / revenue_musd AS expected_value "
        "FROM financial_metrics WHERE company = 'Apple' AND fiscal_year = 2025",
        "scalar",
    ),
    EvaluationCase(
        "filter_lookup", "answer", "What was Microsoft's 2024 gross margin?",
        "SELECT gross_margin FROM financial_metrics "
        "WHERE company = 'Microsoft' AND fiscal_year = 2024",
        "lookup", "gross_margin",
    ),
    EvaluationCase(
        "explicit_single_plot", "chart_ready", "Plot Apple's 2025 revenue.",
        "SELECT company, ticker, fiscal_year, revenue_musd "
        "FROM financial_metrics WHERE company = 'Apple' AND fiscal_year = 2025",
        "single_chart", "revenue_musd",
    ),
    EvaluationCase(
        "company_comparison", "chart_ready",
        "Compare Apple and Microsoft 2025 revenue.",
        "SELECT company, ticker, fiscal_year, revenue_musd "
        "FROM financial_metrics WHERE company IN ('Apple', 'Microsoft') "
        "AND fiscal_year = 2025 ORDER BY company",
        "comparison", "revenue_musd",
    ),
    EvaluationCase(
        "trend", "chart_ready", "Plot Apple's revenue from 2024 to 2025.",
        "SELECT company, ticker, fiscal_year, revenue_musd "
        "FROM financial_metrics WHERE company = 'Apple' "
        "AND fiscal_year IN (2024, 2025) ORDER BY fiscal_year",
        "trend", "revenue_musd",
    ),
    EvaluationCase(
        "gross_margin_comparison", "chart_ready",
        "Compare Apple and Microsoft 2025 gross margin.",
        "SELECT company, ticker, fiscal_year, gross_margin "
        "FROM financial_metrics WHERE company IN ('Apple', 'Microsoft') "
        "AND fiscal_year = 2025 ORDER BY company",
        "comparison", "gross_margin",
    ),
)


@dataclass(frozen=True)
class RepairCase:
    name: str
    mode: SQLResultMode
    question: str
    previous_sql: str
    error_message: str
    reference_sql: str
    kind: CheckKind
    metric: str


REPAIR_CASES = (
    RepairCase(
        "answer_repair", "answer", "What was Apple's 2025 revenue?",
        "SELECT revenue FROM financial_metrics "
        "WHERE company = 'Apple' AND fiscal_year = 2025",
        "no such column: revenue",
        "SELECT revenue_musd FROM financial_metrics "
        "WHERE company = 'Apple' AND fiscal_year = 2025",
        "lookup", "revenue_musd",
    ),
    RepairCase(
        "chart_ready_repair", "chart_ready", "Plot Apple's 2025 revenue.",
        "SELECT company, revenue FROM financial_metrics "
        "WHERE company = 'Apple' AND fiscal_year = 2025",
        "no such column: revenue",
        "SELECT company, ticker, fiscal_year, revenue_musd "
        "FROM financial_metrics WHERE company = 'Apple' AND fiscal_year = 2025",
        "single_chart", "revenue_musd",
    ),
)


@dataclass(frozen=True)
class CaseResult:
    name: str
    mode: SQLResultMode
    question: str
    seconds: float
    generated_sql: str | None
    sql_result: SQLQueryResult | None
    passed: bool
    detail: str | None = None


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _numeric(value: object) -> int | float:
    _require(type(value) in (int, float), f"Expected a numeric value, received {value!r}")
    return cast(int | float, value)


def _same_number(actual: object, expected: object) -> bool:
    return isclose(_numeric(actual), _numeric(expected), rel_tol=1e-6, abs_tol=1e-9)


def _one_row(result: SQLQueryResult) -> dict[str, object]:
    _require(result.row_count == 1 and len(result.rows) == 1, "Expected exactly one row")
    return result.rows[0]


def _answer_value(result: SQLQueryResult, metric: str | None) -> object:
    row = _one_row(result)
    if metric is not None and metric in result.columns:
        return row[metric]
    _require(len(result.columns) == 1, "Expected one scalar or the requested metric")
    return row[result.columns[0]]


def _matching_context(actual: dict[str, object], expected: dict[str, object]) -> None:
    for column in CONTEXT_COLUMNS:
        if column in actual:
            _require(actual[column] == expected[column], f"Wrong {column} context")


def _validate_result(
    result: SQLQueryResult, reference: SQLQueryResult, kind: CheckKind,
    metric: str | None,
) -> None:
    _require(result.row_count == len(result.rows), "Inconsistent result row count")
    _require(reference.row_count == len(reference.rows), "Inconsistent reference row count")
    if kind in ("lookup", "scalar"):
        expected = _answer_value(reference, metric)
        actual = _answer_value(result, metric if kind == "lookup" else None)
        _require(_same_number(actual, expected), "Wrong numeric answer")
        return

    _require(metric is not None and metric in result.columns, "Requested metric is missing")
    _require(result.row_count == reference.row_count, "Wrong row count")
    if kind == "single_chart":
        actual = _one_row(result)
        expected = _one_row(reference)
        _require(any(column in result.columns for column in CONTEXT_COLUMNS),
                 "Chart context is missing")
        _require(_same_number(actual[metric], expected[metric]), "Wrong metric value")
        _matching_context(actual, expected)
        return

    if kind == "comparison":
        _require(reference.row_count == 2, "Reference comparison must have two rows")
        dimension = next(
            (column for column in ENTITY_COLUMNS if column in result.columns), None
        )
        _require(dimension is not None, "Entity dimension is missing")
        expected_rows = {row[dimension]: row for row in reference.rows}
        expected_values = {entity: row[metric] for entity, row in expected_rows.items()}
        actual_values = {row[dimension]: row[metric] for row in result.rows}
        _require(len(actual_values) == result.row_count, "Duplicate entity rows")
        _require(actual_values.keys() == expected_values.keys(), "Wrong entity set")
        for entity, expected in expected_values.items():
            _require(_same_number(actual_values[entity], expected),
                     f"Wrong metric value for {entity}")
        for row in result.rows:
            _matching_context(row, expected_rows[row[dimension]])
        return

    _require(kind == "trend", "Unknown validation kind")
    _require(reference.row_count == 2, "Reference trend must have two rows")
    _require("fiscal_year" in result.columns, "Trend fiscal_year is missing")
    expected_years = [row["fiscal_year"] for row in reference.rows]
    actual_years = [row["fiscal_year"] for row in result.rows]
    _require(actual_years == expected_years, "Wrong trend years or row order")
    for actual, expected in zip(result.rows, reference.rows, strict=True):
        _require(_same_number(actual[metric], expected[metric]), "Wrong trend value")
        _matching_context(actual, expected)


def _run_generation_case(
    case: EvaluationCase, generator: SQLGenerator, executor: SQLExecutor,
) -> CaseResult:
    sql: str | None = None
    result: SQLQueryResult | None = None
    start = perf_counter()
    try:
        reference = executor.execute(case.reference_sql)
        start = perf_counter()
        sql = generator.generate(case.question, FINANCIAL_SCHEMA, result_mode=case.mode)
        result = executor.execute(sql)
        _validate_result(result, reference, case.kind, case.metric)
        detail = None
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
    return CaseResult(
        case.name, case.mode, case.question, perf_counter() - start,
        sql, result, detail is None, detail,
    )


def _run_repair_case(
    case: RepairCase, generator: SQLGenerator, executor: SQLExecutor,
) -> CaseResult:
    sql: str | None = None
    result: SQLQueryResult | None = None
    start = perf_counter()
    try:
        reference = executor.execute(case.reference_sql)
        start = perf_counter()
        sql = generator.repair(
            question=case.question,
            schema=FINANCIAL_SCHEMA,
            previous_sql=case.previous_sql,
            error_message=case.error_message,
            result_mode=case.mode,
        )
        result = executor.execute(sql)
        _validate_result(result, reference, case.kind, case.metric)
        detail = None
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
    return CaseResult(
        case.name, case.mode, case.question, perf_counter() - start,
        sql, result, detail is None, detail,
    )


def _print_case(result: CaseResult, *, cold_start: bool = False) -> None:
    print(f"CASE: {result.name}" + (" (cold start/model load possible)" if cold_start else ""))
    print(f"MODE: {result.mode}")
    print(f"QUESTION: {result.question}")
    print(f"SECONDS: {result.seconds:.3f}")
    print(f"GENERATED SQL: {result.generated_sql or '<none>'}")
    print(f"COLUMNS: {result.sql_result.columns if result.sql_result else '<none>'}")
    print(f"ROWS: {result.sql_result.rows if result.sql_result else '<none>'}")
    print(f"SEMANTIC RESULT: {'PASS' if result.passed else 'FAIL'}")
    if result.detail is not None:
        print(f"DETAIL: {result.detail}")
    print()


def main() -> int:
    if not DATABASE_PATH.is_file():
        print(f"Local financial database is missing: {DATABASE_PATH}")
        return 2

    generator = SQLGenerator(model=CANDIDATE_MODEL)
    executor = SQLExecutor(DATABASE_PATH)
    print(f"Candidate model: {CANDIDATE_MODEL}")
    print(f"Database: {DATABASE_PATH}")
    print("First generation case may include cold-start/model-load time.\n")

    generation_results = []
    for index, case in enumerate(CASES):
        result = _run_generation_case(case, generator, executor)
        generation_results.append(result)
        _print_case(result, cold_start=index == 0)

    print("Case | Mode | Seconds | Result")
    for result in generation_results:
        print(f"{result.name} | {result.mode} | {result.seconds:.3f} | "
              f"{'PASS' if result.passed else 'FAIL'}")
    passed = sum(result.passed for result in generation_results)
    print(f"SUMMARY: {passed}/{len(CASES)} passed")
    print(f"TOTAL SECONDS: {sum(result.seconds for result in generation_results):.3f}")
    print(f"WARM MEDIAN SECONDS: {median(r.seconds for r in generation_results[1:]):.3f}")
    print()

    print("REPAIR DIAGNOSTIC")
    repair_results = []
    for case in REPAIR_CASES:
        result = _run_repair_case(case, generator, executor)
        repair_results.append(result)
        _print_case(result)
    print(f"REPAIR SUMMARY: {sum(result.passed for result in repair_results)}/"
          f"{len(repair_results)} passed")
    return 0 if all(result.passed for result in (*generation_results, *repair_results)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
