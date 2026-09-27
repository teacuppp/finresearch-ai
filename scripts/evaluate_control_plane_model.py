"""Evaluate one candidate control-plane model against curated live cases.

Run: python -m scripts.evaluate_control_plane_model
"""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from statistics import median
from time import perf_counter

from app.agent.router import LLMQuestionRouter
from app.agent.sql_executor import SQLQueryResult
from app.analysis.chart_decision import ChartIntentClassifier
from app.analysis.chart_planner import ChartPlanner
from app.analysis.financial_analyzer import AnalysisOperation
from app.analysis.intent import SQLAnalysisClassifier
from app.analysis.planner import AnalysisPlanner
from app.analysis.retrieval import AnalysisRetrievalPlanner


CANDIDATE_MODEL = "qwen3:4b-instruct"
COMPONENT_TOTALS = (
    ("Router", 8),
    ("SQL Analysis", 7),
    ("Retrieval", 6),
    ("Analysis Planner", 4),
    ("Chart Intent", 6),
    ("Chart Planner", 4),
)


def _identity(value: object) -> object:
    return value


def _equal(actual: object, expected: object) -> bool:
    return actual == expected


def _structured(value: object) -> object:
    return value.model_dump(mode="json")  # type: ignore[attr-defined]


def _chart_intent(value: object) -> object:
    return value.intent  # type: ignore[attr-defined]


def _chart_spec(value: object) -> object:
    return {
        "chart_type": value.chart_type.value,  # type: ignore[attr-defined]
        "x_column": value.x_column,  # type: ignore[attr-defined]
        "y_column": value.y_column,  # type: ignore[attr-defined]
    }


def _retrieval_matches(actual: object, expected: object) -> bool:
    if not isinstance(actual, dict) or not isinstance(expected, dict):
        return False
    if actual.get("entity_column") != expected.get("entity_column"):
        return False
    if actual.get("metric") != expected.get("metric"):
        return False
    for field in ("entities", "fiscal_years"):
        actual_values = actual.get(field)
        expected_values = expected.get(field)
        if not isinstance(actual_values, list) or not isinstance(expected_values, list):
            return False
        if len(actual_values) != len(expected_values) or set(actual_values) != set(expected_values):
            return False
    return True


def _chart_matches(actual: object, expected: object) -> bool:
    if not isinstance(actual, dict) or not isinstance(expected, dict):
        return False
    x_expected = expected.get("x_column")
    valid_x = (
        actual.get("x_column") in x_expected
        if isinstance(x_expected, tuple)
        else actual.get("x_column") == x_expected
    )
    return (
        actual.get("chart_type") == expected.get("chart_type")
        and valid_x
        and actual.get("y_column") == expected.get("y_column")
    )


@dataclass(frozen=True)
class EvaluationCase:
    component: str
    name: str
    expected: object
    invoke: Callable[[], object]
    project: Callable[[object], object] = _identity
    matches: Callable[[object, object], bool] = _equal


@dataclass(frozen=True)
class CaseResult:
    component: str
    name: str
    seconds: float
    expected: object
    actual: object
    passed: bool


def _sql_result(columns: list[str], rows: list[dict[str, object]]) -> SQLQueryResult:
    return SQLQueryResult(columns=columns, rows=rows, row_count=len(rows))


def _cases() -> list[EvaluationCase]:
    router = LLMQuestionRouter(model=CANDIDATE_MODEL)
    sql_analysis = SQLAnalysisClassifier(model=CANDIDATE_MODEL)
    retrieval = AnalysisRetrievalPlanner(model=CANDIDATE_MODEL)
    analysis_planner = AnalysisPlanner(model=CANDIDATE_MODEL)
    chart_intent = ChartIntentClassifier(model=CANDIDATE_MODEL)
    chart_planner = ChartPlanner(model=CANDIDATE_MODEL)
    cases: list[EvaluationCase] = []

    for name, question, expected in (
        ("sql_lookup", "What was Apple's 2025 revenue?", "sql"),
        ("sql_comparison", "Compare Apple and Microsoft net income in 2025.", "sql"),
        ("sql_ranking", "Rank Apple and Microsoft by 2025 revenue.", "sql"),
        ("sql_average", "What was the average 2025 revenue of Apple and Microsoft?", "sql"),
        ("rag_risk", "What risks did Apple disclose?", "rag"),
        ("rag_strategy", "How does Microsoft describe its AI strategy?", "rag"),
        ("rag_driver", "What factors affected Apple's Services business?", "rag"),
        ("rag_why", "Why did Apple's revenue change?", "rag"),
    ):
        cases.append(EvaluationCase(
            "Router", name, expected, partial(router.route, question),
        ))

    for name, question, expected in (
        ("lookup", "What was Apple's revenue in 2025?", {"mode": "direct"}),
        ("average", "What was the average revenue of Apple and Microsoft in 2025?",
         {"mode": "direct"}),
        ("count", "How many financial metric rows exist?", {"mode": "direct"}),
        ("absolute_change", "How much did Apple's revenue increase from 2024 to 2025?",
         {"mode": "analysis", "operation": "absolute_change"}),
        ("percentage_change", "By what percentage did Apple's revenue change from 2024 to 2025?",
         {"mode": "analysis", "operation": "percentage_change"}),
        ("difference", "What is the difference between Apple's and Microsoft's 2025 revenue?",
         {"mode": "analysis", "operation": "difference"}),
        ("ranking", "Rank Apple and Microsoft by 2025 revenue from highest to lowest.",
         {"mode": "analysis", "operation": "ranking"}),
    ):
        cases.append(EvaluationCase(
            "SQL Analysis", name, expected, partial(sql_analysis.classify, question),
            _structured,
        ))

    for name, question, operation, expected in (
        ("absolute_company", "How much did Apple's revenue increase from 2024 to 2025?",
         AnalysisOperation.ABSOLUTE_CHANGE,
         {"entity_column": "company", "entities": ["Apple"],
          "fiscal_years": [2024, 2025], "metric": "revenue_musd"}),
        ("percentage_ticker", "By what percentage did AAPL's net income change from 2024 to 2025?",
         AnalysisOperation.PERCENTAGE_CHANGE,
         {"entity_column": "ticker", "entities": ["AAPL"],
          "fiscal_years": [2024, 2025], "metric": "net_income_musd"}),
        ("difference_company", "What is the difference between Apple's and Microsoft's 2025 revenue?",
         AnalysisOperation.DIFFERENCE,
         {"entity_column": "company", "entities": ["Apple", "Microsoft"],
          "fiscal_years": [2025], "metric": "revenue_musd"}),
        ("difference_ticker", "What is the difference between AAPL's and MSFT's 2025 gross margin?",
         AnalysisOperation.DIFFERENCE,
         {"entity_column": "ticker", "entities": ["AAPL", "MSFT"],
          "fiscal_years": [2025], "metric": "gross_margin"}),
        ("ranking_company", "Rank Apple and Microsoft by 2025 revenue from highest to lowest.",
         AnalysisOperation.RANKING,
         {"entity_column": "company", "entities": ["Apple", "Microsoft"],
          "fiscal_years": [2025], "metric": "revenue_musd"}),
        ("ranking_ticker", "Rank AAPL and MSFT by 2025 operating income from highest to lowest.",
         AnalysisOperation.RANKING,
         {"entity_column": "ticker", "entities": ["AAPL", "MSFT"],
          "fiscal_years": [2025], "metric": "operating_income_musd"}),
    ):
        cases.append(EvaluationCase(
            "Retrieval", name, expected, partial(retrieval.plan, question, operation),
            _structured, _retrieval_matches,
        ))

    apple_years = _sql_result(
        ["company", "fiscal_year", "revenue_musd"],
        [
            {"company": "Apple", "fiscal_year": 2024, "revenue_musd": 391035},
            {"company": "Apple", "fiscal_year": 2025, "revenue_musd": 416161},
        ],
    )
    comparison = _sql_result(
        ["company", "fiscal_year", "revenue_musd"],
        [
            {"company": "Apple", "fiscal_year": 2025, "revenue_musd": 416161},
            {"company": "Microsoft", "fiscal_year": 2025, "revenue_musd": 281724},
        ],
    )
    for name, question, sql_result, expected in (
        ("absolute_change", "How much did Apple's revenue increase from 2024 to 2025?",
         apple_years,
         {"operation": "absolute_change", "old_row": 0, "old_column": "revenue_musd",
          "new_row": 1, "new_column": "revenue_musd"}),
        ("percentage_change", "By what percentage did Apple's revenue change from 2024 to 2025?",
         apple_years,
         {"operation": "percentage_change", "old_row": 0, "old_column": "revenue_musd",
          "new_row": 1, "new_column": "revenue_musd"}),
        ("difference", "What is the difference between Apple's and Microsoft's 2025 revenue?",
         comparison,
         {"operation": "difference", "left_row": 0, "left_column": "revenue_musd",
          "right_row": 1, "right_column": "revenue_musd"}),
        ("ranking", "Rank Apple and Microsoft by 2025 revenue from highest to lowest.",
         comparison,
         {"operation": "ranking", "column": "revenue_musd", "direction": "descending"}),
    ):
        cases.append(EvaluationCase(
            "Analysis Planner", name, expected,
            partial(analysis_planner.plan, question, sql_result), _structured,
        ))

    for name, question, expected in (
        ("plain_lookup", "What was Apple's 2025 revenue?", "lookup"),
        ("explicit_plot_lookup", "Plot Apple's 2025 revenue.", "lookup"),
        ("trend", "Plot Apple's revenue trend from 2024 to 2025.", "trend"),
        ("comparison", "Compare Apple and Microsoft 2025 revenue.", "comparison"),
        ("ranking", "Rank Apple and Microsoft by 2025 revenue.", "ranking"),
        ("list", "List Apple and Microsoft revenue rows.", "list"),
    ):
        cases.append(EvaluationCase(
            "Chart Intent", name, expected,
            partial(chart_intent.classify, question), _chart_intent,
        ))

    single_chart = _sql_result(
        ["company", "fiscal_year", "revenue_musd"],
        [{"company": "Apple", "fiscal_year": 2025, "revenue_musd": 416161}],
    )
    comparison_chart = _sql_result(
        ["company", "ticker", "fiscal_year", "revenue_musd"],
        [
            {"company": "Apple", "ticker": "AAPL", "fiscal_year": 2025,
             "revenue_musd": 416161},
            {"company": "Microsoft", "ticker": "MSFT", "fiscal_year": 2025,
             "revenue_musd": 281724},
        ],
    )
    trend_chart = _sql_result(
        ["fiscal_year", "revenue_musd"],
        [
            {"fiscal_year": 2024, "revenue_musd": 391035},
            {"fiscal_year": 2025, "revenue_musd": 416161},
        ],
    )
    ranking_chart = _sql_result(
        ["company", "revenue_musd"],
        [
            {"company": "Apple", "revenue_musd": 416161},
            {"company": "Microsoft", "revenue_musd": 281724},
        ],
    )
    for name, question, sql_result, expected in (
        ("single_plot", "Plot Apple's 2025 revenue.", single_chart,
         {"chart_type": "bar", "x_column": "company", "y_column": "revenue_musd"}),
        ("comparison", "Compare Apple and Microsoft 2025 revenue.", comparison_chart,
         {"chart_type": "bar", "x_column": ("company", "ticker"),
          "y_column": "revenue_musd"}),
        ("trend", "Plot Apple's revenue trend from 2024 to 2025.", trend_chart,
         {"chart_type": "line", "x_column": "fiscal_year", "y_column": "revenue_musd"}),
        ("ranking", "Visualize the ranking of Apple and Microsoft by 2025 revenue.",
         ranking_chart,
         {"chart_type": "bar", "x_column": "company", "y_column": "revenue_musd"}),
    ):
        cases.append(EvaluationCase(
            "Chart Planner", name, expected,
            partial(chart_planner.plan, question, sql_result), _chart_spec, _chart_matches,
        ))
    return cases


def _evaluate(case: EvaluationCase) -> CaseResult:
    start = perf_counter()
    try:
        actual = case.project(case.invoke())
        passed = case.matches(actual, case.expected)
    except Exception as exc:
        actual = f"{type(exc).__name__}: {exc}"
        passed = False
    return CaseResult(
        case.component, case.name, perf_counter() - start,
        case.expected, actual, passed,
    )


def main() -> int:
    cases = _cases()
    assert len(cases) == 35
    print(f"Candidate model: {CANDIDATE_MODEL}", flush=True)
    print("First call may include candidate-model loading.\n", flush=True)
    results: list[CaseResult] = []
    for case in cases:
        result = _evaluate(case)
        results.append(result)
        print(
            f"COMPONENT: {result.component}\n"
            f"CASE: {result.name}\n"
            f"SECONDS: {result.seconds:.3f}\n"
            f"EXPECTED: {result.expected!r}\n"
            f"ACTUAL: {result.actual!r}\n"
            f"{'PASS' if result.passed else 'FAIL'}\n",
            flush=True,
        )

    for component, total in COMPONENT_TOTALS:
        component_results = [result for result in results if result.component == component]
        assert len(component_results) == total
        print(f"{component}: {sum(result.passed for result in component_results)}/{total}")
    print(f"TOTAL: {sum(result.passed for result in results)}/{len(results)}")
    print(f"TOTAL SECONDS: {sum(result.seconds for result in results):.3f}")
    print(f"WARM MEDIAN SECONDS: {median(result.seconds for result in results[1:]):.3f}")
    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
