"""Profile one real SQL Agent pass. Run: python -m scripts.profile_agent_latency."""

from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import wraps
from math import isclose
from pathlib import Path
from time import perf_counter
from typing import Any

from app.agent.graph import build_agent_graph
from app.agent.router import LLMQuestionRouter
from app.agent.sql_executor import SQLExecutor, SQLQueryResult
from app.agent.sql_generator import SQLGenerator
from app.analysis.chart_data import ChartDataBuilder
from app.analysis.chart_decision import ChartDecisionPolicy, ChartIntentClassifier
from app.analysis.chart_planner import ChartPlanner
from app.analysis.chart_renderer import ChartArtifact, ChartRenderer, ChartSpec
from app.analysis.financial_analyzer import AnalysisOperation, AnalysisResult, FinancialAnalyzer
from app.analysis.intent import SQLAnalysisClassifier
from app.analysis.planner import AnalysisPlanner
from app.services.agent_service import AgentResult, AgentService


DATABASE_PATH = Path(__file__).resolve().parents[1] / "data/financial_demo.db"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@dataclass(frozen=True)
class TimingRecord:
    stage: str
    call_index: int
    elapsed_seconds: float
    succeeded: bool
    order: int
    nested: bool = False


@dataclass(frozen=True)
class StageSummary:
    stage: str
    calls: int
    seconds: float
    failures: int


class TimingCollector:
    """Wrap bound instance methods and retain one timing record per call."""

    def __init__(self, clock: Callable[[], float] = perf_counter) -> None:
        self.clock = clock
        self.records: list[TimingRecord] = []
        self._counts: dict[str, int] = {}
        self._next_order = 0
        self._active = 0

    def wrap_method(self, instance: object, method_name: str, stage: str) -> None:
        original = getattr(instance, method_name)

        @wraps(original)
        def timed(*args: Any, **kwargs: Any) -> Any:
            self._counts[stage] = self._counts.get(stage, 0) + 1
            call_index = self._counts[stage]
            order = self._next_order
            self._next_order += 1
            nested = self._active > 0
            start = self.clock()
            self._active += 1
            succeeded = False
            try:
                result = original(*args, **kwargs)
                succeeded = True
                return result
            finally:
                elapsed = max(0.0, self.clock() - start)
                self._active -= 1
                self.records.append(TimingRecord(
                    stage=stage,
                    call_index=call_index,
                    elapsed_seconds=elapsed,
                    succeeded=succeeded,
                    order=order,
                    nested=nested,
                ))

        setattr(instance, method_name, timed)


def summarize(records: Sequence[TimingRecord]) -> list[StageSummary]:
    stages: OrderedDict[str, tuple[int, float, int]] = OrderedDict()
    for record in sorted(records, key=lambda item: item.order):
        calls, seconds, failures = stages.get(record.stage, (0, 0.0, 0))
        stages[record.stage] = (
            calls + 1,
            seconds + record.elapsed_seconds,
            failures + (not record.succeeded),
        )
    return [
        StageSummary(stage, calls, seconds, failures)
        for stage, (calls, seconds, failures) in stages.items()
    ]


def percentage(seconds: float, total_seconds: float) -> float:
    return 100.0 * seconds / total_seconds if total_seconds > 0 else 0.0


@dataclass(frozen=True)
class ProfileCase:
    name: str
    question: str


CASES = (
    ProfileCase("ordinary_lookup", "What was Apple's 2025 revenue?"),
    ProfileCase("explicit_single_plot", "Plot Apple's 2025 revenue."),
    ProfileCase(
        "percentage_change",
        "By what percentage did Apple's revenue change from 2024 to 2025?",
    ),
    ProfileCase(
        "ranking",
        "Rank Apple and Microsoft by 2025 revenue from highest to lowest.",
    ),
)


@dataclass(frozen=True)
class CaseReport:
    case: ProfileCase
    total_seconds: float
    records: tuple[TimingRecord, ...]
    failure: str | None = None


class ProfileCaseError(AssertionError):
    """A live case failed a required business-result check."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProfileCaseError(message)


def _require_revenue_row(result: SQLQueryResult) -> None:
    _require(result.row_count == 1 and len(result.rows) == 1, "Expected one revenue row")
    _require(result.rows[0].get("revenue_musd") == 416161, "Expected Apple 2025 revenue")


def _require_chart(result: AgentResult) -> None:
    _require(isinstance(result.chart_spec, ChartSpec), "Expected chart specification")
    _require(isinstance(result.chart_artifact, ChartArtifact), "Expected PNG chart")
    _require(result.chart_artifact.media_type == "image/png", "Expected PNG media type")
    _require(result.chart_artifact.content.startswith(PNG_SIGNATURE), "Expected PNG bytes")
    _require(result.chart_spec.y_column == "revenue_musd", "Expected revenue chart metric")


def validate_case(case: ProfileCase, result: AgentResult) -> None:
    """Reject incorrect live results before treating their timings as valid."""
    _require(result.route == "sql", "Expected SQL route")
    _require(isinstance(result.sql_result, SQLQueryResult), "Expected SQL rows")
    _require(bool(result.generated_sql), "Expected generated SQL")
    sql_result = result.sql_result

    if case.name in ("ordinary_lookup", "explicit_single_plot"):
        _require(result.analysis_result is None, "Expected direct SQL result")
        _require_revenue_row(sql_result)
        if case.name == "ordinary_lookup":
            _require(
                result.chart_spec is None and result.chart_artifact is None,
                "Ordinary lookup should have no chart",
            )
        else:
            _require_chart(result)
    elif case.name == "percentage_change":
        analysis = result.analysis_result
        _require(isinstance(analysis, AnalysisResult), "Expected analysis result")
        _require(
            analysis.operation is AnalysisOperation.PERCENTAGE_CHANGE,
            "Expected percentage_change analysis",
        )
        values = {row.get("fiscal_year"): row.get("revenue_musd") for row in sql_result.rows}
        _require(
            values == {2024: 391035, 2025: 416161} and len(sql_result.rows) == 2,
            "Expected Apple 2024 and 2025 raw revenue",
        )
        expected = ((416161 - 391035) / 391035) * 100
        _require(
            isinstance(analysis.value, (int, float))
            and isclose(analysis.value, expected, rel_tol=1e-6),
            "Expected Apple revenue percentage change",
        )
        _require(
            result.chart_spec is None and result.chart_artifact is None,
            "Percentage change should have no chart",
        )
    elif case.name == "ranking":
        analysis = result.analysis_result
        _require(isinstance(analysis, AnalysisResult), "Expected ranking result")
        _require(analysis.operation is AnalysisOperation.RANKING, "Expected ranking analysis")
        _require(len(analysis.ranked_rows) == 2, "Expected two ranked rows")
        names = [row.get("ticker", row.get("company")) for row in analysis.ranked_rows]
        _require(names in (["AAPL", "MSFT"], ["Apple", "Microsoft"]), "Expected Apple before Microsoft")
        _require(
            [row.get("revenue_musd") for row in analysis.ranked_rows] == [416161, 281724],
            "Expected ranked company revenue values",
        )
        _require_chart(result)
    else:
        raise ProfileCaseError("Unknown profile case")


class FailIfRAG:
    def ask(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Unexpected RAG route in SQL latency profile")


def create_profiled_agent(collector: TimingCollector) -> AgentService:
    """Compose the real SQL Agent once, then time calls on its own instances."""
    if not DATABASE_PATH.is_file():
        raise FileNotFoundError(f"Local financial database is missing: {DATABASE_PATH}")

    router = LLMQuestionRouter()
    sql_analysis_classifier = SQLAnalysisClassifier()
    sql_generator = SQLGenerator()
    sql_executor = SQLExecutor(database_path=DATABASE_PATH)
    analysis_planner = AnalysisPlanner()
    financial_analyzer = FinancialAnalyzer()
    chart_data_builder = ChartDataBuilder()
    chart_intent_classifier = ChartIntentClassifier()
    chart_decision_policy = ChartDecisionPolicy()
    chart_planner = ChartPlanner()
    chart_renderer = ChartRenderer()

    for component, method, stage in (
        (router, "route", "router.route"),
        (sql_analysis_classifier, "classify", "sql_analysis.classify"),
        (chart_intent_classifier, "classify", "chart_intent.classify"),
        (sql_generator, "generate", "sql.generate"),
        (sql_generator, "generate_analysis_data", "sql.generate_analysis_data"),
        (sql_generator, "repair", "sql.repair"),
        (sql_generator, "repair_analysis_data", "sql.repair_analysis_data"),
        (analysis_planner, "plan", "analysis.plan"),
        (chart_planner, "plan", "chart.plan"),
        (sql_executor, "execute", "sql.execute"),
        (financial_analyzer, "percentage_change", "financial.percentage_change"),
        (financial_analyzer, "absolute_change", "financial.absolute_change"),
        (financial_analyzer, "difference", "financial.difference"),
        (financial_analyzer, "ranking", "financial.ranking"),
        (chart_data_builder, "build", "chart_data.build"),
        (chart_decision_policy, "decide", "chart_decision.decide"),
        (chart_renderer, "render", "chart.render"),
    ):
        collector.wrap_method(component, method, stage)

    graph = build_agent_graph(
        router=router,
        query_service=FailIfRAG(),  # type: ignore[arg-type]
        sql_generator=sql_generator,
        sql_executor=sql_executor,
        sql_analysis_classifier=sql_analysis_classifier,
        analysis_planner=analysis_planner,
        financial_analyzer=financial_analyzer,
        chart_data_builder=chart_data_builder,
        chart_intent_classifier=chart_intent_classifier,
        chart_decision_policy=chart_decision_policy,
        chart_planner=chart_planner,
        chart_renderer=chart_renderer,
    )
    return AgentService(graph=graph)


def profile_case(
    case: ProfileCase,
    agent: AgentService,
    collector: TimingCollector,
    clock: Callable[[], float] = perf_counter,
) -> CaseReport:
    first_record = len(collector.records)
    start = clock()
    failure: str | None = None
    try:
        validate_case(case, agent.ask(case.question))
    except ProfileCaseError as exc:
        failure = str(exc)
    except Exception as exc:
        failure = f"Agent call raised {type(exc).__name__}"
    elapsed = max(0.0, clock() - start)
    return CaseReport(
        case=case,
        total_seconds=elapsed,
        records=tuple(collector.records[first_record:]),
        failure=failure,
    )


def format_case_report(report: CaseReport) -> str:
    stages = summarize(report.records)
    measured = sum(stage.seconds for stage in stages)
    lines = [
        f"CASE: {report.case.name}",
        f"QUESTION: {report.case.question}",
        f"TOTAL SECONDS: {report.total_seconds:.3f}",
        f"STATUS: {'FAILED: ' + report.failure if report.failure else 'PASS'}",
        "Stage                         Calls    Seconds    % Total    Status",
    ]
    for stage in stages:
        status = "failed" if stage.failures else "ok"
        lines.append(
            f"{stage.stage:<29} {stage.calls:>5} {stage.seconds:>10.3f} "
            f"{percentage(stage.seconds, report.total_seconds):>10.1f} {status:>9}"
        )
        records = sorted(
            (record for record in report.records if record.stage == stage.stage),
            key=lambda item: item.order,
        )
        if len(records) > 1:
            for record in records:
                status = "ok" if record.succeeded else "failed"
                lines.append(
                    f"  call #{record.call_index}: {record.elapsed_seconds:.3f}s {status}"
                )
    lines.extend((
        f"MEASURED STAGE TOTAL: {measured:.3f}s",
        f"UNACCOUNTED / GRAPH OVERHEAD: {report.total_seconds - measured:.3f}s",
    ))
    if any(record.nested for record in report.records):
        lines.append("Nested stage timings overlap; percentages are inclusive.")
    return "\n".join(lines)


def format_overall_summary(reports: Sequence[CaseReport]) -> str:
    lines = [
        "OVERALL SUMMARY",
        "Case                    Total(s)   Slowest stage               Stage(s)   Share   Status",
    ]
    for report in reports:
        stages = summarize(report.records)
        slowest = max(stages, key=lambda item: item.seconds) if stages else None
        stage_name = slowest.stage if slowest else "—"
        stage_seconds = slowest.seconds if slowest else 0.0
        status = "FAIL" if report.failure else "PASS"
        lines.append(
            f"{report.case.name:<23} {report.total_seconds:>8.3f}   "
            f"{stage_name:<27} {stage_seconds:>8.3f} "
            f"{percentage(stage_seconds, report.total_seconds):>6.1f}% {status:>6}"
        )

    all_records = [record for report in reports for record in report.records]
    stages = sorted(summarize(all_records), key=lambda item: item.seconds, reverse=True)
    lines.extend(("", "TOP STAGES ACROSS ALL CASES", "Stage                         Calls    Seconds"))
    for stage in stages:
        lines.append(f"{stage.stage:<29} {stage.calls:>5} {stage.seconds:>10.3f}")
    return "\n".join(lines)


def main() -> int:
    collector = TimingCollector()
    agent = create_profiled_agent(collector)
    print("Execution order: ordinary_lookup, explicit_single_plot, percentage_change, ranking")
    print("Cold-start note: the first case may include model warm-up or loading time.")
    reports: list[CaseReport] = []
    for index, case in enumerate(CASES, start=1):
        print(f"\nRUN {index}/{len(CASES)}")
        report = profile_case(case, agent, collector)
        reports.append(report)
        print(format_case_report(report), flush=True)
    print("\n" + format_overall_summary(reports))
    return 1 if any(report.failure for report in reports) else 0


if __name__ == "__main__":
    raise SystemExit(main())
