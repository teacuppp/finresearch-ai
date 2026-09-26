from collections.abc import Iterator

import pytest

from app.agent.sql_executor import SQLQueryResult
from app.services.agent_service import AgentResult
from scripts.profile_agent_latency import (
    CASES,
    CaseReport,
    ProfileCase,
    TimingCollector,
    TimingRecord,
    format_case_report,
    format_overall_summary,
    percentage,
    profile_case,
    summarize,
)


def clock_from(values: list[float]):
    times: Iterator[float] = iter(values)
    return lambda: next(times)


class Worker:
    def __init__(self, result: object = None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def run(self, *args: object, **kwargs: object) -> object:
        self.calls.append((args, kwargs))
        if self.error is not None:
            raise self.error
        return self.result


def test_successful_wrapper_preserves_args_kwargs_result_identity_and_record():
    result = {"rows": [1, 2]}
    worker = Worker(result)
    collector = TimingCollector(clock_from([10.0, 10.25]))
    args = ([1, 2],)
    options = {"limit": 3}

    collector.wrap_method(worker, "run", "sql.execute")
    returned = worker.run(*args, **options)

    assert returned is result
    assert result == {"rows": [1, 2]}
    assert worker.calls == [(args, options)]
    assert worker.calls[0][0][0] is args[0]
    assert collector.records == [TimingRecord(
        stage="sql.execute", call_index=1, elapsed_seconds=0.25,
        succeeded=True, order=0,
    )]


def test_failed_wrapper_preserves_exception_identity_and_records_failure():
    error = RuntimeError("private detail")
    worker = Worker(error=error)
    collector = TimingCollector(clock_from([2.0, 2.5]))
    collector.wrap_method(worker, "run", "chart.render")

    with pytest.raises(RuntimeError) as raised:
        worker.run("same input", key="same value")

    assert raised.value is error
    assert worker.calls == [(("same input",), {"key": "same value"})]
    assert collector.records == [TimingRecord(
        stage="chart.render", call_index=1, elapsed_seconds=0.5,
        succeeded=False, order=0,
    )]


def test_calls_increment_indexes_and_aggregate_in_first_execution_order():
    sql = Worker(result=object())
    chart = Worker(result=object())
    collector = TimingCollector(clock_from([1, 1.2, 2, 2.3, 3, 3.4]))
    collector.wrap_method(sql, "run", "sql.execute")
    collector.wrap_method(chart, "run", "chart.render")

    sql.run()
    chart.run()
    sql.run()

    assert [(record.stage, record.call_index, record.order) for record in collector.records] == [
        ("sql.execute", 1, 0), ("chart.render", 1, 1), ("sql.execute", 2, 2),
    ]
    stages = summarize(collector.records)
    assert [stage.stage for stage in stages] == ["sql.execute", "chart.render"]
    assert [stage.calls for stage in stages] == [2, 1]
    assert stages[0].seconds == pytest.approx(0.6)
    assert stages[1].seconds == pytest.approx(0.3)
    assert [stage.failures for stage in stages] == [0, 0]


def test_failed_and_successful_repeated_calls_are_both_included_in_aggregate():
    worker = Worker(result="ok", error=ValueError("first"))
    collector = TimingCollector(clock_from([0, 0.1, 1, 1.2]))
    collector.wrap_method(worker, "run", "sql.execute")

    with pytest.raises(ValueError):
        worker.run()
    worker.error = None
    assert worker.run() == "ok"

    summary = summarize(collector.records)
    assert len(summary) == 1
    assert summary[0].calls == 2
    assert summary[0].seconds == pytest.approx(0.3)
    assert summary[0].failures == 1
    assert [record.call_index for record in collector.records] == [1, 2]


@pytest.mark.parametrize("total", [0.0, -1.0])
def test_percentage_handles_zero_or_invalid_total_without_division(total):
    assert percentage(0.1, total) == 0.0


def test_percentage_and_reports_handle_very_small_total():
    assert percentage(1e-9, 2e-9) == pytest.approx(50.0)
    report = CaseReport(
        case=CASES[0], total_seconds=2e-9,
        records=(TimingRecord("sql.execute", 1, 1e-9, True, 0),),
    )

    case_text = format_case_report(report)
    overall_text = format_overall_summary([report])
    assert "50.0" in case_text
    assert "50.0%" in overall_text
    assert "UNACCOUNTED / GRAPH OVERHEAD" in case_text


def test_case_report_includes_multiple_call_details_and_measured_totals():
    report = CaseReport(
        case=CASES[0], total_seconds=1.0,
        records=(
            TimingRecord("sql.execute", 1, 0.1, False, 0),
            TimingRecord("sql.repair", 1, 0.2, True, 1),
            TimingRecord("sql.execute", 2, 0.3, True, 2),
        ),
    )

    text = format_case_report(report)

    assert text.index("sql.execute") < text.index("sql.repair")
    assert "call #1: 0.100s failed" in text
    assert "call #2: 0.300s ok" in text
    assert "MEASURED STAGE TOTAL: 0.600s" in text
    assert "UNACCOUNTED / GRAPH OVERHEAD: 0.400s" in text


def test_overall_summary_sorts_top_stages_by_cumulative_seconds():
    first = CaseReport(
        ProfileCase("first", "Question 1"), 1.0,
        (TimingRecord("router.route", 1, 0.4, True, 0),
         TimingRecord("sql.execute", 1, 0.1, True, 1)),
    )
    second = CaseReport(
        ProfileCase("second", "Question 2"), 1.0,
        (TimingRecord("router.route", 2, 0.3, True, 2),
         TimingRecord("sql.execute", 2, 0.2, True, 3)),
    )

    text = format_overall_summary([first, second])
    top = text.split("TOP STAGES ACROSS ALL CASES", 1)[1]
    assert top.index("router.route") < top.index("sql.execute")
    assert "first" in text and "second" in text


class FakeAgent:
    def __init__(self, result: AgentResult | Exception) -> None:
        self.result = result
        self.questions: list[str] = []

    def ask(self, question: str) -> AgentResult:
        self.questions.append(question)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def lookup_result(value: int) -> AgentResult:
    return AgentResult(
        route="sql", generated_sql="SELECT revenue_musd FROM financial_metrics",
        sql_result=SQLQueryResult(
            columns=["revenue_musd"], rows=[{"revenue_musd": value}], row_count=1,
        ),
    )


def test_profile_case_checks_business_result_and_records_failure():
    agent = FakeAgent(lookup_result(1))
    collector = TimingCollector(clock_from([]))

    report = profile_case(CASES[0], agent, collector, clock=clock_from([1, 2]))

    assert agent.questions == [CASES[0].question]
    assert report.total_seconds == 1.0
    assert report.failure == "Expected Apple 2025 revenue"
    assert "STATUS: FAILED" in format_case_report(report)


def test_profile_case_hides_private_exception_text():
    agent = FakeAgent(RuntimeError("secret model response"))
    report = profile_case(
        CASES[0], agent, TimingCollector(clock_from([])), clock=clock_from([1, 2]),
    )

    assert report.failure == "Agent call raised RuntimeError"
    assert "secret model response" not in format_case_report(report)


def test_profile_case_accepts_correct_lookup_without_mutation():
    result = lookup_result(416161)
    agent = FakeAgent(result)

    report = profile_case(
        CASES[0], agent, TimingCollector(clock_from([])), clock=clock_from([1, 2]),
    )

    assert report.failure is None
    assert result.sql_result.rows == [{"revenue_musd": 416161}]
    assert "STATUS: PASS" in format_case_report(report)
