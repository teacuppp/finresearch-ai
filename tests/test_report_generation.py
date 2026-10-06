"""Report generation composition and provenance tests."""

from dataclasses import FrozenInstanceError

import pytest

from app.agent.sql_executor import SQLQueryResult
from app.analysis.chart_renderer import ChartArtifact, ChartSpec, ChartType
from app.analysis.financial_analyzer import AnalysisOperation, AnalysisResult
from app.rag.models import RetrievedChunk
from app.report.evidence import ReportEvidence, ReportResearchResult
from app.report.executor import ReportResearchExecutionError
from app.report.generation import ReportGenerationResult, ReportGenerationService
from app.report.models import ReportPlan, ReportRAGScope, ReportTask
from app.report.planner import ReportPlanningError
from app.report.synthesis_evidence import ReportSynthesisEvidenceError
from app.report.synthesis_models import ReportClaim, ReportDraft, ReportSection
from app.report.synthesizer import ReportSynthesisError
from app.services.agent_service import AgentResult


class FakeResearchService:
    def __init__(self, response: ReportResearchResult | Exception) -> None:
        self.response = response
        self.calls: list[object] = []

    def research(self, request: str) -> ReportResearchResult:
        self.calls.append(request)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class FakeSynthesizer:
    def __init__(self, response: ReportDraft | Exception) -> None:
        self.response = response
        self.calls: list[object] = []

    def synthesize(self, evidence: object) -> ReportDraft:
        self.calls.append(evidence)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def _draft() -> ReportDraft:
    return ReportDraft(
        title="Apple research",
        sections=(
            ReportSection(
                heading="Findings",
                claims=(
                    ReportClaim(
                        task_id="risks",
                        text="Apple described business risks.",
                        evidence_ids=("E1",),
                    ),
                ),
            ),
        ),
    )


def _rag_source() -> RetrievedChunk:
    return RetrievedChunk(
        text="Apple described business risks.",
        document="apple-2025-10k.pdf",
        page=23,
        chunk_index=4,
        distance=0.1,
        company="Apple",
        ticker="AAPL",
        fiscal_year=2025,
        document_type="10K",
    )


def _rag_research() -> ReportResearchResult:
    plan = ReportPlan(
        title="Apple research",
        tasks=(
            ReportTask(
                task_id="risks",
                question="What risks did Apple describe?",
                expected_route="rag",
                rag_scope=ReportRAGScope(company="Apple"),
            ),
        ),
    )
    return ReportResearchResult(
        plan=plan,
        evidence=(
            ReportEvidence(
                task=plan.tasks[0],
                result=AgentResult(
                    route="rag", answer="Generated answer", sources=[_rag_source()]
                ),
            ),
        ),
    )


def _mixed_research() -> ReportResearchResult:
    plan = ReportPlan(
        title="Apple research",
        tasks=(
            ReportTask(
                task_id="risks",
                question="What risks did Apple describe?",
                expected_route="rag",
                rag_scope=ReportRAGScope(company="Apple"),
            ),
            ReportTask(
                task_id="revenue",
                question="Retrieve Apple revenue.",
                expected_route="sql",
            ),
            ReportTask(
                task_id="change",
                question="Calculate Apple revenue percentage change.",
                expected_route="sql",
            ),
        ),
    )
    chart = ChartArtifact(media_type="image/png", content=b"distinctive-chart-bytes")
    results = (
        AgentResult(route="rag", answer="Generated answer", sources=[_rag_source()]),
        AgentResult(
            route="sql",
            generated_sql="SELECT revenue_musd FROM financials",
            sql_result=SQLQueryResult(
                columns=["revenue_musd"],
                rows=[{"revenue_musd": 416161}],
                row_count=1,
            ),
        ),
        AgentResult(
            route="sql",
            generated_sql="SELECT fiscal_year, revenue_musd FROM financials",
            sql_result=SQLQueryResult(
                columns=["fiscal_year", "revenue_musd"],
                rows=[
                    {"fiscal_year": 2024, "revenue_musd": 391035},
                    {"fiscal_year": 2025, "revenue_musd": 416161},
                ],
                row_count=2,
            ),
            analysis_result=AnalysisResult(
                AnalysisOperation.PERCENTAGE_CHANGE, value=6.425511782832739
            ),
            chart_spec=ChartSpec(ChartType.BAR, "fiscal_year", "revenue_musd", "Revenue"),
            chart_artifact=chart,
        ),
    )
    return ReportResearchResult(
        plan=plan,
        evidence=tuple(
            ReportEvidence(task=task, result=result)
            for task, result in zip(plan.tasks, results, strict=True)
        ),
    )


def test_generate_preserves_request_research_evidence_and_draft_identity() -> None:
    request = "Research Apple's risks"
    research = _rag_research()
    draft = _draft()
    research_service = FakeResearchService(research)
    synthesizer = FakeSynthesizer(draft)

    result = ReportGenerationService(research_service, synthesizer).generate(request)

    assert len(research_service.calls) == 1
    assert research_service.calls[0] is request
    assert len(synthesizer.calls) == 1
    assert result.research is research
    assert result.synthesis_evidence is synthesizer.calls[0]
    assert result.synthesis_evidence.plan is research.plan
    assert [(item.evidence_id, item.task_id, item.kind) for item in result.synthesis_evidence.items] == [
        ("E1", "risks", "rag_source")
    ]
    assert result.draft is draft


def test_generate_retains_mixed_research_and_chart_provenance_by_identity() -> None:
    research = _mixed_research()
    original_results = tuple(evidence.result for evidence in research.evidence)
    chart = original_results[2].chart_artifact
    draft = _draft()
    synthesizer = FakeSynthesizer(draft)

    result = ReportGenerationService(FakeResearchService(research), synthesizer).generate(
        "Research Apple"
    )

    assert result.research is research
    assert tuple(evidence.result for evidence in result.research.evidence) == original_results
    assert all(
        result.research.evidence[index].result is original
        for index, original in enumerate(original_results)
    )
    assert result.research.evidence[0].result.sources is original_results[0].sources
    assert result.research.evidence[1].result.generated_sql == original_results[1].generated_sql
    assert result.research.evidence[1].result.sql_result is original_results[1].sql_result
    assert result.research.evidence[2].result.analysis_result is original_results[2].analysis_result
    assert result.research.evidence[2].result.chart_spec is original_results[2].chart_spec
    assert result.research.evidence[2].result.chart_artifact is chart
    assert chart is not None and chart.content == b"distinctive-chart-bytes"
    assert result.synthesis_evidence is synthesizer.calls[0]
    assert [item.kind for item in result.synthesis_evidence.items] == [
        "rag_source", "sql_result", "sql_result", "analysis_result"
    ]
    assert all("distinctive-chart-bytes" not in item.content for item in result.synthesis_evidence.items)


@pytest.mark.parametrize(
    "error",
    [
        ReportPlanningError("planning failed"),
        ReportResearchExecutionError("execution failed"),
        RuntimeError("unexpected research failure"),
    ],
)
def test_generate_propagates_research_failure_without_synthesis(error: Exception) -> None:
    research_service = FakeResearchService(error)
    synthesizer = FakeSynthesizer(_draft())

    with pytest.raises(type(error)) as caught:
        ReportGenerationService(research_service, synthesizer).generate("Research Apple")

    assert caught.value is error
    assert len(research_service.calls) == 1
    assert synthesizer.calls == []


def test_generate_propagates_evidence_build_failure_without_synthesis() -> None:
    valid = _rag_research()
    malformed = ReportResearchResult(plan=valid.plan, evidence=())
    research_service = FakeResearchService(malformed)
    synthesizer = FakeSynthesizer(_draft())

    with pytest.raises(ReportSynthesisEvidenceError, match="count"):
        ReportGenerationService(research_service, synthesizer).generate("Research Apple")

    assert len(research_service.calls) == 1
    assert synthesizer.calls == []


@pytest.mark.parametrize(
    "error",
    [ReportSynthesisError("synthesis failed"), RuntimeError("unexpected synthesis failure")],
)
def test_generate_propagates_synthesis_failure_without_retry(error: Exception) -> None:
    research_service = FakeResearchService(_rag_research())
    synthesizer = FakeSynthesizer(error)

    with pytest.raises(type(error)) as caught:
        ReportGenerationService(research_service, synthesizer).generate("Research Apple")

    assert caught.value is error
    assert len(research_service.calls) == 1
    assert len(synthesizer.calls) == 1


@pytest.mark.parametrize("candidate", ["", 42])
def test_generate_delegates_request_without_validation(candidate: object) -> None:
    error = ReportPlanningError("invalid request")
    research_service = FakeResearchService(error)
    synthesizer = FakeSynthesizer(_draft())

    with pytest.raises(ReportPlanningError) as caught:
        ReportGenerationService(research_service, synthesizer).generate(
            candidate  # type: ignore[arg-type]
        )

    assert caught.value is error
    assert len(research_service.calls) == 1
    assert research_service.calls[0] is candidate
    assert synthesizer.calls == []


def test_generation_result_is_frozen() -> None:
    result = ReportGenerationService(
        FakeResearchService(_rag_research()), FakeSynthesizer(_draft())
    ).generate("Research Apple")

    with pytest.raises(FrozenInstanceError):
        result.draft = _draft()  # type: ignore[misc]
    assert isinstance(result, ReportGenerationResult)
