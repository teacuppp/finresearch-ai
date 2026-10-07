"""Deterministic, safe Markdown rendering of report generation results."""

import base64
from dataclasses import FrozenInstanceError, replace

import pytest

from app.agent.sql_executor import SQLQueryResult
from app.analysis.chart_renderer import ChartArtifact, ChartSpec, ChartType
from app.analysis.financial_analyzer import AnalysisOperation, AnalysisResult
from app.rag.models import RetrievedChunk
from app.report.evidence import ReportEvidence, ReportResearchResult
from app.report.generation import ReportGenerationResult
from app.report.markdown_renderer import (
    ReportMarkdownArtifact,
    ReportMarkdownRenderer,
    ReportMarkdownRenderingError,
)
from app.report.models import ReportPlan, ReportRAGScope, ReportTask
from app.report.synthesis_evidence import build_synthesis_evidence
from app.report.synthesis_models import ReportClaim, ReportDraft, ReportSection
from app.services.agent_service import AgentResult


def _rag_task(task_id: str = "risks") -> ReportTask:
    return ReportTask(
        task_id=task_id,
        question=f"What risks did Apple describe for {task_id}?",
        expected_route="rag",
        rag_scope=ReportRAGScope(company="Apple"),
    )


def _sql_task(task_id: str = "revenue") -> ReportTask:
    return ReportTask(
        task_id=task_id,
        question=f"Retrieve {task_id}.",
        expected_route="sql",
    )


def _source(
    text: str = "Source disclosure.",
    *,
    document: str = "apple-2025-10k.pdf",
    chunk_index: int = 4,
    company: str | None = "Apple",
    ticker: str | None = "AAPL",
    fiscal_year: int | None = 2025,
    document_type: str | None = "10K",
) -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        document=document,
        page=23,
        chunk_index=chunk_index,
        distance=0.1,
        company=company,
        ticker=ticker,
        fiscal_year=fiscal_year,
        document_type=document_type,
    )


def _rag_result(*sources: RetrievedChunk, answer: str = "Generated RAG answer") -> AgentResult:
    return AgentResult(route="rag", answer=answer, sources=list(sources))


def _sql_result(
    *,
    columns: list[str] | None = None,
    rows: list[dict] | None = None,
) -> SQLQueryResult:
    actual_columns = ["company", "revenue_musd"] if columns is None else columns
    actual_rows = (
        [{"company": "Apple", "revenue_musd": 416161}] if rows is None else rows
    )
    return SQLQueryResult(
        columns=actual_columns,
        rows=actual_rows,
        row_count=len(actual_rows),
    )


def _claim(task_id: str, text: str, *evidence_ids: str) -> ReportClaim:
    return ReportClaim(task_id=task_id, text=text, evidence_ids=evidence_ids)


def _draft(
    *sections: ReportSection,
    title: str = "Apple research",
) -> ReportDraft:
    return ReportDraft(title=title, sections=sections)


def _section(heading: str, *claims: ReportClaim) -> ReportSection:
    return ReportSection(heading=heading, claims=claims)


def _generation(
    entries: tuple[tuple[ReportTask, AgentResult], ...],
    draft: ReportDraft,
) -> ReportGenerationResult:
    plan = ReportPlan(title=draft.title, tasks=tuple(task for task, _ in entries))
    research = ReportResearchResult(
        plan=plan,
        evidence=tuple(
            ReportEvidence(task=plan.tasks[index], result=result)
            for index, (_, result) in enumerate(entries)
        ),
    )
    return ReportGenerationResult(
        research=research,
        synthesis_evidence=build_synthesis_evidence(research),
        draft=draft,
    )


def _basic_generation() -> ReportGenerationResult:
    return _generation(
        ((_rag_task(), _rag_result(_source())),),
        _draft(_section("Risks", _claim("risks", "Apple described risks.", "E1"))),
    )


def test_exact_basic_markdown_and_one_final_newline() -> None:
    artifact = ReportMarkdownRenderer().render(_basic_generation())

    assert artifact == ReportMarkdownArtifact(
        media_type="text/markdown",
        content=(
            "# Apple research\n\n"
            "## Risks\n\n"
            "Apple described risks\\. [E1]\n\n"
            "## Evidence\n\n"
            "### E1 — RAG source\n"
            "- Task: risks\n"
            "- Document: apple\\-2025\\-10k\\.pdf\n"
            "- Page: 23\n"
            "- Chunk: 4\n"
            "- Company: Apple\n"
            "- Ticker: AAPL\n"
            "- Fiscal year: 2025\n"
            "- Document type: 10K\n"
        ),
    )
    assert artifact.content.endswith("\n")
    assert not artifact.content.endswith("\n\n")


def test_section_claim_and_citation_orders_are_preserved() -> None:
    generation = _generation(
        ((_rag_task(), _rag_result(_source("First", chunk_index=1), _source("Second", chunk_index=2))),),
        _draft(
            _section(
                "First section",
                _claim("risks", "First claim", "E2", "E1"),
                _claim("risks", "Second claim", "E1"),
            ),
            _section("Second section", _claim("risks", "Third claim", "E2")),
        ),
    )

    content = ReportMarkdownRenderer().render(generation).content

    assert content.index("## First section") < content.index("## Second section")
    assert content.index("First claim [E2] [E1]") < content.index("Second claim [E1]")
    assert content.index("Second claim [E1]") < content.index("Third claim [E2]")
    assert content.index("### E1 — RAG source") < content.index("### E2 — RAG source")


def test_appendix_contains_only_cited_items_in_registry_order() -> None:
    generation = _generation(
        ((_rag_task(), _rag_result(_source("One"), _source("Two"), _source("Three"))),),
        _draft(_section("Risks", _claim("risks", "Finding", "E3", "E1"))),
    )

    content = ReportMarkdownRenderer().render(generation).content
    appendix = content.split("## Evidence\n\n", 1)[1]

    assert "Finding [E3] [E1]" in content
    assert appendix.index("### E1") < appendix.index("### E3")
    assert "### E2" not in appendix


def test_rag_metadata_omits_nulls_and_full_source_text() -> None:
    source = _source(
        "DISTINCTIVE_FULL_SOURCE_TEXT",
        company=None,
        ticker=None,
        fiscal_year=None,
        document_type=None,
    )
    generation = _generation(
        ((_rag_task(), _rag_result(source)),),
        _draft(_section("Risks", _claim("risks", "Finding", "E1"))),
    )

    content = ReportMarkdownRenderer().render(generation).content

    assert "- Document: apple\\-2025\\-10k\\.pdf" in content
    assert "- Page: 23" in content
    assert "- Chunk: 4" in content
    assert "- Company:" not in content
    assert "- Ticker:" not in content
    assert "- Fiscal year:" not in content
    assert "- Document type:" not in content
    assert "DISTINCTIVE_FULL_SOURCE_TEXT" not in content
    assert "text_characters" not in content


def test_sql_summary_keeps_column_order_without_rows_or_query() -> None:
    result = AgentResult(
        route="sql",
        generated_sql="DISTINCTIVE_GENERATED_SQL",
        sql_result=_sql_result(
            columns=["revenue_musd", "company"],
            rows=[
                {"revenue_musd": 416161, "company": "SECRET_ROW_MARKER"},
                {"revenue_musd": 281724, "company": "Microsoft"},
            ],
        ),
    )
    generation = _generation(
        ((_sql_task(), result),),
        _draft(_section("Revenue", _claim("revenue", "Revenue was reported", "E1"))),
    )

    content = ReportMarkdownRenderer().render(generation).content

    assert "### E1 — SQL result" in content
    assert "- Rows: 2" in content
    assert "- Columns: revenue\\_musd, company" in content
    assert "SECRET_ROW_MARKER" not in content
    assert "DISTINCTIVE_GENERATED_SQL" not in content
    assert "416161" not in content


def test_sql_empty_columns_have_explicit_summary() -> None:
    generation = _generation(
        ((_sql_task(), AgentResult(route="sql", generated_sql="SELECT 1", sql_result=_sql_result(columns=[], rows=[]))),),
        _draft(_section("Revenue", _claim("revenue", "No rows", "E1"))),
    )

    content = ReportMarkdownRenderer().render(generation).content

    assert "- Rows: 0" in content
    assert "- Columns: (none)" in content


def test_scalar_analysis_preserves_exact_value() -> None:
    result = AgentResult(
        route="sql",
        generated_sql="SELECT revenue_musd",
        sql_result=_sql_result(),
        analysis_result=AnalysisResult(
            AnalysisOperation.PERCENTAGE_CHANGE, value=6.425511782832739
        ),
    )
    generation = _generation(
        ((_sql_task(), result),),
        _draft(_section("Change", _claim("revenue", "Revenue increased", "E1", "E2"))),
    )

    content = ReportMarkdownRenderer().render(generation).content

    assert "### E2 — Analysis result" in content
    assert "- Operation: percentage\\_change" in content
    assert "- Value: 6.425511782832739\n" in content
    assert "6.43" not in content


def test_ranking_analysis_shows_count_without_dumping_rows() -> None:
    result = AgentResult(
        route="sql",
        generated_sql="SELECT ranking",
        sql_result=_sql_result(),
        analysis_result=AnalysisResult(
            AnalysisOperation.RANKING,
            ranked_rows=(
                {"company": "RANKED_ROW_SECRET_A", "revenue_musd": 416161},
                {"company": "RANKED_ROW_SECRET_B", "revenue_musd": 281724},
            ),
        ),
    )
    generation = _generation(
        ((_sql_task(), result),),
        _draft(_section("Ranking", _claim("revenue", "Apple ranked first", "E1", "E2"))),
    )

    content = ReportMarkdownRenderer().render(generation).content

    assert "- Operation: ranking" in content
    assert "- Ranked rows: 2" in content
    assert "- Value:" not in content
    assert "RANKED_ROW_SECRET_A" not in content
    assert "RANKED_ROW_SECRET_B" not in content


def test_dynamic_markdown_and_html_are_escaped_without_deleting_text() -> None:
    malicious = (
        "# forged heading\n"
        "<script>alert(1)</script>\n"
        "![image](https://example.com/a.png)\n"
        "[link](https://example.com)\n"
        "`code` *emphasis* 1. fake list ---"
    )
    title = f"Report {malicious}"
    source = _source("Source text", document=f"document\u2028{malicious}")
    generation = _generation(
        ((_rag_task(), _rag_result(source)),),
        _draft(
            _section(
                "Heading # forged heading\n<script>alert(1)</script>",
                _claim("risks", f"Claim {malicious}", "E1"),
            ),
            title=title,
        ),
    )

    content = ReportMarkdownRenderer().render(generation).content
    lines = content.splitlines()

    assert [line for line in lines if line.startswith("# ")] == [lines[0]]
    assert len([line for line in lines if line.startswith("## ")]) == 2
    assert len([line for line in lines if line.startswith("### ")]) == 1
    assert "# forged heading" not in lines
    assert "<script>" not in content
    assert "![image]" not in content
    assert "[link](" not in content
    assert "`code`" not in content
    assert "*emphasis*" not in content
    assert "1. fake list" not in content
    assert "\\# forged heading" in content
    assert "\\<script\\>alert\\(1\\)\\<\\/script\\>" in content
    assert "\\!\\[image\\]" in content
    assert "\\[link\\]" in content
    assert "\\`code\\`" in content
    assert "\\*emphasis\\*" in content
    assert "1\\. fake list" in content
    assert "document \\# forged heading" in content


def test_forged_citation_is_rejected() -> None:
    generation = _generation(
        ((_rag_task(), _rag_result(_source())),),
        _draft(_section("Risks", _claim("risks", "Finding", "E999"))),
    )

    with pytest.raises(ReportMarkdownRenderingError, match="rendering failed") as caught:
        ReportMarkdownRenderer().render(generation)

    assert isinstance(caught.value.__cause__, ValueError)


def test_cross_task_citation_is_rejected() -> None:
    generation = _generation(
        (
            (_rag_task(), _rag_result(_source())),
            (_sql_task(), AgentResult(route="sql", generated_sql="SELECT 1", sql_result=_sql_result())),
        ),
        _draft(_section("Risks", _claim("risks", "Finding", "E2"))),
    )

    with pytest.raises(ReportMarkdownRenderingError) as caught:
        ReportMarkdownRenderer().render(generation)

    assert "another task" in str(caught.value.__cause__)


@pytest.mark.parametrize(
    "kind,bad_content",
    [
        ("rag_source", 'document: "broken"\ntext:\nsource'),
        ("sql_result", "{not JSON"),
        ("analysis_result", "{not JSON"),
    ],
)
def test_malformed_cited_evidence_fails_closed(kind: str, bad_content: str) -> None:
    if kind == "rag_source":
        generation = _basic_generation()
        cited_id = "E1"
    else:
        result = AgentResult(
            route="sql",
            generated_sql="SELECT 1",
            sql_result=_sql_result(),
            analysis_result=(
                AnalysisResult(AnalysisOperation.PERCENTAGE_CHANGE, value=6.5)
                if kind == "analysis_result" else None
            ),
        )
        cited_id = "E2" if kind == "analysis_result" else "E1"
        generation = _generation(
            ((_sql_task(), result),),
            _draft(_section("Numbers", _claim("revenue", "Finding", cited_id))),
        )
    items = tuple(
        replace(item, content=bad_content) if item.evidence_id == cited_id else item
        for item in generation.synthesis_evidence.items
    )
    forged = replace(
        generation,
        synthesis_evidence=replace(generation.synthesis_evidence, items=items),
    )

    with pytest.raises(ReportMarkdownRenderingError) as caught:
        ReportMarkdownRenderer().render(forged)

    assert caught.value.__cause__ is not None


def test_uncited_malformed_evidence_is_not_parsed() -> None:
    generation = _generation(
        ((_rag_task(), _rag_result(_source("Cited"), _source("Uncited"))),),
        _draft(_section("Risks", _claim("risks", "Finding", "E1"))),
    )
    first, second = generation.synthesis_evidence.items
    forged = replace(
        generation,
        synthesis_evidence=replace(
            generation.synthesis_evidence,
            items=(first, replace(second, content="malformed uncited evidence")),
        ),
    )

    content = ReportMarkdownRenderer().render(forged).content

    assert "### E1" in content
    assert "### E2" not in content


def test_internal_research_artifacts_do_not_leak() -> None:
    chart_bytes = b"DISTINCTIVE_CHART_BYTES_739"
    chart_title = "DISTINCTIVE_CHART_TITLE_739"
    research_entries = (
        (
            _rag_task(),
            _rag_result(_source("DISTINCTIVE_SOURCE_TEXT_739"), answer="DISTINCTIVE_RAG_ANSWER_739"),
        ),
        (
            _sql_task(),
            AgentResult(
                route="sql",
                generated_sql="DISTINCTIVE_GENERATED_SQL_739",
                sql_result=_sql_result(rows=[{"company": "DISTINCTIVE_SQL_ROW_739", "revenue_musd": 416161}]),
                chart_spec=ChartSpec(ChartType.BAR, "company", "revenue_musd", chart_title),
                chart_artifact=ChartArtifact(media_type="image/png", content=chart_bytes),
            ),
        ),
    )
    generation = _generation(
        research_entries,
        _draft(_section(
            "Findings",
            _claim("risks", "Risk finding", "E1"),
            _claim("revenue", "Revenue finding", "E2"),
        )),
    )

    content = ReportMarkdownRenderer().render(generation).content

    for marker in (
        "DISTINCTIVE_RAG_ANSWER_739",
        "DISTINCTIVE_SOURCE_TEXT_739",
        "DISTINCTIVE_GENERATED_SQL_739",
        "DISTINCTIVE_SQL_ROW_739",
        chart_title,
        chart_bytes.decode(),
        base64.b64encode(chart_bytes).decode(),
        "ChartSpec(",
        "ChartArtifact(",
        "AgentResult(",
    ):
        assert marker not in content


def test_renderer_output_is_byte_for_byte_deterministic() -> None:
    generation = _basic_generation()
    renderer = ReportMarkdownRenderer()

    first = renderer.render(generation)
    second = renderer.render(generation)

    assert first == second
    assert first.content.encode("utf-8") == second.content.encode("utf-8")


def test_artifact_is_frozen() -> None:
    artifact = ReportMarkdownRenderer().render(_basic_generation())

    with pytest.raises(FrozenInstanceError):
        artifact.content = "changed"  # type: ignore[misc]


def test_structural_preflight_rejects_forged_generation() -> None:
    generation = _basic_generation()
    renderer = ReportMarkdownRenderer()
    with pytest.raises(ReportMarkdownRenderingError):
        renderer.render(None)  # type: ignore[arg-type]

    variants = (
        replace(generation, synthesis_evidence=replace(generation.synthesis_evidence, items=[])),
        replace(generation, synthesis_evidence=replace(
            generation.synthesis_evidence,
            items=(replace(generation.synthesis_evidence.items[0], evidence_id="E2"),),
        )),
        replace(generation, research=ReportResearchResult(
            plan=ReportPlan(title="Different title", tasks=generation.research.plan.tasks),
            evidence=generation.research.evidence,
        )),
        replace(generation, draft=ReportDraft.model_construct(title="Wrong title", sections=generation.draft.sections)),
        replace(generation, draft=ReportDraft.model_construct(title="Apple research", sections=())),
    )
    for variant in variants:
        with pytest.raises(ReportMarkdownRenderingError) as caught:
            renderer.render(variant)
        assert caught.value.__cause__ is not None


def test_semantically_equal_distinct_plans_are_accepted() -> None:
    generation = _basic_generation()
    distinct_equal_plan = ReportPlan.model_validate(generation.research.plan.model_dump())
    assert distinct_equal_plan is not generation.research.plan
    assert distinct_equal_plan == generation.research.plan
    variant = replace(
        generation,
        synthesis_evidence=replace(generation.synthesis_evidence, plan=distinct_equal_plan),
    )

    assert ReportMarkdownRenderer().render(variant).content == ReportMarkdownRenderer().render(
        generation
    ).content
