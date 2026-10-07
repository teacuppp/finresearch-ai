"""Deterministic manifest of existing report chart artifacts."""

import base64
from dataclasses import FrozenInstanceError, replace

import pytest

from app.analysis.chart_renderer import ChartArtifact, ChartSpec, ChartType
from app.report.chart_attachments import (
    ReportAttachment,
    ReportAttachmentManifest,
    ReportChartAttachmentBuilder,
    ReportChartAttachmentError,
)
from app.report.evidence import ReportEvidence, ReportResearchResult
from app.report.generation import ReportGenerationResult
from app.report.models import ReportPlan, ReportRAGScope, ReportTask
from app.report.synthesis_evidence import ReportSynthesisEvidence
from app.report.synthesis_models import ReportClaim, ReportDraft, ReportSection
from app.services.agent_service import AgentResult


def _task(task_id: str, route: str = "sql") -> ReportTask:
    return ReportTask(
        task_id=task_id,
        question=f"Research {task_id}.",
        expected_route=route,
        rag_scope=ReportRAGScope(company="Apple") if route == "rag" else None,
    )


def _chart(
    *,
    title: object = "Revenue chart",
    content: object = b"DISTINCTIVE_CHART_BYTES",
    media_type: str = "image/png",
) -> tuple[ChartSpec, ChartArtifact]:
    return (
        ChartSpec(
            chart_type=ChartType.BAR,
            x_column="year",
            y_column="revenue",
            title=title,
        ),
        ChartArtifact(media_type=media_type, content=content),
    )


def _generation(*entries: tuple[ReportTask, AgentResult]) -> ReportGenerationResult:
    plan = ReportPlan(title="Apple report", tasks=tuple(task for task, _ in entries))
    research = ReportResearchResult(
        plan=plan,
        evidence=tuple(
            ReportEvidence(task=task, result=result) for task, result in entries
        ),
    )
    draft = ReportDraft(
        title=plan.title,
        sections=(
            ReportSection(
                heading="Results",
                claims=(
                    ReportClaim(
                        task_id=plan.tasks[0].task_id,
                        text="Finding",
                        evidence_ids=("E1",),
                    ),
                ),
            ),
        ),
    )
    return ReportGenerationResult(
        research=research,
        synthesis_evidence=ReportSynthesisEvidence(plan=plan, items=()),
        draft=draft,
    )


def _with_result(
    generation: ReportGenerationResult,
    result: AgentResult,
    position: int = 0,
) -> ReportGenerationResult:
    entries = list(generation.research.evidence)
    entries[position] = replace(entries[position], result=result)
    return replace(
        generation,
        research=replace(generation.research, evidence=tuple(entries)),
    )


def _assert_wrapped_failure(generation: ReportGenerationResult) -> None:
    with pytest.raises(
        ReportChartAttachmentError, match="Report chart attachment building failed"
    ) as caught:
        ReportChartAttachmentBuilder().build(generation)
    assert caught.value.__cause__ is not None


def test_mixed_report_without_charts_has_empty_manifest() -> None:
    generation = _generation(
        (_task("risks", "rag"), AgentResult(route="rag")),
        (_task("revenue"), AgentResult(route="sql")),
    )

    assert ReportChartAttachmentBuilder().build(generation) == ReportAttachmentManifest(
        attachments=()
    )


def test_one_sql_chart_uses_exact_metadata_and_original_bytes() -> None:
    spec, artifact = _chart(title="Exact chart title")
    generation = _generation(
        (_task("revenue_ranking"), AgentResult(route="sql", chart_spec=spec, chart_artifact=artifact)),
    )

    manifest = ReportChartAttachmentBuilder().build(generation)

    assert manifest == ReportAttachmentManifest(
        attachments=(
            ReportAttachment(
                attachment_id="chart-revenue_ranking",
                task_id="revenue_ranking",
                media_type="image/png",
                relative_path="attachments/chart-revenue_ranking.png",
                title="Exact chart title",
                content=artifact.content,
            ),
        ),
    )
    assert manifest.attachments[0].content is artifact.content
    assert manifest.attachments[0].title is spec.title


def test_multiple_charts_keep_research_task_order() -> None:
    first_spec, first_artifact = _chart(title="First")
    second_spec, second_artifact = _chart(title="Second")
    generation = _generation(
        (_task("z_first"), AgentResult(route="sql", chart_spec=first_spec, chart_artifact=first_artifact)),
        (_task("no_chart"), AgentResult(route="sql")),
        (_task("a_second"), AgentResult(route="sql", chart_spec=second_spec, chart_artifact=second_artifact)),
    )

    manifest = ReportChartAttachmentBuilder().build(generation)

    assert tuple(item.attachment_id for item in manifest.attachments) == (
        "chart-z_first",
        "chart-a_second",
    )


def test_manifest_is_deterministic_for_identical_generation() -> None:
    spec, artifact = _chart()
    generation = _generation(
        (_task("revenue"), AgentResult(route="sql", chart_spec=spec, chart_artifact=artifact)),
    )
    builder = ReportChartAttachmentBuilder()

    first = builder.build(generation)
    second = builder.build(generation)

    assert first == second
    assert first.attachments[0].relative_path == second.attachments[0].relative_path
    assert first.attachments[0].content is second.attachments[0].content


@pytest.mark.parametrize("missing", ["spec", "artifact"])
def test_chart_pair_mismatch_is_rejected(missing: str) -> None:
    spec, artifact = _chart()
    result = AgentResult(
        route="sql",
        chart_spec=None if missing == "spec" else spec,
        chart_artifact=None if missing == "artifact" else artifact,
    )
    _assert_wrapped_failure(_generation((_task("revenue"), result)))


@pytest.mark.parametrize("wrong", ["spec", "artifact"])
def test_wrong_chart_runtime_type_is_rejected(wrong: str) -> None:
    spec, artifact = _chart()
    result = AgentResult(
        route="sql",
        chart_spec=object() if wrong == "spec" else spec,
        chart_artifact=object() if wrong == "artifact" else artifact,
    )
    _assert_wrapped_failure(_generation((_task("revenue"), result)))


def test_rag_chart_is_rejected() -> None:
    spec, artifact = _chart()
    generation = _generation(
        (_task("risks", "rag"), AgentResult(route="rag", chart_spec=spec, chart_artifact=artifact)),
    )
    _assert_wrapped_failure(generation)


def test_invalid_media_type_is_rejected() -> None:
    spec, artifact = _chart(media_type="image/jpeg")
    _assert_wrapped_failure(
        _generation(
            (_task("revenue"), AgentResult(route="sql", chart_spec=spec, chart_artifact=artifact)),
        )
    )


@pytest.mark.parametrize("content", [b"", "not bytes"])
def test_invalid_chart_content_is_rejected(content: object) -> None:
    spec, artifact = _chart(content=content)
    _assert_wrapped_failure(
        _generation(
            (_task("revenue"), AgentResult(route="sql", chart_spec=spec, chart_artifact=artifact)),
        )
    )


@pytest.mark.parametrize("title", ["", "   ", 42])
def test_invalid_chart_title_is_rejected(title: object) -> None:
    spec, artifact = _chart(title=title)
    _assert_wrapped_failure(
        _generation(
            (_task("revenue"), AgentResult(route="sql", chart_spec=spec, chart_artifact=artifact)),
        )
    )


@pytest.mark.parametrize(
    "mismatch",
    ["count", "task", "route", "synthesis_plan", "draft_title"],
)
def test_generation_structural_mismatch_is_rejected(mismatch: str) -> None:
    generation = _generation((_task("revenue"), AgentResult(route="sql")))
    if mismatch == "count":
        generation = replace(
            generation,
            research=replace(generation.research, evidence=()),
        )
    elif mismatch == "task":
        other = _task("other")
        generation = replace(
            generation,
            research=replace(
                generation.research,
                evidence=(replace(generation.research.evidence[0], task=other),),
            ),
        )
    elif mismatch == "route":
        generation = _with_result(generation, AgentResult(route="rag"))
    elif mismatch == "synthesis_plan":
        other_plan = ReportPlan(title="Other report", tasks=generation.research.plan.tasks)
        generation = replace(
            generation,
            synthesis_evidence=replace(generation.synthesis_evidence, plan=other_plan),
        )
    else:
        generation = replace(
            generation,
            draft=generation.draft.model_copy(update={"title": "Other report"}),
        )
    _assert_wrapped_failure(generation)


def test_semantically_equal_distinct_plans_are_accepted() -> None:
    generation = _generation((_task("revenue"), AgentResult(route="sql")))
    distinct_equal_plan = ReportPlan.model_validate(
        generation.research.plan.model_dump()
    )
    assert distinct_equal_plan is not generation.research.plan
    generation = replace(
        generation,
        synthesis_evidence=replace(
            generation.synthesis_evidence,
            plan=distinct_equal_plan,
        ),
    )

    assert ReportChartAttachmentBuilder().build(generation).attachments == ()


def test_wrong_top_level_input_has_specific_error() -> None:
    with pytest.raises(ReportChartAttachmentError, match="ReportGenerationResult"):
        ReportChartAttachmentBuilder().build(object())


def test_attachment_contracts_are_frozen() -> None:
    spec, artifact = _chart()
    generation = _generation(
        (_task("revenue"), AgentResult(route="sql", chart_spec=spec, chart_artifact=artifact)),
    )
    manifest = ReportChartAttachmentBuilder().build(generation)

    with pytest.raises(FrozenInstanceError):
        manifest.attachments[0].title = "Changed"
    with pytest.raises(FrozenInstanceError):
        manifest.attachments = ()


def test_chart_content_and_metadata_are_not_transformed() -> None:
    original = b"UNIQUE_PNG_BYTE_MARKER_93"
    spec, artifact = _chart(title="<unsafe> **title**", content=original)
    generation = _generation(
        (_task("revenue"), AgentResult(route="sql", chart_spec=spec, chart_artifact=artifact)),
    )

    attachment = ReportChartAttachmentBuilder().build(generation).attachments[0]
    metadata = (
        attachment.attachment_id,
        attachment.task_id,
        attachment.media_type,
        attachment.relative_path,
        attachment.title,
    )

    assert attachment.content is original
    assert attachment.title == "<unsafe> **title**"
    assert attachment.relative_path == "attachments/chart-revenue.png"
    assert all(not value.startswith("/") for value in metadata)
    assert base64.b64encode(original).decode("ascii") not in repr(metadata)
