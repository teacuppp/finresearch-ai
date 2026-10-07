"""Deterministic attachment manifest for charts already produced by research."""

from dataclasses import dataclass
from typing import Literal

from app.analysis.chart_renderer import ChartArtifact, ChartSpec
from app.report.evidence import ReportEvidence, ReportResearchResult
from app.report.generation import ReportGenerationResult
from app.report.models import ReportPlan, ReportTask
from app.report.synthesis_evidence import ReportSynthesisEvidence
from app.report.synthesis_models import ReportDraft
from app.services.agent_service import AgentResult


class ReportChartAttachmentError(ValueError):
    """The report cannot be converted to a chart attachment manifest."""


@dataclass(frozen=True)
class ReportAttachment:
    attachment_id: str
    task_id: str
    media_type: Literal["image/png"]
    relative_path: str
    title: str
    content: bytes


@dataclass(frozen=True)
class ReportAttachmentManifest:
    attachments: tuple[ReportAttachment, ...]


def _validated_research(generation: ReportGenerationResult) -> ReportResearchResult:
    research = generation.research
    if not isinstance(research, ReportResearchResult):
        raise ValueError("research must be a ReportResearchResult")
    plan = research.plan
    if not isinstance(plan, ReportPlan):
        raise ValueError("research plan must be a ReportPlan")
    ReportPlan.model_validate(plan, strict=True)
    if not isinstance(research.evidence, tuple):
        raise ValueError("research evidence must be a tuple")
    if len(research.evidence) != len(plan.tasks):
        raise ValueError("research evidence count differs from plan tasks")

    for task, entry in zip(plan.tasks, research.evidence, strict=True):
        if not isinstance(entry, ReportEvidence):
            raise ValueError("research contains an invalid evidence entry")
        if not isinstance(entry.task, ReportTask) or entry.task != task:
            raise ValueError(f"research evidence task differs from plan task {task.task_id!r}")
        if not isinstance(entry.result, AgentResult):
            raise ValueError(f"task {task.task_id!r} has an invalid agent result")
        if entry.result.route != task.expected_route:
            raise ValueError(f"task {task.task_id!r} has the wrong result route")

    synthesis = generation.synthesis_evidence
    if not isinstance(synthesis, ReportSynthesisEvidence):
        raise ValueError("synthesis evidence must be a ReportSynthesisEvidence")
    if not isinstance(synthesis.plan, ReportPlan) or synthesis.plan != plan:
        raise ValueError("synthesis evidence plan differs from research plan")
    if not isinstance(generation.draft, ReportDraft):
        raise ValueError("draft must be a ReportDraft")
    if generation.draft.title != plan.title:
        raise ValueError("draft title differs from research plan")
    return research


def _attachment(entry: ReportEvidence) -> ReportAttachment | None:
    task_id = entry.task.task_id
    spec = entry.result.chart_spec
    artifact = entry.result.chart_artifact
    if spec is None and artifact is None:
        return None
    if entry.task.expected_route != "sql":
        raise ValueError(f"task {task_id!r} has a chart on a non-SQL route")
    if not isinstance(spec, ChartSpec) or not isinstance(artifact, ChartArtifact):
        raise ValueError(f"task {task_id!r} has an invalid chart pair")
    if type(spec.title) is not str or not spec.title.strip():
        raise ValueError(f"task {task_id!r} has an invalid chart title")
    if type(artifact.media_type) is not str or artifact.media_type != "image/png":
        raise ValueError(f"task {task_id!r} has an invalid chart media type")
    if type(artifact.content) is not bytes or not artifact.content:
        raise ValueError(f"task {task_id!r} has invalid chart bytes")

    attachment_id = f"chart-{task_id}"
    return ReportAttachment(
        attachment_id=attachment_id,
        task_id=task_id,
        media_type="image/png",
        relative_path=f"attachments/{attachment_id}.png",
        title=spec.title,
        content=artifact.content,
    )


class ReportChartAttachmentBuilder:
    def build(self, generation: ReportGenerationResult) -> ReportAttachmentManifest:
        if not isinstance(generation, ReportGenerationResult):
            raise ReportChartAttachmentError("A ReportGenerationResult is required.")
        try:
            research = _validated_research(generation)
            attachments = tuple(
                attachment
                for entry in research.evidence
                if (attachment := _attachment(entry)) is not None
            )
            return ReportAttachmentManifest(attachments=attachments)
        except Exception as exc:
            raise ReportChartAttachmentError(
                "Report chart attachment building failed."
            ) from exc
