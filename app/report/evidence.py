"""Typed evidence collected from report research tasks."""

from dataclasses import dataclass

from app.report.models import ReportPlan, ReportTask
from app.services.agent_service import AgentResult


@dataclass(frozen=True)
class ReportEvidence:
    task: ReportTask
    result: AgentResult


@dataclass(frozen=True)
class ReportResearchResult:
    plan: ReportPlan
    evidence: tuple[ReportEvidence, ...]
