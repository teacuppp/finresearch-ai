"""Compose report research and synthesis without changing either layer."""

from dataclasses import dataclass
from typing import Protocol

from app.report.evidence import ReportResearchResult
from app.report.synthesis_evidence import (
    ReportSynthesisEvidence,
    build_synthesis_evidence,
)
from app.report.synthesis_models import ReportDraft


class ReportResearchServicePort(Protocol):
    def research(self, request: str) -> ReportResearchResult: ...


class ReportSynthesizerPort(Protocol):
    def synthesize(self, evidence: ReportSynthesisEvidence) -> ReportDraft: ...


@dataclass(frozen=True)
class ReportGenerationResult:
    research: ReportResearchResult
    synthesis_evidence: ReportSynthesisEvidence
    draft: ReportDraft


class ReportGenerationService:
    def __init__(
        self,
        research_service: ReportResearchServicePort,
        synthesizer: ReportSynthesizerPort,
    ) -> None:
        self.research_service = research_service
        self.synthesizer = synthesizer

    def generate(self, request: str) -> ReportGenerationResult:
        research = self.research_service.research(request)
        synthesis_evidence = build_synthesis_evidence(research)
        draft = self.synthesizer.synthesize(synthesis_evidence)
        return ReportGenerationResult(
            research=research,
            synthesis_evidence=synthesis_evidence,
            draft=draft,
        )
