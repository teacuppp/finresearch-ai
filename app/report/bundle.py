"""Compose existing report presentation artifacts without changing them."""

from dataclasses import dataclass
from typing import Protocol

from app.report.chart_attachments import ReportAttachmentManifest
from app.report.generation import ReportGenerationResult
from app.report.markdown_renderer import ReportMarkdownArtifact


@dataclass(frozen=True)
class ReportRenderedBundle:
    markdown: ReportMarkdownArtifact
    attachments: ReportAttachmentManifest


class ReportMarkdownRendererPort(Protocol):
    def render(
        self,
        generation: ReportGenerationResult,
    ) -> ReportMarkdownArtifact: ...


class ReportChartAttachmentBuilderPort(Protocol):
    def build(
        self,
        generation: ReportGenerationResult,
    ) -> ReportAttachmentManifest: ...


class ReportBundleRenderer:
    def __init__(
        self,
        markdown_renderer: ReportMarkdownRendererPort,
        attachment_builder: ReportChartAttachmentBuilderPort,
    ) -> None:
        self.markdown_renderer = markdown_renderer
        self.attachment_builder = attachment_builder

    def render(self, generation: ReportGenerationResult) -> ReportRenderedBundle:
        markdown = self.markdown_renderer.render(generation)
        attachments = self.attachment_builder.build(generation)
        return ReportRenderedBundle(markdown=markdown, attachments=attachments)
