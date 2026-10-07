"""JSON report generation endpoint with metadata-only chart attachments."""

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator

from app.dependencies import get_report_bundle_renderer, get_report_generation_service
from app.report.bundle import ReportBundleRenderer
from app.report.chart_attachments import ReportChartAttachmentError
from app.report.executor import ReportResearchExecutionError
from app.report.generation import ReportGenerationService
from app.report.markdown_renderer import ReportMarkdownRenderingError
from app.report.planner import ReportPlanningError
from app.report.synthesis_evidence import ReportSynthesisEvidenceError
from app.report.synthesizer import ReportSynthesisError


router = APIRouter(prefix="/reports", tags=["reports"])


class ReportGenerateRequest(BaseModel):
    request: str = Field(min_length=1, max_length=2000)

    @field_validator("request")
    @classmethod
    def request_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("request must not be empty")
        return value


class ReportMarkdownResponse(BaseModel):
    media_type: Literal["text/markdown"]
    content: str


class ReportAttachmentResponse(BaseModel):
    attachment_id: str
    task_id: str
    media_type: Literal["image/png"]
    relative_path: str
    title: str


class ReportGenerateResponse(BaseModel):
    markdown: ReportMarkdownResponse
    attachments: list[ReportAttachmentResponse]


@router.post("/generate", response_model=ReportGenerateResponse)
def generate_report(
    request: ReportGenerateRequest,
    report_generation_service: Annotated[
        ReportGenerationService, Depends(get_report_generation_service)
    ],
    report_bundle_renderer: Annotated[
        ReportBundleRenderer, Depends(get_report_bundle_renderer)
    ],
) -> ReportGenerateResponse:
    try:
        generation = report_generation_service.generate(request.request)
    except (
        ReportPlanningError,
        ReportResearchExecutionError,
        ReportSynthesisError,
    ) as exc:
        raise HTTPException(
            status_code=502,
            detail="The report could not be generated.",
        ) from exc
    except ReportSynthesisEvidenceError as exc:
        raise HTTPException(
            status_code=500,
            detail="The report could not be generated.",
        ) from exc

    try:
        bundle = report_bundle_renderer.render(generation)
    except (ReportMarkdownRenderingError, ReportChartAttachmentError) as exc:
        raise HTTPException(
            status_code=500,
            detail="The generated report could not be rendered.",
        ) from exc

    return ReportGenerateResponse(
        markdown=ReportMarkdownResponse(
            media_type=bundle.markdown.media_type,
            content=bundle.markdown.content,
        ),
        attachments=[
            ReportAttachmentResponse(
                attachment_id=attachment.attachment_id,
                task_id=attachment.task_id,
                media_type=attachment.media_type,
                relative_path=attachment.relative_path,
                title=attachment.title,
            )
            for attachment in bundle.attachments.attachments
        ],
    )
