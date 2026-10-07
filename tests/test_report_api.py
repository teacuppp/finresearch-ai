"""JSON report endpoint and its metadata-only attachment boundary."""

import base64

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.reports import router
from app.dependencies import get_report_bundle_renderer, get_report_generation_service
from app.main import app as main_app
from app.report.bundle import ReportRenderedBundle
from app.report.chart_attachments import (
    ReportAttachment,
    ReportAttachmentManifest,
    ReportChartAttachmentError,
)
from app.report.executor import ReportResearchExecutionError
from app.report.markdown_renderer import (
    ReportMarkdownArtifact,
    ReportMarkdownRenderingError,
)
from app.report.planner import ReportPlanningError
from app.report.synthesis_evidence import ReportSynthesisEvidenceError
from app.report.synthesizer import ReportSynthesisError


class FakeGenerationService:
    def __init__(self, generation: object, failure: Exception | None = None) -> None:
        self.generation = generation
        self.failure = failure
        self.calls: list[str] = []

    def generate(self, request: str) -> object:
        self.calls.append(request)
        if self.failure is not None:
            raise self.failure
        return self.generation


class FakeBundleRenderer:
    def __init__(
        self, bundle: ReportRenderedBundle, failure: Exception | None = None
    ) -> None:
        self.bundle = bundle
        self.failure = failure
        self.calls: list[object] = []

    def render(self, generation: object) -> ReportRenderedBundle:
        self.calls.append(generation)
        if self.failure is not None:
            raise self.failure
        return self.bundle


def _attachment(task_id: str, content: bytes) -> ReportAttachment:
    attachment_id = f"chart-{task_id}"
    return ReportAttachment(
        attachment_id=attachment_id,
        task_id=task_id,
        media_type="image/png",
        relative_path=f"attachments/{attachment_id}.png",
        title=f"Chart for {task_id}",
        content=content,
    )


def _test_client(
    *,
    markdown: str = "# Report\n",
    attachments: tuple[ReportAttachment, ...] = (),
    generation_failure: Exception | None = None,
    rendering_failure: Exception | None = None,
) -> tuple[TestClient, FakeGenerationService, FakeBundleRenderer, object]:
    generation = object()
    bundle = ReportRenderedBundle(
        markdown=ReportMarkdownArtifact("text/markdown", markdown),
        attachments=ReportAttachmentManifest(attachments),
    )
    service = FakeGenerationService(generation, generation_failure)
    renderer = FakeBundleRenderer(bundle, rendering_failure)
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_report_generation_service] = lambda: service
    app.dependency_overrides[get_report_bundle_renderer] = lambda: renderer
    return TestClient(app), service, renderer, generation


def test_success_without_attachments_preserves_generation_identity() -> None:
    client, service, renderer, generation = _test_client()
    request = "  Analyze Apple revenue.  "

    response = client.post("/reports/generate", json={"request": request})

    assert response.status_code == 200
    assert response.json() == {
        "markdown": {"media_type": "text/markdown", "content": "# Report\n"},
        "attachments": [],
    }
    assert service.calls == [request]
    assert len(renderer.calls) == 1
    assert renderer.calls[0] is generation


def test_multiple_attachments_map_metadata_in_manifest_order() -> None:
    first = _attachment("z_revenue", b"FIRST_PNG")
    second = _attachment("a_growth", b"SECOND_PNG")
    client, service, renderer, generation = _test_client(
        attachments=(first, second)
    )

    response = client.post("/reports/generate", json={"request": "Make a report"})

    assert response.status_code == 200
    assert response.json()["attachments"] == [
        {
            "attachment_id": "chart-z_revenue",
            "task_id": "z_revenue",
            "media_type": "image/png",
            "relative_path": "attachments/chart-z_revenue.png",
            "title": "Chart for z_revenue",
        },
        {
            "attachment_id": "chart-a_growth",
            "task_id": "a_growth",
            "media_type": "image/png",
            "relative_path": "attachments/chart-a_growth.png",
            "title": "Chart for a_growth",
        },
    ]
    assert service.calls == ["Make a report"]
    assert renderer.calls == [generation]
    assert renderer.calls[0] is generation


def test_attachment_bytes_are_absent_from_serialized_json() -> None:
    secret = b"SUPER_SECRET_DISTINCTIVE_PNG_BYTES"
    client, _, _, _ = _test_client(attachments=(_attachment("revenue", secret),))

    response = client.post("/reports/generate", json={"request": "Report"})
    attachment_json = response.json()["attachments"][0]

    assert response.status_code == 200
    assert "content" not in attachment_json
    assert set(attachment_json) == {
        "attachment_id", "task_id", "media_type", "relative_path", "title"
    }
    assert secret not in response.content
    assert base64.b64encode(secret) not in response.content
    assert b"data:image" not in response.content
    assert b"/tmp/" not in response.content


def test_markdown_content_is_returned_exactly() -> None:
    markdown = "# Report\n\n## Evidence\n\nSpecial *emphasis* [E1] <text>\n"
    client, _, _, _ = _test_client(
        markdown=markdown,
        attachments=(_attachment("revenue", b"PNG"),),
    )

    response = client.post("/reports/generate", json={"request": "Report"})

    assert response.status_code == 200
    assert response.json()["markdown"]["content"] == markdown
    assert "attachments/" not in response.json()["markdown"]["content"]


@pytest.mark.parametrize(
    "body",
    [{}, {"request": 42}, {"request": ""}, {"request": "   "}, {"request": "x" * 2001}],
)
def test_invalid_request_uses_normal_422_validation(body: dict) -> None:
    client, service, renderer, _ = _test_client()

    response = client.post("/reports/generate", json=body)

    assert response.status_code == 422
    assert service.calls == []
    assert renderer.calls == []


@pytest.mark.parametrize(
    "failure",
    [
        ReportPlanningError("private planning detail"),
        ReportResearchExecutionError("private research detail"),
        ReportSynthesisError("private synthesis detail"),
    ],
)
def test_generation_errors_map_to_stable_502(failure: Exception) -> None:
    client, service, renderer, _ = _test_client(generation_failure=failure)

    response = client.post("/reports/generate", json={"request": "Report"})

    assert response.status_code == 502
    assert response.json() == {"detail": "The report could not be generated."}
    assert "private" not in response.text
    assert service.calls == ["Report"]
    assert renderer.calls == []


def test_synthesis_evidence_error_maps_to_stable_500() -> None:
    failure = ReportSynthesisEvidenceError("private synthesis evidence detail")
    client, service, renderer, _ = _test_client(generation_failure=failure)

    response = client.post("/reports/generate", json={"request": "Report"})

    assert response.status_code == 500
    assert response.json() == {"detail": "The report could not be generated."}
    assert "private synthesis evidence detail" not in response.text
    assert service.calls == ["Report"]
    assert renderer.calls == []


@pytest.mark.parametrize(
    "failure",
    [
        ReportMarkdownRenderingError("private Markdown detail"),
        ReportChartAttachmentError("private chart detail"),
    ],
)
def test_rendering_errors_map_to_stable_500(failure: Exception) -> None:
    client, service, renderer, generation = _test_client(
        rendering_failure=failure
    )

    response = client.post("/reports/generate", json={"request": "Report"})

    assert response.status_code == 500
    assert response.json() == {
        "detail": "The generated report could not be rendered."
    }
    assert "private" not in response.text
    assert service.calls == ["Report"]
    assert renderer.calls[0] is generation
    assert len(renderer.calls) == 1


def test_unexpected_error_is_not_translated() -> None:
    failure = RuntimeError("programming error")
    client, _, _, _ = _test_client(generation_failure=failure)

    with pytest.raises(RuntimeError) as caught:
        client.post("/reports/generate", json={"request": "Report"})

    assert caught.value is failure


def test_main_app_registers_report_route_and_keeps_existing_routes() -> None:
    paths = main_app.openapi()["paths"]

    assert "post" in paths["/reports/generate"]
    assert "/rag/ask" in paths
    assert "/agent/ask" in paths
    assert "/health" in paths
