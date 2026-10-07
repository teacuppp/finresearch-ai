"""Thin orchestration of existing report presentation artifacts."""

from dataclasses import FrozenInstanceError

import pytest

from app.report.bundle import ReportBundleRenderer, ReportRenderedBundle
from app.report.chart_attachments import (
    ReportAttachment,
    ReportAttachmentManifest,
    ReportChartAttachmentError,
)
from app.report.markdown_renderer import (
    ReportMarkdownArtifact,
    ReportMarkdownRenderingError,
)


class FakeMarkdownRenderer:
    def __init__(
        self,
        artifact: ReportMarkdownArtifact,
        events: list[str],
        failure: Exception | None = None,
    ) -> None:
        self.artifact = artifact
        self.events = events
        self.failure = failure
        self.calls: list[object] = []

    def render(self, generation: object) -> ReportMarkdownArtifact:
        self.calls.append(generation)
        self.events.append("markdown")
        if self.failure is not None:
            raise self.failure
        return self.artifact


class FakeAttachmentBuilder:
    def __init__(
        self,
        manifest: ReportAttachmentManifest,
        events: list[str],
        failure: Exception | None = None,
    ) -> None:
        self.manifest = manifest
        self.events = events
        self.failure = failure
        self.calls: list[object] = []

    def build(self, generation: object) -> ReportAttachmentManifest:
        self.calls.append(generation)
        self.events.append("attachments")
        if self.failure is not None:
            raise self.failure
        return self.manifest


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


def _dependencies(
    markdown: ReportMarkdownArtifact | None = None,
    manifest: ReportAttachmentManifest | None = None,
    *,
    markdown_failure: Exception | None = None,
    attachment_failure: Exception | None = None,
) -> tuple[FakeMarkdownRenderer, FakeAttachmentBuilder, list[str]]:
    events: list[str] = []
    renderer = FakeMarkdownRenderer(
        markdown or ReportMarkdownArtifact("text/markdown", "# Report\n"),
        events,
        markdown_failure,
    )
    builder = FakeAttachmentBuilder(
        manifest or ReportAttachmentManifest(()), events, attachment_failure
    )
    return renderer, builder, events


def test_happy_path_calls_each_dependency_once_in_order_with_same_input() -> None:
    generation = object()
    markdown = ReportMarkdownArtifact("text/markdown", "# Report\n")
    manifest = ReportAttachmentManifest((_attachment("revenue", b"PNG"),))
    renderer, builder, events = _dependencies(markdown, manifest)

    bundle = ReportBundleRenderer(renderer, builder).render(generation)

    assert events == ["markdown", "attachments"]
    assert len(renderer.calls) == len(builder.calls) == 1
    assert renderer.calls[0] is generation
    assert builder.calls[0] is generation
    assert bundle == ReportRenderedBundle(markdown, manifest)
    assert bundle.markdown is markdown
    assert bundle.attachments is manifest


def test_empty_attachment_manifest_is_preserved() -> None:
    manifest = ReportAttachmentManifest(attachments=())
    renderer, builder, _ = _dependencies(manifest=manifest)

    bundle = ReportBundleRenderer(renderer, builder).render(object())

    assert bundle.attachments is manifest
    assert bundle.attachments.attachments == ()


def test_multiple_attachments_keep_tuple_order_and_identity() -> None:
    first = _attachment("z_first", b"FIRST")
    second = _attachment("a_second", b"SECOND")
    manifest = ReportAttachmentManifest((first, second))
    renderer, builder, _ = _dependencies(manifest=manifest)

    bundle = ReportBundleRenderer(renderer, builder).render(object())

    assert bundle.attachments is manifest
    assert bundle.attachments.attachments is manifest.attachments
    assert bundle.attachments.attachments[0] is first
    assert bundle.attachments.attachments[1] is second


def test_markdown_content_is_unchanged_without_chart_references() -> None:
    content = "# Report\n\nA claim [E1]\n\n## Evidence\n\n### E1 — SQL result\n"
    markdown = ReportMarkdownArtifact("text/markdown", content)
    renderer, builder, _ = _dependencies(
        markdown=markdown,
        manifest=ReportAttachmentManifest((_attachment("revenue", b"PNG"),)),
    )

    bundle = ReportBundleRenderer(renderer, builder).render(object())

    assert bundle.markdown is markdown
    assert bundle.markdown.content == content
    assert "attachments/" not in bundle.markdown.content
    assert "![" not in bundle.markdown.content


def test_attachment_bytes_remain_the_original_object() -> None:
    original_bytes = b"DISTINCTIVE_PNG_BYTES"
    attachment = _attachment("revenue", original_bytes)
    manifest = ReportAttachmentManifest((attachment,))
    renderer, builder, _ = _dependencies(manifest=manifest)

    bundle = ReportBundleRenderer(renderer, builder).render(object())

    assert bundle.attachments.attachments[0] is attachment
    assert bundle.attachments.attachments[0].content is original_bytes


def test_markdown_failure_propagates_same_error_without_building() -> None:
    failure = ReportMarkdownRenderingError("existing markdown error")
    renderer, builder, events = _dependencies(markdown_failure=failure)

    with pytest.raises(ReportMarkdownRenderingError) as caught:
        ReportBundleRenderer(renderer, builder).render(object())

    assert caught.value is failure
    assert len(renderer.calls) == 1
    assert builder.calls == []
    assert events == ["markdown"]


def test_attachment_failure_propagates_same_error_without_retry() -> None:
    failure = ReportChartAttachmentError("existing attachment error")
    renderer, builder, events = _dependencies(attachment_failure=failure)

    with pytest.raises(ReportChartAttachmentError) as caught:
        ReportBundleRenderer(renderer, builder).render(object())

    assert caught.value is failure
    assert len(renderer.calls) == len(builder.calls) == 1
    assert events == ["markdown", "attachments"]


@pytest.mark.parametrize("failing_dependency", ["markdown", "attachments"])
def test_unexpected_dependency_error_propagates_unchanged(
    failing_dependency: str,
) -> None:
    failure = RuntimeError("unexpected dependency failure")
    renderer, builder, events = _dependencies(
        markdown_failure=failure if failing_dependency == "markdown" else None,
        attachment_failure=failure if failing_dependency == "attachments" else None,
    )

    with pytest.raises(RuntimeError) as caught:
        ReportBundleRenderer(renderer, builder).render(object())

    assert caught.value is failure
    assert len(renderer.calls) == 1
    assert len(builder.calls) == (0 if failing_dependency == "markdown" else 1)
    assert events == (
        ["markdown"]
        if failing_dependency == "markdown"
        else ["markdown", "attachments"]
    )


def test_invalid_generation_candidate_is_forwarded_without_local_validation() -> None:
    candidate = object()
    failure = ReportMarkdownRenderingError("invalid generation")
    renderer, builder, _ = _dependencies(markdown_failure=failure)

    with pytest.raises(ReportMarkdownRenderingError) as caught:
        ReportBundleRenderer(renderer, builder).render(candidate)

    assert caught.value is failure
    assert renderer.calls == [candidate]
    assert renderer.calls[0] is candidate
    assert builder.calls == []


def test_bundle_is_frozen() -> None:
    renderer, builder, _ = _dependencies()
    bundle = ReportBundleRenderer(renderer, builder).render(object())

    with pytest.raises(FrozenInstanceError):
        bundle.markdown = ReportMarkdownArtifact("text/markdown", "changed")
    with pytest.raises(FrozenInstanceError):
        bundle.attachments = ReportAttachmentManifest(())
