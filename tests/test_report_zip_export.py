"""Deterministic in-memory ZIP export of rendered report bundles."""

import base64
import json
from dataclasses import FrozenInstanceError, replace
from io import BytesIO
from zipfile import ZIP_STORED, ZipFile

import pytest

from app.report.bundle import ReportRenderedBundle
from app.report.chart_attachments import ReportAttachment, ReportAttachmentManifest
from app.report.markdown_renderer import ReportMarkdownArtifact
from app.report.zip_export import (
    ReportZipArtifact,
    ReportZipExportError,
    build_report_zip,
)


def _attachment(
    task_id: str = "revenue", *, content: bytes = b"DISTINCTIVE_PNG_BYTES", title: str = "收入图"
) -> ReportAttachment:
    attachment_id = f"chart-{task_id}"
    return ReportAttachment(
        attachment_id=attachment_id,
        task_id=task_id,
        media_type="image/png",
        relative_path=f"attachments/{attachment_id}.png",
        title=title,
        content=content,
    )


def _bundle(
    *attachments: ReportAttachment,
    markdown: str = "# Report\n",
) -> ReportRenderedBundle:
    return ReportRenderedBundle(
        markdown=ReportMarkdownArtifact("text/markdown", markdown),
        attachments=ReportAttachmentManifest(attachments),
    )


def _names(archive: ZipFile) -> list[str]:
    return [info.filename for info in archive.infolist()]


def _assert_wrapped_failure(bundle: ReportRenderedBundle) -> None:
    with pytest.raises(ReportZipExportError, match="Report ZIP export failed") as caught:
        build_report_zip(bundle)
    assert caught.value.__cause__ is not None


def test_no_attachments_has_only_manifest_and_markdown() -> None:
    artifact = build_report_zip(_bundle())

    assert artifact.media_type == "application/zip"
    assert artifact.filename == "finresearch-report.zip"
    assert type(artifact.content) is bytes and artifact.content
    with ZipFile(BytesIO(artifact.content)) as archive:
        assert _names(archive) == ["manifest.json", "report.md"]
        assert json.loads(archive.read("manifest.json"))["attachments"] == []
        assert archive.read("report.md") == b"# Report\n"


def test_one_attachment_has_exact_member_order_and_png_bytes() -> None:
    original = b"\x89PNG\r\nDISTINCTIVE_BYTES"
    attachment = _attachment(content=original)
    artifact = build_report_zip(_bundle(attachment))

    with ZipFile(BytesIO(artifact.content)) as archive:
        assert _names(archive) == [
            "manifest.json", "report.md", "attachments/chart-revenue.png"
        ]
        assert archive.read(attachment.relative_path) == original
        assert "attachments/" not in _names(archive)


def test_multiple_attachments_preserve_non_alphabetical_manifest_order() -> None:
    first = _attachment("z_first", content=b"FIRST")
    second = _attachment("a_second", content=b"SECOND")
    artifact = build_report_zip(_bundle(first, second))

    with ZipFile(BytesIO(artifact.content)) as archive:
        assert _names(archive) == [
            "manifest.json",
            "report.md",
            first.relative_path,
            second.relative_path,
        ]
        assert archive.read(first.relative_path) == first.content
        assert archive.read(second.relative_path) == second.content


def test_markdown_is_preserved_as_exact_utf8_bytes() -> None:
    markdown = "# 报告\n\n## Evidence\n\nLiteral *punctuation* [E1]  \n"
    artifact = build_report_zip(_bundle(markdown=markdown))

    with ZipFile(BytesIO(artifact.content)) as archive:
        assert archive.read("report.md") == markdown.encode("utf-8")


def test_manifest_schema_metadata_and_serialization_are_exact() -> None:
    first = _attachment("z_first", content=b"SECRET_PNG_ONE", title="收入图")
    second = _attachment("a_second", content=b"SECRET_PNG_TWO", title="Second")
    artifact = build_report_zip(_bundle(first, second))
    expected = {
        "schema_version": 1,
        "markdown": {"media_type": "text/markdown", "path": "report.md"},
        "attachments": [
            {
                "attachment_id": attachment.attachment_id,
                "task_id": attachment.task_id,
                "media_type": "image/png",
                "relative_path": attachment.relative_path,
                "title": attachment.title,
            }
            for attachment in (first, second)
        ],
    }

    with ZipFile(BytesIO(artifact.content)) as archive:
        raw_manifest = archive.read("manifest.json")

    assert json.loads(raw_manifest) == expected
    assert raw_manifest == json.dumps(
        expected,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    assert "收入图".encode("utf-8") in raw_manifest
    assert all("content" not in item for item in expected["attachments"])
    for attachment in (first, second):
        assert attachment.content not in raw_manifest
        assert base64.b64encode(attachment.content) not in raw_manifest


def test_identical_bundle_produces_identical_archive_bytes() -> None:
    bundle = _bundle(_attachment())

    first = build_report_zip(bundle)
    second = build_report_zip(bundle)

    assert first == second
    assert first.content == second.content
    assert first.filename == second.filename == "finresearch-report.zip"
    assert first.media_type == second.media_type == "application/zip"


def test_all_member_metadata_and_archive_comment_are_fixed() -> None:
    artifact = build_report_zip(_bundle(_attachment()))

    with ZipFile(BytesIO(artifact.content)) as archive:
        assert archive.comment == b""
        for info in archive.infolist():
            assert info.date_time == (1980, 1, 1, 0, 0, 0)
            assert info.compress_type == ZIP_STORED
            assert info.create_system == 3
            assert info.external_attr == 0o100644 << 16
            assert info.extra == b""
            assert info.comment == b""


def test_zip_artifact_is_frozen() -> None:
    artifact = build_report_zip(_bundle())

    with pytest.raises(FrozenInstanceError):
        artifact.content = b"changed"


def test_wrong_top_level_input_has_direct_public_error() -> None:
    with pytest.raises(ReportZipExportError, match="ReportRenderedBundle") as caught:
        build_report_zip(object())
    assert caught.value.__cause__ is None


@pytest.mark.parametrize(
    "forgery",
    [
        "markdown_type",
        "markdown_media_type",
        "markdown_content",
        "manifest_type",
        "attachment_container",
        "attachment_type",
    ],
)
def test_forged_bundle_structure_fails_closed(forgery: str) -> None:
    bundle = _bundle(_attachment())
    if forgery == "markdown_type":
        bundle = replace(bundle, markdown=object())
    elif forgery == "markdown_media_type":
        bundle = replace(bundle, markdown=replace(bundle.markdown, media_type="text/html"))
    elif forgery == "markdown_content":
        bundle = replace(bundle, markdown=replace(bundle.markdown, content=b"not text"))
    elif forgery == "manifest_type":
        bundle = replace(bundle, attachments=object())
    elif forgery == "attachment_container":
        bundle = replace(bundle, attachments=ReportAttachmentManifest([_attachment()]))
    else:
        bundle = replace(bundle, attachments=ReportAttachmentManifest((object(),)))
    _assert_wrapped_failure(bundle)


@pytest.mark.parametrize(
    "changes",
    [
        {"media_type": "image/jpeg"},
        {"title": ""},
        {"title": "   "},
        {"title": 42},
        {"content": b""},
        {"content": "not bytes"},
        {"task_id": "Bad_Task"},
        {"task_id": "../evil"},
        {"attachment_id": "chart-other"},
        {"relative_path": "attachments/chart-other.png"},
        {"relative_path": "../evil.png"},
        {"relative_path": "/absolute.png"},
        {"relative_path": "attachments/../evil.png"},
        {"relative_path": "attachments\\evil.png"},
        {"relative_path": "report.md"},
        {"relative_path": "manifest.json"},
    ],
)
def test_forged_attachment_identity_path_or_content_fails_closed(
    changes: dict,
) -> None:
    attachment = replace(_attachment(), **changes)
    _assert_wrapped_failure(_bundle(attachment))


def test_duplicate_zip_member_path_is_rejected() -> None:
    attachment = _attachment()
    _assert_wrapped_failure(_bundle(attachment, attachment))
