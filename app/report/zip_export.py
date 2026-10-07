"""Deterministic, in-memory ZIP export for rendered reports."""

import json
import re
from dataclasses import dataclass
from io import BytesIO
from typing import Literal
from zipfile import ZIP_STORED, ZipFile, ZipInfo

from app.report.bundle import ReportRenderedBundle
from app.report.chart_attachments import ReportAttachment, ReportAttachmentManifest
from app.report.markdown_renderer import ReportMarkdownArtifact


class ReportZipExportError(ValueError):
    """A rendered report cannot be exported as a valid ZIP archive."""


@dataclass(frozen=True)
class ReportZipArtifact:
    media_type: Literal["application/zip"]
    filename: str
    content: bytes


_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


def _zip_member(name: str) -> ZipInfo:
    info = ZipInfo(filename=name, date_time=_ZIP_TIMESTAMP)
    info.compress_type = ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    info.extra = b""
    info.comment = b""
    return info


def _validated_attachments(bundle: ReportRenderedBundle) -> tuple[ReportAttachment, ...]:
    if not isinstance(bundle, ReportRenderedBundle):
        raise ValueError("a ReportRenderedBundle is required")
    markdown = bundle.markdown
    if (
        not isinstance(markdown, ReportMarkdownArtifact)
        or markdown.media_type != "text/markdown"
        or type(markdown.content) is not str
    ):
        raise ValueError("the bundle contains invalid Markdown")
    manifest = bundle.attachments
    if not isinstance(manifest, ReportAttachmentManifest) or not isinstance(
        manifest.attachments, tuple
    ):
        raise ValueError("the bundle contains an invalid attachment manifest")

    paths = {"manifest.json", "report.md"}
    for attachment in manifest.attachments:
        if not isinstance(attachment, ReportAttachment):
            raise ValueError("the manifest contains an invalid attachment")
        if (
            type(attachment.task_id) is not str
            or re.fullmatch(r"[a-z][a-z0-9_]{0,63}", attachment.task_id) is None
            or attachment.attachment_id != f"chart-{attachment.task_id}"
            or attachment.relative_path
            != f"attachments/{attachment.attachment_id}.png"
            or attachment.relative_path in paths
            or attachment.media_type != "image/png"
            or type(attachment.title) is not str
            or not attachment.title.strip()
            or type(attachment.content) is not bytes
            or not attachment.content
        ):
            raise ValueError("the manifest contains invalid chart metadata or bytes")
        paths.add(attachment.relative_path)
    return manifest.attachments


def _manifest_bytes(attachments: tuple[ReportAttachment, ...]) -> bytes:
    manifest = {
        "schema_version": 1,
        "markdown": {"media_type": "text/markdown", "path": "report.md"},
        "attachments": [
            {
                "attachment_id": attachment.attachment_id,
                "task_id": attachment.task_id,
                "media_type": attachment.media_type,
                "relative_path": attachment.relative_path,
                "title": attachment.title,
            }
            for attachment in attachments
        ],
    }
    return json.dumps(
        manifest,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8") + b"\n"


def build_report_zip(bundle: ReportRenderedBundle) -> ReportZipArtifact:
    if not isinstance(bundle, ReportRenderedBundle):
        raise ReportZipExportError("A ReportRenderedBundle is required.")
    try:
        attachments = _validated_attachments(bundle)
        output = BytesIO()
        with ZipFile(output, mode="w", compression=ZIP_STORED) as archive:
            archive.comment = b""
            archive.writestr(_zip_member("manifest.json"), _manifest_bytes(attachments))
            archive.writestr(
                _zip_member("report.md"), bundle.markdown.content.encode("utf-8")
            )
            for attachment in attachments:
                archive.writestr(_zip_member(attachment.relative_path), attachment.content)
        content = output.getvalue()
        if type(content) is not bytes or not content:
            raise ValueError("ZIP archive is empty or not bytes")
        return ReportZipArtifact(
            media_type="application/zip",
            filename="finresearch-report.zip",
            content=content,
        )
    except Exception as exc:
        raise ReportZipExportError("Report ZIP export failed.") from exc
