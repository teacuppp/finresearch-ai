"""Deterministic Markdown presentation of a generated report."""

import json
import re
import string
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from app.report.evidence import ReportResearchResult
from app.report.generation import ReportGenerationResult
from app.report.models import ReportPlan
from app.report.synthesis_evidence import ReportSynthesisEvidence, SynthesisEvidenceItem
from app.report.synthesis_models import ReportDraft


class ReportMarkdownRenderingError(ValueError):
    """The generation result cannot be rendered as safe Markdown."""


@dataclass(frozen=True)
class ReportMarkdownArtifact:
    media_type: Literal["text/markdown"]
    content: str


_LINE_BOUNDARIES = re.compile(r"\r\n|[\r\n\v\f\x1c-\x1e\x85\u2028\u2029]")
_PUNCTUATION = frozenset(string.punctuation)
_RAG_FIELDS = (
    "document",
    "page",
    "chunk_index",
    "company",
    "ticker",
    "fiscal_year",
    "document_type",
)


def _plain_text(value: str) -> str:
    flattened = _LINE_BOUNDARIES.sub(" ", value).strip()
    return "".join(f"\\{char}" if char in _PUNCTUATION else char for char in flattened)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value: {value}")


def _rag_metadata(content: str) -> dict[str, str | int | None]:
    prefix, separator, _source_text = content.partition("\ntext:\n")
    if not separator:
        raise ValueError("RAG evidence has no canonical text boundary")
    lines = prefix.split("\n")
    if len(lines) != len(_RAG_FIELDS) + 1:
        raise ValueError("RAG evidence has an invalid metadata prefix")

    metadata: dict[str, str | int | None] = {}
    for name, line in zip(_RAG_FIELDS, lines, strict=False):
        label = f"{name}: "
        if not line.startswith(label):
            raise ValueError(f"RAG evidence is missing {name}")
        metadata[name] = json.loads(
            line[len(label):], parse_constant=_reject_json_constant
        )
    length_line = lines[-1]
    if not length_line.startswith("text_characters: "):
        raise ValueError("RAG evidence has no text length")
    text_length = int(length_line.removeprefix("text_characters: "))
    if text_length < 0:
        raise ValueError("RAG evidence has invalid text length")

    if (
        not isinstance(metadata["document"], str)
        or type(metadata["page"]) is not int
        or type(metadata["chunk_index"]) is not int
        or any(
            value is not None and not isinstance(value, str)
            for value in (
                metadata["company"],
                metadata["ticker"],
                metadata["document_type"],
            )
        )
        or (
            metadata["fiscal_year"] is not None
            and type(metadata["fiscal_year"]) is not int
        )
    ):
        raise ValueError("RAG evidence metadata has invalid types")
    return metadata


def _sql_summary(content: str) -> tuple[int, list[str]]:
    payload = json.loads(content, parse_constant=_reject_json_constant)
    if not isinstance(payload, dict) or set(payload) != {
        "columns", "row_count", "rows"
    }:
        raise ValueError("SQL evidence has invalid fields")
    columns = payload["columns"]
    rows = payload["rows"]
    row_count = payload["row_count"]
    if (
        not isinstance(columns, list)
        or any(not isinstance(column, str) for column in columns)
        or not isinstance(rows, list)
        or any(not isinstance(row, dict) for row in rows)
        or type(row_count) is not int
        or row_count != len(rows)
    ):
        raise ValueError("SQL evidence has invalid summary values")
    return row_count, columns


def _analysis_summary(content: str) -> tuple[str, Decimal | None, int]:
    payload = json.loads(
        content,
        parse_float=Decimal,
        parse_int=Decimal,
        parse_constant=_reject_json_constant,
    )
    if not isinstance(payload, dict) or set(payload) != {
        "operation", "value", "ranked_rows"
    }:
        raise ValueError("analysis evidence has invalid fields")
    operation = payload["operation"]
    value = payload["value"]
    ranked_rows = payload["ranked_rows"]
    if (
        not isinstance(operation, str)
        or (value is not None and not isinstance(value, Decimal))
        or not isinstance(ranked_rows, list)
        or any(not isinstance(row, dict) for row in ranked_rows)
    ):
        raise ValueError("analysis evidence has invalid summary values")
    return operation, value, len(ranked_rows)


def _appendix_entry(item: SynthesisEvidenceItem) -> str:
    if item.kind == "rag_source":
        metadata = _rag_metadata(item.content)
        lines = [
            f"### {item.evidence_id} — RAG source",
            f"- Task: {_plain_text(item.task_id)}",
            f"- Document: {_plain_text(metadata['document'])}",
            f"- Page: {metadata['page']}",
            f"- Chunk: {metadata['chunk_index']}",
        ]
        for field, label in (
            ("company", "Company"),
            ("ticker", "Ticker"),
            ("fiscal_year", "Fiscal year"),
            ("document_type", "Document type"),
        ):
            value = metadata[field]
            if value is not None:
                lines.append(f"- {label}: {_plain_text(value) if isinstance(value, str) else value}")
        return "\n".join(lines)

    if item.kind == "sql_result":
        row_count, columns = _sql_summary(item.content)
        displayed_columns = ", ".join(_plain_text(column) for column in columns)
        return "\n".join(
            (
                f"### {item.evidence_id} — SQL result",
                f"- Task: {_plain_text(item.task_id)}",
                f"- Rows: {row_count}",
                f"- Columns: {displayed_columns if columns else '(none)'}",
            )
        )

    if item.kind == "analysis_result":
        operation, value, ranked_count = _analysis_summary(item.content)
        lines = [
            f"### {item.evidence_id} — Analysis result",
            f"- Task: {_plain_text(item.task_id)}",
            f"- Operation: {_plain_text(operation)}",
        ]
        if value is not None:
            lines.append(f"- Value: {value}")
        elif operation == "ranking":
            lines.append(f"- Ranked rows: {ranked_count}")
        return "\n".join(lines)

    raise ValueError("unsupported evidence kind")


def _cited_ids(generation: ReportGenerationResult) -> set[str]:
    if not isinstance(generation.research, ReportResearchResult):
        raise ValueError("research must be a ReportResearchResult")
    if not isinstance(generation.draft, ReportDraft):
        raise ValueError("draft must be a ReportDraft")
    ReportDraft.model_validate(generation.draft, strict=True)
    evidence = generation.synthesis_evidence
    if not isinstance(evidence, ReportSynthesisEvidence):
        raise ValueError("synthesis evidence has the wrong type")
    if not isinstance(evidence.plan, ReportPlan) or not isinstance(
        generation.research.plan, ReportPlan
    ):
        raise ValueError("generation contains an invalid plan")
    ReportPlan.model_validate(evidence.plan, strict=True)
    ReportPlan.model_validate(generation.research.plan, strict=True)
    if generation.research.plan != evidence.plan:
        raise ValueError("research and synthesis evidence plans differ")
    if generation.draft.title != evidence.plan.title:
        raise ValueError("draft title differs from the plan")
    if not isinstance(evidence.items, tuple):
        raise ValueError("synthesis evidence items must be a tuple")

    tasks = {task.task_id for task in evidence.plan.tasks}
    by_id: dict[str, SynthesisEvidenceItem] = {}
    for position, item in enumerate(evidence.items, start=1):
        if not isinstance(item, SynthesisEvidenceItem):
            raise ValueError("synthesis evidence contains an invalid item")
        if item.evidence_id != f"E{position}" or item.evidence_id in by_id:
            raise ValueError("synthesis evidence IDs must be consecutive and unique")
        if item.task_id not in tasks or item.kind not in (
            "rag_source", "sql_result", "analysis_result"
        ) or not isinstance(item.content, str):
            raise ValueError("synthesis evidence contains invalid item fields")
        by_id[item.evidence_id] = item

    cited: set[str] = set()
    for section in generation.draft.sections:
        for claim in section.claims:
            if claim.task_id not in tasks:
                raise ValueError("claim refers to an unknown task")
            for evidence_id in claim.evidence_ids:
                item = by_id.get(evidence_id)
                if item is None:
                    raise ValueError("claim cites an unknown evidence ID")
                if item.task_id != claim.task_id:
                    raise ValueError("claim cites evidence from another task")
                cited.add(evidence_id)
    return cited


class ReportMarkdownRenderer:
    def render(self, generation: ReportGenerationResult) -> ReportMarkdownArtifact:
        if not isinstance(generation, ReportGenerationResult):
            raise ReportMarkdownRenderingError("A ReportGenerationResult is required.")
        try:
            cited = _cited_ids(generation)
            blocks = [f"# {_plain_text(generation.draft.title)}"]
            for section in generation.draft.sections:
                blocks.append(f"## {_plain_text(section.heading)}")
                for claim in section.claims:
                    citations = " ".join(f"[{evidence_id}]" for evidence_id in claim.evidence_ids)
                    blocks.append(f"{_plain_text(claim.text)} {citations}")
            blocks.append("## Evidence")
            for item in generation.synthesis_evidence.items:
                if item.evidence_id in cited:
                    blocks.append(_appendix_entry(item))
            return ReportMarkdownArtifact(
                media_type="text/markdown",
                content="\n\n".join(blocks) + "\n",
            )
        except Exception as exc:
            raise ReportMarkdownRenderingError("Report Markdown rendering failed.") from exc
