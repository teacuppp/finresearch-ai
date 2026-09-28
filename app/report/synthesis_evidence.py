"""Deterministic factual evidence for future report synthesis."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, TypeAlias

from app.agent.sql_executor import SQLQueryResult
from app.analysis.financial_analyzer import AnalysisOperation, AnalysisResult
from app.rag.models import RetrievedChunk
from app.report.evidence import ReportEvidence, ReportResearchResult
from app.report.models import ReportPlan, ReportTask
from app.services.agent_service import AgentResult


SynthesisEvidenceKind: TypeAlias = Literal[
    "rag_source", "sql_result", "analysis_result"
]


@dataclass(frozen=True)
class SynthesisEvidenceItem:
    evidence_id: str
    task_id: str
    kind: SynthesisEvidenceKind
    content: str


@dataclass(frozen=True)
class ReportSynthesisEvidence:
    plan: ReportPlan
    items: tuple[SynthesisEvidenceItem, ...]


class ReportSynthesisEvidenceError(ValueError):
    """Research evidence is malformed or inconsistent with its plan."""


def _json_content(payload: object, task_id: str) -> str:
    try:
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ReportSynthesisEvidenceError(
            f"Task {task_id!r} contains non-serializable evidence."
        ) from exc


def _rag_content(source: RetrievedChunk, task_id: str) -> str:
    if (
        not isinstance(source.document, str)
        or not isinstance(source.text, str)
        or type(source.page) is not int
        or type(source.chunk_index) is not int
        or any(
            value is not None and not isinstance(value, str)
            for value in (source.company, source.ticker, source.document_type)
        )
        or (source.fiscal_year is not None and type(source.fiscal_year) is not int)
    ):
        raise ReportSynthesisEvidenceError(
            f"Task {task_id!r} contains a malformed RAG source."
        )

    # JSON-encoded metadata cannot introduce an extra field or line boundary.
    fields = (
        ("document", source.document),
        ("page", source.page),
        ("chunk_index", source.chunk_index),
        ("company", source.company),
        ("ticker", source.ticker),
        ("fiscal_year", source.fiscal_year),
        ("document_type", source.document_type),
    )
    lines = [f"{name}: {_json_content(value, task_id)}" for name, value in fields]
    lines.append(f"text_characters: {len(source.text)}")
    lines.append("text:")
    return "\n".join(lines) + "\n" + source.text


def _sql_content(result: SQLQueryResult, task_id: str) -> str:
    if (
        not isinstance(result.columns, list)
        or any(not isinstance(column, str) for column in result.columns)
        or not isinstance(result.rows, list)
        or any(
            not isinstance(row, dict)
            or any(not isinstance(key, str) for key in row)
            for row in result.rows
        )
        or type(result.row_count) is not int
        or result.row_count != len(result.rows)
    ):
        raise ReportSynthesisEvidenceError(
            f"Task {task_id!r} contains a malformed SQL result."
        )
    return _json_content(
        {
            "columns": result.columns,
            "row_count": result.row_count,
            "rows": result.rows,
        },
        task_id,
    )


def _analysis_content(result: AnalysisResult, task_id: str) -> str:
    if (
        not isinstance(result.operation, AnalysisOperation)
        or (result.value is not None and type(result.value) not in (int, float))
        or not isinstance(result.ranked_rows, tuple)
        or any(not isinstance(row, Mapping) for row in result.ranked_rows)
    ):
        raise ReportSynthesisEvidenceError(
            f"Task {task_id!r} contains a malformed analysis result."
        )
    return _json_content(
        {
            "operation": result.operation.value,
            "value": result.value,
            "ranked_rows": [dict(row) for row in result.ranked_rows],
        },
        task_id,
    )


def build_synthesis_evidence(
    research: ReportResearchResult,
) -> ReportSynthesisEvidence:
    if not isinstance(research, ReportResearchResult):
        raise ReportSynthesisEvidenceError("A ReportResearchResult is required.")
    if not isinstance(research.plan, ReportPlan) or not isinstance(
        research.evidence, tuple
    ):
        raise ReportSynthesisEvidenceError("Research plan or evidence is malformed.")
    if len(research.evidence) != len(research.plan.tasks):
        raise ReportSynthesisEvidenceError(
            "Research evidence count does not match plan task count."
        )

    items: list[SynthesisEvidenceItem] = []

    def add_item(task_id: str, kind: SynthesisEvidenceKind, content: str) -> None:
        items.append(SynthesisEvidenceItem(f"E{len(items) + 1}", task_id, kind, content))

    for task, evidence in zip(research.plan.tasks, research.evidence, strict=True):
        if not isinstance(task, ReportTask):
            raise ReportSynthesisEvidenceError("Research plan contains a malformed task.")
        if not isinstance(evidence, ReportEvidence) or not isinstance(
            evidence.task, ReportTask
        ):
            raise ReportSynthesisEvidenceError(
                f"Task {task.task_id!r} has malformed research evidence."
            )
        if evidence.task.task_id != task.task_id:
            raise ReportSynthesisEvidenceError(
                f"Task {task.task_id!r} does not match research evidence task_id."
            )
        result = evidence.result
        if not isinstance(result, AgentResult) or result.route != task.expected_route:
            raise ReportSynthesisEvidenceError(
                f"Task {task.task_id!r} has an invalid result route."
            )

        if task.expected_route == "rag":
            if (
                not isinstance(result.answer, str)
                or not isinstance(result.sources, list)
                or not result.sources
            ):
                raise ReportSynthesisEvidenceError(
                    f"Task {task.task_id!r} requires an answer and RAG sources."
                )
            for source in result.sources:
                if not isinstance(source, RetrievedChunk):
                    raise ReportSynthesisEvidenceError(
                        f"Task {task.task_id!r} contains a malformed RAG source."
                    )
                add_item(task.task_id, "rag_source", _rag_content(source, task.task_id))
        elif task.expected_route == "sql":
            if not isinstance(result.generated_sql, str) or not isinstance(
                result.sql_result, SQLQueryResult
            ):
                raise ReportSynthesisEvidenceError(
                    f"Task {task.task_id!r} requires a SQL query result."
                )
            add_item(task.task_id, "sql_result", _sql_content(result.sql_result, task.task_id))
            if result.analysis_result is not None:
                if not isinstance(result.analysis_result, AnalysisResult):
                    raise ReportSynthesisEvidenceError(
                        f"Task {task.task_id!r} has a malformed analysis result."
                    )
                add_item(
                    task.task_id,
                    "analysis_result",
                    _analysis_content(result.analysis_result, task.task_id),
                )
        else:
            raise ReportSynthesisEvidenceError(
                f"Task {task.task_id!r} has an unsupported route."
            )

    return ReportSynthesisEvidence(plan=research.plan, items=tuple(items))


def _render_content_json(content: str) -> str:
    return (
        json.dumps(content, ensure_ascii=False)
        .replace("\x85", "\\u0085")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def render_synthesis_context(evidence: ReportSynthesisEvidence) -> str:
    return "\n\n".join(
        f"[EVIDENCE {item.evidence_id}]\n"
        f"task_id: {item.task_id}\n"
        f"kind: {item.kind}\n"
        f"content_characters: {len(item.content)}\n"
        f"content_json: {_render_content_json(item.content)}\n"
        f"[/EVIDENCE {item.evidence_id}]"
        for item in evidence.items
    )
