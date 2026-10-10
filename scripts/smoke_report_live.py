"""Run a predefined report smoke scenario against the real local application."""

import argparse
import json
import re
import sqlite3
import subprocess
import sys
from contextlib import closing
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from io import BytesIO
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Protocol, Sequence
from urllib.error import URLError
from urllib.request import urlopen
from zipfile import BadZipFile, ZipFile


DATABASE_PATH = Path("data/financial_demo.db")
CHROMA_PATH = Path("data/chroma")
OLLAMA_TAGS_URL = "http://127.0.0.1:11434/api/tags"
REQUIRED_MODELS = frozenset({"qwen3:4b", "qwen3:4b-instruct"})
REVENUE_MUSD = {"Apple": 416161, "Microsoft": 281724}
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
ZIP_DISPOSITION = 'attachment; filename="finresearch-report.zip"'
REQUESTS = {
    "comparison": "Create a concise report comparing Apple and Microsoft FY2025 revenue.",
    "rag": (
        "Create a concise report summarizing one supply-chain risk disclosed "
        "by Apple in its FY2025 Form 10-K."
    ),
    "mixed": (
        "Create a concise report comparing Apple and Microsoft FY2025 revenue "
        "and summarizing one supply-chain risk disclosed by Apple in its "
        "FY2025 Form 10-K."
    ),
}


class SmokeFailure(RuntimeError):
    """A live response did not meet the smoke contract."""


class PreconditionBlocked(RuntimeError):
    """The local environment cannot run the live smoke scenario."""

    def __init__(self, category: str, message: str, passed: Sequence[str] = ()) -> None:
        super().__init__(message)
        self.category = category
        self.passed = tuple(passed)


class CLIUsageError(ValueError):
    """A command-line choice is invalid."""


class SmokeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise CLIUsageError(message)


class HttpResponse(Protocol):
    status_code: int
    headers: Mapping[str, str]
    content: bytes

    def json(self) -> Any: ...


class HttpClient(Protocol):
    def post(self, path: str, *, json: dict[str, str]) -> HttpResponse: ...


@dataclass(frozen=True)
class ResponseSummary:
    markdown_length: int
    attachment_count: int
    member_names: tuple[str, ...] = ()
    qualitative_claim: str | None = None
    qualitative_source: str | None = None


def _parser() -> SmokeArgumentParser:
    parser = SmokeArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario", choices=tuple(REQUESTS), default="comparison"
    )
    parser.add_argument("--format", choices=("json", "zip", "both"), default="both")
    return parser


def _fetch_ollama_models() -> set[str]:
    with urlopen(OLLAMA_TAGS_URL, timeout=5) as response:
        payload = json.load(response)
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
        raise ValueError("Ollama returned an invalid model list")
    return {
        model["name"]
        for model in payload["models"]
        if isinstance(model, dict) and isinstance(model.get("name"), str)
    }


def _active_model_contexts() -> dict[str, int]:
    """Read active contexts; unloaded models are intentionally absent."""
    try:
        completed = subprocess.run(
            ["ollama", "ps"], capture_output=True, text=True, timeout=5, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}
    if completed.returncode != 0:
        return {}
    lines = completed.stdout.splitlines()
    if not lines:
        return {}
    headings = re.split(r"\s{2,}", lines[0].strip())
    if "NAME" not in headings or "CONTEXT" not in headings:
        return {}
    name_index = headings.index("NAME")
    context_index = headings.index("CONTEXT")
    contexts: dict[str, int] = {}
    for line in lines[1:]:
        columns = re.split(r"\s{2,}", line.strip())
        if len(columns) <= max(name_index, context_index):
            continue
        try:
            contexts[columns[name_index]] = int(columns[context_index].replace(",", ""))
        except ValueError:
            continue
    return contexts


def _check_database(database_path: Path) -> None:
    if not database_path.is_file():
        raise ValueError(f"{database_path} is missing; seed the local demo database first")
    uri = database_path.resolve().as_uri() + "?mode=ro"
    try:
        with closing(sqlite3.connect(uri, uri=True)) as connection:
            for company, expected in REVENUE_MUSD.items():
                rows = connection.execute(
                    "SELECT revenue_musd FROM financial_metrics "
                    "WHERE company = ? AND fiscal_year = ?",
                    (company, 2025),
                ).fetchall()
                if len(rows) != 1:
                    raise ValueError(
                        f"expected one FY2025 financial_metrics row for {company}; "
                        f"found {len(rows)}"
                    )
                try:
                    actual = Decimal(str(rows[0][0]))
                except (InvalidOperation, TypeError) as exc:
                    raise ValueError(f"invalid FY2025 revenue for {company}") from exc
                if actual != expected:
                    raise ValueError(
                        f"{company} FY2025 revenue_musd is {actual}; expected {expected}"
                    )
    except sqlite3.Error as exc:
        raise ValueError(f"cannot read financial_metrics from {database_path}: {exc}") from exc


def check_preconditions(
    scenario: str,
    *,
    database_path: Path = DATABASE_PATH,
    chroma_path: Path = CHROMA_PATH,
) -> tuple[str, ...]:
    passed: list[str] = []
    try:
        models = _fetch_ollama_models()
    except (OSError, URLError, ValueError) as exc:
        raise PreconditionBlocked(
            "OLLAMA", "Local Ollama is unavailable; start it and retry.", passed
        ) from exc
    passed.append("Ollama reachable")
    missing = sorted(REQUIRED_MODELS - models)
    if missing:
        raise PreconditionBlocked(
            "OLLAMA",
            f"Missing Ollama model(s): {', '.join(missing)}; install them before retrying.",
            passed,
        )
    passed.append("required Qwen models installed")

    contexts = _active_model_contexts()
    low = {name: size for name, size in contexts.items() if name in REQUIRED_MODELS and size < 8192}
    if low:
        detail = ", ".join(f"{name}={size}" for name, size in sorted(low.items()))
        raise PreconditionBlocked(
            "OLLAMA_CONTEXT",
            f"Active Ollama context below 8192 ({detail}); increase the Ollama "
            "context setting, reload the model, and verify with `ollama ps`.",
            passed,
        )
    passed.append(
        "known active Ollama context >=8192"
        if any(name in REQUIRED_MODELS for name in contexts)
        else "active Ollama context unknown (required models not loaded)"
    )

    try:
        _check_database(database_path)
    except ValueError as exc:
        raise PreconditionBlocked("LOCAL_DATA", str(exc), passed) from exc
    passed.append("FY2025 Apple/Microsoft revenue rows match demo values")

    if scenario in ("rag", "mixed"):
        try:
            corpus_present = chroma_path.is_dir() and any(chroma_path.iterdir())
        except OSError as exc:
            raise PreconditionBlocked(
                "LOCAL_DATA", f"Cannot read local corpus at {chroma_path}: {exc}", passed
            ) from exc
        if not corpus_present:
            raise PreconditionBlocked(
                "LOCAL_DATA",
                f"{chroma_path} has no local corpus; index the demo filing before retrying.",
                passed,
            )
        passed.append("local Chroma corpus present")
    return tuple(passed)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeFailure(message)


def _require_financial_values(markdown: str) -> None:
    readable = markdown.replace("\\,", ",")
    companies = re.compile(r"\b(?:Apple|Microsoft)\b", re.I)
    amounts = re.compile(
        r"(?<![\d,])(?P<value>\d{1,3}(?:,\d{3})+|\d+)(?![\d,])"
        r"\s+million\s+USD\b",
        re.I,
    )
    matched: set[str] = set()
    for line in readable.splitlines():
        mentions = list(companies.finditer(line))
        for index, mention in enumerate(mentions):
            company = mention.group().capitalize()
            end = mentions[index + 1].start() if index + 1 < len(mentions) else len(line)
            statement = re.split(r"[.;!?]", line[mention.end():end], 1)[0][:160]
            values = [int(match.group("value").replace(",", ""))
                      for match in amounts.finditer(statement)]
            if not values and index + 1 < len(mentions):
                rest_of_statement = re.split(
                    r"[.;!?]", line[mention.end():], 1
                )[0][:160]
                _require(
                    amounts.search(rest_of_statement) is None,
                    f"ambiguous {company} FY2025 revenue in million USD",
                )
            _require(
                len(values) <= 1,
                f"ambiguous {company} FY2025 revenue in million USD",
            )
            if values:
                _require(
                    values[0] == REVENUE_MUSD[company],
                    f"incorrect {company} FY2025 revenue in million USD",
                )
                matched.add(company)
    for company in REVENUE_MUSD:
        _require(
            company in matched,
            f"missing {company} FY2025 revenue in million USD",
        )


def _check_markdown(markdown: str, scenario: str) -> tuple[str | None, str | None]:
    _require(isinstance(markdown, str) and bool(markdown.strip()), "empty Markdown")
    evidence_heading = re.search(r"(?m)^## Evidence\s*$", markdown)
    _require(evidence_heading is not None, "Markdown has no Evidence section")
    assert evidence_heading is not None
    body = markdown[: evidence_heading.start()]
    evidence_section = markdown[evidence_heading.end() :]
    claim_lines = [
        line.strip()
        for line in body.splitlines()
        if line.strip() and not line.startswith("#")
    ]
    _require(bool(claim_lines), "Markdown has no report claim")
    if scenario in ("comparison", "mixed"):
        _require_financial_values(body)
    if scenario == "comparison":
        _require("SQL result" in evidence_section, "comparison has no SQL evidence")
        return None, None

    rag_ids = set(re.findall(r"(?m)^### (E\d+) — RAG source\s*$", evidence_section))
    citations = set(re.findall(r"\[(E\d+)\]", body))
    cited_rag_ids = rag_ids & citations
    _require(bool(cited_rag_ids), "no cited qualitative RAG source")
    if scenario == "mixed":
        _require("SQL result" in evidence_section, "mixed report has no SQL evidence")
    selected_id = next(
        (match for match in re.findall(r"\[(E\d+)\]", body) if match in cited_rag_ids)
    )
    claim = next(line for line in claim_lines if f"[{selected_id}]" in line)
    source_block = evidence_section.split(f"### {selected_id} — RAG source", 1)[1]
    source_block = source_block.split("\n### ", 1)[0]
    metadata = []
    for label in ("Document", "Page", "Chunk"):
        match = re.search(rf"(?m)^- {label}: (.+)$", source_block)
        if match:
            metadata.append(f"{label.lower()}={match.group(1)}")
    return claim[:240], f"{selected_id} " + " ".join(metadata)


def _check_attachment_count(count: int, scenario: str) -> None:
    if scenario == "comparison":
        _require(count == 1, "comparison requires exactly one PNG attachment")
    elif scenario == "mixed":
        _require(count >= 1, "mixed report requires a chart attachment")
    else:
        _require(count == 0, "RAG-only report must have no attachments")


def validate_json_response(response: HttpResponse, scenario: str) -> ResponseSummary:
    _require(response.status_code == 200, f"JSON endpoint returned HTTP {response.status_code}")
    try:
        payload = response.json()
    except (ValueError, TypeError) as exc:
        raise SmokeFailure("JSON endpoint returned invalid JSON") from exc
    _require(isinstance(payload, dict), "JSON response is not an object")
    markdown_data = payload.get("markdown")
    _require(isinstance(markdown_data, dict), "missing Markdown object")
    _require(markdown_data.get("media_type") == "text/markdown", "wrong Markdown media type")
    markdown = markdown_data.get("content")
    claim, source = _check_markdown(markdown, scenario)
    attachments = payload.get("attachments")
    _require(isinstance(attachments, list), "missing attachments list")
    _check_attachment_count(len(attachments), scenario)
    for attachment in attachments:
        _require(isinstance(attachment, dict), "invalid attachment metadata")
        _require(attachment.get("media_type") == "image/png", "attachment is not PNG")
        _require(
            isinstance(attachment.get("relative_path"), str)
            and attachment["relative_path"].startswith("attachments/")
            and attachment["relative_path"].endswith(".png"),
            "invalid attachment path",
        )
        _require("content" not in attachment, "JSON response exposed attachment bytes")
    return ResponseSummary(len(markdown), len(attachments), qualitative_claim=claim,
                           qualitative_source=source)


def validate_zip_response(response: HttpResponse, scenario: str) -> ResponseSummary:
    _require(response.status_code == 200, f"ZIP endpoint returned HTTP {response.status_code}")
    media_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    _require(media_type == "application/zip", "wrong ZIP media type")
    _require(
        response.headers.get("content-disposition") == ZIP_DISPOSITION,
        "wrong ZIP Content-Disposition",
    )
    try:
        with ZipFile(BytesIO(response.content)) as archive:
            names = tuple(archive.namelist())
            _require(len(names) == len(set(names)), "ZIP has duplicate member names")
            _require("manifest.json" in names and "report.md" in names, "ZIP missing core members")
            manifest = json.loads(archive.read("manifest.json"))
            _require(isinstance(manifest, dict), "ZIP manifest is not an object")
            _require(manifest.get("schema_version") == 1, "wrong manifest schema version")
            _require(
                manifest.get("markdown")
                == {"media_type": "text/markdown", "path": "report.md"},
                "invalid manifest Markdown path",
            )
            attachments = manifest.get("attachments")
            _require(isinstance(attachments, list), "invalid manifest attachments")
            _check_attachment_count(len(attachments), scenario)
            paths: list[str] = []
            for attachment in attachments:
                _require(isinstance(attachment, dict), "invalid manifest attachment")
                path = attachment.get("relative_path")
                _require(
                    isinstance(path, str) and path.startswith("attachments/")
                    and path.endswith(".png"),
                    "invalid manifest attachment path",
                )
                _require(attachment.get("media_type") == "image/png", "chart is not PNG")
                _require(path in names, f"manifest attachment missing from ZIP: {path}")
                _require(archive.read(path).startswith(PNG_SIGNATURE), f"invalid PNG signature: {path}")
                paths.append(path)
            _require(
                set(names) == {"manifest.json", "report.md", *paths},
                "ZIP members do not match manifest paths",
            )
            markdown = archive.read("report.md").decode("utf-8")
    except (BadZipFile, OSError, UnicodeDecodeError, ValueError, KeyError) as exc:
        raise SmokeFailure(f"invalid ZIP archive: {exc}") from exc
    claim, source = _check_markdown(markdown, scenario)
    return ResponseSummary(len(markdown), len(attachments), names, claim, source)


def run_endpoints(client: HttpClient, scenario: str, output_format: str) -> None:
    request = {"request": REQUESTS[scenario]}
    if output_format in ("json", "both"):
        started = perf_counter()
        response = client.post("/reports/generate", json=request)
        elapsed = perf_counter() - started
        print(f"JSON status={response.status_code} duration={elapsed:.2f}s")
        summary = validate_json_response(response, scenario)
        print(f"JSON markdown_chars={summary.markdown_length} attachments={summary.attachment_count}")
        if summary.qualitative_claim:
            print(f"JSON qualitative_claim={summary.qualitative_claim}")
            print(f"JSON qualitative_source={summary.qualitative_source}")
    if output_format in ("zip", "both"):
        started = perf_counter()
        response = client.post("/reports/export", json=request)
        elapsed = perf_counter() - started
        print(f"ZIP status={response.status_code} duration={elapsed:.2f}s")
        summary = validate_zip_response(response, scenario)
        print(f"ZIP markdown_chars={summary.markdown_length} attachments={summary.attachment_count}")
        print(f"ZIP members={', '.join(summary.member_names)}")
        if summary.qualitative_claim:
            print(f"ZIP qualitative_claim={summary.qualitative_claim}")
            print(f"ZIP qualitative_source={summary.qualitative_source}")


def _run_application(scenario: str, output_format: str) -> None:
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as client:
        run_endpoints(client, scenario, output_format)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    try:
        args = parser.parse_args(argv)
    except CLIUsageError as exc:
        print(parser.format_usage().strip(), file=sys.stderr)
        print(f"FAIL invalid arguments: {exc}", file=sys.stderr)
        return 1
    print(f"scenario={args.scenario} format={args.format}")
    try:
        checks = check_preconditions(args.scenario)
    except PreconditionBlocked as exc:
        for check in exc.passed:
            print(f"preflight PASS {check}")
        print(f"BLOCKED {exc.category}: {exc}")
        return 2
    for check in checks:
        print(f"preflight PASS {check}")
    try:
        _run_application(args.scenario, args.format)
    except SmokeFailure as exc:
        print(f"FAIL {exc}")
        return 1
    except Exception as exc:
        print(f"FAIL application error: {type(exc).__name__}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
