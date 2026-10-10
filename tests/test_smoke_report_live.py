"""Deterministic checks for the opt-in live report smoke command."""

import json
import sqlite3
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest

from scripts import smoke_report_live as smoke


COMPARISON_MARKDOWN = (
    "# FY2025 comparison\n\n"
    "Apple reported 416\\,161 million USD and Microsoft reported "
    "281,724 million USD [E1].\n\n"
    "## Evidence\n\n### E1 — SQL result\n"
)
RAG_MARKDOWN = (
    "# Supply-chain risk\n\n"
    "Apple identifies supplier concentration as a risk [E3].\n\n"
    "## Evidence\n\n### E3 — RAG source\n"
    "- Document: Apple FY2025 10-K\n- Page: 12\n"
)
MIXED_MARKDOWN = (
    "# Combined report\n\n"
    "Apple reported 416161 million USD and Microsoft reported "
    "281724 million USD [E1].\n"
    "Apple identifies supplier concentration as a risk [E3].\n\n"
    "## Evidence\n\n### E1 — SQL result\n"
    "### E3 — RAG source\n- Document: Apple FY2025 10-K\n"
)
PNG = smoke.PNG_SIGNATURE + b"fake chart content"
CHART_PATH = "attachments/chart-revenue.png"


class FakeResponse:
    def __init__(self, *, status=200, payload=None, content=b"", headers=None):
        self.status_code = status
        self.payload = payload
        self.content = content
        self.headers = headers or {}

    def json(self):
        return self.payload


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, path, *, json):
        self.calls.append((path, json))
        return self.responses.pop(0)


def json_response(markdown=COMPARISON_MARKDOWN, chart_count=1):
    return FakeResponse(payload={
        "markdown": {"media_type": "text/markdown", "content": markdown},
        "attachments": [
            {"attachment_id": "chart-revenue", "task_id": "revenue",
             "media_type": "image/png", "relative_path": CHART_PATH,
             "title": "Revenue"}
            for _ in range(chart_count)
        ],
    })


def zip_response(markdown=COMPARISON_MARKDOWN, *, paths=(CHART_PATH,),
                 included_paths=None, png=PNG, content=None):
    if content is None:
        manifest = {
            "schema_version": 1,
            "markdown": {"media_type": "text/markdown", "path": "report.md"},
            "attachments": [
                {"relative_path": path, "media_type": "image/png"}
                for path in paths
            ],
        }
        stream = BytesIO()
        with ZipFile(stream, "w") as archive:
            archive.writestr("manifest.json", json.dumps(manifest))
            archive.writestr("report.md", markdown.encode("utf-8"))
            for path in paths if included_paths is None else included_paths:
                archive.writestr(path, png)
        content = stream.getvalue()
    return FakeResponse(
        content=content,
        headers={"content-type": "application/zip",
                 "content-disposition": smoke.ZIP_DISPOSITION},
    )


def demo_database(path: Path, apple=416161, microsoft=281724):
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE financial_metrics "
            "(company TEXT, fiscal_year INTEGER, revenue_musd INTEGER)"
        )
        connection.executemany(
            "INSERT INTO financial_metrics VALUES (?, 2025, ?)",
            [("Apple", apple), ("Microsoft", microsoft)],
        )


@pytest.mark.parametrize(
    ("argv", "scenario", "output_format"),
    [([], "comparison", "both"),
     (["--scenario", "rag", "--format", "json"], "rag", "json"),
     (["--scenario", "mixed", "--format", "zip"], "mixed", "zip")],
)
def test_argument_defaults_and_alternatives(argv, scenario, output_format):
    args = smoke._parser().parse_args(argv)
    assert (args.scenario, args.format) == (scenario, output_format)


@pytest.mark.parametrize("argv", [["--scenario", "unknown"], ["--format", "pdf"]])
def test_invalid_cli_choice_fails(argv, monkeypatch, capsys):
    monkeypatch.setattr(smoke, "check_preconditions", lambda *_: pytest.fail("preflight ran"))
    assert smoke.main(argv) == 1
    assert "FAIL invalid arguments" in capsys.readouterr().err


def test_help_does_not_initialize_application_or_preflight(monkeypatch, capsys):
    monkeypatch.setattr(smoke, "check_preconditions", lambda *_: pytest.fail("preflight ran"))
    monkeypatch.setattr(smoke, "_run_application", lambda *_: pytest.fail("app loaded"))
    with pytest.raises(SystemExit) as result:
        smoke.main(["--help"])
    assert result.value.code == 0
    assert "--scenario" in capsys.readouterr().out


def test_active_context_parsing_does_not_require_unloaded_models(monkeypatch):
    output = (
        "NAME               ID              SIZE      PROCESSOR    CONTEXT    UNTIL\n"
        "qwen3:4b           abc             2.6 GB    100% CPU     8192       4 minutes\n"
    )
    monkeypatch.setattr(smoke.subprocess, "run", lambda *a, **kw: SimpleNamespace(
        returncode=0, stdout=output))
    assert smoke._active_model_contexts() == {"qwen3:4b": 8192}


@pytest.mark.parametrize(
    ("models", "contexts", "category", "message"),
    [(None, {}, "OLLAMA", "unavailable"),
     ({"qwen3:4b"}, {}, "OLLAMA", "qwen3:4b-instruct"),
     (set(smoke.REQUIRED_MODELS), {"qwen3:4b": 4096},
      "OLLAMA_CONTEXT", "8192")],
)
def test_preflight_classifies_ollama_failures(
    monkeypatch, tmp_path, models, contexts, category, message
):
    def fetch():
        if models is None:
            raise OSError("connection refused")
        return models
    monkeypatch.setattr(smoke, "_fetch_ollama_models", fetch)
    monkeypatch.setattr(smoke, "_active_model_contexts", lambda: contexts)
    with pytest.raises(smoke.PreconditionBlocked) as caught:
        smoke.check_preconditions("comparison", database_path=tmp_path / "none")
    assert caught.value.category == category
    assert message in str(caught.value)


def test_database_check_is_read_only_and_detects_mismatch(tmp_path):
    path = tmp_path / "demo.db"
    demo_database(path)
    before = path.read_bytes()
    smoke._check_database(path)
    assert path.read_bytes() == before
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE financial_metrics SET revenue_musd=1 WHERE company='Apple'"
        )
    changed = path.read_bytes()
    with pytest.raises(ValueError, match="Apple FY2025 revenue_musd"):
        smoke._check_database(path)
    assert path.read_bytes() == changed


def test_database_preflight_classifies_local_data(monkeypatch, tmp_path):
    monkeypatch.setattr(smoke, "_fetch_ollama_models", lambda: set(smoke.REQUIRED_MODELS))
    monkeypatch.setattr(smoke, "_active_model_contexts", lambda: {})
    with pytest.raises(smoke.PreconditionBlocked) as caught:
        smoke.check_preconditions("comparison", database_path=tmp_path / "none")
    assert caught.value.category == "LOCAL_DATA"
    assert "missing" in str(caught.value)


def test_rag_preflight_requires_corpus_and_unknown_context_is_allowed(monkeypatch, tmp_path):
    path = tmp_path / "demo.db"
    demo_database(path)
    monkeypatch.setattr(smoke, "_fetch_ollama_models", lambda: set(smoke.REQUIRED_MODELS))
    monkeypatch.setattr(smoke, "_active_model_contexts", lambda: {})
    corpus = tmp_path / "chroma"
    with pytest.raises(smoke.PreconditionBlocked, match="local corpus"):
        smoke.check_preconditions("rag", database_path=path, chroma_path=corpus)
    corpus.mkdir()
    (corpus / "fixture").write_text("exists")
    checks = smoke.check_preconditions("mixed", database_path=path, chroma_path=corpus)
    assert "active Ollama context unknown (required models not loaded)" in checks
    assert "local Chroma corpus present" in checks


@pytest.mark.parametrize("markdown", [
    "Apple reported 416,161 million USD and Microsoft reported 281,724 million USD.",
    "Apple: 416161 million USD\nMicrosoft: 281724 million USD",
    "Apple 416\\,161 million USD\nMicrosoft 281\\,724 million USD",
    "| Company | FY2025 revenue |\n| Apple | 416,161 million USD |\n"
    "| Microsoft | 281,724 million USD |",
])
def test_financial_markdown_accepts_company_value_associations(markdown):
    smoke._require_financial_values(markdown)


def test_financial_markdown_rejects_wrong_values():
    with pytest.raises(smoke.SmokeFailure):
        smoke._require_financial_values("Apple 1 million USD; Microsoft 2 million USD")


@pytest.mark.parametrize("markdown", [
    "Apple reported 281,724 million USD and Microsoft reported 416,161 million USD.",
    "Apple reported 281,724 million USD and Microsoft reported "
    "416,161 million USD.\nThe figures were 416,161 and 281,724 million USD.",
    "Apple: 416161 million USD\nMicrosoft: 1 million USD",
    "Apple and Microsoft reported 416161 million USD and 281724 million USD.",
    "Apple and Microsoft reported 281724 million USD.\n"
    "Apple: 416161 million USD. Microsoft: 281724 million USD.",
    "Apple: 416161 million USD. Apple: 281724 million USD. "
    "Microsoft: 281724 million USD.",
    "Apple: 416161 million USD and 281724 million USD; "
    "Microsoft: 281724 million USD",
])
def test_financial_markdown_rejects_wrong_or_ambiguous_associations(markdown):
    with pytest.raises(smoke.SmokeFailure):
        smoke._require_financial_values(markdown)


@pytest.mark.parametrize("scenario,markdown,charts", [
    ("comparison", COMPARISON_MARKDOWN, 1),
    ("rag", RAG_MARKDOWN, 0),
    ("mixed", MIXED_MARKDOWN, 1),
])
def test_json_response_structure(scenario, markdown, charts):
    result = smoke.validate_json_response(json_response(markdown, charts), scenario)
    assert result.markdown_length == len(markdown)
    assert result.attachment_count == charts
    if scenario != "comparison":
        assert "supplier concentration" in result.qualitative_claim
        assert "Apple FY2025 10-K" in result.qualitative_source


def test_json_response_rejects_exposed_attachment_bytes():
    response = json_response()
    response.payload["attachments"][0]["content"] = "binary"
    with pytest.raises(smoke.SmokeFailure, match="exposed attachment bytes"):
        smoke.validate_json_response(response, "comparison")


@pytest.mark.parametrize("change,reason", [
    ("status", "HTTP 502"),
    ("media_type", "Markdown media type"),
    ("evidence", "Evidence section"),
    ("chart_count", "exactly one PNG"),
])
def test_json_response_rejects_invalid_structure(change, reason):
    response = json_response()
    if change == "status":
        response.status_code = 502
    elif change == "media_type":
        response.payload["markdown"]["media_type"] = "text/plain"
    elif change == "evidence":
        response.payload["markdown"]["content"] = "# Report\nNo evidence."
    else:
        response.payload["attachments"] = []
    with pytest.raises(smoke.SmokeFailure, match=reason):
        smoke.validate_json_response(response, "comparison")


def test_rag_response_requires_citation_to_rag_source():
    response = json_response(RAG_MARKDOWN.replace("[E3]", "[E99]"), 0)
    with pytest.raises(smoke.SmokeFailure, match="cited qualitative RAG source"):
        smoke.validate_json_response(response, "rag")


@pytest.mark.parametrize("scenario,markdown,paths", [
    ("comparison", COMPARISON_MARKDOWN, (CHART_PATH,)),
    ("rag", RAG_MARKDOWN, ()),
    ("mixed", MIXED_MARKDOWN, (CHART_PATH,)),
])
def test_zip_response_manifest_members_and_markdown(scenario, markdown, paths):
    result = smoke.validate_zip_response(zip_response(markdown, paths=paths), scenario)
    assert result.member_names == ("manifest.json", "report.md", *paths)
    assert result.markdown_length == len(markdown)


@pytest.mark.parametrize("response,reason", [
    (zip_response(content=b"not a zip"), "invalid ZIP archive"),
    (zip_response(png=b"not a png"), "invalid PNG signature"),
    (zip_response(included_paths=()), "missing from ZIP"),
])
def test_zip_response_rejects_corruption_and_missing_chart(response, reason):
    with pytest.raises(smoke.SmokeFailure, match=reason):
        smoke.validate_zip_response(response, "comparison")


def test_zip_response_rejects_wrong_media_type_and_status():
    response = zip_response()
    response.headers["content-type"] = "application/json"
    with pytest.raises(smoke.SmokeFailure, match="ZIP media type"):
        smoke.validate_zip_response(response, "comparison")
    response = zip_response()
    response.status_code = 500
    with pytest.raises(smoke.SmokeFailure, match="HTTP 500"):
        smoke.validate_zip_response(response, "comparison")


def test_both_mode_uses_sequential_independent_generations(capsys):
    client = FakeClient([json_response(), zip_response()])
    smoke.run_endpoints(client, "comparison", "both")
    assert client.calls == [
        ("/reports/generate", {"request": smoke.REQUESTS["comparison"]}),
        ("/reports/export", {"request": smoke.REQUESTS["comparison"]}),
    ]
    assert "JSON status=200" in capsys.readouterr().out


@pytest.mark.parametrize("outcome,code,verdict", [
    ("pass", 0, "PASS"), ("fail", 1, "FAIL"), ("blocked", 2, "BLOCKED")
])
def test_main_exit_codes(monkeypatch, capsys, outcome, code, verdict):
    def preflight(_scenario):
        if outcome == "blocked":
            raise smoke.PreconditionBlocked("LOCAL_DATA", "seed demo data")
        return ("demo ready",)
    def application(_scenario, _format):
        if outcome == "fail":
            raise smoke.SmokeFailure("bad response")
    monkeypatch.setattr(smoke, "check_preconditions", preflight)
    monkeypatch.setattr(smoke, "_run_application", application)
    assert smoke.main([]) == code
    assert verdict in capsys.readouterr().out
