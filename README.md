# FinResearch AI

[![CI](https://github.com/teacuppp/finresearch-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/teacuppp/finresearch-ai/actions/workflows/ci.yml)

An Agentic RAG platform for financial research combining document retrieval, structured financial data analysis, grounded report generation, chart rendering, and deterministic report export.

FinResearch AI is a **local development demo** built with FastAPI, LangGraph, Ollama/Qwen3, ChromaDB, Sentence Transformers, a CrossEncoder reranker, and read-only SQLite. It shows how document evidence and structured financial results can flow through one agent and into a cited Markdown report. It does not include enterprise data infrastructure or a hosted service.

## What it can do

| Workflow | Implemented behavior |
| --- | --- |
| Document RAG | Upload PDFs with company, ticker, fiscal year, and document type metadata; index chunks in Chroma; retrieve dense candidates within metadata scope; rerank with a CrossEncoder; generate Qwen answers with source citations and validation. |
| Agent analysis | Route each question to **one** primary RAG or SQL path. SQL generation runs against read-only local SQLite with at most two execution repairs by default. Structured analysis supports absolute change, percentage change, difference, and ranking through deterministic Python calculations. SQL results can also drive chart decisions, plans, and PNG rendering. |
| Reports | Plan independent SQL and/or RAG research tasks, execute each through the shared AgentService, synthesize task-associated claims with evidence citations, render Markdown, attach existing SQL-task PNG charts, and return either JSON metadata or an in-memory ZIP archive. |

The report workflow can combine multiple SQL and RAG tasks. POST /agent/ask itself selects one primary route per request.

## Architecture

~~~mermaid
flowchart TD
    API["POST /reports/generate or /reports/export"] --> Generation["ReportGenerationService"]
    Generation --> Research["ReportService"]
    Research --> Planner["ReportPlanner"]
    Planner --> Executor["ReportResearchExecutor"]
    Executor --> Agent["Shared AgentService / LangGraph"]
    Agent --> SQL["SQL generation and bounded repair"]
    SQL --> SQLite["Read-only SQLite SQL result"]
    SQLite --> Analysis["Optional deterministic financial analysis"]
    SQLite --> ChartPlan["Optional model-generated ChartSpec"]
    ChartPlan --> ChartRender["Deterministic PNG chart rendering"]
    Agent --> Scope["RAG metadata filtering"]
    Scope --> Dense["Dense Chroma retrieval"]
    Dense --> Rerank["CrossEncoder reranking"]
    Rerank --> Sources["Retrieved RAG source chunks"]
    Rerank --> Answer["Grounded Qwen answer and validation; not synthesis input"]
    SQLite --> Results["ReportResearchResult"]
    Analysis --> Results
    ChartRender --> Results
    Sources --> Results
    Results -->|SQL rows, analysis results, RAG chunks| Evidence["ReportSynthesisEvidence"]
    Evidence --> Synth["ReportSynthesizer"]
    Synth --> Draft["ReportDraft"]
    Results -->|ChartSpec and PNG ChartArtifact| Attach["ReportChartAttachmentBuilder"]
    Draft --> Bundle["ReportBundleRenderer"]
    Attach --> Bundle
    Bundle --> Markdown["Markdown"]
    Bundle --> Charts["PNG attachment manifest"]
    Markdown --> JSON["JSON: Markdown plus attachment metadata"]
    Charts --> JSON
    Markdown --> ZIP["Deterministic ZIP: manifest, Markdown, PNG bytes"]
    Charts --> ZIP
~~~

The report executor calls the same AgentService used by POST /agent/ask; it does not create a separate agent graph. Each report task has an expected route and is checked against the actual agent result. The RAG branch reuses QueryService and RAGPipeline, which generates and validates an intermediate answer. Report synthesis receives the retrieved RAG source chunks, not that answer. For SQL tasks, synthesis evidence contains raw SQL rows and any deterministic analysis result.

Chart planning uses a local model to produce a ChartSpec; chart rendering then deterministically produces a PNG ChartArtifact. Both stay on the SQL research result, where ReportChartAttachmentBuilder reads them to create the attachment manifest. Chart bytes do not pass through ReportSynthesisEvidence or ReportSynthesizer. ReportBundleRenderer combines the draft's Markdown with that manifest for JSON or deterministic ZIP presentation. Generated prose and chart specifications still depend on model output.

## Quick start

These commands are for local macOS/Linux development from the repository root. Use Python 3.12 and install Ollama separately; keep its local service running.

~~~bash
git clone https://github.com/teacuppp/finresearch-ai.git
cd finresearch-ai
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
ollama pull qwen3:4b
ollama pull qwen3:4b-instruct
~~~

The tested report demo used an **active Ollama context length of 8192**. Check loaded models with `ollama ps`. If an active context is lower, adjust Ollama settings and reload the model before running the demo. Unloaded models need not appear in `ollama ps`. A context of 8192 does not guarantee every report will fit; larger reports can require more context and memory. Embedding and reranking models may also download on first startup.

### Prepare local data

The repository does not commit data/financial_demo.db, local Chroma state, or source PDFs. The initializer seeds Apple FY2025 revenue of **416161 million USD** and Microsoft FY2025 revenue of **281724 million USD**.

**The initializer deletes and replaces an existing data/financial_demo.db. Inspect or back up that file before running it.**

~~~bash
python -m scripts.init_financial_db
~~~

### Start the API

~~~bash
uvicorn app.main:app --reload
~~~

Open the local Swagger UI at http://127.0.0.1:8000/docs. GET /health returns a simple health response. Run the upload example below in another terminal. The development server uses the local service configuration; it is not a public hosted deployment.

RAG and mixed demos require the relevant financial filings to be uploaded and indexed locally. For example, after starting the API, upload your own Apple FY2025 Form 10-K PDF:

~~~bash
curl -X POST "http://127.0.0.1:8000/documents/upload" \
  -F "file=@/path/to/apple-2025-10k.pdf;type=application/pdf" \
  -F "company=Apple" \
  -F "ticker=AAPL" \
  -F "fiscal_year=2025" \
  -F "document_type=10-K"
~~~

The upload fields are file, company, ticker, fiscal_year, and document_type. This project does not download sample filings automatically.

## Live demo

The opt-in smoke command uses the real FastAPI lifespan and production service composition through TestClient; it does not require a separate uvicorn process. Its default is the comparison scenario with both JSON and ZIP checks:

~~~bash
python -m scripts.smoke_report_live
python -m scripts.smoke_report_live --scenario comparison --format both
python -m scripts.smoke_report_live --scenario rag --format json
python -m scripts.smoke_report_live --scenario mixed --format both
~~~

The choices are `--scenario comparison|rag|mixed` and `--format json|zip|both`. Exit status is **0 PASS**, **1 FAIL**, or **2 BLOCKED** by a local precondition. Preflight checks Ollama, both required model tags, known active context, and the canonical SQL demo rows; RAG and mixed scenarios also require a local Chroma corpus. The command validates JSON/ZIP structure, known company-to-revenue associations, manifest paths, and PNG signatures. It reports cited qualitative claims and sources for human review; those checks do not prove full semantic accuracy.

In `both` mode, JSON and ZIP requests generate reports independently inside one TestClient lifespan. The Markdown and ZIP bytes need not match across the two model runs. This smoke command is not part of CI.

## API examples

All examples assume the local development API is running.

| Endpoint | Purpose |
| --- | --- |
| GET /health | Local health response |
| POST /documents/upload | Multipart PDF upload and indexing; fields shown in [Prepare local data](#prepare-local-data) |
| POST /rag/ask | Direct cited RAG answer |
| POST /agent/ask | One primary RAG or SQL route with typed result |
| POST /reports/generate | Markdown plus chart attachment metadata as JSON |
| POST /reports/export | A complete ZIP report |

### Generate a JSON report

~~~bash
curl -X POST "http://127.0.0.1:8000/reports/generate" \
  -H "Content-Type: application/json" \
  -d '{"request":"Create a concise report comparing Apple and Microsoft FY2025 revenue."}'
~~~

The response has this **illustrative shape**, not guaranteed wording or task IDs:

~~~json
{
  "markdown": {
    "media_type": "text/markdown",
    "content": "# Example report\n\n..."
  },
  "attachments": [
    {
      "attachment_id": "chart-sql_revenue_comparison",
      "task_id": "sql_revenue_comparison",
      "media_type": "image/png",
      "relative_path": "attachments/chart-sql_revenue_comparison.png",
      "title": "Example chart"
    }
  ]
}
~~~

JSON attachments contain metadata only; PNG bytes are not embedded. A report without charts returns an empty attachments list.

### Export a ZIP report

~~~bash
curl -X POST "http://127.0.0.1:8000/reports/export" \
  -H "Content-Type: application/json" \
  -d '{"request":"Create a concise report comparing Apple and Microsoft FY2025 revenue."}' \
  --output finresearch-report.zip
~~~

The server responds with Content-Type `application/zip` and the fixed filename `finresearch-report.zip`. The archive contains `manifest.json`, `report.md`, and any chart attachments at `attachments/chart-<task_id>.png`. The ZIP builder uses fixed member metadata and in-memory bytes; the `--output` flag writes the response on the **client**. Report content itself varies with local models and evidence.

### Ask the existing RAG or agent APIs

~~~bash
curl -X POST "http://127.0.0.1:8000/rag/ask" \
  -H "Content-Type: application/json" \
  -d '{"question":"What supply-chain risk did Apple disclose in its FY2025 filing?","company":"Apple","ticker":"AAPL","fiscal_year":2025,"document_type":"10-K","top_k":5}'

curl -X POST "http://127.0.0.1:8000/agent/ask" \
  -H "Content-Type: application/json" \
  -d '{"question":"What was Apple revenue in fiscal 2025?"}'
~~~

Both accept a question, optional top_k (1–20, default 5), and optional company, ticker, fiscal_year, and document_type fields for RAG scope. The agent response is either a cited `rag` result or a `sql` result with generated SQL, rows, and an optional analysis result. SQL questions should include their company and year in the question because the API metadata fields filter the RAG path.

## Engineering highlights

- **Explicit composition:** FastAPI lifespan creates shared services, and typed LangGraph state routes agent work above QueryService and RAGPipeline.
- **Scoped retrieval:** metadata filters narrow the Chroma candidate corpus before dense retrieval; production retrieval then reranks with a CrossEncoder. BM25 and hybrid RRF are experimental alternatives.
- **Guardrails:** RAG validates citations and permits one answer repair. Report synthesis instructs the model to answer each task and avoid new arithmetic; validation checks task IDs, citation ownership and coverage, and numbers against cited evidence. Deterministic analysis results must be cited with their SQL source and communicated exactly. These checks reduce unsupported output but cannot guarantee semantic truth or relevance.
- **Read-only analysis:** SQL generation and execution enforce read-only access. Execution repair has a bounded retry budget; supported financial calculations run in deterministic Python rather than generated code.
- **Presentation boundaries:** chart attachment paths derive from task IDs; Markdown rendering and ZIP member metadata are deterministic for the same validated inputs. The JSON transport omits binary chart content.

## Evaluation and testing

The test suite uses fakes for external model behavior where practical. GitHub Actions runs `pytest -q` on pushes and pull requests targeting `main`; it does **not** run the live Ollama smoke command. A local snapshot after Phase 6.1 was **1156 passing tests**, not a permanent test-count promise.

~~~bash
HF_HUB_OFFLINE=1 python -m pytest -q
python -m scripts.evaluate_retrieval
python -m scripts.evaluate_routing
python -m scripts.evaluate_routing --benchmark evaluation/routing_challenge_v1.json
~~~

Retrieval evaluation requires the local indexed corpus. Routing evaluation requires local Ollama. The measurements below come from small labeled financial datasets; they are not a general accuracy claim.

### Retrieval benchmarks

The retriever is evaluated independently from the language model using a manually labeled benchmark built from Apple and Microsoft 2025 annual reports.

Each benchmark query contains one or more relevant source chunks identified by:

```text
(document, page, chunk_index)
```

A source is considered relevant when the retrieved chunk contains enough information to independently support the answer.

The evaluator retrieves the top 20 chunks for each query and reports:

* **Hit@1** — whether a relevant result appears at rank 1
* **Hit@3** — whether a relevant result appears in the top 3
* **Hit@5** — whether a relevant result appears in the top 5
* **MRR@20** — reciprocal rank of the first relevant result within the top 20, averaged across queries

#### Current Baseline

Benchmark size: **4 manually labeled queries**

| Metric |  Score |
| ------ | -----: |
| Hit@1  | 0.5000 |
| Hit@3  | 0.5000 |
| Hit@5  | 0.5000 |
| MRR@20 | 0.5385 |

Per-query results:

| Query                      | Relevant Ranks | Hit@1 | Hit@3 | Hit@5 |  RR@20 |
| -------------------------- | -------------- | ----: | ----: | ----: | -----: |
| AAPL Total Revenue 2025    | 13, 14         |     0 |     0 |     0 | 0.0769 |
| AAPL Services Revenue 2025 | 1, 2, 18       |     1 |     1 |     1 | 1.0000 |
| MSFT Total Revenue 2025    | 13             |     0 |     0 |     0 | 0.0769 |
| MSFT Operating Income 2025 | 1, 2           |     1 |     1 |     1 | 1.0000 |

#### Initial Finding

The benchmark exposes a recurring weakness in the current fixed-size character chunking strategy.

For some financial tables, a semantically strong chunk contains the table header and metric name but is split immediately before the final value. As a result, incomplete but highly similar chunks can outrank the chunks that contain complete answer evidence.

For example:

```text
Query:
What was Apple's total revenue in 2025?

Highly ranked partial chunk:
2025
Net sales
Products
Services
Total net sal
                 ← chunk boundary

Relevant chunk:
Services ...
Total net sales 416,161
```

This indicates that retrieval quality is currently limited not only by embedding similarity, but also by **chunk structure and evidence completeness**.

The labeled benchmark makes future chunking, embedding, reranking, and hybrid-retrieval comparisons reproducible; the expanded eight-query results follow.

Run the retrieval benchmark with:

```bash
python -m scripts.evaluate_retrieval
```

#### Retrieval Chunking Experiments

| Strategy | Hit@1 | Hit@3 | Hit@5 | MRR@20 |
|---|---:|---:|---:|---:|
| Fixed-size | 0.2500 | 0.2500 | 0.2500 | 0.2692 |
| Boundary-aware | 0.2500 | 0.2500 | 0.2500 | 0.2500 |
| Line + table-header aware | 0.0000 | 0.2500 | 1.0000 | 0.2458 |

The line-aware, table-header-preserving strategy increased
Hit@5 from 0.25 to 1.00, meaning all benchmark questions
retrieved self-contained answer evidence within the top five
candidates.

However, MRR did not improve because relevant chunks were
typically ranked between positions 3 and 5.

This shifted the primary bottleneck from evidence completeness
to candidate ranking.

#### 8-query retrieval benchmark

Benchmark size:
4 → 8 manually labeled queries

Dense:
Hit@1  = 0.3750
Hit@3  = 0.6250
Hit@5  = 1.0000
MRR@20 = 0.5333

Dense + CrossEncoder:
Hit@1  = 0.5000
Hit@3  = 1.0000
Hit@5  = 1.0000
MRR@20 = 0.7500

Observed failure modes:
- exact financial metric confusion
- broad revenue semantic confusion
- evidence year-binding limitations

Cross-encoder reranking improved MRR@20 from
0.5333 to 0.7500 while preserving Hit@5 = 1.0000.

Hit@3 improved from 0.6250 to 1.0000.

#### Hybrid Retrieval Experiments

An additional BM25 lexical retriever was evaluated alongside
dense retrieval using Reciprocal Rank Fusion (RRF).

The final experiment compared five retrieval strategies over the
same 8-query financial benchmark.

| Strategy | Hit@1 | Hit@3 | Hit@5 | MRR@20 |
|---|---:|---:|---:|---:|
| Dense | 0.3750 | 0.6250 | 1.0000 | 0.5333 |
| BM25 | 0.3750 | 0.5000 | 0.6250 | 0.4620 |
| Hybrid RRF | 0.7500 | 0.8750 | 0.8750 | 0.8281 |
| Dense + CrossEncoder | 0.5000 | 1.0000 | 1.0000 | 0.7500 |
| Hybrid RRF + CrossEncoder | 0.5000 | 1.0000 | 1.0000 | 0.7292 |

Hybrid RRF substantially improved early ranking quality, doubling
Hit@1 relative to dense retrieval and increasing MRR@20 from
0.5333 to 0.8281.

However, one Apple total-revenue query regressed from Dense Rank 5
to Hybrid Rank 8, reducing Hit@5 from 1.0000 to 0.8750.

Dense retrieval followed by CrossEncoder reranking therefore remains
the default production retrieval strategy because it preserves
Hit@5 = 1.0000 and Hit@3 = 1.0000 across the current benchmark.

Hybrid retrieval remains available as an experimental retrieval
strategy for future benchmark expansion and domain-specific reranker
experiments.

### Routing evaluation

The frozen Router v1 uses Qwen3:4b structured output to choose exactly one of two classes: `rag` or `sql`. Saved results for two curated, labeled sets are:

| Evaluation set | Examples | SQL / RAG | Correct |
| --- | ---: | ---: | ---: |
| Initial routing benchmark | 36 | 18 / 18 | 36/36 |
| Routing challenge v1 | 24 | 12 / 12 | 24/24 |

Router v1 was **60/60 correct across the two curated labeled routing evaluation sets**. This is not a measurement of real-world routing accuracy.

Those saved results used the router's default `qwen3:4b` model. The application currently injects `qwen3:4b-instruct` for routing, so the saved score is not a measurement of that current model configuration.

Run the evaluations with a local Ollama `qwen3:4b` model:

```bash
.venv/bin/python -m scripts.evaluate_routing
.venv/bin/python -m scripts.evaluate_routing \
  --benchmark evaluation/routing_challenge_v1.json
```

### Historical agent smoke observations

A controlled local test repaired a generated query that selected the nonexistent `revenue` column. The repaired query selected `revenue_musd` for Apple FY2025 and returned `416161` after one repair attempt. That observation verifies the bounded generate → execute → repair → execute path for this case; it is not a general SQL-repair accuracy benchmark.

Other selected live agent runs against the local demo data observed:

| Task | Observed result |
| --- | --- |
| Apple FY2025 revenue | Direct SQL: `revenue_musd = 416161` |
| Apple revenue change, 2024 to 2025 | Percentage change ≈ 6.4255%; absolute change 25126 million USD |
| Apple versus Microsoft FY2025 revenue | Difference 134437 million USD; ranking placed Apple first |
| Apple risk question | RAG answer citing filing evidence |

These are case observations, not an analysis accuracy benchmark.

## Project structure

~~~text
finresearch-ai/
├── app/
│   ├── agent/          # LangGraph, routing, SQL generation/execution
│   ├── analysis/       # deterministic calculations and chart components
│   ├── api/            # document, RAG, agent, and report routes
│   ├── evaluation/     # benchmark loading and metrics
│   ├── rag/            # ingestion, retrieval, reranking, generation
│   ├── report/         # planning, research, synthesis, presentation, ZIP
│   ├── services/       # application composition and service boundaries
│   └── main.py
├── scripts/
│   ├── init_financial_db.py
│   ├── smoke_report_live.py
│   ├── evaluate_retrieval.py
│   └── evaluate_routing.py
├── evaluation/          # labeled retrieval and routing cases
├── tests/
└── README.md
~~~

## Known limitations

- Inference depends on a running local Ollama service and available model context/memory. Reports remain model-dependent and can contain semantic errors despite evidence and citation checks.
- SQL uses a local SQLite demo rather than Databricks or another enterprise data platform. The SQL route does not consume RAG API metadata filters as structured SQL constraints.
- RAG and mixed reports need locally supplied and indexed filings; the small labeled retrieval benchmarks do not establish broad-domain performance.
- POST /agent/ask selects one primary route. The report workflow supports multiple SQL and RAG tasks, but that does not make the single-question agent endpoint a general mixed-route API.
- Reports are generated on demand. There is no production report persistence, background job queue, or report ID. The repository does not include a public hosted deployment or production authentication/security system.

## Roadmap

Completed in this local implementation: document RAG and retrieval evaluation, LangGraph routing, read-only SQL analysis, chart rendering, grounded report generation, JSON report APIs, deterministic ZIP export, and the opt-in live demo.

Future possibilities include Databricks integration, deployment and authentication hardening, broader retrieval benchmarks, optional persistence and background jobs, and further reproducible demos. These are plans, not current capabilities.
