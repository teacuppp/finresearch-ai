# FinResearch AI — Agentic RAG Platform

**Personal Project | Python, FastAPI, LangGraph, ChromaDB, SQLite, Ollama/Qwen3**

## Project summary

Built a local financial research backend that combines filing retrieval and structured financial analysis through a reusable agent workflow, producing cited Markdown reports, PNG charts, and downloadable ZIP bundles.

## Four-bullet CV version

- Built reusable LangGraph orchestration with typed state and dependency-injected services, routing financial questions to document RAG or SQL and reusing the same agent for mixed report research tasks.
- Implemented dense ChromaDB retrieval with CrossEncoder reranking; historical evaluation on eight manually labeled financial queries recorded MRR@20 of 0.7500 with reranking versus 0.5333 for dense retrieval alone.
- Added read-only SQLite execution with a bounded SQL repair loop and deterministic Python calculations for absolute change, percentage change, difference, and ranking, keeping supported derived results outside model-generated arithmetic.
- Delivered FastAPI report generation and export APIs with task-associated evidence citations, PNG chart attachments, and deterministic in-memory ZIP packaging; verified one real mixed report example alongside a last verified local suite snapshot of 1,156 passing tests.

## Shorter three-bullet version

- Built a Python/FastAPI financial research backend with reusable LangGraph orchestration across document RAG and read-only SQLite tools.
- Combined dense retrieval and CrossEncoder reranking with typed planning, deterministic financial calculations, and evidence-cited report synthesis.
- Delivered Markdown reports, PNG charts, and deterministic ZIP export, supported by tests with model fakes and a reviewed real mixed report demo.

## Relevant technical skills

- **Backend:** Python, FastAPI, Pydantic, dependency injection, typed state and protocols, HTTP request/response contracts.
- **AI orchestration:** LangGraph, Ollama/Qwen3, OpenAI-compatible client integration, structured model output, bounded repair workflows.
- **Retrieval:** Sentence Transformers embeddings, ChromaDB, metadata filtering, CrossEncoder reranking, experimental BM25 and Reciprocal Rank Fusion, Hit@k and MRR evaluation.
- **Data and presentation:** SQLite, read-only query validation, deterministic financial analysis, Matplotlib PNG rendering, Markdown, in-memory ZIP serialization.
- **Verification:** pytest, test doubles, FastAPI TestClient lifespan, opt-in live smoke checks, Git/GitHub workflows.

## Evidence and claim boundaries

This belongs under **Projects**, with “Personal Project” visible. It describes the local implementation and reviewed examples, without implying employment or a production deployment.

| Claim | Proof and qualification |
| --- | --- |
| Reusable agent orchestration | [Agent graph](../../app/agent/graph.py), [typed state](../../app/agent/state.py), and [report executor](../../app/report/executor.py). Each agent request selects one primary route; reports execute multiple independent tasks. |
| Retrieval MRR@20: 0.5333 → 0.7500 | [Historical eight-query results](../../README.md#retrieval-benchmarks), [labeled benchmark](../../evaluation/retrieval_benchmark.json), and [evaluation script](../../scripts/evaluate_retrieval.py). These measure retrieval ranking on a small financial dataset, not general answer accuracy. |
| Read-only SQL and deterministic analysis | [SQL executor](../../app/agent/sql_executor.py) and [FinancialAnalyzer](../../app/analysis/financial_analyzer.py). The default graph allows at most two SQL execution repairs. |
| Reports, charts, ZIP export | [Report API](../../app/api/reports.py), [chart renderer](../../app/analysis/chart_renderer.py), and [ZIP exporter](../../app/report/zip_export.py). Determinism applies to presentation for the same inputs; fresh model outputs can differ. |
| 1,156 passing tests | [Last verified local snapshot](../../README.md#evaluation-and-testing). This is a recorded suite result, not a live inference accuracy score or a permanent test count. |
| Validated real mixed report | [Demo notes](../demo/README.md), [unchanged report](../demo/report.md), and [chart](../demo/attachments/chart-sql_revenue_comparison.png). One reviewed example does not establish broad reliability. |

The saved 60/60 routing result used an earlier `qwen3:4b` configuration. The current application injects `qwen3:4b-instruct` for routing, so that historical score should not be presented as current-model accuracy.
