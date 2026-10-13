# FinResearch AI interview preparation

**Personal project for Junior AI Engineer, GenAI Engineer, RAG Engineer, and Python AI Backend Engineer interviews in Hong Kong.**

## 30-second project pitch

FinResearch AI is my personal Python project for financial research. I built a FastAPI backend that uses LangGraph to route questions to filing retrieval or read-only SQL. It combines dense retrieval with CrossEncoder reranking, performs supported financial calculations in Python, and generates cited reports with charts and ZIP export. I tested the contracts with model fakes and reviewed a real mixed report against its source evidence.

## 90-second technical explanation

I built FinResearch AI to handle two kinds of financial evidence: database metrics and narrative filing disclosures. FastAPI creates shared services, and a typed LangGraph workflow routes each question to RAG or SQL. The RAG branch retrieves dense candidates from Chroma within metadata filters, reranks them with a CrossEncoder, builds context, and generates a cited answer.

For SQL, I validate read-only statements and use a read-only SQLite connection. A bounded repair loop handles execution errors. Supported change, difference, and ranking operations use typed plans and deterministic Python calculations.

Reports sit above the same agent. A planner creates tasks, an executor gathers results, and a synthesizer creates claims linked to task-specific evidence IDs. It receives retrieved filing chunks and SQL results, rather than the intermediate RAG answer. Code checks citation ownership, coverage, and numeric support, while charts and ZIP packaging have separate presentation boundaries.

One useful debugging lesson was that correct retrieval did not guarantee a relevant report claim: synthesis chose unrelated evidence until I strengthened the task-relevance prompt. The project has a reviewed real demo, but its small benchmarks and structural checks do not establish general factual accuracy.

## Technical questions and grounded answers

### 1. Why use LangGraph, and where does it sit in the architecture?

LangGraph makes route selection, SQL mode, execution outcomes, and repair transitions explicit. Its state carries the question, selected route, rows, errors, analysis, and chart results. Nodes delegate to injected tools and services. RAGPipeline retains the retrieve → rerank → context → generate → validate boundary. ReportResearchExecutor calls the existing AgentService sequentially for each task, so report generation shares the agent implementation rather than adding another graph. See the [graph](../../app/agent/graph.py), [state](../../app/agent/state.py), and [report executor](../../app/report/executor.py).

### 2. How do you decide between RAG and SQL, including mixed questions?

The structured router selects SQL when the financial schema fully answers a question, such as revenue or net income. Narrative risks, strategy, and explanatory drivers use RAG. A single `/agent/ask` call chooses one primary route. Mixed reports use a separate planner to create independent SQL and RAG tasks. Their expected routes are checked after execution, not forced into the router. The application currently injects `qwen3:4b-instruct`; the saved 60/60 result used the earlier `qwen3:4b` configuration and should not be quoted as current-model accuracy. See [routing](../../app/agent/router.py) and [composition](../../app/services/rag_service.py).

### 3. Why rerank dense results, and how did you measure it?

Dense retrieval provides candidate recall using embeddings. The CrossEncoder then scores each query/chunk pair jointly and orders the candidates before selecting final context. RAGPipeline retrieves 20 candidates and defaults to five final chunks. The historical eight-query evaluation recorded MRR@20 of 0.5333 for dense retrieval and 0.7500 with reranking. It evaluates labeled evidence ranking, not generated-answer correctness. BM25 and RRF remain experimental because the small comparison included a top-five regression. See the [reranker](../../app/rag/reranker.py), [benchmark](../../evaluation/retrieval_benchmark.json), and [results](../../README.md#retrieval-benchmarks).

### 4. How does metadata filtering prevent evidence from the wrong filing?

Company, ticker, fiscal-year, and filing-form fields become Chroma predicates before dense retrieval. Multiple constraints use AND. Document forms are normalized, and queries match canonical and legacy spellings such as `10-K` and `10K`. Report RAG scopes require a company or ticker; QueryService rejects an unspecified company/ticker when multiple tickers exist. Filters rely on correct indexed metadata and do not establish truth. API metadata fields constrain the RAG route; SQL questions need company and year semantics in the question itself. See [metadata handling](../../app/rag/metadata.py), [QueryService](../../app/services/query_service.py), and [executor](../../app/report/executor.py).

### 5. What prevents generated SQL from modifying the database?

SQL validation permits one SELECT or WITH statement, rejects comments and extra statements, and blocks write or administrative keywords. Execution uses SQLite `mode=ro` and `PRAGMA query_only = ON`. A row limit rejects oversized results. The graph permits at most two execution repairs by default and then raises an error. The keyword validator is conservative, not a full SQL parser, and read-only access does not guarantee correct query semantics or bound query cost. See the [executor](../../app/agent/sql_executor.py), [graph](../../app/agent/graph.py), and [guardrail tests](../../tests/test_sql_executor.py).

### 6. Which calculations are deterministic, and what still depends on a model?

FinancialAnalyzer implements absolute change, percentage change, difference, and ranking. Models classify the operation and select retrieval parameters or row/column references; code retrieves raw values and computes the result. Validation rejects booleans, non-finite values, invalid indices, and a zero old value for percentage change. A wrong model plan can still choose the wrong data. Direct SQL handles lookups and other supported database queries, including aggregates; not every computation uses FinancialAnalyzer. Chart specification is also model-generated, while rendering uses existing rows. See [analysis](../../app/analysis/financial_analyzer.py), [SQL builder](../../app/analysis/sql_builder.py), and [tests](../../tests/test_financial_analyzer.py).

### 7. How do you integrate structured model output and handle invalid responses?

The OpenAI-compatible client calls Ollama with Pydantic response schemas for routes, plans, chart specifications, and drafts. Report models forbid extra fields, are frozen, and enforce identifiers, bounds, and route/scope consistency. Deterministic checks then validate relationships beyond the schema, such as requested years and citation ownership. Planner and synthesizer failures become domain errors; those layers make a single model call and do not silently retry or repair. The HTTP boundary maps known failures to stable responses while preserving causes. Structured output constrains shape, not meaning. See [models](../../app/report/models.py), [synthesis models](../../app/report/synthesis_models.py), and [API](../../app/api/reports.py).

### 8. Does citation validation prevent hallucinations or prompt injection?

It reduces some unsupported outputs but cannot guarantee factual correctness. RAG answer validation checks citation syntax or the insufficient-information response; it does not prove entailment. Report synthesis additionally checks existing evidence IDs, same-task ownership, coverage, numeric support, and analysis/SQL pairing. Source text is preserved but JSON-escaped to protect evidence framing, and the prompt treats it as untrusted data. Numeric membership alone can miss swapped company/value associations, while supported facts can still be irrelevant. Human review and broader semantic evaluations remain necessary. See [RAG validation](../../app/rag/validation.py), [evidence preparation](../../app/report/synthesis_evidence.py), and [synthesis checks](../../app/report/synthesizer.py).

### 9. How do your unit tests differ from real end-to-end verification?

Tests with fake clients verify contracts and failure paths without measuring real model behavior. The recorded local suite snapshot is 1,156 passing tests. The opt-in live smoke command uses the real FastAPI lifespan and service composition with SQLite, Chroma, and Ollama. It checks response structure, known financial associations, evidence references, ZIP paths, and PNG signatures, then prints qualitative claims for review. The [demo](../demo/README.md) is one source-reviewed mixed ZIP example. Its chart compares reported FY2025 revenue despite different fiscal-year ends. Neither the suite count nor one smoke result establishes broad reliability. See [smoke checks](../../scripts/smoke_report_live.py).

### 10. How did you diagnose model-related failures without redesigning the system?

I inspected outputs at the boundaries: plan questions, route and chart intent, retrieved chunks, intermediate answer, and synthesized claims. Comparison intent was lost when planning changed “compare” to “retrieve”; preserving comparison wording restored the intended chart path. In another case, retrieval and the intermediate answer were relevant, but synthesis chose wearables evidence. A controlled A/B/C prompt experiment supported a narrow task-relevance prompt repair. The shipped tests check those prompt contracts using fakes. I treated the 4096 → 8192 Ollama context change separately as runtime configuration. These observed fixes do not eliminate future model errors. See [planner](../../app/report/planner.py) and [synthesizer regressions](../../tests/test_report_synthesizer.py).

## Three limitations and realistic next improvements

1. **Limited semantic coverage.** Eight retrieval queries and one reviewed report provide narrow evidence. Citation and number checks can miss attribution errors and topical drift. Expand labeled cases across filings, years, and metrics, including competing RAG evidence and swapped company/value claims; measure retrieval, routing, and report relevance separately.
2. **Local synchronous application.** SQL uses seeded SQLite, filings must be indexed locally, and reports have no persistence, background queue, or production authentication. Define data-access and deployment requirements, then add authentication, request limits, job handling, and observable failures before considering a public service.
3. **Model and runtime sensitivity.** Generated plans, chart specifications, and prose vary; 8192 tokens is not enough for every report. Record model tags, context settings, corpus versions, and per-stage timings in future evaluations. Evaluate evidence-size budgets and context needs before adding another model or infrastructure layer.
