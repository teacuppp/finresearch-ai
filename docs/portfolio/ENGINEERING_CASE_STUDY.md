# FinResearch AI: engineering a local financial research workflow

**Personal engineering project — Python, FastAPI, LangGraph, ChromaDB, SQLite, Ollama/Qwen3**

## Problem and scope

Financial research combines structured metrics with narrative disclosures. A revenue comparison can use database rows, while a supply-chain question needs filing text. FinResearch AI connects these sources in a local backend that produces reviewable answers and reports. The goal was to make model decisions explicit and keep calculations, provenance checks, and presentation inspectable. This is a personal project with locally seeded financial data and locally indexed PDFs. Its [reviewed demo](../demo/README.md) compares Apple and Microsoft reported FY2025 revenue and cites an Apple supply-chain disclosure; their fiscal years end on different dates.

## System architecture

[FastAPI lifespan](../../app/main.py) creates shared services through an [application composition function](../../app/services/rag_service.py). A [LangGraph graph](../../app/agent/graph.py) uses typed state to route a question to RAG or SQL. Nodes delegate to services and tools rather than reproduce their internals. The single-question agent chooses one primary route.

Mixed reports add orchestration above that graph. ReportPlanner creates independent tasks, ReportResearchExecutor calls the same AgentService for each, and ReportGenerationService prepares evidence and invokes synthesis. Research is sequential. Each task's expected route is checked against the actual agent result; it does not bypass routing. Presentation then combines Markdown and chart attachments. This separation lets tests substitute model clients while exercising the same contracts used by the application.

## RAG design and retrieval evaluation

PDF ingestion creates chunks with document, page, chunk, company, ticker, fiscal-year, and document-type metadata. [Metadata filters](../../app/rag/metadata.py) narrow Chroma retrieval before ranking; canonical filing forms also match legacy values such as `10K`. Sentence Transformers supplies embeddings. The [RAGPipeline](../../app/rag/pipeline.py) retrieves 20 dense candidates, applies a [CrossEncoder](../../app/rag/reranker.py), builds context from the final top-k sources, generates an answer, and validates citation syntax. One answer repair is allowed if validation fails.

The [historical eight-query benchmark](../../README.md#retrieval-benchmarks) recorded MRR@20 of 0.5333 for dense retrieval and 0.7500 with CrossEncoder reranking. MRR measures the reciprocal rank of the first relevant result, averaged across queries. The [labels](../../evaluation/retrieval_benchmark.json) identify document/page evidence with metric-row terms and nearby year context, making evidence completeness relevant to scoring.

BM25 and weighted RRF were also explored. Hybrid retrieval improved early ranking in the small dataset but regressed one total-revenue query outside the top five. Dense retrieval plus reranking remained the application default. These historical results guide a local design choice; eight queries cannot establish broad retrieval or answer accuracy.

## SQL safety and deterministic computation

The [SQL executor](../../app/agent/sql_executor.py) accepts a single SELECT or WITH statement and rejects comments, multiple statements, and forbidden write or administrative keywords. It opens SQLite with `mode=ro`, enables `query_only`, and enforces a result-row limit. These controls supplement the generation prompt. The graph permits two execution repairs by default before failing; its recursion limit also bounds traversal.

A SQL classifier separates direct queries from four supported analysis operations. For analysis, a typed retrieval plan selects raw values, a deterministic SQL builder constructs the initial query, and a model plan identifies rows and columns. [FinancialAnalyzer](../../app/analysis/financial_analyzer.py) performs absolute change, percentage change, difference, or ranking in Python. It rejects invalid indices, booleans, non-finite values, and zero percentage-change denominators. Deterministic arithmetic removes calculation from that model step, but cannot fix an incorrectly selected company, year, or metric. Direct SQL can still answer aggregations outside those four operations.

## Report generation and evidence grounding

Strict [report models](../../app/report/models.py) constrain task routes, scope, identifiers, and plan size. Planner validation checks requested years, structured metrics, and RAG scope. The [evidence builder](../../app/report/synthesis_evidence.py) assigns stable E1/E2 identifiers to SQL rows, optional analysis results, and retrieved RAG chunks. It excludes generated SQL, the intermediate RAG answer, and chart bytes from synthesis input.

Source content remains unchanged. Prompt rendering JSON-encodes it and escapes Unicode line separators so retrieved text cannot imitate evidence framing through literal line boundaries. The [synthesizer](../../app/report/synthesizer.py) validates title equality, known task/evidence IDs, same-task citations, task coverage, analysis/SQL pairing, and numeric values against cited evidence. Scalar analysis results must be communicated exactly. These checks constrain output; they do not prove sentence-level entailment or correct company-to-value associations.

Chart specifications are model-generated and validated against SQL columns. Matplotlib renders PNGs from those rows. Attachments remain on research results and reach the attachment builder directly. [ZIP export](../../app/report/zip_export.py) preserves Markdown and PNG bytes with fixed member metadata and ordering. Repeated export of the same bundle is deterministic; new model runs need not create the same bundle.

## Model integration challenges

### Comparison intent loss

In a live comparison case, the planner rewrote “compare Apple and Microsoft revenue” into a request to retrieve values. The downstream chart-intent classifier treated the rewritten task as a lookup, so no chart was selected. The [planner prompt](../../app/report/planner.py) was changed to preserve comparison semantics and ask for values side by side. This retains the user's intent without inventing a calculated difference. [Planner regressions](../../tests/test_report_planner.py) check the prompt contract; they do not guarantee every future model decomposition.

### RAG synthesis relevance drift

A separate investigation found relevant supply-chain chunks and a correct intermediate RAG answer, yet report synthesis selected unrelated wearables evidence. The defect was evidence selection during synthesis. A controlled A/B/C prompt experiment found a prompt-only task-relevance repair sufficient for the observed case. The shipped prompt requires every claim to answer its task and closely paraphrase supported facts, avoiding unsupported “significant risk” interpretations. The [synthesizer regression](../../tests/test_report_synthesizer.py) uses relevant and competing sources with a fake client to check prompt construction; it does not rerun the live experiment. This repair does not eliminate semantic errors.

### Ollama context configuration

An active context of 4096 also proved insufficient for the tested report workflow. Moving the runtime configuration to 8192 was a separate environment correction, not a code repair. The [live smoke command](../../scripts/smoke_report_live.py) checks known active contexts; unloaded models may have unknown context. Larger reports can exceed 8192, and extra capacity has memory costs.

## Testing and reproducibility

Tests use fakes for external models to exercise validation, failure paths, bounded repairs, and presentation determinism. The [last verified local snapshot](../../README.md#evaluation-and-testing) recorded 1,156 passing tests. This count is historical and does not measure model accuracy. Opt-in smoke checks run the real FastAPI lifespan, SQLite, Chroma, and Ollama composition. They check status, financial associations, citations, manifest paths, and PNG signatures; qualitative claims still require source review. The committed [demo report](../demo/report.md) and [chart](../demo/attachments/chart-sql_revenue_comparison.png) came from one reviewed real ZIP, preserving their original bytes. JSON and ZIP smoke requests generate independently, so their text may differ.

## Trade-offs and limitations

The project demonstrates backend composition, not a hosted production service. Model routing and synthesis remain fallible; small curated evaluations offer limited coverage. Numeric presence checks do not fully validate attribution, and citation syntax does not prove factual support. Local seeded SQL data is separate from the filing corpus. Reports run synchronously without production authentication, background jobs, or persistence. Next work would expand relevance and attribution evaluations, record model/context settings with evaluation runs, and add deployment controls only after defining operational requirements.
