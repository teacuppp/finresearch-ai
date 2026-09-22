# FinResearch AI

[![CI](https://github.com/teacuppp/finresearch-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/teacuppp/finresearch-ai/actions/workflows/ci.yml)

**Agentic financial research over documents and local structured data.**

FinResearch AI ingests financial reports such as 10-K filings, retrieves document evidence, and generates grounded answers with source citations. Its Agent v1 API can also route structured financial questions to a read-only local SQLite database.

Retrieval, indexing, grounding, and evaluation remain explicit and independently testable. LangGraph orchestrates the existing services above the deterministic RAG pipeline.

## Overview

The document workflow is:

1. Upload a financial PDF with structured metadata.
2. Extract document text with PyMuPDF.
3. Split extracted text into overlapping chunks.
4. Generate dense embeddings with Sentence Transformers.
5. Store chunks, embeddings, and metadata in ChromaDB.
6. Filter the candidate corpus by company, ticker, fiscal year, or document type.
7. Retrieve the most relevant financial evidence.
8. Detect ambiguous multi-company queries before generation.
9. Generate an answer using a local Qwen3 model through Ollama.
10. Validate citations and repair malformed LLM responses.
11. Refuse unsupported questions when retrieved evidence is insufficient.
12. Evaluate retrieval quality against a manually labeled benchmark.

`POST /agent/ask` adds a router above this workflow: it selects document RAG or a read-only SQL query for each question. The existing `POST /rag/ask` endpoint remains available for direct RAG requests.

## Key Features

* PDF ingestion with **PyMuPDF**
* Fixed-size character chunking with overlap
* Sentence Transformer embeddings
* Persistent **ChromaDB** vector storage
* Multi-document financial corpus
* Metadata-aware retrieval using:

  * company
  * ticker
  * fiscal year
  * document type
* Entity-aware vector search
* Ambiguous query detection for multi-company retrieval
* Local **Qwen3** generation through Ollama
* Grounded answers restricted to retrieved context
* Source citations in `[Source N]` format
* Citation validation and one-step answer repair
* Safe refusal for unsupported questions
* Duplicate document replacement during re-indexing
* Shared application services through FastAPI lifespan
* Dependency injection for testable API components
* Manually labeled retrieval benchmark
* Hit@1, Hit@3, Hit@5, and MRR@20 evaluation
* Automated pytest test suite
* GitHub Actions CI
* FastAPI Swagger documentation
* Typed LangGraph state and conditional RAG/SQL routing
* Structured Qwen3:4b route selection
* Read-only SQL generation and execution against local SQLite
* Bounded SQL execution repair with explicit graph state
* Route-specific typed responses from `POST /agent/ask`
* Labeled routing evaluation

## Project Status

| Capability                      | Status      |
| ------------------------------- | ----------- |
| Multi-document indexing         | ✅ Completed |
| Entity-aware metadata filtering | ✅ Completed |
| Duplicate document replacement  | ✅ Completed |
| Ambiguous query detection       | ✅ Completed |
| Grounded generation             | ✅ Completed |
| Citation validation and repair  | ✅ Completed |
| Retrieval benchmark             | ✅ Completed |
| Hit@K and MRR evaluation        | ✅ Completed |
| GitHub Actions CI               | ✅ Completed |
| Cross-encoder reranking         | ✅ Completed |
| Agent v1 RAG/SQL routing        | ✅ Completed |
| Agent HTTP API                  | ✅ Completed |
| Bounded SQL self-repair         | ✅ Completed |

Current focus: **multi-step analysis and reporting.** Retrieval quality remains measured separately against the labeled benchmark below.

## Retrieval Evaluation

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

### Current Baseline

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

### Initial Finding

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

The benchmark will be reused unchanged when comparing future chunking, embedding, reranking, and hybrid-retrieval strategies.

Run the retrieval benchmark with:

```bash
python -m scripts.evaluate_retrieval
```

### Retrieval Chunking Experiments

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

### 8-query retrieval benchmark

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

### Hybrid Retrieval Experiments

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

## Routing Evaluation

The frozen Router v1 uses Qwen3:4b structured output to choose exactly one of two classes: `rag` or `sql`. Saved results for two curated, labeled sets are:

| Evaluation set | Examples | SQL / RAG | Correct |
| --- | ---: | ---: | ---: |
| Initial routing benchmark | 36 | 18 / 18 | 36/36 |
| Routing challenge v1 | 24 | 12 / 12 | 24/24 |

Router v1 was **60/60 correct across the two curated labeled routing evaluation sets**. This is not a measurement of real-world routing accuracy.

Run the evaluations with a local Ollama `qwen3:4b` model:

```bash
.venv/bin/python -m scripts.evaluate_routing
.venv/bin/python -m scripts.evaluate_routing \
  --benchmark evaluation/routing_challenge_v1.json
```

## Agent Architecture

```text
POST /agent/ask
  → AgentService → compiled LangGraph → LLMQuestionRouter (Qwen3:4b)
  → conditional routing
      rag → QueryService → existing RAGPipeline
          → dense candidate retrieval → CrossEncoder reranking
          → grounded Qwen answer → validated source citations
      sql → generate_sql → execute_sql
          ├─ success → END
          └─ SQLExecutionError → record sql_error → check retry budget
              ├─ budget remains → repair_sql → execute_sql
              └─ exhausted → sql_failure → raise SQLExecutionError
```

The API constructs the same metadata filter used by `/rag/ask`. The RAG branch passes the filter, `top_k`, company, and ticker through `AgentService` and the graph to `QueryService`. Unscoped RAG questions return HTTP 400 when multiple companies are indexed. The deterministic `RAGPipeline` still owns retrieval, reranking, context building, generation, and validation; LangGraph sits above it and does not reimplement those steps. PDF ingestion continues to use PyMuPDF, chunking, Sentence Transformers, and ChromaDB.

### Agent v1 capabilities

* Structured SQL generation and read-only SQLite execution
* Bounded SQL self-repair through explicit generate, execute, repair, and failure nodes
* Typed LangGraph state and conditional SQL/RAG routing
* Structured Qwen routing and labeled routing benchmarks
* AgentService composition through the FastAPI lifespan
* RAG query-option propagation and ambiguity protection for unscoped multi-company RAG queries
* `POST /agent/ask` with route-specific typed API responses

## Tech Stack

| Layer                    | Technology                            |
| ------------------------ | ------------------------------------- |
| API                      | FastAPI                               |
| Language                 | Python 3.12                           |
| PDF processing           | PyMuPDF                               |
| Embeddings               | Sentence Transformers                 |
| Baseline embedding model | `all-MiniLM-L6-v2`                    |
| Vector database          | ChromaDB                              |
| Local LLM                | Qwen3                                 |
| Model runtime            | Ollama                                |
| LLM client               | OpenAI-compatible Python SDK          |
| Agent orchestration      | LangGraph                            |
| Structured data backend  | Local SQLite                         |
| Validation               | Pydantic + custom citation validation |
| Retrieval evaluation     | Hit@K + MRR@20                        |
| Testing                  | pytest                                |
| CI                       | GitHub Actions                        |

## API

### Health Check

```http
GET /health
```

### Upload and Index a Financial Document

```http
POST /documents/upload
```

The upload endpoint accepts:

* PDF file
* company
* ticker
* fiscal year
* document type

Example:

```bash
curl -X POST "http://127.0.0.1:8000/documents/upload" \
  -F "file=@annual-report.pdf" \
  -F "company=Microsoft" \
  -F "ticker=MSFT" \
  -F "fiscal_year=2025" \
  -F "document_type=10-K"
```

### Agent: Ask a Financial Question

```http
POST /agent/ask
```

JSON request fields:

| Field | Type | Use |
| --- | --- | --- |
| `question` | string | Required, nonblank; at most 1,000 characters |
| `top_k` | integer | RAG result limit; defaults to 5 (1–20) |
| `company` | string or null | RAG company scope and ambiguity check |
| `ticker` | string or null | RAG ticker scope and ambiguity check |
| `fiscal_year` | integer or null | RAG metadata filter |
| `document_type` | string or null | RAG metadata filter |

Illustrative RAG request for narrative evidence:

```json
{
  "question": "How does Apple describe supply chain risks in its 2025 filing?",
  "company": "Apple",
  "ticker": "AAPL",
  "fiscal_year": 2025,
  "document_type": "10-K",
  "top_k": 5
}
```

RAG response shape, using the existing source fields:

```json
{
  "route": "rag",
  "answer": "Apple describes supply chain disruption as a risk. [Source 1]",
  "sources": [
    {
      "document": "apple.pdf",
      "page": 12,
      "chunk_index": 3,
      "distance": 0.2,
      "company": "Apple",
      "ticker": "AAPL",
      "fiscal_year": 2025,
      "document_type": "10-K"
    }
  ]
}
```

Illustrative SQL request for a structured metric (using a value in the local demo database):

```json
{
  "question": "What was Apple's revenue in fiscal 2025?"
}
```

SQL response shape; generated SQL may vary:

```json
{
  "route": "sql",
  "generated_sql": "SELECT revenue_musd FROM financial_metrics WHERE ticker = 'AAPL' AND fiscal_year = 2025",
  "sql_result": {
    "columns": ["revenue_musd"],
    "rows": [{"revenue_musd": 416161.0}],
    "row_count": 1
  }
}
```

The SQL branch returns the executed query and rows; it does not generate a narrative answer. After a repair, `generated_sql` is the final SQL that executed successfully. The public `/agent/ask` request and response contract did not change when bounded repair was added. Metadata fields filter the RAG branch. For SQL questions, include required company and year information in `question`.

### Direct RAG Endpoint

```http
POST /rag/ask
```

Example:

```bash
curl -X POST "http://127.0.0.1:8000/rag/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What was total revenue in 2025?",
    "ticker": "MSFT",
    "fiscal_year": 2025,
    "top_k": 5
  }'
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/teacuppp/finresearch-ai.git
cd finresearch-ai
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Install and prepare Ollama

Install Ollama, then download the local model:

```bash
ollama pull qwen3:4b
```

Make sure the Ollama service is running.

### 5. Initialize the local SQL demo database

```bash
python -m scripts.init_financial_db
```

This creates `data/financial_demo.db` with sample `financial_metrics` rows. Running the script again replaces that local demo database.

### 6. Start the API

```bash
uvicorn app.main:app --reload
```

Open Swagger UI:

```text
http://127.0.0.1:8000/docs
```

On the first run, Sentence Transformers may download the embedding model from Hugging Face.

## Project Structure

Selected implementation and evaluation files:

```text
finresearch-ai/
├── app/
│   ├── agent/
│   │   ├── graph.py
│   │   ├── router.py
│   │   ├── schema.py
│   │   ├── sql_executor.py
│   │   ├── sql_generator.py
│   │   └── state.py
│   ├── api/
│   │   ├── agent.py
│   │   ├── documents.py
│   │   ├── filters.py
│   │   └── rag.py
│   ├── evaluation/
│   │   ├── benchmark.py
│   │   ├── models.py
│   │   ├── retrieval_metrics.py
│   │   ├── routing_benchmark.py
│   │   └── routing_metrics.py
│   ├── rag/
│   │   ├── context.py
│   │   ├── embeddings.py
│   │   ├── generator.py
│   │   ├── hybrid_retriever.py
│   │   ├── ingestion.py
│   │   ├── lexical_retriever.py
│   │   ├── models.py
│   │   ├── pipeline.py
│   │   ├── reranker.py
│   │   ├── retriever.py
│   │   ├── splitter.py
│   │   ├── validation.py
│   │   └── vector_store.py
│   ├── services/
│   │   ├── agent_service.py
│   │   ├── document_service.py
│   │   ├── query_service.py
│   │   └── rag_service.py
│   ├── dependencies.py
│   └── main.py
├── evaluation/
│   ├── retrieval_benchmark.json
│   ├── routing_benchmark.json
│   ├── routing_baseline_v1.txt
│   ├── routing_challenge_v1.json
│   └── routing_challenge_v1_baseline.txt
├── scripts/
│   ├── debug_retrieval.py
│   ├── evaluate_retrieval.py
│   ├── evaluate_routing.py
│   ├── init_financial_db.py
│   └── inspect_document_chunks.py
├── tests/                      # API, agent, RAG, SQL, and evaluation tests
├── requirements.txt
├── pytest.ini
└── README.md
```

## Testing

Run the complete test suite:

```bash
pytest
```

Run the retrieval benchmark:

```bash
python -m scripts.evaluate_retrieval
```

The tests cover:

* document ingestion
* text chunking
* embeddings
* vector storage
* document replacement
* metadata filtering
* retrieval
* ambiguity detection
* RAG pipeline behavior
* citation validation
* answer repair
* benchmark loading and validation
* retrieval evaluation metrics
* FastAPI endpoints
* agent graph, router, and AgentService behavior
* read-only SQL generation and execution
* route-specific agent API responses
* routing benchmark loading and metrics

GitHub Actions runs the test suite automatically for pushes and pull requests targeting `main`.

## Engineering Decisions

### Explicit RAG Components

The retrieval pipeline is implemented through separate ingestion, embedding, vector-store, retrieval, context-building, generation, and validation layers.

This keeps the underlying RAG mechanics visible and independently testable.

AgentService and LangGraph call the existing QueryService and RAGPipeline rather than duplicating RAG logic in graph nodes.

### Metadata-aware Retrieval

Financial documents are indexed with structured metadata such as ticker and fiscal year.

Chroma metadata filters restrict the candidate corpus before semantic retrieval.

### Ambiguity Detection

When multiple companies are indexed, a query that does not specify a company or ticker is rejected before RAG generation.

This prevents the model from silently combining evidence across different entities.

### Grounded Generation

The generator is instructed to answer only from retrieved evidence and refuse unsupported questions rather than fabricate financial values.

### Citation Guardrail

Generated factual answers must contain source citations.

Invalid responses trigger a single repair attempt before the API returns a controlled error.

### Shared Model Lifecycle

Embedding models and shared services are initialized through FastAPI lifespan rather than being recreated for every HTTP request.

### Document Replacement

Re-indexing a document removes its previous vector records before inserting the new version, preventing stale chunks from remaining in the collection.

### Retrieval Benchmark

Retrieval quality is measured independently from LLM generation.

The benchmark uses manually labeled financial queries and relevant source chunks to make retrieval changes measurable and reproducible.

This allows chunking, embeddings, hybrid retrieval, and reranking strategies to be compared against the same baseline rather than evaluated through anecdotal examples.

### Read-only SQL and Bounded Self-repair

`generate_sql` calls SQLGenerator once, stores `generated_sql`, initializes `sql_retry_count` to `0`, and clears `sql_error`. `execute_sql` then sends that SQL to SQLExecutor, which applies the existing read-only validation and opens the local SQLite database in read-only mode.

If execution raises `SQLExecutionError`, the graph records a safe error message in the explicit `sql_error` state field and checks the retry budget. When a repair is available, `repair_sql` calls `SQLGenerator.repair()` with the original question, financial schema, failed SQL, and execution error. It replaces `generated_sql`, increments the explicit `sql_retry_count` state field by one, clears `sql_error`, and routes the repaired SQL back through `execute_sql`.

`max_sql_retries` defaults to `2`. It counts repair attempts, so a request permits at most two repairs and three total SQL executions: the initial query plus two repaired queries. When that budget is exhausted, `sql_failure` raises `SQLExecutionError` with the final execution error. The graph never returns a partial SQL result.

Every repaired query still passes SQLGenerator's existing read-only validation and SQLExecutor's validation and execution path. `SQLValidationError` is not retried or repaired, so forbidden write or administrative SQL cannot enter the recovery loop.

#### Controlled live smoke test

A controlled live smoke test exercised one known column-name failure against the local `financial_metrics` database:

```text
Initial SQL:
SELECT revenue FROM financial_metrics
WHERE ticker = 'AAPL' AND fiscal_year = 2025

SQLite error:
no such column: revenue

Repaired SQL:
SELECT revenue_musd
FROM financial_metrics
WHERE ticker = 'AAPL' AND fiscal_year = 2025

Result:
revenue_musd = 416161
sql_retry_count = 1
```

This verifies the controlled generate → execute → repair → execute path for that case. It is not a general SQL-repair accuracy benchmark.

## Known Limitations

* Each request selects one route: RAG **or** SQL. Mixed-intent questions requiring both are not yet supported.
* The SQL route does not consume API metadata filters as structured SQL constraints. Put required company and year information in the natural-language question.
* SQLite remains the local structured-data backend.
* There is no Python analysis, chart, or report workflow yet.
* There is no Databricks integration yet.

## Roadmap

1. [x] Add bounded SQL repair and retry in the graph.
2. [ ] Add multi-step analysis and reporting, including mixed SQL/RAG questions and later Python analysis, charts, and reports.
3. [ ] Continue expanding the retrieval benchmark and evaluating retrieval changes against it.

Other future integrations, including Databricks, remain outside Agent v1.

## Milestones

### v0.2.0 — Multi-document Entity-aware Retrieval

* Structured financial document metadata
* Metadata-aware Chroma retrieval
* Company / ticker / fiscal-year filtering
* FastAPI RAG endpoint
* Grounded generation with citations
* Citation repair and controlled failures
* Duplicate document replacement
* Automated tests

### Retrieval Engineering — Measured Ranking

* Manually labeled financial retrieval benchmark
* Hit@1, Hit@3, Hit@5, and MRR@20 metrics
* Automated retrieval evaluator
* Retrieval inspection utilities
* Baseline retrieval measurements
* Identification of chunk-boundary retrieval failures
* Dense candidate retrieval followed by CrossEncoder reranking in production
* Experimental BM25 and hybrid RRF retrieval

### Agent v1 — RAG/SQL Routing

* Structured Qwen3:4b routing into `rag` or `sql`
* Typed LangGraph state, compiled conditional graph, and AgentService
* Existing RAGPipeline reused through QueryService with scoped query options
* Structured SQL generation and read-only local SQLite execution
* Bounded SQL execution repair with two repair attempts by default
* `POST /agent/ask` with distinct typed RAG and SQL responses
* Two curated routing evaluations totaling 60/60 correct decisions

## Status

FinResearch AI is under active development.

Agent v1 and bounded SQL self-repair are implemented. Multi-step analysis and reporting are next; retrieval improvements continue to be evaluated against labeled evidence.
