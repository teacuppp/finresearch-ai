# FinResearch AI

[![CI](https://github.com/teacuppp/finresearch-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/teacuppp/finresearch-ai/actions/workflows/ci.yml)

**Entity-aware RAG platform for financial research.**

FinResearch AI ingests financial reports such as 10-K filings, converts them into searchable vector representations, performs metadata-aware semantic retrieval, and generates grounded financial answers with source citations.

The project is built from first principles to keep the retrieval, indexing, grounding, and evaluation mechanics explicit and independently testable before introducing higher-level agent frameworks.

## Overview

FinResearch AI supports a complete document-to-answer workflow:

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

Current focus: **retrieval quality optimization, especially financial-table chunking and evidence completeness.**

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

## Architecture

```mermaid
flowchart LR
    A[Financial PDF] --> B[Document Upload API]
    B --> C[PyMuPDF Extraction]
    C --> D[Chunking + Metadata]
    D --> E[Sentence Transformer]
    E --> F[(ChromaDB)]

    Q[User Question] --> G[Query Validation]
    G --> H[Metadata Filter]
    H --> I[Retriever]
    F --> I

    I --> J[Top-K Retrieved Chunks]

    J --> EV[Retrieval Evaluation]
    EV --> EM[Hit@K + MRR]

    J --> K[Context Builder]
    K --> L[Qwen3 via Ollama]
    L --> M[Citation Validation]

    M -->|Valid| N[Grounded Answer + Sources]
    M -->|Invalid| O[Answer Repair]
    O --> M
```

## Example

The same natural-language question can retrieve different financial evidence depending on metadata filters.

### Microsoft

Request:

```json
{
  "question": "What was total revenue in 2025?",
  "ticker": "MSFT",
  "fiscal_year": 2025,
  "top_k": 5
}
```

Example response:

```json
{
  "answer": "Microsoft reported total revenue of $281.724 billion in 2025. [Source 2]"
}
```

### Apple

Request:

```json
{
  "question": "What was total revenue in 2025?",
  "ticker": "AAPL",
  "fiscal_year": 2025,
  "top_k": 5
}
```

Example response:

```json
{
  "answer": "Apple reported total net sales of $416.161 billion in fiscal year 2025. [Source 2]"
}
```

This demonstrates entity-aware retrieval over a shared multi-document vector database.

### Ambiguous Query

When multiple companies are indexed, an unfiltered query such as:

```json
{
  "question": "What was total revenue in 2025?"
}
```

is rejected before retrieval and generation:

```json
{
  "detail": "The query is ambiguous. Please specify a company or ticker."
}
```

This prevents the system from silently mixing evidence from different companies.

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

### Ask a Financial Question

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

### 5. Start the API

```bash
uvicorn app.main:app --reload
```

Open Swagger UI:

```text
http://127.0.0.1:8000/docs
```

On the first run, Sentence Transformers may download the embedding model from Hugging Face.

## Project Structure

```text
finresearch-ai/
├── app/
│   ├── api/
│   │   ├── documents.py
│   │   ├── filters.py
│   │   └── rag.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── benchmark.py
│   │   ├── models.py
│   │   └── retrieval_metrics.py
│   ├── rag/
│   │   ├── context.py
│   │   ├── embeddings.py
│   │   ├── generator.py
│   │   ├── ingestion.py
│   │   ├── models.py
│   │   ├── pipeline.py
│   │   ├── retriever.py
│   │   ├── splitter.py
│   │   ├── validation.py
│   │   └── vector_store.py
│   ├── services/
│   ├── dependencies.py
│   └── main.py
├── evaluation/
│   └── retrieval_benchmark.json
├── scripts/
│   ├── debug_retrieval.py
│   ├── evaluate_retrieval.py
│   └── inspect_document_chunks.py
├── tests/
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

GitHub Actions runs the test suite automatically for pushes and pull requests targeting `main`.

## Engineering Decisions

### Explicit RAG Components

The retrieval pipeline is implemented through separate ingestion, embedding, vector-store, retrieval, context-building, generation, and validation layers.

This keeps the underlying RAG mechanics visible and independently testable.

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

## Roadmap

* [x] Multi-document financial indexing
* [x] Metadata-aware retrieval
* [x] Grounded generation and citation repair
* [x] Duplicate document replacement
* [x] Ambiguous query detection
* [x] Retrieval evaluation with Hit@1, Hit@3, Hit@5, and MRR@20
* [ ] Expand the retrieval benchmark dataset
* [ ] Structure-aware financial-document chunking
* [ ] Hybrid retrieval
* [ ] Reranking
* [ ] Multilingual financial retrieval
* [ ] Structured LLM outputs
* [ ] Financial market data tools
* [ ] SQL / Python analysis tools
* [ ] LangGraph agent workflows
* [ ] React frontend
* [ ] PostgreSQL / pgvector
* [ ] Docker and deployment

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

### Current Development — Measurable Retrieval

* Manually labeled financial retrieval benchmark
* Hit@1, Hit@3, Hit@5, and MRR@20 metrics
* Automated retrieval evaluator
* Retrieval inspection utilities
* Baseline retrieval measurements
* Identification of chunk-boundary retrieval failures

## Status

FinResearch AI is under active development.

The current focus is improving financial-document retrieval quality using measurable experiments before introducing agentic workflows.
