# Repository Guidelines

## Project Overview

FinResearch AI is an Agentic RAG platform for financial research.

The project combines:

- document ingestion and financial-document chunking,
- dense and lexical retrieval,
- CrossEncoder reranking,
- grounded LLM answer generation,
- retrieval evaluation,
- read-only SQL analysis,
- and an emerging LangGraph agent layer.

Preserve existing working abstractions unless a task explicitly requires changing them.

## Project Structure

Application code lives under `app/`.

- `app/api/`: FastAPI routes and request/response contracts.
- `app/services/`: application orchestration and dependency composition.
- `app/rag/`: deterministic document-RAG components.
- `app/agent/`: agent state, routing, graphs, and agent tools.
- `app/evaluation/`: retrieval benchmark models and metrics.
- `tests/`: pytest tests mirroring application components.
- `scripts/`: operational, debugging, indexing, and evaluation entry points.
- `evaluation/retrieval_benchmark.json`: labeled retrieval benchmark.

Local PDFs, Chroma state, SQLite databases, model caches, `.env`, and backup files must not be committed.

## Architecture Boundaries

Preserve `RAGPipeline` as the deterministic:

retrieve → rerank → context → generate → validate

boundary.

Agent workflows must orchestrate existing services rather than reimplement RAG internals.

LangGraph orchestration belongs above the existing service/RAG layer.

Prefer:

API
→ AgentService / graph
→ routing
→ QueryService / SQL tools
→ existing deterministic components

Do not embed LangGraph routing logic inside `RAGPipeline`.

Reuse `QueryService`, `RAGPipeline`, `VectorStore`, `SQLGenerator`, and `SQLExecutor` rather than duplicating their behavior inside graph nodes.

Existing `/rag/ask` behavior must remain compatible unless an explicit task changes its API contract.

## Retrieval Strategy

The current production retrieval strategy is:

dense candidate retrieval
→ CrossEncoder reranking
→ final top-k context

BM25 and weighted RRF hybrid retrieval are implemented for experimentation and evaluation, but are not the production default.

Do not change the production retrieval strategy merely because an alternative implementation exists.

Any change that can affect retrieval ranking, chunking, filtering, or reranking must be evaluated against `evaluation/retrieval_benchmark.json`.

Report benchmark changes rather than claiming retrieval improvement without measurements.

## Agent and SQL Safety

SQL execution must remain read-only.

Do not allow destructive or schema-mutating SQL statements.

SQL generation and repair loops must have explicit retry limits. Never introduce an unbounded agent or SQL-repair loop.

Keep agent state explicit and typed where practical.

Graph nodes should remain small and delegate domain behavior to existing services/tools.

Start with local SQLite abstractions before introducing external data platforms unless a task explicitly requires otherwise.

## Development Environment

Use Python 3.12.

Create and activate the environment with:

`python -m venv .venv && source .venv/bin/activate`

Install dependencies with:

`pip install -r requirements.txt`

Run the API locally with:

`uvicorn app.main:app --reload`

Ollama must be running with the configured `qwen3:4b` model for live answer generation.

## Coding Conventions

Use:

- four-space indentation,
- type hints,
- `snake_case` for functions and modules,
- `PascalCase` for classes,
- uppercase names for constants,
- small focused modules,
- dependency injection instead of constructing models or stores inside request handlers.

Follow nearby style. Do not perform unrelated refactors during a focused feature task.

Prefer existing abstractions over introducing parallel implementations.

## Testing

Use pytest.

Test files should be named `test_<component>.py` and tests `test_<behavior>()`.

Use fakes for LLMs, rerankers, and external model behavior where possible.

Use `tmp_path` for temporary Chroma, SQLite, PDF, or filesystem artifacts.

Behavior changes require focused tests covering relevant success, validation, empty-result, filtering, and failure paths.

After changing code:

1. Run the focused tests for the affected component.
2. Run `pytest -q` when practical.
3. Report any tests that were not run and why.
4. Never describe test collection alone as test execution.

Useful commands include:

- `pytest -q`
- `pytest tests/test_pipeline.py -v`
- `python -m scripts.evaluate_retrieval`

## Git and Change Discipline

Keep changes narrowly scoped.

Use short imperative Conventional Commit-style subjects, for example:

- `feat: add agent routing graph`
- `fix: exclude zero-score lexical matches`
- `test: cover SQL repair failure path`
- `docs: document retrieval benchmark results`

Do not create, amend, push, or merge commits unless explicitly requested by the user.

Do not rewrite existing history.

Before completing a coding task, report:

- files changed,
- important design decisions,
- tests executed and their results,
- `git diff --stat`,
- and any remaining uncertainty.

## Security

Never commit credentials, `.env`, local financial documents, generated databases, model caches, or backup directories.

Validate uploaded filenames, size, type, and target-path containment when changing ingestion behavior.

Keep SQL execution read-only.

Avoid introducing network calls, external services, or new infrastructure when a local deterministic implementation is sufficient for the requested feature.
