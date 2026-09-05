# FinResearch AI

An entity-aware RAG platform for financial research, built with
FastAPI, Sentence Transformers, ChromaDB and local LLMs.

FinResearch AI ingests financial reports such as 10-K filings,
indexes them into a vector database, retrieves company-specific
evidence using metadata-aware semantic search, and generates
grounded answers with source citations.

## Key Features

- PDF ingestion and chunking with PyMuPDF
- Sentence Transformer embeddings
- Persistent ChromaDB vector storage
- Multi-document metadata filtering
- Company / ticker / fiscal-year aware retrieval
- Local Qwen3 generation through Ollama
- Citation validation and automatic answer repair
- Grounded refusal for unsupported questions
- FastAPI REST endpoints
- Automated unit tests