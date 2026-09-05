## 🚀 Milestones

### ✅ v0.2.0 - Multi-document Entity-aware Retrieval *(2026-09-05)*
- **Metadata Filtering**: Upload PDFs with structured tags (`company`, `ticker`, `fiscal_year`, `document_type`).
- **Entity-aware Search**: Retrieval pipeline respects these tags, ensuring queries only search within specific entities (e.g., only "Apple" documents).
- **API Integration**: `POST /documents/upload` accepts metadata via `Form` data; `POST /rag/ask` supports optional filters.
- **Test Coverage**: Full unit tests with `dependency_overrides` to validate the metadata pipeline without heavy model loads.