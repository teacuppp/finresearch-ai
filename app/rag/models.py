from dataclasses import dataclass


@dataclass
class DocumentChunk:
    text: str
    document: str
    page: int
    chunk_index: int

    company: str | None = None
    ticker: str | None = None
    fiscal_year: int | None = None
    document_type: str | None = None


@dataclass
class RetrievedChunk:
    text: str
    document: str
    page: int
    chunk_index: int
    distance: float

    company: str | None = None
    ticker: str | None = None
    fiscal_year: int | None = None
    document_type: str | None = None