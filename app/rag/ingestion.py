from pathlib import Path

import pymupdf

from app.rag.models import DocumentChunk
from app.rag.splitter import split_text


def process_pdf(
    file_path: Path,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    company: str | None = None,
    ticker: str | None = None,
    fiscal_year: int | None = None,
    document_type: str | None = None,
) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []

    with pymupdf.open(file_path) as document:
        for page_number, page in enumerate(document):
            text = page.get_text("text", sort=True)

            page_chunks = split_text(
                text=text,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            for chunk_index, chunk_text in enumerate(page_chunks):
                chunks.append(
                    DocumentChunk(
                        text=chunk_text,
                        document=file_path.name,
                        page=page_number + 1,
                        chunk_index=chunk_index,
                        company=company,
                        ticker=ticker,
                        fiscal_year=fiscal_year,
                        document_type=document_type,
                    )
                )

    return chunks