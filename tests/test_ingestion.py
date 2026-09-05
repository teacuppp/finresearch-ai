import pymupdf

from app.rag.ingestion import process_pdf


def test_process_pdf(tmp_path):
    pdf_path = tmp_path / "test.pdf"

    document = pymupdf.open()

    page = document.new_page()
    page.insert_text(
        (72, 72),
        "FinResearch AI " * 100,
    )

    document.save(pdf_path)
    document.close()

    chunks = process_pdf(
    pdf_path,
    chunk_size=500,
    chunk_overlap=100,
    company="Microsoft",
    ticker="MSFT",
    fiscal_year=2025,
    document_type="10-K",
    )

    assert len(chunks) > 0

    first_chunk = chunks[0]

    assert first_chunk.document == "test.pdf"
    assert first_chunk.page == 1
    assert first_chunk.chunk_index == 0
    assert len(first_chunk.text) > 0
    assert first_chunk.company == "Microsoft"
    assert first_chunk.ticker == "MSFT"
    assert first_chunk.fiscal_year == 2025
    assert first_chunk.document_type == "10-K"