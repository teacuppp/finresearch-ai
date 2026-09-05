import io

import pymupdf
from fastapi.testclient import TestClient

from app.dependencies import get_document_service
from app.main import app


class FakeDocumentService:
    def __init__(self):
        self.last_file_path = None
        self.last_company = None
        self.last_ticker = None
        self.last_fiscal_year = None
        self.last_document_type = None

    def index_pdf(
        self,
        file_path,
        company=None,
        ticker=None,
        fiscal_year=None,
        document_type=None,
    ) -> int:
        self.last_file_path = file_path
        self.last_company = company
        self.last_ticker = ticker
        self.last_fiscal_year = fiscal_year
        self.last_document_type = document_type

        return 3


fake_document_service = FakeDocumentService()


def override_get_document_service():
    return fake_document_service


app.dependency_overrides[
    get_document_service
] = override_get_document_service


client = TestClient(app)


def create_test_pdf() -> bytes:
    document = pymupdf.open()

    page = document.new_page()

    page.insert_text(
        (72, 72),
        "FinResearch AI financial document.",
    )

    pdf_bytes = document.tobytes()

    document.close()

    return pdf_bytes


def test_upload_pdf():
    pdf_bytes = create_test_pdf()

    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "test.pdf",
                io.BytesIO(pdf_bytes),
                "application/pdf",
            )
        },
        data={
        "company": "Microsoft",
        "ticker": "MSFT",
        "fiscal_year": "2025",
        "document_type": "10-K",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["filename"] == "test.pdf"

    assert (
        data["content_type"]
        == "application/pdf"
    )

    assert data["chunk_count"] == 3
    assert data["indexed"] is True

    assert (
        fake_document_service.last_file_path.name
        == "test.pdf"
    )

    assert fake_document_service.last_company == "Microsoft"
    assert fake_document_service.last_ticker == "MSFT"
    assert fake_document_service.last_fiscal_year == 2025
    assert fake_document_service.last_document_type == "10-K"


def test_upload_rejects_non_pdf():
    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "notes.txt",
                io.BytesIO(b"hello"),
                "text/plain",
            )
        },
        data={
        "company": "Microsoft",
        "ticker": "MSFT",
        "fiscal_year": "2025",
        "document_type": "10-K",
        },  
    )

    assert response.status_code == 400


class FailingDocumentService:
    def index_pdf(
        self,
        file_path,
    ) -> int:
        raise RuntimeError(
            "Indexing failed"
        )


def test_upload_handles_indexing_failure():
    def override_failure():
        return FailingDocumentService()

    app.dependency_overrides[
        get_document_service
    ] = override_failure

    pdf_bytes = create_test_pdf()

    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "broken.pdf",
                io.BytesIO(pdf_bytes),
                "application/pdf",
            )
        },
        data={
        "company": "Microsoft",
        "ticker": "MSFT",
        "fiscal_year": "2025",
        "document_type": "10-K",
        },
    )

    assert response.status_code == 400

    assert (
        "Failed to process document"
        in response.json()["detail"]
    )

    app.dependency_overrides[
        get_document_service
    ] = override_get_document_service