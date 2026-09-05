from pathlib import Path
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    Form,
    HTTPException,
    UploadFile,
)

from app.dependencies import get_document_service
from app.services.document_service import DocumentService


router = APIRouter(
    prefix="/documents",
    tags=["documents"],
)


DOCUMENT_DIR = Path("data/documents")
DOCUMENT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)



@router.post("/upload")
async def upload_document(
    file: UploadFile,
    company: Annotated[str, Form()],
    ticker: Annotated[str, Form()],
    fiscal_year: Annotated[int, Form()],
    document_type: Annotated[str, Form()],
    document_service: Annotated[
        DocumentService,
        Depends(get_document_service),
    ],
):
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required.",
        )

    file_path = DOCUMENT_DIR / file.filename

    content = await file.read()

    file_path.write_bytes(content)

    try:
        chunk_count = document_service.index_pdf(
            file_path=file_path,
            company=company,
            ticker=ticker,
            fiscal_year=fiscal_year,
            document_type=document_type,
        )

    except Exception as exc:
        file_path.unlink(
            missing_ok=True,
        )

        raise HTTPException(
            status_code=400,
            detail=f"Failed to process document: {exc}",
        )

    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "size": len(content),
        "chunk_count": chunk_count,
        "indexed": True,
        "company": company,
        "ticker": ticker,
        "fiscal_year": fiscal_year,
        "document_type": document_type,
    }