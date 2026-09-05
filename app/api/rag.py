from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
)
from app.rag.validation import AnswerValidationError
from pydantic import BaseModel, Field

from app.dependencies import get_rag_pipeline
from app.rag.pipeline import RAGPipeline
from app.dependencies import get_query_service
from app.services.query_service import (
    AmbiguousQueryError,
    QueryService,
)


from app.api.filters import build_metadata_filter



router = APIRouter(
    prefix="/rag",
    tags=["rag"],
)


class AskRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=1000,
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
    )

    company: str | None = None
    ticker: str | None = None
    fiscal_year: int | None = None
    document_type: str | None = None


class SourceResponse(BaseModel):
    document: str
    page: int
    chunk_index: int
    distance: float

    company: str | None = None
    ticker: str | None = None
    fiscal_year: int | None = None
    document_type: str | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]


@router.post(
    "/ask",
    response_model=AskResponse,
)
def ask_question(
    request: AskRequest,
    query_service: Annotated[
        QueryService,
        Depends(get_query_service),
    ],
):
    where = build_metadata_filter(
        company=request.company,
        ticker=request.ticker,
        fiscal_year=request.fiscal_year,
        document_type=request.document_type,
    )

    try:
        result = query_service.ask(
            question=request.question,
            top_k=request.top_k,
            where=where,
            company=request.company,
            ticker=request.ticker,
        )

    except AmbiguousQueryError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except AnswerValidationError as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                "The language model produced an invalid "
                "source-grounded response."
            ),
        ) from exc

    return AskResponse(
        answer=result.answer,
        sources=[
            SourceResponse(
                document=source.document,
                page=source.page,
                chunk_index=source.chunk_index,
                distance=source.distance,
                company=source.company,
                ticker=source.ticker,
                fiscal_year=source.fiscal_year,
                document_type=source.document_type,
            )
            for source in result.sources
        ],
    )