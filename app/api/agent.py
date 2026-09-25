from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator

from app.agent.graph import UnsupportedRouteError
from app.agent.router import QuestionRoutingError
from app.agent.sql_executor import SQLExecutionError, SQLValidationError
from app.agent.sql_generator import SQLGenerationError
from app.analysis.financial_analyzer import AnalysisError, AnalysisOperation
from app.analysis.intent import SQLAnalysisClassificationError
from app.analysis.planner import AnalysisPlanningError
from app.api.filters import build_metadata_filter
from app.api.rag import AskRequest as RAGAskRequest, SourceResponse
from app.dependencies import get_agent_service
from app.rag.validation import AnswerValidationError
from app.services.agent_service import AgentService, InvalidAgentResultError
from app.services.query_service import AmbiguousQueryError


router = APIRouter(prefix="/agent", tags=["agent"])


class AgentAskRequest(RAGAskRequest):
    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be empty")
        return value


class RAGAgentResponse(BaseModel):
    route: Literal["rag"]
    answer: str
    sources: list[SourceResponse]


class SQLResultResponse(BaseModel):
    columns: list[str]
    rows: list[dict[str, Any]]
    row_count: int


class AnalysisResultResponse(BaseModel):
    operation: AnalysisOperation
    value: int | float | None
    ranked_rows: list[dict[str, Any]]


class SQLAgentResponse(BaseModel):
    route: Literal["sql"]
    generated_sql: str
    sql_result: SQLResultResponse
    analysis_result: AnalysisResultResponse | None = None


AgentAskResponse = Annotated[
    RAGAgentResponse | SQLAgentResponse,
    Field(discriminator="route"),
]


@router.post(
    "/ask", response_model=AgentAskResponse, response_model_exclude_unset=True
)
def ask_question(
    request: AgentAskRequest,
    agent_service: Annotated[AgentService, Depends(get_agent_service)],
) -> AgentAskResponse:
    where = build_metadata_filter(
        company=request.company,
        ticker=request.ticker,
        fiscal_year=request.fiscal_year,
        document_type=request.document_type,
    )

    try:
        result = agent_service.ask(
            question=request.question,
            top_k=request.top_k,
            where=where,
            company=request.company,
            ticker=request.ticker,
        )
    except AmbiguousQueryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except AnswerValidationError as exc:
        raise HTTPException(
            status_code=502,
            detail="The language model produced an invalid source-grounded response.",
        ) from exc
    except (
        QuestionRoutingError,
        UnsupportedRouteError,
        SQLGenerationError,
        SQLExecutionError,
        SQLValidationError,
        InvalidAgentResultError,
        SQLAnalysisClassificationError,
        AnalysisPlanningError,
        AnalysisError,
    ) as exc:
        raise HTTPException(
            status_code=502,
            detail="The agent could not complete the request.",
        ) from exc

    if result.route == "rag":
        if result.answer is None:
            raise HTTPException(
                status_code=502,
                detail="The agent could not complete the request.",
            )
        return RAGAgentResponse(
            route="rag",
            answer=result.answer,
            sources=[
                SourceResponse.model_validate(source, from_attributes=True)
                for source in result.sources
            ],
        )

    if result.route != "sql" or result.generated_sql is None or result.sql_result is None:
        raise HTTPException(
            status_code=502,
            detail="The agent could not complete the request.",
        )
    response = SQLAgentResponse(
        route="sql",
        generated_sql=result.generated_sql,
        sql_result=SQLResultResponse(
            columns=result.sql_result.columns,
            rows=result.sql_result.rows,
            row_count=result.sql_result.row_count,
        ),
    )
    if result.analysis_result is not None:
        response.analysis_result = AnalysisResultResponse(
            operation=result.analysis_result.operation,
            value=result.analysis_result.value,
            ranked_rows=[dict(row) for row in result.analysis_result.ranked_rows],
        )
    return response
