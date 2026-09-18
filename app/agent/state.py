from typing import Literal, NotRequired, TypedDict

from app.agent.sql_executor import SQLQueryResult
from app.rag.models import RetrievedChunk


Route = Literal["rag", "sql"]


class AgentState(TypedDict):
    question: str
    route: NotRequired[Route]
    answer: NotRequired[str]
    sources: NotRequired[list[RetrievedChunk]]
    generated_sql: NotRequired[str]
    sql_result: NotRequired[SQLQueryResult]
