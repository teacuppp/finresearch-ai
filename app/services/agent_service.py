from dataclasses import dataclass, field
from typing import Mapping, Protocol, cast

from app.agent.sql_executor import SQLQueryResult
from app.agent.state import AgentState, Route
from app.analysis.financial_analyzer import AnalysisOperation, AnalysisResult
from app.rag.models import RetrievedChunk


class AgentGraph(Protocol):
    def invoke(
        self,
        input: AgentState,
    ) -> Mapping[str, object]: ...


class InvalidAgentResultError(RuntimeError):
    pass


@dataclass(frozen=True)
class AgentResult:
    route: Route
    answer: str | None = None
    sources: list[RetrievedChunk] = field(
        default_factory=list
    )
    generated_sql: str | None = None
    sql_result: SQLQueryResult | None = None
    analysis_result: AnalysisResult | None = None


class AgentService:
    def __init__(
        self,
        graph: AgentGraph,
    ):
        self.graph = graph

    def ask(
        self,
        question: str,
        top_k: int = 5,
        where: dict | None = None,
        company: str | None = None,
        ticker: str | None = None,
    ) -> AgentResult:
        if not question.strip():
            raise ValueError(
                "question must not be empty"
            )

        initial_state: AgentState = {
            "question": question,
            "top_k": top_k,
            "where": where,
            "company": company,
            "ticker": ticker,
        }
        result = self.graph.invoke(
            initial_state
        )
        route = result.get("route")

        if route == "rag":
            answer = result.get("answer")
            sources = result.get("sources")

            if not isinstance(answer, str):
                raise InvalidAgentResultError(
                    "RAG graph result is missing an answer."
                )

            if not isinstance(sources, list):
                raise InvalidAgentResultError(
                    "RAG graph result is missing sources."
                )

            return AgentResult(
                route="rag",
                answer=answer,
                sources=cast(
                    list[RetrievedChunk],
                    list(sources),
                ),
            )

        if route == "sql":
            generated_sql = result.get(
                "generated_sql"
            )
            sql_result = result.get(
                "sql_result"
            )

            if not isinstance(
                generated_sql,
                str,
            ):
                raise InvalidAgentResultError(
                    "SQL graph result is missing generated SQL."
                )

            if not isinstance(
                sql_result,
                SQLQueryResult,
            ):
                raise InvalidAgentResultError(
                    "SQL graph result is missing a query result."
                )

            analysis_result = result.get("analysis_result")
            sql_task_mode = result.get("sql_task_mode", "direct")
            if sql_task_mode == "analysis":
                operation = result.get("analysis_operation")
                if (
                    not isinstance(analysis_result, AnalysisResult)
                    or not isinstance(operation, AnalysisOperation)
                    or analysis_result.operation is not operation
                ):
                    raise InvalidAgentResultError(
                        "Analysis graph result is missing or inconsistent."
                    )
            elif sql_task_mode != "direct" or any(
                result.get(field) is not None
                for field in ("analysis_operation", "analysis_plan", "analysis_result")
            ):
                raise InvalidAgentResultError("SQL graph result has inconsistent mode.")

            return AgentResult(
                route="sql",
                generated_sql=generated_sql,
                sql_result=sql_result,
                analysis_result=cast(AnalysisResult | None, analysis_result),
            )

        raise InvalidAgentResultError(
            f"Agent graph returned an unsupported route: {route!r}."
        )
