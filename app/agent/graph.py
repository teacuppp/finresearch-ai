from __future__ import annotations

from typing import Literal, Protocol, cast

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agent.schema import FINANCIAL_SCHEMA
from app.agent.sql_executor import SQLExecutionError, SQLExecutor
from app.agent.sql_generator import SQLGenerator
from app.agent.state import AgentState, Route
from app.services.query_service import QueryService


class UnsupportedRouteError(ValueError):
    pass


class QuestionRouter(Protocol):
    def route(self, question: str) -> str: ...


def _validated_route(route: str | None) -> Route:
    if route not in {"rag", "sql"}:
        raise UnsupportedRouteError(
            f"Unsupported route: {route!r}. Expected 'rag' or 'sql'."
        )

    return cast(Route, route)


def build_agent_graph(
    router: QuestionRouter,
    query_service: QueryService,
    sql_generator: SQLGenerator,
    sql_executor: SQLExecutor,
    max_sql_retries: int = 2,
) -> CompiledStateGraph:
    if max_sql_retries < 0:
        raise ValueError("max_sql_retries must be non-negative")

    def route_question(state: AgentState) -> dict[str, Route]:
        route = router.route(state["question"])
        return {"route": _validated_route(route)}

    def select_route(state: AgentState) -> Route:
        return _validated_route(state.get("route"))

    def rag_node(state: AgentState) -> dict[str, object]:
        result = query_service.ask(
            question=state["question"],
            top_k=state.get("top_k", 5),
            where=state.get("where"),
            company=state.get("company"),
            ticker=state.get("ticker"),
        )

        return {
            "answer": result.answer,
            "sources": result.sources,
        }

    def generate_sql(state: AgentState) -> dict[str, object]:
        generated_sql = sql_generator.generate(
            question=state["question"],
            schema=FINANCIAL_SCHEMA,
        )

        return {
            "generated_sql": generated_sql,
            "sql_retry_count": 0,
            "sql_error": None,
        }

    def execute_sql(state: AgentState) -> dict[str, object]:
        try:
            sql_result = sql_executor.execute(state["generated_sql"])
        except SQLExecutionError as exc:
            return {
                "sql_error": str(exc).strip() or "SQL execution failed.",
            }

        return {
            "sql_result": sql_result,
            "sql_error": None,
        }

    def select_sql_outcome(
        state: AgentState,
    ) -> Literal["success", "repair", "failure"]:
        if state.get("sql_error") is None:
            return "success"
        if state["sql_retry_count"] < max_sql_retries:
            return "repair"
        return "failure"

    def repair_sql(state: AgentState) -> dict[str, object]:
        error_message = state["sql_error"]
        if error_message is None:
            raise RuntimeError("SQL repair requires an execution error.")

        repaired_sql = sql_generator.repair(
            question=state["question"],
            schema=FINANCIAL_SCHEMA,
            previous_sql=state["generated_sql"],
            error_message=error_message,
        )
        return {
            "generated_sql": repaired_sql,
            "sql_retry_count": state["sql_retry_count"] + 1,
            "sql_error": None,
        }

    def sql_failure(state: AgentState) -> None:
        raise SQLExecutionError(state.get("sql_error") or "SQL execution failed.")

    graph = StateGraph(AgentState)

    graph.add_node("route_question", route_question)
    graph.add_node("rag_node", rag_node)
    graph.add_node("generate_sql", generate_sql)
    graph.add_node("execute_sql", execute_sql)
    graph.add_node("repair_sql", repair_sql)
    graph.add_node("sql_failure", sql_failure)

    graph.add_edge(START, "route_question")
    graph.add_conditional_edges(
        "route_question",
        select_route,
        {
            "rag": "rag_node",
            "sql": "generate_sql",
        },
    )
    graph.add_edge("rag_node", END)
    graph.add_edge("generate_sql", "execute_sql")
    graph.add_conditional_edges(
        "execute_sql",
        select_sql_outcome,
        {
            "success": END,
            "repair": "repair_sql",
            "failure": "sql_failure",
        },
    )
    graph.add_edge("repair_sql", "execute_sql")
    graph.add_edge("sql_failure", END)

    return graph.compile().with_config({
        "recursion_limit": max(25, 2 * max_sql_retries + 6),
    })
