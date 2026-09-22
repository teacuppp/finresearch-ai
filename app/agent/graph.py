from __future__ import annotations

from typing import Protocol, cast

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agent.schema import FINANCIAL_SCHEMA
from app.agent.sql_executor import SQLExecutor
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
) -> CompiledStateGraph:
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

    def sql_node(state: AgentState) -> dict[str, object]:
        generated_sql = sql_generator.generate(
            question=state["question"],
            schema=FINANCIAL_SCHEMA,
        )
        sql_result = sql_executor.execute(generated_sql)

        return {
            "generated_sql": generated_sql,
            "sql_result": sql_result,
        }

    graph = StateGraph(AgentState)

    graph.add_node("route_question", route_question)
    graph.add_node("rag_node", rag_node)
    graph.add_node("sql_node", sql_node)

    graph.add_edge(START, "route_question")
    graph.add_conditional_edges(
        "route_question",
        select_route,
        {
            "rag": "rag_node",
            "sql": "sql_node",
        },
    )
    graph.add_edge("rag_node", END)
    graph.add_edge("sql_node", END)

    return graph.compile()
