from __future__ import annotations

from typing import Literal, Protocol, cast

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agent.schema import FINANCIAL_SCHEMA
from app.agent.sql_executor import SQLExecutionError, SQLExecutor
from app.agent.sql_generator import SQLGenerator
from app.agent.state import AgentState, Route, SQLTaskMode
from app.analysis.financial_analyzer import FinancialAnalyzer
from app.analysis.intent import (
    AnalysisSQLTask,
    DirectSQLTask,
    SQLAnalysisClassificationError,
    SQLAnalysisClassifier,
)
from app.analysis.planner import (
    AbsoluteChangePlan,
    AnalysisPlanner,
    AnalysisPlanningError,
    DifferencePlan,
    PercentageChangePlan,
    RankingPlan,
)
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
    *,
    sql_analysis_classifier: SQLAnalysisClassifier,
    analysis_planner: AnalysisPlanner,
    financial_analyzer: FinancialAnalyzer,
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

    def classify_sql_task(state: AgentState) -> dict[str, object]:
        decision = sql_analysis_classifier.classify(state["question"])
        if isinstance(decision, DirectSQLTask):
            return {"sql_task_mode": "direct"}
        if isinstance(decision, AnalysisSQLTask):
            return {
                "sql_task_mode": "analysis",
                "analysis_operation": decision.operation,
            }
        raise SQLAnalysisClassificationError("Unexpected SQL task decision.")

    def select_sql_mode(state: AgentState) -> SQLTaskMode:
        return state["sql_task_mode"]

    def generate_analysis_data(state: AgentState) -> dict[str, object]:
        generated_sql = sql_generator.generate_analysis_data(
            question=state["question"],
            schema=FINANCIAL_SCHEMA,
            operation=state["analysis_operation"],
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
    ) -> Literal[
        "direct_success", "analysis_success", "repair", "analysis_repair", "failure"
    ]:
        if state.get("sql_error") is None:
            return (
                "analysis_success"
                if state["sql_task_mode"] == "analysis"
                else "direct_success"
            )
        if state["sql_retry_count"] < max_sql_retries:
            return (
                "analysis_repair" if state["sql_task_mode"] == "analysis" else "repair"
            )
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

    def repair_analysis_sql(state: AgentState) -> dict[str, object]:
        error_message = state["sql_error"]
        if error_message is None:
            raise RuntimeError("SQL repair requires an execution error.")
        repaired_sql = sql_generator.repair_analysis_data(
            question=state["question"],
            schema=FINANCIAL_SCHEMA,
            operation=state["analysis_operation"],
            previous_sql=state["generated_sql"],
            error_message=error_message,
        )
        return {
            "generated_sql": repaired_sql,
            "sql_retry_count": state["sql_retry_count"] + 1,
            "sql_error": None,
        }

    def plan_analysis(state: AgentState) -> dict[str, object]:
        plan = analysis_planner.plan(
            question=state["question"],
            sql_result=state["sql_result"],
        )
        if not isinstance(plan, (
            PercentageChangePlan, AbsoluteChangePlan, DifferencePlan, RankingPlan
        )):
            raise AnalysisPlanningError("Unexpected analysis plan type.")
        if plan.operation != state["analysis_operation"]:
            raise AnalysisPlanningError(
                "Analysis plan does not match SQL task operation."
            )
        return {"analysis_plan": plan}

    def execute_analysis(state: AgentState) -> dict[str, object]:
        plan = state["analysis_plan"]
        sql_result = state["sql_result"]
        if isinstance(plan, PercentageChangePlan):
            result = financial_analyzer.percentage_change(
                sql_result,
                old_row=plan.old_row,
                old_column=plan.old_column,
                new_row=plan.new_row,
                new_column=plan.new_column,
            )
        elif isinstance(plan, AbsoluteChangePlan):
            result = financial_analyzer.absolute_change(
                sql_result,
                old_row=plan.old_row,
                old_column=plan.old_column,
                new_row=plan.new_row,
                new_column=plan.new_column,
            )
        elif isinstance(plan, DifferencePlan):
            result = financial_analyzer.difference(
                sql_result,
                left_row=plan.left_row,
                left_column=plan.left_column,
                right_row=plan.right_row,
                right_column=plan.right_column,
            )
        elif isinstance(plan, RankingPlan):
            result = financial_analyzer.ranking(
                sql_result, column=plan.column, direction=plan.direction,
            )
        else:
            raise AnalysisPlanningError("Unexpected analysis plan type.")
        return {"analysis_result": result}

    graph = StateGraph(AgentState)

    graph.add_node("route_question", route_question)
    graph.add_node("rag_node", rag_node)
    graph.add_node("generate_sql", generate_sql)
    graph.add_node("execute_sql", execute_sql)
    graph.add_node("repair_sql", repair_sql)
    graph.add_node("sql_failure", sql_failure)
    graph.add_node("classify_sql_task", classify_sql_task)
    graph.add_node("generate_analysis_data", generate_analysis_data)
    graph.add_node("repair_analysis_sql", repair_analysis_sql)
    graph.add_node("plan_analysis", plan_analysis)
    graph.add_node("execute_analysis", execute_analysis)

    graph.add_edge(START, "route_question")
    graph.add_conditional_edges(
        "route_question",
        select_route,
        {
            "rag": "rag_node",
            "sql": "classify_sql_task",
        },
    )
    graph.add_edge("rag_node", END)
    graph.add_conditional_edges(
        "classify_sql_task",
        select_sql_mode,
        {"direct": "generate_sql", "analysis": "generate_analysis_data"},
    )
    graph.add_edge("generate_sql", "execute_sql")
    graph.add_edge("generate_analysis_data", "execute_sql")
    graph.add_conditional_edges(
        "execute_sql",
        select_sql_outcome,
        {
            "direct_success": END,
            "analysis_success": "plan_analysis",
            "repair": "repair_sql",
            "analysis_repair": "repair_analysis_sql",
            "failure": "sql_failure",
        },
    )
    graph.add_edge("repair_sql", "execute_sql")
    graph.add_edge("repair_analysis_sql", "execute_sql")
    graph.add_edge("sql_failure", END)
    graph.add_edge("plan_analysis", "execute_analysis")
    graph.add_edge("execute_analysis", END)

    return graph.compile().with_config({
        "recursion_limit": max(25, 2 * max_sql_retries + 9),
    })
