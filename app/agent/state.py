from typing import Literal, NotRequired, TypedDict

from app.agent.sql_executor import SQLQueryResult
from app.agent.sql_generator import SQLResultMode
from app.analysis.chart_decision import ChartDecision, ChartIntentResponse
from app.analysis.chart_renderer import ChartArtifact, ChartSpec
from app.analysis.financial_analyzer import AnalysisOperation, AnalysisResult
from app.analysis.planner import AnalysisPlan
from app.rag.models import RetrievedChunk


Route = Literal["rag", "sql"]
SQLTaskMode = Literal["direct", "analysis"]


class AgentState(TypedDict):
    question: str
    top_k: NotRequired[int]
    where: NotRequired[dict | None]
    company: NotRequired[str | None]
    ticker: NotRequired[str | None]
    route: NotRequired[Route]
    answer: NotRequired[str]
    sources: NotRequired[list[RetrievedChunk]]
    generated_sql: NotRequired[str]
    sql_result: NotRequired[SQLQueryResult]
    sql_error: NotRequired[str | None]
    sql_retry_count: NotRequired[int]
    sql_task_mode: NotRequired[SQLTaskMode]
    analysis_operation: NotRequired[AnalysisOperation]
    analysis_plan: NotRequired[AnalysisPlan]
    analysis_result: NotRequired[AnalysisResult]
    chart_intent: NotRequired[ChartIntentResponse]
    explicit_visualization: NotRequired[bool]
    sql_result_mode: NotRequired[SQLResultMode]
    chart_data: NotRequired[SQLQueryResult | None]
    chart_decision: NotRequired[ChartDecision]
    chart_spec: NotRequired[ChartSpec]
    chart_artifact: NotRequired[ChartArtifact]
