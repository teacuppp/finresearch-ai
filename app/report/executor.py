"""Execute a report plan through the existing agent service."""

from typing import Protocol

from pydantic import ValidationError

from app.report.evidence import ReportEvidence, ReportResearchResult
from app.report.models import ReportPlan, ReportRAGScope
from app.services.agent_service import AgentResult


class ReportResearchExecutionError(RuntimeError):
    """A report task could not produce route-consistent evidence."""


class ReportAgent(Protocol):
    def ask(
        self,
        question: str,
        top_k: int = 5,
        where: dict | None = None,
        company: str | None = None,
        ticker: str | None = None,
    ) -> AgentResult: ...


def _scope_filter(scope: ReportRAGScope) -> dict:
    conditions = [
        {field: {"$eq": value}}
        for field in ("company", "ticker", "fiscal_year", "document_type")
        if (value := getattr(scope, field)) is not None
    ]
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


class ReportResearchExecutor:
    def __init__(self, agent_service: ReportAgent, top_k: int = 5) -> None:
        if type(top_k) is not int or not 1 <= top_k <= 20:
            raise ValueError("top_k must be an integer from 1 to 20")
        self.agent_service = agent_service
        self.top_k = top_k

    def execute(self, plan: ReportPlan) -> ReportResearchResult:
        if not isinstance(plan, ReportPlan):
            raise ReportResearchExecutionError("A ReportPlan is required.")
        try:
            ReportPlan.model_validate(plan, strict=True)
        except ValidationError as exc:
            raise ReportResearchExecutionError("Invalid report plan.") from exc

        evidence: list[ReportEvidence] = []
        for task in plan.tasks:
            try:
                if task.expected_route == "rag":
                    scope = task.rag_scope
                    assert scope is not None  # ReportTask validation requires RAG scope.
                    result = self.agent_service.ask(
                        question=task.question,
                        top_k=self.top_k,
                        where=_scope_filter(scope),
                        company=scope.company,
                        ticker=scope.ticker,
                    )
                else:
                    result = self.agent_service.ask(question=task.question)
            except Exception as exc:
                raise ReportResearchExecutionError(
                    f"Report research task {task.task_id!r} failed."
                ) from exc

            if result.route != task.expected_route:
                raise ReportResearchExecutionError(
                    f"Report research task {task.task_id!r} expected route "
                    f"{task.expected_route!r}, got {result.route!r}."
                )
            evidence.append(ReportEvidence(task=task, result=result))

        return ReportResearchResult(plan=plan, evidence=tuple(evidence))
