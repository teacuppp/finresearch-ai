"""Compose report planning and research execution."""

from typing import Protocol

from app.report.evidence import ReportResearchResult
from app.report.models import ReportPlan


class ReportPlannerPort(Protocol):
    def plan(self, request: str) -> ReportPlan: ...


class ReportResearchExecutorPort(Protocol):
    def execute(self, plan: ReportPlan) -> ReportResearchResult: ...


class ReportService:
    def __init__(
        self,
        planner: ReportPlannerPort,
        executor: ReportResearchExecutorPort,
    ) -> None:
        self.planner = planner
        self.executor = executor

    def research(self, request: str) -> ReportResearchResult:
        plan = self.planner.plan(request)
        return self.executor.execute(plan)
