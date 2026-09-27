from dataclasses import FrozenInstanceError

import pytest

from app.report.evidence import ReportEvidence, ReportResearchResult
from app.report.models import ReportPlan, ReportTask
from app.services.agent_service import AgentResult


def test_report_evidence_retains_exact_objects_and_is_frozen() -> None:
    task = ReportTask(task_id="revenue", question="Retrieve revenue.", expected_route="sql")
    result = AgentResult(route="sql")

    evidence = ReportEvidence(task=task, result=result)

    assert evidence.task is task
    assert evidence.result is result
    with pytest.raises(FrozenInstanceError):
        evidence.result = AgentResult(route="rag")


def test_research_result_retains_plan_and_ordered_evidence_tuple() -> None:
    first = ReportTask(task_id="first", question="First?", expected_route="sql")
    second = ReportTask(task_id="second", question="Second?", expected_route="sql")
    plan = ReportPlan(title="Research", tasks=(first, second))
    first_evidence = ReportEvidence(task=first, result=AgentResult(route="sql"))
    second_evidence = ReportEvidence(task=second, result=AgentResult(route="sql"))

    research = ReportResearchResult(
        plan=plan, evidence=(first_evidence, second_evidence)
    )

    assert research.plan is plan
    assert isinstance(research.evidence, tuple)
    assert research.evidence[0] is first_evidence
    assert research.evidence[1] is second_evidence
    with pytest.raises(FrozenInstanceError):
        research.evidence = ()
