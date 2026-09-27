"""Structured evidence-gathering plan for one research request."""

import re

from openai import OpenAI

from app.report.models import ReportPlan, ReportRAGScope, ReportTask


REPORT_PLANNER_SYSTEM_PROMPT = """
Convert the user's research request into a small, minimal set of independent
evidence-gathering questions. Each task must be independently answerable by the
existing AgentService. Return only a structured ReportPlan with a concise title
and tasks. Do not answer the request or any task.

The existing Agent has exactly two top-level routes:
- rag: narrative or qualitative facts from filings and documents, including
  business risks, management discussion, strategy, and explanatory drivers.
- sql: structured financial metrics, comparisons, aggregations, numeric
  calculations, rankings, and trends from the financial database.

The expected_route is an expectation for validation and diagnostics, not a
command that bypasses the existing Agent router. A request needing both
qualitative filing evidence and quantitative financial data may require both
rag and sql tasks.

SQL decomposition:
- NEVER create one task per entity × metric cell. For ordinary direct metric
  lookups and comparisons, combine entities and metrics when they share the
  same context. When the same entities and year share multiple requested
  metrics, prefer one comparative SQL task if those metrics can be retrieved
  together. Keep all relevant entities in the same comparison question for
  direct SQL.
- Create separate SQL tasks only for genuinely distinct analytical operations
  or when an operation's execution cardinality requires it; a ranking and a
  comparison of a different metric may be separate tasks.
- BAD: "Apple 2025 revenue" and "Microsoft 2025 revenue" as separate tasks.
  GOOD: "Retrieve Apple and Microsoft 2025 revenue values."
  GOOD: "Retrieve Apple and Microsoft 2025 revenue, net income, and gross margin
  values." This remains ONE direct SQL evidence task.
- For ordinary historical metric retrieval or comparison, ask for underlying
  values, not a calculated difference. BAD: "Compare Apple and Microsoft 2025
  operating income." GOOD: "Retrieve Apple and Microsoft 2025 operating income
  values." Keep explicit absolute change, percentage change, difference, and
  ranking requests in their operation-specific wording.
- Explicit deterministic operation precedence: if the original request asks
  for absolute change, percentage change, difference, or ranking, preserve
  that operation in its evidence task. The ordinary retrieve-underlying-values
  rule applies ONLY to ordinary lookup or comparison, never to an explicitly
  requested deterministic operation.
  Difference request: "What is the difference between Apple and Microsoft 2025
  revenue?" GOOD: "What is the difference between Apple and Microsoft 2025
  revenue?" BAD: "Retrieve Apple and Microsoft 2025 revenue values."
  Ranking request: "Rank Apple and Microsoft by 2025 revenue." GOOD: "Rank
  Apple and Microsoft by 2025 revenue from highest to lowest." BAD: "Retrieve
  Apple and Microsoft 2025 revenue values."
- A task must request at most one supported deterministic analysis operation
  and be independently executable by the current AgentService, not merely
  routeable to SQL.
- For absolute_change and percentage_change, use exactly one entity, exactly
  two fiscal years, and exactly one metric per task. For multi-company change
  requests, create one task per entity.
  BAD: "Compare Apple and Microsoft revenue growth from 2024 to 2025."
  GOOD: "By what percentage did Apple's revenue change from 2024 to 2025?"
  GOOD: "By what percentage did Microsoft's revenue change from 2024 to 2025?"
- For difference, use exactly two entities, exactly one fiscal year, and
  exactly one metric per task.
- For ranking, use at least two named entities (or all entities), exactly one
  fiscal year, and exactly one metric per task.
- If a request combines a ranking with an unrelated metric comparison, split
  them into separate SQL tasks. For example, "Rank Apple and Microsoft by 2025
  revenue and compare 2025 operating income" becomes:
  1. "Rank Apple and Microsoft by 2025 revenue from highest to lowest."
  2. "Retrieve Apple and Microsoft 2025 operating income values."
- A deterministic analysis task already retrieves and retains its required raw
  SQL evidence. Do NOT add a redundant direct SQL raw-value task for the same
  entity, year, and metric already consumed by absolute_change,
  percentage_change, difference, or ranking. In the ranking example above,
  the ranking task already provides raw revenue evidence; do not add a third
  task to retrieve Apple and Microsoft 2025 revenue values.
- For structured SQL evidence tasks, use EXACTLY the structured metrics
  explicitly requested by the user. Never add another financial metric for
  completeness, merely because it is available in the database, or because
  it appears in an example elsewhere in this system prompt. Do not expand
  revenue + net income into revenue + net income + gross margin.
  Request: "Apple FY2025 revenue, net income, and business risks"
  GOOD SQL task: "Retrieve Apple's FY2025 revenue and net income values."
  BAD SQL task: "Retrieve Apple's FY2025 revenue, net income, and gross margin
  values."

RAG scope:
- Every RAG task must contain rag_scope identifying one company or ticker.
  Scope a single RAG task to ONE company/ticker. Never combine narrative
  evidence for multiple companies in one RAG task. For a multi-company
  narrative request, create one RAG task per company.
- Copy company, ticker, fiscal year, and document type into rag_scope only
  when supported by the original request. Do not invent ticker symbols or
  metadata not supplied by the user.
- When multiple qualitative topics share the same company/ticker, fiscal year,
  and document scope, prefer one combined RAG evidence task if it remains
  independently answerable. For example, ask one Apple 2025 filing question
  about competition, regulation, and supply-chain risks with company="Apple",
  fiscal_year=2025, ticker=None, and document_type=None.
- rag_scope.document_type is either 10-K, 10-Q, 8-K, 20-F, or 40-F explicitly
  named by the user, or None. Generic "filing", "annual filing", "report",
  "annual report", and "forward-looking statements" require
  document_type=None. Do NOT copy "filing" into document_type.

Semantic preservation:
- Preserve whether the request is historical, current, or forecast-oriented.
  NEVER introduce forecast, forecasted, projected, estimated, or expected
  semantics unless the original request explicitly asks for forecasting.
- Do not alter requested fiscal years. Do not invent entities, years, metrics,
  or analysis objectives. Preserve metric names and company names.
- Every company or ticker named in any generated task must come from the
  original request. Never infer comparison peers. Never add another company
  merely because a comparison would be useful. Never infer a ticker from a
  company name.
- Historical observed metrics in the structured financial database are SQL
  evidence. Forward-looking forecasts or projections from company filings
  are narrative document evidence and may be RAG tasks.
- For a RAG forecast task, rag_scope.fiscal_year is the SOURCE DOCUMENT fiscal
  year, not the target forecast year. Do not set it to the target year unless
  the request explicitly identifies that as the source document year. If no
  source filing year is specified, leave rag_scope.fiscal_year=None. For a
  request about projected Apple and Microsoft revenue for 2026 with no source
  year, use one RAG task for each company and no fiscal_year, ticker, or
  document_type scope.

Do not generate SQL, calculate financial values, invent evidence, write report
prose, or generate chart specifications. Avoid duplicate or overlapping tasks.
Preserve company names, ticker symbols, fiscal years, and metric names from the
request. Do not add tasks unrelated to the requested report.
""".strip()


class ReportPlanningError(RuntimeError):
    """A valid structured report plan could not be produced."""


_YEAR_PATTERN = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
_FILING_FORM_PATTERN = re.compile(
    r"(?<!\w)(?:10-K|10-Q|8-K|20-F|40-F)(?!\w)", re.IGNORECASE
)
_FORECAST_PATTERN = re.compile(
    r"\b(?:forecasts?|forecasted|forecasting|projected|projections?|"
    r"estimates?|estimated|estimating|expected)\b",
    re.IGNORECASE,
)
_METRIC_PATTERNS = {
    "revenue": re.compile(r"(?<!\w)revenue(?:_musd)?(?!\w)", re.IGNORECASE),
    "net income": re.compile(
        r"(?<!\w)net[\s_-]+income(?:_musd)?(?!\w)", re.IGNORECASE
    ),
    "operating income": re.compile(
        r"(?<!\w)operating[\s_-]+income(?:_musd)?(?!\w)", re.IGNORECASE
    ),
    "gross margin": re.compile(r"(?<!\w)gross[\s_-]+margin(?!\w)", re.IGNORECASE),
}


def _metric_concepts(text: str) -> set[str]:
    return {
        metric for metric, pattern in _METRIC_PATTERNS.items() if pattern.search(text)
    }


def _canonicalize_optional_rag_scope(plan: ReportPlan, request: str) -> ReportPlan:
    request_years = {int(year) for year in _YEAR_PATTERN.findall(request)}
    request_forms = {form.upper() for form in _FILING_FORM_PATTERN.findall(request)}
    canonical_tasks: list[ReportTask] = []
    changed = False

    for task in plan.tasks:
        scope = task.rag_scope
        if scope is None:
            canonical_tasks.append(task)
            continue
        if scope.company is not None and not _contains_explicit_name(
            request, scope.company
        ):
            raise ValueError("RAG company was not present in the original request.")
        if scope.fiscal_year is not None and scope.fiscal_year not in request_years:
            raise ValueError("RAG fiscal year was not present in the original request.")

        ticker = scope.ticker
        if ticker is not None and not _contains_explicit_name(request, ticker):
            ticker = None
        document_type = scope.document_type
        if document_type is not None and document_type.upper() not in request_forms:
            document_type = None

        if ticker != scope.ticker or document_type != scope.document_type:
            canonical_scope = ReportRAGScope.model_validate(
                {**scope.model_dump(), "ticker": ticker, "document_type": document_type}
            )
            canonical_tasks.append(
                ReportTask.model_validate(
                    {**task.model_dump(), "rag_scope": canonical_scope}
                )
            )
            changed = True
        else:
            canonical_tasks.append(task)

    if not changed:
        return plan
    return ReportPlan.model_validate({"title": plan.title, "tasks": tuple(canonical_tasks)})


def _validated_plan_against_request(plan: ReportPlan, request: str) -> ReportPlan:
    request_years = {int(year) for year in _YEAR_PATTERN.findall(request)}
    generated_years = {int(year) for year in _YEAR_PATTERN.findall(plan.title)}
    request_forms = {form.upper() for form in _FILING_FORM_PATTERN.findall(request)}
    request_metrics = _metric_concepts(request)

    for task in plan.tasks:
        generated_years.update(
            int(year) for year in _YEAR_PATTERN.findall(task.question)
        )
        if task.expected_route == "sql" and not _metric_concepts(
            task.question
        ).issubset(request_metrics):
            raise ValueError("SQL task introduced a metric absent from the request.")
        scope = task.rag_scope
        if scope is None:
            continue
        if scope.fiscal_year is not None:
            generated_years.add(scope.fiscal_year)
        if scope.company is not None and not _contains_explicit_name(
            request, scope.company
        ):
            raise ValueError("RAG company was not present in the original request.")
        if scope.ticker is not None and not _contains_explicit_name(
            request, scope.ticker
        ):
            raise ValueError("RAG ticker was not present in the original request.")
        if (
            scope.document_type is not None
            and scope.document_type.strip().upper() not in request_forms
        ):
            raise ValueError("RAG document type was not present in the original request.")

    if not generated_years.issubset(request_years):
        raise ValueError("Plan introduced a year absent from the original request.")
    if _FORECAST_PATTERN.search(request) is None:
        if _FORECAST_PATTERN.search(plan.title) or any(
            _FORECAST_PATTERN.search(task.question) for task in plan.tasks
        ):
            raise ValueError(
                "Plan introduced forecast semantics absent from the request."
            )
    return plan


def _contains_explicit_name(request: str, name: str) -> bool:
    pattern = rf"(?<!\w){re.escape(name.strip())}(?!\w)"
    return re.search(pattern, request, re.IGNORECASE) is not None


class ReportPlanner:
    def __init__(
        self,
        model: str = "qwen3:4b-instruct",
        base_url: str = "http://localhost:11434/v1/",
        api_key: str = "ollama",
        client: OpenAI | None = None,
    ) -> None:
        self.model = model
        self.client = (
            client if client is not None else OpenAI(base_url=base_url, api_key=api_key)
        )

    def plan(self, request: str) -> ReportPlan:
        if not isinstance(request, str):
            raise ReportPlanningError("request must be a string")
        if not request.strip():
            raise ReportPlanningError("request must not be blank")
        if len(request) > 2000:
            raise ReportPlanningError("request must be at most 2000 characters")

        try:
            response = self.client.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": REPORT_PLANNER_SYSTEM_PROMPT},
                    {"role": "user", "content": request},
                ],
                response_format=ReportPlan,
                temperature=0,
            )
            parsed = response.choices[0].message.parsed
            if not isinstance(parsed, ReportPlan):
                raise TypeError("model returned no valid ReportPlan")
            ReportPlan.model_validate(parsed)
            canonical = _canonicalize_optional_rag_scope(parsed, request)
            return _validated_plan_against_request(canonical, request)
        except Exception as exc:
            raise ReportPlanningError("Report planning failed.") from exc
