from __future__ import annotations

from openai import OpenAI
from pydantic import BaseModel, ConfigDict

from app.agent.schema import FINANCIAL_SCHEMA
from app.agent.state import Route


ROUTER_SYSTEM_PROMPT = f"""
You are the execution-path router for a financial research system.

Your only task is to classify the user's question as "sql" or "rag".
Do not answer the user's financial question. Return only the structured
route decision requested by the response schema.

Choose "sql" when the question can be fully answered from structured fields
in this database schema:

{FINANCIAL_SCHEMA}

This includes questions about company or ticker, fiscal year, revenue,
operating income, net income, or gross margin, as well as comparisons,
filters, ordering, and aggregations over those fields.

Choose "rag" when answering requires qualitative or narrative document text,
including risk disclosures, strategy, management commentary, competitive
factors, explanations or drivers, and qualitative descriptions from financial
filings. A question asking why or how a financial result changed requires
"rag" when the structured fields alone do not provide the explanation.

If both the document corpus and the structured schema could answer the
question, choose "sql" whenever the structured schema can fully answer it.

Examples:
- "What was Apple's revenue in 2025?" -> {{"route": "sql"}}
- "Compare Apple and Microsoft net income in 2025." -> {{"route": "sql"}}
- "Which company had the higher gross margin?" -> {{"route": "sql"}}
- "What risks did Apple disclose?" -> {{"route": "rag"}}
- "How does Microsoft describe its AI strategy?" -> {{"route": "rag"}}
- "What factors affected Apple's Services business?" -> {{"route": "rag"}}
""".strip()


class RouteDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    route: Route


class QuestionRoutingError(RuntimeError):
    pass


class LLMQuestionRouter:
    def __init__(
        self,
        model: str = "qwen3:4b",
        base_url: str = "http://localhost:11434/v1/",
        api_key: str = "ollama",
        client: OpenAI | None = None,
    ):
        self.model = model
        self.client = (
            client
            if client is not None
            else OpenAI(
                base_url=base_url,
                api_key=api_key,
            )
        )

    def route(self, question: str) -> Route:
        if not question.strip():
            raise ValueError("question must not be empty")

        try:
            response = self.client.chat.completions.parse(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": ROUTER_SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": question,
                    },
                ],
                response_format=RouteDecision,
                temperature=0,
                reasoning_effort="none",
            )
        except Exception as exc:
            raise QuestionRoutingError(
                "Question routing request failed."
            ) from exc

        try:
            decision = response.choices[0].message.parsed
        except (AttributeError, IndexError, KeyError, TypeError) as exc:
            raise QuestionRoutingError(
                "LLM returned an invalid structured routing response."
            ) from exc

        if decision is None:
            raise QuestionRoutingError(
                "LLM returned no parsed route decision."
            )

        if not isinstance(decision, RouteDecision):
            raise QuestionRoutingError(
                "LLM returned an invalid structured route decision."
            )

        try:
            route = decision.route
        except AttributeError as exc:
            raise QuestionRoutingError(
                "LLM returned an invalid structured route decision."
            ) from exc

        if route not in ("rag", "sql"):
            raise QuestionRoutingError(
                "LLM returned an invalid structured route decision."
            )

        return route
