from dataclasses import dataclass

from app.agent.state import Route


@dataclass
class RelevantSource:
    document: str
    page: int
    row_terms: list[str]
    preceding_terms: list[str]
    preceding_line_window: int = 8


@dataclass
class RetrievalExample:
    id: str
    question: str
    ticker: str
    fiscal_year: int
    relevant_sources: list[RelevantSource]


@dataclass(frozen=True)
class RoutingBenchmarkExample:
    id: str
    question: str
    expected_route: Route
    category: str | None = None


@dataclass(frozen=True)
class RoutingMetrics:
    total_examples: int
    correct_examples: int
    accuracy: float
    sql_precision: float
    sql_recall: float
    rag_precision: float
    rag_recall: float
