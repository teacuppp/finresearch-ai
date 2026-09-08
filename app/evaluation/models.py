from dataclasses import dataclass


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