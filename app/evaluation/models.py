from dataclasses import dataclass

@dataclass
class RelevantSource:
    document: str
    page: int
    chunk_index: int


@dataclass
class RetrievalExample:
    id: str
    question: str
    ticker: str
    fiscal_year: int
    relevant_sources: list[RelevantSource]