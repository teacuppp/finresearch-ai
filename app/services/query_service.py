from app.rag.pipeline import RAGPipeline, RAGResult
from app.rag.vector_store import VectorStore


class AmbiguousQueryError(Exception):
    pass


class QueryService:
    def __init__(
        self,
        rag_pipeline: RAGPipeline,
        vector_store: VectorStore,
    ):
        self.rag_pipeline = rag_pipeline
        self.vector_store = vector_store

    def ask(
        self,
        question: str,
        top_k: int = 5,
        where: dict | None = None,
        company: str | None = None,
        ticker: str | None = None,
    ) -> RAGResult:
        if company is None and ticker is None:
            tickers = self.vector_store.list_tickers()

            if len(tickers) > 1:
                raise AmbiguousQueryError(
                    "The query is ambiguous. "
                    "Please specify a company or ticker."
                )

        return self.rag_pipeline.ask(
            question=question,
            top_k=top_k,
            where=where,
        )