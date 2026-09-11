from dataclasses import replace

from app.rag.lexical_retriever import (
    LexicalRetriever,
)
from app.rag.models import RetrievedChunk
from app.rag.retriever import Retriever


DEFAULT_RRF_K = 60


def _chunk_key(
    chunk: RetrievedChunk,
) -> tuple[str, int, int]:
    return (
        chunk.document,
        chunk.page,
        chunk.chunk_index,
    )


class HybridRetriever:
    def __init__(
    self,
    dense_retriever: Retriever,
    lexical_retriever: LexicalRetriever,
    rrf_k: int = DEFAULT_RRF_K,
    dense_weight: float = 1.0,
    lexical_weight: float = 1.0,
    ):
        if rrf_k <= 0:
            raise ValueError(
                "rrf_k must be positive"
            )

        if dense_weight <= 0:
            raise ValueError(
                "dense_weight must be positive"
            )

        if lexical_weight <= 0:
            raise ValueError(
                "lexical_weight must be positive"
            )

        self.dense_retriever = (
            dense_retriever
        )

        self.lexical_retriever = (
            lexical_retriever
        )

        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.lexical_weight = lexical_weight

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 20,
        where: dict | None = None,
    ) -> list[RetrievedChunk]:
        if top_k <= 0:
            return []

        if candidate_k <= 0:
            return []

        dense_results = (
            self.dense_retriever.retrieve(
                query=query,
                top_k=candidate_k,
                where=where,
            )
        )

        lexical_results = (
            self.lexical_retriever.retrieve(
                query=query,
                top_k=candidate_k,
                where=where,
            )
        )

        scores = {}
        chunks = {}

        for rank, chunk in enumerate(
            dense_results,
            start=1,
        ):
            key = _chunk_key(chunk)

            chunks[key] = chunk

            scores[key] = (
                scores.get(key, 0.0)
                + self.dense_weight
                / (
                    self.rrf_k
                    + rank
                )
            )

        for rank, chunk in enumerate(
            lexical_results,
            start=1,
        ):
            key = _chunk_key(chunk)

            chunks.setdefault(
                key,
                chunk,
            )

            scores[key] = (
                scores.get(key, 0.0)
                + self.lexical_weight
                / (
                    self.rrf_k
                    + rank
                )
            )

        ranked_keys = sorted(
            scores,
            key=scores.get,
            reverse=True,
        )

        results = []

        for key in ranked_keys[:top_k]:
            chunk = chunks[key]

            results.append(
                replace(
                    chunk,
                    distance=-scores[key],
                )
            )

        return results