from sentence_transformers import CrossEncoder

from app.rag.models import RetrievedChunk


class Reranker:
    def __init__(
        self,
        model_name: str = (
            "cross-encoder/"
            "ms-marco-MiniLM-L6-v2"
        ),
    ):
        self.model = CrossEncoder(
            model_name
        )

    def rerank(
        self,
        query: str,
        results: list[RetrievedChunk],
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        if not results:
            return []

        pairs = [
            (
                query,
                result.text,
            )
            for result in results
        ]

        scores = self.model.predict(
            pairs
        )

        ranked_results = [
            result
            for _, result in sorted(
                zip(
                    scores,
                    results,
                ),
                key=lambda item: item[0],
                reverse=True,
            )
        ]

        if top_k is not None:
            return ranked_results[:top_k]

        return ranked_results