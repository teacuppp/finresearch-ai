import re

from rank_bm25 import BM25Okapi

from app.rag.models import (
    DocumentChunk,
    RetrievedChunk,
)
from app.rag.vector_store import VectorStore


_TOKEN_PATTERN = re.compile(
    r"[A-Za-z0-9]+"
)


def _tokenize(
    text: str,
) -> list[str]:
    return [
        match.group(0).casefold()
        for match
        in _TOKEN_PATTERN.finditer(text)
    ]


class LexicalRetriever:
    def __init__(
        self,
        vector_store: VectorStore,
    ):
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        where: dict | None = None,
    ) -> list[RetrievedChunk]:
        chunks = self.vector_store.get_chunks(
            where=where
        )

        if not chunks:
            return []

        tokenized_corpus = [
            _tokenize(chunk.text)
            for chunk in chunks
        ]

        bm25 = BM25Okapi(
            tokenized_corpus
        )

        query_tokens = _tokenize(
            query
        )

        scores = bm25.get_scores(
            query_tokens
        )

        # if not any(score > 0 for score in scores):
        #     return []

        ranked = sorted(
            zip(
                chunks,
                scores,
            ),
            key=lambda item: item[1],
            reverse=True,
        )

        results = []

        for chunk, score in ranked[:top_k]:
            # if score <= 0:
            #     continue

            results.append(
                RetrievedChunk(
                    text=chunk.text,
                    document=chunk.document,
                    page=chunk.page,
                    chunk_index=chunk.chunk_index,
                    distance=-float(score),
                    company=chunk.company,
                    ticker=chunk.ticker,
                    fiscal_year=chunk.fiscal_year,
                    document_type=(
                        chunk.document_type
                    ),
                )
            )

        return results