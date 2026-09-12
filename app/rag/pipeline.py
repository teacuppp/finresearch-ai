from dataclasses import dataclass

from app.rag.context import build_context
from app.rag.generator import AnswerGenerator
from app.rag.models import RetrievedChunk
from app.rag.reranker import Reranker
from app.rag.retriever import Retriever
from app.rag.validation import (
    AnswerValidationError,
    validate_answer,
)


@dataclass
class RAGResult:
    answer: str
    sources: list[RetrievedChunk]


class RAGPipeline:
    def __init__(
        self,
        retriever: Retriever,
        reranker: Reranker,
        generator: AnswerGenerator,
        retrieval_depth: int = 20,
    ):
        if retrieval_depth <= 0:
            raise ValueError(
                "retrieval_depth must be positive"
            )

        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator
        self.retrieval_depth = retrieval_depth

    def ask(
        self,
        question: str,
        top_k: int = 5,
        where: dict | None = None,
    ) -> RAGResult:
        if top_k <= 0:
            raise ValueError(
                "top_k must be positive"
            )

        candidates = self.retriever.retrieve(
            query=question,
            top_k=self.retrieval_depth,
            where=where,
        )

        chunks = self.reranker.rerank(
            query=question,
            results=candidates,
            top_k=top_k,
        )

        context = build_context(
            chunks
        )

        answer = self.generator.generate(
            question=question,
            context=context,
        )

        try:
            validate_answer(
                answer
            )
        except AnswerValidationError:
            answer = (
                self.generator.repair(
                    question=question,
                    context=context,
                    previous_answer=answer,
                )
            )

            validate_answer(
                answer
            )

        return RAGResult(
            answer=answer,
            sources=chunks,
        )