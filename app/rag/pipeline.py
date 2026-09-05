from dataclasses import dataclass

from app.rag.context import build_context
from app.rag.generator import AnswerGenerator
from app.rag.models import RetrievedChunk
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
        generator: AnswerGenerator,
    ):
        self.retriever = retriever
        self.generator = generator

    def ask(
        self,
        question: str,
        top_k: int = 5,
        where: dict | None = None,
    ) -> RAGResult:
        chunks = self.retriever.retrieve(
            query=question,
            top_k=top_k,
            where=where,
)
        context = build_context(chunks)

        answer = self.generator.generate(
            question=question,
            context=context,
        )

        try:
            validate_answer(answer)
        except AnswerValidationError:
            # Attempt to repair the answer
            answer = self.generator.repair(
                question=question,
                context=context,
                previous_answer=answer,
            )
            validate_answer(answer)

        return RAGResult(
            answer=answer,
            sources=chunks,
        )