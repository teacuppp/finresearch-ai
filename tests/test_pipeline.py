import pytest

from app.rag.models import RetrievedChunk
from app.rag.pipeline import RAGPipeline


# class FakeRetriever:
#     def retrieve(
#         self,
#         query: str,
#         top_k: int = 5,
#     ) -> list[RetrievedChunk]:
#         return [
#             RetrievedChunk(
#                 text="Apple reported total net sales of $416.161 billion.",
#                 document="apple.pdf",
#                 page=35,
#                 chunk_index=0,
#                 distance=0.5,
#             )
#         ]


class FakeRetriever:
    def __init__(self):
        self.last_where = None

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        where: dict | None = None,
    ) -> list[RetrievedChunk]:
        self.last_where = where

        return [
            RetrievedChunk(
                text="Apple reported total net sales of $416.161 billion.",
                document="apple.pdf",
                page=35,
                chunk_index=0,
                distance=0.5,
                company="Apple",
                ticker="AAPL",
                fiscal_year=2025,
                document_type="10-K",
            )
        ]


class ValidGenerator:
    def generate(
        self,
        question: str,
        context: str,
    ) -> str:
        return (
            "Apple reported total net sales of "
            "$416.161 billion in 2025. [Source 1]"
        )


class InvalidGenerator:
    def generate(
        self,
        question: str,
        context: str,
    ) -> str:
        return (
            "Apple reported total net sales of "
            "$416.161 billion in 2025."
        )

    def repair(
            self,
            question: str,
            context: str,
            previous_answer: str,
        ) -> str:
            return (
                "Apple reported total net sales of "
                "$416.161 billion in 2025."
            )
    


def test_pipeline_returns_valid_answer():
    pipeline = RAGPipeline(
        retriever=FakeRetriever(),
        generator=ValidGenerator(),
    )

    result = pipeline.ask(
        "What was Apple's revenue?"
    )

    assert "[Source 1]" in result.answer
    assert len(result.sources) == 1
    assert result.sources[0].page == 35


def test_pipeline_rejects_answer_without_citation():
    pipeline = RAGPipeline(
        retriever=FakeRetriever(),
        generator=InvalidGenerator(),
    )

    with pytest.raises(AnswerValidationError):
        pipeline.ask(
            "What was Apple's revenue?"
        )

#测试 retry，而不是靠 Qwen 碰运气
class RepairableGenerator:
    def __init__(self):
        self.repair_called = False

    def generate(
        self,
        question: str,
        context: str,
    ) -> str:
        return (
            "Microsoft reported operating income "
            "of $128.528 billion."
        )

    def repair(
        self,
        question: str,
        context: str,
        previous_answer: str,
    ) -> str:
        self.repair_called = True

        return (
            "Microsoft reported operating income "
            "of $128.528 billion in fiscal year 2025. "
            "[Source 1]"
        )

def test_pipeline_repairs_missing_citation():
    generator = RepairableGenerator()

    pipeline = RAGPipeline(
        retriever=FakeRetriever(),
        generator=generator,
    )

    result = pipeline.ask(
        "What was Microsoft's operating income in 2025?"
    )

    assert generator.repair_called is True
    assert "[Source 1]" in result.answer



#测试 repair 也失败
class AlwaysInvalidGenerator:
    def generate(
        self,
        question: str,
        context: str,
    ) -> str:
        return "Microsoft operating income was $128.528 billion."

    def repair(
        self,
        question: str,
        context: str,
        previous_answer: str,
    ) -> str:
        return "Microsoft operating income was $128.528 billion."

from app.rag.validation import AnswerValidationError


def test_pipeline_rejects_failed_repair():
    pipeline = RAGPipeline(
        retriever=FakeRetriever(),
        generator=AlwaysInvalidGenerator(),
    )

    with pytest.raises(AnswerValidationError):
        pipeline.ask(
            "What was Microsoft's operating income in 2025?"
        )