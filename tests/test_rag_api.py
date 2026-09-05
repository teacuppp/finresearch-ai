from fastapi.testclient import TestClient

from app.dependencies import get_rag_pipeline
from app.main import app
from app.rag.models import RetrievedChunk
from app.rag.pipeline import RAGResult
from app.rag.validation import AnswerValidationError


class FakeRAGPipeline:
    def __init__(self):
        self.last_question = None
        self.last_top_k = None

    def ask(
        self,
        question: str,
        top_k: int = 5,
        where: dict | None = None,
    ) -> RAGResult:
        self.last_question = question
        self.last_top_k = top_k
        self.last_where = where

        return RAGResult(
            answer=(
                "Apple reported total net sales of "
                "$416.161 billion in fiscal year 2025. "
                "[Source 1]"
            ),
            sources=[
                RetrievedChunk(
                    text=(
                        "Apple reported total net sales "
                        "of $416.161 billion."
                    ),
                    document="apple.pdf",
                    page=35,
                    chunk_index=0,
                    distance=0.6041,
                )
            ],
        )


fake_pipeline = FakeRAGPipeline()


def override_get_rag_pipeline():
    return fake_pipeline


app.dependency_overrides[
    get_rag_pipeline
] = override_get_rag_pipeline


client = TestClient(app)


def test_ask_question():
    response = client.post(
        "/rag/ask",
        json={
            "question": (
                "What was Apple's total revenue in 2025?"
            ),
            "top_k": 5,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert (
        data["answer"]
        == (
            "Apple reported total net sales of "
            "$416.161 billion in fiscal year 2025. "
            "[Source 1]"
        )
    )

    assert len(data["sources"]) == 1

    source = data["sources"][0]

    assert source["document"] == "apple.pdf"
    assert source["page"] == 35
    assert source["chunk_index"] == 0
    assert source["distance"] == 0.6041


def test_ask_question_uses_default_top_k():
    response = client.post(
        "/rag/ask",
        json={
            "question": "What was Apple's revenue?"
        },
    )

    assert response.status_code == 200

    assert fake_pipeline.last_top_k == 5


def test_ask_question_custom_top_k():
    response = client.post(
        "/rag/ask",
        json={
            "question": "What was Apple's revenue?",
            "top_k": 3,
        },
    )

    assert response.status_code == 200

    assert fake_pipeline.last_top_k == 3


def test_ask_question_rejects_empty_question():
    response = client.post(
        "/rag/ask",
        json={
            "question": "",
            "top_k": 5,
        },
    )

    assert response.status_code == 422


def test_ask_question_rejects_zero_top_k():
    response = client.post(
        "/rag/ask",
        json={
            "question": "What was Apple's revenue?",
            "top_k": 0,
        },
    )

    assert response.status_code == 422


def test_ask_question_rejects_large_top_k():
    response = client.post(
        "/rag/ask",
        json={
            "question": "What was Apple's revenue?",
            "top_k": 21,
        },
    )

    assert response.status_code == 422




#RAG API 也增加 502 测试
# class FailingRAGPipeline:
#     def ask(
#         self,
#         question: str,
#         top_k: int = 5,
#     ):
#         raise AnswerValidationError(
#             "Missing citation"
#         )

class FailingRAGPipeline:
    def ask(
        self,
        question: str,
        top_k: int = 5,
        where: dict | None = None,
    ):
        raise AnswerValidationError(
            "Missing citation"
        )


def test_rag_api_handles_generation_validation_error():
    def override_failure():
        return FailingRAGPipeline()

    app.dependency_overrides[
        get_rag_pipeline
    ] = override_failure

    response = client.post(
        "/rag/ask",
        json={
            "question": (
                "What was Microsoft's operating income in 2025?"
            ),
            "top_k": 5,
        },
    )

    assert response.status_code == 502

    assert response.json() == {
        "detail": (
            "The language model produced an invalid "
            "source-grounded response."
        )
    }

    app.dependency_overrides[
        get_rag_pipeline
    ] = override_get_rag_pipeline


#增加 API test：
def test_ask_question_builds_metadata_filter():
    response = client.post(
        "/rag/ask",
        json={
            "question": "What was total revenue?",
            "ticker": "MSFT",
            "fiscal_year": 2025,
            "top_k": 5,
        },
    )

    assert response.status_code == 200

    assert fake_pipeline.last_where == {
        "$and": [
            {
                "ticker": {
                    "$eq": "MSFT"
                }
            },
            {
                "fiscal_year": {
                    "$eq": 2025
                }
            },
        ]
    }


    app.dependency_overrides[
        get_rag_pipeline
    ] = override_get_rag_pipeline
    

#。  pytest tests/test_rag_api.py -v