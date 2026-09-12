from app.rag.models import RetrievedChunk
from app.rag.pipeline import RAGPipeline


def _chunk(
    text: str,
    document: str,
    page: int,
    chunk_index: int,
) -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        document=document,
        page=page,
        chunk_index=chunk_index,
        distance=0.1,
        company="Apple",
        ticker="AAPL",
        fiscal_year=2025,
        document_type="10-K",
    )


class FakeRetriever:
    def __init__(
        self,
        results: list[RetrievedChunk],
    ):
        self.results = results
        self.calls = []

    def retrieve(
        self,
        query: str,
        top_k: int,
        where: dict | None = None,
    ) -> list[RetrievedChunk]:
        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
                "where": where,
            }
        )

        return self.results[:top_k]


class FakeReranker:
    def __init__(
        self,
        results: list[RetrievedChunk],
    ):
        self.results = results
        self.calls = []

    def rerank(
        self,
        query: str,
        results: list[RetrievedChunk],
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        self.calls.append(
            {
                "query": query,
                "results": results,
                "top_k": top_k,
            }
        )

        if top_k is None:
            return self.results

        return self.results[:top_k]


class FakeGenerator:
    def __init__(
        self,
        answer: str = (
            "Apple reported revenue "
            "of $391 billion [Source 1]."
        ),
    ):
        self.answer = answer
        self.generate_calls = []
        self.repair_calls = []

    def generate(
        self,
        question: str,
        context: str,
    ) -> str:
        self.generate_calls.append(
            {
                "question": question,
                "context": context,
            }
        )

        return self.answer

    def repair(
        self,
        question: str,
        context: str,
        previous_answer: str,
    ) -> str:
        self.repair_calls.append(
            {
                "question": question,
                "context": context,
                "previous_answer": (
                    previous_answer
                ),
            }
        )

        return (
            "Apple reported revenue "
            "of $391 billion [Source 1]."
        )


def test_pipeline_retrieves_candidate_depth():
    chunks = [
        _chunk(
            text=f"Chunk {index}",
            document="a.pdf",
            page=index + 1,
            chunk_index=index,
        )
        for index in range(20)
    ]

    retriever = FakeRetriever(
        results=chunks,
    )

    reranker = FakeReranker(
        results=chunks,
    )

    generator = FakeGenerator()

    pipeline = RAGPipeline(
        retriever=retriever,
        reranker=reranker,
        generator=generator,
        retrieval_depth=20,
    )

    pipeline.ask(
        question=(
            "What was Apple's revenue?"
        ),
        top_k=5,
    )

    assert len(
        retriever.calls
    ) == 1

    assert (
        retriever.calls[0]["top_k"]
        == 20
    )


def test_pipeline_reranks_retrieved_candidates():
    candidates = [
        _chunk(
            text="Candidate 1",
            document="a.pdf",
            page=1,
            chunk_index=0,
        ),
        _chunk(
            text="Candidate 2",
            document="a.pdf",
            page=2,
            chunk_index=1,
        ),
    ]

    reranked = [
        candidates[1],
        candidates[0],
    ]

    retriever = FakeRetriever(
        results=candidates,
    )

    reranker = FakeReranker(
        results=reranked,
    )

    generator = FakeGenerator()

    pipeline = RAGPipeline(
        retriever=retriever,
        reranker=reranker,
        generator=generator,
    )

    result = pipeline.ask(
        question="test question",
        top_k=2,
    )

    assert len(
        reranker.calls
    ) == 1

    assert (
        reranker.calls[0]["results"]
        == candidates
    )

    assert (
        reranker.calls[0]["top_k"]
        == 2
    )

    assert (
        result.sources
        == reranked
    )


def test_pipeline_passes_where_filter():
    candidates = [
        _chunk(
            text="Revenue evidence",
            document="a.pdf",
            page=1,
            chunk_index=0,
        ),
    ]

    retriever = FakeRetriever(
        results=candidates,
    )

    reranker = FakeReranker(
        results=candidates,
    )

    generator = FakeGenerator()

    pipeline = RAGPipeline(
        retriever=retriever,
        reranker=reranker,
        generator=generator,
    )

    where = {
        "$and": [
            {
                "ticker": {
                    "$eq": "AAPL"
                }
            },
            {
                "fiscal_year": {
                    "$eq": 2025
                }
            },
        ]
    }

    pipeline.ask(
        question=(
            "What was Apple's revenue?"
        ),
        where=where,
    )

    assert (
        retriever.calls[0]["where"]
        == where
    )


def test_pipeline_builds_context_from_reranked_chunks():
    first = _chunk(
        text="First relevant evidence",
        document="a.pdf",
        page=1,
        chunk_index=0,
    )

    second = _chunk(
        text="Second relevant evidence",
        document="a.pdf",
        page=2,
        chunk_index=1,
    )

    retriever = FakeRetriever(
        results=[
            second,
            first,
        ],
    )

    reranker = FakeReranker(
        results=[
            first,
            second,
        ],
    )

    generator = FakeGenerator()

    pipeline = RAGPipeline(
        retriever=retriever,
        reranker=reranker,
        generator=generator,
    )

    result = pipeline.ask(
        question="test question",
        top_k=2,
    )

    context = (
        generator.generate_calls[0][
            "context"
        ]
    )

    assert (
        "First relevant evidence"
        in context
    )

    assert (
        "Second relevant evidence"
        in context
    )

    assert (
        context.index(
            "First relevant evidence"
        )
        < context.index(
            "Second relevant evidence"
        )
    )

    assert result.sources == [
        first,
        second,
    ]


def test_pipeline_limits_final_sources_to_top_k():
    candidates = [
        _chunk(
            text=f"Chunk {index}",
            document="a.pdf",
            page=index + 1,
            chunk_index=index,
        )
        for index in range(10)
    ]

    retriever = FakeRetriever(
        results=candidates,
    )

    reranker = FakeReranker(
        results=candidates,
    )

    generator = FakeGenerator()

    pipeline = RAGPipeline(
        retriever=retriever,
        reranker=reranker,
        generator=generator,
        retrieval_depth=20,
    )

    result = pipeline.ask(
        question="test question",
        top_k=5,
    )

    assert len(
        result.sources
    ) == 5

def test_pipeline_handles_empty_retrieval_results():
    retriever = FakeRetriever(
        results=[],
    )

    reranker = FakeReranker(
        results=[],
    )

    generator = FakeGenerator(
        answer=(
            "I could not find enough "
            "information in the "
            "provided documents."
        ),
    )

    pipeline = RAGPipeline(
        retriever=retriever,
        reranker=reranker,
        generator=generator,
    )

    result = pipeline.ask(
        question="test question",
        top_k=5,
    )

    assert result.sources == []

    assert (
        reranker.calls[0]["results"]
        == []
    )

    assert (
        result.answer
        == (
            "I could not find enough "
            "information in the "
            "provided documents."
        )
    )

    assert (
        generator.repair_calls
        == []
    )

def test_pipeline_rejects_nonpositive_top_k():
    retriever = FakeRetriever(
        results=[],
    )

    reranker = FakeReranker(
        results=[],
    )

    generator = FakeGenerator()

    pipeline = RAGPipeline(
        retriever=retriever,
        reranker=reranker,
        generator=generator,
    )

    try:
        pipeline.ask(
            question="test question",
            top_k=0,
        )

        assert False

    except ValueError as error:
        assert str(error) == (
            "top_k must be positive"
        )


def test_pipeline_rejects_nonpositive_retrieval_depth():
    retriever = FakeRetriever(
        results=[],
    )

    reranker = FakeReranker(
        results=[],
    )

    generator = FakeGenerator()

    try:
        RAGPipeline(
            retriever=retriever,
            reranker=reranker,
            generator=generator,
            retrieval_depth=0,
        )

        assert False

    except ValueError as error:
        assert str(error) == (
            "retrieval_depth must be positive"
        )