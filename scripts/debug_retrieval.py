from pathlib import Path

from app.rag.embeddings import EmbeddingModel
from app.rag.retriever import Retriever
from app.rag.vector_store import VectorStore


# QUESTION = "What was Apple's total revenue in 2025?"
 
# QUESTION = "What was Microsoft's total revenue in 2025?"
# EXPECTED_TERMS = [
#     "MSFT",
#     281,724,
#     "2025",
# ]

QUESTION = "What was Microsoft's operating income in 2025?"
EXPECTED_TEXT = "128,528"
TICKET = "MSFT"
# QUESTION = (
#     "How much revenue did Apple generate "
#     "from services in 2025?"
# )


def main():
    print(
        "Chroma path:",
        Path("data/chroma").resolve(),
    )

    embedding_model = EmbeddingModel()

    vector_store = VectorStore(
        path="data/chroma",
        collection_name="financial_documents",
    )

    print(
        "Collection count:",
        vector_store.count(),
    )

    retriever = Retriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
    )

    where = {
        "$and": [
            {
                "ticker": {
                    "$eq": TICKET
                }
            },
            {
                "fiscal_year": {
                    "$eq": 2025
                }
            },
        ]
    }

    results = retriever.retrieve(
        query=QUESTION,
        top_k=20,
        where=where,
    )

    for rank, result in enumerate(
        results,
        start=1,
    ):
        print(
            "\n"
            + "=" * 100
        )

        print(f"Rank: {rank}")
        print(f"Document: {result.document}")
        print(f"Page: {result.page}")
        print(
            f"Chunk: {result.chunk_index}"
        )
        print(
            f"Distance: {result.distance:.6f}"
        )

        print(
            "Contains "+EXPECTED_TEXT+":",
            EXPECTED_TEXT in result.text,
        )

        print("-" * 100)
        print(result.text)


if __name__ == "__main__":
    main()

#.  python -m scripts.debug_retrieval