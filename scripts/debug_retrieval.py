from pathlib import Path

# from app.evaluation.retrieval_metrics import (
#     get_evidence_span,
# )
from app.rag.embeddings import EmbeddingModel
from app.rag.retriever import Retriever
from app.rag.vector_store import VectorStore
#.  python -m scripts.debug_retrieval
QUESTION = (
    "What was Microsoft's net income in 2025?"
)

TICKER = "MSFT"

NUMBER = "101,832"
# QUESTION = (
#     "How much revenue did Apple generate "
#     "from services in 2025?"
# )

# TICKER = "AAPL"

# REQUIRED_TERMS = [
#     "Services",
#     "109,158",
#     "2025",
# ]


# QUESTION = (
#     "What was Microsoft's total revenue "
#     "in 2025?"
# )

# TICKER = "MSFT"

# REQUIRED_TERMS = [
#     "Revenue",
#     "281,724",
#     "2025",
# ]

# QUESTION = (
#     "What was Apple's total revenue in 2025?"
# )

# TICKER = "AAPL"

# TARGET_RANKS = {
#     13,
#     14,
# }


# QUESTION = (
#     "What was Microsoft's operating income in 2025?"
# )

# TICKER = "MSFT"

# TARGET_RANKS = {
#     1,
#     2,
# }

# QUESTION = (
#     "What was Apple's total revenue in 2025?"
# )

# TICKER = "AAPL"

TARGET_RANKS = set(range(1, 21))


# QUESTION = (
#     "What was Microsoft's total revenue in 2025?"
# )

# TICKER = "MSFT"


# QUESTION = (
#     "How much revenue did Apple generate from services in 2025?"
# )

# TICKER = "AAPL"

# QUESTION = (
#     "What was Microsoft's operating income in 2025?"
# )

# TICKER = "MSFT"


def _print_numbered_lines(
    text: str,
) -> None:
    lines = [
        line
        for line in text.splitlines()
        if line.strip()
    ]

    for index, line in enumerate(
        lines,
        start=1,
    ):
        print(
            f"{index:02d}: {line}"
        )

def _normalize_text(
    text: str,
) -> str:
    return " ".join(
        text.casefold().split()
    )


def _find_term_positions(
    text: str,
    terms: list[str],
) -> dict[str, int]:
    normalized_text = _normalize_text(
        text
    )

    positions = {}

    for term in terms:
        normalized_term = _normalize_text(
            term
        )

        positions[term] = (
            normalized_text.find(
                normalized_term
            )
        )

    return positions


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
                    "$eq": TICKER
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
        # if rank not in TARGET_RANKS:
        #     continue

        # if "281,724" not in result.text:
        #   continue

        if NUMBER not in result.text:
                continue

        # if "416,161" not in result.text:
        #   continue

        # if "109,158" not in result.text:
        #   continue

        # if "128,528" not in result.text:
        #     continue

        # if "416,161" not in result.text:
        #     continue
        # positions = _find_term_positions(
        #     result.text,
        #     REQUIRED_TERMS,
        # )

        # span = get_evidence_span(
        #     result.text,
        #     REQUIRED_TERMS,
        # )

        print(
            "\n"
            + "=" * 100
        )

        print(f"Rank: {rank}")
        print(
            f"Document: {result.document}"
        )
        print(
            f"Page: {result.page}"
        )
        print(
            f"Chunk: {result.chunk_index}"
        )
        print(
            f"Distance: {result.distance:.6f}"
        )

        # print(
        #     f"Required terms: "
        #     f"{REQUIRED_TERMS}"
        # )

        # print(
        #     f"Term positions: "
        #     f"{positions}"
        # )

        # print(
        #     f"Evidence span: {span}"
        # )

        print("-" * 100)
        _print_numbered_lines(
            result.text
        )


if __name__ == "__main__":
    main()

#.  python -m scripts.debug_retrieval