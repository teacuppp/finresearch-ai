from pathlib import Path

from app.evaluation.benchmark import (
    load_retrieval_benchmark,
)
from app.evaluation.retrieval_metrics import (
    find_relevant_ranks,
    hit_at_k,
    mean,
    reciprocal_rank,
)
from app.rag.embeddings import EmbeddingModel
from app.rag.retriever import Retriever
from app.rag.vector_store import VectorStore


BENCHMARK_PATH = Path(
    "evaluation/retrieval_benchmark.json"
)

RETRIEVAL_DEPTH = 20


def main():
    examples = load_retrieval_benchmark(
        BENCHMARK_PATH
    )

    embedding_model = EmbeddingModel()

    vector_store = VectorStore(
        path="data/chroma",
        collection_name="financial_documents",
    )

    retriever = Retriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
    )

    hit_1_scores = []
    hit_3_scores = []
    hit_5_scores = []
    reciprocal_ranks = []

    print(
        "\n"
        + "=" * 100
    )
    print("Retrieval Benchmark")
    print("=" * 100)

    for example in examples:
        where = {
            "$and": [
                {
                    "ticker": {
                        "$eq": example.ticker
                    }
                },
                {
                    "fiscal_year": {
                        "$eq": example.fiscal_year
                    }
                },
            ]
        }

        results = retriever.retrieve(
            query=example.question,
            top_k=RETRIEVAL_DEPTH,
            where=where,
        )

        relevant_ranks = find_relevant_ranks(
            results=results,
            relevant_sources=example.relevant_sources,
        )

        hit_1 = hit_at_k(
            relevant_ranks,
            1,
        )

        hit_3 = hit_at_k(
            relevant_ranks,
            3,
        )

        hit_5 = hit_at_k(
            relevant_ranks,
            5,
        )

        rr = reciprocal_rank(
            relevant_ranks
        )

        hit_1_scores.append(hit_1)
        hit_3_scores.append(hit_3)
        hit_5_scores.append(hit_5)
        reciprocal_ranks.append(rr)

        print(
            "\n"
            + "-" * 100
        )

        print(f"ID: {example.id}")
        print(
            f"Question: {example.question}"
        )
        print(
            f"Ticker: {example.ticker}"
        )

        print(
            "Relevant ranks:",
            relevant_ranks
            if relevant_ranks
            else "None in Top 20",
        )

        print(
            f"Hit@1: {hit_1:.0f}"
        )

        print(
            f"Hit@3: {hit_3:.0f}"
        )

        print(
            f"Hit@5: {hit_5:.0f}"
        )

        print(
            f"RR@20: {rr:.4f}"
        )

    print(
        "\n"
        + "=" * 100
    )
    print("Aggregate Results")
    print("=" * 100)

    print(
        f"Queries: {len(examples)}"
    )

    print(
        f"Hit@1:  "
        f"{mean(hit_1_scores):.4f}"
    )

    print(
        f"Hit@3:  "
        f"{mean(hit_3_scores):.4f}"
    )

    print(
        f"Hit@5:  "
        f"{mean(hit_5_scores):.4f}"
    )

    print(
        f"MRR@20: "
        f"{mean(reciprocal_ranks):.4f}"
    )


if __name__ == "__main__":
    main()