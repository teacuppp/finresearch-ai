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
from app.rag.reranker import Reranker
from app.rag.retriever import Retriever
from app.rag.vector_store import VectorStore


BENCHMARK_PATH = Path(
    "evaluation/retrieval_benchmark.json"
)

RETRIEVAL_DEPTH = 20


def _calculate_metrics(
    results,
    relevant_sources,
) -> dict:
    relevant_ranks = find_relevant_ranks(
        results=results,
        relevant_sources=relevant_sources,
    )

    return {
        "relevant_ranks": relevant_ranks,
        "hit_1": hit_at_k(
            relevant_ranks,
            1,
        ),
        "hit_3": hit_at_k(
            relevant_ranks,
            3,
        ),
        "hit_5": hit_at_k(
            relevant_ranks,
            5,
        ),
        "rr": reciprocal_rank(
            relevant_ranks
        ),
    }


def _print_query_metrics(
    label: str,
    metrics: dict,
) -> None:
    relevant_ranks = metrics[
        "relevant_ranks"
    ]

    print(f"\n{label}")

    print(
        "Relevant ranks:",
        relevant_ranks
        if relevant_ranks
        else "None in Top 20",
    )

    print(
        f"Hit@1: "
        f"{metrics['hit_1']:.0f}"
    )

    print(
        f"Hit@3: "
        f"{metrics['hit_3']:.0f}"
    )

    print(
        f"Hit@5: "
        f"{metrics['hit_5']:.0f}"
    )

    print(
        f"RR@20: "
        f"{metrics['rr']:.4f}"
    )


def _append_metrics(
    aggregate: dict,
    metrics: dict,
) -> None:
    aggregate[
        "hit_1"
    ].append(
        metrics["hit_1"]
    )

    aggregate[
        "hit_3"
    ].append(
        metrics["hit_3"]
    )

    aggregate[
        "hit_5"
    ].append(
        metrics["hit_5"]
    )

    aggregate[
        "rr"
    ].append(
        metrics["rr"]
    )


def _new_aggregate() -> dict:
    return {
        "hit_1": [],
        "hit_3": [],
        "hit_5": [],
        "rr": [],
    }


def _print_aggregate(
    label: str,
    aggregate: dict,
    query_count: int,
) -> None:
    print(
        "\n"
        + "=" * 100
    )

    print(label)

    print(
        "=" * 100
    )

    print(
        f"Queries: {query_count}"
    )

    print(
        f"Hit@1:  "
        f"{mean(aggregate['hit_1']):.4f}"
    )

    print(
        f"Hit@3:  "
        f"{mean(aggregate['hit_3']):.4f}"
    )

    print(
        f"Hit@5:  "
        f"{mean(aggregate['hit_5']):.4f}"
    )

    print(
        f"MRR@20: "
        f"{mean(aggregate['rr']):.4f}"
    )


def main():
    examples = load_retrieval_benchmark(
        BENCHMARK_PATH
    )

    embedding_model = EmbeddingModel()

    vector_store = VectorStore(
        path="data/chroma",
        collection_name=(
            "financial_documents"
        ),
    )

    retriever = Retriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
    )

    reranker = Reranker()

    dense_aggregate = (
        _new_aggregate()
    )

    reranked_aggregate = (
        _new_aggregate()
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "Retrieval Benchmark: "
        "Dense vs Dense + Reranker"
    )

    print(
        "=" * 100
    )

    for example in examples:
        where = {
            "$and": [
                {
                    "ticker": {
                        "$eq": (
                            example.ticker
                        )
                    }
                },
                {
                    "fiscal_year": {
                        "$eq": (
                            example.fiscal_year
                        )
                    }
                },
            ]
        }

        candidates = (
            retriever.retrieve(
                query=example.question,
                top_k=RETRIEVAL_DEPTH,
                where=where,
            )
        )

        reranked_results = (
            reranker.rerank(
                query=example.question,
                results=candidates,
            )
        )

        dense_metrics = (
            _calculate_metrics(
                results=candidates,
                relevant_sources=(
                    example.relevant_sources
                ),
            )
        )

        reranked_metrics = (
            _calculate_metrics(
                results=reranked_results,
                relevant_sources=(
                    example.relevant_sources
                ),
            )
        )

        _append_metrics(
            dense_aggregate,
            dense_metrics,
        )

        _append_metrics(
            reranked_aggregate,
            reranked_metrics,
        )

        print(
            "\n"
            + "-" * 100
        )

        print(
            f"ID: {example.id}"
        )

        print(
            f"Question: "
            f"{example.question}"
        )

        print(
            f"Ticker: {example.ticker}"
        )

        _print_query_metrics(
            label="Dense",
            metrics=dense_metrics,
        )

        _print_query_metrics(
            label="Dense + Reranker",
            metrics=reranked_metrics,
        )

    _print_aggregate(
        label=(
            "Aggregate Results: Dense"
        ),
        aggregate=dense_aggregate,
        query_count=len(examples),
    )

    _print_aggregate(
        label=(
            "Aggregate Results: "
            "Dense + Reranker"
        ),
        aggregate=reranked_aggregate,
        query_count=len(examples),
    )


if __name__ == "__main__":
    main()