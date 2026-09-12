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
from app.rag.hybrid_retriever import (
    HybridRetriever,
)
from app.rag.lexical_retriever import (
    LexicalRetriever,
)
from app.rag.reranker import Reranker
from app.rag.retriever import Retriever
from app.rag.vector_store import VectorStore


BENCHMARK_PATH = Path(
    "evaluation/retrieval_benchmark.json"
)

RETRIEVAL_DEPTH = 20

HYBRID_CANDIDATE_DEPTH = 20


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
        (
            relevant_ranks
            if relevant_ranks
            else "None in Top 20"
        ),
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


def _new_aggregate() -> dict:
    return {
        "hit_1": [],
        "hit_3": [],
        "hit_5": [],
        "rr": [],
    }


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

    dense_retriever = Retriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
    )

    lexical_retriever = LexicalRetriever(
        vector_store=vector_store,
    )

    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        lexical_retriever=lexical_retriever,
        dense_weight=1.0,
        lexical_weight=1.0,
    )

    reranker = Reranker()

    dense_aggregate = (
        _new_aggregate()
    )

    lexical_aggregate = (
        _new_aggregate()
    )

    hybrid_aggregate = (
        _new_aggregate()
    )

    dense_reranked_aggregate = (
        _new_aggregate()
    )

    hybrid_reranked_aggregate = (
        _new_aggregate()
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "Retrieval Benchmark: "
        "Dense vs BM25 vs Hybrid RRF "
        "vs Dense + Reranker "
        "vs Hybrid + Reranker"
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

        dense_results = (
            dense_retriever.retrieve(
                query=example.question,
                top_k=RETRIEVAL_DEPTH,
                where=where,
            )
        )

        lexical_results = (
            lexical_retriever.retrieve(
                query=example.question,
                top_k=RETRIEVAL_DEPTH,
                where=where,
            )
        )

        hybrid_results = (
            hybrid_retriever.retrieve(
                query=example.question,
                top_k=RETRIEVAL_DEPTH,
                candidate_k=(
                    HYBRID_CANDIDATE_DEPTH
                ),
                where=where,
            )
        )

        dense_reranked_results = (
            reranker.rerank(
                query=example.question,
                results=dense_results,
            )
        )

        hybrid_reranked_results = (
            reranker.rerank(
                query=example.question,
                results=hybrid_results,
            )
        )

        dense_metrics = (
            _calculate_metrics(
                results=dense_results,
                relevant_sources=(
                    example.relevant_sources
                ),
            )
        )

        lexical_metrics = (
            _calculate_metrics(
                results=lexical_results,
                relevant_sources=(
                    example.relevant_sources
                ),
            )
        )

        hybrid_metrics = (
            _calculate_metrics(
                results=hybrid_results,
                relevant_sources=(
                    example.relevant_sources
                ),
            )
        )

        dense_reranked_metrics = (
            _calculate_metrics(
                results=(
                    dense_reranked_results
                ),
                relevant_sources=(
                    example.relevant_sources
                ),
            )
        )

        hybrid_reranked_metrics = (
            _calculate_metrics(
                results=(
                    hybrid_reranked_results
                ),
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
            lexical_aggregate,
            lexical_metrics,
        )

        _append_metrics(
            hybrid_aggregate,
            hybrid_metrics,
        )

        _append_metrics(
            dense_reranked_aggregate,
            dense_reranked_metrics,
        )

        _append_metrics(
            hybrid_reranked_aggregate,
            hybrid_reranked_metrics,
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
            f"Ticker: "
            f"{example.ticker}"
        )

        _print_query_metrics(
            label="Dense",
            metrics=dense_metrics,
        )

        _print_query_metrics(
            label="BM25",
            metrics=lexical_metrics,
        )

        _print_query_metrics(
            label="Hybrid RRF",
            metrics=hybrid_metrics,
        )

        _print_query_metrics(
            label="Dense + Reranker",
            metrics=(
                dense_reranked_metrics
            ),
        )

        _print_query_metrics(
            label=(
                "Hybrid RRF + Reranker"
            ),
            metrics=(
                hybrid_reranked_metrics
            ),
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
            "Aggregate Results: BM25"
        ),
        aggregate=lexical_aggregate,
        query_count=len(examples),
    )

    _print_aggregate(
        label=(
            "Aggregate Results: Hybrid RRF"
        ),
        aggregate=hybrid_aggregate,
        query_count=len(examples),
    )

    _print_aggregate(
        label=(
            "Aggregate Results: "
            "Dense + Reranker"
        ),
        aggregate=(
            dense_reranked_aggregate
        ),
        query_count=len(examples),
    )

    _print_aggregate(
        label=(
            "Aggregate Results: "
            "Hybrid RRF + Reranker"
        ),
        aggregate=(
            hybrid_reranked_aggregate
        ),
        query_count=len(examples),
    )


if __name__ == "__main__":
    main()