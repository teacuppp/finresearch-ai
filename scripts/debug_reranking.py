import argparse

from app.evaluation.benchmark import (
    load_retrieval_benchmark,
)
from app.evaluation.retrieval_metrics import (
    find_relevant_ranks,
)
from app.rag.embeddings import EmbeddingModel
from app.rag.reranker import Reranker
from app.rag.retriever import Retriever
from app.rag.vector_store import VectorStore


RETRIEVAL_DEPTH = 20


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Inspect dense retrieval and "
            "cross-encoder reranking results."
        )
    )

    parser.add_argument(
        "--benchmark-id",
        required=True,
        help=(
            "Benchmark example ID, e.g. "
            "aapl_operating_income_2025"
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of reranked results to print.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    examples = load_retrieval_benchmark(
        "evaluation/retrieval_benchmark.json"
    )

    example = next(
        (
            item
            for item in examples
            if item.id == args.benchmark_id
        ),
        None,
    )

    if example is None:
        available_ids = [
            item.id
            for item in examples
        ]

        raise ValueError(
            "Unknown benchmark ID: "
            f"{args.benchmark_id}. "
            "Available IDs: "
            f"{available_ids}"
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

    reranker = Reranker()

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

    dense_results = retriever.retrieve(
        query=example.question,
        top_k=RETRIEVAL_DEPTH,
        where=where,
    )

    scores = reranker.score(
        query=example.question,
        results=dense_results,
    )

    scored_results = list(
        zip(
            dense_results,
            scores,
            range(
                1,
                len(dense_results) + 1,
            ),
        )
    )

    scored_results.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    print("=" * 100)
    print(f"ID: {example.id}")
    print(f"Question: {example.question}")
    print(f"Ticker: {example.ticker}")
    print("=" * 100)

    for rerank_index, (
        result,
        score,
        dense_rank,
    ) in enumerate(
        scored_results[: args.top_k],
        start=1,
    ):
        relevant = bool(
            find_relevant_ranks(
                results=[result],
                relevant_sources=(
                    example.relevant_sources
                ),
            )
        )

        print()
        print("-" * 100)

        print(
            f"Rerank Rank: {rerank_index}"
        )
        print(
            f"Dense Rank: {dense_rank}"
        )
        print(
            f"Relevant: "
            f"{'YES' if relevant else 'NO'}"
        )
        print(
            f"Dense Distance: "
            f"{result.distance:.6f}"
        )
        print(
            f"Reranker Score: "
            f"{float(score):.6f}"
        )
        print(
            f"Document: {result.document}"
        )
        print(
            f"Page: {result.page}"
        )
        print(
            f"Chunk: {result.chunk_index}"
        )

        print("-" * 100)

        lines = [
            line.strip()
            for line in result.text.splitlines()
            if line.strip()
        ]

        for line_number, line in enumerate(
            lines,
            start=1,
        ):
            print(
                f"{line_number:02d}: {line}"
            )


if __name__ == "__main__":
    main()