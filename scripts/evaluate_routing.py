"""Run with `.venv/bin/python -m scripts.evaluate_routing` from the repo root."""

from pathlib import Path

from app.agent.router import LLMQuestionRouter
from app.agent.state import Route
from app.evaluation.models import RoutingBenchmarkExample
from app.evaluation.routing_benchmark import (
    load_routing_benchmark,
)
from app.evaluation.routing_metrics import (
    calculate_routing_metrics,
)


BENCHMARK_PATH = Path(
    "evaluation/routing_benchmark.json"
)


def _print_misclassification(
    example: RoutingBenchmarkExample,
    predicted_route: Route,
) -> None:
    print("\nMisclassified example")
    print(f"ID: {example.id}")
    print(f"Question: {example.question}")
    print(f"Expected route: {example.expected_route}")
    print(f"Predicted route: {predicted_route}")


def main() -> None:
    examples = load_routing_benchmark(
        BENCHMARK_PATH
    )
    router = LLMQuestionRouter()
    predicted_routes: list[Route] = []

    print("Routing Benchmark")
    print("=" * 80)

    for example in examples:
        predicted_route = router.route(
            example.question
        )
        predicted_routes.append(
            predicted_route
        )

        if predicted_route != example.expected_route:
            _print_misclassification(
                example=example,
                predicted_route=predicted_route,
            )

    metrics = calculate_routing_metrics(
        expected_routes=[
            example.expected_route
            for example in examples
        ],
        predicted_routes=predicted_routes,
    )

    print("\n" + "=" * 80)
    print("Aggregate Results")
    print("=" * 80)
    print(f"Total examples: {metrics.total_examples}")
    print(f"Correct examples: {metrics.correct_examples}")
    print(f"Accuracy: {metrics.accuracy:.4f}")
    print(f"SQL precision: {metrics.sql_precision:.4f}")
    print(f"SQL recall: {metrics.sql_recall:.4f}")
    print(f"RAG precision: {metrics.rag_precision:.4f}")
    print(f"RAG recall: {metrics.rag_recall:.4f}")


if __name__ == "__main__":
    main()
