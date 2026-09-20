from collections.abc import Sequence

from app.agent.state import Route
from app.evaluation.models import RoutingMetrics


def _safe_divide(
    numerator: int,
    denominator: int,
) -> float:
    if denominator == 0:
        return 0.0

    return numerator / denominator


def _validate_routes(
    routes: Sequence[Route],
    label: str,
) -> None:
    for route in routes:
        if route not in {"rag", "sql"}:
            raise ValueError(
                f"Invalid {label} route: {route!r}"
            )


def calculate_routing_metrics(
    expected_routes: Sequence[Route],
    predicted_routes: Sequence[Route],
) -> RoutingMetrics:
    if len(expected_routes) != len(predicted_routes):
        raise ValueError(
            "Expected and predicted route counts must match"
        )

    _validate_routes(
        expected_routes,
        "expected",
    )
    _validate_routes(
        predicted_routes,
        "predicted",
    )

    total_examples = len(expected_routes)
    correct_examples = sum(
        expected == predicted
        for expected, predicted in zip(
            expected_routes,
            predicted_routes,
        )
    )

    sql_true_positives = sum(
        expected == "sql" and predicted == "sql"
        for expected, predicted in zip(
            expected_routes,
            predicted_routes,
        )
    )
    sql_false_positives = sum(
        expected == "rag" and predicted == "sql"
        for expected, predicted in zip(
            expected_routes,
            predicted_routes,
        )
    )
    sql_false_negatives = sum(
        expected == "sql" and predicted == "rag"
        for expected, predicted in zip(
            expected_routes,
            predicted_routes,
        )
    )

    rag_true_positives = sum(
        expected == "rag" and predicted == "rag"
        for expected, predicted in zip(
            expected_routes,
            predicted_routes,
        )
    )
    rag_false_positives = sum(
        expected == "sql" and predicted == "rag"
        for expected, predicted in zip(
            expected_routes,
            predicted_routes,
        )
    )
    rag_false_negatives = sum(
        expected == "rag" and predicted == "sql"
        for expected, predicted in zip(
            expected_routes,
            predicted_routes,
        )
    )

    return RoutingMetrics(
        total_examples=total_examples,
        correct_examples=correct_examples,
        accuracy=_safe_divide(
            correct_examples,
            total_examples,
        ),
        sql_precision=_safe_divide(
            sql_true_positives,
            sql_true_positives + sql_false_positives,
        ),
        sql_recall=_safe_divide(
            sql_true_positives,
            sql_true_positives + sql_false_negatives,
        ),
        rag_precision=_safe_divide(
            rag_true_positives,
            rag_true_positives + rag_false_positives,
        ),
        rag_recall=_safe_divide(
            rag_true_positives,
            rag_true_positives + rag_false_negatives,
        ),
    )
