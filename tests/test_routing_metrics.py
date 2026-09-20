import pytest

from app.evaluation.models import RoutingMetrics
from app.evaluation.routing_metrics import (
    calculate_routing_metrics,
)


def test_calculates_perfect_routing_metrics():
    metrics = calculate_routing_metrics(
        expected_routes=[
            "sql",
            "rag",
        ],
        predicted_routes=[
            "sql",
            "rag",
        ],
    )

    assert metrics == RoutingMetrics(
        total_examples=2,
        correct_examples=2,
        accuracy=1.0,
        sql_precision=1.0,
        sql_recall=1.0,
        rag_precision=1.0,
        rag_recall=1.0,
    )


def test_calculates_known_confusion_matrix():
    metrics = calculate_routing_metrics(
        expected_routes=[
            "sql",
            "sql",
            "sql",
            "sql",
            "rag",
            "rag",
        ],
        predicted_routes=[
            "sql",
            "sql",
            "rag",
            "rag",
            "sql",
            "rag",
        ],
    )

    assert metrics.total_examples == 6
    assert metrics.correct_examples == 3
    assert metrics.accuracy == 0.5
    assert metrics.sql_precision == pytest.approx(
        2 / 3
    )
    assert metrics.sql_recall == 0.5
    assert metrics.rag_precision == pytest.approx(
        1 / 3
    )
    assert metrics.rag_recall == 0.5


def test_returns_zero_for_undefined_class_metrics():
    metrics = calculate_routing_metrics(
        expected_routes=[
            "rag",
            "rag",
        ],
        predicted_routes=[
            "rag",
            "rag",
        ],
    )

    assert metrics.sql_precision == 0.0
    assert metrics.sql_recall == 0.0
    assert metrics.rag_precision == 1.0
    assert metrics.rag_recall == 1.0


def test_rejects_mismatched_route_counts():
    with pytest.raises(
        ValueError,
        match="route counts must match",
    ):
        calculate_routing_metrics(
            expected_routes=["sql"],
            predicted_routes=[],
        )
