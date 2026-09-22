import json
from collections import Counter
from pathlib import Path

import pytest

from app.evaluation.models import RoutingBenchmarkExample
from app.evaluation.routing_benchmark import (
    load_routing_benchmark,
)


def _write_benchmark(
    tmp_path,
    data,
) -> Path:
    benchmark_path = (
        tmp_path
        / "routing_benchmark.json"
    )
    benchmark_path.write_text(
        json.dumps(data),
        encoding="utf-8",
    )

    return benchmark_path


def test_load_routing_benchmark(
    tmp_path,
):
    benchmark_path = _write_benchmark(
        tmp_path,
        [
            {
                "id": "apple_revenue",
                "question": "What was Apple's revenue?",
                "expected_route": "sql",
                "category": "direct_metric_lookup",
            },
            {
                "id": "apple_risks",
                "question": "What risks did Apple disclose?",
                "expected_route": "rag",
            },
        ],
    )

    examples = load_routing_benchmark(
        benchmark_path
    )

    assert examples == [
        RoutingBenchmarkExample(
            id="apple_revenue",
            question="What was Apple's revenue?",
            expected_route="sql",
            category="direct_metric_lookup",
        ),
        RoutingBenchmarkExample(
            id="apple_risks",
            question="What risks did Apple disclose?",
            expected_route="rag",
            category=None,
        ),
    ]


def test_rejects_non_list_outer_structure(
    tmp_path,
):
    benchmark_path = _write_benchmark(
        tmp_path,
        {
            "id": "not_a_list",
        },
    )

    with pytest.raises(
        ValueError,
        match="must be a JSON list",
    ):
        load_routing_benchmark(
            benchmark_path
        )


@pytest.mark.parametrize(
    "missing_field",
    [
        "id",
        "question",
        "expected_route",
    ],
)
def test_rejects_missing_required_fields(
    tmp_path,
    missing_field,
):
    item = {
        "id": "example",
        "question": "What was Apple's revenue?",
        "expected_route": "sql",
    }
    del item[missing_field]
    benchmark_path = _write_benchmark(
        tmp_path,
        [item],
    )

    with pytest.raises(
        ValueError,
        match=(
            "missing required field: "
            f"{missing_field}"
        ),
    ):
        load_routing_benchmark(
            benchmark_path
        )


def test_rejects_duplicate_ids(
    tmp_path,
):
    benchmark_path = _write_benchmark(
        tmp_path,
        [
            {
                "id": "duplicate",
                "question": "What was Apple's revenue?",
                "expected_route": "sql",
            },
            {
                "id": "duplicate",
                "question": "What risks did Apple disclose?",
                "expected_route": "rag",
            },
        ],
    )

    with pytest.raises(
        ValueError,
        match="Duplicate benchmark id: duplicate",
    ):
        load_routing_benchmark(
            benchmark_path
        )


def test_rejects_invalid_expected_route(
    tmp_path,
):
    benchmark_path = _write_benchmark(
        tmp_path,
        [
            {
                "id": "invalid_route",
                "question": "Analyze Apple.",
                "expected_route": "hybrid",
            }
        ],
    )

    with pytest.raises(
        ValueError,
        match="Invalid expected route",
    ):
        load_routing_benchmark(
            benchmark_path
        )


def test_rejects_empty_question(
    tmp_path,
):
    benchmark_path = _write_benchmark(
        tmp_path,
        [
            {
                "id": "empty_question",
                "question": "   ",
                "expected_route": "rag",
            }
        ],
    )

    with pytest.raises(
        ValueError,
        match=(
            "field must not be empty: "
            "question"
        ),
    ):
        load_routing_benchmark(
            benchmark_path
        )


def test_real_routing_benchmark_loads():
    benchmark_path = Path(
        "evaluation/routing_benchmark.json"
    )

    examples = load_routing_benchmark(
        benchmark_path
    )

    assert len(examples) == 36
    assert len(
        {
            example.id
            for example in examples
        }
    ) == 36
    assert Counter(
        example.expected_route
        for example in examples
    ) == {
        "sql": 18,
        "rag": 18,
    }
    assert Counter(
        example.category
        for example in examples
    ) == {
        "direct_metric_lookup": 3,
        "company_ticker_filtering": 2,
        "fiscal_year_filtering": 2,
        "multi_company_comparison": 2,
        "multi_year_comparison": 2,
        "arithmetic_change": 3,
        "ordering_ranking": 2,
        "aggregation": 2,
        "risk_disclosures": 3,
        "strategy": 3,
        "management_commentary": 3,
        "competitive_factors": 2,
        "qualitative_explanations": 3,
        "business_drivers": 2,
        "filing_narrative": 2,
    }
