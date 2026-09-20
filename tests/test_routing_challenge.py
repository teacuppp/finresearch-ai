from collections import Counter
from pathlib import Path

from app.evaluation.routing_benchmark import (
    load_routing_benchmark,
)


CHALLENGE_PATH = Path(
    "evaluation/routing_challenge_v1.json"
)


def test_routing_challenge_v1_shape():
    examples = load_routing_benchmark(
        CHALLENGE_PATH
    )

    assert len(examples) == 24
    assert len(
        {
            example.id
            for example in examples
        }
    ) == 24
    assert Counter(
        example.expected_route
        for example in examples
    ) == {
        "sql": 12,
        "rag": 12,
    }


def test_routing_challenge_v1_category_distribution():
    examples = load_routing_benchmark(
        CHALLENGE_PATH
    )

    assert Counter(
        example.category
        for example in examples
    ) == {
        "ambiguous_surface_wording": 4,
        "filing_language_structured": 4,
        "metric_and_number_cues": 4,
        "quantitative_vs_explanatory": 4,
        "keyword_light_paraphrase": 4,
        "comparative_intent": 4,
    }

    for category in {
        example.category
        for example in examples
    }:
        assert Counter(
            example.expected_route
            for example in examples
            if example.category == category
        ) == {
            "sql": 2,
            "rag": 2,
        }
