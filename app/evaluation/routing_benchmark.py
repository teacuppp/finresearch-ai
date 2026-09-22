import json
from pathlib import Path
from typing import cast

from app.agent.state import Route
from app.evaluation.models import (
    RoutingBenchmarkExample,
)


def _required_text(
    item: dict,
    field: str,
    index: int,
) -> str:
    if field not in item:
        raise ValueError(
            "Routing benchmark item "
            f"{index} is missing required field: {field}"
        )

    value = item[field]

    if not isinstance(value, str):
        raise ValueError(
            "Routing benchmark field must be a string: "
            f"{field} at item {index}"
        )

    if not value.strip():
        raise ValueError(
            "Routing benchmark field must not be empty: "
            f"{field} at item {index}"
        )

    return value


def load_routing_benchmark(
    path: str | Path,
) -> list[RoutingBenchmarkExample]:
    benchmark_path = Path(path)

    data = json.loads(
        benchmark_path.read_text(
            encoding="utf-8",
        )
    )

    if not isinstance(data, list):
        raise ValueError(
            "Routing benchmark must be a JSON list"
        )

    examples: list[RoutingBenchmarkExample] = []
    seen_ids: set[str] = set()

    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(
                "Routing benchmark items must be JSON objects: "
                f"item {index}"
            )

        example_id = _required_text(
            item,
            "id",
            index,
        )

        if example_id in seen_ids:
            raise ValueError(
                f"Duplicate benchmark id: {example_id}"
            )

        seen_ids.add(example_id)

        question = _required_text(
            item,
            "question",
            index,
        )
        expected_route_value = _required_text(
            item,
            "expected_route",
            index,
        )

        if expected_route_value not in {"rag", "sql"}:
            raise ValueError(
                "Invalid expected route for benchmark example "
                f"{example_id}: {expected_route_value!r}"
            )

        category = item.get("category")

        if category is not None:
            if not isinstance(category, str) or not category.strip():
                raise ValueError(
                    "Routing benchmark category must be a "
                    f"non-empty string: {example_id}"
                )

        examples.append(
            RoutingBenchmarkExample(
                id=example_id,
                question=question,
                expected_route=cast(
                    Route,
                    expected_route_value,
                ),
                category=category,
            )
        )

    return examples
