from pathlib import Path

import pytest

from app.evaluation.models import RoutingBenchmarkExample
from scripts import evaluate_routing


@pytest.mark.parametrize(
    ("argv", "expected_path"),
    [
        (
            [],
            evaluate_routing.BENCHMARK_PATH,
        ),
        (
            [
                "--benchmark",
                "evaluation/routing_challenge_v1.json",
            ],
            Path(
                "evaluation/routing_challenge_v1.json"
            ),
        ),
    ],
)
def test_main_uses_selected_benchmark_without_live_router(
    argv,
    expected_path,
    monkeypatch,
):
    loaded_paths = []

    def fake_load_routing_benchmark(path):
        loaded_paths.append(path)
        return [
            RoutingBenchmarkExample(
                id="example",
                question="What was the revenue?",
                expected_route="sql",
            )
        ]

    class FakeRouter:
        def route(self, question):
            return "sql"

    monkeypatch.setattr(
        evaluate_routing,
        "load_routing_benchmark",
        fake_load_routing_benchmark,
    )
    monkeypatch.setattr(
        evaluate_routing,
        "LLMQuestionRouter",
        FakeRouter,
    )

    evaluate_routing.main(argv)

    assert loaded_paths == [expected_path]
