import json
from pathlib import Path

import pytest

from app.evaluation.benchmark import (
    load_retrieval_benchmark,
)


def test_load_retrieval_benchmark(
    tmp_path,
):
    benchmark_path = (
        tmp_path
        / "benchmark.json"
    )

    benchmark_path.write_text(
        json.dumps(
            [
                {
                    "id": "aapl_services_2025",
                    "question": (
                        "How much revenue did Apple "
                        "generate from services in 2025?"
                    ),
                    "ticker": "AAPL",
                    "fiscal_year": 2025,
                    "relevant_sources": [
                        {
                            "document": "apple.pdf",
                            "page": 35,
                            "row_terms": [
                                "Services",
                                "109,158",
                            ],
                            "preceding_terms": [
                                "2025",
                            ],
                            "preceding_line_window": 8,
                        }
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    examples = load_retrieval_benchmark(
        benchmark_path
    )

    assert len(examples) == 1

    example = examples[0]

    assert (
        example.id
        == "aapl_services_2025"
    )

    assert example.ticker == "AAPL"
    assert example.fiscal_year == 2025

    assert len(
        example.relevant_sources
    ) == 1

    source = example.relevant_sources[0]

    assert source.document == "apple.pdf"
    assert source.page == 35

    assert source.row_terms == [
        "Services",
        "109,158",
    ]

    assert source.preceding_terms == [
        "2025",
    ]

    assert (
        source.preceding_line_window
        == 8
    )


def test_default_preceding_line_window(
    tmp_path,
):
    benchmark_path = (
        tmp_path
        / "benchmark.json"
    )

    data = [
        {
            "id": "default_window",
            "question": "Question",
            "ticker": "AAPL",
            "fiscal_year": 2025,
            "relevant_sources": [
                {
                    "document": "apple.pdf",
                    "page": 1,
                    "row_terms": [
                        "Revenue",
                        "100",
                    ],
                    "preceding_terms": [
                        "2025",
                    ],
                }
            ],
        }
    ]

    benchmark_path.write_text(
        json.dumps(data),
        encoding="utf-8",
    )

    examples = load_retrieval_benchmark(
        benchmark_path
    )

    assert (
        examples[0]
        .relevant_sources[0]
        .preceding_line_window
        == 8
    )


def test_rejects_duplicate_benchmark_ids(
    tmp_path,
):
    benchmark_path = (
        tmp_path
        / "benchmark.json"
    )

    data = [
        {
            "id": "duplicate",
            "question": "Question one",
            "ticker": "AAPL",
            "fiscal_year": 2025,
            "relevant_sources": [
                {
                    "document": "apple.pdf",
                    "page": 1,
                    "row_terms": [
                        "Revenue",
                        "100",
                    ],
                    "preceding_terms": [
                        "2025",
                    ],
                }
            ],
        },
        {
            "id": "duplicate",
            "question": "Question two",
            "ticker": "MSFT",
            "fiscal_year": 2025,
            "relevant_sources": [
                {
                    "document": "microsoft.pdf",
                    "page": 1,
                    "row_terms": [
                        "Revenue",
                        "200",
                    ],
                    "preceding_terms": [
                        "2025",
                    ],
                }
            ],
        },
    ]

    benchmark_path.write_text(
        json.dumps(data),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Duplicate benchmark id",
    ):
        load_retrieval_benchmark(
            benchmark_path
        )


def test_rejects_empty_relevant_sources(
    tmp_path,
):
    benchmark_path = (
        tmp_path
        / "benchmark.json"
    )

    data = [
        {
            "id": "missing_ground_truth",
            "question": "Question",
            "ticker": "AAPL",
            "fiscal_year": 2025,
            "relevant_sources": [],
        }
    ]

    benchmark_path.write_text(
        json.dumps(data),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="has no relevant sources",
    ):
        load_retrieval_benchmark(
            benchmark_path
        )


def test_rejects_empty_row_terms(
    tmp_path,
):
    benchmark_path = (
        tmp_path
        / "benchmark.json"
    )

    data = [
        {
            "id": "empty_row_terms",
            "question": "Question",
            "ticker": "AAPL",
            "fiscal_year": 2025,
            "relevant_sources": [
                {
                    "document": "apple.pdf",
                    "page": 35,
                    "row_terms": [],
                    "preceding_terms": [
                        "2025",
                    ],
                }
            ],
        }
    ]

    benchmark_path.write_text(
        json.dumps(data),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Relevant source has no row terms",
    ):
        load_retrieval_benchmark(
            benchmark_path
        )


def test_rejects_negative_preceding_line_window(
    tmp_path,
):
    benchmark_path = (
        tmp_path
        / "benchmark.json"
    )

    data = [
        {
            "id": "negative_window",
            "question": "Question",
            "ticker": "AAPL",
            "fiscal_year": 2025,
            "relevant_sources": [
                {
                    "document": "apple.pdf",
                    "page": 35,
                    "row_terms": [
                        "Revenue",
                        "100",
                    ],
                    "preceding_terms": [
                        "2025",
                    ],
                    "preceding_line_window": -1,
                }
            ],
        }
    ]

    benchmark_path.write_text(
        json.dumps(data),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=(
            "preceding_line_window "
            "must not be negative"
        ),
    ):
        load_retrieval_benchmark(
            benchmark_path
        )


def test_real_retrieval_benchmark_loads():
    benchmark_path = Path(
        "evaluation/retrieval_benchmark.json"
    )

    examples = load_retrieval_benchmark(
        benchmark_path
    )

    assert len(examples) == 4

    assert {
        example.id
        for example in examples
    } == {
        "aapl_revenue_2025",
        "aapl_services_revenue_2025",
        "msft_revenue_2025",
        "msft_operating_income_2025",
    }