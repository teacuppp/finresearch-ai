import json
from pathlib import Path

from app.evaluation.models import (
    RelevantSource,
    RetrievalExample,
)


def load_retrieval_benchmark(
    path: str | Path,
) -> list[RetrievalExample]:
    benchmark_path = Path(path)

    data = json.loads(
        benchmark_path.read_text(
            encoding="utf-8",
        )
    )

    if not isinstance(data, list):
        raise ValueError(
            "Retrieval benchmark must be a JSON list"
        )

    examples: list[RetrievalExample] = []
    seen_ids: set[str] = set()

    for item in data:
        example_id = item["id"]

        if example_id in seen_ids:
            raise ValueError(
                f"Duplicate benchmark id: {example_id}"
            )

        seen_ids.add(example_id)

        if not item["relevant_sources"]:
            raise ValueError(
                f"Benchmark example has no relevant sources: "
                f"{example_id}"
            )

        relevant_sources = [
            RelevantSource(
                document=source["document"],
                page=source["page"],
                chunk_index=source["chunk_index"],
            )
            for source in item["relevant_sources"]
        ]

        examples.append(
            RetrievalExample(
                id=example_id,
                question=item["question"],
                ticker=item["ticker"],
                fiscal_year=item["fiscal_year"],
                relevant_sources=relevant_sources,
            )
        )

    return examples