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

        relevant_source_data = (
            item["relevant_sources"]
        )

        if not relevant_source_data:
            raise ValueError(
                f"Benchmark example has no relevant sources: "
                f"{example_id}"
            )

        relevant_sources: list[
            RelevantSource
        ] = []

        for source in relevant_source_data:
            row_terms = source["row_terms"]

            if not row_terms:
                raise ValueError(
                    f"Relevant source has no row terms: "
                    f"{example_id}"
                )

            preceding_terms = source.get(
                "preceding_terms",
                [],
            )

            preceding_line_window = source.get(
                "preceding_line_window",
                8,
            )

            if preceding_line_window < 0:
                raise ValueError(
                    f"preceding_line_window must not "
                    f"be negative: {example_id}"
                )

            relevant_sources.append(
                RelevantSource(
                    document=source["document"],
                    page=source["page"],
                    row_terms=row_terms,
                    preceding_terms=preceding_terms,
                    preceding_line_window=(
                        preceding_line_window
                    ),
                )
            )

        examples.append(
            RetrievalExample(
                id=example_id,
                question=item["question"],
                ticker=item["ticker"],
                fiscal_year=item["fiscal_year"],
                relevant_sources=(
                    relevant_sources
                ),
            )
        )

    return examples