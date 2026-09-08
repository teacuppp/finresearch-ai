from typing import List


def _find_split_position(
    text: str,
    start: int,
    target_end: int,
    min_chunk_size: int,
) -> int:
    if target_end >= len(text):
        return len(text)

    search_start = min(
        start + min_chunk_size,
        target_end,
    )

    separators = (
        "\n\n",
        "\n",
        ". ",
    )

    for separator in separators:
        position = text.rfind(
            separator,
            search_start,
            target_end,
        )

        if position != -1:
            return position + len(separator)

    return target_end


def split_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[str]:
    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be positive"
        )

    if chunk_overlap < 0:
        raise ValueError(
            "chunk_overlap must not be negative"
        )

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size"
        )

    chunks: List[str] = []

    start = 0
    text_length = len(text)

    min_chunk_size = max(
        1,
        int(chunk_size * 0.6),
    )

    while start < text_length:
        target_end = min(
            start + chunk_size,
            text_length,
        )

        end = _find_split_position(
            text=text,
            start=start,
            target_end=target_end,
            min_chunk_size=min_chunk_size,
        )

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        next_start = max(
            0,
            end - chunk_overlap,
        )

        # Defensive guard against non-progress.
        if next_start <= start:
            next_start = end

        start = next_start

    return chunks