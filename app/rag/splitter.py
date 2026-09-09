import re
from typing import List


_YEAR_PATTERN = re.compile(
    r"\b(?:19|20)\d{2}\b"
)

DEFAULT_CHUNK_SIZE = 1000

# A detected table header should only remain active
# for a limited number of following non-empty lines.
TABLE_HEADER_MAX_AGE_LINES = 12


def _normalize_line(
    line: str,
) -> str:
    return " ".join(
        line.split()
    )


def _split_oversized_line(
    line: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    chunks = []

    start = 0

    while start < len(line):
        end = min(
            start + chunk_size,
            len(line),
        )

        chunks.append(
            line[start:end]
        )

        if end >= len(line):
            break

        start = (
            end
            - chunk_overlap
        )

    return chunks


def _prepare_lines(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    lines: list[str] = []

    for raw_line in text.splitlines():
        normalized_line = _normalize_line(
            raw_line
        )

        if not normalized_line:
            continue

        if (
            len(normalized_line)
            <= chunk_size
        ):
            lines.append(
                normalized_line
            )
            continue

        lines.extend(
            _split_oversized_line(
                line=normalized_line,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )

    return lines


def _looks_like_year_header(
    line: str,
) -> bool:
    normalized = _normalize_line(
        line
    )

    years = set(
        _YEAR_PATTERN.findall(
            normalized
        )
    )

    if len(years) < 2:
        return False

    lowered = normalized.casefold()

    # Narrative comparison headings are not
    # financial-table column headers.
    if (
        "compared with" in lowered
        or "compared to" in lowered
    ):
        return False

    # Long prose containing several years should
    # not become a table header.
    if len(normalized) > 180:
        return False

    # Table-introduction sentences such as:
    # "The following table shows ...:"
    # are context, not the actual column header.
    if normalized.endswith(
        (".", ":", ";")
    ):
        return False

    return True


def _looks_like_table_row(
    line: str,
) -> bool:
    normalized = _normalize_line(
        line
    )

    if not normalized:
        return False

    digit_count = sum(
        char.isdigit()
        for char in normalized
    )

    return digit_count >= 3


def _select_overlap_lines(
    lines: list[str],
    chunk_overlap: int,
) -> list[str]:
    if chunk_overlap <= 0:
        return []

    selected: list[str] = []
    selected_length = 0

    for line in reversed(lines):
        line_length = (
            len(line)
            + 1
        )

        if (
            selected_length
            + line_length
            > chunk_overlap
        ):
            break

        selected.append(
            line
        )

        selected_length += (
            line_length
        )

    selected.reverse()

    return selected


def _lines_length(
    lines: list[str],
) -> int:
    return sum(
        len(line) + 1
        for line in lines
    )


def split_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
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

    normalized_lines = [
        normalized
        for line in text.splitlines()
        if (
            normalized := _normalize_line(
                line
            )
        )
    ]

    # No usable structure exists. Preserve the
    # original fixed-size fallback behavior.
    if (
        len(normalized_lines) == 1
        and len(normalized_lines[0])
        > chunk_size
    ):
        return _split_oversized_line(
            line=normalized_lines[0],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    lines = _prepare_lines(
        text=text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    if not lines:
        return []

    chunks: list[str] = []

    current_lines: list[str] = []
    current_length = 0

    active_table_header: str | None = None
    table_header_age: int | None = None

    for line in lines:
        normalized_line = _normalize_line(
            line
        )

        if _looks_like_year_header(
            normalized_line
        ):
            active_table_header = (
                normalized_line
            )

            table_header_age = 0

        elif active_table_header is not None:
            assert table_header_age is not None

            table_header_age += 1

            if (
                table_header_age
                > TABLE_HEADER_MAX_AGE_LINES
            ):
                active_table_header = None
                table_header_age = None

        additional_length = (
            len(normalized_line)
            + 1
        )

        if (
            current_lines
            and current_length
            + additional_length
            > chunk_size
        ):
            chunks.append(
                "\n".join(
                    current_lines
                )
            )

            current_lines = (
                _select_overlap_lines(
                    current_lines,
                    chunk_overlap,
                )
            )

            current_length = (
                _lines_length(
                    current_lines
                )
            )

            should_carry_header = (
                active_table_header
                is not None
                and table_header_age
                is not None
                and table_header_age
                <= TABLE_HEADER_MAX_AGE_LINES
                and _looks_like_table_row(
                    normalized_line
                )
                and active_table_header
                not in current_lines
            )

            if should_carry_header:
                header_length = (
                    len(active_table_header)
                    + 1
                )

                # Remove old overlap lines if necessary
                # so header + incoming line still obey
                # chunk_size.
                while (
                    current_lines
                    and current_length
                    + header_length
                    + additional_length
                    > chunk_size
                ):
                    removed_line = (
                        current_lines.pop(0)
                    )

                    current_length -= (
                        len(removed_line)
                        + 1
                    )

                if (
                    header_length
                    + additional_length
                    <= chunk_size
                ):
                    current_lines.insert(
                        0,
                        active_table_header,
                    )

                    current_length += (
                        header_length
                    )

            # Even without a carried header,
            # overlap must never force the next
            # chunk beyond chunk_size.
            while (
                current_lines
                and current_length
                + additional_length
                > chunk_size
            ):
                removed_line = (
                    current_lines.pop(0)
                )

                current_length -= (
                    len(removed_line)
                    + 1
                )

        current_lines.append(
            normalized_line
        )

        current_length += (
            additional_length
        )

    if current_lines:
        chunks.append(
            "\n".join(
                current_lines
            )
        )

    return chunks