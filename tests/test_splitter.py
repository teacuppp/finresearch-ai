# import sys
# from pathlib import Path

# # 获取项目根目录（即当前文件的父目录的父目录）
# project_root = Path(__file__).parent.parent
# # 把项目根目录强行插进 Python 的搜索路径第一位
# sys.path.insert(0, str(project_root))

# 现在就可以正常导入了
import pytest

from app.rag.splitter import split_text


def test_split_text():
    text = "A" * 2500

    chunks = split_text(
        text,
        chunk_size=1000,
        chunk_overlap=200,
    )

    assert len(chunks) == 3
    assert len(chunks[0]) == 1000
    assert len(chunks[1]) == 1000
    assert len(chunks[2]) == 900


def test_short_text():
    text = "Hello FinResearch AI"

    chunks = split_text(
        text,
        chunk_size=1000,
        chunk_overlap=200,
    )

    assert len(chunks) == 1
    assert chunks[0] == text


def test_invalid_overlap():
    with pytest.raises(ValueError):
        split_text(
            "Hello",
            chunk_size=1000,
            chunk_overlap=1000,
        )


#chuncks
def test_split_text_prefers_line_boundary():
    text = (
        "A" * 60
        + "\n"
        + "B" * 60
    )

    chunks = split_text(
        text,
        chunk_size=100,
        chunk_overlap=0,
    )

    assert chunks[0] == "A" * 60


#测表格行不会从中间切：
def test_split_text_keeps_financial_row_together():
    text = (
        "A" * 65
        + "\n"
        + "Total net sales $ 416,161 $ 391,035 $ 383,285"
        + "\n"
        + "B" * 100
    )

    chunks = split_text(
        text,
        chunk_size=120,
        chunk_overlap=20,
    )

    assert any(
        "Total net sales $ 416,161"
        in chunk
        for chunk in chunks
    )


# #fallback
def test_split_text_falls_back_to_fixed_size():
    text = "A" * 250

    chunks = split_text(
        text,
        chunk_size=100,
        chunk_overlap=20,
    )

    assert len(chunks) > 1

    assert all(
        len(chunk) <= 100
        for chunk in chunks
    )



#参数校验
import pytest


def test_split_text_rejects_non_positive_chunk_size():
    with pytest.raises(
        ValueError,
        match="chunk_size must be positive",
    ):
        split_text(
            "example",
            chunk_size=0,
            chunk_overlap=0,
        )


def test_split_text_rejects_negative_overlap():
    with pytest.raises(
        ValueError,
        match="chunk_overlap must not be negative",
    ):
        split_text(
            "example",
            chunk_size=100,
            chunk_overlap=-1,
        )


def test_split_text_preserves_complete_lines():
    text = (
        "Header\n"
        + "A" * 80
        + "\n"
        + "Revenue 281,724 245,122\n"
        + "B" * 80
    )

    chunks = split_text(
        text,
        chunk_size=100,
        chunk_overlap=20,
    )

    assert any(
        "Revenue 281,724 245,122"
        in chunk
        for chunk in chunks
    )

def test_split_text_carries_year_header_into_table_chunk():
    text = (
        "2025 2024 2023\n"
        "Revenue 281,724 245,122 211,915\n"
        "Gross margin 193,893 171,008 146,052\n"
        "Operating expenses 65,365 61,575 57,246\n"
        "Operating income 128,528 109,433 88,806\n"
        "Net income 101,832 88,136 72,361\n"
    )

    chunks = split_text(
        text,
        chunk_size=110,
        chunk_overlap=20,
    )

    matching_chunks = [
        chunk
        for chunk in chunks
        if (
            "Operating income"
            in chunk
        )
    ]

    assert matching_chunks

    assert any(
        "2025 2024 2023"
        in chunk
        for chunk in matching_chunks
    )


def test_split_text_preserves_financial_table_header_context():
    text = (
        "Products and Services Performance\n"
        "The following table shows net sales "
        "by category for 2025, 2024 and 2023\n"
        "2025 Change 2024 Change 2023\n"
        "iPhone 209,586 201,183 200,583\n"
        "Mac 33,708 29,984 29,357\n"
        "iPad 28,023 26,694 28,300\n"
        "Services 109,158 96,169 85,200\n"
        "Total net sales 416,161 391,035 383,285\n"
    )

    chunks = split_text(
        text,
        chunk_size=180,
        chunk_overlap=20,
    )

    total_sales_chunks = [
        chunk
        for chunk in chunks
        if (
            "Total net sales"
            in chunk
            and "416,161"
            in chunk
        )
    ]

    assert total_sales_chunks

    assert any(
        "2025"
        in chunk
        for chunk in total_sales_chunks
    )

def test_comparison_heading_is_not_table_header():
    text = (
        "Fiscal Year 2025 Compared with "
        "Fiscal Year 2024\n"
        + "A" * 90
        + "\n"
        + "Revenue 281,724 245,122\n"
    )

    chunks = split_text(
        text,
        chunk_size=100,
        chunk_overlap=0,
    )

    revenue_chunks = [
        chunk
        for chunk in chunks
        if "Revenue 281,724" in chunk
    ]

    assert revenue_chunks

    assert all(
        not chunk.startswith(
            "Fiscal Year 2025 Compared"
        )
        for chunk in revenue_chunks
    )

#再验证 overlap 参数：
def test_zero_overlap_does_not_repeat_lines():
    text = (
        "line one\n"
        "line two\n"
        "line three\n"
        "line four\n"
    )

    chunks = split_text(
        text,
        chunk_size=20,
        chunk_overlap=0,
    )

    combined = "\n".join(
        chunks
    )

    assert combined.count(
        "line two"
    ) == 1

#最后验证 header 生命周期：
def test_table_header_expires_after_many_lines():
    lines = [
        "2025 2024 2023",
    ]

    lines.extend(
        f"Narrative line {index}"
        for index in range(20)
    )

    lines.append(
        "Revenue 281,724 245,122 211,915"
    )

    text = "\n".join(
        lines
    )

    chunks = split_text(
        text,
        chunk_size=120,
        chunk_overlap=0,
    )

    revenue_chunks = [
        chunk
        for chunk in chunks
        if "Revenue 281,724" in chunk
    ]

    assert revenue_chunks

    assert all(
        not chunk.startswith(
            "2025 2024 2023"
        )
        for chunk in revenue_chunks
    )