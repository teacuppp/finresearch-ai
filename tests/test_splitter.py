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


#fallback
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