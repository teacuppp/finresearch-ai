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