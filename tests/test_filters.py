from app.api.filters import build_metadata_filter


def test_build_metadata_filter_returns_none():
    result = build_metadata_filter()

    assert result is None


def test_build_metadata_filter_single_condition():
    result = build_metadata_filter(
        ticker="MSFT"
    )

    assert result == {
        "ticker": {
            "$eq": "MSFT"
        }
    }


def test_build_metadata_filter_multiple_conditions():
    result = build_metadata_filter(
        ticker="MSFT",
        fiscal_year=2025,
    )

    assert result == {
        "$and": [
            {
                "ticker": {
                    "$eq": "MSFT"
                }
            },
            {
                "fiscal_year": {
                    "$eq": 2025
                }
            },
        ]
    }


#.     pytest tests/test_filters.py -v

# test_filters.py
# → filter 构造是否正确

# test_rag_api.py
# → HTTP 参数有没有正确传进去

# test_retrieval.py
# → where 有没有真正限制检索

# test_vector_store.py
# → Chroma filtering 是否工作