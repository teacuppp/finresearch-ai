from app.evaluation.retrieval_metrics import (
    hit_at_k,
    mean,
    reciprocal_rank,
)


def test_hit_at_k():
    assert hit_at_k([1], 1) == 1.0
    assert hit_at_k([2], 1) == 0.0
    assert hit_at_k([2], 3) == 1.0
    assert hit_at_k([], 5) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank([1]) == 1.0
    assert reciprocal_rank([2]) == 0.5
    assert reciprocal_rank([3]) == 1 / 3
    assert reciprocal_rank([]) == 0.0


def test_reciprocal_rank_uses_first_relevant_result():
    assert reciprocal_rank([4, 2, 5]) == 0.5


def test_mean():
    assert mean([1.0, 0.5, 0.0]) == 0.5
    assert mean([]) == 0.0