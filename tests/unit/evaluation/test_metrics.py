from __future__ import annotations

import pytest

from hybrid_retrieval_lab.evaluation.metrics import ndcg_at, recall_at


def test_recall_counts_both_relevance_grades_and_uses_all_relevant_as_denominator() -> None:
    grades = {"a": 2, "b": 1, "c": 0, "d": 2}
    assert recall_at(["c", "b", "a"], grades, 2) == pytest.approx(1 / 3)
    assert recall_at(["c", "b", "a"], grades, 3) == pytest.approx(2 / 3)
    assert recall_at(["c"], {"c": 0}, 5) == 0.0


def test_ndcg_rewards_grade_two_and_earlier_positions() -> None:
    grades = {"a": 2, "b": 1, "c": 0}
    assert ndcg_at(["a", "b"], grades, 2) == pytest.approx(1.0)
    assert ndcg_at(["b", "a"], grades, 2) < 1.0
    assert ndcg_at(["a", "c"], grades, 2) < ndcg_at(["a", "b"], grades, 2)
    assert ndcg_at(["c"], {"c": 0}, 5) == 0.0
