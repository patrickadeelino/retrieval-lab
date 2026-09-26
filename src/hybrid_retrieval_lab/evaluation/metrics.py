from __future__ import annotations

from math import log2


def recall_at(ids: list[str], grades: dict[str, int], k: int) -> float:
    relevant = {chunk_id for chunk_id, grade in grades.items() if grade >= 1}
    return sum(chunk_id in relevant for chunk_id in ids[:k]) / len(relevant) if relevant else 0.0


def ndcg_at(ids: list[str], grades: dict[str, int], k: int) -> float:
    dcg = sum((2 ** grades.get(chunk_id, 0) - 1) / log2(rank + 1) for rank, chunk_id in enumerate(ids[:k], start=1))
    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg = sum((2**grade - 1) / log2(rank + 1) for rank, grade in enumerate(ideal, start=1))
    return dcg / idcg if idcg else 0.0
