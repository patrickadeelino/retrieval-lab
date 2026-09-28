from __future__ import annotations

import json

import pytest

from hybrid_retrieval_lab.evaluation import runner


def test_evaluation_records_rankings_metrics_warmups_and_input_hashes(evaluation_case) -> None:
    case = evaluation_case
    report = runner.evaluate(*case.paths, repeats=2, reviewed_qrels=True)
    assert not report["provisional_qrels"]
    assert all(len(report["inputs"][key + "_sha256"]) == 64 for key in ("corpus", "queries", "qrels"))
    query = report["queries"][0]
    assert query["coverage_before_colbert_at_10"] == 1.0
    assert query["missing_before_colbert"] == []
    for name, result in query["strategies"].items():
        assert case.calls.count(name) == 3  # One warmup and two timed runs.
        assert [hit["id"] for hit in result["ranking"]] == case.rankings[name]
        assert len(result["latency_ms"]["runs"]) == 2
        assert result["metrics"]["recall_at_10"] == 1.0
    assert (
        query["strategies"]["hybrid_colbert"]["metrics"]["ndcg_at_5"]
        > query["strategies"]["hybrid"]["metrics"]["ndcg_at_5"]
    )


@pytest.mark.parametrize("invalid", ["missing", "duplicate", "out_of_range", "boolean", "no_relevant"])
def test_invalid_judgments_fail_before_connecting_to_qdrant(evaluation_case, monkeypatch, invalid) -> None:
    case = evaluation_case
    rows = [json.loads(line) for line in case.qrels.read_text().splitlines()]
    if invalid == "missing":
        rows.pop()
    elif invalid == "duplicate":
        rows.append(rows[0])
    elif invalid == "out_of_range":
        rows[0]["grade"] = 3
    elif invalid == "boolean":
        rows[0]["grade"] = True
    else:
        for row in rows:
            row["grade"] = 0
    case.qrels.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def forbidden_client(**kwargs):
        pytest.fail("Invalid judgments must fail before opening a Qdrant connection")

    monkeypatch.setattr(runner, "QdrantClient", forbidden_client)
    with pytest.raises(ValueError):
        runner.evaluate(*case.paths)


def test_colbert_cannot_change_candidate_set(evaluation_case) -> None:
    case = evaluation_case
    case.rankings["hybrid_colbert"] = ["a", "b"]
    with pytest.raises(RuntimeError, match="changed the candidate set"):
        runner.evaluate(*case.paths, repeats=1)


def test_evaluation_requires_colbert_pool_to_match_hybrid_ranking(evaluation_case) -> None:
    case = evaluation_case
    case.colbert_candidates[:] = ["a", "b", "c"]
    with pytest.raises(RuntimeError, match="candidate pool differs from hybrid"):
        runner.evaluate(*case.paths, repeats=1)
