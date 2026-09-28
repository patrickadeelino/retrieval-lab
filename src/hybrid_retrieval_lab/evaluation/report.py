from __future__ import annotations

from dataclasses import dataclass
from html import escape
from statistics import median

from hybrid_retrieval_lab.evaluation.models import EvaluatedStrategy, PilotReport, QueryReport

LABELS = {"bm25": "BM25", "dense": "Dense · E5", "hybrid": "Hybrid · RRF", "hybrid_colbert": "Hybrid + ColBERT"}


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _metric(value: float) -> str:
    return f"{value:.3f}"


def _one_decimal(value: float) -> str:
    return f"{value:.1f}"


def _e(value: object) -> str:
    return escape(str(value), quote=True)


def _tip(label: str, description: str) -> str:
    return (
        f'<span class="metric-tip" tabindex="0" aria-label="{_e(label)}: {_e(description)}">'
        f'{_e(label)} <span class="tip-icon" aria-hidden="true">ⓘ</span>'
        f'<span class="tip-body" role="tooltip">{_e(description)}</span></span>'
    )


def _signed(value: float, percentage_points: bool = False) -> str:
    if abs(value) <= 1e-9:
        return "—"
    scaled = value * 100 if percentage_points else value
    unit = " pp" if percentage_points else ""
    return f"{scaled:+.1f}" + unit if percentage_points else f"{scaled:+.3f}"


def _delta_cell(value: float, percentage_points: bool = False) -> str:
    kind = "positive" if value > 1e-9 else "negative" if value < -1e-9 else "neutral"
    return f'<td class="delta {kind}">{_signed(value, percentage_points)}</td>'


def _coverage_before(query: QueryReport) -> float:
    coverage = query.get("coverage_before_colbert_at_10")
    if coverage is None:
        coverage = query.get("coverage_before_colbert_at_20")
    if coverage is None:
        raise ValueError("Query report is missing pre-ColBERT coverage")
    return coverage


def _render_strategy(label: str, result: EvaluatedStrategy) -> tuple[str, str]:
    metrics = result["metrics"]
    latency = result["latency_ms"]
    summary_row = (
        f'<tr><th scope="row">{_e(label)}</th><td>{_pct(metrics["recall_at_5"])}</td>'
        f"<td>{_pct(metrics['recall_at_10'])}</td><td>{_metric(metrics['ndcg_at_5'])}</td>"
        f"<td>{_metric(metrics['ndcg_at_10'])}</td><td>{_metric(latency['median'])} ms</td></tr>"
    )
    ranking = []
    for hit in result["ranking"][:10]:
        grade = hit["grade"]
        ranking.append(
            f'<li class="hit grade-{grade}"><div class="hit-head"><span class="rank">#{hit["rank"]}</span>'
            f'<strong>{_e(hit["id"])}</strong><span class="grade">Grade {grade}</span></div>'
            f"<p>{_e(hit['text'])}</p><small>Score {_metric(hit['score'])} · "
            f'<a href="{_e(hit["source_url"])}" target="_blank" rel="noopener">{_e(hit["title"])}</a></small></li>'
        )
    strategy_card = (
        f'<details class="strategy"><summary><span>{_e(label)}</span>'
        f"<span>top 10 · {_pct(metrics['recall_at_10'])} recall</span></summary>"
        f'<ol class="hits">{"".join(ranking)}</ol></details>'
    )
    return summary_row, strategy_card


def _render_query_strategies(
    results: dict[str, EvaluatedStrategy],
) -> tuple[list[str], list[str]]:
    summary_rows: list[str] = []
    strategy_cards: list[str] = []
    for strategy, label in LABELS.items():
        summary_row, strategy_card = _render_strategy(label, results[strategy])
        summary_rows.append(summary_row)
        strategy_cards.append(strategy_card)
    return summary_rows, strategy_cards


@dataclass(frozen=True)
class RenderedQuery:
    query_id: str
    section: str
    candidate_row: str
    impact_row: str
    candidate_coverage: dict[str, float]
    fusion_lost_relevant_chunk: bool
    impact_outcomes: dict[str, str]


def _render_rank_changes(before: list[str], after: list[str], grades: dict[str, int]) -> str:
    rows = []
    for index, chunk_id in enumerate(after, 1):
        previous_rank = before.index(chunk_id) + 1
        delta = previous_rank - index
        movement = f"↑ {delta}" if delta > 0 else f"↓ {abs(delta)}" if delta < 0 else "—"
        rows.append(
            f"<tr><td>{_e(chunk_id)}</td><td>{grades.get(chunk_id, 0)}</td>"
            f"<td>{previous_rank}</td><td>{index}</td><td>{movement}</td></tr>"
        )
    return "".join(rows)


def _render_query_impact(query_id: str, results: dict[str, EvaluatedStrategy]) -> tuple[str, dict[str, str]]:
    deltas = {
        metric: results["hybrid_colbert"]["metrics"][metric] - results["hybrid"]["metrics"][metric]
        for metric in ("ndcg_at_5", "recall_at_10")
    }
    outcomes = {
        metric: "win" if delta > 1e-9 else "loss" if delta < -1e-9 else "tie" for metric, delta in deltas.items()
    }
    latency_ratio = results["hybrid_colbert"]["latency_ms"]["median"] / results["hybrid"]["latency_ms"]["median"]
    row = (
        f'<tr><th scope="row"><a href="#{_e(query_id)}">{_e(query_id)}</a></th>'
        + _delta_cell(deltas["ndcg_at_5"])
        + _delta_cell(deltas["recall_at_10"], True)
        + f"<td>{_one_decimal(latency_ratio)}×</td></tr>"
    )
    return row, outcomes


def _render_query(
    query: QueryReport, metric_tips: dict[str, str], retrieval_limit: int, rerank_limit: int
) -> RenderedQuery:
    results = query["strategies"]
    summary_rows, strategy_cards = _render_query_strategies(results)
    before = results["hybrid_colbert"]["inspection"]["candidate_ids_before"]
    after = [row["id"] for row in results["hybrid_colbert"]["ranking"]]
    relevant_ids = {item["id"] for item in query["relevant"]}
    bm25_ids = {hit["id"] for hit in results["bm25"]["ranking"][:retrieval_limit]}
    dense_ids = {hit["id"] for hit in results["dense"]["ranking"][:retrieval_limit]}
    candidate_coverage = {
        "bm25": len(relevant_ids & bm25_ids) / len(relevant_ids),
        "dense": len(relevant_ids & dense_ids) / len(relevant_ids),
        "union": len(relevant_ids & (bm25_ids | dense_ids)) / len(relevant_ids),
        "rrf": len(relevant_ids & set(before)) / len(relevant_ids),
    }
    candidate_cells = (
        f"<td>{_pct(candidate_coverage['bm25'])}</td><td>{_pct(candidate_coverage['dense'])}</td>"
        f"<td>{_pct(candidate_coverage['union'])}</td><td>{_pct(candidate_coverage['rrf'])}</td>"
    )
    candidate_row = (
        f'<tr><th scope="row"><a href="#{_e(query["id"])}">{_e(query["id"])}</a></th>' + candidate_cells + "</tr>"
    )
    impact_row, impact_outcomes = _render_query_impact(query["id"], results)
    grades = {item["id"]: item["grade"] for item in query["relevant"]}
    moves = _render_rank_changes(before, after, grades)
    relevant_chips = "".join(
        f'<span class="chip">{_e(item["id"])} · {item["grade"]}</span>' for item in query["relevant"]
    )
    missing = query["missing_before_colbert"]
    missing_text = (
        ", ".join(missing) if missing else f"No relevant chunks were excluded from the {rerank_limit} candidates."
    )
    section = (
        f'<section class="query" id="{_e(query["id"])}"><div class="eyebrow">{_e(query["id"])} · information need</div>'
        f'<h2>{_e(query["query"])}</h2><p class="need">{_e(query["information_need"])}</p>'
        f'<div class="callout"><strong>Coverage before ColBERT: {_pct(_coverage_before(query))}</strong>'
        f'<span>Missing: {_e(missing_text)}</span></div><div class="chips">{relevant_chips}</div>'
        f'<div class="table-wrap"><table><caption>Strategy comparison</caption><thead><tr><th>Strategy</th>'
        f"<th>{metric_tips['recall_at_5']}</th><th>{metric_tips['recall_at_10']}</th>"
        f"<th>{metric_tips['ndcg_at_5']}</th><th>{metric_tips['ndcg_at_10']}</th><th>Median</th>"
        f"</tr></thead><tbody>{''.join(summary_rows)}</tbody></table></div>"
        f'<h3>Returned chunks</h3><div class="strategies">{"".join(strategy_cards)}</div>'
        f'<details class="moves"><summary>View rank changes for the {rerank_limit} candidates</summary>'
        f'<div class="table-wrap"><table><thead><tr><th>Chunk</th><th>Grade</th><th>RRF</th>'
        f"<th>ColBERT</th><th>Change</th></tr></thead><tbody>{moves}</tbody></table></div></details></section>"
    )
    return RenderedQuery(
        query_id=query["id"],
        section=section,
        candidate_row=candidate_row,
        impact_row=impact_row,
        candidate_coverage=candidate_coverage,
        fusion_lost_relevant_chunk=candidate_coverage["union"] > candidate_coverage["rrf"],
        impact_outcomes=impact_outcomes,
    )


def _impact_count(rendered: list[RenderedQuery], metric: str, outcome: str) -> int:
    return sum(query.impact_outcomes[metric] == outcome for query in rendered)


def _mean_coverage(rendered: list[RenderedQuery], stage: str) -> float:
    return sum(query.candidate_coverage[stage] for query in rendered) / len(rendered)


def _mean_metric(queries: list[QueryReport], strategy: str, metric: str) -> float:
    return sum(query["strategies"][strategy]["metrics"][metric] for query in queries) / len(queries)


def _strategy_averages(queries: list[QueryReport]) -> dict[str, dict[str, float]]:
    return {
        strategy: {
            "recall_at_5": _mean_metric(queries, strategy, "recall_at_5"),
            "recall_at_10": _mean_metric(queries, strategy, "recall_at_10"),
            "ndcg_at_5": _mean_metric(queries, strategy, "ndcg_at_5"),
            "ndcg_at_10": _mean_metric(queries, strategy, "ndcg_at_10"),
        }
        for strategy in LABELS
    }


def _strategy_median_latency(queries: list[QueryReport], strategy: str) -> float:
    return median(query["strategies"][strategy]["latency_ms"]["median"] for query in queries)


def _strategy_median_latencies(queries: list[QueryReport]) -> dict[str, float]:
    return {strategy: _strategy_median_latency(queries, strategy) for strategy in LABELS}


def _strategy_top_one_count(queries: list[QueryReport], strategy: str) -> int:
    return sum(query["strategies"][strategy]["ranking"][0]["grade"] == 2 for query in queries)


def _strategy_top_one_counts(queries: list[QueryReport]) -> dict[str, int]:
    return {strategy: _strategy_top_one_count(queries, strategy) for strategy in LABELS}


def render_html(data: PilotReport) -> str:
    runtime = data["config"].get("runtime", {})
    revision = runtime.get("git_commit", "unknown")
    short_revision = revision[:12] if revision != "unknown" else revision
    provenance = (
        f"Source revision {_e(short_revision)} ({_e(runtime.get('git_tree_state', 'unknown'))}); "
        f"Python {_e(runtime.get('python', 'unknown'))}; uv {_e(runtime.get('uv', 'unknown'))}; "
        f"Qdrant {_e(runtime.get('qdrant', 'unknown'))}; uv.lock SHA-256 {_e(runtime.get('uv_lock_sha256', 'unknown'))}."
    )
    metric_tips = {
        "recall_at_5": _tip(
            "Recall@5",
            "Fraction of grade-1/2 chunks retrieved in the top 5. Higher means more relevant chunks found.",
        ),
        "recall_at_10": _tip(
            "Recall@10",
            "Fraction of grade-1/2 chunks retrieved in the top 10. Higher means more relevant chunks found.",
        ),
        "ndcg_at_5": _tip(
            "nDCG@5",
            "Top-5 ordering quality: rewards grade 2 and higher positions. Ranges from 0 to 1; 1 is the ideal ranking.",
        ),
        "ndcg_at_10": _tip(
            "nDCG@10",
            "Top-10 ordering quality: rewards grade 2 and higher positions. Ranges from 0 to 1; 1 is the ideal ranking.",
        ),
    }
    retrieval_limit = data["config"]["candidates_per_strategy"]
    rerank_limit = data["config"]["colbert_candidates"]
    union_limit = retrieval_limit * 2
    rendered_queries = [_render_query(query, metric_tips, retrieval_limit, rerank_limit) for query in data["queries"]]
    sections = [query.section for query in rendered_queries]
    lost_after_fusion = [query.query_id for query in rendered_queries if query.fusion_lost_relevant_chunk]
    candidate_rows = [query.candidate_row for query in rendered_queries]
    impact_rows = [query.impact_row for query in rendered_queries]
    impact_counts = {
        "ndcg_at_5": {
            "win": _impact_count(rendered_queries, "ndcg_at_5", "win"),
            "tie": _impact_count(rendered_queries, "ndcg_at_5", "tie"),
            "loss": _impact_count(rendered_queries, "ndcg_at_5", "loss"),
        },
        "recall_at_10": {
            "win": _impact_count(rendered_queries, "recall_at_10", "win"),
            "tie": _impact_count(rendered_queries, "recall_at_10", "tie"),
            "loss": _impact_count(rendered_queries, "recall_at_10", "loss"),
        },
    }
    query_count = len(rendered_queries)
    source_count = data["config"].get("source_chunk_count", data["config"]["point_count"])
    skipped_ids = data["config"].get("skipped_chunk_ids", [])
    skipped_notice = (
        f'<p class="notice">{len(skipped_ids)} chunks were skipped for exceeding the E5 window: {_e(", ".join(skipped_ids))}. They remain in the evaluation relevance denominator.</p>'
        if skipped_ids
        else ""
    )
    judgment_status = "Author-reviewed judgments." if not data["provisional_qrels"] else "Provisional judgments."
    averages = _strategy_averages(data["queries"])
    latency = _strategy_median_latencies(data["queries"])
    top_one = _strategy_top_one_counts(data["queries"])
    overview = "".join(
        f'<div class="overview-card"><span>{_e(label)}</span><strong>{_metric(averages[key]["ndcg_at_5"])}</strong>'
        f'<small>{metric_tips["ndcg_at_5"]} mean</small><div class="card-line">{metric_tips["recall_at_10"]} <b>{_pct(averages[key]["recall_at_10"])}</b></div>'
        f'<div class="card-line">Grade 2 at rank 1 <b>{top_one[key]}/{query_count}</b></div>'
        f'<div class="card-line">Typical latency <b>{_one_decimal(latency[key])} ms</b></div></div>'
        for key, label in LABELS.items()
    )
    comparison = "".join(
        f'<tr><th scope="row">{_e(label)}</th><td>{_metric(averages[key]["ndcg_at_5"])}</td>'
        f"<td>{_metric(averages[key]['ndcg_at_10'])}</td><td>{_pct(averages[key]['recall_at_5'])}</td>"
        f"<td>{_pct(averages[key]['recall_at_10'])}</td><td>{top_one[key]}/{query_count}</td>"
        f"<td>{_one_decimal(latency[key])} ms</td><td>{_one_decimal(latency[key] / latency['bm25'])}×</td></tr>"
        for key, label in LABELS.items()
    )
    fusion_loss_note = (
        "The RRF cutoff excludes relevant candidates for " + ", ".join(lost_after_fusion) + "."
        if lost_after_fusion
        else "No relevant chunks found in the union were excluded by the RRF cutoff."
    )
    candidate_average = (
        f"<td><strong>{_pct(_mean_coverage(rendered_queries, 'bm25'))}</strong></td>"
        f"<td><strong>{_pct(_mean_coverage(rendered_queries, 'dense'))}</strong></td>"
        f"<td><strong>{_pct(_mean_coverage(rendered_queries, 'union'))}</strong></td>"
        f"<td><strong>{_pct(_mean_coverage(rendered_queries, 'rrf'))}</strong></td>"
    )
    impact_explanation = (
        "RRF and ColBERT contain the same 10 IDs, so Recall@10 ties by construction. "
        "nDCG@5 shows whether reranking improved top-ranked results."
        if rerank_limit == 10
        else "nDCG@5 describes top-ranked ordering. Recall@10 measures relevant chunks retained in the top ten."
    )
    impact_summary = (
        f'<tr><th scope="row">nDCG@5</th><td>{impact_counts["ndcg_at_5"]["win"]}</td>'
        f"<td>{impact_counts['ndcg_at_5']['tie']}</td><td>{impact_counts['ndcg_at_5']['loss']}</td></tr>"
        f'<tr><th scope="row">Recall@10</th><td>{impact_counts["recall_at_10"]["win"]}</td>'
        f"<td>{impact_counts['recall_at_10']['tie']}</td><td>{impact_counts['recall_at_10']['loss']}</td></tr>"
    )
    links = "".join(f'<a href="#{_e(q["id"])}">{_e(q["id"])} · {_e(q["query"])}</a>' for q in data["queries"])
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hybrid Retrieval Lab · Pilot evaluation</title><style>
:root{{--ink:#172229;--muted:#5d6d73;--paper:#f5f7f4;--surface:#fff;--line:#dbe3df;--accent:#155f59;--soft:#e0f2eb;--amber:#a86716}}
*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.55 system-ui,-apple-system,sans-serif}}
a{{color:var(--accent)}}.wrap{{max-width:1180px;margin:auto;padding:32px 24px 80px}}header{{background:#173b38;color:#fff;padding:42px 0 48px}}
header .wrap{{padding-top:0;padding-bottom:0}}header p{{color:#d0e4dd;max-width:760px}}h1{{font-size:clamp(2rem,5vw,3.4rem);line-height:1.05;margin:8px 0 18px}}
h2{{font-size:clamp(1.45rem,3vw,2.2rem);line-height:1.18;margin:8px 0}}h3{{margin:32px 0 12px}}.eyebrow{{text-transform:uppercase;letter-spacing:.13em;font-size:.75rem;font-weight:800;color:#8bd4bd}}
.meta{{display:flex;gap:12px;flex-wrap:wrap;margin-top:24px}}.meta span{{border:1px solid #547a70;padding:6px 11px;border-radius:999px;font-size:.82rem}}
.notice{{background:#fff8e9;border:1px solid #efd9a7;border-left:5px solid var(--amber);padding:14px 18px;border-radius:8px;margin:24px 0}}
.overview{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:24px 0}}.overview-card{{background:var(--surface);padding:18px;border:1px solid var(--line);border-radius:12px;display:flex;flex-direction:column}}
.overview-card strong{{font-size:2.1rem;color:var(--accent)}}.overview-card small{{color:var(--muted)}}.card-line{{display:flex;justify-content:space-between;gap:10px;border-top:1px solid var(--line);margin-top:9px;padding-top:8px;font-size:.82rem}}.card-line b{{white-space:nowrap}}.intro,.method{{color:var(--muted)}}.method{{font-size:.84rem;margin:14px 0 26px}}.comparison{{margin-top:8px}}.comparison th:first-child{{min-width:150px}}.comparison td{{white-space:nowrap}}.takeaways{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin:18px 0}}.takeaways>div{{background:#eaf2ed;border:1px solid #cbded3;border-radius:12px;padding:16px}}.takeaways strong{{color:var(--accent)}}.takeaways p{{font-size:.9rem;margin:8px 0 0}}nav{{display:flex;gap:8px;flex-wrap:wrap;margin:28px 0}}
.definitions{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;margin:18px 0 28px}}.definition{{background:#eaf2ed;border:1px solid #cbded3;border-radius:12px;padding:16px 18px}}.definition strong{{display:block;color:var(--accent)}}.definition p{{margin:6px 0 0;font-size:.9rem}}
.metric-tip{{display:inline-flex;align-items:center;gap:4px;position:relative;cursor:help;outline-offset:3px;white-space:nowrap}}.tip-icon{{color:var(--accent);font-size:.9em}}.tip-body{{display:none;position:absolute;z-index:10;top:calc(100% + 8px);left:0;width:min(280px,70vw);white-space:normal;background:#173b38;color:#fff;padding:10px 12px;border-radius:8px;box-shadow:0 8px 24px #0003;font-size:.82rem;font-weight:400;line-height:1.4}}.metric-tip:hover .tip-body,.metric-tip:focus .tip-body,.metric-tip:focus-within .tip-body{{display:block}}nav a{{background:#e6ede9;padding:8px 12px;border-radius:8px;text-decoration:none;font-size:.88rem}}.query{{background:var(--surface);border:1px solid var(--line);border-radius:16px;padding:30px;margin:24px 0;scroll-margin-top:20px}}
.query .eyebrow{{color:var(--accent)}}.need{{color:var(--muted);margin:10px 0 20px}}.callout{{display:flex;gap:18px;flex-wrap:wrap;background:var(--soft);padding:12px 16px;border-radius:9px}}
.chips{{display:flex;gap:7px;flex-wrap:wrap;margin:14px 0 20px}}.chip{{font:12px ui-monospace,monospace;background:#edf1f0;padding:5px 8px;border-radius:6px}}
.table-wrap{{overflow:auto}}table{{border-collapse:collapse;width:100%;min-width:630px;font-size:.9rem}}caption{{text-align:left;font-weight:700;margin:10px 0}}th,td{{padding:10px;border-bottom:1px solid var(--line);text-align:left}}thead{{background:#edf3ef}}tbody tr:hover{{background:#f8faf8}}
.strategies{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}}details{{border:1px solid var(--line);border-radius:10px;background:#fff}}summary{{cursor:pointer;padding:13px 16px;font-weight:700}}.strategy summary{{display:flex;justify-content:space-between;gap:10px}}.strategy summary span:last-child{{font-size:.8rem;color:var(--muted)}}
.hits{{list-style:none;padding:0 14px 10px;margin:0}}.hit{{border-top:1px solid var(--line);padding:13px 4px}}.hit-head{{display:flex;gap:8px;align-items:center;flex-wrap:wrap}}.rank{{color:var(--muted)}}.grade{{margin-left:auto;border-radius:99px;padding:2px 8px;background:#e7ecea;font-size:.75rem}}
.grade-1 .grade{{background:#fff0cf;color:#684300}}.grade-2 .grade{{background:#d9f0df;color:#145528}}.hit p{{font-size:.88rem;line-height:1.5;margin:7px 0;max-height:6em;overflow:auto;white-space:pre-wrap}}.hit small{{color:var(--muted)}}
.analysis-block{{background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:22px;margin:24px 0}}.section-head{{display:flex;justify-content:space-between;gap:20px;align-items:end;margin-bottom:14px}}.section-head h2{{font-size:1.35rem;margin:3px 0}}.section-head p{{max-width:420px;color:var(--muted);font-size:.87rem;margin:0}}.section-kicker{{color:var(--accent);font-size:.72rem;text-transform:uppercase;letter-spacing:.1em;font-weight:800}}.compact{{min-width:580px}}.compact th,.compact td{{padding:8px 10px}}.compact th:first-child{{white-space:nowrap}}tfoot{{background:#e9f1ed}}.table-note{{color:var(--muted);font-size:.83rem;margin:12px 0 0}}.impact-layout{{display:grid;grid-template-columns:minmax(0,1.4fr) minmax(220px,1fr);gap:12px;align-items:start;margin-bottom:22px}}.impact-layout table{{min-width:430px}}.impact-note{{border-left:3px solid var(--accent);padding:10px 14px;background:#eaf2ed;display:grid;gap:4px;font-size:.86rem}}.delta.positive{{color:#11613d;font-weight:700}}.delta.negative{{color:#a43e32;font-weight:700}}.delta.neutral{{color:var(--muted)}}.definitions-details{{margin:24px 0}}.definitions-details>.definitions{{padding:0 14px 14px;margin:0}}@media(max-width:700px){{.section-head{{display:block}}.section-head p{{margin-top:6px}}.impact-layout{{grid-template-columns:1fr}}.analysis-block{{padding:16px}}}}.moves{{margin-top:16px}}footer{{font-size:.83rem;color:var(--muted);margin-top:30px;word-break:break-all}}@media(max-width:850px){{.overview,.strategies{{grid-template-columns:repeat(2,1fr)}}}}@media(max-width:600px){{.overview,.strategies,.definitions,.takeaways{{grid-template-columns:1fr}}.query{{padding:18px}}}}
</style></head><body><header><div class="wrap"><div class="eyebrow">Hybrid Retrieval Lab · phase 6</div><h1>Pilot evaluation</h1>
<p>Four strategies, {query_count} information needs, {source_count} corpus chunks. Explore coverage, ordering and rank changes.</p>
<div class="meta"><span>{data["config"]["point_count"]} indexed chunks</span><span>{query_count} queries</span><span>4 strategies</span><span>Generated {_e(data["created_at_utc"])}</span></div></div></header>
<main class="wrap"><div class="notice"><strong>{judgment_status}</strong> Grades 0/1/2 are stored in the judgments file. Averages across {query_count} queries describe this corpus and these questions; they do not estimate general quality.</div>
{skipped_notice}<h2>Overview</h2><p class="intro">Compare top-ranked quality, relevant chunks retrieved and latency in this local run.</p><div class="overview">{overview}</div>
<div class="table-wrap"><table class="comparison"><caption>Quality and latency by strategy</caption><thead><tr><th>Strategy</th><th>{metric_tips["ndcg_at_5"]}</th><th>{metric_tips["ndcg_at_10"]}</th><th>{metric_tips["recall_at_5"]}</th><th>{metric_tips["recall_at_10"]}</th><th>Grade 2 at rank 1</th><th>Typical latency</th><th>vs. BM25</th></tr></thead><tbody>{comparison}</tbody></table></div>
<section class="analysis-block"><div class="section-head"><div><span class="section-kicker">Stage 1 · retrieval</span><h2>Candidate coverage</h2></div><p>Fraction of relevant chunks in each list. The union contains up to {union_limit} IDs; RRF selects {rerank_limit} for ColBERT.</p></div>
<div class="table-wrap"><table class="compact"><thead><tr><th>Query</th><th>BM25 · {retrieval_limit}</th><th>E5 · {retrieval_limit}</th><th>Union · up to {union_limit}</th><th>RRF → ColBERT · {rerank_limit}</th></tr></thead><tbody>{"".join(candidate_rows)}</tbody><tfoot><tr><th>Query average</th>{candidate_average}</tr></tfoot></table></div>
<p class="table-note">{_e(fusion_loss_note)}</p></section>
<section class="analysis-block"><div class="section-head"><div><span class="section-kicker">Stage 2 · reranking</span><h2>What ColBERT changed</h2></div><p>Paired comparison against hybrid RRF. Absolute differences of at most 1e-9 count as ties.</p></div>
<div class="impact-layout"><div class="table-wrap"><table class="compact"><thead><tr><th>Metric</th><th>Improved</th><th>Tied</th><th>Worsened</th></tr></thead><tbody>{impact_summary}</tbody></table></div>
<div class="impact-note"><strong>Interpretation</strong><span>{_e(impact_explanation)}</span></div></div>
<div class="table-wrap"><table class="compact"><caption>Per-query difference: ColBERT − hybrid RRF</caption><thead><tr><th>Query</th><th>Δ nDCG@5</th><th>Δ Recall@10</th><th>ColBERT / hybrid latency</th></tr></thead><tbody>{"".join(impact_rows)}</tbody></table></div>
<p class="table-note">Positive deltas favor ColBERT. Latency compares medians for the same query after warmup; it does not represent financial cost.</p></section>
<details class="definitions-details"><summary>How to read Recall and nDCG</summary><div class="definitions"><div class="definition"><strong>Recall@k · how many did we find?</strong><p>What fraction of grade-1/2 chunks appears in the first k results? Example: 4 of 5 relevant chunks in the top 10 = 80%.</p></div>
<div class="definition"><strong>nDCG@k · how good is the order?</strong><p>Ranges from 0 to 1. Rewards grade-2 chunks and higher positions; 1 represents the ideal order for this query's judgments.</p></div></div></details>
<p class="method">Typical latency = median of per-query medians ({data["config"]["timed_runs_per_query_strategy"]} timed runs after warmup). Includes query encoding and Qdrant; excludes HTTP. This small local run does not represent production. {provenance}</p>
<nav aria-label="Jump to query">{links}</nav>{"".join(sections)}
<footer>Recall treats grades 1 and 2 as relevant. nDCG uses gain 2<sup>grade</sup> − 1. Latencies are medians of {data["config"]["timed_runs_per_query_strategy"]} runs after warmup, including query encoding and Qdrant, excluding HTTP. Full data in <code>pilot.json</code>.<br>Judgments SHA-256: <code>{_e(data["inputs"]["qrels_sha256"])}</code>.</footer></main></body></html>"""
