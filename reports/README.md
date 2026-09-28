# Published evaluation report

The canonical baseline uses the token-only chunker, the frozen 50-chunk Portuguese corpus and ten information needs:

- `baseline/index.html` — browser-friendly report
- `baseline/pilot.json` — rankings, metrics, timings and reproduction provenance

The report contains rankings and metrics for all 40 query/strategy combinations. Its qrels include some grades carried forward from the previous 53-chunk snapshot; this limitation is disclosed in the report and [evaluation method](../docs/evaluation.md). Reproduce it with the [documented procedure](../docs/reproduction-and-recovery.md). Historical and intermediate reports are not included in the current tree.
