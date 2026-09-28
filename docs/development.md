# Development and verification

From the repository root with Python 3.11+ and uv 0.11.3:

```bash
uv sync --locked --extra dev
uv run --locked ruff check src scripts tests
uv run --locked ruff format --check src scripts tests
uv run --locked mypy
uv run --locked pytest -q -m unit tests/unit
```

| Suite | Dependencies | Purpose |
| --- | --- | --- |
| `tests/unit/` | Explicit stubs/mocks; no downloads or live Qdrant | RRF, metrics, factory/service, HTTP, logging, token policy, index identity. |
| `tests/integration/` | Live Qdrant and real encoders | Indexing, vectors, strategies, candidate invariants and token limits. |

Within each suite, test modules follow the corresponding production package: for example, API tests live under `api/`, encoder tests under `encoders/`, and retrieval strategy tests under `services/search/strategy/`. Integration tests use the same service path beneath `tests/integration/`. The package-level `logger.py` test remains at `tests/unit/test_logger.py`.

Run integration separately after installing development dependencies:

```bash
docker compose up -d qdrant
QDRANT_URL=http://127.0.0.1:6333 pytest -q -m integration tests/integration
```

Tests create isolated collections and clean up. Host execution uses the host model cache, not Compose's named volume, and may download models again. Avoid concurrent evaluation workloads.

Ruff checks source, scripts and tests. Strict mypy covers source and scripts; dynamic test doubles are outside that scope. Embedded report HTML has a line-length exemption.

[The workflow](../.github/workflows/quality.yml) uses the same `uv.lock` and pinned uv release for lint, format, types and unit tests on push/PR with Python 3.11. It does not run heavy integration or evaluation. Verify remote CI after pushing; local results are not a GitHub Actions result.

The [index validation and token-budget guide](index-validation-and-token-budget.md) records a previous run with 51 unit and six integration tests; those counts describe that execution. Follow the [reproduction and recovery guide](reproduction-and-recovery.md) for the current clean-clone baseline procedure and index recovery steps.

## Coverage

Install the development dependencies and run:

```bash
pytest -q -m unit tests/unit --cov \
  --cov-report=term-missing --cov-report=xml:coverage.xml \
  --cov-report=json:coverage.json --cov-report=html:htmlcov
```

Open `htmlcov/index.html` to inspect missed statements and branches. Coverage configuration in `pyproject.toml` includes the entire Python package and scripts, including unimported modules. Tests are outside those source roots. Generated coverage files are ignored by Git. No minimum threshold is enforced yet.

The workflow puts coverage in its run summary and saves XML, JSON and HTML as the `unit-test-coverage` artifact. Download that artifact from the GitHub Actions run to inspect the HTML report. No external coverage service is required.

The README links the main-branch CI badge and a static unit-coverage badge. After changing tests or production code, rerun the command above, read the total percentage from its terminal summary, and update the badge value in `README.md` before pushing. The badge is intentionally manual, so it never depends on an external service and may be stale if it is not refreshed. Coverage does not establish assertion quality or retrieval quality.

The distribution and GitHub project are named `retrieval-lab`; the existing Python import package remains `hybrid_retrieval_lab`, so documented CLI commands stay compatible.
