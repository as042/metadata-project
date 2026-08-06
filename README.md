# metadata-project

LLM-driven reconstruction of SRA study metadata, after the *Metappuccino* approach.

This repository currently covers the **data layer**: finding studies in the Sequence Read
Archive, summarising them, and shortlisting the ones whose linked paper is openly readable —
the context an LLM needs to reconstruct fields SRA does not reliably provide.

## Layout

| File | Purpose |
|---|---|
| `project.py` | SRA client and object model. `Project` (study → experiment → run → sample), `search_studies()`, `scan_iter()`, `classify_publication()`. Every HTTP call lives here. |
| `dataset.py` | The two pipeline stages: `save_recent_studies()` and `filter_oa_studies()`. Owns checkpointing. |
| `main.py` | Entry point — supplies credentials and parameters, nothing else. |
| `tests/test_offline.py` | Network-free unit tests. Safe for CI, runs in under a second. |
| `tests/test_project.py` | Live tests against NCBI. Cannot run offline; a flaky response fails them. |
| `tests/test_dataset.py` | Network-free tests for checkpoint and resume. |
| `datasets/` | Harvest output. |

## Documentation

* **[PIPELINE.md](PIPELINE.md)** — the full map: every API call the harvest makes, in
  order, with the service and database each one talks to, and why each branch exists.
* **[metappuccino-findings.md](metappuccino-findings.md)** — notes on the Metappuccino
  paper and the target schema it implies.

## Running it

Two credential files are needed at the repo root, both gitignored:

* `api_key.txt` — an NCBI API key. Raises the E-utilities limit from 3 to 10 requests/second.
* `email.txt` — a contact address. NCBI uses it to warn you before blocking your IP, and
  Unpaywall requires one.

```sh
uv sync
uv run python main.py          # harvest, then filter to open-access studies
uv run pytest tests/           # everything
uv run pytest tests/test_offline.py tests/test_dataset.py   # no network
```

A harvest checkpoints every 25 studies to `<output>.partial` and removes it on success.
Re-running the same call resumes from where it stopped; nothing needs to be re-fetched.
Requesting *n* studies yields roughly 0.93*n*, as oversized surveillance umbrellas are
dropped — see PIPELINE.md §3.
