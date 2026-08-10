# metadata-project

**Reconstructing missing sequencing metadata with an LLM.**

Public sequencing archives hold millions of datasets, but the descriptions attached to them are
patchy and inconsistent. Two labs uploading the same kind of experiment will label it in
completely different ways — one writes `tissue: lung`, the next writes `source_name: Lung (FFPE)`,
a third leaves both blank and only mentions it in their paper. Anyone trying to *reuse* that data
at scale — "find me every human lung RNA-seq dataset from a tumour" — hits a wall almost
immediately, because the archive cannot answer the question even when the answer exists.

This repository is a proof of concept for filling those gaps automatically: it harvests studies
from the NCBI **Sequence Read Archive (SRA)**, then rebuilds each one into a fixed 65-field schema,
using a language model only where the archive itself falls short — and recording, field by field,
*how* every value was arrived at.

The approach is modelled on [Metappuccino](metappuccino-findings.md), which did this for ~19
hard-coded fields of human cancer data. This version is broader: all of SRA, any organism, a
larger schema, and an explicit provenance trail.

> **Status: early proof of concept.** The harvest and schema layers are complete and tested; the
> model layers work and have been measured on real studies, but have not been run at scale. See
> [Where this actually stands](#where-this-actually-stands).

---

## Background: what SRA gives you, and what it doesn't

If you haven't worked with sequencing archives, three things are worth knowing up front.

**1. The data is a four-level hierarchy.** A submission is organised as:

```
Study (SRP…)          one research project — "gut microbiome of honey bees"
└── Experiment (SRX…) one library prepared from one sample, on one instrument
    ├── Run (SRR…)    the actual sequencing output — one file set
    └── Sample (SRS…) the biological material — organism, tissue, collection site
```

Accession prefixes tell you which of the three mirror archives issued the record: `S` = NCBI (US),
`E` = EBI (Europe), `D` = DDBJ (Japan). They're synchronised, so `SRP…` and `ERP…` records sit side
by side in the same search results.

**This repo emits one record per Experiment**, with the runs attached as a list. That's the level
at which the interesting biology varies — a single sample can be sequenced several times, and
those runs share every field you'd want to query on.

**2. Most descriptive metadata is free text with no controlled vocabulary.** Beyond a small set of
mandatory fields, submitters attach arbitrary key–value pairs. There is no fixed field list and no
enforced spelling. `geo_loc_name`, `geographic location`, and `country` are the same concept, and
all three appear in the wild.

**3. "Missing" is ambiguous.** A blank field might mean the submitter didn't record it, didn't
think it applied, or is withholding it. INSDC (the international body governing these archives)
defines a vocabulary for saying which — `not applicable`, `not collected`, `not provided`,
`restricted access` — and this project treats those as *real values worth preserving*, distinct
from an unanswered field. Guessing over a submitter's deliberate "not provided" is a mistake, and
one this pipeline made once at scale before it was caught (see [below](#what-has-been-measured)).

---

## The two pipelines

Work is split into a **harvest** (find studies worth reconstructing) and a **reconstruction**
(fill in the schema). They're separate because the harvest is slow and free, while reconstruction
is fast and costs money — you want to run the first once and the second many times.

### Pipeline A — Harvest

Finds real studies and shortlists the ones a model stands a chance with.

| Stage | Function | What it does |
|---|---|---|
| 1 | `save_recent_studies()` | Search SRA over a date window, de-duplicate experiment hits down to studies, fetch a summary of each, save as JSON. |
| 2 | `filter_oa_studies()` | For each study, find its linked publications and work out whether the full text is **openly readable**. Keep only studies where at least one is. |

Stage 2 matters more than it looks. Layer 4 of the reconstruction reads the paper, so a study
whose paper is paywalled is worth less — and "indexed in PubMed" is *not* the same as "readable".
The classifier chases PubMed → PMC → Unpaywall to find out. It is also the main filter on yield:
in the reference harvest only **305 of 1,858 studies** had an open-access paper.

Both stages checkpoint to `<output>.partial` every 25 studies and resume automatically, because a
full harvest runs for hours and a dropped connection shouldn't cost you all of it.

[**PIPELINE.md**](PIPELINE.md) documents every HTTP call this makes, in order, with the service and
database each one hits and why each branch exists.

### Pipeline B — Reconstruction

Takes harvested studies and produces schema records. Four layers run in sequence, each only
touching fields the previous ones left open, from cheapest and most trustworthy to most expensive
and most speculative.

| # | Layer | Cost | Source of the value |
|---|---|---|---|
| 1 | **direct** | free | The archive states it outright. Accessions, instrument, read counts, dates. |
| 2 | **harmonized** | free | A synonym table. `geo_loc_name` → `country`, `tissue` → `tissue_type`, and ~20 others. No model involved — this is deterministic renaming of a field the submitter did provide. |
| 3 | **inferred_from_text** | ~$0.0035/record | A model reads the study's own free text (title, abstract, sample attributes) and infers what it implies. "Isolated from ileal biopsies" → `tissue_type: ileum`. |
| 4 | **inferred_from_paper** | ~$0.013/study | A model reads up to 30,000 characters of the linked open-access paper — Methods section first — and fills what the archive never said. |

The ordering is the whole design. **A layer never overwrites an earlier one**, so a value the
archive stated directly can't be clobbered by a guess, and a paper can't overrule the submitter.

Layer 3 costs scale with *records*; layer 4 with *studies* — one paper, one call, applied to every
record in that study. Layer 4 is also blindfolded to 22 of the 65 fields (run identifiers,
submission dates, checksums) that a paper could not possibly know, which cuts both noise and
tokens.

---

## The target schema

`schema.py` defines `TargetSchema`: 65 fields, one object per experiment. 64 of them are genuine
[ENA portal API](https://www.ebi.ac.uk/ena/portal/api/) field names. That's deliberate on two
counts: the output is immediately intelligible to anyone who already works with these archives,
and because ENA publishes its own values for the same accessions, layer 2's synonym table can be
checked against them for free. The sole extension is `treatment` — ENA has no equivalent under any
spelling, but it's one of the commonest submitter attributes there is, and without a slot of its
own the model just posts it into whichever adjacent field looks closest.

Fields are tagged by the level they describe:

| Level | Count | Examples |
|---|---|---|
| sample | 32 | `tissue_type`, `country`, `host`, `isolation_source` |
| experiment | 12 | `library_strategy`, `instrument_model` |
| submission | 7 | `center_name`, `submission_accession` |
| study | 6 | `study_title`, `study_abstract` |
| run | 6 | `read_count`, `base_count` |
| record | 2 | `id`, `description` |

Alongside the values, each record carries three sidecars:

```python
record.tissue_type            # "ileum"
record.provenance["tissue_type"]   # "inferred_from_text"  — which layer produced it
record.confidence["tissue_type"]   # "medium"              — how sure, if inferred
record.runs                        # [{...}, {...}]        — the runs under this experiment
```

**Provenance is structural, not self-reported.** The cascade records which layer wrote each field;
the model is never asked to describe its own reliability. Confidence (`high`/`medium`/`low`) *is*
model-reported, and answers a narrow question — "given that this field applies, how sure are you
this is the right value?" — so it only exists on inferred fields. Both dictionaries validate keys
and values on every write, so a typo'd field name or an invented provenance class raises
immediately rather than silently entering the dataset.

---

## Running it

Two credential files at the repo root, both gitignored:

* `api_key.txt` — an NCBI API key. Raises the E-utilities limit from 3 to 10 requests/second.
* `email.txt` — a contact address. NCBI uses it to warn you before blocking your IP; Unpaywall requires one.
* `claude_api_key.txt` — an Anthropic API key. **Only needed for layers 3 and 4.**

```sh
uv sync

# The whole thing: harvest → filter → reconstruct
uv run python main.py

# Free run — layers 1 and 2 only, no API key needed, no spend
uv run python -c "import main; main.full_pipeline(from_text=False, from_paper=False)"

uv run pytest tests/                                        # everything (~100s)
uv run pytest tests/test_offline.py tests/test_dataset.py   # no network, no tokens
```

`full_pipeline()` takes `file_location` and `dataset_prefix` to control output paths, and creates
the directory for you. Each stage reads the previous stage's file, so you can run them
independently — reconstruct an existing harvest without re-fetching it, or re-harvest without
touching reconstruction.

### Reusing a harvest

Harvesting is the slow part, so `combine_studies()` merges saved study files into one corpus:

```python
from dataset import combine_studies
combine_studies(["datasets/dataset1/oa_studies.json", "datasets/dataset2/oa_studies2.json"],
                out_path="datasets/oa_corpus.json")
```

It de-duplicates by accession, preferring the copy with the most *classified* publications (that
classification cost real requests to compute), sorts the output for a stable diff, and prints what
reconstruction would cost. `datasets/oa_corpus.json` is the current one: **346 studies, 102,227
records**, every one with an open-access paper.

### Cost control

The model layers spend real money per sample, and a small harvest can hide a very large bill — a
20-study run once cost $7 because two of those studies held 1,664 experiments between them, and
nothing connected those facts before it spent.

So reconstruction estimates its cost first, prints a per-study breakdown, and **refuses to start**
above `max_spend` (default `$1.00`), having spent nothing:

```
Estimated cost: $362.29  (102,227 records x $0.0035 layer 3 + 346 papers x $0.013 layer 4)
SpendLimitExceeded: estimated $362.29 exceeds max_spend=$1.00 — nothing has been spent.
```

Pass a higher `max_spend` to authorise a bigger run, or `None` to disable the check. The cost
distribution is very lopsided — a handful of surveillance studies carry most of it, while ~200 of
the 346 are under 50 records — so the printed breakdown is how you pick an affordable slice.

If a layer fails on one study (a refusal, an exhausted retry budget, an expired API key) that study
falls back to its direct-only records and the run continues.

---

## What has been measured

Numbers below are from real runs against real studies, not simulations.

**Coverage.** Layers 1–2 alone fill ~36% of the schema (23.5/65 fields per record). Adding layer 3
takes it to ~50% (32.6/65). The remainder is what layer 4 and future work would have to reach.

**Layer 2 precision: 8,582/8,591 = 99.9%** across six fields, checked against NCBI ground truth on
a 1,664-record study. Deterministic renaming is nearly free and nearly perfect, and every mapping
moved from layer 3 to layer 2 is a strict win.

**One policy line cost 472 records.** In a surveillance study, 380 samples declared
`geo_loc_name: "not provided"`. Layer 2 recorded that faithfully — then the "is this field still
open?" rule treated a declared-missing value as fair game, and layer 3 overwrote it with guesses
inferred from the sequencing centre's address (Brazil ×167, Thailand ×58, Uganda ×51, Senegal ×39).
369 of 380 were wrong, plus 103 more on `host`.

The fix was to make that rule provenance-aware: a missing-value verdict can only be revisited by
the layer that *made* it, never one that read it from the archive. Re-verified across 9,117
ground-truth checks: **95.3% → 99.9%, wrong answers 416 → 0.**

**Prompt caching has a per-model floor.** Haiku 4.5 will not cache a prefix below 4,096 tokens, and
a `cache_control` marker below that limit is silently ignored — no error, no warning, just full
price on every call. [layer_3_haiku_findings.md](layer_3_haiku_findings.md) records this and the
rest of what the model layers cost and why.

---

## Layout

| File | Purpose |
|---|---|
| `project.py` | SRA client and object model. `Project` (study → experiment → run → sample), `search_studies()`, `scan_iter()`, `classify_publication()`, `fetch_open_access_text()`. Every HTTP call to NCBI/EBI lives here. |
| `schema.py` | `TargetSchema` — the 65 fields, the provenance/confidence sidecars and their validation, and `from_project()`, which is layer 1. |
| `reconstruct.py` | The four-layer cascade. Synonym table, prompts, field routing, and the rules about what each layer may touch. |
| `claude.py` | Anthropic API transport. Structured outputs, prompt caching, the Batch API, and token/cost accounting. Nothing pipeline-specific. |
| `dataset.py` | Pipeline orchestration — the three saved stages, checkpointing, `combine_studies()`, and the spend guard. |
| `main.py` | Entry point: `full_pipeline()`. Supplies credentials and parameters, nothing else. |
| `tests/test_offline.py` | 136 network-free tests. Cannot reach the API by construction, so no test can spend money. |
| `tests/test_project.py` | Live tests against NCBI. |
| `tests/test_dataset.py` | Network-free tests for checkpoint and resume. |
| `datasets/` | Harvest and reconstruction output. |

**Further reading:** [PIPELINE.md](PIPELINE.md) (every API call the harvest makes) ·
[layer_3_haiku_findings.md](layer_3_haiku_findings.md) (what the model layers cost and why) ·
[metappuccino-findings.md](metappuccino-findings.md) (the prior work and the schema it implies).

---

## Where this actually stands

Honest limitations, since this is a proof of concept and not a finished tool:

* **The model layers have not been run at scale.** Accuracy figures come from individual studies,
  not a representative sample. Nothing here is a benchmark.
* **Confidence is not yet a usable signal.** The values validate and round-trip, but they haven't
  been shown to correlate with correctness well enough to filter on.
* **`sort="random"` is coarser than it sounds.** The sampler shuffles across pages of 100
  experiment records, so a small harvest is satisfied by the first page and returns one
  contiguous block — often several studies from the same submitter. Fine for large harvests,
  misleading for small ones.
* **Layer 4 is the expensive, least-tested layer.** It carries the largest requests in the
  pipeline and has been run only on small study sets.
* **Requesting *n* studies yields roughly 0.93*n***, as oversized surveillance umbrellas are
  dropped before they can cost thousands of requests (PIPELINE.md §3).