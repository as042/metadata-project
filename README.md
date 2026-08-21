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
`restricted access`, and the bare `missing` for an absence with no reason attached — and this
project treats those as *real values worth preserving*, distinct from an unanswered field. Guessing
over a submitter's deliberate "not provided" is a mistake, and one this pipeline made once at scale
before it was caught (see [below](#what-has-been-measured)).

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

The two paid figures are for the default model (Haiku 4.5, no thinking) and scale with whatever
model you select — see [Choosing models per layer](#choosing-models-per-layer).

The ordering is the whole design. **A layer never overwrites an earlier one**, so a value the
archive stated directly can't be clobbered by a guess, and a paper can't overrule the submitter.

Layer 3 costs scale with *records*; layer 4 with *studies* — one paper, one call, applied to every
record in that study.

Both model layers are blindfolded to fields they cannot answer. Layer 3 skips the 15 run-,
submission- and record-level fields the archive assigns at deposition; layer 4 skips 22 — those
same 15 plus seven submitter-chosen identifiers a paper could not possibly know. Each guard drops
its fields *before* anything is planned, so a blind field never reaches a prompt, a token budget or
an answer, which cuts noise and cost together.

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

### The Anthropic key

**There is no default key file.** Layers 3 and 4 spend real money, so the account that pays is
named per run with `claude_key_file=`; a run that turns a model layer on without naming one is
refused before it does any work:

```sh
uv run python -c "import main; main.full_pipeline(claude_key_file='anton_claude_api_key.txt')"
```

Keep each key in its own gitignored file. The path is read and validated *first* — ahead of the
output directory, ahead of NCBI, ahead of the harvest — so a missing file, an empty one, or one
holding the wrong credential entirely (an NCBI key, say) costs a local file read rather than
surfacing after stage 1 has already spent its requests. Reading the file spends nothing; the key
is not used until layer 3 makes its first call. Each run prints `Billing Claude key: <file>` next
to the cost estimate, so which account pays is visible before anything is spent.

This selects *which* account pays; it does not raise `max_spend`, which caps every run regardless.
Layers 1 and 2 are free and need no key at all.

```sh
uv sync

# The whole thing: harvest → filter → reconstruct
uv run python main.py

# Free run — layers 1 and 2 only, no API key needed, no spend
uv run python -c "import main; main.full_pipeline(from_text=False, from_paper=False)"

uv run pytest tests/                                        # 236 tests, ~50s (test_project.py hits NCBI)
uv run pytest tests/test_offline.py tests/test_dataset.py   # 197 tests, ~1s, no network, no tokens
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

That file holds *summaries*. Everything a model layer actually reads — per-sample attribute bags,
experiment and run rows, paper text — was re-fetched from SRA and Europe PMC on every run, which
makes two runs a week apart different experiments. `corpus.py` expands a shortlist once into a
self-contained file instead:

```python
import corpus
corpus.build_full_corpus("datasets/oa_corpus.json", "datasets/oa_corpus_full.json")
```

`datasets/oa_corpus_full.json` is the current build: **346 studies, 102,240 records, 88,560
samples, 330 papers** (61 of those turned out to have no retrievable text). It costs no tokens —
NCBI, Europe PMC and Unpaywall only — and takes roughly two and a half hours, almost all of it
waiting on SRA. Checkpointed, so an interrupted build resumes.

Three things this buys: runs become reproducible, anything consuming reconstruction input only has
to speak JSON rather than three archive APIs, and the paper text is persisted rather than
discarded. Papers are stored once and keyed by publication id, since 17 of them serve more than one
study.

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

Every run ends with what it actually billed, priced per call at that call's own model rate:

```
Claude usage: $1.25 | 61 calls | in 94,554 (+299,756 cached, 73,929 written) | out 81,786
```

Compare that against the estimate printed at the start. They should be close; a large gap means a
multiplier in `dataset.THINKING_MULTIPLIER` needs replacing with the measured number.

### Choosing models per layer

Each model layer takes its own `model` / `effort` / `thinking`, because the two layers have
opposite cost shapes: layer 3 makes one small call per *record*, layer 4 one large call per
open-access *study*. Upgrading layer 4 alone is cheap; upgrading layer 3 is what scales.

Models and effort levels have named constants in `claude.py`, so they autocomplete and a typo is an
`AttributeError` at import rather than a string that travels until the cost estimate refuses it:

```python
import claude, main

main.full_pipeline(
    claude_key_file="anton_claude_api_key.txt",
    paper_model=claude.OPUS_5, paper_effort=claude.EFFORT_MEDIUM, paper_thinking=True)
```

`claude.MODELS` and `claude.EFFORT_LEVELS` list what is available. The constants are the API's own
IDs, so the equivalent raw strings (`"claude-opus-5"`, `"medium"`) still work everywhere — including
in datasets and checkpoints already on disk.

**The cost estimate follows these settings**, so `max_spend` keeps protecting the run — the
per-unit figures are measured on Haiku 4.5 and scaled by the model's price and by thinking.
Relative to that baseline (`test2` = 52 records / 5 papers; full harvest = 102,227 records / 305
papers):

| layer settings | multiplier | test2 | full harvest |
|---|---|---|---|
| `claude-haiku-4-5` (default) | 1.0× | $0.25 | $362 |
| `claude-sonnet-5`, no thinking | 2.0× | $0.49 | $724 |
| `claude-opus-5`, no thinking | 5.0× | $1.24 | $1,809 |
| `claude-opus-5`, `medium` + thinking | 9.5× | $2.35 | $3,437 |
| `claude-opus-5`, `xhigh` + thinking | 20.0× | $4.94 | $7,235 |

Price scaling is exact — every model here bills output at 5× input, so the ratio is unambiguous.
The thinking multipliers are partly measured (Opus 5 `medium` = 1.9×; Sonnet 5 at default effort =
2.5×) and partly interpolated, rounded up so the guard errs toward refusing too early. Every run
now prints its actual dollar spend, which is the measurement — replace a multiplier as soon as you
have a run at that setting.

> **`thinking=False` used to be a lie on Sonnet 5 and Opus 5.** Omitting the `thinking` field runs
> *adaptive* thinking on those models, so a run configured "thinking off" thought anyway and billed
> $1.25 against a $0.49 estimate and a $1.00 cap. The flag now sends an explicit `disabled`, and an
> unset `effort` is priced as the API's real default (`high`) rather than as cheap.

Combinations the API rejects are refused before the harvest, not at the first request: `effort` or
thinking on Haiku 4.5, and `thinking=False` on Opus 5 above `high` effort. A model with no entry in
`claude.PRICES` is refused outright rather than costed at the baseline.

### Auditing the confidence label

`confidence` records **where a value came from**, not how likely it is to be right:

| level | meaning |
|---|---|
| `high` | quoted — appears in the evidence word for word |
| `medium` | rephrased, or one of several spans the model could have quoted |
| `low` | inferred — the evidence does not carry the value |

The point of a mechanical axis is that the top level is *checkable*. `audit.py` verifies it by
string matching — no gold set, no model, **no spend**:

```sh
uv run python audit.py datasets/test3/test_reconstructed.json                        # offline
uv run python audit.py datasets/test2/test_reconstructed2.json \
                       datasets/test2/test_filtered.json                             # exact
```

The offline mode is blind to the sample attribute bag (it is not saved in the output), so it can
prove a value was quoted but not disprove it — non-matches come back `unknown`, never as failures.
Passing a studies file re-fetches from SRA to rebuild the exact evidence string, which costs NCBI
requests and no tokens. Layer 4 is **not auditable by `audit.py`**: reconstruction output does not
persist the paper text a value was drawn from. The expanded corpus does store it, so the evidence
now exists — `audit.py` has not been taught to read it from there.

A quoted value can still be wrong — the model may have quoted the wrong span. This measures how
far the model reached, not whether it landed; accuracy still needs a labelled set.

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

**A synonym key that collides with a field name silently wins.** `description` is a common
submitter *sample* attribute and also the name of the STUDY-level abstract field, and the
exact-name match resolved first — so 87 sample descriptions across the corpus were written into the
study abstract instead of `sample_description`. The fix derives the shadowed keys from the synonym
table rather than listing them, so a future row cannot reintroduce it. Note it is deliberately not
a blanket "a sample attribute may not fill a study-level field" rule: `project_name` is study-level
and is legitimately filled from the sample bag 8,109 times.

**Layer 3 was answering questions the archive alone can settle.** Across the 1,782-record and
52-record runs it filled 4,342 run-, submission- and record-level slots — a BioProject accession
written into `submission_accession` on 146 records, a strain id into a run accession, and thousands
of "not provided" at full token price. Those 15 fields are now dropped from the ask entirely,
mirroring what layer 4 already did with its 22.

**Prompt caching has a per-model floor.** Haiku 4.5 will not cache a prefix below 4,096 tokens, and
a `cache_control` marker below that limit is silently ignored — no error, no warning, just full
price on every call. [layer_3_haiku_findings.md](layer_3_haiku_findings.md) records this and the
rest of what the model layers cost and why.

---

## Why there's also a Rust implementation

`rust/` holds a port of the reconstruction side of this pipeline, and it is where the work
continues. It is intended to be production-level, and it does not re-harvest anything: it starts
from the expanded corpus file this Python code produces, so the two read the same input and their
outputs are directly comparable.

Three things the type system buys there that are runtime checks here:

* **Provenance folded into the value.** `Field<T>` is `Unknown | Missing(reason, provenance) |
  Known(value, provenance)`, so the sidecar dictionaries and the sweep that validates them are
  gone. A confidence on a direct field is unrepresentable rather than caught.
* **`Directness` in place of `confidence`.** It records *what the model did* — quoted, rephrased,
  inferred — rather than how sure it claims to be. `quoted` is machine-checkable against stored
  evidence, and because the corpus persists paper text, the Rust auditor can check both model
  layers where `audit.py` can only check layer 3.
* **Four independent spend guards** rather than one pre-flight estimate: an env var the paid layers
  do not exist without, a typed confirmation that prices the real plan, a running ledger, and
  volume caps.

Run over the same five studies as a Python layers-1–4 run, compared on the 40 fields the two
schemas share, the port fills **21% more real values per record** — and all of that comes from the
free direct layer reading source objects Python was paying a model to infer. The two model layers
are statistically indistinguishable between the implementations.

Python remains the reference implementation and the comparison baseline, and this README remains
the description of it. **[rust/README.md](rust/README.md)** documents the Rust crate: its cascade,
its type model, the transport and its spend guards.

---

## Layout

| File | Purpose |
|---|---|
| `project.py` | SRA client and object model. `Project` (study → experiment → run → sample), plus the Submission, BioSample and BioProject records, `search_studies()`, `scan_iter()`, `classify_publication()`, `publication_oa_status()`, `fetch_open_access_text()`. Every HTTP call to NCBI/EBI lives here. |
| `schema.py` | `TargetSchema` — the 65 fields, the provenance/confidence sidecars and their validation, and `from_project()`, which is layer 1. |
| `reconstruct.py` | The four-layer cascade. Synonym table, prompts, field routing, and the rules about what each layer may touch. |
| `claude.py` | Anthropic API transport. Structured outputs, prompt caching, the Batch API, and token/cost accounting. Nothing pipeline-specific. |
| `corpus.py` | Expands a study shortlist into a self-contained corpus file — attribute bags, experiments, runs and paper text — so reconstruction reads JSON instead of re-fetching three archives on every run. |
| `dataset.py` | Pipeline orchestration — the three saved stages, checkpointing, `combine_studies()`, and the spend guard. |
| `main.py` | Entry point: `full_pipeline()`. Supplies credentials and parameters, nothing else. |
| `tests/test_offline.py` | 190 network-free tests. Cannot reach the API by construction, so no test can spend money. |
| `tests/test_project.py` | Live tests against NCBI. |
| `tests/test_dataset.py` | Network-free tests for checkpoint and resume. |
| `datasets/` | Harvest and reconstruction output. |

**Further reading:** [PIPELINE.md](PIPELINE.md) (every API call the harvest makes) ·
[layer_3_haiku_findings.md](layer_3_haiku_findings.md) (what the model layers cost and why) ·
[layer_order_findings.md](layer_order_findings.md) (every cascade ordering, measured) ·
[cost_findings.md](cost_findings.md) (unit costs, archive-scale projections, what the guards bound) ·
[metappuccino-findings.md](metappuccino-findings.md) (the prior work and the schema it implies) ·
[rust/README.md](rust/README.md) (the Rust implementation).

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