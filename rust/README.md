# metadata-project — Rust implementation

Reconstructs structured, queryable metadata for SRA studies: takes a harvested corpus of
archive records plus their open-access papers, and produces one flat record per experiment
with 62 schema fields, each carrying where its value came from.

This is a port of the Python pipeline at the repository root, which remains the reference
implementation and the comparison baseline. See `../README.md` for the project's goals and
`../PIPELINE.md` for how the corpus is harvested — nothing in this crate touches NCBI or
Europe PMC. It starts from a corpus JSON file that the Python side produced.

```
cargo test                  # 330 tests, no network, no key, no spend
cargo run                   # free layers only
SPEND=1 cargo run           # adds the model layers, after a typed confirmation
```

Built on Rust edition 2024. Four dependencies: `chrono`, `serde`, `serde_json`, and
`reqwest` (blocking, rustls). There is no LLM SDK — the Anthropic transport is hand-rolled
in `model/claude.rs`, because structured outputs, prompt caching and the effort control are
not exposed by any published Rust client.

---

## The shape of a run

```
oa_corpus_full.json
        │
        ▼
   Corpus ──► Project ×N          input model: what the archives said
        │
        ▼
   TargetSchema::from_corpus(corpus, &settings)
        │
        │   for each project, for each layer in settings.layers():
        │       Direct ──► Harmonized ──► LLMNaive ──► LLMPaper
        │
        ▼
   Vec<TargetSchema>              output model: one record per experiment
        │
        ├──► Export ──► runs/<timestamp>.json
        └──► audit::verbatim(&export)
```

`SchemaSettings` carries the layer list, the caps, and the issue sink. It is the only
argument `from_corpus` needs beyond the corpus itself.

---

## The cascade

Layers are named, not numbered. The order is a property of the `Vec<Layer>` in
`SchemaSettings`, not of the enum — a colleague can reorder the list, drop a layer, or
insert a new one without editing `layer.rs`, so any ordinal name would go stale the first
time that happened. (Some code comments still say "layer 3" and "layer 4"; read those as
referring to `LLMNaive` and `LLMPaper`.)

The rule that makes the ordering meaningful: **a layer only ever fills fields no earlier
layer settled.** Nothing enforces it as an invariant — the ordering is a cost decision, and
running the free deterministic layers first means the paid ones pay only for what was left
open.

| Layer | Reads | Runs once per | Costs |
|---|---|---|---|
| `Direct` | the `Project` graph: study, submission, BioProject, experiment, runs, sample | **experiment** — or per (experiment, sample) pair when the experiment is pooled | free, offline |
| `Harmonized` | four submitter attribute bags, in `BAG_PRIORITY` order | **record**, keyed on that record's sample — so in effect one decision per sample | free, offline |
| `LLMNaive` | study title and abstract; per sample: organism, sample title, raw attribute bag, experiment titles, library strategy | **study** (for `Level::Study` fields) and **distinct sample** (for everything else), plus one grouped call for records whose sample does not resolve | one model call per job |
| `LLMPaper` | up to 30,000 characters of open-access full text, Methods-first | **paper** — a study with two linked papers makes two calls, each covering every record in the study | one model call per paper |

### `Direct`

The only layer that *creates* records; every other one fills fields on records this
produced. A layer scheduled before it sees an empty slice and does nothing.

Everything it writes is something an archive stated outright, which is what makes these the
anchors no later layer may overwrite. Run-level rollups (`total_spots`, `total_bases`) are
all-or-nothing per field: if any run in the experiment fails to report a count, the field
stays `Unknown` rather than presenting the sum of the reporting runs as an experiment total.

### `Harmonized`

A synonym table plus a normalisation pass mapping submitter attribute keys onto schema
fields. No model, no network. The *value* is the submitter's and is trustworthy; only the
key mapping is ours, which is exactly what `Provenance::Harmonized` records — and why these
values carry no `Directness`, since nobody chose them.

It reads four bags, and `BAG_PRIORITY` order *is* the precedence rule, because `assign`
never overwrites a settled field:

1. `SraSample` — the submitter's bag on the SRA record
2. `BioSample` — the submitter's bag on the BioSample record; overlaps heavily and disagrees
   with the SRA bag 433 times corpus-wide, which is what makes this an ordering decision
   rather than a free merge
3. `BioSampleHarmonized` — NCBI's `harmonized_name` view; last, because that key mapping is
   second-hand in the same way ours is
4. `Experiment` — `EXPERIMENT_ATTRIBUTES`; contributes nothing today, wired so a real
   attribute appearing there is not silently missed

Running it before the model layers pays twice: it fixes values the model was getting wrong
by hand, and every field it fills is closed to the model afterwards — fewer questions, on a
smaller schema, for fewer output tokens.

### `LLMNaive`

The first layer that costs money, and the first that can be wrong in interesting ways.
Everything above it is a lookup; this one asks a model.

Two call shapes, split by what the evidence can settle. Study-level fields get one call per
study — asking them per sample was both redundant and wrong, producing two different
`study_alias` answers across one study's fifteen samples. Everything else gets one call per
*sample*, since every experiment on a sample shares its biology.

Jobs with nothing open are never planned, so both shapes routinely disappear. In the
reference run below, `Direct` filled all six study-level fields on every record, so the
study call had an empty `wanted` and was skipped entirely.

Nothing in the file names a provider. It holds a `&dyn Model`, builds a provider-neutral
`Request`, and reads back text and usage. `plan` is a pure function over the records and is
separated from the sending, because *what gets asked, of whom, about which fields* is the
part that costs money when it is wrong — every test exercises it with no model.

### `LLMPaper`

Runs last, so it is only ever asked what nothing cheaper could answer. A full paper is tens
of thousands of tokens against ~540 for a sample's archive evidence, so a single call here
can cost more than every `LLMNaive` call in the same study combined.

One call per paper, not per sample: a paper says "we sequenced tumour and adjacent normal
tissue" without saying which sample is which, so a per-sample ask would invite a distinction
the text does not draw.

Because one answer covers the whole study, the two sides of the call are different sets:

- **What it is asked** (`wanted`) is the **union** of open fields across all the study's
  records. Open on any record → asked about.
- **What it is shown** (`established`) is the **intersection**: settled on *every* record,
  agreeing byte-for-byte across all of them, and an ordinary value rather than a declared
  absence. `records.first()` supplies the candidate string; the unanimity check is the gate,
  so a field where records disagree is dropped rather than letting the first one win. It
  goes over as `ALREADY ESTABLISHED (do not answer these again)`.
- **What it writes** goes to every record in the study, but through `assign`, which never
  overwrites a settled field — so the study-wide answer lands only in the slots still open
  on each individual record.

That last asymmetry is worth understanding before comparing runs: the cascade guarantees
*precedence*, not *consistency*. `LLMNaive` can settle a field per sample and leave the rest
open, and `LLMPaper`'s single answer then reaches only the leftovers, so one study can hold
two different answers for the same field.

### Blind fields

Some fields are assigned by the archive at deposition, and no abstract, attribute bag or
manuscript states them. `fields::level_of` sorts every field into `Study`, `Sample`,
`Experiment`, `Run`, `Submission` or `Record`, and:

- `LLMNaive` refuses `Run`, `Submission` and `Record` levels.
- `LLMPaper` refuses those plus `PAPER_BLIND_FIELDS` — `checklist`, `biosample_package`,
  `sample_alias`, `experiment_alias`, `study_alias`, `library_name`,
  `sample_capture_status`. These sit at sample or experiment level but are registration
  artefacts, not science.

Both guards drop fields before anything is planned, so a blind field never reaches a schema,
a token budget or an answer. This is a structural guard rather than a prompt instruction on
purpose: the Python prompt already said not to infer these and the model did it anyway,
because the field list it is shown outranks the prose telling it not to.

### Prompts

`layer/prompts/` holds the instruction texts, sent as the cached prefix on every call.
`text_system_full.txt` is the default — field definitions, routing rules for loose attribute
keys, common mistakes, and worked examples. `text_system_short.txt` is the framing only, and
is below the 4,096-token minimum cacheable prefix on some models, which makes its
`cache_control` a silent no-op. `text_system_targeted.txt` was an attempt at raising the
verbatim rate that measured worse on both axes and stays unused. `LLMPaper` composes its
prompt at compile time from the full text plus `paper_addendum.txt`, so the field definitions
cannot drift between the two layers.

---

## The record

`TargetSchema` is one fully denormalised row per experiment: `id` plus 62 fields, with study,
sample, experiment, run and submission values flattened onto the same record.

**Field names follow `project`, not ENA**, and two renames invert their old meaning. ENA
calls the BioProject `study_accession` and the SRP `secondary_study_accession`; likewise the
BioSample is `sample_accession` and the SRS is `secondary_sample_accession`. Here
`study_accession` is the **SRP** and `sample_accession` is the **SRS**. Anything comparing
against an ENA-named schema must swap those two pairs — a positional mapping produces
wrong-but-plausible values silently.

Every value is a `Field<T>`:

```rust
enum Field<T> {
    Unknown,                                // nothing has settled this yet
    Missing(MissingReason, Provenance),     // someone stated it does not apply
    Known(T, Provenance),                   // an ordinary value
}
```

`Provenance` is folded into the value rather than kept in a parallel map. Python carries
provenance and confidence as two side maps and then needs a runtime sweep to catch illegal
combinations — a confidence on a direct field, or on a field with no value. Making the
inferred variants the only ones that carry a `Directness` deletes that class of bug at the
type level.

```rust
enum Provenance {
    Direct,
    Harmonized,
    InferredFromText(Directness),
    InferredFromPaper(Directness),
}

enum Directness { Quoted, Rephrased, Inferred }
```

**`Directness` replaced a confidence score**, and records *what the model did* rather than
how sure it is. Asked for a confidence, the model produced 391 high / 41 medium / 0 low
across 432 inferences, with the largest error class uniformly `high` — a scale whose bottom
rung is never used cannot separate right answers from wrong ones. `Quoted` is the only claim
that is machine-checkable, which is what `audit` exists to check.

**`MissingReason` preserves INSDC's missing-value vocabulary** — `not applicable`,
`not collected`, `not provided`, `restricted access`, and the bare `missing` as
`Unspecified`. A submitter who wrote "not applicable" has *answered*; storing that as a
literal string would assert the country is called "not applicable". The recogniser is
deliberately closed to the controlled terms only: a submitter's own "unknown" or "na" is
free text that happens to read like an absence, and promoting it would be inferring a
determination nobody made.

`PartialDate` keeps `Year` / `YearMonth` / `Date` apart rather than padding up, because 31%
of collection dates in the corpus are a bare year.

---

## Input model

`project.rs` holds everything the four sources can say about one study, in one owned value:
`Project` with its `Study`, `Submission`, `BioProject`, `Sample`, `BioSample`, `Experiment`,
`Run` and `Paper`. Design notes are in `../RUST_PROJECT_DESIGN.md`.

Two conventions worth knowing before editing anything here:

- **Accession newtypes.** `StudyAccession`, `SampleAccession`, `ExperimentAccession`,
  `RunAccession`, `BioSampleAccession`, `BioProjectAccession`. They cost nothing and make
  `samples[&experiment.accession]` a compile error instead of a `None` at runtime. The
  `Archive` enum records which of NCBI / ENA / DDBJ an accession came from — mirroring makes
  the *data* seamless across the three and the *identifiers* not, since NCBI's efetch
  mis-resolves non-NCBI accessions on the numeric part alone.
- **`BTreeMap`, never `HashMap`.** The evidence string handed to the model is built by
  serialising the attribute bags, and prompt caching is a prefix match, so non-deterministic
  iteration order silently loses every cache hit. It also keeps corpus round-trips
  byte-reproducible.

`dto.rs` holds wire types that mirror the corpus JSON exactly and exist only to be converted
into the domain types. A direct `Deserialize` onto the domain types will not work: `papers`
is a map keyed by publication id, study fields are flat on the wire and nested in `Project`,
controlled vocabularies arrive as strings and become enums, and dates arrive as bare strings
and become `ArchiveDate` / `ZonedDate` carrying the granularity the source committed to.

`corpus.rs` loads the file and refuses an unrecognised `format_version` rather than
mis-parsing one — the format has already changed once, and v1 carried no submission,
BioProject record, run statistics or per-publication papers.

---

## Talking to a model

`model.rs` defines the seam:

```rust
pub trait Model: Send + Sync {
    fn complete(&self, request: &Request) -> Result<Response, ModelError>;
    fn price(&self, usage: Usage) -> f64;
    fn complete_many(&self, /* … */) -> /* per-key Results */;
}
```

**Synchronous on purpose.** The only I/O in the pipeline is here, and async would colour
every caller above it — `Layer::process`, `from_project`, `from_corpus` — none of which
touch the network. The useful concurrency is bounded by the provider's rate limit, and the
batch path turns thousands of records into three requests and some polling.

`price` lives on the trait rather than beside the Anthropic rates so a budget can meter any
provider: a local model answers zero and the accounting still works. `complete_many`
defaults to sending requests one at a time, so a provider with no batch endpoint works
unchanged, and returns a per-key `Result` so one bad request does not cost the rest.

Implementations and decorators:

- **`model/claude.rs`** — the Messages API over raw HTTP. `body_for` and `parse` are pure and
  `complete` is the only function that performs I/O, so the payload and the result handling
  are fully testable without a network or a key. Handles structured outputs, cache control,
  effort, thinking budgets, and the model-specific combinations the API rejects.
- **`model/claude/batch.rs`** — the Message Batches API: same work, half price, minutes to
  hours of latency. Deliberately *not* part of the `Model` trait, since its shape is
  different and no other planned provider offers one. (Measured effective saving is ~45%
  rather than 50%, because the fan-out pays for an extra cache write.)
- **`model/retry.rs`** — `Retrying<M>`, five attempts with exponential backoff by default.
  What is *not* retried matters as much: refusals, malformed schema replies, 400s from a bad
  parameter combination and budget refusals are deterministic, and re-sending bills again for
  nothing. `ModelError::is_retryable` draws that line. A server-supplied `retry-after` is
  capped, so an unattended run fails rather than hangs.
- **`model/budget.rs`** — `Budget` (a shared ledger) and `Budgeted<M>` (the decorator that
  meters through it).

Compose them **budget outermost**: `Budgeted<Retrying<Claude>>`, so retries underneath count
as one billed call. Each layer gets its own client, but they share one `Budget`, which is
what makes the ceiling a limit on the run rather than on each layer separately.

---

## Spend guards

Four, failing independently, in the order they fire:

1. **`SPEND=1`** — the model layers are not constructed without it, so a bare `cargo run` is
   free by construction rather than by a flag that could be read the wrong way.
2. **`confirmation_prompt`** in `main.rs` — prices the *actual plan* via `estimate::for_corpus`
   before anything is sent, and requires a typed `y`. Anything else declines, including
   end-of-input, so a piped or unattended run cannot spend. Free runs are not interrupted:
   a prompt that appears when the answer cannot matter is a prompt that gets typed through.
3. **`Budget` / `MAX_SPEND`** — the in-the-moment ledger, metering what has actually been
   billed and refusing the next call once the ceiling is reached. It cannot un-spend the call
   that crossed the line, so the ledger can end slightly above the limit; what it guarantees
   is that the run cannot continue past it.
4. **`MAX_STUDIES` / `MAX_RECORDS`** — volume caps applied *before* the layers run, so a paid
   layer is never asked about a record that is going to be discarded.

Guards 2 and 3 are not redundant. A ledger learns a call's cost only after paying for it, and
it stops a run *part-way*, which leaves a half-finished paid layer nobody can compare against
anything. The estimate answers the question a ledger cannot: should this begin at all.

**`estimate.rs`** works from the same `plan` functions the run executes, so it prices the real
workload rather than a model of it. Its constants are measured, not assumed —
`CHARS_PER_TOKEN` is **2.7**, from a run that sent 16,652 characters of instructions and was
billed 6,148 tokens for them; the folklore figure of four would have under-counted by a third,
in the direction that matters least for a guard whose job is to prevent a surprise.

---

## Selecting studies, and comparing runs

`SchemaSettings::only_studies([...])` picks studies by accession, matching either the SRP or
the BioProject form, case-insensitively, and reports unmatched accessions through the issue
sink *before* anything runs — a mistyped accession otherwise produces a run that looks correct
and covers a different study set than the one it will be compared against. Selection happens
before the caps, so five named studies capped to thirty records is thirty records of those
five.

This exists because coverage varies by several fields per record *between studies* on the
model layers alone. One study cannot answer a coverage question; the same studies twice can.

**`export.rs`** saves each run to `runs/<timestamp>.json` with its parameters and a provenance
histogram beside the records. The point is not backup — two runs differing in one variable are
only comparable if both were kept along with what produced them. `format_version` is **4**,
which records `params.models` as one entry per model layer, since `LLMNaive` and `LLMPaper`
can now run different models and settings. Older versions stay readable as long as every field
added since has a default; `READABLE_VERSIONS` names them, and that is a real constraint on how
the shape may change rather than a courtesy.

**`audit.rs`** checks the one claim a model makes that can be falsified. `quoted` says the value
appears in the evidence word for word — a statement about a string, which a string match can
settle. No model, no network, no spend; the input is a saved run. It reports `verified`,
`unsupported` (claimed `quoted`, not found), `understated` (labelled `rephrased` or `inferred`
but verbatim anyway — not an error, but a run with many of them has labels that are not
tracking what was done), and `unauditable` (inferred answers whose record stored no evidence,
reported rather than counted as passes). Auditing needs `SchemaSettings::keep_evidence(true)`,
which is off by default because a corpus run would store every sample's attribute bag twice.

---

## Module map

| File | What it is |
|---|---|
| `main.rs` | the experiment harness: corpus path, caps, per-layer model settings, the confirmation prompt |
| `lib.rs` | module list and `prelude` |
| `corpus.rs` | loading a harvested corpus, version-gated; `papers_of` |
| `dto.rs` | wire types mirroring the corpus JSON, and their conversions into `project` |
| `project.rs` | the input model — accession newtypes, controlled vocabularies, the archive object graph |
| `target_schema.rs` | the output record, `Field`/`Provenance`/`Directness`/`MissingReason`, `SchemaSettings`, `from_corpus` |
| `layer.rs` | the `Layer` enum, `ModelConfig`, and dispatch to `process` / `plan` / `estimate` |
| `layer/direct.rs` | `Direct` |
| `layer/harmonized.rs` | `Harmonized`, the bag priority list and the synonym table |
| `layer/llm_naive.rs` | `LLMNaive`, plus `Job`/`JobKey`/`apply`/`answer_schema` shared with the paper layer |
| `layer/llm_paper.rs` | `LLMPaper`, `established`, `PAPER_BLIND_FIELDS` |
| `layer/fields.rs` | field name list, `level_of`, `open_fields`, `assign`, missing-value and date parsing |
| `layer/prompts/` | the instruction texts |
| `model.rs` | the `Model` trait, `Request`/`Response`/`Usage`/`ModelError`, `Effort`, `Thinking` |
| `model/claude.rs` | the Anthropic Messages API |
| `model/claude/batch.rs` | the Message Batches API |
| `model/retry.rs` | `Retrying<M>` and its policy |
| `model/budget.rs` | `Budget`, `Ledger`, `Budgeted<M>` |
| `estimate.rs` | pre-flight pricing over the same plan the run executes |
| `export.rs` | saving a run with its parameters and provenance histogram |
| `audit.rs` | verbatim checking of `quoted` claims |

---

## A reference run

`runs/20260818T044453Z.json` — five studies, 52 records, 52 distinct samples, all four layers,
Sonnet 5 with thinking disabled and both model layers batched. 55 API calls, $0.2627.

| Layer | Real values per record | Most common fields |
|---|---|---|
| `Direct` | 29.00 | 29 fields at 100%: accessions, titles, the whole library block, `platform`, `instrument_model`, `total_spots`, `total_bases`, `earliest_run_published` |
| `Harmonized` | 2.50 | `strain` 85%, `cell_line` 42%, `tissue_type` 42%, `dev_stage` 29%, `treatment` 27%, `cell_type` 25% |
| `LLMNaive` | 2.60 | `library_name` 48%, `cell_type` 48%, `treatment` 48%, `sample_description` 29%, `age` 29%, `sequencing_method` 25% |
| `LLMPaper` | 0.81 | `isolation_source` 44%, `age` 13%, `sequencing_method` 10% |

Free layers produced 1,638 real values for nothing; the paid layers produced 177 for $0.26.

Two caveats before reading anything into the field lists. This study set is *Mus musculus*,
*C. elegans* and *H. sapiens*, so the MIxS environmental triad never fires and the frequencies
are specific to these studies. And a coverage comparison must separate real values from
declared absences — raw "settled" counts include fields where the answer is "not applicable",
which cost full token price and carry no information.

---

## Conventions

- **`//` comments only, never `///`.** Developer comments are wanted; user-facing doc comments
  are not, because the comments are eventually to be human-written throughout.
- Pure planning separated from I/O everywhere it appears: `plan` from `process`, `body_for`
  and `parse` from `complete`, `batch_body` and `parse_results` from the polling loop. That
  separation is what lets 330 tests run with no network, no key and no spend.
- Tests live in-file, in `mod tests` at the bottom of each module.