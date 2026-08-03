# How Metappuccino Works — Key Findings

Notes on the *Metappuccino* approach ("large language model-driven reconstruction of
sequence read archive metadata for cancer research") and what it implies for building our own
version on top of `pysradb`.

## 1. The core problem Metappuccino solves

SRA metadata is **inconsistent and incomplete**. The fields researchers want to query by
(age, sex, disease, is-cancer, biopsy site, treatment…) are *not* guaranteed structured
columns. They exist — if at all — as free-text, submitter-defined key–value tags with wildly
varying names, or buried in titles/descriptions, or simply absent. Metappuccino imposes a
**fixed target schema** on top of this mess and uses an LLM to normalize and infer the fields
SRA doesn't reliably provide.

## 2. Metappuccino's target fields are reconstruction *targets*, not raw SRA headers

Metappuccino v1.0.0 extracts **19 fields** (Table 1 of the paper). They fall into tiers by how
they're obtained:

| Tier | How it's obtained | Example Table-1 fields |
|------|-------------------|------------------------|
| **T1 — structured** | Read directly from the INSDC core (maybe collapsed to a discrete vocab) | Study accession, Instrument Platform, Number of base pairs, Library strategy, Library selection |
| **T2 — attribute** | Read from a submitter-provided sample/BioSample tag, after harmonizing synonym keys; present only if submitted | Age, Sex, Ethnicity, Cell line, Disease, Biopsy site |
| **T3 — inferred** | Reasoned by the LLM from free text (title / description / paper) when no structured value exists | Is cancer, Organ, Response, Treatment time, Sequencing source |

Most biological fields are really **"T2 if the tag exists, else T3."** Only fields 1–5 are
"free" (structured); the rest are where the LLM earns its keep.

## 3. What SRA actually gives us (via `pysradb`)

A detailed `pysradb` record (`SRAweb().sra_metadata(srp, detailed=True)`) contains **four
buckets** of columns:

1. **INSDC structured core** — accessions, `library_strategy/source/selection/layout`,
   `instrument`/`instrument_model`/`instrument_model_desc`, `organism`/`taxid`, spot/base
   counts. Schema-guaranteed, always populated. Maps ~directly to Table-1 fields 1–5.
   *(Note: "Instrument Platform" is not missing from pysradb — it's `instrument_model_desc`,
   e.g. `ILLUMINA`. pysradb just splits/renames it.)*
2. **File / access info** — `public_*`, `aws_*`, `gcp_*`, `ncbi_*`, `ena_*` URLs and
   checksums. Near-universal pysradb plumbing, **not biology → drop it**.
3. **Submitter attributes** (sample *and* experiment) — optional, highly variable free-key
   tags. These are the harmonization targets (T2).
4. **Free-text descriptions/titles** — `study_title`, `experiment_title`, `experiment_desc`.
   Always present and information-dense; the primary **inference** source (T3).

### Critical `pysradb` detail
- `sra_metadata()` defaults to `detailed=False` and returns only ~24 summary columns.
  **`detailed=True` is required** to fetch the sample/experiment attribute columns
  (`sraweb.py:689-692`). `expand_sample_attributes` / `sample_attribute` params exist but are
  inert in v2.5.1.

## 4. There is no fixed or maximum header set

SRA attributes are arbitrary `TAG`/`VALUE` pairs (`TAG` is `xs:string`) — **any key is valid**,
even `foo`. BioSample "packages" enforce *mandatory* fields (a floor) but allow unlimited
custom attributes (no ceiling); NCBI harmonizes known synonyms and keeps unknown keys verbatim.

Consequences:
- The column set is **emergent per study** — the union of tags seen across that study's
  samples. Different studies → different column names *and* counts (observed: SRP098789 = 24
  summary cols, SRP157974 = 56, SRP035988 = 52, SRP066834 = 53).
- **Never hard-code expected attribute columns.** Treat them as a dynamic key–value bag.
- This permissiveness (the long tail of one-off / synonym keys) is exactly why a fixed lookup
  table can't harmonize everything, and why an LLM is needed.

## 5. Why fields show up blank vs. absent

Three distinct mechanisms:
1. **Submission-driven** — an optional schema field left empty (`sample_title`,
   `library_name`) shows as *present-but-blank*; an *attribute* the submitter never provided
   (e.g. `age`) produces *no column at all* (absent). Both = "data genuinely not in SRA."
2. **pysradb-structural** — pysradb emits both single- and paired-end URL slots and all
   cloud-mirror slots; whichever doesn't apply is blank (e.g. `ena_fastq_http` blank for a
   PAIRED study while `_1`/`_2` are filled). Not a real gap — just plumbing.
3. **Union/merge** — when samples within a study carry heterogeneous tags, every tag becomes a
   column and rows are blank where their sample lacked it.

For reconstruction, only mechanisms **1 and 3** represent real gaps to target; mechanism 2 is
disposable.

### Worked example: `SRP098789` (this repo's default study)
A human **cell-line drug-treatment ribosome-profiling** study — no human subjects. So
`Age`/`Sex`/`Ethnicity`/`Is cancer` are absent even with `detailed=True`. Meanwhile
`Treatment` (`PF-06446846`) and `Treatment time` (`10 min`) exist **only in the experiment
title** — a textbook T3 inference case. Contrast `SRP157974` (breast cancer), which *does*
expose `age`, `sex`, `tissue` columns because its depositor submitted them.

## 6. Architecture for our own version

1. **Define the target schema** — the queryable fields, each with a declared value space
   (discrete/enum vs open-vocab).
2. **Deterministic extraction**, in two parts:
   - **2a. Direct maps** from the structured core (no LLM) — Table-1 fields 1–5.
   - **2b. Attribute harmonization** — synonym lookup table + **LLM fallback** for the long
     tail of unknown keys.
3. **LLM reconstruction** of the remaining fields:
   - **Ground it** — pass known structured values as authoritative anchors so the model can't
     overwrite them.
   - **Constrain outputs** — enum/structured output for discrete fields; optional ontology
     grounding (Disease→MONDO/DOID, tissue→UBERON, cell line→Cellosaurus) for open-vocab.
   - **Record provenance + confidence** per field (direct / harmonized / inferred).
4. **Evaluate** against a hand-labeled gold set (~50–100 samples) with per-field accuracy.

### Cross-cutting constraints
- **Granularity/join** — SRA is study → experiment → sample → run. Extract each field at its
  correct level, then join. Run study-level LLM calls **once per study** (cost), not per
  sample.
- **Source paper** — great for study-level fields but often unavailable (no PubMed link /
  paywalled) and can't be mapped to individual samples. Treat "paper absent" as the common
  case.

---

*One honest caveat: the tier assignment per Table-1 field above is inferred from the field
definitions plus the SRA/BioSample data model, not from reading Metappuccino's source. Confirm
against the paper's Methods before treating it as ground truth.*
