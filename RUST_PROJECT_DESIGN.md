# The Rust `Project` — input model

A draft of the **input** type: everything the four sources can tell us about one study,
in one owned value. Not the output `TargetSchema` — that is a separate document.

The goal is that `Project::reconstruct(&self, cfg: &ModelConfig) -> Vec<TargetRecord>`
needs no other argument. Everything it reads lives here.

Field inventory is drawn from live records, not documentation: 199 distinct
element/attribute paths in an SRA `EXPERIMENT_PACKAGE`, 40 in a BioSample record,
60 in a BioProject record, plus the paper. The coverage table at the end maps every
one of them to a home.

---

## Principles applied

| principle | where it lands |
|---|---|
| Struct fields only for data guaranteed present and valid | accessions, and little else — nearly everything archive-supplied is `Option` |
| Maps for open, unbounded bags | 619 distinct sample-attribute keys measured across 88,560 samples; must be a map |
| Enums only where the vocabulary is genuinely bounded, always with an escape | `Other(String)` on every one |
| Sub-objects for each archive object | `Study`, `Sample`, `BioSample`, `Experiment`, `Run`, `Submission`, `BioProject`, `Paper` |
| ID references, not nested ownership, where the relation is many-to-many | `Experiment.sample_ids` → `Project.samples` |

**`Option<T>`, not the `Field<T>` missing-value enum.** That enum belongs to the *output*.
An archive record either states a thing or does not; there is no "not applicable"
determination at input time. Where a submitter literally typed `not applicable` into an
attribute bag, it is preserved verbatim as a map value — normalising it here would destroy
evidence the model is supposed to read.

**`BTreeMap`, not `HashMap`.** Deterministic iteration is not cosmetic here. The evidence
string handed to the model is built by serialising the attribute bag, and the API's prompt
caching is a **prefix match** — a map that iterates in a different order between runs
produces a different prefix and silently loses every cache hit. It also makes runs
byte-reproducible, which the benchmark work depends on.

---

## Newtypes

Four accession kinds circulate in the same graph and are trivially confusable. Newtypes cost
nothing and make `samples[&experiment.accession]` a compile error.

```rust
pub struct StudyAccession(pub String);       // SRP…/ERP…/DRP…
pub struct SampleAccession(pub String);      // SRS…/ERS…/DRS…
pub struct ExperimentAccession(pub String);  // SRX…/ERX…/DRX…
pub struct RunAccession(pub String);         // SRR…/ERR…/DRR…
pub struct BioSampleAccession(pub String);   // SAMN…/SAMEA…/SAMD…
pub struct BioProjectAccession(pub String);  // PRJNA…/PRJEB…/PRJDB…

/// First letter of any INSDC accession encodes the archive of origin.
/// Worth having as a type: the mirroring is seamless for *data* and not for
/// *identifiers* — NCBI's efetch mis-resolves non-NCBI BioSample and BioProject
/// accessions by matching on the numeric part alone.
pub enum Archive { Ncbi, Ena, Ddbj }
```

---

## Root

```rust
pub struct Project {
    /// The only field guaranteed present and valid.
    pub accession: StudyAccession,
    pub archive: Archive,

    pub study: Study,
    pub submission: Option<Submission>,
    pub bioproject: Option<BioProject>,

    /// Keyed, because experiments reference samples many-to-many through pools.
    pub samples: BTreeMap<SampleAccession, Sample>,
    /// Owns its runs (that relation really is 1:many).
    pub experiments: Vec<Experiment>,

    /// Owned, not shared. Only 17 of 326 papers are referenced by more than one
    /// study and denormalising costs 540 KB across the whole corpus — not worth
    /// an Arc, and a bare id would break the no-other-input goal.
    pub paper: Option<Paper>,

    pub source: SourceMeta,
}

/// Provenance of the corpus record itself — when it was fetched and with what.
/// Not archive data; needed to tell a stale corpus from a fresh one.
pub struct SourceMeta {
    pub corpus_format_version: u32,
    pub fetched_at: DateTime<Utc>,
    pub record_count: Option<u64>,
    /// Set when expansion degraded (oversized study, transient failure).
    pub build_note: Option<String>,
}
```

---

## Study, Submission, BioProject

```rust
pub struct Study {
    pub title: Option<String>,
    pub abstract_text: Option<String>,
    pub study_type: Option<StudyType>,
    /// STUDY@alias — often the GEO series (GSE123534). Currently dropped by the
    /// Python parser; maps to the `study_alias` target field.
    pub alias: Option<String>,
    pub center_name: Option<String>,
    /// DESCRIPTOR/CENTER_PROJECT_NAME — maps to the `project_name` target field.
    pub center_project_name: Option<String>,
    /// DERIVED, not a source field: the earliest RUN@published across every run
    /// in the study. The archive has no study-level release date. See Dates.
    pub earliest_run_published: Option<ArchiveDate>,
    /// IDENTIFIERS/EXTERNAL_ID, keyed by @namespace (GEO, Coriell, …).
    pub external_ids: BTreeMap<String, String>,
    /// STUDY_LINKS/STUDY_LINK/XREF_LINK — db → id.
    pub xrefs: BTreeMap<String, String>,
}

pub struct Submission {
    pub accession: Option<String>,          // SRA822677
    pub alias: Option<String>,
    pub center_name: Option<String>,
    pub broker_name: Option<String>,        // e.g. "GEO"
    pub lab_name: Option<String>,
    pub comment: Option<String>,
    /// EXPERIMENT_PACKAGE/Organization — who deposited it.
    pub organization: Option<Organization>,
}

pub struct Organization {
    pub name: Option<String>,
    pub abbreviation: Option<String>,
    pub org_type: Option<String>,
    pub contact_email: Option<String>,
    pub contact_first: Option<String>,
    pub contact_last: Option<String>,
}

pub struct BioProject {
    pub accession: BioProjectAccession,
    pub uid: Option<String>,
    pub archive: Archive,
    pub title: Option<String>,
    pub description: Option<String>,
    pub name: Option<String>,
    pub relevance: Option<String>,
    pub model_organism: Option<String>,
    /// ProjectTypeSubmission/Target
    pub target_organism: Option<String>,
    pub target_taxid: Option<String>,
    pub target_sample_scope: Option<String>,
    pub target_material: Option<String>,
    pub target_capture: Option<String>,
    pub method_type: Option<String>,
    pub data_types: Vec<String>,
    pub objectives: Vec<String>,
    pub submitting_organization: Option<String>,
    pub submitted: Option<ArchiveDate>,
    pub last_update: Option<ArchiveDate>,
    /// ExternalLink/dbXREF — db → id.
    pub external_links: BTreeMap<String, String>,
    pub publications: Vec<Publication>,
}

pub struct Publication {
    pub id: String,
    pub db_type: PublicationDb,             // ePubmed (PMID) / eDOI
    pub date: Option<ZonedDate>,                    // the one zoned date
    pub status: Option<String>,
    pub reference: Option<String>,
    /// Ours, not the archive's — computed by the harvest's stage 2.
    pub accessibility: Option<Accessibility>,
}
```

---

## Sample and BioSample

Two bags, kept apart on purpose. They overlap heavily and disagree often enough that merging
would erase which archive said what — and would change what the model is credited with
inferring. EBI additionally normalises key style (`geo loc name` vs `geo_loc_name`), so
compare them normalised before concluding one adds a field.

```rust
pub struct Sample {
    pub accession: SampleAccession,
    pub alias: Option<String>,              // SAMPLE@alias, e.g. GSM3506244
    pub title: Option<String>,
    pub taxon_id: Option<String>,
    pub scientific_name: Option<String>,
    pub biosample_id: Option<BioSampleAccession>,
    /// SAMPLE_ATTRIBUTES — the submitter's open EAV bag. 619 distinct keys observed.
    /// Values preserved verbatim, including literal "not applicable" strings.
    pub attributes: BTreeMap<String, String>,
    pub external_ids: BTreeMap<String, String>,
    pub xrefs: BTreeMap<String, String>,    // SAMPLE_LINKS/XREF_LINK
    /// Present only when the BioSample record was fetched.
    pub biosample: Option<BioSample>,
}

pub struct BioSample {
    pub accession: BioSampleAccession,
    pub archive: Archive,
    pub title: Option<String>,
    pub organism_name: Option<String>,
    pub taxonomy_id: Option<String>,
    /// Which checklist governs it (MIMS.me, Human, Pathogen.cl …). Determines
    /// which attributes were *mandatory*, which is useful signal about absence.
    pub package: Option<String>,
    pub models: Vec<String>,
    pub owner: Option<String>,
    pub contact_first: Option<String>,
    pub contact_last: Option<String>,
    pub status: Option<String>,
    pub status_when: Option<ArchiveDate>,          // see Dates
    pub access: Option<String>,
    pub submission_date: Option<ArchiveDate>,
    pub publication_date: Option<ArchiveDate>,
    pub last_update: Option<ArchiveDate>,
    /// Ids/Id keyed by @db, with the primary flagged.
    pub ids: BTreeMap<String, String>,
    pub links: BTreeMap<String, String>,
    /// Attribute@attribute_name → value (the submitter's spelling).
    pub attributes: BTreeMap<String, String>,
    /// Attribute@harmonized_name → value. NCBI's normalisation, a *separate* view
    /// of the same data — keep both, they are not interchangeable.
    pub harmonized: BTreeMap<String, String>,
}
```

---

## Experiment and Run

```rust
pub struct Experiment {
    pub accession: ExperimentAccession,
    pub alias: Option<String>,              // EXPERIMENT@alias
    pub title: Option<String>,
    /// Vec because a pooled experiment references several samples.
    pub sample_ids: Vec<SampleAccession>,
    pub design_description: Option<String>,
    pub library_name: Option<String>,
    pub library_strategy: Option<LibraryStrategy>,
    pub library_source: Option<LibrarySource>,
    pub library_selection: Option<LibrarySelection>,
    pub library_layout: Option<LibraryLayout>,
    pub library_construction_protocol: Option<String>,
    pub platform: Option<Platform>,
    /// 37 distinct values and growing with every new sequencer — String, not enum.
    pub instrument_model: Option<String>,
    /// EXPERIMENT_ATTRIBUTES. Only 4 distinct keys corpus-wide, but still a bag.
    pub attributes: BTreeMap<String, String>,
    pub xrefs: BTreeMap<String, String>,
    pub pool_members: Vec<PoolMember>,
    pub runs: Vec<Run>,
}

/// Pool/Member — how a pooled experiment splits across samples.
pub struct PoolMember {
    pub sample_accession: Option<SampleAccession>,
    pub member_name: Option<String>,
    pub sample_name: Option<String>,
    pub sample_title: Option<String>,
    pub organism: Option<String>,
    pub tax_id: Option<String>,
    pub spots: Option<u64>,
    pub bases: Option<u64>,
}

pub struct Run {
    pub accession: RunAccession,
    pub alias: Option<String>,
    pub published: Option<ArchiveDate>,             // RUN@published; bumps on re-release
    pub is_public: Option<bool>,
    pub size_bytes: Option<u64>,
    pub total_spots: Option<u64>,
    pub total_bases: Option<u64>,
    pub cluster_name: Option<String>,
    pub statistics: Option<RunStatistics>,
    pub base_composition: BTreeMap<String, u64>,   // Bases/Base @value → @count
    pub files: Vec<FileRef>,
    pub cloud_files: Vec<CloudFile>,
    pub submitter_id: Option<String>,
}

pub struct RunStatistics {
    pub n_reads: Option<u32>,
    pub n_spots: Option<u64>,
    pub reads: Vec<ReadStat>,               // per-read index/count/average/stdev
}

pub struct ReadStat {
    pub index: u32,
    pub count: Option<u64>,
    pub average: Option<f64>,
    pub stdev: Option<f64>,
}

pub struct FileRef {
    pub url: Option<String>,
    pub filename: Option<String>,
    pub md5: Option<String>,
    pub size: Option<u64>,
    pub date: Option<ZonedDate>,                    // the one zoned date
    pub semantic_name: Option<String>,
    pub supertype: Option<String>,
    pub sratoolkit: Option<bool>,
    pub alternatives: Vec<FileAlternative>,
}

pub struct FileAlternative {
    pub url: Option<String>,
    pub org: Option<String>,
    pub access_type: Option<String>,
    pub free_egress: Option<String>,
}

pub struct CloudFile {
    pub provider: Option<String>,           // s3 / gs
    pub location: Option<String>,
    pub filetype: Option<String>,
}
```

---

## Paper

```rust
pub struct Paper {
    pub id: String,                          // PMID or DOI
    pub db_type: PublicationDb,
    pub text: Option<String>,
    /// 91% of stored papers hit the 30,000-char cap. Code that treats this as a
    /// whole document is reasoning over a fragment; make that impossible to miss.
    pub truncated: bool,
    pub char_count: usize,
    /// Why it is retrievable (or not): gold/hybrid deposit to PMC, vs bronze at the
    /// publisher or green in a repository — the latter two are classified open
    /// access but yield no text through the Europe PMC route.
    pub oa_status: Option<String>,
}
```

---

## Enums

Every INSDC vocabulary gets an escape variant. The counts below are what this corpus
actually contains; the official lists are longer and grow, so exhaustive matching would
break on data from outside it.

```rust
pub enum LibraryLayout   { Single, Paired, Other(String) }                    // 2 observed
pub enum StudyType       { /* 4 observed */ Other(String) }
pub enum LibrarySource   { /* 7 observed */ Other(String) }
pub enum Platform        { /* 8 observed */ Other(String) }
pub enum LibraryStrategy { /* 18 observed */ Other(String) }
pub enum LibrarySelection{ /* 21 observed */ Other(String) }

pub enum PublicationDb   { EPubmed, EDoi, Other(String) }
/// Ours, closed by definition — no escape variant needed.
pub enum Accessibility   { Oa, Partial, Paywall, Unknown }
```

`instrument_model` stays `String`: 37 values in this corpus, unbounded in practice, and
nothing branches on it.

---

## Dates

Eleven date fields across four sources, in **six different formats**, and — measured on
live records — **exactly one of them carries a timezone**:

| source path | example value | format | tz? |
|---|---|---|---|
| SRA `RUN@published` | `2020-05-21 16:29:22` | space-separated | no |
| SRA `SRAFile@date` | `2020-05-24 16:28:12` | space-separated | no |
| BioSample `@submission_date` | `2018-12-10T08:06:05.510` | ISO-T, millis | no |
| BioSample `@publication_date` | `2019-10-15T00:00:00.000` | ISO-T, millis | no |
| BioSample `@last_update` | `2019-10-15T10:24:38.287` | ISO-T, millis | no |
| BioSample `Status@when` | `2019-10-15T00:50:07.373` | ISO-T, millis | no |
| BioProject `Submission@submitted` | `2018-12-10` | date only | n/a |
| BioProject `Submission@last_update` | `2020-05-24` | date only | n/a |
| BioProject `Publication@date` | `2020-05-24T00:00:00Z` | ISO-8601 | **yes** |
| Entrez esummary `CreateDate` | `2018/02/17` | slash-separated | no |
| Entrez esummary `UpdateDate` | `2018/02/16` | slash-separated | no |

Three rules follow.

**1. Do not type these as `DateTime<Utc>`.** Ten of the eleven are timezone-naive; calling
them UTC asserts something the archive never said. Use `NaiveDateTime` (or `NaiveDate` for
the date-only ones), and reserve `DateTime<Utc>` for the single field that is actually
zoned.

**2. Name every field for its source and its meaning.** There is no such thing as "the"
publication date of a study. The three that get confused:

* `RUN@published` — when *that run's data* was released. **Bumps on re-release**, so a
  study first released in 2014 can read 2026.
* Entrez `pdat` — the release date that `esearch` date filters actually use. Does **not**
  bump. Not present in the XML at all; it is an Entrez-level concept.
* esummary `CreateDate` / `UpdateDate` — Entrez *record* dates, which bump on re-index.

A `before_date=2024` search therefore correctly returns studies whose `published` reads
2025 or 2026. That is not a bug, and a field called plain `published` invites treating it
as one.

**3. Keep the raw string.** A parse that fails, or that silently coerces, is unrecoverable
once the source text is gone. The wrapper costs a `String` per date and makes every
mis-parse debuggable:

```rust
pub struct ArchiveDate {
    /// Exactly as the source wrote it. Never normalised.
    pub raw: String,
    /// None when the raw form did not parse — keep the record, flag the field.
    pub parsed: Option<NaiveDateTime>,
    pub granularity: DateGranularity,
}

pub enum DateGranularity { Day, Second, Millisecond }

/// The one zoned field in the whole dataset.
pub struct ZonedDate { pub raw: String, pub parsed: Option<DateTime<Utc>> }
```

### Where each date lands

```rust
pub struct Study {
    /// DERIVED, not a source field: the *earliest* RUN@published across every run
    /// in the study. Named for what it is because the archive has no study-level
    /// release date, and because the naive reading (the first record's date) was
    /// wrong by 12 years on SRP049009 — esearch returns newest-first.
    pub earliest_run_published: Option<ArchiveDate>,
    // …
}

pub struct Run {
    /// RUN@published. Bumps on re-release; not the Entrez pdat.
    pub published: Option<ArchiveDate>,
    // …
}

pub struct FileRef {
    pub file_date: Option<ArchiveDate>,          // SRAFile@date
    // …
}

pub struct BioSample {
    pub submission_date: Option<ArchiveDate>,    // @submission_date
    pub publication_date: Option<ArchiveDate>,   // @publication_date
    pub last_update: Option<ArchiveDate>,        // @last_update
    pub status_when: Option<ArchiveDate>,        // Status@when
    // …
}

pub struct BioProject {
    pub submitted: Option<ArchiveDate>,          // Submission@submitted (day granularity)
    pub last_update: Option<ArchiveDate>,        // Submission@last_update (day granularity)
    // …
}

pub struct Publication {
    /// The only timezone-bearing date in the dataset.
    pub date: Option<ZonedDate>,                 // Publication@date
    // …
}

pub struct SourceMeta {
    /// Ours, not the archive's — stamped by the corpus builder in real UTC.
    pub fetched_at: DateTime<Utc>,
    // …
}
```

Entrez `CreateDate` / `UpdateDate` are deliberately **not** in the type. They are properties
of the Entrez index rather than of the study, they bump on re-index, and nothing in
reconstruction reads them. Add them only if something starts to.

---

## Source coverage

Every enumerated path from all four sources, and where it lands. **"gap"** means the
value exists at the source but the current Python corpus does not carry it — those
require Python-side changes before this type can be filled.

| source | paths | status |
|---|---|---|
| SRA `STUDY` | accession, alias, center_name, title, abstract, type, external_ids, links | alias / center_name / CENTER_PROJECT_NAME are **gaps** |
| SRA `SAMPLE` | accession, alias, title, taxon, scientific_name, attributes, external_ids, links | alias, xrefs are **gaps** |
| SRA `EXPERIMENT` | accession, alias, title, design, library_*, platform, instrument, attributes, links | alias, design_description, construction_protocol, xrefs are **gaps** |
| SRA `RUN` | accession, alias, published, size, spots, bases, statistics, bases composition, files, cloud files | alias, statistics, composition, cloud files, is_public are **gaps** |
| SRA `SUBMISSION` | accession, alias, center, broker, lab, comment | entire object is a **gap** |
| SRA `Pool/Member` | accession, names, organism, taxid, spots, bases | entire object is a **gap** |
| SRA `Organization` | name, abbr, type, contact | entire object is a **gap** |
| BioSample | 40 paths: ids, description, organism, package, models, owner, status, dates, attributes, harmonized, links | only `attributes` carried; rest are **gaps** |
| BioProject | 60 paths: title, description, relevance, target, method, data types, objectives, org, dates, links, publications | only accession + publications carried; rest are **gaps** |
| Paper | id, type, text, chars | `oa_status` and an explicit `truncated` flag are **gaps** |

---

## Decisions settled

1. **Enums** — keep the enum-with-`Other(String)` layout as drafted; `instrument_model`
   stays `String`. Revisit per-vocabulary later if a match arm actually wants it.
2. **`Study`** stays a sub-object of `Project`, mirroring the archive's own structure.
3. **`Archive`** is stored as a field rather than derived from the prefix on demand.
4. **Dates** — see the section below. Named per source, typed by what the source actually
   guarantees, and never silently normalised.
5. **`PoolMember` stays**, though this corpus exercises none of it: 0 of 102,240
   experiments reference more than one sample, so `Pool` is always present with exactly
   one member. Kept because it is real SRA structure and it is why `sample_ids` is a
   `Vec` — a pooled study would otherwise silently lose its sample split. Removable later
   if pooled data never appears.
6. **File metadata stays** (`FileRef`, `FileAlternative`, `CloudFile`), though nothing in
   reconstruction reads it — no prompt infers tissue type from an md5. It is currently
   33% of the corpus by bytes (458,864 file refs, 67 MB of 203 MB) and the full form is
   3–4x that, so it is the largest block of inert data in the type. Kept for the
   everything-in-one-place goal; the first thing to cut if size becomes a problem.

## What must change on the Python side

The type above cannot be filled from the current corpus. To populate it:

- parse the ~9 dropped SRA fields that map to target-schema fields (aliases,
  center/broker/submission, construction protocol) plus design description, pool members,
  run statistics, organization, and xrefs;
- capture the full BioSample record rather than only its attribute bag;
- fetch the BioProject record for real, rather than only to extract publications;
- record `oa_status` and a `truncated` flag on papers.

That is a `project.py` change and a corpus rebuild (~3 hours, free).
