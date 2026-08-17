use std::collections::BTreeMap;

use serde::{Serialize, Deserialize};

// BTreeMap rather than HashMap throughout. The evidence string handed to the model
// is built by serialising these bags, and the API's prompt caching is a prefix
// match, so non-deterministic iteration order silently loses every cache hit.
// It also keeps corpus round-trips byte-reproducible.

// ---------------------------------------------------------------------------
// Accession newtypes
// ---------------------------------------------------------------------------
// Four accession kinds circulate in the same graph. Newtypes make
// `samples[&experiment.accession]` a compile error instead of a None at runtime.
// Ord/Hash are required because SampleAccession keys a BTreeMap.

#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct StudyAccession(pub String);

#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SampleAccession(pub String);

#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ExperimentAccession(pub String);

#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct RunAccession(pub String);

#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct BioSampleAccession(pub String);

#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct BioProjectAccession(pub String);

// The first letter of any INSDC accession encodes the archive of origin. Stored
// rather than derived on demand. Mirroring makes the *data* seamless across the
// three archives but not the *identifiers*: NCBI's efetch mis-resolves non-NCBI
// BioSample and BioProject accessions by matching on the numeric part alone
// (PRJEB13694 -> PRJNA13694, a different project entirely).
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Archive {
    #[default]
    Ncbi,
    Ena,
    Ddbj,
}

// ---------------------------------------------------------------------------
// Dates
// ---------------------------------------------------------------------------
// Eleven date fields across four sources in six formats, of which exactly one
// carries a timezone (BioProject Publication@date). Typing the rest as UTC would
// assert something the archive never said, so the raw string is always kept and
// `granularity` records what the source actually committed to.
//
// The parsed value is deliberately absent: it forces a date-library choice
// (chrono / time / jiff) that belongs to whoever owns this crate. Add a
// `parsed` field alongside `raw` once that is decided; nothing here needs to
// change to accommodate it.

#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum DateGranularity {
    #[default]
    Day,
    Second,
    Millisecond,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ArchiveDate {
    pub raw: String,
    pub granularity: DateGranularity,
}

// Publication@date is the only zoned date in the dataset. Separate type so the
// distinction cannot be lost by assignment.
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ZonedDate {
    pub raw: String,
}

// ---------------------------------------------------------------------------
// Controlled vocabularies
// ---------------------------------------------------------------------------
// Every INSDC vocabulary gets an escape variant. The counts in comments are what
// the 346-study corpus contains; the official lists are longer and grow, so
// exhaustive matching would break on data from outside it.

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum LibraryLayout {
    Single,
    Paired,
    Other(String),
}

// STUDY_TYPE@existing_study_type. 4 of these appear in the 346-study corpus;
// the INSDC list is longer, so Other() is the escape for the rest. Note the
// archive's own literal "Other" is a distinct thing from an unrecognised value.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum StudyType {
    TranscriptomeAnalysis,
    WholeGenomeSequencing,
    Metagenomics,
    ArchiveOther,
    Other(String),
}

// 7 observed. Values are upper-case in the XML.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum LibrarySource {
    Genomic,
    Transcriptomic,
    Metagenomic,
    Metatranscriptomic,
    TranscriptomicSingleCell,
    ViralRna,
    ArchiveOther,
    Other(String),
}

// 8 observed.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Platform {
    Illumina,
    PacbioSmrt,
    OxfordNanopore,
    IonTorrent,
    Bgiseq,
    Dnbseq,
    Genemind,
    Ls454,
    Other(String),
}

// 18 observed of a longer INSDC list.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum LibraryStrategy {
    RnaSeq,
    Wgs,
    Wxs,
    Wga,
    Amplicon,
    RadSeq,
    TargetedCapture,
    ChipSeq,
    MirnaSeq,
    NcrnaSeq,
    BisulfiteSeq,
    AtacSeq,
    TnSeq,
    MbdSeq,
    HiC,
    Clone,
    DnaseHypersensitivity,
    ArchiveOther,
    Other(String),
}

// 21 observed. Case is inconsistent in the source ("other" here, "OTHER" in
// library_strategy), which is one reason these are parsed into variants rather
// than compared as strings.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum LibrarySelection {
    Cdna,
    Random,
    Pcr,
    RandomPcr,
    RtPcr,
    PolyA,
    OligoDt,
    ReducedRepresentation,
    RestrictionDigest,
    SizeFractionation,
    RepeatFractionation,
    HybridSelection,
    InverseRrna,
    Chip,
    Dnase,
    Mbd2MethylCpgBinding,
    Mda,
    Race,
    Cage,
    Unspecified,
    ArchiveOther,
    Other(String),
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum PublicationDb {
    EPubmed,
    EDoi,
    Other(String),
}

// Manual because #[default] cannot sit on a variant carrying data, and neither
// concrete kind is a sensible stand-in for "not yet known".
impl Default for PublicationDb {
    #[inline]
    fn default() -> Self {
        PublicationDb::Other(String::new())
    }
}

// Ours, not the archive's: computed by the harvest's publication classifier.
// Closed by definition, so no escape variant.
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum AccessibilityType {
    Oa,
    Partial,
    Paywall,
    #[default]
    Unknown,
}

// ---------------------------------------------------------------------------
// Root
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Project {
    pub accession: StudyAccession,
    pub archive: Archive,

    pub study: Study,
    pub submission: Option<Submission>,
    pub bioproject: Option<BioProject>,

    // Keyed, because experiments reference samples many-to-many through pools.
    pub samples: BTreeMap<SampleAccession, Sample>,
    // Owns its runs; that relation really is 1:many.
    pub experiments: Vec<Experiment>,

    // Owned, not shared. Only 17 of 326 papers are referenced by more than one
    // study and denormalising costs 540 KB across the whole corpus, so an Arc
    // buys nothing. A bare id would break the no-other-input goal outright.
    // Study-level, not on BioProject: these are classified by the harvest and
    // survive even when the BioProject fetch fails, which is exactly when
    // hanging them off that record would lose them.
    pub publications: Vec<Publication>,

    pub source: SourceMeta,
}

// Provenance of the corpus record itself, not archive data. Needed to tell a
// stale corpus from a fresh one.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SourceMeta {
    pub corpus_format_version: u32,
    // Stamped by the corpus builder in real UTC, unlike every archive date.
    pub fetched_at: ZonedDate,
    pub record_count: Option<u64>,
    // Set when expansion degraded: oversized study, transient fetch failure.
    pub build_note: Option<String>,
}

// ---------------------------------------------------------------------------
// Study, Submission, BioProject
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Study {
    pub title: Option<String>,
    pub abstract_text: Option<String>,
    pub study_type: Option<StudyType>,
    // STUDY@alias, often the GEO series (GSE123534). Maps to the study_alias
    // target field, which the model currently pays to guess.
    pub alias: Option<String>,
    pub center_name: Option<String>,
    // DESCRIPTOR/CENTER_PROJECT_NAME. Maps to the project_name target field.
    pub center_project_name: Option<String>,
    // Derived, not a source field: the earliest RUN@published across every run
    // in the study. The archive has no study-level release date, and reading
    // only the first record was wrong by 12 years on SRP049009 because esearch
    // returns newest-first.
    pub earliest_run_published: Option<ArchiveDate>,
    // IDENTIFIERS/EXTERNAL_ID keyed by @namespace (GEO, Coriell).
    pub external_ids: BTreeMap<String, String>,
    // STUDY_LINKS/STUDY_LINK/XREF_LINK: db -> id.
    pub xrefs: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Submission {
    pub accession: Option<String>,
    pub alias: Option<String>,
    pub center_name: Option<String>,
    pub broker_name: Option<String>,
    pub lab_name: Option<String>,
    pub comment: Option<String>,
    // EXPERIMENT_PACKAGE/Organization: who deposited it.
    pub organization: Option<Organization>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Organization {
    pub name: Option<String>,
    pub abbreviation: Option<String>,
    pub org_type: Option<String>,
    pub contact_email: Option<String>,
    pub contact_first: Option<String>,
    pub contact_last: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct BioProject {
    pub accession: BioProjectAccession,
    // Stored alongside the accession because efetch does not resolve accessions:
    // it strips the PRJ?? prefix and treats the digits as an internal uid. That
    // coincides for PRJNA and is wrong for PRJEB/PRJDB, so the real uid must come
    // from esearch and is what makes the mis-resolution detectable afterwards.
    pub uid: Option<String>,
    pub archive: Archive,
    pub title: Option<String>,
    pub description: Option<String>,
    pub name: Option<String>,
    pub relevance: Option<String>,
    pub model_organism: Option<String>,
    // ProjectTypeSubmission/Target
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
    // ExternalLink/dbXREF: db -> id.
    pub external_links: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Publication {
    pub id: String,
    pub db_type: PublicationDb,
    pub date: Option<ZonedDate>,
    pub status: Option<String>,
    pub reference: Option<String>,
    pub accessibility_type: Option<AccessibilityType>,
    pub paper: Option<Paper>,
}

// ---------------------------------------------------------------------------
// Sample and BioSample
// ---------------------------------------------------------------------------
// Two attribute bags, kept apart on purpose. They overlap heavily and disagree
// often enough that merging would erase which archive said what, and would
// change what the model is credited with inferring. EBI additionally normalises
// key style (`geo loc name` vs `geo_loc_name`), so compare them normalised
// before concluding one adds a field.

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Sample {
    pub accession: SampleAccession,
    pub alias: Option<String>,
    pub title: Option<String>,
    // String on purpose, even though every one of the 88,560 values in the
    // corpus is integral. `Project` mirrors the wire: the archive hands this
    // over as text, and parsing during deserialisation would let one malformed
    // id anywhere in a 630 MB file abort the whole load. `TargetSchema` holds
    // the interpreted `u64` and parses at that boundary, where a bad value can
    // be handled per record. Do not "align" the two by changing this.
    pub taxon_id: Option<String>,
    pub scientific_name: Option<String>,
    pub biosample_id: Option<BioSampleAccession>,
    // SAMPLE_ATTRIBUTES, the submitter's open bag. 619 distinct keys observed
    // across 88,560 samples. Values are verbatim, including literal
    // "not applicable" strings, which are evidence the model is meant to read.
    pub attributes: BTreeMap<String, String>,
    pub external_ids: BTreeMap<String, String>,
    pub xrefs: BTreeMap<String, String>,
    // The BioSample record's own bag, kept here beside the SRA bag rather than
    // inside BioSample so the two views sit side by side. They overlap heavily
    // and disagree often enough to be worth telling apart.
    pub biosample_attributes: BTreeMap<String, String>,
    pub biosample: Option<BioSample>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct BioSample {
    pub accession: BioSampleAccession,
    pub archive: Archive,
    pub title: Option<String>,
    pub organism_name: Option<String>,
    pub taxonomy_id: Option<String>,
    // Which checklist governs the record (MIMS.me, Human, Pathogen.cl). Decides
    // which attributes were mandatory, which is signal about why one is absent.
    pub package: Option<String>,
    pub models: Vec<String>,
    pub owner: Option<String>,
    pub contact_first: Option<String>,
    pub contact_last: Option<String>,
    pub status: Option<String>,
    pub status_when: Option<ArchiveDate>,
    pub access: Option<String>,
    pub submission_date: Option<ArchiveDate>,
    pub publication_date: Option<ArchiveDate>,
    pub last_update: Option<ArchiveDate>,
    // Ids/Id keyed by @db.
    pub ids: BTreeMap<String, String>,
    pub links: BTreeMap<String, String>,
    // Attribute@harmonized_name -> value: NCBI's normalisation of the submitter's
    // spelling. A separate view, not a replacement. The submitter's own bag is
    // on Sample::biosample_attributes.
    pub harmonized: BTreeMap<String, String>,
}

// ---------------------------------------------------------------------------
// Experiment and Run
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Experiment {
    pub accession: ExperimentAccession,
    pub alias: Option<String>,
    pub title: Option<String>,
    // Vec because a pooled experiment references several samples. Zero of the
    // 102,240 experiments in the current corpus do, but the relation is
    // many-to-many in SRA and collapsing it would lose a pooled study's split.
    pub sample_ids: Vec<SampleAccession>,
    pub design_description: Option<String>,
    pub library_name: Option<String>,
    pub library_strategy: Option<LibraryStrategy>,
    pub library_source: Option<LibrarySource>,
    pub library_selection: Option<LibrarySelection>,
    pub library_layout: Option<LibraryLayout>,
    pub library_construction_protocol: Option<String>,
    pub platform: Option<Platform>,
    // 37 values in the corpus and unbounded in practice; nothing branches on it.
    pub instrument_model: Option<String>,
    // EXPERIMENT_ATTRIBUTES. Only 4 distinct keys corpus-wide, but still a bag.
    pub attributes: BTreeMap<String, String>,
    pub xrefs: BTreeMap<String, String>,
    pub pool_members: Vec<PoolMember>,
    pub runs: Vec<Run>,
}

// Pool/Member, present at both experiment and run level. Records how reads split
// across samples when several are multiplexed into one run. Always exactly one
// member in the current corpus.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
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

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Run {
    pub accession: RunAccession,
    pub alias: Option<String>,
    // RUN@published. Bumps on re-release, so it is not the Entrez pdat that
    // date-filtered searches use, and a 2014 study can read 2026 here.
    pub published: Option<ArchiveDate>,
    pub is_public: Option<bool>,
    pub size_bytes: Option<u64>,
    pub total_spots: Option<u64>,
    pub total_bases: Option<u64>,
    pub cluster_name: Option<String>,
    pub statistics: Option<RunStatistics>,
    // Bases/Base: @value -> @count.
    pub base_composition: BTreeMap<String, u64>,
    pub files: Vec<FileRef>,
    pub cloud_files: Vec<CloudFile>,
    pub submitter_id: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RunStatistics {
    pub n_reads: Option<u32>,
    pub n_spots: Option<u64>,
    pub reads: Vec<ReadStat>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ReadStat {
    pub index: u32,
    pub count: Option<u64>,
    pub average: Option<f64>,
    pub stdev: Option<f64>,
}

// One downloadable artifact of a run. A run has several: the submitter's
// `Original` uploads and NCBI's `Primary ETL` normalisations (.sra, .sralite).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FileRef {
    // Frequently None on Original uploads, which exist only via requester-pays
    // cloud delivery. That is why alternatives cannot collapse into this field.
    pub url: Option<String>,
    pub filename: Option<String>,
    pub md5: Option<String>,
    pub size: Option<u64>,
    pub file_date: Option<ArchiveDate>,
    pub semantic_name: Option<String>,
    pub supertype: Option<String>,
    pub sratoolkit: Option<bool>,
    pub alternatives: Vec<FileAlternative>,
}

// The same file at a different host, with different access and egress terms
// (NCBI anonymous/worldwide vs AWS or GCP requester-pays, region-locked).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FileAlternative {
    pub url: Option<String>,
    pub org: Option<String>,
    pub access_type: Option<String>,
    pub free_egress: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CloudFile {
    pub provider: Option<String>,
    pub location: Option<String>,
    pub filetype: Option<String>,
}

// ---------------------------------------------------------------------------
// Paper
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Paper {
    pub id: String,
    pub db_type: PublicationDb,
    pub text: Option<String>,
    // 91% of stored papers hit the 30,000-char cap, so code treating this as a
    // whole document is reasoning over a fragment.
    pub truncated: bool,
    pub char_count: usize,
    // Why it is retrievable or not: gold/hybrid deposits reach PMC, bronze
    // (free at the publisher) and green (repository) do not, and both classify
    // as open access while yielding no text through the Europe PMC route.
    pub oa_status: Option<String>,
}