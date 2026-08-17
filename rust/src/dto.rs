// Wire types. These mirror the corpus JSON exactly — field for field, name for
// name — and exist only to be converted into the domain types in `project`.
//
// A direct Deserialize onto the domain types will not work. The two shapes
// differ in four ways that no amount of #[serde(rename)] fixes:
//
//   * `papers` is a map keyed by publication id; the domain wants owned Papers,
//     reached through each publication's `paper_ids`.
//   * a study object is flat, while `Project` nests its study fields under
//     `study`.
//   * controlled vocabularies arrive as strings and become enums.
//   * dates arrive as bare strings and become ArchiveDate/ZonedDate, which
//     carry the granularity the source committed to.
//
// Keeping that translation here means the domain types never have to be shaped
// by what the file happens to look like.

use serde::Deserialize;
use std::collections::BTreeMap;

use crate::corpus::{CorpusCounts, CorpusParams};
use crate::project::*;


// Attribute bags can carry a null value: a submitter wrote a TAG with an empty
// VALUE, which the harvest stores faithfully because present-but-blank is not
// the same fact as absent — the submitter considered the field and had nothing
// to put in it. 87 such entries exist in the reference corpus.
//
// Mapped to an empty string rather than dropped, so the key survives and the
// domain map stays a plain String->String. Callers that care about the
// distinction test for `is_empty()`; callers that do not are unaffected.
fn string_map<'de, D>(d: D) -> Result<BTreeMap<String, String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let raw = BTreeMap::<String, Option<String>>::deserialize(d)?;
    Ok(raw
        .into_iter()
        .map(|(k, v)| (k, v.unwrap_or_default()))
        .collect())
}

// ---------------------------------------------------------------------------
// Wire shapes
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct CorpusDto {
    pub format_version: u32,
    pub created: String,
    pub params: CorpusParams,
    pub counts: CorpusCounts,
    pub papers: BTreeMap<String, PaperDto>,
    pub studies: Vec<StudyDto>,
}

#[derive(Debug, Deserialize)]
pub struct PaperDto {
    pub id: String,
    #[serde(rename = "type")]
    pub db_type: Option<String>,
    pub text: Option<String>,
    pub chars: usize,
    pub truncated: bool,
    pub oa_status: Option<String>,
}

// The study object is flat on the wire: study-level fields sit beside samples,
// experiments and the corpus's own bookkeeping.
#[derive(Debug, Deserialize)]
pub struct StudyDto {
    pub accession: String,
    pub bioproject: Option<String>,
    pub title: Option<String>,
    #[serde(rename = "abstract")]
    pub abstract_text: Option<String>,
    pub study_type: Option<String>,
    pub published: Option<String>,
    pub record_count: Option<u64>,
    #[serde(deserialize_with = "string_map")]
    pub external_ids: BTreeMap<String, String>,
    pub study_alias: Option<String>,
    pub center_name: Option<String>,
    pub center_project_name: Option<String>,
    #[serde(deserialize_with = "string_map")]
    pub xrefs: BTreeMap<String, String>,
    pub submission: Option<SubmissionDto>,
    pub bioproject_record: Option<BioProjectDto>,
    pub samples: BTreeMap<String, SampleDto>,
    pub experiments: Vec<ExperimentDto>,
    pub publications: Vec<PublicationDto>,
    pub build_note: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct SubmissionDto {
    pub accession: Option<String>,
    pub alias: Option<String>,
    pub center_name: Option<String>,
    pub broker_name: Option<String>,
    pub lab_name: Option<String>,
    pub comment: Option<String>,
    pub organization: Option<OrganizationDto>,
}

#[derive(Debug, Deserialize)]
pub struct OrganizationDto {
    pub name: Option<String>,
    pub abbreviation: Option<String>,
    pub org_type: Option<String>,
    pub contact_email: Option<String>,
    pub contact_first: Option<String>,
    pub contact_last: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct BioProjectDto {
    pub accession: String,
    pub uid: Option<String>,
    pub title: Option<String>,
    pub description: Option<String>,
    pub name: Option<String>,
    pub relevance: Option<String>,
    pub model_organism: Option<String>,
    pub target_organism: Option<String>,
    pub target_taxid: Option<String>,
    pub target_sample_scope: Option<String>,
    pub target_material: Option<String>,
    pub target_capture: Option<String>,
    pub method_type: Option<String>,
    pub data_types: Vec<String>,
    pub objectives: Vec<String>,
    pub submitting_organization: Option<String>,
    pub submitted: Option<String>,
    pub last_update: Option<String>,
    #[serde(deserialize_with = "string_map")]
    pub external_links: BTreeMap<String, String>,
}

#[derive(Debug, Deserialize)]
pub struct PublicationDto {
    pub id: String,
    #[serde(rename = "type")]
    pub db_type: Option<String>,
    pub accessibility_type: Option<String>,
    pub date: Option<String>,
    pub status: Option<String>,
    pub reference: Option<String>,
    pub oa_status: Option<String>,
    // Keys into the corpus-level `papers` map. Usually empty; a list because a
    // study can carry several open-access papers (37 of 346 do).
    pub paper_ids: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct SampleDto {
    pub accession: String,
    pub alias: Option<String>,
    pub title: Option<String>,
    pub taxon_id: Option<String>,
    pub scientific_name: Option<String>,
    pub biosample: Option<String>,
    #[serde(deserialize_with = "string_map")]
    pub attributes: BTreeMap<String, String>,
    #[serde(deserialize_with = "string_map")]
    pub external_ids: BTreeMap<String, String>,
    #[serde(deserialize_with = "string_map")]
    pub xrefs: BTreeMap<String, String>,
    #[serde(deserialize_with = "string_map")]
    pub biosample_attributes: BTreeMap<String, String>,
    pub biosample_record: Option<BioSampleDto>,
}

#[derive(Debug, Deserialize)]
pub struct BioSampleDto {
    pub accession: String,
    pub title: Option<String>,
    pub organism_name: Option<String>,
    pub taxonomy_id: Option<String>,
    pub package: Option<String>,
    pub models: Vec<String>,
    pub owner: Option<String>,
    pub contact_first: Option<String>,
    pub contact_last: Option<String>,
    pub status: Option<String>,
    pub status_when: Option<String>,
    pub access: Option<String>,
    pub submission_date: Option<String>,
    pub publication_date: Option<String>,
    pub last_update: Option<String>,
    #[serde(deserialize_with = "string_map")]
    pub ids: BTreeMap<String, String>,
    #[serde(deserialize_with = "string_map")]
    pub links: BTreeMap<String, String>,
    #[serde(deserialize_with = "string_map")]
    pub harmonized: BTreeMap<String, String>,
}

#[derive(Debug, Deserialize)]
pub struct ExperimentDto {
    pub accession: String,
    pub alias: Option<String>,
    pub title: Option<String>,
    pub sample_ids: Vec<String>,
    pub design_description: Option<String>,
    pub library_name: Option<String>,
    pub library_strategy: Option<String>,
    pub library_source: Option<String>,
    pub library_selection: Option<String>,
    pub library_layout: Option<String>,
    pub library_construction_protocol: Option<String>,
    pub platform: Option<String>,
    pub instrument_model: Option<String>,
    #[serde(deserialize_with = "string_map")]
    pub attributes: BTreeMap<String, String>,
    #[serde(deserialize_with = "string_map")]
    pub xrefs: BTreeMap<String, String>,
    pub pool_members: Vec<PoolMemberDto>,
    pub runs: Vec<RunDto>,
}

#[derive(Debug, Deserialize)]
pub struct PoolMemberDto {
    pub accession: Option<String>,
    pub member_name: Option<String>,
    pub sample_name: Option<String>,
    pub sample_title: Option<String>,
    pub organism: Option<String>,
    pub tax_id: Option<String>,
    pub spots: Option<u64>,
    pub bases: Option<u64>,
}

#[derive(Debug, Deserialize)]
pub struct RunDto {
    pub accession: String,
    pub alias: Option<String>,
    pub published: Option<String>,
    pub is_public: Option<bool>,
    pub size_bytes: Option<u64>,
    pub total_spots: Option<u64>,
    pub total_bases: Option<u64>,
    pub cluster_name: Option<String>,
    pub submitter_id: Option<String>,
    pub statistics: Option<RunStatisticsDto>,
    pub base_composition: BTreeMap<String, u64>,
    pub files: Vec<FileRefDto>,
    pub cloud_files: Vec<CloudFileDto>,
}

#[derive(Debug, Deserialize)]
pub struct RunStatisticsDto {
    pub n_reads: Option<u32>,
    pub n_spots: Option<u64>,
    pub reads: Vec<ReadStatDto>,
}

#[derive(Debug, Deserialize)]
pub struct ReadStatDto {
    pub index: Option<u32>,
    pub count: Option<u64>,
    pub average: Option<f64>,
    pub stdev: Option<f64>,
}

#[derive(Debug, Deserialize)]
pub struct FileRefDto {
    pub url: Option<String>,
    pub md5: Option<String>,
    pub size: Option<u64>,
    pub source: Option<String>,
    pub filename: Option<String>,
    pub file_date: Option<String>,
    pub semantic_name: Option<String>,
    pub supertype: Option<String>,
    pub sratoolkit: Option<String>,
    pub alternatives: Vec<FileAlternativeDto>,
}

#[derive(Debug, Deserialize)]
pub struct FileAlternativeDto {
    pub url: Option<String>,
    pub org: Option<String>,
    pub access_type: Option<String>,
    pub free_egress: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct CloudFileDto {
    pub provider: Option<String>,
    pub location: Option<String>,
    pub filetype: Option<String>,
}

// ---------------------------------------------------------------------------
// Scalar conversions
// ---------------------------------------------------------------------------

// The first letter of an INSDC accession is the archive of origin. Defaults to
// NCBI rather than failing: an unrecognised prefix is a new archive or a typo,
// neither of which should cost the whole record.
#[inline]
fn archive_of(accession: &str) -> Archive {
    match accession.chars().next() {
        Some('E') => Archive::Ena,
        Some('D') => Archive::Ddbj,
        _ => Archive::Ncbi,
    }
}

// Granularity is inferred from the shape the source wrote, not assumed. Six
// formats appear across the four sources and only one carries a timezone, so
// the raw string is always kept alongside.
#[inline]
fn archive_date(raw: Option<String>) -> Option<ArchiveDate> {
    raw.map(|raw| {
        let granularity = if raw.contains('.') {
            DateGranularity::Millisecond
        } else if raw.contains(':') {
            DateGranularity::Second
        } else {
            DateGranularity::Day
        };
        ArchiveDate { raw, granularity }
    })
}

#[inline]
fn zoned_date(raw: Option<String>) -> Option<ZonedDate> {
    raw.map(|raw| ZonedDate { raw })
}

#[inline]
fn publication_db(s: Option<String>) -> PublicationDb {
    match s.as_deref() {
        Some("ePubmed") => PublicationDb::EPubmed,
        Some("eDOI") => PublicationDb::EDoi,
        Some(other) => PublicationDb::Other(other.to_string()),
        None => PublicationDb::Other(String::new()),
    }
}

#[inline]
fn accessibility(s: Option<String>) -> Option<AccessibilityType> {
    s.map(|s| match s.as_str() {
        "oa" => AccessibilityType::Oa,
        "partial" => AccessibilityType::Partial,
        "paywall" => AccessibilityType::Paywall,
        _ => AccessibilityType::Unknown,
    })
}

// The vocabulary parsers below all fall through to Other(String). The variants
// cover what the 346-study corpus contains; the official INSDC lists are longer
// and grow, so an unlisted value is expected rather than exceptional.
//
// `ArchiveOther` is the archive's own literal "Other"/"OTHER"/"other" — a
// submitter saying "none of the listed categories" — which is a different fact
// from our parser not recognising a value. Case is inconsistent between fields
// in the source, which is why these match on a lowercased string.

#[inline]
fn study_type(s: Option<String>) -> Option<StudyType> {
    s.map(|s| match s.to_ascii_lowercase().as_str() {
        "transcriptome analysis" => StudyType::TranscriptomeAnalysis,
        "whole genome sequencing" => StudyType::WholeGenomeSequencing,
        "metagenomics" => StudyType::Metagenomics,
        "other" => StudyType::ArchiveOther,
        _ => StudyType::Other(s),
    })
}

#[inline]
pub(crate) fn library_source(s: Option<String>) -> Option<LibrarySource> {
    s.map(|s| match s.to_ascii_lowercase().as_str() {
        "genomic" => LibrarySource::Genomic,
        "transcriptomic" => LibrarySource::Transcriptomic,
        "metagenomic" => LibrarySource::Metagenomic,
        "metatranscriptomic" => LibrarySource::Metatranscriptomic,
        "transcriptomic single cell" => LibrarySource::TranscriptomicSingleCell,
        "viral rna" => LibrarySource::ViralRna,
        "other" => LibrarySource::ArchiveOther,
        _ => LibrarySource::Other(s),
    })
}

#[inline]
pub(crate) fn platform(s: Option<String>) -> Option<Platform> {
    s.map(|s| match s.to_ascii_uppercase().as_str() {
        "ILLUMINA" => Platform::Illumina,
        "PACBIO_SMRT" => Platform::PacbioSmrt,
        "OXFORD_NANOPORE" => Platform::OxfordNanopore,
        "ION_TORRENT" => Platform::IonTorrent,
        "BGISEQ" => Platform::Bgiseq,
        "DNBSEQ" => Platform::Dnbseq,
        "GENEMIND" => Platform::Genemind,
        "LS454" => Platform::Ls454,
        _ => Platform::Other(s),
    })
}

#[inline]
pub(crate) fn library_layout(s: Option<String>) -> Option<LibraryLayout> {
    s.map(|s| match s.to_ascii_uppercase().as_str() {
        "SINGLE" => LibraryLayout::Single,
        "PAIRED" => LibraryLayout::Paired,
        _ => LibraryLayout::Other(s),
    })
}

#[inline]
pub(crate) fn library_strategy(s: Option<String>) -> Option<LibraryStrategy> {
    s.map(|s| match s.to_ascii_lowercase().as_str() {
        "rna-seq" => LibraryStrategy::RnaSeq,
        "wgs" => LibraryStrategy::Wgs,
        "wxs" => LibraryStrategy::Wxs,
        "wga" => LibraryStrategy::Wga,
        "amplicon" => LibraryStrategy::Amplicon,
        "rad-seq" => LibraryStrategy::RadSeq,
        "targeted-capture" => LibraryStrategy::TargetedCapture,
        "chip-seq" => LibraryStrategy::ChipSeq,
        "mirna-seq" => LibraryStrategy::MirnaSeq,
        "ncrna-seq" => LibraryStrategy::NcrnaSeq,
        "bisulfite-seq" => LibraryStrategy::BisulfiteSeq,
        "atac-seq" => LibraryStrategy::AtacSeq,
        "tn-seq" => LibraryStrategy::TnSeq,
        "mbd-seq" => LibraryStrategy::MbdSeq,
        "hi-c" => LibraryStrategy::HiC,
        "clone" => LibraryStrategy::Clone,
        "dnase-hypersensitivity" => LibraryStrategy::DnaseHypersensitivity,
        "other" => LibraryStrategy::ArchiveOther,
        _ => LibraryStrategy::Other(s),
    })
}

#[inline]
pub(crate) fn library_selection(s: Option<String>) -> Option<LibrarySelection> {
    s.map(|s| match s.to_ascii_lowercase().as_str() {
        "cdna" => LibrarySelection::Cdna,
        "random" => LibrarySelection::Random,
        "pcr" => LibrarySelection::Pcr,
        "random pcr" => LibrarySelection::RandomPcr,
        "rt-pcr" => LibrarySelection::RtPcr,
        "polya" => LibrarySelection::PolyA,
        "oligo-dt" => LibrarySelection::OligoDt,
        "reduced representation" => LibrarySelection::ReducedRepresentation,
        "restriction digest" => LibrarySelection::RestrictionDigest,
        "size fractionation" => LibrarySelection::SizeFractionation,
        "repeat fractionation" => LibrarySelection::RepeatFractionation,
        "hybrid selection" => LibrarySelection::HybridSelection,
        "inverse rrna" => LibrarySelection::InverseRrna,
        "chip" => LibrarySelection::Chip,
        "dnase" => LibrarySelection::Dnase,
        "mbd2 protein methyl-cpg binding domain" => LibrarySelection::Mbd2MethylCpgBinding,
        "mda" => LibrarySelection::Mda,
        "race" => LibrarySelection::Race,
        "cage" => LibrarySelection::Cage,
        "unspecified" => LibrarySelection::Unspecified,
        "other" => LibrarySelection::ArchiveOther,
        _ => LibrarySelection::Other(s),
    })
}

// ---------------------------------------------------------------------------
// Leaf conversions
// ---------------------------------------------------------------------------

impl From<PaperDto> for Paper {
    #[inline]
    fn from(d: PaperDto) -> Self {
        Paper {
            id: d.id,
            db_type: publication_db(d.db_type),
            text: d.text,
            truncated: d.truncated,
            char_count: d.chars,
            oa_status: d.oa_status,
        }
    }
}

impl PublicationDto {
    // Needs the papers map, so this is not a From impl: text is stored once at
    // the corpus level and keyed by publication id, and the publication owns a
    // clone of the one it points at. `paper_ids` is a list on the wire to leave
    // room for a second copy of the same article (a repository version beside
    // the PMC one); today at most one resolves, so the first is taken.
    #[inline]
    fn into_publication(self, papers: &BTreeMap<String, Paper>) -> Publication {
        Publication {
            id: self.id,
            db_type: publication_db(self.db_type),
            date: zoned_date(self.date),
            status: self.status,
            reference: self.reference,
            accessibility_type: accessibility(self.accessibility_type),
            paper: self
                .paper_ids
                .iter()
                .find_map(|id| papers.get(id).cloned()),
        }
    }
}

impl From<OrganizationDto> for Organization {
    #[inline]
    fn from(d: OrganizationDto) -> Self {
        Organization {
            name: d.name,
            abbreviation: d.abbreviation,
            org_type: d.org_type,
            contact_email: d.contact_email,
            contact_first: d.contact_first,
            contact_last: d.contact_last,
        }
    }
}

impl From<SubmissionDto> for Submission {
    #[inline]
    fn from(d: SubmissionDto) -> Self {
        Submission {
            accession: d.accession,
            alias: d.alias,
            center_name: d.center_name,
            broker_name: d.broker_name,
            lab_name: d.lab_name,
            comment: d.comment,
            organization: d.organization.map(Into::into),
        }
    }
}

impl From<BioProjectDto> for BioProject {
    #[inline]
    fn from(d: BioProjectDto) -> Self {
        let archive = archive_of(d.accession.trim_start_matches("PRJ"));
        BioProject {
            accession: BioProjectAccession(d.accession),
            uid: d.uid,
            archive,
            title: d.title,
            description: d.description,
            name: d.name,
            relevance: d.relevance,
            model_organism: d.model_organism,
            target_organism: d.target_organism,
            target_taxid: d.target_taxid,
            target_sample_scope: d.target_sample_scope,
            target_material: d.target_material,
            target_capture: d.target_capture,
            method_type: d.method_type,
            data_types: d.data_types,
            objectives: d.objectives,
            submitting_organization: d.submitting_organization,
            submitted: archive_date(d.submitted),
            last_update: archive_date(d.last_update),
            external_links: d.external_links,
        }
    }
}

impl From<BioSampleDto> for BioSample {
    #[inline]
    fn from(d: BioSampleDto) -> Self {
        // SAMN / SAMEA / SAMD — the archive letter follows the SAM prefix.
        let archive = archive_of(d.accession.trim_start_matches("SAM"));
        BioSample {
            accession: BioSampleAccession(d.accession),
            archive,
            title: d.title,
            organism_name: d.organism_name,
            taxonomy_id: d.taxonomy_id,
            package: d.package,
            models: d.models,
            owner: d.owner,
            contact_first: d.contact_first,
            contact_last: d.contact_last,
            status: d.status,
            status_when: archive_date(d.status_when),
            access: d.access,
            submission_date: archive_date(d.submission_date),
            publication_date: archive_date(d.publication_date),
            last_update: archive_date(d.last_update),
            ids: d.ids,
            links: d.links,
            harmonized: d.harmonized,
        }
    }
}

impl From<SampleDto> for Sample {
    #[inline]
    fn from(d: SampleDto) -> Self {
        Sample {
            accession: SampleAccession(d.accession),
            alias: d.alias,
            title: d.title,
            taxon_id: d.taxon_id,
            scientific_name: d.scientific_name,
            biosample_id: d.biosample.map(BioSampleAccession),
            attributes: d.attributes,
            external_ids: d.external_ids,
            xrefs: d.xrefs,
            biosample_attributes: d.biosample_attributes,
            biosample: d.biosample_record.map(Into::into),
        }
    }
}

impl From<PoolMemberDto> for PoolMember {
    #[inline]
    fn from(d: PoolMemberDto) -> Self {
        PoolMember {
            sample_accession: d.accession.map(SampleAccession),
            member_name: d.member_name,
            sample_name: d.sample_name,
            sample_title: d.sample_title,
            organism: d.organism,
            tax_id: d.tax_id,
            spots: d.spots,
            bases: d.bases,
        }
    }
}

impl From<ReadStatDto> for ReadStat {
    #[inline]
    fn from(d: ReadStatDto) -> Self {
        ReadStat {
            index: d.index.unwrap_or(0),
            count: d.count,
            average: d.average,
            stdev: d.stdev,
        }
    }
}

impl From<RunStatisticsDto> for RunStatistics {
    #[inline]
    fn from(d: RunStatisticsDto) -> Self {
        RunStatistics {
            n_reads: d.n_reads,
            n_spots: d.n_spots,
            reads: d.reads.into_iter().map(Into::into).collect(),
        }
    }
}

impl From<FileAlternativeDto> for FileAlternative {
    #[inline]
    fn from(d: FileAlternativeDto) -> Self {
        FileAlternative {
            url: d.url,
            org: d.org,
            access_type: d.access_type,
            free_egress: d.free_egress,
        }
    }
}

impl From<FileRefDto> for FileRef {
    #[inline]
    fn from(d: FileRefDto) -> Self {
        FileRef {
            url: d.url,
            filename: d.filename,
            md5: d.md5,
            size: d.size,
            file_date: archive_date(d.file_date),
            semantic_name: d.semantic_name,
            supertype: d.supertype,
            // The wire carries the toolkit flag as a string ("true"/"false").
            sratoolkit: d.sratoolkit.map(|s| s.eq_ignore_ascii_case("true")),
            alternatives: d.alternatives.into_iter().map(Into::into).collect(),
        }
    }
}

impl From<CloudFileDto> for CloudFile {
    #[inline]
    fn from(d: CloudFileDto) -> Self {
        CloudFile {
            provider: d.provider,
            location: d.location,
            filetype: d.filetype,
        }
    }
}

impl From<RunDto> for Run {
    #[inline]
    fn from(d: RunDto) -> Self {
        Run {
            accession: RunAccession(d.accession),
            alias: d.alias,
            published: archive_date(d.published),
            is_public: d.is_public,
            size_bytes: d.size_bytes,
            total_spots: d.total_spots,
            total_bases: d.total_bases,
            cluster_name: d.cluster_name,
            statistics: d.statistics.map(Into::into),
            base_composition: d.base_composition,
            files: d.files.into_iter().map(Into::into).collect(),
            cloud_files: d.cloud_files.into_iter().map(Into::into).collect(),
            submitter_id: d.submitter_id,
        }
    }
}

impl From<ExperimentDto> for Experiment {
    #[inline]
    fn from(d: ExperimentDto) -> Self {
        Experiment {
            accession: ExperimentAccession(d.accession),
            alias: d.alias,
            title: d.title,
            sample_ids: d.sample_ids.into_iter().map(SampleAccession).collect(),
            design_description: d.design_description,
            library_name: d.library_name,
            library_strategy: library_strategy(d.library_strategy),
            library_source: library_source(d.library_source),
            library_selection: library_selection(d.library_selection),
            library_layout: library_layout(d.library_layout),
            library_construction_protocol: d.library_construction_protocol,
            platform: platform(d.platform),
            instrument_model: d.instrument_model,
            attributes: d.attributes,
            xrefs: d.xrefs,
            pool_members: d.pool_members.into_iter().map(Into::into).collect(),
            runs: d.runs.into_iter().map(Into::into).collect(),
        }
    }
}

// ---------------------------------------------------------------------------
// Study -> Project
// ---------------------------------------------------------------------------

impl StudyDto {
    // Takes the papers map rather than owning it: papers are shared (17 serve
    // more than one study), so each publication clones the one it points at.
    // Cloning costs ~540 KB across the whole corpus, which is why the domain
    // owns its papers instead of reference-counting them.
    #[inline]
    pub fn into_project(
        self,
        papers: &BTreeMap<String, Paper>,
        corpus_format_version: u32,
        fetched_at: &str,
    ) -> Project {
        let archive = archive_of(&self.accession);

        Project {
            accession: StudyAccession(self.accession),
            archive,
            study: Study {
                title: self.title,
                abstract_text: self.abstract_text,
                study_type: study_type(self.study_type),
                alias: self.study_alias,
                center_name: self.center_name,
                center_project_name: self.center_project_name,
                earliest_run_published: archive_date(self.published),
                external_ids: self.external_ids,
                xrefs: self.xrefs,
            },
            submission: self.submission.map(Into::into),
            bioproject: self.bioproject_record.map(Into::into),
            samples: self
                .samples
                .into_iter()
                .map(|(k, v)| (SampleAccession(k), Sample::from(v)))
                .collect(),
            experiments: self.experiments.into_iter().map(Into::into).collect(),
            publications: self
                .publications
                .into_iter()
                .map(|p| p.into_publication(papers))
                .collect(),
            source: SourceMeta {
                corpus_format_version,
                fetched_at: ZonedDate {
                    raw: fetched_at.to_string(),
                },
                record_count: self.record_count,
                build_note: self.build_note,
            },
        }
    }
}