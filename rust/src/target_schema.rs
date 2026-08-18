use std::collections::{BTreeMap, BTreeSet};

use chrono::{NaiveDate, NaiveDateTime};
use serde::{Deserialize, Serialize};

use crate::{corpus::Corpus, layer::Layer, project::*};

// The reconstruction output: one record per experiment, fully denormalised —
// study, sample, experiment, run and submission fields all flattened onto the
// same row. 65 fields, mirroring the Python schema exactly.
//
// FIELD NAMES FOLLOW `project`, NOT ENA.
// Each field is named and typed after the Project field it is drawn from, so a
// value can be traced to its source without a lookup table. The original ENA
// names are kept in comments as `// ENA: <name>`.
//
// ‼️ TWO RENAMES INVERT THEIR OLD MEANING — the ENA naming trap.
// ENA calls the BioProject `study_accession` and the SRP `secondary_study_
// accession`; likewise the BioSample is `sample_accession` and the SRS is
// `secondary_sample_accession`. Project stores them the other way round, and
// these names now follow Project. So `study_accession` here is the SRP (it was
// the PRJNA) and `sample_accession` is the SRS (it was the SAMN). Anything
// comparing against the ENA-named schema must swap those two pairs, and a
// positional mapping will silently produce wrong-but-plausible values.

// How a field came to hold what it holds.
//
// Confidence is folded in rather than kept parallel. Python carries provenance
// and confidence as two maps and then needs a runtime sweep
// (`inconsistent_confidence`) to catch the illegal combinations: a confidence
// on a `direct` field, or on a field with no value. Making the inferred
// variants the only ones that carry a Directness deletes that whole class of
// bug — `Direct` and `Harmonized` structurally cannot hold one.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Provenance {
    // The archive stated it outright.
    Direct,
    // A synonym table mapped the submitter's key onto a schema field. The
    // value is still the submitter's; only the key mapping is ours.
    Harmonized,
    InferredFromText(Directness),
    InferredFromPaper(Directness),
}

// Where an inferred value came from, not how likely it is to be right. The
// distinction matters: `Quoted` is machine-checkable by string matching against
// the evidence, which is the only reason the audit can verify anything.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Directness {
    // Appears in the evidence word for word.
    Quoted,
    // Same value, different words — a unit normalised, a synonym chosen, or one
    // of several candidate spans picked.
    Rephrased,
    // The evidence does not carry the value; it was concluded from what is there.
    Inferred,
}

// INSDC's missing-value vocabulary. Each is a *stated reason* a field has no
// ordinary value — an answer, not an absence — which is why these sit beside
// `Known` rather than collapsing into `Unknown`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum MissingReason {
    // Cannot apply here: sex on a soil metagenome, host on a free-living isolate.
    NotApplicable,
    // Nobody measured it; the value does not exist anywhere.
    NotCollected,
    // It exists but the submitter did not deposit it.
    NotProvided,
    // It exists but is behind controlled access.
    RestrictedAccess,
    // INSDC's bare `missing`: the submitter states the value is absent without
    // saying why. Unlike the four above it carries no reason, which is the
    // point — it is still a statement by the submitter, and collapsing it into
    // `Unknown` would lose that they answered at all. The commonest of the lot:
    // 20,804 harmonised values in the corpus, against 15,403 for the reasoned
    // terms combined. INSDC also defines qualified forms (`missing: control
    // sample`, `missing: lab stock`); none occur in this corpus.
    Unspecified,
}

impl MissingReason {
    // The INSDC term this stands for. The audit needs it: a stated absence
    // claiming to be quoted is claiming *this string* appears in the evidence.
    #[inline]
    pub fn as_str(self) -> &'static str {
        match self {
            MissingReason::NotApplicable => "not applicable",
            MissingReason::NotCollected => "not collected",
            MissingReason::NotProvided => "not provided",
            MissingReason::RestrictedAccess => "restricted access",
            MissingReason::Unspecified => "missing",
        }
    }
}

// One field's value together with how it got there.
//
// `Unknown` is the only variant without a Provenance, and that is the point:
// nothing has been concluded, so there is nothing to attribute. It is a
// statement about this pipeline. Every other variant is a statement about the
// archive, and carries the layer that made it.
//
// Generic over T so a date or a count can hold a missing-value reason too.
// Python cannot: its typed fields reject the sentinels outright, which it
// documents as a known casualty — a cell line arguably has no collection date
// and there is currently no way to record that.
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Field<T> {
    #[default]
    Unknown,
    Missing(MissingReason, Provenance),
    Known(T, Provenance),
}

impl<T> Field<T> {
    #[inline]
    pub fn from_option(option: Option<T>) -> Self {
        match option {
            Some(t) => {
                Self::Known(t, Provenance::Direct)
            },
            None => Self::Unknown,
        }
    }

    #[inline]
    pub fn value(&self) -> Option<&T> {
        match self {
            Field::Known(v, _) => Some(v),
            _ => None,
        }
    }

    #[inline]
    pub fn provenance(&self) -> Option<&Provenance> {
        match self {
            Field::Unknown => None,
            Field::Missing(_, p) | Field::Known(_, p) => Some(p),
        }
    }

    // A field the cascade has settled, whether or not it holds an ordinary
    // value. Missing-value terms count: they are determinations, and a later
    // layer must not overwrite them.
    #[inline]
    pub fn is_settled(&self) -> bool {
        !matches!(self, Field::Unknown)
    }
}

// A date the submitter stated at whatever precision they had.
//
// Measured across the 79,046 `collection_date` values in the corpus: 53% are a
// full date, **31% are a bare year** and 5.6% year-month. Those are not
// malformed — MIxS and INSDC explicitly allow a reduced-precision collection
// date, and "collected in 2019" is a complete answer at the precision available.
//
// A plain NaiveDate would force ~29,000 of them to be dropped or given a
// fabricated January 1st that later reads as fact. Inventing precision is the
// exact failure this project exists to avoid, so the type carries only what was
// stated.
//
// Ranges (`2019-05/2019-08`, 638 values, 0.8%) have no variant yet and parse to
// nothing; add one if something needs to query them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum PartialDate {
    Year(i32),
    YearMonth(i32, u32),
    Date(NaiveDate),
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct TargetSchema {
    // The experiment accession. Not a Field: every record has one by
    // construction, nobody inferred it, and there is nothing to attribute.
    pub id: ExperimentAccession,

    // -- study ------------------------------------------------------------
    // Project::bioproject                                  ENA: study_accession
    pub bioproject_accession: Field<BioProjectAccession>,
    // Project::accession                         ENA: secondary_study_accession
    pub study_accession: Field<StudyAccession>,
    // Study::title                                             ENA: study_title
    pub study_title: Field<String>,
    // Study::abstract_text                                      ENA: description
    pub abstract_text: Field<String>,
    // Study::alias                                             ENA: study_alias
    pub study_alias: Field<String>,
    // Study::center_project_name                              ENA: project_name
    pub center_project_name: Field<String>,
    // Study::center_name / Submission::center_name             ENA: center_name
    pub center_name: Field<String>,

    // -- submission -------------------------------------------------------
    // Submission::accession                             ENA: submission_accession
    pub submission_accession: Field<String>,
    // Submission::broker_name                                  ENA: broker_name
    pub broker_name: Field<String>,

    // -- sample -----------------------------------------------------------
    // Sample::biosample_id                                  ENA: sample_accession
    pub biosample_accession: Field<BioSampleAccession>,
    // Sample::accession                          ENA: secondary_sample_accession
    pub sample_accession: Field<SampleAccession>,
    // Sample::title                                            ENA: sample_title
    pub sample_title: Field<String>,
    // Sample::alias                                            ENA: sample_alias
    pub sample_alias: Field<String>,
    // Sample::scientific_name                              ENA: scientific_name
    pub scientific_name: Field<String>,
    // Sample::taxon_id                                            ENA: tax_id
    //
    // Integer, though `Sample::taxon_id` is a String: an NCBI taxonomy id is
    // conceptually a number, and measured across all 88,560 samples in the
    // corpus every value parses as one, with no nulls and a maximum of
    // 3,713,471. The conversion therefore parses; a value that ever fails to
    // is a real anomaly and should surface rather than pass through as text.
    pub taxon_id: Field<u64>,

    // -- biosample --------------------------------------------------------
    // BioSample::package                          ENA: ncbi_reporting_standard
    pub biosample_package: Field<String>,

    // -- experiment -------------------------------------------------------
    // Experiment::accession                          ENA: experiment_accession
    pub experiment_accession: Field<ExperimentAccession>,
    // Experiment::title                                   ENA: experiment_title
    pub experiment_title: Field<String>,
    // Experiment::alias                                   ENA: experiment_alias
    pub experiment_alias: Field<String>,
    // Experiment::library_strategy                       ENA: library_strategy
    pub library_strategy: Field<LibraryStrategy>,
    // Experiment::library_source                           ENA: library_source
    pub library_source: Field<LibrarySource>,
    // Experiment::library_selection                     ENA: library_selection
    pub library_selection: Field<LibrarySelection>,
    // Experiment::library_layout                           ENA: library_layout
    pub library_layout: Field<LibraryLayout>,
    // Experiment::library_name                               ENA: library_name
    pub library_name: Field<String>,
    // Experiment::library_construction_protocol
    //                                  ENA: library_construction_protocol
    pub library_construction_protocol: Field<String>,
    // Experiment::platform                             ENA: instrument_platform
    pub platform: Field<Platform>,
    // Experiment::instrument_model                       ENA: instrument_model
    pub instrument_model: Field<String>,

    // -- run --------------------------------------------------------------
    // Run::total_spots, summed across the experiment's runs     ENA: read_count
    pub total_spots: Field<u64>,
    // Run::total_bases, summed across the experiment's runs     ENA: base_count
    pub total_bases: Field<u64>,
    // Run::published, earliest across the experiment's runs;
    // falls back to Study::earliest_run_published             ENA: first_public
    pub earliest_run_published: Field<NaiveDateTime>,

    // -- no Project counterpart -------------------------------------------
    // Everything below comes from an attribute bag (layer 2/3), from the paper
    // (layer 4), or from nowhere in this corpus at all. Names are unchanged
    // because there is no Project field to follow.
    pub age: Field<String>,
    pub broad_scale_environmental_context: Field<String>,
    pub cell_line: Field<String>,
    pub cell_type: Field<String>,
    // Present as the `ENA-CHECKLIST` key in the attribute bag (9,996 entries),
    // not as a structural element — so it arrives via harmonisation, not direct.
    pub checklist: Field<String>,
    pub collected_by: Field<String>,
    // PartialDate, not NaiveDate: this is submitter free text and MIxS allows
    // reduced precision. See the type for the measured breakdown.
    pub collection_date: Field<PartialDate>,
    pub country: Field<String>,
    // No source in this corpus: an ENA filereport column with no SRA equivalent.
    pub datahub: Field<String>,
    pub dev_stage: Field<String>,
    pub environment_biome: Field<String>,
    pub environment_feature: Field<String>,
    pub environment_material: Field<String>,
    pub environmental_medium: Field<String>,
    pub first_created: Field<NaiveDateTime>,
    pub host: Field<String>,
    pub host_scientific_name: Field<String>,
    pub host_sex: Field<String>,
    pub host_tax_id: Field<u64>,
    pub isolation_source: Field<String>,
    pub last_updated: Field<NaiveDateTime>,
    pub local_environmental_context: Field<String>,
    pub sample_capture_status: Field<String>,
    pub sample_description: Field<String>,
    pub sequencing_method: Field<String>,
    pub sex: Field<String>,
    pub strain: Field<String>,
    // No source in this corpus.
    pub submitted_format: Field<String>,
    // No source in this corpus.
    pub submitted_read_type: Field<String>,
    // No source in this corpus; appears to be a scratch field.
    pub tag: Field<String>,
    pub tissue_type: Field<String>,
    pub treatment: Field<String>,
}

// How to build the records: which layers run, and in what order.
//
// A list rather than a set of `do_this_layer: bool` flags, so the sequence is
// data a caller can rearrange instead of control flow they would have to edit.
// The same layer may appear twice, and the layers need not be in cascade order
// — neither is useful today, but forbidding them would put this type back in
// the business of deciding the pipeline instead of describing it.
pub struct SchemaSettings {
    // Order matters. Each layer processes the data in the order of the vec.
    layers: Vec<Layer>,
    // The evidence each record was shown, kept only when asked for.
    //
    // Off by default because it is large — a corpus run would store the
    // attribute bag of every sample twice over — and useful only when something
    // is going to read it back. The verbatim audit is that something: a
    // `quoted` claim says a value appears in the evidence word for word, and
    // without the evidence the claim cannot be checked at all.
    evidence: Option<std::sync::Arc<std::sync::Mutex<BTreeMap<String, String>>>>,
    // Volume ceilings, independent of the spend ceiling.
    //
    // A budget stops a run once it has cost too much, which needs the cost to
    // be known — and the cost of a study is only known once it has been paid.
    // These stop it before that: `SRP049009` plans 2,150 calls, and capping the
    // records means the paid layers never see more than a handful of them
    // whatever the ledger says. Two limits that fail independently.
    max_studies: Option<usize>,
    max_total_records: Option<usize>,
    // Named studies rather than a prefix. `None` means every study; an empty
    // set means none, which is a real configuration a caller can ask for and
    // not the same thing as "unset".
    only_studies: Option<BTreeSet<String>>,
    // Where a layer reports something it could not do. `None` discards them,
    // which is what the first version did unconditionally — and a paid call
    // that failed then looked exactly like a layer that had nothing to say.
    on_issue: Option<IssueSink>,
}

// Where issues go. Boxed because a run decides at construction what to do
// with them, and Send + Sync so a run can still fan out over records.
pub type IssueSink = Box<dyn Fn(&Issue) + Send + Sync>;

// What separates one layer's evidence from another's inside a record's entry.
//
// A line rather than a nested map, so a run saved before layer 3 and layer 4
// could differ still loads: the export's `evidence` field keeps its type, and a
// file with no headers at all reads as one undifferentiated block, which is
// exactly what it was.
pub const EVIDENCE_HEADER: &str = "=== evidence: ";

// Something a layer could not do, reported rather than swallowed.
//
// Not an error return: one sample failing must not abandon the other three
// hundred, so the run continues and says so. The distinction that matters is
// between "the model declined this field" (normal, silent) and "the call did
// not happen" (worth knowing about).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Issue {
    pub layer: &'static str,
    pub context: String,
    pub error: String,
}

impl std::fmt::Display for Issue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}] {}: {}", self.layer, self.context, self.error)
    }
}

// Direct only, which is the one configuration that is free, offline and
// deterministic.
//
// Deliberately not `#[derive(Default)]`: that would produce an empty layer list
// and so a silent no-op, which looks identical to a pipeline that ran and found
// nothing. Erring toward the free layer also means a caller who forgets to
// configure anything cannot be billed for it.
impl Default for SchemaSettings {
    #[inline]
    fn default() -> Self {
        Self {
            layers: vec![Layer::Direct],
            evidence: None,
            max_studies: None,
            max_total_records: None,
            only_studies: None,
            on_issue: None,
        }
    }
}

impl SchemaSettings {
    #[inline]
    pub fn new(layers: Vec<Layer>) -> Self {
        Self {
            layers,
            evidence: None,
            max_studies: None,
            max_total_records: None,
            only_studies: None,
            on_issue: None,
        }
    }









    // Keep the evidence shown to each record, so a saved run can be audited
    // without regenerating it from the corpus.
    pub fn keep_evidence(mut self, keep: bool) -> Self {
        self.evidence = keep.then(|| std::sync::Arc::new(std::sync::Mutex::new(BTreeMap::new())));
        self
    }

    // Appends what one record was shown, tagged with the layer that showed it.
    //
    // A record can be covered by more than one call — the study-level one and
    // its own — and the audit wants the union of what that layer saw.
    //
    // Tagged because the layers do not see the same thing. Layer 4 appends up
    // to 30,000 characters of publication to the same record, and an audit that
    // could not tell the blocks apart would check layer 3's `quoted` claims
    // against a paper its call was never shown — turning a falsifiable claim
    // back into an unfalsifiable one, quietly and in the flattering direction.
    pub fn record_evidence(&self, id: &str, layer: &str, evidence: &str) {
        let Some(store) = &self.evidence else { return };
        let mut store = store.lock().expect("evidence store poisoned");
        let entry = store.entry(id.to_string()).or_default();
        if !entry.is_empty() {
            entry.push('\n');
        }
        entry.push_str(EVIDENCE_HEADER);
        entry.push_str(layer);
        entry.push_str(" ===\n");
        entry.push_str(evidence);
    }

    pub fn evidence(&self) -> Option<BTreeMap<String, String>> {
        self.evidence.as_ref().map(|s| s.lock().expect("evidence store poisoned").clone())
    }


    // At most this many studies, taken from the front of the corpus.
    pub fn max_studies(mut self, studies: usize) -> Self {
        self.max_studies = Some(studies);
        self
    }

    // Run only these studies, named by accession.
    //
    // The caps take a *prefix* of the corpus, which is the right shape for "a
    // small cheap run" and the wrong shape for "the same studies as last time".
    // A controlled comparison needs the second: re-running the five studies a
    // Python run covered is the only way to tell an improved layer from an easy
    // study, and coverage varies enough between studies (sd 3.8 fields/record on
    // the model layers) to hide any effect smaller than that.
    //
    // Matches either the study accession (SRP/ERP/DRP) or the BioProject one
    // (PRJNA/PRJEB/PRJDB), because which of the two names a study is an accident
    // of where the list came from. Case-insensitive; whitespace trimmed.
    //
    // Selection happens before the caps, so the two compose: pick five studies
    // and cap the records, and the cap applies to those five.
    pub fn only_studies<S: AsRef<str>>(mut self, accessions: impl IntoIterator<Item = S>) -> Self {
        self.only_studies = Some(
            accessions
                .into_iter()
                .map(|a| a.as_ref().trim().to_ascii_uppercase())
                .collect(),
        );
        self
    }

    // Whether this project is one the run was asked for. `true` for every
    // project when no selection was made.
    //
    // One predicate, called by both the run and the estimate — the estimate is
    // only a guard if it prices the studies that are actually going to run.
    pub fn selects(&self, project: &Project) -> bool {
        let Some(wanted) = &self.only_studies else {
            return true;
        };
        let names = [
            Some(project.accession.0.to_ascii_uppercase()),
            project
                .bioproject
                .as_ref()
                .map(|b| b.accession.0.to_ascii_uppercase()),
        ];
        names.iter().flatten().any(|name| wanted.contains(name))
    }

    // Accessions that were asked for and are not in the corpus.
    //
    // Reported rather than ignored: a mistyped accession otherwise produces a
    // smaller run that looks exactly like a correct one, and the comparison it
    // was for is quietly against a different study set.
    pub fn unmatched<'a>(&self, projects: impl IntoIterator<Item = &'a Project>) -> Vec<String> {
        let Some(wanted) = &self.only_studies else {
            return Vec::new();
        };
        let mut missing = wanted.clone();
        for project in projects {
            missing.remove(&project.accession.0.to_ascii_uppercase());
            if let Some(bioproject) = &project.bioproject {
                missing.remove(&bioproject.accession.0.to_ascii_uppercase());
            }
        }
        missing.into_iter().collect()
    }

    // At most this many records in total, across every study.
    //
    // Applied by trimming each study's experiments *before* its layers run, so
    // it bounds what the paid layers are asked to do rather than only what is
    // returned. A study is therefore delivered as a prefix of itself.
    pub fn max_total_records(mut self, records: usize) -> Self {
        self.max_total_records = Some(records);
        self
    }


    #[inline]
    pub fn study_limit(&self) -> Option<usize> {
        self.max_studies
    }

    #[inline]
    pub fn record_limit(&self) -> Option<usize> {
        self.max_total_records
    }

    // Hand every issue to `sink`. Without this they are discarded, so a run
    // that wants to know a call failed has to ask.
    pub fn on_issue(mut self, sink: impl Fn(&Issue) + Send + Sync + 'static) -> Self {
        self.on_issue = Some(Box::new(sink));
        self
    }

    #[inline]
    pub fn report(&self, issue: Issue) {
        if let Some(sink) = &self.on_issue {
            sink(&issue);
        }
    }

    // Appends one layer, for building a sequence up from `default()`.
    #[inline]
    pub fn with(mut self, layer: Layer) -> Self {
        self.layers.push(layer);
        self
    }

    #[inline]
    pub fn layers(&self) -> &[Layer] {
        &self.layers
    }

    // Whether this configuration can bill an API. The check a caller should
    // make before running a corpus rather than after.
    #[inline]
    pub fn is_paid(&self) -> bool {
        self.layers.iter().any(Layer::is_paid)
    }
}

impl TargetSchema {
    // One record per experiment, or per (experiment, sample) pair when the
    // experiment is pooled, built by running each configured layer in order.
    //
    // The layers are what fill the record; this function only sequences them.
    // Which ones run, and in what order, is entirely `settings` — see
    // `SchemaSettings`, whose default is the free direct layer alone.
    #[inline]
    pub fn from_project(project: Project, settings: &SchemaSettings) -> Vec<Self> {
        let mut schemas = Vec::new();
        for layer in settings.layers() {
            layer.process(&project, &mut schemas, settings);
        }

        schemas
    }

    // Every project in a corpus, flattened into one record list.
    //
    // `counts.records` is the builder's own tally and is only a capacity hint —
    // a stale or wrong count costs a reallocation, never a dropped record.
    #[inline]
    pub fn from_corpus(corpus: Corpus, settings: &SchemaSettings) -> Vec<TargetSchema> {
        let mut schemas = Vec::with_capacity(corpus.counts.records);
        let studies = settings.study_limit().unwrap_or(usize::MAX);

        // Said once, before anything runs. A mistyped accession otherwise makes
        // a run that looks correct and covers a different study set than the
        // one it is going to be compared against.
        for accession in settings.unmatched(&corpus.projects) {
            settings.report(Issue {
                layer: "corpus",
                context: accession,
                error: "study was selected but is not in this corpus".into(),
            });
        }

        // Selection first, then the caps, so `only_studies` and `max_studies`
        // compose rather than fight: five named studies capped to thirty records
        // is thirty records of those five.
        for mut project in corpus
            .projects
            .into_iter()
            .filter(|p| settings.selects(p))
            .take(studies)
        {
            if let Some(limit) = settings.record_limit() {
                let room = limit.saturating_sub(schemas.len());
                if room == 0 {
                    break;
                }
                // Trimmed before the layers run, so a paid layer is never asked
                // about a record that is going to be discarded. One experiment
                // yields one record except on a pooled study, which is why the
                // result is trimmed again below.
                if project.experiments.len() > room {
                    project.experiments.truncate(room);
                }
            }

            let mut built = Self::from_project(project, settings);
            if let Some(limit) = settings.record_limit() {
                built.truncate(limit.saturating_sub(schemas.len()));
            }
            schemas.extend(built);
        }
        schemas
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::corpus::Corpus;

    // The same real slice the corpus tests use. This module tests sequencing —
    // which layers run and how their output is combined — so it wants a whole
    // corpus rather than the hand-built projects in `layer::direct`.
    const MINI: &str = include_str!("../test_data/mini_corpus.json");

    fn mini() -> Corpus {
        Corpus::from_json(MINI, false).expect("fixture should parse")
    }

    // -- Field -------------------------------------------------------------

    #[test]
    fn from_option_stamps_direct_and_maps_none_to_unknown() {
        assert_eq!(
            Field::from_option(Some("value".to_string())),
            Field::Known("value".to_string(), Provenance::Direct)
        );
        assert_eq!(Field::from_option(None::<String>), Field::<String>::Unknown);
    }

    #[test]
    fn from_option_does_not_read_missing_value_terms() {
        // Worth pinning down because it is a real limit rather than an
        // oversight: INSDC's vocabulary appears in the submitter attribute
        // bags, which layer 2 reads, and in no direct field measured across the
        // corpus. If that ever changes this assertion is where to start.
        let field = Field::from_option(Some("not applicable".to_string()));
        assert_eq!(field, Field::Known("not applicable".to_string(), Provenance::Direct));
        assert!(!matches!(field, Field::Missing(..)));
    }

    #[test]
    fn unknown_is_the_only_variant_without_a_provenance() {
        // The structural point of folding confidence into provenance: there is
        // no way to express an attributed nothing, or an unattributed value.
        assert_eq!(Field::<u64>::Unknown.provenance(), None);
        assert_eq!(
            Field::Known(1u64, Provenance::Harmonized).provenance(),
            Some(&Provenance::Harmonized)
        );
        assert_eq!(
            Field::<u64>::Missing(MissingReason::NotApplicable, Provenance::Direct).provenance(),
            Some(&Provenance::Direct)
        );
    }

    #[test]
    fn a_missing_value_term_counts_as_settled() {
        // A stated reason is a determination, not an absence, so a later layer
        // must not treat it as an open field and overwrite it.
        let missing = Field::<PartialDate>::Missing(
            MissingReason::NotCollected,
            Provenance::InferredFromText(Directness::Quoted),
        );
        assert!(missing.is_settled());
        assert!(missing.value().is_none());
        assert!(!Field::<PartialDate>::Unknown.is_settled());
        assert!(Field::Known(42u64, Provenance::Direct).is_settled());
    }

    #[test]
    fn a_partial_date_keeps_the_precision_it_was_given() {
        // 31% of corpus collection_dates are a bare year and 5.6% year-month.
        // The variants must stay distinguishable: promoting a year to a January
        // 1st would invent precision the submitter never stated.
        let year = Field::Known(PartialDate::Year(2019), Provenance::Harmonized);
        let full = Field::Known(
            PartialDate::Date(NaiveDate::from_ymd_opt(2019, 1, 1).unwrap()),
            Provenance::Harmonized,
        );
        assert_ne!(year, full);
        assert_ne!(PartialDate::Year(2019), PartialDate::YearMonth(2019, 1));
    }

    // -- SchemaSettings ----------------------------------------------------

    #[test]
    fn the_default_configuration_is_the_free_one() {
        // Deliberately not derived: a derived Default would be an empty list,
        // and so a silent no-op. Erring toward the direct layer also means a
        // caller who configures nothing cannot be billed.
        let settings = SchemaSettings::default();
        assert_eq!(settings.layers().len(), 1);
        assert!(matches!(settings.layers()[0], Layer::Direct));
        assert!(!settings.is_paid());
    }

    #[test]
    fn is_paid_tracks_the_llm_layers() {
        assert!(!SchemaSettings::new(vec![Layer::Direct, Layer::Harmonized]).is_paid());
        assert!(!Layer::Direct.is_paid());
        assert!(!Layer::Harmonized.is_paid());
    }

    #[test]
    fn with_appends_in_order() {
        let settings = SchemaSettings::default().with(Layer::Harmonized);
        assert_eq!(settings.layers().len(), 2);
        assert!(matches!(settings.layers()[0], Layer::Direct));
        assert!(matches!(settings.layers()[1], Layer::Harmonized));
    }

    #[test]
    fn a_duplicate_layer_is_allowed() {
        // The list describes the pipeline rather than policing it. Reordering
        // and repetition are the caller's to make - and to regret.
        let settings = SchemaSettings::new(vec![Layer::Direct, Layer::Direct]);
        assert_eq!(settings.layers().len(), 2);
    }

    // -- from_project ------------------------------------------------------

    #[test]
    fn from_project_builds_one_record_per_experiment() {
        let project = mini().projects.into_iter()
            .find(|p| p.accession.0 == "DRP006604").unwrap();
        let records = TargetSchema::from_project(project, &SchemaSettings::default());

        assert_eq!(records.len(), 2);
        assert!(records.iter().all(|r| r.study_accession
            == Field::Known(StudyAccession("DRP006604".into()), Provenance::Direct)));
        assert!(records.iter().all(|r| r.experiment_accession.is_settled()));
    }

    #[test]
    fn an_empty_layer_list_produces_nothing() {
        // The failure the manual Default exists to avoid: indistinguishable
        // from a pipeline that ran and found nothing.
        let project = mini().projects.into_iter().next().unwrap();
        let records = TargetSchema::from_project(project, &SchemaSettings::new(vec![]));
        assert!(records.is_empty());
    }

    #[test]
    fn running_direct_twice_duplicates_every_record() {
        let project = mini().projects.into_iter()
            .find(|p| p.accession.0 == "DRP006604").unwrap();
        let once = TargetSchema::from_project(project.clone(), &SchemaSettings::default());
        let twice = TargetSchema::from_project(
            project, &SchemaSettings::new(vec![Layer::Direct, Layer::Direct]));
        assert_eq!(twice.len(), once.len() * 2);
    }

    // -- from_corpus -------------------------------------------------------

    #[test]
    fn from_corpus_flattens_every_project() {
        let corpus = mini();
        let expected: usize = corpus.projects.iter().map(|p| p.experiments.len()).sum();
        let records = TargetSchema::from_corpus(corpus, &SchemaSettings::default());

        assert_eq!(records.len(), expected);
        let studies: Vec<_> = records.iter()
            .filter_map(|r| r.study_accession.value().map(|a| a.0.as_str()))
            .collect();
        assert!(studies.contains(&"DRP006604"));
        assert!(studies.contains(&"DRP003937"));
        assert!(studies.contains(&"SRP999999"));
    }

    #[test]
    fn from_corpus_ids_are_unique() {
        // The invariant the pooled-experiment suffix exists to preserve. Across
        // the full 630 MB corpus this holds for all 102,240 records.
        let records = TargetSchema::from_corpus(mini(), &SchemaSettings::default());
        let unique: std::collections::BTreeSet<_> =
            records.iter().map(|r| r.id.0.as_str()).collect();
        assert_eq!(unique.len(), records.len());
    }

    #[test]
    fn from_corpus_matches_from_project_run_per_project() {
        // from_corpus is only a fold over from_project; this pins that down so
        // a future short-cut in one cannot silently diverge from the other.
        let corpus = mini();
        let settings = SchemaSettings::default();
        let per_project: Vec<_> = corpus.projects.iter()
            .flat_map(|p| TargetSchema::from_project(p.clone(), &settings))
            .collect();
        let whole = TargetSchema::from_corpus(corpus, &settings);
        assert_eq!(whole, per_project);
    }

    // -- volume caps ---------------------------------------------------------

    #[test]
    fn max_studies_takes_a_prefix_of_the_corpus() {
        let corpus = mini();
        let all = TargetSchema::from_corpus(corpus.clone(), &SchemaSettings::default());
        let one = TargetSchema::from_corpus(
            corpus,
            &SchemaSettings::default().max_studies(1),
        );
        assert!(one.len() < all.len());
        assert!(one.iter().all(|r| r.study_accession == all[0].study_accession));
    }

    #[test]
    fn max_total_records_is_exact_across_studies() {
        let corpus = mini();
        for limit in 0..=5 {
            let records = TargetSchema::from_corpus(
                corpus.clone(),
                &SchemaSettings::default().max_total_records(limit),
            );
            assert!(records.len() <= limit, "asked for {limit}, got {}", records.len());
        }
    }

    #[test]
    fn the_two_caps_compose() {
        let records = TargetSchema::from_corpus(
            mini(),
            &SchemaSettings::default().max_studies(2).max_total_records(1),
        );
        assert_eq!(records.len(), 1);
    }

    // -- selecting studies by name -----------------------------------------

    fn accessions(records: &[TargetSchema]) -> std::collections::BTreeSet<String> {
        records
            .iter()
            .filter_map(|r| r.study_accession.value().map(|a| a.0.clone()))
            .collect()
    }

    #[test]
    fn naming_a_study_runs_that_study_and_no_other() {
        let records =
            TargetSchema::from_corpus(mini(), &SchemaSettings::default().only_studies(["DRP003937"]));
        assert_eq!(accessions(&records), ["DRP003937".to_string()].into());
    }

    #[test]
    fn a_study_can_be_named_by_its_bioproject_instead() {
        // Which of the two names a study is an accident of where the list came
        // from: the Python runs to compare against recorded BioProject
        // accessions, and this corpus is keyed by the SRP.
        let records = TargetSchema::from_corpus(
            mini(),
            &SchemaSettings::default().only_studies(["PRJDB7399"]),
        );
        assert_eq!(accessions(&records), ["DRP006604".to_string()].into());
    }

    #[test]
    fn one_bioproject_covering_two_studies_selects_both() {
        // Real in this fixture and real in the corpus: the mapping is not 1:1,
        // so selecting by BioProject can widen the run. Better to run both than
        // to silently pick one.
        let records = TargetSchema::from_corpus(
            mini(),
            &SchemaSettings::default().only_studies(["PRJDB4784"]),
        );
        assert_eq!(
            accessions(&records),
            ["DRP003937".to_string(), "SRP999999".to_string()].into()
        );
    }

    #[test]
    fn accessions_are_matched_case_insensitively_and_trimmed() {
        // Pasted from a paper, a spreadsheet or a shell, an accession arrives
        // with whatever whitespace and case it had.
        let records = TargetSchema::from_corpus(
            mini(),
            &SchemaSettings::default().only_studies(["  drp003937  "]),
        );
        assert_eq!(accessions(&records), ["DRP003937".to_string()].into());
    }

    #[test]
    fn no_selection_still_runs_everything() {
        let all = TargetSchema::from_corpus(mini(), &SchemaSettings::default()).len();
        assert!(all > 1);
    }

    #[test]
    fn an_empty_selection_runs_nothing_rather_than_everything() {
        // `None` and "an empty list" are different requests, and reading the
        // second as the first would run the whole corpus on a paid layer.
        let none: [&str; 0] = [];
        let records = TargetSchema::from_corpus(mini(), &SchemaSettings::default().only_studies(none));
        assert!(records.is_empty());
    }

    #[test]
    fn a_selected_study_that_is_not_in_the_corpus_is_reported() {
        // The failure this exists to prevent: a mistyped accession makes a run
        // that looks correct and covers a different study set than the one it
        // is about to be compared against.
        use std::sync::{Arc, Mutex};
        let seen = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&seen);
        let settings = SchemaSettings::default()
            .only_studies(["DRP003937", "SRP000000", "PRJNA999999"])
            .on_issue(move |i| sink.lock().unwrap().push(i.to_string()));

        let records = TargetSchema::from_corpus(mini(), &settings);
        assert_eq!(accessions(&records), ["DRP003937".to_string()].into());

        let seen = seen.lock().unwrap();
        assert_eq!(seen.len(), 2, "{seen:?}");
        assert!(seen.iter().any(|i| i.contains("SRP000000")));
        assert!(seen.iter().any(|i| i.contains("PRJNA999999")));
        assert!(!seen.iter().any(|i| i.contains("DRP003937")), "a match was reported missing");
    }

    #[test]
    fn selection_and_the_caps_compose() {
        // Selection first, then the caps: "these studies, capped" rather than
        // "the first N of the corpus, then filtered" — which would silently
        // return nothing whenever a selected study sits past the cap.
        let late = mini().projects.last().unwrap().accession.0.clone();
        let records = TargetSchema::from_corpus(
            mini(),
            &SchemaSettings::default().only_studies([&late]).max_studies(1),
        );
        assert_eq!(accessions(&records), [late.clone()].into(), "the cap ate the selected study");

        let capped = TargetSchema::from_corpus(
            mini(),
            &SchemaSettings::default()
                .only_studies(["DRP006604"])
                .max_total_records(1),
        );
        assert_eq!(capped.len(), 1);
    }

    #[test]
    fn no_cap_means_everything() {
        let corpus = mini();
        let expected: usize = corpus.projects.iter().map(|p| p.experiments.len()).sum();
        assert_eq!(
            TargetSchema::from_corpus(corpus, &SchemaSettings::default()).len(),
            expected
        );
    }

    #[test]
    fn the_record_cap_bounds_what_a_paid_layer_is_asked_to_do() {
        // The safeguard, not the slice. Trimming only the returned list would
        // let a 2,150-call study run in full and then throw most of it away —
        // billed. So this counts the calls the model actually receives.
        use crate::layer::Layer;
        use crate::model::{Model, ModelError, Request, Response, Usage};
        use std::sync::{Arc, Mutex};

        struct Counting(Arc<Mutex<u32>>);
        impl Model for Counting {
            fn price(&self, _u: Usage) -> f64 {
                0.0
            }
            fn complete(&self, _r: &Request) -> Result<Response, ModelError> {
                *self.0.lock().unwrap() += 1;
                let reply = serde_json::json!({"answers": []});
                Ok(Response { text: reply.to_string(), json: Some(reply), stop_reason: None, usage: Usage::default() })
            }
        }

        let calls_for = |limit: Option<usize>| {
            let calls = Arc::new(Mutex::new(0));
            let mut settings = SchemaSettings::new(vec![
                Layer::Direct,
                Layer::Harmonized,
                Layer::LLMNaive {
                model: Box::new(Counting(Arc::clone(&calls))),
                config: crate::layer::ModelConfig::new("test", ""),
            },
            ]);
            if let Some(limit) = limit {
                settings = settings.max_total_records(limit);
            }
            let records = TargetSchema::from_corpus(mini(), &settings);
            let calls = *calls.lock().unwrap();
            (records.len(), calls)
        };

        let (capped_records, capped_calls) = calls_for(Some(1));
        let (all_records, all_calls) = calls_for(None);

        assert_eq!(capped_records, 1);
        assert!(all_records > 1);
        assert!(
            capped_calls < all_calls,
            "capped run made {capped_calls} calls, uncapped made {all_calls} — \
             the cap is not reaching the paid layer"
        );
    }

    #[test]
    fn a_record_cap_of_zero_makes_no_calls_at_all() {
        use crate::layer::Layer;
        use crate::model::{Model, ModelError, Request, Response, Usage};
        use std::sync::{Arc, Mutex};

        struct Counting(Arc<Mutex<u32>>);
        impl Model for Counting {
            fn price(&self, _u: Usage) -> f64 {
                0.0
            }
            fn complete(&self, _r: &Request) -> Result<Response, ModelError> {
                *self.0.lock().unwrap() += 1;
                panic!("a capped-to-zero run must not reach the model");
            }
        }
        let calls = Arc::new(Mutex::new(0));
        let settings = SchemaSettings::new(vec![
            Layer::Direct,
            Layer::LLMNaive {
                model: Box::new(Counting(Arc::clone(&calls))),
                config: crate::layer::ModelConfig::new("test", ""),
            },
        ])
        .max_total_records(0);

        assert!(TargetSchema::from_corpus(mini(), &settings).is_empty());
        assert_eq!(*calls.lock().unwrap(), 0);
    }

    #[test]
    fn an_empty_corpus_yields_no_records() {
        let mut corpus = mini();
        corpus.projects.clear();
        assert!(TargetSchema::from_corpus(corpus, &SchemaSettings::default()).is_empty());
    }
}