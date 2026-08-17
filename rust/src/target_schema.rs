use chrono::{NaiveDate, NaiveDateTime};

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
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
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
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
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
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
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
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
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
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PartialDate {
    Year(i32),
    YearMonth(i32, u32),
    Date(NaiveDate),
}

#[derive(Clone, Debug, Default, PartialEq)]
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
        Self { layers: vec![Layer::Direct] }
    }
}

impl SchemaSettings {
    #[inline]
    pub fn new(layers: Vec<Layer>) -> Self {
        Self { layers }
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
            layer.process(&project, &mut schemas);
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
        for project in corpus.projects {
            schemas.extend(Self::from_project(project, settings));
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

    #[test]
    fn an_empty_corpus_yields_no_records() {
        let mut corpus = mini();
        corpus.projects.clear();
        assert!(TargetSchema::from_corpus(corpus, &SchemaSettings::default()).is_empty());
    }
}