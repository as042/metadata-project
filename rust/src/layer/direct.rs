use chrono::NaiveDateTime;

use crate::project::*;
use crate::target_schema::{Field, Provenance, TargetSchema};

// Layer 1. Everything here is something an archive stated outright, which is
// what makes these the anchors no later layer may overwrite.
//
// The only layer that *creates* records; every other one fills fields on
// records this produced. That asymmetry is why `process` appends to `schemas`
// rather than mapping over it, and why a settings list that runs another layer
// before this one has nothing to work on.
//
// It is also the only layer that is free: no network, no model, no spend.

// Run@published is the one date field with a single format: all 117,915 runs in
// the corpus are `NNNN-NN-NN NN:NN:NN`, length 19, no exceptions. Parsing it is
// therefore exact, and a failure is a real anomaly rather than a format the
// archive also uses. The shortcut does not generalise — the other date fields
// across the four sources span six formats between them.
const RUN_PUBLISHED_FORMAT: &str = "%Y-%m-%d %H:%M:%S";

#[inline]
fn parse_run_published(date: &ArchiveDate) -> Option<NaiveDateTime> {
    NaiveDateTime::parse_from_str(&date.raw, RUN_PUBLISHED_FORMAT).ok()
}

// One record per experiment, or per (experiment, sample) pair when the
// experiment is pooled.
//
// Appends rather than replacing, so listing Direct twice duplicates every
// record instead of quietly discarding the first pass. Neither is useful, but
// a visible duplicate is easier to diagnose than a silent overwrite.
//
// Not implemented, unlike Python: the sample-level and study-level fallback
// records for a Project built without experiments. Every one of the 346
// studies in the corpus has experiments, and a study-level stub would have to
// put an SRP accession in `id`, whose type is ExperimentAccession. That is a
// type question to settle before writing the fallback, not after.
#[inline]
pub(crate) fn process(project: &Project, schemas: &mut Vec<TargetSchema>) {
    schemas.reserve(project.experiments.len());

    for experiment in &project.experiments {
        // Several SRS under one SRX — a multiplexed library, the reason
        // `Experiment::sample_ids` is a Vec. Zero of the 102,240 experiments in
        // the corpus are pooled, but the relation is many-to-many in SRA and
        // collapsing it would drop every sample in a pool but one.
        let samples: Vec<Option<&Sample>> = if experiment.sample_ids.is_empty() {
            vec![None]
        } else {
            experiment.sample_ids.iter().map(|id| project.samples.get(id)).collect()
        };
        let pooled = samples.len() > 1;

        for sample in samples {
            // `id` must stay unique, so a pooled experiment's records are keyed
            // `<experiment>.<sample>`. An unresolvable sample cannot
            // disambiguate anything, so it keeps the bare accession — as does
            // the overwhelming majority, which are not pooled and so stay
            // directly comparable against ENA's own ids.
            let id = match sample {
                Some(sample) if pooled => ExperimentAccession(format!(
                    "{}.{}",
                    experiment.accession.0, sample.accession.0
                )),
                _ => experiment.accession.clone(),
            };

            // The SRP lives on Project, not Study, so it has no helper.
            let mut schema = TargetSchema {
                id,
                study_accession: Field::Known(project.accession.clone(), Provenance::Direct),
                ..Default::default()
            };

            // Experiment first: it owns the runs, and the study's release date
            // is only a fallback for what the runs could not supply. The guards
            // inside the helpers make this order a preference rather than a
            // requirement.
            schema.add_direct_from_experiment(experiment.clone());
            schema.add_direct_from_study(&project.study);
            if let Some(submission) = &project.submission {
                schema.add_direct_from_submission(submission);
            }
            if let Some(bioproject) = &project.bioproject {
                schema.add_direct_from_bioproject(bioproject);
            }
            if let Some(sample) = sample {
                schema.add_direct_from_sample(sample);
            }

            schemas.push(schema);
        }
    }
}

// An inherent impl in the layer module rather than in target_schema, so the
// direct layer's logic lives in one file while the call sites keep reading as
// methods on the record being built.
impl TargetSchema {
    // Adds fields to `self` using only direct with only experiment/run metadata.
    #[inline]
    pub fn add_direct_from_experiment(&mut self, experiment: Experiment) {
        self.experiment_accession = Field::Known(experiment.accession, Provenance::Direct);
        self.experiment_title = Field::from_option(experiment.title);
        self.experiment_alias = Field::from_option(experiment.alias);
        self.library_strategy = Field::from_option(experiment.library_strategy);
        self.library_source = Field::from_option(experiment.library_source);
        self.library_selection = Field::from_option(experiment.library_selection);
        self.library_layout = Field::from_option(experiment.library_layout);
        self.library_name = Field::from_option(experiment.library_name);
        self.library_construction_protocol =
            Field::from_option(experiment.library_construction_protocol);
        self.platform = Field::from_option(experiment.platform);
        self.instrument_model = Field::from_option(experiment.instrument_model);
        self.add_direct_from_runs(experiment.runs);
    }

    // Adds fields to `self` using only direct with only run metadata.
    //
    // All-or-nothing per field, deliberately: an experiment whose runs only
    // partly report a count would otherwise get the sum of the reporting ones
    // presented as the experiment total — a silent undercount that looks
    // authoritative. Python sums whatever is present; this does not. Measured
    // on the corpus, no experiment has partial counts, so the two agree today.
    #[inline]
    pub fn add_direct_from_runs(&mut self, runs: Vec<Run>) {
        // `all()` is vacuously true on an empty slice, which would make a
        // runless experiment report a total of zero rather than nothing known.
        if runs.is_empty() {
            return;
        }

        if runs.iter().all(|x| x.total_spots.is_some()) {
            let total = runs.iter().map(|x| x.total_spots.unwrap()).sum();
            self.total_spots = Field::Known(total, Provenance::Direct);
        } else {
            self.total_spots = Field::Unknown;
        }

        if runs.iter().all(|x| x.total_bases.is_some()) {
            let total = runs.iter().map(|x| x.total_bases.unwrap()).sum();
            self.total_bases = Field::Known(total, Provenance::Direct);
        } else {
            self.total_bases = Field::Unknown;
        }

        // Earliest, not first-listed: runs of one experiment can be released on
        // different days (281 experiments in the corpus), and SRA returns them
        // newest-first, so taking `runs[0]` would report the latest date.
        let earliest = runs
            .iter()
            .map(|x| x.published.as_ref().and_then(parse_run_published))
            .collect::<Option<Vec<_>>>()
            .and_then(|dates| dates.into_iter().min());
        self.earliest_run_published = match earliest {
            Some(date) => Field::Known(date, Provenance::Direct),
            None => Field::Unknown,
        };
    }

    // Adds fields to `self` using only direct with only study metadata.
    //
    // Borrowed, unlike `add_direct_from_experiment`. The split is deliberate:
    // an experiment maps 1:1 onto a record and can be consumed, while a study,
    // submission, BioProject and sample are each read once per experiment in
    // the study — taking them owned would clone an attribute bag 102,240 times.
    #[inline]
    pub fn add_direct_from_study(&mut self, study: &Study) {
        self.study_title = Field::from_option(study.title.clone());
        self.abstract_text = Field::from_option(study.abstract_text.clone());
        self.study_alias = Field::from_option(study.alias.clone());
        self.center_project_name = Field::from_option(study.center_project_name.clone());
        self.center_name = Field::from_option(study.center_name.clone());

        // Study-level fallback for the release date, guarded so the run-derived
        // value always wins: this one is the earliest run date across the whole
        // study, so on a multi-experiment study it is a lower bound for this
        // record rather than this record's own date. Only reachable when the
        // experiment reported no runs at all.
        if !self.earliest_run_published.is_settled() {
            let date = study.earliest_run_published.as_ref().and_then(parse_run_published);
            if let Some(date) = date {
                self.earliest_run_published = Field::Known(date, Provenance::Direct);
            }
        }
    }

    // Adds fields to `self` using only direct with only submission metadata.
    #[inline]
    pub fn add_direct_from_submission(&mut self, submission: &Submission) {
        self.submission_accession = Field::from_option(submission.accession.clone());
        self.broker_name = Field::from_option(submission.broker_name.clone());

        // Two sources for one target field, and they are not interchangeable:
        // both are set on 341 of 346 studies and they disagree on 157 of those.
        // The study's own is preferred because it is the one the study declares
        // about itself, and because it is the one that is never absent (346/346
        // vs 341/346), so this branch is the genuine fallback rather than a
        // coin toss. Guarded on is_settled, not called conditionally, so the
        // precedence does not depend on call order.
        if !self.center_name.is_settled() {
            self.center_name = Field::from_option(submission.center_name.clone());
        }
    }

    // Adds fields to `self` using only direct with only BioProject metadata.
    //
    // One field. The BioProject record's title/description are *not* folded into
    // study_title/abstract_text: those already come from the SRA study, and the
    // two are different texts written for different registries.
    #[inline]
    pub fn add_direct_from_bioproject(&mut self, bioproject: &BioProject) {
        self.bioproject_accession = Field::Known(bioproject.accession.clone(), Provenance::Direct);
    }

    // Adds fields to `self` using only direct with only sample/BioSample metadata.
    #[inline]
    pub fn add_direct_from_sample(&mut self, sample: &Sample) {
        self.sample_accession = Field::Known(sample.accession.clone(), Provenance::Direct);
        self.biosample_accession = Field::from_option(sample.biosample_id.clone());
        self.sample_title = Field::from_option(sample.title.clone());
        self.sample_alias = Field::from_option(sample.alias.clone());
        self.scientific_name = Field::from_option(sample.scientific_name.clone());

        // The one direct field that changes type on the way in. A taxon id that
        // does not parse becomes Unknown rather than being carried as text:
        // there is no error channel here, and a Field<u64> that silently held a
        // non-number would be worse than an absent one. All 88,560 values in
        // the corpus parse, so this branch is a guard, not a code path.
        self.taxon_id = match sample.taxon_id.as_ref().map(|x| x.parse::<u64>()) {
            Some(Ok(id)) => Field::Known(id, Provenance::Direct),
            Some(Err(_)) | None => Field::Unknown,
        };

        if let Some(biosample) = &sample.biosample {
            self.add_direct_from_biosample(biosample);
        }
    }

    // Adds fields to `self` using only direct with only BioSample metadata.
    #[inline]
    pub fn add_direct_from_biosample(&mut self, biosample: &BioSample) {
        self.biosample_package = Field::from_option(biosample.package.clone());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Hand-built rather than sliced from the corpus, unlike the tests in
    // `corpus`. Most of what this layer guards against does not occur in the
    // 346-study corpus at all — no experiment is pooled, none has partial run
    // counts, no taxon id fails to parse — so a fixture cut from real data
    // could not reach the branches that most need covering.

    fn date(raw: &str) -> ArchiveDate {
        ArchiveDate { raw: raw.into(), granularity: DateGranularity::Second }
    }

    fn at(raw: &str) -> NaiveDateTime {
        NaiveDateTime::parse_from_str(raw, RUN_PUBLISHED_FORMAT).unwrap()
    }

    fn run(spots: Option<u64>, bases: Option<u64>, published: Option<&str>) -> Run {
        Run {
            accession: RunAccession("SRR000001".into()),
            total_spots: spots,
            total_bases: bases,
            published: published.map(date),
            ..Default::default()
        }
    }

    fn a_sample(accession: &str) -> Sample {
        Sample { accession: SampleAccession(accession.into()), ..Default::default() }
    }

    fn experiment(accession: &str, sample_ids: &[&str]) -> Experiment {
        Experiment {
            accession: ExperimentAccession(accession.into()),
            sample_ids: sample_ids.iter().map(|s| SampleAccession((*s).into())).collect(),
            ..Default::default()
        }
    }

    fn project(experiments: Vec<Experiment>, samples: Vec<Sample>) -> Project {
        Project {
            accession: StudyAccession("SRP000001".into()),
            samples: samples.into_iter().map(|s| (s.accession.clone(), s)).collect(),
            experiments,
            ..Default::default()
        }
    }

    fn run_direct(project: &Project) -> Vec<TargetSchema> {
        let mut schemas = Vec::new();
        process(project, &mut schemas);
        schemas
    }

    // -- runs -------------------------------------------------------------

    #[test]
    fn runs_sum_spots_and_bases_across_the_experiment() {
        let mut schema = TargetSchema::default();
        schema.add_direct_from_runs(vec![
            run(Some(10), Some(100), Some("2019-01-01 00:00:00")),
            run(Some(20), Some(200), Some("2019-01-02 00:00:00")),
        ]);
        assert_eq!(schema.total_spots, Field::Known(30, Provenance::Direct));
        assert_eq!(schema.total_bases, Field::Known(300, Provenance::Direct));
    }

    #[test]
    fn runs_with_partial_counts_report_nothing() {
        // The rule that differs from Python: summing only the runs that report
        // would present an undercount as the experiment total. No experiment in
        // the corpus is like this, so only a unit test can reach it.
        let mut schema = TargetSchema::default();
        schema.add_direct_from_runs(vec![
            run(Some(10), Some(100), Some("2019-01-01 00:00:00")),
            run(None, None, Some("2019-01-02 00:00:00")),
        ]);
        assert_eq!(schema.total_spots, Field::Unknown);
        assert_eq!(schema.total_bases, Field::Unknown);
        // the date is independent of the counts and still resolves
        assert!(schema.earliest_run_published.is_settled());
    }

    #[test]
    fn spots_and_bases_are_judged_separately() {
        // A copy-paste bug once had the date branch clear total_bases. If the
        // two fields were coupled, the reporting one would be lost with it.
        let mut schema = TargetSchema::default();
        schema.add_direct_from_runs(vec![
            run(Some(10), Some(100), Some("2019-01-01 00:00:00")),
            run(Some(20), None, Some("2019-01-02 00:00:00")),
        ]);
        assert_eq!(schema.total_spots, Field::Known(30, Provenance::Direct));
        assert_eq!(schema.total_bases, Field::Unknown);
    }

    #[test]
    fn no_runs_settles_nothing() {
        // `all()` is vacuously true on an empty slice: without the early return
        // a runless experiment would assert a total of zero reads.
        let mut schema = TargetSchema::default();
        schema.add_direct_from_runs(vec![]);
        assert_eq!(schema.total_spots, Field::Unknown);
        assert_eq!(schema.total_bases, Field::Unknown);
        assert_eq!(schema.earliest_run_published, Field::Unknown);
    }

    #[test]
    fn published_takes_the_earliest_not_the_first() {
        // SRA returns runs newest-first, so a first-listed read would report
        // the latest date. This is the ordering that was wrong by 12 years on
        // SRP049009 in the Python harvest.
        let mut schema = TargetSchema::default();
        schema.add_direct_from_runs(vec![
            run(Some(1), Some(1), Some("2026-05-04 11:22:33")),
            run(Some(1), Some(1), Some("2014-03-28 14:17:07")),
            run(Some(1), Some(1), Some("2020-01-01 00:00:00")),
        ]);
        assert_eq!(
            schema.earliest_run_published,
            Field::Known(at("2014-03-28 14:17:07"), Provenance::Direct)
        );
    }

    #[test]
    fn published_is_all_or_nothing_too() {
        // One undated run makes the minimum unknowable: the earliest of the
        // rest is only a bound, and reporting it would look like a fact.
        let mut schema = TargetSchema::default();
        schema.add_direct_from_runs(vec![
            run(Some(1), Some(1), Some("2019-01-01 00:00:00")),
            run(Some(1), Some(1), None),
        ]);
        assert_eq!(schema.earliest_run_published, Field::Unknown);
    }

    #[test]
    fn published_in_an_unexpected_format_is_not_guessed_at() {
        let mut schema = TargetSchema::default();
        schema.add_direct_from_runs(vec![run(Some(1), Some(1), Some("2019-01-01"))]);
        assert_eq!(schema.earliest_run_published, Field::Unknown);
    }

    // -- study, submission, bioproject -------------------------------------

    #[test]
    fn study_fills_its_five_fields() {
        let study = Study {
            title: Some("Gut microbiome of urban foxes".into()),
            abstract_text: Some("We sequenced...".into()),
            alias: Some("GSE123534".into()),
            center_project_name: Some("FOX-2019".into()),
            center_name: Some("CGBIU".into()),
            ..Default::default()
        };
        let mut schema = TargetSchema::default();
        schema.add_direct_from_study(&study);

        assert_eq!(schema.study_title.value().map(String::as_str), Some("Gut microbiome of urban foxes"));
        assert_eq!(schema.abstract_text.value().map(String::as_str), Some("We sequenced..."));
        assert_eq!(schema.study_alias.value().map(String::as_str), Some("GSE123534"));
        assert_eq!(schema.center_project_name.value().map(String::as_str), Some("FOX-2019"));
        assert_eq!(schema.center_name.value().map(String::as_str), Some("CGBIU"));
    }

    #[test]
    fn study_date_is_only_a_fallback() {
        let study = Study {
            earliest_run_published: Some(date("2010-01-01 00:00:00")),
            ..Default::default()
        };

        // unsettled: the study-level date fills in
        let mut empty = TargetSchema::default();
        empty.add_direct_from_study(&study);
        assert_eq!(
            empty.earliest_run_published,
            Field::Known(at("2010-01-01 00:00:00"), Provenance::Direct)
        );

        // settled by the runs: the study's own value must not overwrite it,
        // since it is the earliest across the whole study rather than this
        // record's date
        let mut from_runs = TargetSchema::default();
        from_runs.add_direct_from_runs(vec![run(Some(1), Some(1), Some("2019-06-06 06:06:06"))]);
        from_runs.add_direct_from_study(&study);
        assert_eq!(
            from_runs.earliest_run_published,
            Field::Known(at("2019-06-06 06:06:06"), Provenance::Direct)
        );
    }

    #[test]
    fn submission_center_name_loses_to_the_study() {
        // Both are set on 341 of 346 studies and they disagree on 157, so which
        // one wins is a real choice rather than a formality.
        let study = Study { center_name: Some("STUDY CENTER".into()), ..Default::default() };
        let submission = Submission {
            accession: Some("SRA123456".into()),
            broker_name: Some("ENA".into()),
            center_name: Some("SUBMISSION CENTER".into()),
            ..Default::default()
        };

        let mut schema = TargetSchema::default();
        schema.add_direct_from_study(&study);
        schema.add_direct_from_submission(&submission);
        assert_eq!(schema.center_name.value().map(String::as_str), Some("STUDY CENTER"));
        assert_eq!(schema.submission_accession.value().map(String::as_str), Some("SRA123456"));
        assert_eq!(schema.broker_name.value().map(String::as_str), Some("ENA"));

        // The guard is on is_settled, not on call order, so the study still
        // wins when the submission is applied first.
        let mut reversed = TargetSchema::default();
        reversed.add_direct_from_submission(&submission);
        reversed.add_direct_from_study(&study);
        assert_eq!(reversed.center_name.value().map(String::as_str), Some("STUDY CENTER"));
    }

    #[test]
    fn submission_center_name_is_used_when_the_study_has_none() {
        let submission = Submission {
            center_name: Some("SUBMISSION CENTER".into()),
            ..Default::default()
        };
        let mut schema = TargetSchema::default();
        schema.add_direct_from_study(&Study::default());
        schema.add_direct_from_submission(&submission);
        assert_eq!(schema.center_name.value().map(String::as_str), Some("SUBMISSION CENTER"));
    }

    #[test]
    fn bioproject_supplies_only_its_accession() {
        // Its title and description are deliberately not folded into
        // study_title / abstract_text: different registries, different texts.
        let bioproject = BioProject {
            accession: BioProjectAccession("PRJDB4237".into()),
            title: Some("BioProject title".into()),
            description: Some("BioProject description".into()),
            ..Default::default()
        };
        let mut schema = TargetSchema::default();
        schema.add_direct_from_bioproject(&bioproject);
        assert_eq!(
            schema.bioproject_accession,
            Field::Known(BioProjectAccession("PRJDB4237".into()), Provenance::Direct)
        );
        assert_eq!(schema.study_title, Field::Unknown);
        assert_eq!(schema.abstract_text, Field::Unknown);
    }

    // -- sample and biosample ----------------------------------------------

    #[test]
    fn sample_parses_taxon_id_into_an_integer() {
        let sample = Sample {
            taxon_id: Some("749906".into()),
            scientific_name: Some("gut metagenome".into()),
            biosample_id: Some(BioSampleAccession("SAMD00041293".into())),
            ..a_sample("DRS029834")
        };
        let mut schema = TargetSchema::default();
        schema.add_direct_from_sample(&sample);

        assert_eq!(schema.taxon_id, Field::Known(749906, Provenance::Direct));
        assert_eq!(
            schema.sample_accession,
            Field::Known(SampleAccession("DRS029834".into()), Provenance::Direct)
        );
        assert_eq!(
            schema.biosample_accession,
            Field::Known(BioSampleAccession("SAMD00041293".into()), Provenance::Direct)
        );
    }

    #[test]
    fn unparseable_taxon_id_is_dropped_rather_than_carried_as_text() {
        // All 88,560 corpus values parse, so this branch is a guard. A Field<u64>
        // silently holding a non-number would be worse than an absent one.
        for raw in ["not applicable", "", "9606.0", "-1", "taxid:9606"] {
            let sample = Sample { taxon_id: Some(raw.into()), ..a_sample("SRS000001") };
            let mut schema = TargetSchema::default();
            schema.add_direct_from_sample(&sample);
            assert_eq!(schema.taxon_id, Field::Unknown, "taxon_id {raw:?} should not parse");
        }
    }

    #[test]
    fn biosample_package_comes_through_the_nested_record() {
        let sample = Sample {
            biosample: Some(BioSample {
                package: Some("MIMARKS: survey, host-associated; version 6.0".into()),
                ..Default::default()
            }),
            ..a_sample("SRS000001")
        };
        let mut schema = TargetSchema::default();
        schema.add_direct_from_sample(&sample);
        assert_eq!(
            schema.biosample_package.value().map(String::as_str),
            Some("MIMARKS: survey, host-associated; version 6.0")
        );

        // 1,521 corpus samples have no BioSample record at all
        let mut without = TargetSchema::default();
        without.add_direct_from_sample(&a_sample("SRS000002"));
        assert_eq!(without.biosample_package, Field::Unknown);
    }

    // -- process -----------------------------------------------------------

    #[test]
    fn one_record_per_experiment() {
        let project = project(
            vec![experiment("SRX000001", &["SRS000001"]), experiment("SRX000002", &["SRS000002"])],
            vec![a_sample("SRS000001"), a_sample("SRS000002")],
        );
        let schemas = run_direct(&project);

        assert_eq!(schemas.len(), 2);
        assert_eq!(schemas[0].id, ExperimentAccession("SRX000001".into()));
        assert_eq!(schemas[1].id, ExperimentAccession("SRX000002".into()));
        // the SRP is set here rather than by a helper, since it lives on Project
        assert!(schemas.iter().all(|s| s.study_accession
            == Field::Known(StudyAccession("SRP000001".into()), Provenance::Direct)));
    }

    #[test]
    fn a_pooled_experiment_expands_to_one_record_per_sample() {
        // Zero of the 102,240 corpus experiments are pooled, but the relation is
        // many-to-many in SRA: collapsing it would drop every sample but one.
        let project = project(
            vec![experiment("SRX000001", &["SRS000001", "SRS000002"])],
            vec![a_sample("SRS000001"), a_sample("SRS000002")],
        );
        let schemas = run_direct(&project);

        assert_eq!(schemas.len(), 2);
        // ids must stay unique, so they are suffixed with the sample
        assert_eq!(schemas[0].id, ExperimentAccession("SRX000001.SRS000001".into()));
        assert_eq!(schemas[1].id, ExperimentAccession("SRX000001.SRS000002".into()));
        // the bare experiment accession is still recorded on both
        assert!(schemas.iter().all(|s| s.experiment_accession
            == Field::Known(ExperimentAccession("SRX000001".into()), Provenance::Direct)));
        assert_eq!(
            schemas[1].sample_accession,
            Field::Known(SampleAccession("SRS000002".into()), Provenance::Direct)
        );
    }

    #[test]
    fn an_unpooled_experiment_keeps_the_bare_accession() {
        // The majority case, and the reason for the suffix rule rather than
        // suffixing unconditionally: these stay comparable against ENA's ids.
        let project = project(
            vec![experiment("SRX000001", &["SRS000001"])],
            vec![a_sample("SRS000001")],
        );
        assert_eq!(run_direct(&project)[0].id, ExperimentAccession("SRX000001".into()));
    }

    #[test]
    fn an_unresolvable_sample_id_still_yields_a_record() {
        // The experiment points at a sample the project does not carry. The
        // record is built without sample fields rather than skipped.
        let project = project(vec![experiment("SRX000001", &["SRS999999"])], vec![]);
        let schemas = run_direct(&project);

        assert_eq!(schemas.len(), 1);
        assert_eq!(schemas[0].id, ExperimentAccession("SRX000001".into()));
        assert_eq!(schemas[0].sample_accession, Field::Unknown);
        assert_eq!(schemas[0].taxon_id, Field::Unknown);
        assert!(schemas[0].experiment_accession.is_settled());
    }

    #[test]
    fn an_experiment_with_no_sample_ids_yields_one_record() {
        let project = project(vec![experiment("SRX000001", &[])], vec![]);
        let schemas = run_direct(&project);
        assert_eq!(schemas.len(), 1);
        assert_eq!(schemas[0].sample_accession, Field::Unknown);
    }

    #[test]
    fn a_project_with_no_experiments_yields_no_records() {
        // Python falls back to sample- then study-level stubs; this does not,
        // because `id` is an ExperimentAccession and a study-level stub would
        // have to hold an SRP. Asserted so the difference is deliberate.
        let project = project(vec![], vec![a_sample("SRS000001")]);
        assert!(run_direct(&project).is_empty());
    }

    #[test]
    fn process_appends_rather_than_replacing() {
        // Listing Direct twice duplicates instead of silently discarding the
        // first pass; a visible duplicate is easier to diagnose.
        let project = project(
            vec![experiment("SRX000001", &["SRS000001"])],
            vec![a_sample("SRS000001")],
        );
        let mut schemas = Vec::new();
        process(&project, &mut schemas);
        process(&project, &mut schemas);
        assert_eq!(schemas.len(), 2);
        assert_eq!(schemas[0].id, schemas[1].id);
    }

    #[test]
    fn every_value_the_layer_sets_is_stamped_direct() {
        let mut sample = a_sample("SRS000001");
        sample.taxon_id = Some("9606".into());
        sample.title = Some("liver".into());
        let mut project = project(
            vec![Experiment {
                title: Some("RNA-seq of liver".into()),
                runs: vec![run(Some(5), Some(50), Some("2019-01-01 00:00:00"))],
                ..experiment("SRX000001", &["SRS000001"])
            }],
            vec![sample],
        );
        project.study.title = Some("A study".into());
        project.submission = Some(Submission {
            accession: Some("SRA000001".into()),
            ..Default::default()
        });
        project.bioproject = Some(BioProject {
            accession: BioProjectAccession("PRJNA000001".into()),
            ..Default::default()
        });

        let schemas = run_direct(&project);
        let s = &schemas[0];
        // Nothing here was inferred, so no field may carry a Directness.
        for provenance in [
            s.study_accession.provenance(), s.bioproject_accession.provenance(),
            s.study_title.provenance(), s.submission_accession.provenance(),
            s.sample_accession.provenance(), s.sample_title.provenance(),
            s.taxon_id.provenance(), s.experiment_accession.provenance(),
            s.experiment_title.provenance(), s.total_spots.provenance(),
            s.earliest_run_published.provenance(),
        ] {
            assert_eq!(provenance, Some(&Provenance::Direct));
        }
    }
}