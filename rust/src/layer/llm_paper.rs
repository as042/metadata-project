use std::collections::{BTreeMap, BTreeSet};

use super::fields::{level_of, open_fields, Level};
use super::llm_naive::{answer_schema, apply, read_answers, Answer, Job, JobKey};
use super::ModelConfig;
use crate::model::{Model, ModelError, Request, Response};
use crate::project::{Paper, Project};
use crate::target_schema::{Issue, Provenance, SchemaSettings, TargetSchema};

// Layer 4 — reason from the study's publication.
//
// The last layer, and the one with the worst ratio of cost to coverage. A full
// paper is tens of thousands of tokens against ~540 for a sample's archive
// evidence, so this layer's single call can cost more than every layer-3 call
// in the same study combined. It runs last so it is only ever asked about what
// nothing cheaper could answer.
//
// One call per paper, not per sample. A paper describes the *study*: it says
// "we sequenced tumour and adjacent normal tissue" without saying which sample
// is which, so a per-sample ask would invite a distinction the text does not
// draw. The answers apply to every record the paper covers.
//
// Model-agnostic on the same terms as layer 3: a `&dyn Model` and a
// `ModelConfig`, no provider named anywhere in the file.

// The instructions: layer 3's, plus what changes when the evidence is prose.
//
// Composed at compile time from the same file layer 3 uses, so the field
// definitions, routing rules and worked examples cannot drift between the two
// layers — a field meaning one thing in one layer and another in the next is
// exactly the confusion this schema exists to remove.
pub const PAPER_SYSTEM: &str = concat!(
    include_str!("prompts/text_system_full.txt"),
    include_str!("prompts/paper_addendum.txt")
);

// Characters of paper text sent per study.
//
// A tight budget on purpose. The stored text is Methods-first, which is where
// sample provenance actually lives, and 91% of the corpus's papers already hit
// this cap when they were harvested — so raising it here would not recover the
// rest of the article, only what the harvest kept.
pub const PAPER_MAX_CHARS: usize = 30_000;

// Fields a paper cannot speak to, on top of the archive-assigned levels layer 3
// already refuses.
//
// Not a prompt instruction. The Python prompt already said "never infer these
// from a paper or an abstract" and the model did it anyway, 14 times per field,
// because the field list it is shown outranks the prose telling it not to.
//
// These sit at sample or experiment level in the schema but are the same kind
// of thing as a run accession: registration artefacts and submitter-chosen
// identifiers, not science. On the measured Python run they accounted for ~150
// of layer 4's 262 filled fields, essentially all of them "not provided".
pub const PAPER_BLIND_FIELDS: &[&str] = &[
    "checklist",           // ENA checklist identifier
    "biosample_package",   // Python's ncbi_reporting_standard
    "sample_alias",        // submitter-chosen identifiers, all three
    "experiment_alias",
    "study_alias",
    "library_name",
    "sample_capture_status", // controlled INSDC vocabulary, not prose
];

fn is_blind(name: &str) -> bool {
    matches!(
        level_of(name),
        Some(Level::Run) | Some(Level::Submission) | Some(Level::Record)
    ) || PAPER_BLIND_FIELDS.contains(&name)
}

// What earlier layers already settled, shown to the model as context.
//
// Two jobs at once. It stops the model re-answering what is already known —
// which costs tokens and risks a worse guess arriving alongside a direct value
// — and it gives the paper something to reason against: knowing the organism is
// a gut metagenome is what makes "honey bee" a host rather than the subject.
//
// Only fields every record agrees on. One call answers for the whole study, so
// the context it reasons from has to be true of the whole study: a tissue type
// settled on one sample and not another is exactly the kind of fact that would
// invite the model to generalise it across the rest.
//
// Only ordinary values, too. A field settled as a stated absence is not context
// worth reasoning from, and listing it invites the model to argue with it.
pub fn established(records: &[TargetSchema]) -> BTreeMap<&'static str, String> {
    let mut out = BTreeMap::new();
    let Some(first) = records.first() else {
        return out;
    };
    let open: BTreeSet<&str> = records.iter().flat_map(open_fields).collect();
    for name in super::fields::FIELD_NAMES {
        if open.contains(name) {
            continue;
        }
        let Some(text) = value_text(first, name) else { continue };
        if records.iter().all(|r| value_text(r, name).as_deref() == Some(text.as_str())) {
            out.insert(*name, text);
        }
    }
    out
}

// Reads a field's value as text, if it has an ordinary one.
//
// Through the serialisation rather than by matching 62 fields by hand: `Field`
// is externally tagged, so a settled ordinary value is `{"Known": [value,
// provenance]}` whatever the value's type is. A stated absence serialises as
// `Missing` and falls through, which is the behaviour wanted. Typed values that
// do not render as a scalar — a partial date, an accession newtype — are
// skipped rather than half-rendered; this is context for a reader.
fn value_text(record: &TargetSchema, name: &str) -> Option<String> {
    let json = serde_json::to_value(record).ok()?;
    let known = json.get(name)?.get("Known")?.as_array()?;
    match known.first()? {
        serde_json::Value::String(text) => Some(text.clone()),
        serde_json::Value::Number(number) => Some(number.to_string()),
        _ => None,
    }
}

// The paper, plus the study title and what is already known.
pub fn paper_evidence(project: &Project, paper: &Paper, records: &[TargetSchema]) -> String {
    let mut text = paper.text.clone().unwrap_or_default();
    if text.chars().count() > PAPER_MAX_CHARS {
        text = text.chars().take(PAPER_MAX_CHARS).collect();
    }
    let known = serde_json::to_string(&established(records)).unwrap_or_default();
    format!(
        "STUDY TITLE: {}\nALREADY ESTABLISHED (do not answer these again): {}\n\n\
         PUBLICATION ({}):\n{}",
        project.study.title.clone().unwrap_or_default(),
        known,
        paper.id,
        text
    )
}

// Every call this layer would make, without making any of them.
//
// One job per paper with text. A study can carry several — 37 of 346 do — and
// each is asked separately rather than concatenated: two papers on one study
// are two sources, and merging them would make a claim about which said what
// unanswerable.
pub fn plan(project: &Project, schemas: &[TargetSchema]) -> Vec<Job> {
    if schemas.is_empty() {
        return Vec::new();
    }
    let wanted: Vec<&'static str> = {
        let mut union: BTreeSet<&'static str> = BTreeSet::new();
        for schema in schemas {
            union.extend(open_fields(schema).into_iter().filter(|n| !is_blind(n)));
        }
        union.into_iter().collect()
    };
    if wanted.is_empty() {
        return Vec::new();
    }

    let targets: Vec<usize> = (0..schemas.len()).collect();
    crate::corpus::papers_of(project)
        .into_iter()
        .map(|paper| Job {
            // Layer 3's `Job`, so both layers share `apply` and the batch
            // plumbing and cannot drift in how an answer reaches a record.
            key: JobKey::Paper(paper.id.clone()),
            evidence: paper_evidence(project, paper, schemas),
            wanted: wanted.clone(),
            targets: targets.clone(),
        })
        .collect()
}

// Runs the planned calls. Mirrors layer 3 deliberately: the difference between
// the layers is what they read, not how they are driven.
pub fn process(
    project: &Project,
    schemas: &mut [TargetSchema],
    model: &dyn Model,
    config: &ModelConfig,
    settings: &SchemaSettings,
) {
    let jobs = plan(project, schemas);
    if jobs.is_empty() {
        return;
    }

    let request_for = |job: &Job| {
        let mut request = Request::new(&job.evidence)
            .system(config.prompt)
            .schema(answer_schema(&job.wanted))
            .thinking(config.thinking);
        request.effort = config.effort;
        request.max_tokens = config.max_tokens;
        request
    };

    let absorb = |job: &Job, response: &Response, schemas: &mut [TargetSchema]| {
        if response.stop_reason.as_deref() == Some("max_tokens") {
            settings.report(Issue {
                layer: "llm_paper",
                context: format!("{:?}", job.key),
                error: format!(
                    "response stopped at max_tokens ({} output tokens) — the answer may be \
                     incomplete",
                    response.usage.output
                ),
            });
        }
        let answers: Vec<Answer> = response.json.as_ref().map(read_answers).unwrap_or_default();
        // The one thing this layer does differently once an answer is in hand.
        // A paper-sourced value has to be distinguishable from an archive-text
        // one: it is the weaker source, and every count and audit downstream
        // separates them.
        apply(schemas, job, &answers, Provenance::InferredFromPaper);
    };
    let report = |job: &Job, error: &ModelError| {
        settings.report(Issue {
            layer: "llm_paper",
            context: format!("{:?}", job.key),
            error: error.to_string(),
        });
    };

    // The evidence is per-paper and applies to every record, so each record is
    // credited with all of it.
    for job in &jobs {
        for index in &job.targets {
            settings.record_evidence(&schemas[*index].id.0, "llm_paper", &job.evidence);
        }
    }

    if config.batch {
        let requests: BTreeMap<String, Request> = jobs
            .iter()
            .enumerate()
            .map(|(index, job)| (format!("p{index}"), request_for(job)))
            .collect();
        let results = model.complete_many(&requests);
        for (index, job) in jobs.iter().enumerate() {
            match results.get(&format!("p{index}")) {
                Some(Ok(response)) => absorb(job, response, schemas),
                Some(Err(error)) => report(job, error),
                None => report(
                    job,
                    &ModelError::Api {
                        status: 0,
                        kind: Some("batch_result_missing".into()),
                        message: "no result came back for this paper".into(),
                    },
                ),
            }
        }
    } else {
        for job in &jobs {
            match model.complete(&request_for(job)) {
                Ok(response) => absorb(job, &response, schemas),
                Err(error) => {
                    let stop = matches!(error, ModelError::BudgetExceeded { .. });
                    report(job, &error);
                    if stop {
                        return;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Thinking, Usage};
    use crate::project::*;
    use crate::target_schema::{Directness, Field, Provenance};
    use serde_json::json;

    fn paper(id: &str, text: &str) -> Publication {
        Publication {
            id: id.into(),
            paper: Some(Paper {
                id: id.into(),
                text: Some(text.into()),
                char_count: text.len(),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    fn project(papers: Vec<Publication>) -> Project {
        Project {
            accession: StudyAccession("SRP000001".into()),
            study: Study {
                title: Some("Honey bee gut microbiome".into()),
                ..Default::default()
            },
            publications: papers,
            ..Default::default()
        }
    }

    fn record() -> TargetSchema {
        TargetSchema {
            id: ExperimentAccession("SRX1".into()),
            ..Default::default()
        }
    }

    fn config() -> ModelConfig {
        ModelConfig::new("test", PAPER_SYSTEM)
    }

    struct Scripted {
        reply: serde_json::Value,
        seen: std::sync::Mutex<Vec<Request>>,
    }

    impl Scripted {
        fn new(reply: serde_json::Value) -> Self {
            Self { reply, seen: std::sync::Mutex::new(Vec::new()) }
        }
    }

    impl Model for Scripted {
        fn price(&self, _u: Usage) -> f64 {
            0.0
        }
        fn complete(&self, request: &Request) -> Result<Response, ModelError> {
            self.seen.lock().unwrap().push(request.clone());
            Ok(Response {
                text: self.reply.to_string(),
                json: Some(self.reply.clone()),
                stop_reason: None,
                usage: Usage::default(),
            })
        }
    }

    // -- what gets asked ----------------------------------------------------

    #[test]
    fn one_call_per_paper_covering_every_record() {
        // A paper describes the study, so asking per sample would invite a
        // distinction the text does not draw — and cost one full paper per
        // sample instead of one per study.
        let project = project(vec![paper("PMID1", "we sequenced honey bee guts")]);
        let schemas = vec![record(), record(), record()];
        let jobs = plan(&project, &schemas);
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].targets, vec![0, 1, 2]);
    }

    #[test]
    fn several_papers_are_asked_separately() {
        // 37 of 346 studies carry more than one. Concatenating them would make
        // "which paper said this" unanswerable.
        let project = project(vec![paper("PMID1", "first paper"), paper("PMID2", "second paper")]);
        let jobs = plan(&project, &[record()]);
        assert_eq!(jobs.len(), 2);
        assert!(jobs[0].evidence.contains("first paper"));
        assert!(jobs[1].evidence.contains("second paper"));
    }

    #[test]
    fn a_study_with_no_retrievable_text_is_not_asked_at_all() {
        // The common case: a publication classified `oa` whose text never came
        // back. Asking anyway would spend a paper-sized call on nothing.
        let mut without = project(vec![]);
        without.publications = vec![Publication { id: "PMID1".into(), paper: None, ..Default::default() }];
        assert!(plan(&without, &[record()]).is_empty());
    }

    #[test]
    fn the_paper_blind_fields_are_never_asked_about() {
        // Registration artefacts, not science. On the measured Python run these
        // were ~150 of layer 4's 262 filled fields, essentially all of them
        // "not provided".
        let project = project(vec![paper("PMID1", "text")]);
        let jobs = plan(&project, &[record()]);
        for blind in PAPER_BLIND_FIELDS {
            assert!(!jobs[0].wanted.contains(blind), "{blind} reached the ask");
        }
        // and layer 3's archive-level guard still applies on top
        for archive in ["submission_accession", "total_spots", "tag", "center_name"] {
            assert!(!jobs[0].wanted.contains(&archive), "{archive} reached the ask");
        }
    }

    #[test]
    fn layer_four_asks_a_strict_subset_of_what_layer_three_asks() {
        // The two guards compose rather than diverge: everything the archive
        // layer refuses to ask, the paper layer refuses too, plus its own list.
        let project = project(vec![paper("PMID1", "text")]);
        let schemas = vec![record()];
        let paper_wanted: BTreeSet<&str> = plan(&project, &schemas)[0].wanted.iter().copied().collect();
        let text_wanted: BTreeSet<&str> = super::super::llm_naive::plan(&project, &schemas)
            .iter()
            .flat_map(|job| job.wanted.iter().copied())
            .collect();

        assert!(paper_wanted.is_subset(&text_wanted));
        assert!(paper_wanted.len() < text_wanted.len());
        let dropped: BTreeSet<&str> = text_wanted.difference(&paper_wanted).copied().collect();
        assert_eq!(dropped, PAPER_BLIND_FIELDS.iter().copied().collect());
    }

    #[test]
    fn nothing_is_planned_when_nothing_is_open() {
        let project = project(vec![paper("PMID1", "text")]);
        let mut schema = record();
        for name in super::super::fields::FIELD_NAMES {
            super::super::fields::assign(&mut schema, name, "not applicable", Provenance::Direct);
        }
        assert!(plan(&project, &[schema]).is_empty());
    }

    // -- the evidence -------------------------------------------------------

    #[test]
    fn the_paper_is_truncated_to_the_budget() {
        // 91% of stored papers already hit this cap at harvest, so this mostly
        // guards against a long one rather than trimming the typical case.
        let long = "x".repeat(PAPER_MAX_CHARS * 2);
        let project = project(vec![paper("PMID1", &long)]);
        let evidence = &plan(&project, &[record()])[0].evidence;
        let body = evidence.split("PUBLICATION").nth(1).unwrap();
        assert!(body.chars().filter(|c| *c == 'x').count() == PAPER_MAX_CHARS);
    }

    #[test]
    fn what_earlier_layers_settled_is_shown_as_context() {
        // Two jobs: it stops the model re-answering what is known, and gives it
        // something to reason against.
        let project = project(vec![paper("PMID1", "text")]);
        let mut schema = record();
        schema.scientific_name = Field::Known("gut metagenome".into(), Provenance::Direct);
        schema.host = Field::Known("Apis mellifera".into(), Provenance::Harmonized);

        let evidence = &plan(&project, &[schema])[0].evidence;
        assert!(evidence.contains("ALREADY ESTABLISHED"));
        assert!(evidence.contains("gut metagenome"));
        assert!(evidence.contains("Apis mellifera"));
    }

    #[test]
    fn a_settled_field_is_context_and_not_a_question() {
        let project = project(vec![paper("PMID1", "text")]);
        let mut schema = record();
        schema.host = Field::Known("Apis mellifera".into(), Provenance::Direct);
        let job = &plan(&project, &[schema])[0];
        assert!(job.evidence.contains("Apis mellifera"));
        assert!(!job.wanted.contains(&"host"), "a settled field must not be asked again");
    }

    #[test]
    fn a_declared_absence_is_not_offered_as_context() {
        // "not applicable" is not something to reason from, and listing it
        // invites the model to argue with it.
        let mut schema = record();
        schema.sex = Field::Missing(
            crate::target_schema::MissingReason::NotApplicable,
            Provenance::Harmonized,
        );
        assert!(!established(&[schema]).contains_key("sex"));
    }

    #[test]
    fn context_is_only_what_the_whole_study_agrees_on() {
        // One call answers for every record, so context true of only one of
        // them is worse than no context: it is exactly the fact the model would
        // generalise across the rest.
        let mut a = record();
        let mut b = record();
        a.tissue_type = Field::Known("tumour".into(), Provenance::Direct);
        b.tissue_type = Field::Known("adjacent normal".into(), Provenance::Direct);
        a.host = Field::Known("Homo sapiens".into(), Provenance::Direct);
        b.host = Field::Known("Homo sapiens".into(), Provenance::Direct);

        let known = established(&[a, b]);
        assert_eq!(known.get("host").map(String::as_str), Some("Homo sapiens"));
        assert!(!known.contains_key("tissue_type"), "a per-sample fact was offered as study context");
    }

    #[test]
    fn established_reads_numbers_as_well_as_strings() {
        let mut schema = record();
        schema.taxon_id = Field::Known(7460, Provenance::Direct);
        assert_eq!(established(&[schema]).get("taxon_id").map(String::as_str), Some("7460"));
    }

    // -- the prompt ---------------------------------------------------------

    #[test]
    fn the_paper_prompt_is_the_text_prompt_plus_the_addendum() {
        // Composed at compile time from the same file, so the field definitions
        // cannot drift between the two layers.
        assert!(PAPER_SYSTEM.starts_with(super::super::llm_naive::TEXT_SYSTEM_FULL));
        assert!(PAPER_SYSTEM.contains("READING A PAPER"));
        assert!(PAPER_SYSTEM.len() > super::super::llm_naive::TEXT_SYSTEM_FULL.len());
    }

    #[test]
    fn the_addendum_speaks_the_directness_vocabulary() {
        // Python's version said "medium at best"; the axis it belongs to no
        // longer exists.
        assert!(PAPER_SYSTEM.contains("rephrased at best"));
        for stale in ["(high)", "(medium)", "(low)", "medium at best"] {
            assert!(!PAPER_SYSTEM.contains(stale), "{stale} survived");
        }
    }

    // -- running ------------------------------------------------------------

    #[test]
    fn answers_reach_every_record_the_paper_covered() {
        let project = project(vec![paper("PMID1", "honey bee guts")]);
        let mut schemas = vec![record(), record()];
        let model = Scripted::new(json!({"answers": [
            {"field": "host", "value": "Apis mellifera", "directness": "rephrased"}
        ]}));
        process(&project, &mut schemas, &model, &config(), &SchemaSettings::default());

        for schema in &schemas {
            assert_eq!(
                schema.host,
                Field::Known(
                    "Apis mellifera".into(),
                    Provenance::InferredFromPaper(Directness::Rephrased)
                )
            );
        }
    }

    #[test]
    fn the_layer_sends_its_own_config_not_the_text_layers() {
        // The whole point of per-layer settings: layer 4 reads 30,000
        // characters of prose and layer 3 reads an attribute bag, so they
        // should be tunable apart.
        let project = project(vec![paper("PMID1", "text")]);
        let mut schemas = vec![record()];
        let model = Scripted::new(json!({"answers": []}));
        let config = ModelConfig::new("paper-model", PAPER_SYSTEM)
            .thinking(Thinking::Disabled)
            .max_tokens(8000)
            .effort(None);
        process(&project, &mut schemas, &model, &config, &SchemaSettings::default());

        let seen = model.seen.lock().unwrap();
        assert_eq!(seen[0].system.as_deref(), Some(PAPER_SYSTEM));
        assert_eq!(seen[0].thinking, Thinking::Disabled);
        assert_eq!(seen[0].max_tokens, 8000);
        assert_eq!(seen[0].effort, None);
    }

    #[test]
    fn a_failure_is_reported_and_leaves_the_records_alone() {
        struct Refusing;
        impl Model for Refusing {
            fn price(&self, _u: Usage) -> f64 {
                0.0
            }
            fn complete(&self, _r: &Request) -> Result<Response, ModelError> {
                Err(ModelError::Refused { category: None, explanation: None, usage: Usage::default() })
            }
        }
        use std::sync::{Arc, Mutex};
        let seen = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&seen);
        let settings = SchemaSettings::default()
            .on_issue(move |i| sink.lock().unwrap().push(i.to_string()));

        let project = project(vec![paper("PMID1", "text")]);
        let mut schemas = vec![record()];
        process(&project, &mut schemas, &Refusing, &config(), &settings);

        assert_eq!(schemas[0].host, Field::Unknown);
        let seen = seen.lock().unwrap();
        assert_eq!(seen.len(), 1);
        assert!(seen[0].contains("llm_paper"));
    }

    #[test]
    fn a_budget_stop_ends_the_layer_instead_of_paying_to_be_refused() {
        // The one error worth abandoning the study over: it means every
        // remaining call would be refused too. A paper call is the most
        // expensive in the cascade, so continuing is the costliest way to fail.
        struct Broke(std::sync::Mutex<usize>);
        impl Model for Broke {
            fn price(&self, _u: Usage) -> f64 {
                0.0
            }
            fn complete(&self, _r: &Request) -> Result<Response, ModelError> {
                *self.0.lock().unwrap() += 1;
                Err(ModelError::BudgetExceeded { limit: 0.25, spent: 0.25 })
            }
        }
        let model = Broke(std::sync::Mutex::new(0));
        let project = project(vec![paper("PMID1", "a"), paper("PMID2", "b"), paper("PMID3", "c")]);
        let mut schemas = vec![record()];
        process(&project, &mut schemas, &model, &config(), &SchemaSettings::default());
        assert_eq!(*model.0.lock().unwrap(), 1, "the layer kept calling after the budget stop");
    }

    #[test]
    fn a_batch_result_that_never_arrives_is_reported() {
        // A silently absent result reads exactly like a paper with nothing to
        // say, which is the failure mode the issue sink exists for.
        struct Empty;
        impl Model for Empty {
            fn price(&self, _u: Usage) -> f64 {
                0.0
            }
            fn complete(&self, _r: &Request) -> Result<Response, ModelError> {
                unreachable!("batched")
            }
            fn supports_batch(&self) -> bool {
                true
            }
            fn complete_many(
                &self,
                _requests: &BTreeMap<String, Request>,
            ) -> BTreeMap<String, Result<Response, ModelError>> {
                BTreeMap::new()
            }
        }
        use std::sync::{Arc, Mutex};
        let seen = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&seen);
        let settings = SchemaSettings::default()
            .on_issue(move |i| sink.lock().unwrap().push(i.to_string()));

        let project = project(vec![paper("PMID1", "a")]);
        let mut schemas = vec![record()];
        process(&project, &mut schemas, &Empty, &config().batch(true), &settings);

        let seen = seen.lock().unwrap();
        assert_eq!(seen.len(), 1);
        assert!(seen[0].contains("PMID1"), "{}", seen[0]);
    }

    #[test]
    fn batching_asks_the_same_questions_as_the_sequential_path() {
        struct Batching(Scripted);
        impl Model for Batching {
            fn price(&self, u: Usage) -> f64 {
                self.0.price(u)
            }
            fn complete(&self, r: &Request) -> Result<Response, ModelError> {
                self.0.complete(r)
            }
            fn supports_batch(&self) -> bool {
                true
            }
            fn complete_many(
                &self,
                requests: &BTreeMap<String, Request>,
            ) -> BTreeMap<String, Result<Response, ModelError>> {
                requests.iter().map(|(k, r)| (k.clone(), self.0.complete(r))).collect()
            }
        }

        let project = project(vec![paper("PMID1", "a"), paper("PMID2", "b")]);
        let reply = json!({"answers": [
            {"field": "host", "value": "Apis mellifera", "directness": "rephrased"}
        ]});

        let mut seq = vec![record()];
        let m1 = Scripted::new(reply.clone());
        process(&project, &mut seq, &m1, &config(), &SchemaSettings::default());

        let mut bat = vec![record()];
        let m2 = Batching(Scripted::new(reply));
        process(&project, &mut bat, &m2, &config().batch(true), &SchemaSettings::default());

        assert_eq!(seq, bat);
        assert_eq!(m1.seen.lock().unwrap().len(), m2.0.seen.lock().unwrap().len());
    }
}
