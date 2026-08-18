use std::collections::{BTreeMap, BTreeSet};

use serde_json::{json, Value};

use super::fields::{assign, level_of, open_fields, Level};
use super::ModelConfig;
use crate::model::{Model, ModelError, Request};
use crate::project::{Project, Sample, SampleAccession};
use crate::target_schema::{Directness, Issue, Provenance, SchemaSettings, TargetSchema};

// Layer 3 — reason from titles, abstracts and the sample attribute bag.
//
// The first layer that costs money, and the first that can be wrong in
// interesting ways. Everything above it is a lookup; this one asks a model.
//
// MODEL-AGNOSTIC BY CONSTRUCTION.
// Nothing in this file names a provider. It holds a `&dyn Model`, builds a
// `Request` out of provider-neutral types, and reads back text and usage. It
// does not know what an Anthropic model id is, cannot price a call, and has no
// idea the batch endpoint exists. Adding OpenRouter or a local model is an
// `impl Model` and nothing here changes — which is the point, because this file
// is where the expensive judgements live and it should not be re-edited every
// time the transport does.
//
// The plan is separated from the sending for the same reason `body_for` is
// separated from `complete`: *what gets asked, of whom, about which fields* is
// the part that costs money when wrong, and `plan` is a pure function over the
// records. Every test below exercises it without a model.

// The instructions, sent as the cached prefix on every call.
//
// Two forms, because which one is used is the most consequential variable in
// this layer and worth being able to change in one line.
//
// SHORT is the framing only: answer from the evidence, how to label directness,
// when a missing-value term is an answer. FULL adds what Python's comment calls
// "not padding" — per-field definitions, routing rules for loose attribute
// keys, the common mistakes, and five worked examples. Three error classes were
// traced to its absence there, and the same ones showed up here on the first
// two live studies: the MIxS environmental triad filled for a cultivated plant
// and a gut microbiome, `sequencing_method` duplicating
// `library_construction_protocol`, a filename in `library_name`.
//
// Size is not incidental either. SHORT is ~567 tokens, below Haiku 4.5's
// 4,096-token minimum cacheable prefix, so `cache_control` on it is a silent
// no-op and every call re-bills the instructions — measured, cache_write and
// cache_read both zero across a two-call run. FULL is ~4,163 and clears it.
pub const TEXT_SYSTEM_SHORT: &str = include_str!("prompts/text_system_short.txt");
pub const TEXT_SYSTEM_FULL: &str = include_str!("prompts/text_system_full.txt");

// FULL plus three edits aimed at the two failure classes the audit measured
// over six runs: 60 of 93 `quoted` claims held, and the misses were 4 concluded
// absences and 2 composed sentences, both claiming to be verbatim.
//
// The suspected mechanism for the first is a collision inside the prompt
// itself: MISSING VALUES says to "answer with the exact string", which sits one
// paragraph from a label meaning "word for word". An earlier attempt annotated
// the worked examples instead and moved the rate not at all, so this one says
// it where the confusion is, adds a concrete test for `quoted` (find the span),
// and tells `sample_description` outright that a sentence you composed is
// inferred.
//
// Untested. FULL stays the default until this is measured on the same six-run
// protocol.
pub const TEXT_SYSTEM_TARGETED: &str = include_str!("prompts/text_system_targeted.txt");

// FULL is the default: it is the one that fixes measured errors, and a run that
// wants the cheaper prefix should have to say so.
pub const TEXT_SYSTEM_DEFAULT: &str = TEXT_SYSTEM_FULL;

// The vocabulary the model answers in, and what each token means on the record.
//
// One table for three uses — the wording in TEXT_SYSTEM, the `enum` in the
// answer schema, and the parser below — so a token cannot be added to one and
// missed by another.
//
// The tokens name *what the model did*, not how sure it is. That is the whole
// reason this axis exists: asked for a confidence, the model produced 391 high
// / 41 medium / **0 low** across 432 inferences, with the largest error class
// uniformly `high`. A scale whose bottom rung is never used cannot separate
// right answers from wrong ones. `quoted` is also the only claim that is
// machine-checkable — it says the value appears in the evidence word for word,
// and a string match can say whether it does.
const DIRECTNESS: [(&str, Directness); 3] = [
    ("quoted", Directness::Quoted),
    ("rephrased", Directness::Rephrased),
    ("inferred", Directness::Inferred),
];

fn directness_of(token: &str) -> Directness {
    DIRECTNESS
        .iter()
        .find(|(name, _)| *name == token)
        .map(|(_, directness)| *directness)
        // Anything unrecognised is treated as the weakest claim rather than
        // rejected: the value may still be right, and overstating how it was
        // arrived at is the failure that matters here.
        .unwrap_or(Directness::Inferred)
}

// Fields this layer is never asked about, whatever is open.
//
// Run-, submission- and record-level fields are assigned by the archive at
// deposition — release dates, upload formats, the submitting centre. An
// abstract and an attribute bag state none of them. Python had no such guard
// and layer 3 filled 4,342 of these: a *BioProject* accession written into
// `submission_accession` on 146 records, a strain id written into a run
// accession, and thousands of "not provided" at full token price.
//
// Dropped before anything is planned, so a blind field never reaches a schema,
// a token budget, or an answer.
fn is_blind(name: &str) -> bool {
    matches!(
        level_of(name),
        Some(Level::Run) | Some(Level::Submission) | Some(Level::Record)
    )
}

// Which call a job belongs to.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum JobKey {
    // The one study-level call. `center_project_name` is a property of the
    // study, not of a sample: asking it once per sample was both redundant and
    // wrong, producing two different `study_alias` answers across one study's
    // 15 samples. One study, one answer, by construction.
    Study,
    Sample(SampleAccession),
    // Records whose sample does not resolve. Grouped and asked once rather than
    // skipped — the study text still applies to them.
    Unresolved,
    // Layer 4's shape: one call per publication, covering every record in the
    // study. Here rather than in a second enum because both layers share `Job`
    // and `apply`, and a key that could not say "a paper asked this" would make
    // a reported failure ambiguous between the two layers.
    Paper(String),
}

// One planned call: what to ask, about which fields, on behalf of which records.
#[derive(Clone, Debug, PartialEq)]
pub struct Job {
    pub key: JobKey,
    pub evidence: String,
    pub wanted: Vec<&'static str>,
    // Indices into the schema slice. Indices rather than references because the
    // records are mutated once the answers come back.
    pub targets: Vec<usize>,
}

// Everything the archive offers about one sample, as plain text.
//
// The sample's raw attributes go over un-harmonized. With layer 2 enabled most
// of them are already settled and so are not asked about; with it skipped, this
// is the only place most biological values appear at all.
pub fn sample_evidence(
    project: &Project,
    sample: Option<&Sample>,
    schemas: &[TargetSchema],
    targets: &[usize],
) -> String {
    let mut lines = Vec::new();
    if let Some(title) = &project.study.title {
        lines.push(format!("STUDY TITLE: {title}"));
    }
    if let Some(abstract_text) = &project.study.abstract_text {
        lines.push(format!("STUDY ABSTRACT: {abstract_text}"));
    }
    if let Some(sample) = sample {
        if let Some(name) = &sample.scientific_name {
            lines.push(format!("ORGANISM: {name}"));
        }
        if let Some(title) = &sample.title {
            lines.push(format!("SAMPLE TITLE: {title}"));
        }
        if !sample.attributes.is_empty() {
            let bag = serde_json::to_string(&sample.attributes).unwrap_or_default();
            lines.push(format!("SAMPLE ATTRIBUTES: {bag}"));
        }
    }

    // Sorted and de-duplicated: several experiments on one sample usually share
    // a title, and the evidence is part of the cache key.
    let titles: BTreeSet<&str> = targets
        .iter()
        .filter_map(|i| schemas[*i].experiment_title.value().map(String::as_str))
        .collect();
    if !titles.is_empty() {
        lines.push(format!(
            "EXPERIMENT TITLES: {}",
            titles.into_iter().collect::<Vec<_>>().join("; ")
        ));
    }
    let assays: BTreeSet<String> = targets
        .iter()
        .filter_map(|i| schemas[*i].library_strategy.value().map(|s| format!("{s:?}")))
        .collect();
    if !assays.is_empty() {
        lines.push(format!(
            "LIBRARY STRATEGY: {}",
            assays.into_iter().collect::<Vec<_>>().join(", ")
        ));
    }
    lines.join("\n")
}

// Just the study — no sample attributes, because they cannot inform a
// study-level answer and would only invite one sample's detail to be
// generalised across the whole study.
pub fn study_evidence(project: &Project) -> String {
    let mut lines = Vec::new();
    if let Some(title) = &project.study.title {
        lines.push(format!("STUDY TITLE: {title}"));
    }
    if let Some(abstract_text) = &project.study.abstract_text {
        lines.push(format!("STUDY ABSTRACT: {abstract_text}"));
    }
    if let Some(study_type) = &project.study.study_type {
        lines.push(format!("STUDY TYPE: {study_type:?}"));
    }
    lines.join("\n")
}

// A JSON Schema asking for a *list* of answers.
//
// The obvious shape — one property per field, each an optional
// `{value, confidence}` — is rejected outright. Structured outputs enforce three
// separate limits and asking about ~30 open fields trips all three: at most 16
// union-typed parameters, at most 24 optional ones, and an overall complexity
// ceiling that 24 nested objects exceeds on its own.
//
// A list sidesteps all of it. One property, one item shape, and the field name
// is an `enum` — enums are cheap where properties are not. Declining is simply
// not emitting an item.
//
// `value` is a string for every field, including the integer and date ones:
// `fields::assign` already parses `"9606"` and `"2019-05-04"` on the way in, and
// a per-field value type would reintroduce the union limit for no gain.
pub fn answer_schema(fields: &[&str]) -> Value {
    json!({
        "type": "object",
        "properties": {
            "answers": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "field": { "type": "string", "enum": fields },
                        "value": { "type": "string" },
                        "directness": {
                            "type": "string",
                            "enum": DIRECTNESS.map(|(name, _)| name)
                        }
                    },
                    "required": ["field", "value", "directness"],
                    "additionalProperties": false
                }
            }
        },
        "required": ["answers"],
        "additionalProperties": false
    })
}

// Every call this layer would make, without making any of them.
//
// Two call shapes, split by what the evidence can actually settle: one call per
// study for the study-level fields, and one call per sample for everything
// else. Every experiment on a sample shares its biology, so asking once and
// applying to all of them is cheaper and self-consistent.
pub fn plan(project: &Project, schemas: &[TargetSchema]) -> Vec<Job> {
    // Open fields per record, with the blind ones already gone.
    let open: Vec<Vec<&'static str>> = schemas
        .iter()
        .map(|schema| {
            open_fields(schema)
                .into_iter()
                .filter(|name| !is_blind(name))
                .collect()
        })
        .collect();

    let mut jobs = Vec::new();

    let study_wanted: Vec<&'static str> = union_of(&open, 0..schemas.len())
        .into_iter()
        .filter(|name| level_of(name) == Some(Level::Study))
        .collect();
    if !study_wanted.is_empty() {
        jobs.push(Job {
            key: JobKey::Study,
            evidence: study_evidence(project),
            wanted: study_wanted,
            targets: (0..schemas.len()).collect(),
        });
    }

    let mut by_sample: BTreeMap<Option<SampleAccession>, Vec<usize>> = BTreeMap::new();
    for (index, schema) in schemas.iter().enumerate() {
        by_sample
            .entry(schema.sample_accession.value().cloned())
            .or_default()
            .push(index);
    }

    for (accession, targets) in by_sample {
        let wanted: Vec<&'static str> = union_of(&open, targets.iter().copied())
            .into_iter()
            .filter(|name| level_of(name) != Some(Level::Study))
            .collect();
        if wanted.is_empty() {
            continue;
        }
        let sample = accession.as_ref().and_then(|a| project.samples.get(a));
        jobs.push(Job {
            key: match &accession {
                Some(accession) => JobKey::Sample(accession.clone()),
                None => JobKey::Unresolved,
            },
            evidence: sample_evidence(project, sample, schemas, &targets),
            wanted,
            targets,
        });
    }
    jobs
}

// Sorted so the ask — and therefore the schema, and therefore the cache key —
// is stable across runs.
fn union_of(open: &[Vec<&'static str>], indices: impl IntoIterator<Item = usize>) -> Vec<&'static str> {
    let mut union: BTreeSet<&'static str> = BTreeSet::new();
    for index in indices {
        union.extend(open[index].iter().copied());
    }
    union.into_iter().collect()
}

// One answer as the model gives it.
#[derive(Clone, Debug, PartialEq)]
pub struct Answer {
    pub field: String,
    pub value: String,
    pub directness: String,
}

// Reads the reply. A malformed or absent `answers` list yields nothing rather
// than failing the job: one unusable reply must not cost the whole study.
pub fn read_answers(reply: &Value) -> Vec<Answer> {
    reply["answers"]
        .as_array()
        .map(|items| {
            items
                .iter()
                .filter_map(|item| {
                    Some(Answer {
                        field: item["field"].as_str()?.to_string(),
                        value: item["value"].as_str()?.to_string(),
                        directness: item["directness"].as_str().unwrap_or("inferred").to_string(),
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}

// Writes a job's answers onto the records it was asked on behalf of.
//
// A field is only written where it is still open for *that* record: the ask was
// the union across the group, so an answer can arrive for a record that already
// had the field settled. `assign` refuses those anyway; filtering here keeps the
// intent visible rather than relying on it.
//
// `provenance` is a parameter for the same reason `fields::assign` takes one:
// the routing is identical for layer 3 and layer 4 and only the attribution
// differs, so passing it in is what lets the paper layer reuse this rather than
// keep a second copy that could drift.
pub fn apply(
    schemas: &mut [TargetSchema],
    job: &Job,
    answers: &[Answer],
    provenance: fn(Directness) -> Provenance,
) -> usize {
    let wanted: BTreeSet<&str> = job.wanted.iter().copied().collect();
    let mut filled = 0;
    for answer in answers {
        // The schema constrains this to the enum, but a reply is not a promise.
        if !wanted.contains(answer.field.as_str()) || answer.value.trim().is_empty() {
            continue;
        }
        let provenance = provenance(directness_of(&answer.directness));
        for index in &job.targets {
            if assign(
                &mut schemas[*index],
                &answer.field,
                answer.value.trim(),
                provenance.clone(),
            ) == super::fields::Outcome::Set
            {
                filled += 1;
            }
        }
    }
    filled
}

// Runs the planned calls against whatever model it was handed.
//
// Errors on individual jobs are swallowed deliberately: a refusal or a bad reply
// on one sample leaves that sample's fields open for the paper layer, and must
// not abandon the rest of the study. A budget stop is different — it means every
// remaining call would be refused too — so it ends the layer.
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

    // The plan is identical either way — only the sending differs — so the two
    // paths cannot drift into asking different questions.
    if config.batch {
        process_batched(&jobs, schemas, model, config, settings);
    } else {
        process_sequentially(&jobs, schemas, model, config, settings);
    }
}

fn process_sequentially(
    jobs: &[Job],
    schemas: &mut [TargetSchema],
    model: &dyn Model,
    config: &ModelConfig,
    settings: &SchemaSettings,
) {
    for job in jobs {
        for index in &job.targets {
            settings.record_evidence(&schemas[*index].id.0, "llm_naive", &job.evidence);
        }
        match model.complete(&request_for(job, config)) {
            Ok(response) => absorb(job, &response, schemas, settings),
            Err(error) => {
                let stop = matches!(error, ModelError::BudgetExceeded { .. });
                report(settings, job, &error);
                // A budget stop means every remaining call would be refused
                // too, so continuing is pure waste. Anything else leaves this
                // sample's fields open for the paper layer.
                if stop {
                    return;
                }
            }
        }
    }
}

fn process_batched(
    jobs: &[Job],
    schemas: &mut [TargetSchema],
    model: &dyn Model,
    config: &ModelConfig,
    settings: &SchemaSettings,
) {
    let mut requests = BTreeMap::new();
    for (index, job) in jobs.iter().enumerate() {
        for target in &job.targets {
            settings.record_evidence(&schemas[*target].id.0, "llm_naive", &job.evidence);
        }
        // Positional keys: a job key can be a sample accession containing dots,
        // and the transport maps these onto its own ids anyway.
        requests.insert(format!("j{index}"), request_for(job, config));
    }

    let results = model.complete_many(&requests);
    for (index, job) in jobs.iter().enumerate() {
        match results.get(&format!("j{index}")) {
            Some(Ok(response)) => absorb(job, response, schemas, settings),
            Some(Err(error)) => report(settings, job, error),
            None => report(
                settings,
                job,
                &ModelError::Api {
                    status: 0,
                    kind: Some("batch_result_missing".into()),
                    message: "no result came back for this job".into(),
                },
            ),
        }
    }
}

fn request_for(job: &Job, config: &ModelConfig) -> Request {
    let mut request = Request::new(&job.evidence)
        .system(config.prompt)
        .schema(answer_schema(&job.wanted))
        .thinking(config.thinking);
    request.effort = config.effort;
    request.max_tokens = config.max_tokens;
    request
}

fn absorb(
    job: &Job,
    response: &crate::model::Response,
    schemas: &mut [TargetSchema],
    settings: &SchemaSettings,
) {
    // A clipped answer parses fine and is simply short, so without this a
    // truncated run is indistinguishable from a terse one. Thinking counts
    // toward the same ceiling, which is why it is worth reporting.
    if response.stop_reason.as_deref() == Some("max_tokens") {
        settings.report(Issue {
            layer: "llm_naive",
            context: format!("{:?}", job.key),
            error: format!(
                "response stopped at max_tokens ({} output tokens) — the answer may be incomplete",
                response.usage.output
            ),
        });
    }
    let answers = response.json.as_ref().map(read_answers).unwrap_or_default();
    apply(schemas, job, &answers, Provenance::InferredFromText);
}

fn report(settings: &SchemaSettings, job: &Job, error: &ModelError) {
    settings.report(Issue {
        layer: "llm_naive",
        context: format!("{:?}", job.key),
        error: error.to_string(),
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Response, Usage};
    use crate::project::*;
    use crate::target_schema::Field;

    // No test here reaches a network. The two that exercise `process` hand it a
    // scripted model that answers from memory, which is the whole point of the
    // layer holding a `&dyn Model` rather than a client.

    // The settings under test unless a test says otherwise.
    fn config() -> ModelConfig {
        ModelConfig::new("test", TEXT_SYSTEM_DEFAULT)
    }

    fn sample(accession: &str, attributes: &[(&str, &str)]) -> Sample {
        Sample {
            accession: SampleAccession(accession.into()),
            title: Some(format!("title of {accession}")),
            scientific_name: Some("Mus musculus".into()),
            attributes: attributes
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
            ..Default::default()
        }
    }

    fn project(samples: Vec<Sample>) -> Project {
        Project {
            accession: StudyAccession("SRP000001".into()),
            study: Study {
                title: Some("Gut microbiome of urban foxes".into()),
                abstract_text: Some("We sequenced caecal contents.".into()),
                ..Default::default()
            },
            samples: samples.into_iter().map(|s| (s.accession.clone(), s)).collect(),
            ..Default::default()
        }
    }

    fn record(sample: Option<&str>) -> TargetSchema {
        TargetSchema {
            sample_accession: match sample {
                Some(a) => Field::Known(SampleAccession(a.into()), Provenance::Direct),
                None => Field::Unknown,
            },
            ..Default::default()
        }
    }

    // -- what gets asked ----------------------------------------------------

    #[test]
    fn one_study_call_and_one_call_per_sample() {
        // Asking study-level fields per sample made ~106 duplicate asks over a
        // 60-sample run and produced two different study_alias answers within
        // one study.
        let project = project(vec![sample("SRS1", &[]), sample("SRS2", &[])]);
        let schemas = vec![record(Some("SRS1")), record(Some("SRS1")), record(Some("SRS2"))];
        let jobs = plan(&project, &schemas);

        assert_eq!(jobs.len(), 3);
        assert_eq!(jobs[0].key, JobKey::Study);
        assert_eq!(jobs[0].targets, vec![0, 1, 2], "the study answer applies to every record");
        assert_eq!(jobs[1].key, JobKey::Sample(SampleAccession("SRS1".into())));
        assert_eq!(jobs[1].targets, vec![0, 1], "both experiments on one sample, asked once");
        assert_eq!(jobs[2].key, JobKey::Sample(SampleAccession("SRS2".into())));
    }

    #[test]
    fn the_study_call_asks_only_study_level_fields() {
        let project = project(vec![sample("SRS1", &[])]);
        let jobs = plan(&project, &[record(Some("SRS1"))]);
        let study = &jobs[0];
        assert!(study.wanted.iter().all(|f| level_of(f) == Some(Level::Study)));
        assert!(study.wanted.contains(&"study_alias"));
        assert!(study.wanted.contains(&"center_project_name"));
        // and the sample call asks for none of them
        assert!(jobs[1].wanted.iter().all(|f| level_of(f) != Some(Level::Study)));
    }

    #[test]
    fn the_study_call_never_sees_sample_attributes() {
        // Sample detail cannot inform a study-level answer, and showing it
        // invites one sample's value to be generalised across the study.
        let project = project(vec![sample("SRS1", &[("tissue", "liver")])]);
        let jobs = plan(&project, &[record(Some("SRS1"))]);
        assert!(!jobs[0].evidence.contains("liver"));
        assert!(!jobs[0].evidence.contains("SAMPLE ATTRIBUTES"));
        assert!(jobs[1].evidence.contains("liver"));
    }

    #[test]
    fn archive_assigned_fields_are_never_asked_about() {
        // The guard Python lacked. Measured there: 4,342 of these filled,
        // including a BioProject accession written into submission_accession on
        // 146 records.
        let project = project(vec![sample("SRS1", &[])]);
        let jobs = plan(&project, &[record(Some("SRS1"))]);
        for job in &jobs {
            for field in &job.wanted {
                assert!(!is_blind(field), "{field} should not have been asked");
            }
            for blind in ["submission_accession", "center_name", "total_spots",
                          "submitted_format", "earliest_run_published", "tag"] {
                assert!(!job.wanted.contains(&blind), "{blind} reached the ask");
            }
        }
    }

    #[test]
    fn a_settled_field_is_not_asked_about() {
        let project = project(vec![sample("SRS1", &[])]);
        let mut schema = record(Some("SRS1"));
        schema.host = Field::Known("Mus musculus".into(), Provenance::Direct);
        let jobs = plan(&project, &[schema]);
        assert!(jobs.iter().all(|j| !j.wanted.contains(&"host")));
    }

    #[test]
    fn a_record_with_no_sample_is_still_asked_about() {
        // Grouped rather than skipped: the study text applies to it.
        let project = project(vec![]);
        let jobs = plan(&project, &[record(None), record(None)]);
        let unresolved = jobs.iter().find(|j| j.key == JobKey::Unresolved).unwrap();
        assert_eq!(unresolved.targets, vec![0, 1]);
        assert!(!unresolved.wanted.is_empty());
    }

    #[test]
    fn nothing_is_planned_when_nothing_is_open() {
        // Every field settled by earlier layers means no call, and no spend.
        let project = project(vec![sample("SRS1", &[])]);
        let mut schema = record(Some("SRS1"));
        // A missing-value term settles a field of any type — it is checked
        // before parsing — which is what makes this a one-liner rather than a
        // type-appropriate value per field.
        for name in super::super::fields::FIELD_NAMES {
            assign(&mut schema, name, "not applicable", Provenance::Direct);
        }
        assert!(open_fields(&schema).is_empty(), "precondition: nothing left open");
        assert!(plan(&project, &[schema]).is_empty());
    }

    #[test]
    fn no_records_means_no_calls() {
        assert!(plan(&project(vec![]), &[]).is_empty());
    }

    #[test]
    fn the_ask_is_ordered_so_the_cache_prefix_is_stable() {
        // The field list becomes the schema, which becomes part of what is sent.
        // An unstable order is a different request every run.
        let project = project(vec![sample("SRS1", &[])]);
        let a = plan(&project, &[record(Some("SRS1"))]);
        let b = plan(&project, &[record(Some("SRS1"))]);
        assert_eq!(a, b);
        assert!(a[0].wanted.windows(2).all(|w| w[0] < w[1]), "not sorted");
    }

    // -- the schema sent ----------------------------------------------------

    #[test]
    fn the_answer_schema_is_a_list_not_one_property_per_field() {
        // One property per field trips three separate structured-output limits
        // at ~30 fields. A list has one property and an enum, which is cheap.
        let schema = answer_schema(&["host", "sex", "strain"]);
        assert_eq!(schema["properties"]["answers"]["type"], "array");
        let item = &schema["properties"]["answers"]["items"];
        assert_eq!(item["properties"]["field"]["enum"], json!(["host", "sex", "strain"]));
        assert_eq!(item["properties"]["value"]["type"], "string");
        assert_eq!(
            item["properties"]["directness"]["enum"],
            json!(["quoted", "rephrased", "inferred"])
        );
        // structured outputs reject a schema that allows extra keys
        assert_eq!(item["additionalProperties"], false);
        assert_eq!(schema["additionalProperties"], false);
    }

    #[test]
    fn every_value_is_a_string_whatever_the_field_type() {
        // taxon_id is an integer on the record and a string here: `assign`
        // parses it, and a per-field value type would reintroduce the union
        // limit for no gain.
        let schema = answer_schema(&["taxon_id", "collection_date"]);
        let item = &schema["properties"]["answers"]["items"];
        assert_eq!(item["properties"]["value"]["type"], "string");
    }

    #[test]
    fn the_prompt_the_schema_and_the_parser_share_one_vocabulary() {
        // The failure this guards: adding a token to the enum and forgetting the
        // parser silently downgrades every answer carrying it to `inferred`, and
        // nothing looks broken. One table drives all three; this checks it.
        let schema = answer_schema(&["host"]);
        let enum_tokens = schema["properties"]["answers"]["items"]["properties"]
            ["directness"]["enum"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect::<Vec<_>>();

        assert_eq!(enum_tokens.len(), DIRECTNESS.len());
        for (token, expected) in DIRECTNESS {
            assert!(enum_tokens.contains(&token.to_string()), "{token} missing from the schema");
            assert_eq!(directness_of(token), expected, "{token} not understood by the parser");
            for form in [TEXT_SYSTEM_SHORT, TEXT_SYSTEM_FULL] {
                assert!(form.contains(token), "{token} is never explained to the model");
            }
        }
        // and the axis it replaced is gone from the instructions
        for form in [TEXT_SYSTEM_SHORT, TEXT_SYSTEM_FULL] {
            assert!(!form.contains("high, medium"), "the replaced axis is still explained");
        }
    }

    #[test]
    fn the_full_form_carries_what_the_short_one_omits() {
        // Not a size check for its own sake: each of these sections exists
        // because an error class was traced to its absence, and the first two
        // live studies reproduced them.
        for section in [
            "FIELD DEFINITIONS",
            "ROUTING FREE-TEXT ATTRIBUTES",
            "COMMON MISTAKES TO AVOID",
            "WORKED EXAMPLES",
        ] {
            assert!(TEXT_SYSTEM_FULL.contains(section), "{section} missing");
            assert!(!TEXT_SYSTEM_SHORT.contains(section), "{section} unexpectedly in the short form");
        }
        // The specific guidance for the two errors both live studies produced.
        // Matched on single lines: the prose is hard-wrapped, so a phrase that
        // reads as one sentence is not one substring.
        assert!(TEXT_SYSTEM_FULL.contains("is NOT an environmental"),
                "the triad guidance is missing — the error it prevents occurred twice");
        assert!(TEXT_SYSTEM_FULL.contains("never put"),
                "the sequencing_method / library_construction_protocol rule is missing");
    }

    #[test]
    fn effort_and_thinking_reach_the_request() {
        // Both used to come from Request::default() with no way to change them,
        // so a Sonnet run was silently adaptive and billed 801 output tokens a
        // call for reasoning that is discarded on the way in.
        use crate::model::{Effort, Thinking};

        let project = project(vec![sample("SRS1", &[])]);
        for (effort, thinking) in [
            (Some(Effort::High), Thinking::Disabled),
            (None, Thinking::Unset),
            (Some(Effort::Max), Thinking::Adaptive),
        ] {
            let mut schemas = vec![record(Some("SRS1"))];
            let model = Scripted::new(json!({"answers": []}));
            let config = config().effort(effort).thinking(thinking);
            process(&project, &mut schemas, &model, &config, &SchemaSettings::default());

            let seen = model.seen.lock().unwrap();
            assert_eq!(seen[0].effort, effort, "{effort:?}");
            assert_eq!(seen[0].thinking, thinking, "{thinking:?}");
        }
    }

    #[test]
    fn a_truncated_response_is_reported_not_silently_short() {
        // The gap this closes: a clipped answer parses cleanly and is simply
        // short, so a run stopped at the ceiling looked identical to a terse
        // one. Thinking counts toward the same ceiling, which is why it matters.
        use std::sync::{Arc, Mutex};

        struct Clipped;
        impl Model for Clipped {
            fn price(&self, _u: Usage) -> f64 { 0.0 }
            fn complete(&self, _r: &Request) -> Result<Response, ModelError> {
                let reply = json!({"answers": []});
                Ok(Response {
                    text: reply.to_string(),
                    json: Some(reply),
                    stop_reason: Some("max_tokens".into()),
                    usage: Usage { output: 16_000, ..Default::default() },
                })
            }
        }

        let seen = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&seen);
        let settings = SchemaSettings::default()
            .on_issue(move |i| sink.lock().unwrap().push(i.to_string()));

        let project = project(vec![sample("SRS1", &[])]);
        let mut schemas = vec![record(Some("SRS1"))];
        process(&project, &mut schemas, &Clipped, &config(), &settings);

        let seen = seen.lock().unwrap();
        assert!(!seen.is_empty(), "truncation went unreported");
        assert!(seen[0].contains("max_tokens"), "{}", seen[0]);
        assert!(seen[0].contains("16000"), "{}", seen[0]);
    }

    #[test]
    fn an_ordinary_response_is_not_reported_as_truncated() {
        use std::sync::{Arc, Mutex};
        let seen = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&seen);
        let settings = SchemaSettings::default()
            .on_issue(move |i| sink.lock().unwrap().push(i.to_string()));
        let project = project(vec![sample("SRS1", &[])]);
        let mut schemas = vec![record(Some("SRS1"))];
        process(&project, &mut schemas, &Scripted::new(json!({"answers": []})), &config(), &settings);
        assert!(seen.lock().unwrap().is_empty());
    }

    #[test]
    fn the_token_ceiling_reaches_the_request() {
        let project = project(vec![sample("SRS1", &[])]);
        let mut schemas = vec![record(Some("SRS1"))];
        let model = Scripted::new(json!({"answers": []}));
        process(&project, &mut schemas, &model, &config().max_tokens(4096), &SchemaSettings::default());
        assert_eq!(model.seen.lock().unwrap()[0].max_tokens, 4096);
    }

    #[test]
    fn the_defaults_are_what_every_run_so_far_used() {
        // Pinned rather than assumed: these are what produced the measurements,
        // and changing one silently would invalidate the comparisons.
        use crate::model::{Effort, Thinking};
        let config = ModelConfig::new("test", TEXT_SYSTEM_DEFAULT);
        assert_eq!(config.effort, Some(Effort::Medium));
        assert_eq!(config.thinking, Thinking::Adaptive);
        assert_eq!(config.max_tokens, 16_000, "the value every run so far used");
        assert!(!config.batch, "batching changes latency from seconds to hours");
    }

    #[test]
    fn the_targeted_form_is_full_plus_the_three_edits() {
        // A variant that changed anything else would not be a one-variable test.
        assert!(TEXT_SYSTEM_TARGETED.len() > TEXT_SYSTEM_FULL.len());
        for section in [
            "FIELD DEFINITIONS",
            "ROUTING FREE-TEXT ATTRIBUTES",
            "COMMON MISTAKES TO AVOID",
            "WORKED EXAMPLES",
            "DIRECTNESS",
            "MISSING VALUES",
        ] {
            assert!(TEXT_SYSTEM_TARGETED.contains(section), "{section} lost");
        }
        // the worked examples are untouched
        assert!(TEXT_SYSTEM_TARGETED.contains("Ammonia-oxidising archaea, Monterey Bay"));
    }

    #[test]
    fn the_targeted_form_separates_the_exact_string_from_the_quoted_label() {
        // The suspected mechanism: "answer with the exact string" reads as
        // licence to call the answer quoted. Both prompts say the first; only
        // this one says they are different questions.
        assert!(TEXT_SYSTEM_FULL.contains("answer with the exact string"));
        assert!(TEXT_SYSTEM_TARGETED.contains("does not make your answer quoted"));
        assert!(TEXT_SYSTEM_TARGETED.contains("almost every missing-value term you write is"));
        assert!(!TEXT_SYSTEM_FULL.contains("does not make your answer quoted"));
    }

    #[test]
    fn the_targeted_form_gives_quoted_a_test_and_forbids_composed_descriptions() {
        // The second failure class: two composed sentences claimed verbatim.
        assert!(TEXT_SYSTEM_TARGETED.contains("Before writing quoted, find the span"));
        assert!(TEXT_SYSTEM_TARGETED.contains("composed is inferred, never quoted"));
        assert!(!TEXT_SYSTEM_FULL.contains("find the span"));
    }

    #[test]
    fn every_form_still_speaks_only_the_current_vocabulary() {
        for form in [TEXT_SYSTEM_SHORT, TEXT_SYSTEM_FULL, TEXT_SYSTEM_TARGETED] {
            for stale in ["(high)", "(medium)", "(low)"] {
                assert!(!form.contains(stale), "{stale} present");
            }
            for token in ["quoted", "rephrased", "inferred"] {
                assert!(form.contains(token), "{token} never explained");
            }
        }
    }

    #[test]
    fn the_default_is_still_the_measured_form_not_the_new_one() {
        // TARGETED is untested. Promoting an unmeasured prompt to the default
        // is how the thing being measured stops being the thing that runs.
        assert_eq!(TEXT_SYSTEM_DEFAULT, TEXT_SYSTEM_FULL);
    }

    #[test]
    fn both_forms_share_the_framing() {
        // Whatever else differs, the rules the parser depends on must not.
        for section in ["DIRECTNESS", "MISSING VALUES"] {
            assert!(TEXT_SYSTEM_SHORT.contains(section), "{section} missing from short");
            assert!(TEXT_SYSTEM_FULL.contains(section), "{section} missing from full");
        }
    }

    #[test]
    fn the_full_form_clears_the_cache_floor_the_short_one_does_not() {
        // Measured on a two-call run: cache_write and cache_read both zero,
        // because a prefix below the model's minimum is a silent no-op and the
        // instructions are re-billed every call. Haiku 4.5's floor is 4,096
        // tokens; this approximates at four characters per token.
        const HAIKU_FLOOR_TOKENS: usize = 4096;
        assert!(TEXT_SYSTEM_SHORT.len() / 4 < HAIKU_FLOOR_TOKENS);
        assert!(TEXT_SYSTEM_FULL.len() / 4 >= HAIKU_FLOOR_TOKENS,
                "full form is ~{} tokens", TEXT_SYSTEM_FULL.len() / 4);
    }

    #[test]
    fn no_worked_example_still_speaks_the_old_vocabulary() {
        // The examples label every answer, so a missed translation would teach
        // the model a token the schema does not allow.
        for form in [TEXT_SYSTEM_SHORT, TEXT_SYSTEM_FULL] {
            for stale in ["(high)", "(medium)", "(low)"] {
                assert!(!form.contains(stale), "{stale} survived the translation");
            }
        }
        // and the examples do use the new ones
        assert!(TEXT_SYSTEM_FULL.contains("(quoted)"));
        assert!(TEXT_SYSTEM_FULL.contains("(rephrased)"));
        assert!(TEXT_SYSTEM_FULL.contains("(inferred)"));
    }

    #[test]
    fn a_declared_absence_is_taught_as_inferred() {
        // The one mislabel both live runs produced: `collected_by` answered
        // "not provided" and labelled quoted, when nothing in the evidence
        // said it. The examples now say so where they demonstrate DECLARE.
        assert!(TEXT_SYSTEM_FULL.contains("DECLARE  (inferred"));
        assert!(TEXT_SYSTEM_FULL.contains("every DECLARE in the examples below is"));
    }

    #[test]
    fn the_layer_sends_whichever_prompt_its_config_names() {
        // The A/B, as a test: one line changes what every call carries. Per
        // layer, so layer 4 can be reading its own prompt at the same time.
        let project = project(vec![sample("SRS1", &[])]);
        for prompt in [TEXT_SYSTEM_SHORT, TEXT_SYSTEM_FULL, TEXT_SYSTEM_TARGETED] {
            let mut schemas = vec![record(Some("SRS1"))];
            let model = Scripted::new(json!({"answers": []}));
            let config = ModelConfig::new("test", prompt);
            process(&project, &mut schemas, &model, &config, &SchemaSettings::default());
            assert_eq!(model.seen.lock().unwrap()[0].system.as_deref(), Some(prompt));
        }
    }

    // -- applying answers ---------------------------------------------------

    fn answer(field: &str, value: &str, directness: &str) -> Answer {
        Answer {
            field: field.into(),
            value: value.into(),
            directness: directness.into(),
        }
    }

    #[test]
    fn an_answer_reaches_every_record_the_job_covered() {
        let project = project(vec![sample("SRS1", &[])]);
        let mut schemas = vec![record(Some("SRS1")), record(Some("SRS1"))];
        let jobs = plan(&project, &schemas);
        let job = jobs.iter().find(|j| matches!(j.key, JobKey::Sample(_))).unwrap();

        apply(&mut schemas, job, &[answer("host", "Mus musculus", "quoted")], Provenance::InferredFromText);
        for schema in &schemas {
            assert_eq!(
                schema.host,
                Field::Known(
                    "Mus musculus".into(),
                    Provenance::InferredFromText(Directness::Quoted)
                )
            );
        }
    }

    #[test]
    fn the_token_becomes_the_directness() {
        // The token names what the model did, not how sure it is — which is why
        // it lands on the provenance rather than beside it as a score.
        for (token, expected) in [
            ("quoted", Directness::Quoted),
            ("rephrased", Directness::Rephrased),
            ("inferred", Directness::Inferred),
            ("something else", Directness::Inferred), // weakest claim, not a reject
            ("high", Directness::Inferred),           // the old vocabulary is not accepted
        ] {
            let project = project(vec![sample("SRS1", &[])]);
            let mut schemas = vec![record(Some("SRS1"))];
            let jobs = plan(&project, &schemas);
            let job = jobs.iter().find(|j| matches!(j.key, JobKey::Sample(_))).unwrap();
            apply(&mut schemas, job, &[answer("host", "Mus musculus", token)], Provenance::InferredFromText);
            assert_eq!(
                schemas[0].host.provenance(),
                Some(&Provenance::InferredFromText(expected)),
                "{token}"
            );
        }
    }

    #[test]
    fn an_answer_outside_the_ask_is_refused() {
        // The schema constrains the model to the enum, but a reply is not a
        // promise — and this is the guard that stopped 4,342 archive fields.
        let project = project(vec![sample("SRS1", &[])]);
        let mut schemas = vec![record(Some("SRS1"))];
        let jobs = plan(&project, &schemas);
        let job = jobs.iter().find(|j| matches!(j.key, JobKey::Sample(_))).unwrap();

        let filled = apply(
            &mut schemas,
            job,
            &[
                answer("submission_accession", "PRJNA293224", "quoted"),
                answer("host", "Mus musculus", "quoted"),
            ],
            Provenance::InferredFromText,
        );
        assert_eq!(filled, 1);
        assert_eq!(schemas[0].submission_accession, Field::Unknown);
    }

    #[test]
    fn an_answer_never_overwrites_an_earlier_layer() {
        let project = project(vec![sample("SRS1", &[])]);
        let mut schemas = vec![record(Some("SRS1"))];
        let jobs = plan(&project, &schemas);
        let job = jobs.iter().find(|j| matches!(j.key, JobKey::Sample(_))).unwrap();

        schemas[0].host = Field::Known("from layer 1".into(), Provenance::Direct);
        apply(&mut schemas, job, &[answer("host", "from the model", "quoted")], Provenance::InferredFromText);
        assert_eq!(
            schemas[0].host,
            Field::Known("from layer 1".into(), Provenance::Direct)
        );
    }

    #[test]
    fn an_empty_value_is_not_an_answer() {
        let project = project(vec![sample("SRS1", &[])]);
        let mut schemas = vec![record(Some("SRS1"))];
        let jobs = plan(&project, &schemas);
        let job = jobs.iter().find(|j| matches!(j.key, JobKey::Sample(_))).unwrap();
        assert_eq!(apply(&mut schemas, job, &[answer("host", "   ", "quoted")], Provenance::InferredFromText), 0);
    }

    #[test]
    fn a_missing_value_term_is_recorded_as_a_declared_absence() {
        // The model is told to answer these rather than omit the field, and the
        // record has a variant for them.
        let project = project(vec![sample("SRS1", &[])]);
        let mut schemas = vec![record(Some("SRS1"))];
        let jobs = plan(&project, &schemas);
        let job = jobs.iter().find(|j| matches!(j.key, JobKey::Sample(_))).unwrap();
        apply(&mut schemas, job, &[answer("host", "not applicable", "quoted")], Provenance::InferredFromText);
        assert!(matches!(schemas[0].host, Field::Missing(..)));
    }

    // -- reading the reply --------------------------------------------------

    #[test]
    fn answers_are_read_out_of_the_reply() {
        let reply = json!({"answers": [
            {"field": "host", "value": "Mus musculus", "directness": "quoted"},
            {"field": "sex", "value": "female", "directness": "rephrased"}
        ]});
        assert_eq!(
            read_answers(&reply),
            vec![
                answer("host", "Mus musculus", "quoted"),
                answer("sex", "female", "rephrased")
            ]
        );
    }

    #[test]
    fn a_reply_with_no_answers_is_not_a_failure() {
        // Declining every field is a legitimate outcome and the common one.
        assert!(read_answers(&json!({"answers": []})).is_empty());
        assert!(read_answers(&json!({})).is_empty());
        assert!(read_answers(&json!({"answers": "nonsense"})).is_empty());
    }

    #[test]
    fn a_malformed_item_is_skipped_rather_than_failing_the_job() {
        let reply = json!({"answers": [
            {"field": "host"},
            {"field": "sex", "value": "female", "directness": "quoted"}
        ]});
        assert_eq!(read_answers(&reply).len(), 1);
    }

    // -- model agnosticism --------------------------------------------------

    // Answers from memory. It is not a Claude client, has no key, no prices and
    // no network — and the layer cannot tell.
    struct Scripted {
        reply: Value,
        seen: std::sync::Mutex<Vec<Request>>,
    }

    impl Scripted {
        fn new(reply: Value) -> Self {
            Self { reply, seen: std::sync::Mutex::new(Vec::new()) }
        }
    }

    impl Model for Scripted {
        fn price(&self, _usage: Usage) -> f64 {
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

    #[test]
    fn the_layer_runs_against_any_model_at_all() {
        // The requirement, as a test: nothing in this file names a provider, so
        // a model that never opens a socket drives it just as well as one that
        // does. Adding OpenRouter is an `impl Model` and no edit here.
        let project = project(vec![sample("SRS1", &[("tissue", "liver")])]);
        let mut schemas = vec![record(Some("SRS1"))];
        let model = Scripted::new(json!({"answers": [
            {"field": "host", "value": "Mus musculus", "directness": "quoted"}
        ]}));

        process(&project, &mut schemas, &model, &config(), &SchemaSettings::default());

        assert_eq!(
            schemas[0].host,
            Field::Known("Mus musculus".into(), Provenance::InferredFromText(Directness::Quoted))
        );
        let seen = model.seen.lock().unwrap();
        assert_eq!(seen.len(), 2, "one study call and one sample call");
        // and what it was handed is provider-neutral
        assert_eq!(seen[0].system.as_deref(), Some(TEXT_SYSTEM_DEFAULT));
        assert!(seen[0].schema.is_some());
        assert!(seen[0].cache_system, "the instructions are the cached prefix");
    }

    #[test]
    fn batching_asks_the_same_questions_as_the_sequential_path() {
        // The plan is shared, so the two paths must differ only in how the
        // requests are sent. A batch that asked different questions would make
        // every cost comparison between them meaningless.
        let project = project(vec![sample("SRS1", &[]), sample("SRS2", &[])]);
        let reply = json!({"answers": [
            {"field": "host", "value": "Mus musculus", "directness": "quoted"}
        ]});

        let mut seq = vec![record(Some("SRS1")), record(Some("SRS2"))];
        let m1 = Scripted::new(reply.clone());
        process(&project, &mut seq, &m1, &config(), &SchemaSettings::default());

        let mut batched = vec![record(Some("SRS1")), record(Some("SRS2"))];
        let m2 = Batching(Scripted::new(reply));
        process(&project, &mut batched, &m2, &config().batch(true), &SchemaSettings::default());

        assert_eq!(seq, batched, "the two paths produced different records");
        let a = m1.seen.lock().unwrap();
        let b = m2.0.seen.lock().unwrap();
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x, y, "a batched request differs from its live twin");
        }
    }

    // Routes complete_many through one call, the way a real batch endpoint
    // does, so the layer's batch path is exercised without a network.
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

    #[test]
    fn a_per_key_batch_failure_does_not_cost_the_other_jobs() {
        // One rejected request in a batch must leave the rest applied, the same
        // rule the sequential path follows for one failed call.
        struct HalfFailing;
        impl Model for HalfFailing {
            fn price(&self, _u: Usage) -> f64 { 0.0 }
            fn complete(&self, _r: &Request) -> Result<Response, ModelError> {
                unreachable!("the batch path should be used")
            }
            fn supports_batch(&self) -> bool { true }
            fn complete_many(
                &self,
                requests: &BTreeMap<String, Request>,
            ) -> BTreeMap<String, Result<Response, ModelError>> {
                let reply = json!({"answers": [
                    {"field": "host", "value": "Mus musculus", "directness": "quoted"}
                ]});
                // Fails the LAST job. The plan is [Study, Sample(SRS1),
                // Sample(SRS2)], so failing the first would only lose the
                // study-level call and both records would still be filled by
                // their own sample jobs — the test would pass for the wrong
                // reason.
                let last = requests.len() - 1;
                requests.keys().enumerate().map(|(i, k)| {
                    (k.clone(), if i == last {
                        Err(ModelError::Refused { category: None, explanation: None })
                    } else {
                        Ok(Response { text: reply.to_string(), json: Some(reply.clone()),
                                      stop_reason: None, usage: Usage::default() })
                    })
                }).collect()
            }
        }

        let project = project(vec![sample("SRS1", &[]), sample("SRS2", &[])]);
        let mut schemas = vec![record(Some("SRS1")), record(Some("SRS2"))];
        process(&project, &mut schemas, &HalfFailing, &config().batch(true), &SchemaSettings::default());

        // `host` is sample-level, so only the failed sample's record misses it.
        assert!(schemas[0].host.is_settled(), "the surviving job should have applied");
        assert_eq!(schemas[1].host, Field::Unknown, "the refused job filled nothing");
    }

    #[test]
    fn a_provider_without_a_batch_endpoint_still_works_batched() {
        // The default `complete_many` is a loop, so asking for a batch from a
        // model that has none is a no-op rather than an error.
        let project = project(vec![sample("SRS1", &[])]);
        let mut schemas = vec![record(Some("SRS1"))];
        let model = Scripted::new(json!({"answers": [
            {"field": "host", "value": "Mus musculus", "directness": "quoted"}
        ]}));
        assert!(!model.supports_batch());
        process(&project, &mut schemas, &model, &config().batch(true), &SchemaSettings::default());
        assert!(schemas[0].host.is_settled());
    }

    #[test]
    fn a_failing_job_does_not_abandon_the_study() {
        // A refusal on one sample leaves its fields open for the paper layer.
        // Ending the whole study there would forfeit every other sample too.
        struct Refusing;
        impl Model for Refusing {
            fn price(&self, _u: Usage) -> f64 {
                0.0
            }
            fn complete(&self, _r: &Request) -> Result<Response, ModelError> {
                Err(ModelError::Refused { category: None, explanation: None })
            }
        }
        let project = project(vec![sample("SRS1", &[])]);
        let mut schemas = vec![record(Some("SRS1"))];
        process(&project, &mut schemas, &Refusing, &config(), &SchemaSettings::default());
        assert_eq!(schemas[0].host, Field::Unknown);
    }

    #[test]
    fn a_failure_is_reported_rather_than_discarded() {
        // The bug this exists for: the first live run printed a full record and
        // "calls 0, billed $0.00", which looked like a layer with nothing to
        // say. The call had been made and had failed, and the error was thrown
        // away.
        use std::sync::{Arc, Mutex};

        struct Refusing;
        impl Model for Refusing {
            fn price(&self, _u: Usage) -> f64 {
                0.0
            }
            fn complete(&self, _r: &Request) -> Result<Response, ModelError> {
                Err(ModelError::Api {
                    status: 400,
                    kind: Some("invalid_request_error".into()),
                    message: "credit balance is too low".into(),
                })
            }
        }

        let seen = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&seen);
        let settings = SchemaSettings::default()
            .on_issue(move |issue| sink.lock().unwrap().push(issue.to_string()));

        let project = project(vec![sample("SRS1", &[])]);
        let mut schemas = vec![record(Some("SRS1"))];
        process(&project, &mut schemas, &Refusing, &config(), &settings);

        // One per failed job — this fixture plans a study call and a sample
        // call, and both failed, so both are reported. Reporting only the first
        // would understate how much of the study went unanswered.
        let seen = seen.lock().unwrap();
        assert_eq!(seen.len(), plan(&project, &[record(Some("SRS1"))]).len());
        assert!(seen.iter().all(|line| line.contains("llm_naive")));
        assert!(seen.iter().all(|line| line.contains("credit balance is too low")));
        assert!(seen.iter().any(|line| line.contains("Study")), "{seen:?}");
    }

    #[test]
    fn a_run_with_no_sink_still_does_not_abort() {
        // Reporting is opt-in; not opting in must not change what happens.
        struct Refusing;
        impl Model for Refusing {
            fn price(&self, _u: Usage) -> f64 {
                0.0
            }
            fn complete(&self, _r: &Request) -> Result<Response, ModelError> {
                Err(ModelError::Overloaded)
            }
        }
        let project = project(vec![sample("SRS1", &[]), sample("SRS2", &[])]);
        let mut schemas = vec![record(Some("SRS1")), record(Some("SRS2"))];
        process(&project, &mut schemas, &Refusing, &config(), &SchemaSettings::default());
        assert!(schemas.iter().all(|s| s.host == Field::Unknown));
    }

    #[test]
    fn a_budget_stop_ends_the_layer_rather_than_retrying_every_job() {
        // Unlike a refusal, this one means every remaining call would be
        // refused as well, so continuing is pure waste.
        struct Broke {
            calls: std::sync::Mutex<u32>,
        }
        impl Model for Broke {
            fn price(&self, _u: Usage) -> f64 {
                0.0
            }
            fn complete(&self, _r: &Request) -> Result<Response, ModelError> {
                *self.calls.lock().unwrap() += 1;
                Err(ModelError::BudgetExceeded { spent: 2.0, limit: 1.0 })
            }
        }
        let project = project(vec![sample("SRS1", &[]), sample("SRS2", &[])]);
        let mut schemas = vec![record(Some("SRS1")), record(Some("SRS2"))];
        let model = Broke { calls: std::sync::Mutex::new(0) };
        process(&project, &mut schemas, &model, &config(), &SchemaSettings::default());
        assert_eq!(*model.calls.lock().unwrap(), 1, "stopped at the first refusal");
    }
}
