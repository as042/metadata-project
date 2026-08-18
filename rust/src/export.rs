use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::model::budget::Budget;
use crate::model::Usage;
use crate::target_schema::{Directness, Provenance, SchemaSettings, TargetSchema};

// Saving a run so it can be compared against another one.
//
// The point is not backup, it is science: two runs differing in one variable —
// a prompt, a model, an effort level — are only comparable if both were kept
// along with what produced them. A file of records with no record of the
// settings is a result nobody can interpret six weeks later.
//
// So every export carries its parameters and a provenance histogram beside the
// records. The histogram is the thing worth diffing: how many fields each layer
// settled, and for the inferred ones, how the model said it got there.

// What new exports are stamped with. Bumped whenever the shape changes, so a
// file says which shape it is rather than leaving a reader to guess.
//
// v1 -> v2 added `text_system_chars`, `effort`, `thinking`, `max_tokens` and
// `evidence`. v2 -> v3 added `batch`. v3 -> v4 moved the model settings into
// `models`, one entry per model layer, because a run can now give layer 3 and
// layer 4 different models and settings and the flat fields could only describe
// one of them. Every one of them defaults, which is what makes the older
// versions still readable — see READABLE_VERSIONS.
pub const FORMAT_VERSION: u32 = 4;

// Versions this build can load.
//
// A version stays readable as long as every field added since has a default:
// the missing ones then read as absent rather than failing. That is a real
// constraint on how the shape may change, not a courtesy — `Usage::thinking`
// was once added without one and silently orphaned 26 saved runs, which
// surfaced only as a comparison that had quietly lost two thirds of its data.
//
// Drop a version from this list when a change cannot be defaulted, and the
// error below will say so instead of a file half-loading.
pub const READABLE_VERSIONS: &[u32] = &[1, 2, 3, 4];

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Export {
    pub format_version: u32,
    // RFC 3339, UTC. Also the filename, so runs sort chronologically and one
    // cannot silently overwrite another.
    pub created: String,
    pub params: Params,
    pub counts: Counts,
    // Present only when the run asked for it. Keyed by record id, holding the
    // union of every call's evidence that record was covered by.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evidence: Option<std::collections::BTreeMap<String, String>>,
    pub records: Vec<TargetSchema>,
}

// One model layer's settings, as they were when the run happened.
//
// A list of these rather than a flat block, because layers are tuned apart:
// layer 3 asks two dozen short questions against an attribute bag and layer 4
// asks a handful against 30,000 characters of prose, and the whole reason
// `ModelConfig` is per-layer is that the best model for one is not obviously
// the best for the other. A run that puts Haiku on layer 3 and Sonnet on layer
// 4 has to be able to say so, and the v3 shape could only name one of them.
//
// `model` comes from `ModelConfig::label` — caller-supplied, because a layer
// holds a `dyn Model` and deliberately cannot ask it what it is.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LayerParams {
    pub layer: String,
    pub model: String,
    // Which instructions this layer sent, by size. The single most consequential
    // variable in the cascade, so a saved run that does not record it cannot be
    // compared against one that used another form.
    pub prompt_chars: usize,
    // Both change what a run costs and what it answers, so a saved run that
    // does not name them cannot be compared with one that used other values.
    pub effort: Option<String>,
    pub thinking: String,
    pub max_tokens: u32,
    // Whether this layer's calls went as one batch. Recorded because it halves
    // the bill and nothing else about the request changes — so without it, two
    // runs differing only in this are told apart by dividing cost by usage,
    // which works while the discount is exactly 0.5 and stops the moment it is
    // not (measured: 45%, because the fan-out adds a cache write).
    pub batch: bool,
}

// What produced this run.
//
// Every field defaults, so a Params written by an older build still loads with
// the fields it never had reading as absent. `Option` already behaves this way
// in serde; the attribute is written out anyway so the rule stays visible for
// the next field added, which may not be an Option.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Params {
    #[serde(default)]
    pub layers: Vec<String>,
    #[serde(default)]
    pub max_studies: Option<usize>,
    #[serde(default)]
    pub max_total_records: Option<usize>,
    #[serde(default)]
    pub max_spend: Option<f64>,
    // One entry per model layer, in cascade order. Empty on a free run.
    #[serde(default)]
    pub models: Vec<LayerParams>,
    // -- v1..v3 shape, still written when it is not a lie ------------------
    //
    // These described the run's single model layer. They are derived now rather
    // than set, and left out when the run's model layers disagree — a v3 reader
    // seeing one value is then reading something that was true of every layer.
    // Kept because the 15 saved runs are all v2/v3 and every comparison against
    // them goes through these fields; a bump that silently stopped writing them
    // would strand that history without saying so.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text_system_chars: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effort: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub batch: Option<bool>,
    #[serde(default)]
    pub note: Option<String>,
}

impl Params {
    // Free-text, for the variable under test: "full TEXT_SYSTEM", "effort=high".
    pub fn note(mut self, note: impl Into<String>) -> Self {
        self.note = Some(note.into());
        self
    }

    // The v1..v3 flat fields, filled from the model layers when they agree.
    //
    // Agreeing is the condition, not "there is exactly one layer": two layers
    // on the same model with the same settings describe themselves perfectly
    // well in one row, and refusing that would strand the ordinary case.
    fn summarise(mut self, models: &[LayerParams]) -> Self {
        let Some(first) = models.first() else { return self };
        let uniform = |f: fn(&LayerParams) -> String| models.iter().all(|m| f(m) == f(first));
        if uniform(|m| m.model.clone()) {
            self.model = Some(first.model.clone());
        }
        if uniform(|m| format!("{:?}", (m.prompt_chars, &m.effort, &m.thinking, m.max_tokens, m.batch)))
        {
            self.text_system_chars = Some(first.prompt_chars);
            self.effort = first.effort.clone();
            self.thinking = Some(first.thinking.clone());
            self.max_tokens = Some(first.max_tokens);
            self.batch = Some(first.batch);
        }
        self
    }
}

// Field-slots by how they were settled, summed across every record.
//
// Slots rather than fields: a run over 40 records has 40 chances at `host`, and
// the interesting number is how many of them landed.
// Same rule as Params: a counter added later must read as zero on an older
// file rather than failing the load.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Counts {
    pub records: usize,
    pub direct: usize,
    pub harmonized: usize,
    pub inferred_from_text: usize,
    pub inferred_from_paper: usize,
    // Of the inferred ones, how the model said it got there. `quoted` is the
    // only claim a checker can falsify, which is why it is worth its own count.
    pub quoted: usize,
    pub rephrased: usize,
    pub inferred: usize,
    // Settled as a stated absence rather than an ordinary value, across every
    // layer. Counted separately because "answered not applicable" and "filled"
    // are different outcomes that both close a field.
    pub declared_missing: usize,
    pub open: usize,
    pub calls: u64,
    pub usage: Usage,
    pub spent: f64,
}

impl Counts {
    fn add(&mut self, provenance: Option<&Provenance>, missing: bool) {
        let Some(provenance) = provenance else {
            self.open += 1;
            return;
        };
        if missing {
            self.declared_missing += 1;
        }
        match provenance {
            Provenance::Direct => self.direct += 1,
            Provenance::Harmonized => self.harmonized += 1,
            Provenance::InferredFromText(d) => {
                self.inferred_from_text += 1;
                self.add_directness(*d);
            }
            Provenance::InferredFromPaper(d) => {
                self.inferred_from_paper += 1;
                self.add_directness(*d);
            }
        }
    }

    fn add_directness(&mut self, directness: Directness) {
        match directness {
            Directness::Quoted => self.quoted += 1,
            Directness::Rephrased => self.rephrased += 1,
            Directness::Inferred => self.inferred += 1,
        }
    }

    // Every slot is accounted for exactly once.
    pub fn settled(&self) -> usize {
        self.direct + self.harmonized + self.inferred_from_text + self.inferred_from_paper
    }
}

// The field list appears once here. A field missing from it is simply not
// counted, which the drift test at the bottom of this module catches.
fn tally(records: &[TargetSchema]) -> Counts {
    let mut counts = Counts { records: records.len(), ..Default::default() };
    for record in records {
        macro_rules! count {
            ($($field:ident),+ $(,)?) => {
                $( counts.add(
                    record.$field.provenance(),
                    matches!(record.$field, crate::target_schema::Field::Missing(..)),
                ); )+
            };
        }
        count!(
            bioproject_accession, study_accession, study_title, abstract_text,
            study_alias, center_project_name, center_name, submission_accession,
            broker_name, biosample_accession, sample_accession, sample_title,
            sample_alias, scientific_name, taxon_id, biosample_package,
            experiment_accession, experiment_title, experiment_alias,
            library_strategy, library_source, library_selection, library_layout,
            library_name, library_construction_protocol, platform,
            instrument_model, total_spots, total_bases, earliest_run_published,
            age, broad_scale_environmental_context, cell_line, cell_type,
            checklist, collected_by, collection_date, country, datahub,
            dev_stage, environment_biome, environment_feature,
            environment_material, environmental_medium, first_created, host,
            host_scientific_name, host_sex, host_tax_id, isolation_source,
            last_updated, local_environmental_context, sample_capture_status,
            sample_description, sequencing_method, sex, strain,
            submitted_format, submitted_read_type, tag, tissue_type, treatment,
        );
    }
    counts
}

impl Export {
    // Builds an export from a finished run. `params` carries whatever the
    // settings cannot know — chiefly which model answered.
    pub fn new(
        records: Vec<TargetSchema>,
        settings: &SchemaSettings,
        budget: &Budget,
        params: Params,
    ) -> Self {
        let mut counts = tally(&records);
        let ledger = budget.ledger();
        counts.calls = ledger.calls;
        counts.usage = ledger.usage;
        counts.spent = ledger.spent;

        let models: Vec<LayerParams> = settings
            .layers()
            .iter()
            .filter_map(|layer| Some((layer, layer.config()?)))
            .map(|(layer, config)| LayerParams {
                layer: layer.name().to_string(),
                model: config.label.clone(),
                prompt_chars: config.prompt.len(),
                effort: config.effort.map(|e| e.as_str().to_string()),
                thinking: config.thinking.as_str().to_string(),
                max_tokens: config.max_tokens,
                batch: config.batch,
            })
            .collect();

        let mut params = Params {
            layers: settings.layers().iter().map(|l| l.name().to_string()).collect(),
            max_studies: settings.study_limit(),
            max_total_records: settings.record_limit(),
            max_spend: budget.limit(),
            ..params
        }
        .summarise(&models);
        params.models = models;

        Self {
            format_version: FORMAT_VERSION,
            created: chrono::Utc::now().to_rfc3339(),
            params,
            counts,
            evidence: settings.evidence(),
            records,
        }
    }

    // Writes to `dir/<timestamp>.json`, creating the directory if needed.
    //
    // Timestamped rather than named, because the failure this exists to prevent
    // is a run overwriting the one it was supposed to be compared against.
    pub fn save(&self, dir: impl AsRef<Path>) -> std::io::Result<PathBuf> {
        let dir = dir.as_ref();
        std::fs::create_dir_all(dir)?;
        // Second precision and no colons: `created` keeps the exact instant,
        // and the filename only has to sort and be legal on every filesystem.
        let stamp = chrono::DateTime::parse_from_rfc3339(&self.created)
            .map(|t| t.format("%Y%m%dT%H%M%SZ").to_string())
            .unwrap_or_else(|_| self.created.replace(':', "-"));
        let path = dir.join(format!("{stamp}.json"));
        std::fs::write(&path, serde_json::to_string_pretty(self)?)?;
        Ok(path)
    }

    pub fn load(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let raw = std::fs::read_to_string(path)?;
        let export: Self = serde_json::from_str(&raw)?;
        if !READABLE_VERSIONS.contains(&export.format_version) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "export is format_version {}; this build reads {READABLE_VERSIONS:?} \
                     and writes {FORMAT_VERSION}",
                    export.format_version
                ),
            ));
        }
        Ok(export)
    }
}

// A one-line summary, which is what a run usually wants printed.
impl std::fmt::Display for Export {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let c = &self.counts;
        write!(
            f,
            "{} records | direct {} harmonized {} text {} (quoted {} rephrased {} inferred {}) \
             | missing {} open {} | {} calls ${:.6}{}",
            c.records, c.direct, c.harmonized, c.inferred_from_text,
            c.quoted, c.rephrased, c.inferred, c.declared_missing, c.open,
            c.calls, c.spent,
            if c.usage.thinking > 0 {
                format!(" ({} thinking tokens)", c.usage.thinking)
            } else {
                String::new()
            }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layer::fields::FIELD_NAMES;
    use crate::target_schema::{Field, MissingReason};

    fn record() -> TargetSchema {
        TargetSchema {
            study_title: Field::Known("t".into(), Provenance::Direct),
            country: Field::Known("USA".into(), Provenance::Harmonized),
            host: Field::Known(
                "Apis mellifera".into(),
                Provenance::InferredFromText(Directness::Rephrased),
            ),
            sequencing_method: Field::Known(
                "16S".into(),
                Provenance::InferredFromText(Directness::Quoted),
            ),
            collected_by: Field::Missing(
                MissingReason::NotProvided,
                Provenance::InferredFromText(Directness::Inferred),
            ),
            ..Default::default()
        }
    }

    #[test]
    fn every_field_slot_is_accounted_for_exactly_once() {
        // The field list in `tally` is a second copy of the schema's fields; a
        // field missing from it would simply not be counted, and the totals
        // would quietly stop adding up.
        let counts = tally(&[TargetSchema::default(), TargetSchema::default()]);
        assert_eq!(counts.open + counts.settled(), FIELD_NAMES.len() * 2);
        assert_eq!(counts.open, FIELD_NAMES.len() * 2);
    }

    #[test]
    fn the_histogram_splits_by_layer_and_by_directness() {
        let counts = tally(&[record()]);
        assert_eq!(counts.direct, 1);
        assert_eq!(counts.harmonized, 1);
        assert_eq!(counts.inferred_from_text, 3);
        assert_eq!(counts.quoted, 1);
        assert_eq!(counts.rephrased, 1);
        assert_eq!(counts.inferred, 1);
        assert_eq!(counts.settled(), 5);
        assert_eq!(counts.open, FIELD_NAMES.len() - 5);
    }

    #[test]
    fn a_declared_absence_counts_as_settled_and_as_missing() {
        // Both, deliberately: it closed a field *and* it is not an ordinary
        // value, and a comparison between runs wants to see both facts.
        let counts = tally(&[record()]);
        assert_eq!(counts.declared_missing, 1);
        assert_eq!(counts.inferred_from_text, 3, "the missing one is still a text answer");
    }

    #[test]
    fn counts_scale_with_records() {
        let one = tally(&[record()]);
        let two = tally(&[record(), record()]);
        assert_eq!(two.records, 2);
        assert_eq!(two.settled(), one.settled() * 2);
    }

    #[test]
    fn an_export_round_trips_through_json() {
        // The whole point of saving is reading back later, so this has to hold
        // for every variant a record can contain — a partial date, a stated
        // absence, an inferred provenance.
        let export = Export {
            format_version: FORMAT_VERSION,
            created: "2026-08-17T12:00:00+00:00".into(),
            params: Params::default().note("baseline"),
            counts: tally(&[record()]),
            evidence: None,
            records: vec![record()],
        };
        let json = serde_json::to_string(&export).unwrap();
        let back: Export = serde_json::from_str(&json).unwrap();
        assert_eq!(back, export);
        assert_eq!(back.records[0].collected_by, export.records[0].collected_by);
    }

    #[test]
    fn saving_twice_does_not_overwrite() {
        // The failure this file naming exists to prevent: a second run landing
        // on the first one, which is only noticed when the comparison is due.
        let dir = std::env::temp_dir().join(format!("mp-export-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        let mut a = Export {
            format_version: FORMAT_VERSION,
            created: "2026-08-17T12:00:00+00:00".into(),
            params: Params::default(),
            counts: Counts::default(),
            evidence: None,
            records: vec![],
        };
        let first = a.save(&dir).unwrap();
        a.created = "2026-08-17T12:00:01+00:00".into();
        let second = a.save(&dir).unwrap();

        assert_ne!(first, second);
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 2);
        assert!(!first.to_string_lossy().contains(':'), "filename must be portable");
        // and they sort chronologically, which is how a comparison finds them
        assert!(first < second);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_saved_export_loads_back() {
        let dir = std::env::temp_dir().join(format!("mp-load-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let export = Export {
            format_version: FORMAT_VERSION,
            created: "2026-08-17T12:00:00+00:00".into(),
            params: Params::default().note("baseline"),
            counts: tally(&[record()]),
            evidence: None,
            records: vec![record()],
        };
        let path = export.save(&dir).unwrap();
        assert_eq!(Export::load(&path).unwrap(), export);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_v1_export_still_loads_after_the_bump() {
        // The point of the version list. v1 predates five fields; every one of
        // them defaults, so the file reads with those absent rather than being
        // orphaned. Written as raw JSON on purpose — constructing it through
        // the current struct would not reproduce an older shape.
        let dir = std::env::temp_dir().join(format!("mp-v1-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("old.json");
        std::fs::write(
            &path,
            r#"{
                "format_version": 1,
                "created": "2026-08-17T12:00:00+00:00",
                "params": {"layers": ["direct"], "max_studies": 1,
                           "max_total_records": 2, "max_spend": 0.25,
                           "model": "claude-haiku-4-5", "note": null},
                "counts": {"records": 1, "direct": 5, "harmonized": 0,
                           "inferred_from_text": 0, "inferred_from_paper": 0,
                           "quoted": 0, "rephrased": 0, "inferred": 0,
                           "declared_missing": 0, "open": 57, "calls": 1,
                           "usage": {"input": 10, "output": 2,
                                     "cache_write": 0, "cache_read": 0},
                           "spent": 0.001},
                "records": []
            }"#,
        )
        .unwrap();

        let export = Export::load(&path).unwrap();
        assert_eq!(export.format_version, 1);
        assert_eq!(export.params.model.as_deref(), Some("claude-haiku-4-5"));
        // the fields v1 never had read as absent, not as an error
        assert_eq!(export.params.text_system_chars, None);
        assert_eq!(export.params.effort, None);
        assert_eq!(export.params.thinking, None);
        assert_eq!(export.params.max_tokens, None);
        assert_eq!(export.params.batch, None);
        assert_eq!(export.evidence, None);
        // and the counter added to Usage reads as zero
        assert_eq!(export.counts.usage.thinking, 0);
        assert_eq!(export.counts.usage.input, 10);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_v2_export_still_loads_after_the_bump() {
        // The two 30-record batching runs are v2. A bump that orphaned them
        // would lose the only comparison that measured the discount.
        let dir = std::env::temp_dir().join(format!("mp-v2-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("v2.json");
        std::fs::write(&path, r#"{
            "format_version": 2,
            "created": "2026-08-18T00:39:42+00:00",
            "params": {"layers": ["direct","harmonized","llm_naive"], "max_studies": 5,
                       "max_total_records": 30, "max_spend": 0.25,
                       "model": "claude-sonnet-5", "text_system_chars": 16652,
                       "effort": "medium", "thinking": "disabled", "max_tokens": 16000,
                       "note": null},
            "counts": {"records": 30, "calls": 30, "spent": 0.1769,
                       "usage": {"input": 11660, "output": 10256,
                                 "cache_write": 6148, "cache_read": 178292}},
            "records": []
        }"#).unwrap();

        let export = Export::load(&path).unwrap();
        assert_eq!(export.format_version, 2);
        assert_eq!(export.params.thinking.as_deref(), Some("disabled"));
        // the field v2 never had reads as absent — which is exactly the state
        // that forced the batching run to be identified by its price
        assert_eq!(export.params.batch, None);
        assert_eq!(export.counts.usage.thinking, 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // -- per-layer parameters (v4) ------------------------------------------

    // A model that answers nothing, so `Export::new` can be exercised over a
    // real cascade without a network or a bill.
    struct Silent;
    impl crate::model::Model for Silent {
        fn price(&self, _u: Usage) -> f64 {
            0.0
        }
        fn complete(
            &self,
            _r: &crate::model::Request,
        ) -> Result<crate::model::Response, crate::model::ModelError> {
            Ok(crate::model::Response {
                text: String::new(),
                json: Some(serde_json::json!({"answers": []})),
                stop_reason: None,
                usage: Usage::default(),
            })
        }
    }

    fn settings_with(configs: Vec<(&'static str, crate::layer::ModelConfig)>) -> SchemaSettings {
        use crate::layer::Layer;
        SchemaSettings::new(
            configs
                .into_iter()
                .map(|(which, config)| match which {
                    "naive" => Layer::LLMNaive { model: Box::new(Silent), config },
                    _ => Layer::LLMPaper { model: Box::new(Silent), config },
                })
                .collect(),
        )
    }

    #[test]
    fn each_model_layer_records_its_own_settings() {
        // The reason for the bump. A run that puts a cheap model on layer 3 and
        // an expensive one on layer 4 is the whole point of per-layer configs,
        // and the v3 shape had one slot for both.
        use crate::model::{Effort, Thinking};
        let settings = settings_with(vec![
            (
                "naive",
                crate::layer::ModelConfig::new("claude-haiku-4-5", "short prompt")
                    .effort(None)
                    .thinking(Thinking::Disabled)
                    .batch(true),
            ),
            (
                "paper",
                crate::layer::ModelConfig::new("claude-sonnet-5", "a much longer prompt")
                    .effort(Some(Effort::High))
                    .max_tokens(8000),
            ),
        ]);
        let export = Export::new(vec![], &settings, &Budget::new(0.0), Params::default());

        assert_eq!(export.params.models.len(), 2);
        assert_eq!(export.params.models[0].layer, "llm_naive");
        assert_eq!(export.params.models[0].model, "claude-haiku-4-5");
        assert_eq!(export.params.models[0].effort, None);
        assert_eq!(export.params.models[0].thinking, "disabled");
        assert!(export.params.models[0].batch);
        assert_eq!(export.params.models[1].layer, "llm_paper");
        assert_eq!(export.params.models[1].model, "claude-sonnet-5");
        assert_eq!(export.params.models[1].effort.as_deref(), Some("high"));
        assert_eq!(export.params.models[1].max_tokens, 8000);
        assert!(!export.params.models[1].batch);
    }

    #[test]
    fn the_flat_summary_is_written_only_when_the_layers_agree() {
        // The v1..v3 fields are what every saved comparison reads. Filling them
        // from one layer while another ran different settings would make a
        // v3-shaped reader quietly wrong rather than merely blind.
        let same = || crate::layer::ModelConfig::new("claude-sonnet-5", "prompt");
        let agreeing =
            Export::new(vec![], &settings_with(vec![("naive", same()), ("paper", same())]),
                        &Budget::new(0.0), Params::default());
        assert_eq!(agreeing.params.model.as_deref(), Some("claude-sonnet-5"));
        assert_eq!(agreeing.params.text_system_chars, Some("prompt".len()));
        assert_eq!(agreeing.params.batch, Some(false));

        let split = Export::new(
            vec![],
            &settings_with(vec![
                ("naive", crate::layer::ModelConfig::new("claude-haiku-4-5", "prompt")),
                ("paper", crate::layer::ModelConfig::new("claude-sonnet-5", "a longer prompt")),
            ]),
            &Budget::new(0.0),
            Params::default(),
        );
        assert_eq!(split.params.model, None, "two models cannot be summarised as one");
        assert_eq!(split.params.text_system_chars, None);
        // and the per-layer rows still say exactly what ran
        assert_eq!(split.params.models.len(), 2);
    }

    #[test]
    fn a_free_run_records_no_models_at_all() {
        let export = Export::new(vec![], &SchemaSettings::default(), &Budget::new(0.0), Params::default());
        assert!(export.params.models.is_empty());
        assert_eq!(export.params.model, None);
        // and the flat fields stay out of the file rather than reading as
        // "effort was none", which is a different claim
        let json = serde_json::to_string(&export.params).unwrap();
        assert!(!json.contains("text_system_chars"), "{json}");
    }

    #[test]
    fn a_v3_export_still_loads_after_the_bump() {
        // The 15 saved runs are v2 and v3, including the pair that measured the
        // batch discount. `models` is new and defaults to empty, so they load
        // with their flat fields intact.
        let dir = std::env::temp_dir().join(format!("mp-v3-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("v3.json");
        std::fs::write(&path, r#"{
            "format_version": 3,
            "created": "2026-08-18T00:39:42+00:00",
            "params": {"layers": ["direct","harmonized","llm_naive"], "max_studies": 5,
                       "max_total_records": 30, "max_spend": 0.25,
                       "model": "claude-sonnet-5", "text_system_chars": 16652,
                       "effort": "medium", "thinking": "disabled", "max_tokens": 16000,
                       "batch": true, "note": null},
            "counts": {"records": 30, "calls": 30, "spent": 0.0973},
            "records": []
        }"#).unwrap();

        let export = Export::load(&path).unwrap();
        assert_eq!(export.format_version, 3);
        assert_eq!(export.params.batch, Some(true));
        assert_eq!(export.params.model.as_deref(), Some("claude-sonnet-5"));
        // the field v3 never had reads as empty, not as an error
        assert!(export.params.models.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_new_export_is_stamped_with_the_current_version() {
        let export = Export {
            format_version: FORMAT_VERSION,
            created: "2026-08-17T12:00:00+00:00".into(),
            params: Params::default(),
            counts: Counts::default(),
            evidence: None,
            records: vec![],
        };
        assert_eq!(export.format_version, 4);
        assert!(READABLE_VERSIONS.contains(&FORMAT_VERSION),
                "a build must be able to read what it writes");
    }

    #[test]
    fn the_refusal_names_both_what_it_reads_and_what_it_writes() {
        // So a version mismatch says which direction the problem is in.
        let dir = std::env::temp_dir().join(format!("mp-ver2-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("future.json");
        std::fs::write(&path, r#"{"format_version": 99, "created": "x",
            "params": {}, "counts": {}, "records": []}"#).unwrap();
        let error = Export::load(&path).unwrap_err().to_string();
        assert!(error.contains("99"), "{error}");
        assert!(error.contains("reads"), "{error}");
        assert!(error.contains("writes"), "{error}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_future_format_is_refused_rather_than_misread() {
        let dir = std::env::temp_dir().join(format!("mp-ver-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut export = Export {
            format_version: FORMAT_VERSION + 1,
            created: "2026-08-17T12:00:00+00:00".into(),
            params: Params::default(),
            counts: Counts::default(),
            evidence: None,
            records: vec![],
        };
        export.format_version = FORMAT_VERSION + 1;
        let path = export.save(&dir).unwrap();
        assert!(Export::load(&path).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
