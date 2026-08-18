use crate::model::{Effort, Model, Thinking};
use crate::project::Project;
use crate::target_schema::TargetSchema;

pub mod direct;
pub mod fields;
pub mod harmonized;
pub mod llm_naive;
pub mod llm_paper;

// One stage of the reconstruction cascade.
//
// A layer only ever fills fields no earlier layer settled, which is what makes
// the order in `SchemaSettings` meaningful rather than merely cosmetic: run the
// cheap deterministic layers first and a later one pays only for what they
// could not answer. Nothing enforces that ordering, deliberately — the point of
// the list is that a colleague can reorder it, including into arrangements that
// cost more, without editing this enum.
//
// The variants divide on cost, not on technique. Direct and Harmonized are free
// and offline; both LLM variants bill per record against whichever `Model` they
// are handed.
// Everything a model layer needs beyond the model itself.
//
// Per layer rather than per run, because the layers do different work and the
// right settings differ: layer 3 asks 24 short questions per sample against an
// attribute bag, layer 4 asks a handful against 30,000 characters of prose.
// Sharing one prompt, one effort and one batch flag between them would force
// every future optimisation to move both at once.
pub struct ModelConfig {
    // Named by the caller, because a layer cannot ask a `dyn Model` what it is
    // — the same property that lets a local model drop in unchanged.
    pub label: String,
    pub prompt: &'static str,
    pub effort: Option<Effort>,
    pub thinking: Thinking,
    pub max_tokens: u32,
    // Half price, minutes to hours of latency. Per layer because layer 3 plans
    // thousands of calls and layer 4 plans one per study, so the trade differs.
    pub batch: bool,
}

impl ModelConfig {
    pub fn new(label: impl Into<String>, prompt: &'static str) -> Self {
        Self {
            label: label.into(),
            prompt,
            effort: Some(Effort::Medium),
            thinking: Thinking::Adaptive,
            max_tokens: 16_000,
            batch: false,
        }
    }

    pub fn effort(mut self, effort: Option<Effort>) -> Self {
        self.effort = effort;
        self
    }

    pub fn thinking(mut self, thinking: Thinking) -> Self {
        self.thinking = thinking;
        self
    }

    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    pub fn batch(mut self, batch: bool) -> Self {
        self.batch = batch;
        self
    }
}

pub enum Layer {
    // Fields the archive states outright. The only layer that creates records.
    Direct,
    // A synonym table mapping submitter attribute keys onto schema fields. Free
    // and offline, but unlike Direct the *key* mapping is ours, which is why it
    // gets its own provenance rather than folding into Direct.
    Harmonized,
    // Reads the archive's own text and attribute bags.
    LLMNaive {
        model: Box<dyn Model>,
        config: ModelConfig,
    },
    // Reads the linked publication. Only the studies with retrievable text can
    // be asked at all, so this layer is a no-op for the rest.
    LLMPaper {
        model: Box<dyn Model>,
        config: ModelConfig,
    },
}

impl Layer {
    // Runs one layer over one project.
    //
    // `schemas` is in-out because layers are not uniform: Direct appends the
    // records, and every other layer fills fields on records already there.
    // A layer scheduled before Direct therefore sees an empty slice and does
    // nothing — it is not an error, just wasted work.
    //
    // Takes `&self` so one SchemaSettings can drive every project in a corpus;
    // consuming the layer would make the settings single-use.
    #[inline]
    pub(crate) fn process(
        &self,
        project: &Project,
        schemas: &mut Vec<TargetSchema>,
        settings: &crate::target_schema::SchemaSettings,
    ) {
        match self {
            Layer::Direct => direct::process(project, schemas),
            Layer::Harmonized => harmonized::process(project, schemas),
            Layer::LLMNaive { model, config } => {
                llm_naive::process(project, schemas, model.as_ref(), config, settings)
            }
            Layer::LLMPaper { model, config } => {
                llm_paper::process(project, schemas, model.as_ref(), config, settings)
            }
        }
    }

    // A stable name for a saved run's parameters, so a comparison six weeks
    // later can say what was in the cascade.
    #[inline]
    pub fn name(&self) -> &'static str {
        match self {
            Layer::Direct => "direct",
            Layer::Harmonized => "harmonized",
            Layer::LLMNaive { .. } => "llm_naive",
            Layer::LLMPaper { .. } => "llm_paper",
        }
    }

    // The model settings this layer runs with, when it has any.
    #[inline]
    pub fn config(&self) -> Option<&ModelConfig> {
        match self {
            Layer::LLMNaive { config, .. } | Layer::LLMPaper { config, .. } => Some(config),
            _ => None,
        }
    }

    // The model this layer would ask, when it asks one. Exposed so a cost
    // estimate can price a layer without naming a provider, the same way the
    // layer itself does.
    #[inline]
    pub fn model(&self) -> Option<&dyn Model> {
        match self {
            Layer::LLMNaive { model, .. } | Layer::LLMPaper { model, .. } => Some(model.as_ref()),
            _ => None,
        }
    }

    // What this layer would spend on one project, without spending it.
    //
    // `None` for the free layers, which is what tells the estimator to run them
    // for real instead — their output is what the paid layers are not asked
    // about. Dispatch lives here beside `process` so the plan being priced and
    // the plan being executed are the same function call.
    #[inline]
    pub(crate) fn estimate(
        &self,
        project: &Project,
        schemas: &[TargetSchema],
    ) -> Option<crate::estimate::Estimate> {
        let (model, config, jobs) = match self {
            Layer::LLMNaive { model, config } => {
                (model, config, llm_naive::plan(project, schemas))
            }
            Layer::LLMPaper { model, config } => {
                (model, config, llm_paper::plan(project, schemas))
            }
            _ => return None,
        };
        Some(crate::estimate::of_jobs(self.name(), &jobs, model.as_ref(), config))
    }

    // Whether running this layer can bill an API. Not currently consulted by
    // anything; it exists so a spend guard has one place to ask rather than
    // matching on the enum at each call site.
    #[inline]
    pub fn is_paid(&self) -> bool {
        matches!(self, Layer::LLMNaive { .. } | Layer::LLMPaper { .. })
    }
}