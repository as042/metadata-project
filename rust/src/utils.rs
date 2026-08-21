// Wiring a run: the pieces that assemble a cascade and put the spend decision
// in front of a person. Kept out of the library's core because none of it is
// reconstruction — it is the glue a binary needs, and a caller driving the
// pipeline from its own code may want none of it.

use std::io::{self, Write};
use std::sync::Arc;

use crate::corpus::Corpus;
use crate::estimate;
use crate::layer::{Layer, ModelConfig};
use crate::model::budget::{Budget, Budgeted};
use crate::model::claude::{Claude, ModelId};
use crate::model::retry::{RetryPolicy, Retrying};
use crate::model::{Effort, Model, Thinking};
use crate::target_schema::SchemaSettings;

// Prices the configured run before anything is sent, and makes the spend an
// explicit decision.
//
// The preventative half of the spend guard. `max_spend` is the in-the-moment
// one: it learns what a call cost only after paying for it, so the smallest
// mistake it can catch has already cost a call — and it stops a run *part-way*,
// which leaves a half-finished paid layer nobody can compare against anything.
// This one runs over the same plan the run will execute, before the first call.
//
// Free runs are not interrupted: there is nothing to decide, and a prompt that
// appears when the answer cannot matter is a prompt that gets typed through.
//
// Anything other than "y" declines, including end-of-input — so a piped or
// unattended `cargo run` cannot spend by default.
pub fn confirmation_prompt(corpus: &Corpus, settings: &SchemaSettings, budget: &Budget) -> bool {
    let estimate = estimate::for_corpus(corpus, settings, budget);
    if estimate.calls() == 0 {
        return true;
    }

    println!("\n{estimate}");
    print!("proceed with an estimated ${:.4}? [y/N] ", estimate.cost());
    io::stdout().flush().unwrap();

    let mut answer = String::new();
    if io::stdin().read_line(&mut answer).unwrap_or(0) == 0 {
        println!("no answer — nothing sent");
        return false;
    }
    let yes = answer.trim().eq_ignore_ascii_case("y");
    if !yes {
        println!("declined — nothing sent");
    }
    yes
}

// Direct -> Harmonized -> LLMNaive -> LLMPaper, backed by Claude.
//
// The arrangement the pipeline has always run, and the baseline every ordering
// experiment is measured against.
//
// One named arrangement rather than a general builder: the cascade order is a
// measured claim and not a preference, so a function that hands back *this*
// order can be pointed at. Any other arrangement is a `Vec<Layer>` literal,
// which is what `SchemaSettings::new` already takes.
//
// The models arrive as configs rather than being chosen here, so what a run
// asks and of whom stays legible in the caller's own file.
//
// The paid layers only exist when SPEND=1, so a bare `cargo run` is free by
// construction rather than by a flag that could be read the wrong way.
pub fn dhnp_claude_layers(
    key_file: &str,
    budget: &Arc<Budget>,
    text: (ModelId, ModelConfig),
    paper: Option<(ModelId, ModelConfig)>,
) -> Vec<Layer> {
    let mut layers = vec![Layer::Direct, Layer::Harmonized];
    if !spending() {
        return layers;
    }
    layers.push(naive_layer(key_file, budget, text));
    if let Some(paper) = paper {
        layers.push(paper_layer(key_file, budget, paper));
    }
    layers
}

// Direct -> Harmonized -> LLMPaper -> LLMNaive, backed by Claude.
//
// The same layers, with the publication asked first. Measured over seven runs
// it fills +0.76 real values per record against the arrangement above (Welch
// p = 0.014, non-overlapping ranges) and costs 9-16% less, because a study-wide
// answer settles fields the per-sample layer would otherwise be asked about one
// sample at a time — and that layer is the one billing per sample.
//
// Deliberately the same signature as `dhnp_claude_layers`, so a run switches
// between them by changing the name and nothing else.
//
// With `paper` absent the two are identical: there is no paper layer to move,
// and both reduce to Direct -> Harmonized -> LLMNaive.
pub fn dhpn_claude_layers(
    key_file: &str,
    budget: &Arc<Budget>,
    text: (ModelId, ModelConfig),
    paper: Option<(ModelId, ModelConfig)>,
) -> Vec<Layer> {
    let mut layers = vec![Layer::Direct, Layer::Harmonized];
    if !spending() {
        return layers;
    }
    // Built before the paper layer is pushed, so the two functions differ in
    // ordering alone and not in what either layer is handed.
    let naive = naive_layer(key_file, budget, text);
    if let Some(paper) = paper {
        layers.push(paper_layer(key_file, budget, paper));
    }
    layers.push(naive);
    layers
}

// The free/paid boundary, in one place so the two cascades cannot disagree
// about it.
fn spending() -> bool {
    std::env::var("SPEND").as_deref() == Ok("1")
}

fn naive_layer(key_file: &str, budget: &Arc<Budget>, (model, config): (ModelId, ModelConfig)) -> Layer {
    Layer::LLMNaive { model: paid(key_file, model, budget), config }
}

fn paper_layer(key_file: &str, budget: &Arc<Budget>, (model, config): (ModelId, ModelConfig)) -> Layer {
    Layer::LLMPaper { model: paid(key_file, model, budget), config }
}

// Budget outermost, so retries beneath it count as one billed call. Each layer
// gets its own client but they share the one ledger, which is what makes the
// caller's `max_spend` a limit on the run rather than on each layer separately.
fn paid(key_file: &str, model: ModelId, budget: &Arc<Budget>) -> Box<dyn Model> {
    let claude = Claude::from_key_file(key_file, model).unwrap();
    Box::new(Budgeted::new(
        Retrying::new(claude, RetryPolicy::default()),
        Arc::clone(budget),
    ))
}

// One model and one set of send settings applied to every layer in a run.
//
// Deliberately less expressive than `ModelConfig`, which is per layer on
// purpose. An experiment comparing two orderings, or one ordering twice, wants
// both model layers held identical — and a type that cannot express otherwise
// makes that constraint visible rather than merely conventional. A run that
// needs the layers to differ builds its `Vec<Layer>` directly, as `main` does.
pub struct UniformClaude {
    pub key_file: &'static str,
    pub model: ModelId,
    pub text_prompt: &'static str,
    pub paper_prompt: &'static str,
    pub effort: Option<Effort>,
    pub thinking: Thinking,
    pub max_tokens: u32,
    pub batch: bool,
}

impl UniformClaude {
    // Wrapped innermost-first: `budgets[0]` is nearest the transport and the
    // last is outermost, so a caller metering a single run inside a whole sweep
    // passes `[per_run, global]` and both ledgers see every call.
    //
    // Retrying sits beneath every budget, so a retried call is one billed call
    // to all of them.
    pub fn client(&self, budgets: &[Arc<Budget>]) -> Box<dyn Model> {
        let claude = Claude::from_key_file(self.key_file, self.model).unwrap();
        let mut model: Box<dyn Model> = Box::new(Retrying::new(claude, RetryPolicy::default()));
        for budget in budgets {
            model = Box::new(Budgeted::new(model, Arc::clone(budget)));
        }
        model
    }

    pub fn config(&self, prompt: &'static str) -> ModelConfig {
        ModelConfig::new(self.model.as_str(), prompt)
            .effort(self.effort)
            .thinking(self.thinking)
            .max_tokens(self.max_tokens)
            .batch(self.batch)
    }

    // A cascade from a whitespace-separated spec: "D H N P".
    //
    // No SPEND gate here, unlike `dhnp_claude_layers`. The estimating binaries
    // need the paid layers to *exist* in order to price them, and pricing sends
    // nothing — so the gate belongs at the point a run would actually spend,
    // not at the point a layer is constructed.
    //
    // Each occurrence builds its own client and config, so a spec may repeat a
    // layer and get two independent ones rather than a shared handle.
    pub fn layers(&self, spec: &str, budgets: &[Arc<Budget>]) -> Vec<Layer> {
        spec.split_whitespace()
            .map(|token| match token {
                "D" => Layer::Direct,
                "H" => Layer::Harmonized,
                "N" => Layer::LLMNaive {
                    model: self.client(budgets),
                    config: self.config(self.text_prompt),
                },
                "P" => Layer::LLMPaper {
                    model: self.client(budgets),
                    config: self.config(self.paper_prompt),
                },
                other => panic!("unknown layer token {other:?} in spec {spec:?}"),
            })
            .collect()
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::layer::llm_naive::TEXT_SYSTEM_FULL;
    use crate::layer::llm_paper::PAPER_SYSTEM;

    // A key file is never read: the free tokens build no client, and every
    // assertion below is about arrangement rather than transport.
    const CLAUDE: UniformClaude = UniformClaude {
        key_file: "/nonexistent-on-purpose",
        model: ModelId::Sonnet5,
        text_prompt: TEXT_SYSTEM_FULL,
        paper_prompt: PAPER_SYSTEM,
        effort: Some(Effort::Medium),
        thinking: Thinking::Disabled,
        max_tokens: 16_000,
        batch: true,
    };

    #[test]
    fn config_carries_every_send_setting() {
        let config = CLAUDE.config(PAPER_SYSTEM);
        assert_eq!(config.label, ModelId::Sonnet5.as_str());
        assert_eq!(config.prompt, PAPER_SYSTEM);
        assert_eq!(config.effort, Some(Effort::Medium));
        assert_eq!(config.thinking, Thinking::Disabled);
        assert_eq!(config.max_tokens, 16_000);
        assert!(config.batch);
    }

    #[test]
    fn a_spec_builds_its_layers_in_the_order_written() {
        let layers = CLAUDE.layers("H D", &[]);
        assert_eq!(layers.len(), 2);
        assert!(matches!(layers[0], Layer::Harmonized));
        assert!(matches!(layers[1], Layer::Direct));
    }

    #[test]
    fn a_spec_may_repeat_a_layer() {
        // Not a mistake to reject: `layer.rs` treats the list as a description
        // of the pipeline rather than a thing to police.
        assert_eq!(CLAUDE.layers("D D H", &[]).len(), 3);
    }

    #[test]
    #[should_panic(expected = "unknown layer token")]
    fn an_unknown_spec_token_is_refused() {
        // Loudly, and at construction: a silently dropped token would build a
        // shorter cascade than the caller wrote and still run.
        CLAUDE.layers("D H X", &[]);
    }

    #[test]
    fn neither_cascade_builds_a_paid_layer_without_spend() {
        // The key file above does not exist, so reaching the paid path at all
        // would panic rather than quietly pass.
        if spending() {
            return;
        }
        let budget = Arc::new(Budget::unlimited());
        for layers in [
            dhnp_claude_layers("/nonexistent-on-purpose", &budget, text(), Some(paper())),
            dhpn_claude_layers("/nonexistent-on-purpose", &budget, text(), Some(paper())),
        ] {
            assert_eq!(layers.len(), 2);
            assert!(matches!(layers[0], Layer::Direct));
            assert!(matches!(layers[1], Layer::Harmonized));
        }
    }

    fn text() -> (ModelId, ModelConfig) {
        (ModelId::Sonnet5, CLAUDE.config(TEXT_SYSTEM_FULL))
    }

    fn paper() -> (ModelId, ModelConfig) {
        (ModelId::Sonnet5, CLAUDE.config(PAPER_SYSTEM))
    }
}
