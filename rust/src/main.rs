use std::fs;
use std::io::{self, Write};
use std::sync::Arc;

use metadata_project::audit;
use metadata_project::corpus::Corpus;
use metadata_project::estimate;
use metadata_project::export::{Export, Params};
#[allow(unused_imports)]
use metadata_project::layer::llm_naive::{TEXT_SYSTEM_FULL, TEXT_SYSTEM_SHORT, TEXT_SYSTEM_TARGETED};
use metadata_project::layer::llm_paper::PAPER_SYSTEM;
use metadata_project::layer::{Layer, ModelConfig};
use metadata_project::model::budget::{Budget, Budgeted};
use metadata_project::model::claude::{Claude, ModelId};
use metadata_project::model::retry::{RetryPolicy, Retrying};
use metadata_project::model::{Effort, Model, Thinking};
use metadata_project::target_schema::{SchemaSettings, TargetSchema};

const CORPUS: &str = "../datasets/oa_corpus_full.json";
const KEY_FILE: &str = "../anton_claude_api_key.txt";
const RUNS: &str = "runs";

// Three limits that fail independently. The budget stops the run once it has
// cost too much, which needs the cost to be known — and a study's cost is only
// known once it has been paid. The other two stop it before that, on volume.
const MAX_SPEND: f64 = 0.50;
const MAX_STUDIES: usize = 5;
const MAX_RECORDS: usize = 60;

// Which studies to run. Empty means "whatever the caps take from the front".
//
// The caps take a prefix, which is the right shape for a small cheap run and
// the wrong shape for "the same studies as last time". These five are the ones
// `datasets/test2/test_reconstructed4.json` covers, so a run over them is
// comparable against a Python run already on disk — the only way to tell an
// improved layer from an easy study, given coverage varies by 3.8 fields per
// record between studies on the model layers alone.
//
// Named by BioProject because that is what the Python run recorded; this corpus
// is keyed by the SRP and the selector matches either.
const STUDIES: &[&str] = &[
    "PRJNA474216",
    "PRJNA509126",
    "PRJNA509132",
    "PRJNA509134",
    "PRJNA509140",
];

// Stores what each record was shown, so `quoted` claims can be checked
// afterwards. Off by default in the library — a corpus run would store every
// sample's attribute bag twice — and on here because these runs are experiments.
const KEEP_EVIDENCE: bool = true;

// Whether layer 4 is in the cascade at all.
//
// A constant rather than an environment variable, so what a run does is legible
// from the file rather than from how it was invoked. `SPEND=1` stays a variable
// because it is the free/paid boundary and belongs on the command that spends;
// which paid layers run is a property of the experiment.
const READ_PAPERS: bool = true;

// -- what each paid layer sends ---------------------------------------------
//
// One block per layer, because the layers read different things: layer 3 asks
// two dozen short questions against an attribute bag, layer 4 asks a handful
// against 30,000 characters of prose. The cheapest model that answers one well
// is not automatically the right one for the other, and every one of these is a
// variable a run might be comparing. Setting them apart is the point; setting
// them identical is a choice this still lets you make.
//
// `PROMPT` — the most consequential variable in the cascade. FULL beats SHORT on
// coverage (19.2 fields against 11.2) and loses on verbatim rate (65% against
// 92%); TARGETED was measured and lost on both, so it stays unused.
//
// `EFFORT` — None omits the parameter, which is not the same as asking for low.
// Dropped automatically on models that reject it (Haiku 4.5 does).
//
// `THINKING` — Disabled is sent explicitly to the models that would otherwise
// think anyway; on the ones that predate the field it is omitted, so this is
// safe to set on any model. Measured: thinking loses on both models tried,
// costing 8× on Haiku for no coverage gain and halving coverage on Sonnet.
//
// `MAX_TOKENS` — the hard ceiling on one response, thinking included. Effort is
// soft guidance; this is the strict limit.
//
// `BATCH` — 45% cheaper, minutes to hours of latency. Measured at 45 rather than
// the advertised 50 because the fan-out pays for a second cache write.
// Providers with no batch endpoint fall back to sequential calls unchanged.

const TEXT_MODEL: ModelId = ModelId::Sonnet5;
const TEXT_PROMPT: &str = TEXT_SYSTEM_FULL;
const TEXT_EFFORT: Option<Effort> = Some(Effort::Medium);
const TEXT_THINKING: Thinking = Thinking::Disabled;
const TEXT_MAX_TOKENS: u32 = 16_000;
const TEXT_BATCH: bool = true;

const PAPER_MODEL: ModelId = ModelId::Sonnet5;
const PAPER_PROMPT: &str = PAPER_SYSTEM;
const PAPER_EFFORT: Option<Effort> = Some(Effort::Medium);
const PAPER_THINKING: Thinking = Thinking::Disabled;
const PAPER_MAX_TOKENS: u32 = 16_000;
// One call per paper against a whole study, so there is far less to batch and
// far less latency to trade away for the discount.
const PAPER_BATCH: bool = true;

fn main() {
    let corpus = Corpus::from_json(&fs::read_to_string(CORPUS).unwrap(), false).unwrap();
    let budget = Arc::new(Budget::new(MAX_SPEND));
    let mut settings = SchemaSettings::new(layers(&budget))
        .keep_evidence(KEEP_EVIDENCE)
        .max_studies(MAX_STUDIES)
        .max_total_records(MAX_RECORDS)
        .on_issue(|issue| eprintln!("!! {issue}"));
    if !STUDIES.is_empty() {
        settings = settings.only_studies(STUDIES);
    }

    if !confirmation_prompt(&corpus, &settings, &budget) {
        return;
    }

    let records = TargetSchema::from_corpus(corpus, &settings);

    println!("{:#?}", records.first());

    let export = Export::new(records, &settings, &budget, Params::default());
    println!("{export}\n{}saved {}", audit::verbatim(&export), export.save(RUNS).unwrap().display());
}

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
fn confirmation_prompt(corpus: &Corpus, settings: &SchemaSettings, budget: &Budget) -> bool {
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

// The paid layers only exist when SPEND=1, so a bare `cargo run` is free by
// construction rather than by a flag that could be read the wrong way.
fn layers(budget: &Arc<Budget>) -> Vec<Layer> {
    let mut layers = vec![Layer::Direct, Layer::Harmonized];
    if std::env::var("SPEND").as_deref() != Ok("1") {
        return layers;
    }

    layers.push(Layer::LLMNaive {
        model: paid(TEXT_MODEL, budget),
        config: ModelConfig::new(TEXT_MODEL.as_str(), TEXT_PROMPT)
            .effort(TEXT_EFFORT)
            .thinking(TEXT_THINKING)
            .max_tokens(TEXT_MAX_TOKENS)
            .batch(TEXT_BATCH),
    });

    if READ_PAPERS {
        layers.push(Layer::LLMPaper {
            model: paid(PAPER_MODEL, budget),
            config: ModelConfig::new(PAPER_MODEL.as_str(), PAPER_PROMPT)
                .effort(PAPER_EFFORT)
                .thinking(PAPER_THINKING)
                .max_tokens(PAPER_MAX_TOKENS)
                .batch(PAPER_BATCH),
        });
    }
    layers
}

// Budget outermost, so retries beneath it count as one billed call. Each layer
// gets its own client but they share the one ledger, which is what makes
// MAX_SPEND a limit on the run rather than on each layer separately.
fn paid(model: ModelId, budget: &Arc<Budget>) -> Box<dyn Model> {
    let claude = Claude::from_key_file(KEY_FILE, model).unwrap();
    Box::new(Budgeted::new(
        Retrying::new(claude, RetryPolicy::default()),
        Arc::clone(budget),
    ))
}