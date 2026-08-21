// Configuration for one run, and nothing else.
//
// Every knob a run varies lives here as a constant rather than an argument or
// an environment variable, so what a run did is legible from this file rather
// than from how it was invoked. The wiring those constants feed lives in
// `utils`.

use std::fs;
use std::sync::Arc;

use metadata_project::prelude::*;

const CORPUS: &str = "../datasets/oa_corpus_full.json";
const KEY_FILE: &str = "../anton_claude_api_key.txt";
const RUNS: &str = "runs";

// Three limits that fail independently. The budget stops the run once it has
// cost too much, which needs the cost to be known — and a study's cost is only
// known once it has been paid. The other two stop it before that, on volume.
const MAX_SPEND: f64 = 1.00;
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
const TEXT_BATCH: bool = false;

const PAPER_MODEL: ModelId = ModelId::Sonnet5;
const PAPER_PROMPT: &str = PAPER_SYSTEM;
const PAPER_EFFORT: Option<Effort> = Some(Effort::Medium);
const PAPER_THINKING: Thinking = Thinking::Disabled;
const PAPER_MAX_TOKENS: u32 = 16_000;
// One call per paper against a whole study, so there is far less to batch and
// far less latency to trade away for the discount.
const PAPER_BATCH: bool = false;

fn main() {
    let corpus = Corpus::from_json(&fs::read_to_string(CORPUS).unwrap(), false).unwrap();
    let budget = Arc::new(Budget::new(MAX_SPEND));

    let text = (
        TEXT_MODEL,
        ModelConfig::new(TEXT_MODEL.as_str(), TEXT_PROMPT)
            .effort(TEXT_EFFORT)
            .thinking(TEXT_THINKING)
            .max_tokens(TEXT_MAX_TOKENS)
            .batch(TEXT_BATCH),
    );
    let paper = READ_PAPERS.then(|| {
        (
            PAPER_MODEL,
            ModelConfig::new(PAPER_MODEL.as_str(), PAPER_PROMPT)
                .effort(PAPER_EFFORT)
                .thinking(PAPER_THINKING)
                .max_tokens(PAPER_MAX_TOKENS)
                .batch(PAPER_BATCH),
        )
    });

    let mut settings = SchemaSettings::new(dhpn_claude_layers(KEY_FILE, &budget, text, paper))
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