// Runs every layer ordering for real, under a hard aggregate ceiling.
//
// Three independent limits, because the per-run budget alone does not bound a
// batched run: `Budgeted::complete_many` only calls `Budget::check`, which
// refuses when the ledger is already full rather than reserving what the batch
// will cost. The batch that crosses the line is therefore unbounded by the
// per-run limit — bounded in practice only by one project's share of a layer.
//
//   1. SPEND=1, or this binary does nothing.
//   2. A GLOBAL budget shared by every client in every ordering. This is the
//      real ceiling; it is set below the authorised figure so that a single
//      overshooting batch still lands under it.
//   3. A per-ordering budget, so one ordering cannot consume the whole pot.
//   4. A pre-flight estimate gate: an ordering is skipped unless the global
//      ledger has room for its full estimate, which is measured to err high.

use std::fs;
use std::sync::Arc;

use metadata_project::prelude::*;

const CORPUS: &str = "../datasets/oa_corpus_full.json";
const RUNS: &str = "runs";

// Authorised: $4.50. The ceiling is set to $4.00 so that the worst case — the
// global ledger full, and one more batch already submitted — still lands under
// the authorised figure with room to spare. One batch is one project's jobs for
// one layer, estimated at $0.045 (naive) and $0.022 (paper).
const GLOBAL_LIMIT: f64 = 4.00;
const PER_RUN_LIMIT: f64 = 0.50;

const MAX_STUDIES: usize = 5;
const MAX_RECORDS: usize = 60;
const STUDIES: &[&str] = &[
    "PRJNA474216",
    "PRJNA509126",
    "PRJNA509132",
    "PRJNA509134",
    "PRJNA509140",
];

const CLAUDE: UniformClaude = UniformClaude {
    key_file: "../anton_claude_api_key.txt",
    model: ModelId::Sonnet5,
    text_prompt: TEXT_SYSTEM_FULL,
    paper_prompt: PAPER_SYSTEM,
    effort: Some(Effort::Medium),
    thinking: Thinking::Disabled,
    max_tokens: 16_000,
    batch: true,
};

const SPECS: &[(&str, &str)] = &[
    ("D H N P", "baseline"),
    ("D H P N", "paper before naive"),
    ("D N H P", "harmonized after naive"),
    ("D N P H", "harmonized last"),
    ("D P H N", "paper first"),
    ("D P N H", "fully inverted"),
    ("H D N P", "free layer before Direct"),
    ("N D H P", "PAID layer before Direct"),
    ("D H N P N", "naive repeated"),
];

// Rebuilt per use rather than reused: a `Vec<Layer>` owns its clients, so the
// dry pass and the live run cannot share one.
fn settings(spec: &str, global: &Arc<Budget>, per_run: &Arc<Budget>) -> SchemaSettings {
    SchemaSettings::new(CLAUDE.layers(spec, &[Arc::clone(per_run), Arc::clone(global)]))
        .keep_evidence(true)
        .max_studies(MAX_STUDIES)
        .max_total_records(MAX_RECORDS)
        .only_studies(STUDIES)
        .on_issue(|issue| eprintln!("!! {issue}"))
}

fn main() {
    if std::env::var("SPEND").as_deref() != Ok("1") {
        println!("SPEND=1 not set — nothing to do, nothing sent.");
        return;
    }

    let corpus = Corpus::from_json(&fs::read_to_string(CORPUS).unwrap(), false).unwrap();
    let global = Arc::new(Budget::new(GLOBAL_LIMIT));

    println!("global ceiling ${GLOBAL_LIMIT:.2}, per-ordering ${PER_RUN_LIMIT:.2}\n");

    for (spec, note) in SPECS {
        let per_run = Arc::new(Budget::new(PER_RUN_LIMIT));

        // Priced against this ordering's own plan before anything is sent.
        let estimate = estimate::for_corpus(&corpus, &settings(spec, &global, &per_run), &per_run);
        let spent = global.spent();
        if spent + estimate.cost() > GLOBAL_LIMIT {
            println!(
                "[{spec}] {note}: SKIPPED — ${spent:.4} spent, estimate ${:.4}, \
                 ceiling ${GLOBAL_LIMIT:.2}",
                estimate.cost()
            );
            continue;
        }

        println!(
            "\n=== [{spec}] {note} — estimate ${:.4}, global spent ${spent:.4} ===",
            estimate.cost()
        );

        let live = settings(spec, &global, &per_run);
        let records = TargetSchema::from_corpus(corpus.clone(), &live);
        let export = Export::new(records, &live, &per_run, Params::default().note(*spec));
        let path = export.save(RUNS).unwrap();

        println!("{export}");
        println!("{}", audit::verbatim(&export));
        println!(
            "[{spec}] billed ${:.4} | global ${:.4} of ${GLOBAL_LIMIT:.2} | {}",
            per_run.spent(),
            global.spent(),
            path.display()
        );
    }

    let ledger = global.ledger();
    println!(
        "\n==== TOTAL ${:.4} over {} calls (ceiling ${GLOBAL_LIMIT:.2}) ====",
        ledger.spent, ledger.calls
    );
}
