// Replicates two orderings several times to get a variance estimate.
//
// Unbatched on purpose: sequential calls return in minutes rather than hours,
// and the spend guard is strictly tighter that way — `Budgeted::complete`
// checks the ledger before every individual call, so an overshoot is bounded by
// one call rather than by one whole batch submission. The cost of that is the
// 45% batch discount, so each run bills roughly twice its batched equivalent.
//
// Replicates are interleaved rather than grouped, so that any drift over the
// hour the sweep takes falls on both orderings equally instead of on whichever
// ran second.
//
// Without SPEND=1 this prints the estimates and sends nothing.

use std::fs;
use std::sync::Arc;

use metadata_project::prelude::*;

const CORPUS: &str = "../datasets/oa_corpus_full.json";
const RUNS: &str = "runs";

// Unbatched runs bill about $0.47 each, so the per-run ceiling is raised from
// the $0.50 used for the batched sweep: a limit that sits on top of the
// expected cost stops a run part-way and wastes it, which is the one failure
// this experiment cannot absorb.
const GLOBAL_LIMIT: f64 = 2.50;
const PER_RUN_LIMIT: f64 = 0.75;

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
    batch: false,
};

const PLAN: &[(&str, &str)] = &[
    ("D H N P", "rep2"),
    ("D H P N", "rep2"),
    ("D H N P", "rep3"),
    ("D H P N", "rep3"),
];

fn settings(spec: &str, global: &Arc<Budget>, per_run: &Arc<Budget>) -> SchemaSettings {
    SchemaSettings::new(CLAUDE.layers(spec, &[Arc::clone(per_run), Arc::clone(global)]))
        .keep_evidence(true)
        .max_studies(MAX_STUDIES)
        .max_total_records(MAX_RECORDS)
        .only_studies(STUDIES)
        .on_issue(|issue| eprintln!("!! {issue}"))
}

fn main() {
    let spend = std::env::var("SPEND").as_deref() == Ok("1");
    let corpus = Corpus::from_json(&fs::read_to_string(CORPUS).unwrap(), false).unwrap();
    let global = Arc::new(Budget::new(GLOBAL_LIMIT));

    if !spend {
        println!("DRY RUN — SPEND=1 not set, nothing will be sent.\n");
        let mut total = 0.0;
        for (spec, rep) in PLAN {
            let per_run = Arc::new(Budget::new(PER_RUN_LIMIT));
            let estimate =
                estimate::for_corpus(&corpus, &settings(spec, &global, &per_run), &per_run);
            total += estimate.cost();
            println!("[{spec} {rep}] estimate ${:.4} over {} calls", estimate.cost(), estimate.calls());
        }
        println!(
            "\nplanned total ${total:.4}, global ceiling ${GLOBAL_LIMIT:.2}, \
             per-run ${PER_RUN_LIMIT:.2}, batch={}",
            CLAUDE.batch
        );
        return;
    }

    println!(
        "global ceiling ${GLOBAL_LIMIT:.2}, per-run ${PER_RUN_LIMIT:.2}, batch={}\n",
        CLAUDE.batch
    );

    for (spec, rep) in PLAN {
        let per_run = Arc::new(Budget::new(PER_RUN_LIMIT));
        let estimate = estimate::for_corpus(&corpus, &settings(spec, &global, &per_run), &per_run);
        let spent = global.spent();
        if spent + estimate.cost() > GLOBAL_LIMIT {
            println!("[{spec} {rep}] SKIPPED — ${spent:.4} spent, estimate ${:.4}", estimate.cost());
            continue;
        }

        println!("\n=== [{spec} {rep}] estimate ${:.4}, global ${spent:.4} ===", estimate.cost());
        let live = settings(spec, &global, &per_run);
        let records = TargetSchema::from_corpus(corpus.clone(), &live);
        let export =
            Export::new(records, &live, &per_run, Params::default().note(format!("{spec} {rep}")));
        let path = export.save(RUNS).unwrap();
        println!("{export}");
        println!("{}", audit::verbatim(&export).to_string().lines().next().unwrap_or(""));
        println!(
            "[{spec} {rep}] billed ${:.4} | global ${:.4} of ${GLOBAL_LIMIT:.2} | {}",
            per_run.spent(), global.spent(), path.display()
        );
    }

    let ledger = global.ledger();
    println!("\n==== TOTAL ${:.4} over {} calls ====", ledger.spent, ledger.calls);
}
