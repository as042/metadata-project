// Throwaway: prices every layer ordering without sending anything.
//
// `estimate::for_corpus` walks the layer list in the order given, runs the free
// layers for real and only *plans* the paid ones, so a sweep over permutations
// costs nothing. Constructing a Claude client reads the key file and validates
// the prefix; no request is made anywhere in this binary.

use std::fs;

use metadata_project::prelude::*;

const CORPUS: &str = "../datasets/oa_corpus_full.json";

// Mirrors main.rs so the numbers are comparable with the measured baseline run.
const MAX_SPEND: f64 = 0.50;
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

fn settings(spec: &str) -> SchemaSettings {
    SchemaSettings::new(CLAUDE.layers(spec, &[]))
        .max_studies(MAX_STUDIES)
        .max_total_records(MAX_RECORDS)
        .only_studies(STUDIES)
        .on_issue(|issue| eprintln!("!! {issue}"))
}

fn main() {
    let corpus = Corpus::from_json(&fs::read_to_string(CORPUS).unwrap(), false).unwrap();
    let budget = Budget::new(MAX_SPEND);

    println!(
        "{:<12} {:<26} {:>7} {:>7} {:>7} {:>10}",
        "order", "note", "records", "N call", "P call", "cost"
    );
    for (spec, note) in SPECS {
        let estimate = estimate::for_corpus(&corpus, &settings(spec), &budget);
        let calls = |name: &str| -> usize {
            estimate.layers.iter().filter(|l| l.layer == name).map(|l| l.calls).sum()
        };
        println!(
            "{:<12} {:<26} {:>7} {:>7} {:>7} {:>10}",
            spec,
            note,
            estimate.records,
            calls("llm_naive"),
            calls("llm_paper"),
            format!("${:.4}", estimate.cost())
        );
    }

    println!("\n-- full breakdown per ordering --");
    for (spec, note) in SPECS {
        println!("\n[{spec}] {note}");
        print!("{}", estimate::for_corpus(&corpus, &settings(spec), &budget));
    }
}
