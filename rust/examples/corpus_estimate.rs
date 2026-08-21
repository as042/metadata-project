// Prices a whole-corpus reconstruction without sending anything.
//
// No caps and no study selection, so `plan` runs over all 346 studies and the
// figure is the real workload rather than a per-record rate extrapolated from a
// small run. That distinction matters here: the five-study comparison set
// averages ~10 samples per study and the corpus averages ~256, and the naive
// layer bills per sample.

use std::fs;

use metadata_project::prelude::*;

const CORPUS: &str = "../datasets/oa_corpus_full.json";

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

fn main() {
    let corpus = Corpus::from_json(&fs::read_to_string(CORPUS).unwrap(), false).unwrap();
    println!(
        "corpus: {} studies, {} records, {} samples, {} papers",
        corpus.counts.studies, corpus.counts.records, corpus.counts.samples, corpus.counts.papers
    );

    for spec in ["D H N P", "D H P N"] {
        let settings = SchemaSettings::new(CLAUDE.layers(spec, &[]));
        println!("\n[{spec}] whole corpus, no caps");
        print!("{}", estimate::for_corpus(&corpus, &settings, &Budget::unlimited()));
    }
}
