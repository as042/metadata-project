use std::fs;

use metadata_project::corpus::Corpus;

fn main() {
    let contents = fs::read_to_string("../datasets/oa_corpus_full.json").unwrap();
    let corpus = Corpus::from_json(&contents, true).unwrap();
}