use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::dto::CorpusDto;
use crate::project::{Paper, Project, ZonedDate};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Corpus {
    pub format_version: u32,
    pub created: ZonedDate,
    pub params: CorpusParams,
    pub counts: CorpusCounts,
    pub projects: Vec<Project>,
}

// The version this build understands. Refusing an unknown version beats
// mis-parsing one: the corpus format has already changed once (v1 carried no
// submission, BioProject record, run statistics or per-publication papers).
pub const SUPPORTED_FORMAT_VERSION: u32 = 2;

#[derive(Debug)]
pub enum CorpusError {
    Json(serde_json::Error),
    UnsupportedVersion { found: u32, supported: u32 },
}

impl std::fmt::Display for CorpusError {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CorpusError::Json(e) => write!(f, "corpus JSON is malformed: {e}"),
            CorpusError::UnsupportedVersion { found, supported } => write!(
                f,
                "corpus is format_version {found}; this build understands \
                 {supported}. Rebuild it, or read it with the matching version."
            ),
        }
    }
}

impl std::error::Error for CorpusError {}

impl From<serde_json::Error> for CorpusError {
    #[inline]
    fn from(e: serde_json::Error) -> Self {
        CorpusError::Json(e)
    }
}

impl Corpus {
    // Two passes by necessity: publications reference papers by id, so the map
    // has to be built before any study can resolve its own. It is local to this
    // function — once every publication owns its paper the index is redundant,
    // and keeping it would duplicate ~353 texts (roughly 10 MB) for nothing.
    #[inline]
    pub fn from_json(json: &str, print: bool) -> Result<Self, CorpusError> {
        let dto: CorpusDto = serde_json::from_str(json)?;
        if dto.format_version != SUPPORTED_FORMAT_VERSION {
            return Err(CorpusError::UnsupportedVersion {
                found: dto.format_version,
                supported: SUPPORTED_FORMAT_VERSION,
            });
        }

        let papers: BTreeMap<String, Paper> = dto
            .papers
            .into_iter()
            .map(|(id, p)| (id, Paper::from(p)))
            .collect();

        let projects = dto
            .studies
            .into_iter()
            .map(|s| s.into_project(&papers, dto.format_version, &dto.created))
            .collect();

        let corpus = Corpus {
            format_version: dto.format_version,
            created: ZonedDate { raw: dto.created },
            params: dto.params,
            counts: dto.counts,
            projects,
        };

        if print {
            println!("format_version {}", corpus.format_version);
            println!("projects       {}", corpus.projects.len());
            println!("with text      {}", corpus.with_paper_text().count());

            let owned: usize = corpus.projects.iter().map(|p| papers_of(p).len()).sum();
            let multi: Vec<_> = corpus
                .projects
                .iter()
                .filter(|p| papers_of(p).len() > 1)
                .collect();
            println!("papers owned by publications {owned}");
            println!("studies with >1 paper        {}", multi.len());

            if let Some(p) = multi.first() {
                println!("\n{:?}", p.accession);
                for pub_ in &p.publications {
                    match &pub_.paper {
                        Some(paper) => println!(
                            "   {} {:?} -> {} chars, truncated={}, oa_status={:?}",
                            pub_.id,
                            pub_.accessibility_type,
                            paper.char_count,
                            paper.truncated,
                            paper.oa_status
                        ),
                        None => println!("   {} {:?} -> no paper", pub_.id, pub_.accessibility_type),
                    }
                }
            }

            let s = &corpus.projects[0];
            let sample = s.samples.values().next().unwrap();
            println!("\n{:?} sample {:?}", s.accession, sample.accession);
            println!("   sra attrs       {}", sample.attributes.len());
            println!("   biosample attrs {}", sample.biosample_attributes.len());
            println!("   biosample pkg   {:?}",
                    sample.biosample.as_ref().and_then(|b| b.package.as_ref()));
        }

        Ok(corpus)
    }

    // Studies whose paper text actually came back. A study without it does not
    // look wrong in the data — its publications are populated and classified
    // `oa` — because nothing is wrong: bronze and green routes are genuinely
    // open access and still deposit nothing to Europe PMC, which is the only
    // source the harvest reads. The sole signal is absent or empty text.
    #[inline]
    pub fn with_paper_text(&self) -> impl Iterator<Item = &Project> {
        self.projects.iter().filter(|p| !papers_of(p).is_empty())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CorpusParams {
    pub source: String,
    pub max_records: usize,
    pub biosample: bool,
    pub bioproject: bool,
    pub papers: bool,
    pub study_count: usize,
    pub format_version: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CorpusCounts {
    pub studies: usize,
    pub records: usize,
    pub samples: usize,
    pub papers: usize,
    pub papers_empty: usize,
    pub summary_only: usize,
}

// Every retrieved text a study actually has. 37 of 346 studies carry more than
// one, which is why this is a Vec and not an Option — the short-circuit that
// used to cap studies at a single paper was removed from the harvest.
#[inline]
pub fn papers_of(project: &Project) -> Vec<&Paper> {
    project
        .publications
        .iter()
        .filter_map(|p| p.paper.as_ref())
        .filter(|p| p.text.is_some())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::project::PublicationDb;

    // A real slice of `datasets/oa_corpus_full.json`, not a hand-written file:
    // the point of the fixture is to prove the deserializer matches what the
    // builder actually emits, and a fixture written from the DTO definitions
    // would only prove it matches itself. Trimmed to 3 studies / 4 samples and
    // paper texts cut to 280 chars, with `chars` corrected to match.
    //
    // Three studies, chosen to cover each branch of the paper lookup:
    //   DRP006604  3 publications, all resolving to papers with text
    //   DRP003937  1 publication, `oa`, but no paper_ids at all - the bronze /
    //              green case, open access with nothing deposited to Europe PMC
    //   SRP999999  SYNTHETIC. A publication pointing at a paper whose text is
    //              empty. No study in the real corpus does this - all 61
    //              empty-text papers are orphans, referenced by nothing - but
    //              the text filter in `papers_of` is reachable code, so the
    //              case has to be manufactured to cover it. The accession is
    //              deliberately not a real one.
    const MINI: &str = include_str!("../test_data/mini_corpus.json");

    fn corpus() -> Corpus {
        Corpus::from_json(MINI, false).expect("fixture should parse")
    }

    fn project(accession: &str) -> Project {
        corpus()
            .projects
            .into_iter()
            .find(|p| p.accession.0 == accession)
            .unwrap_or_else(|| panic!("{accession} missing from fixture"))
    }

    #[test]
    fn test_from_json() {
        let corpus = corpus();

        assert_eq!(corpus.format_version, SUPPORTED_FORMAT_VERSION);
        assert_eq!(corpus.created.raw, "2026-08-16T04:42:51+00:00");
        assert_eq!(corpus.params.source, "datasets/oa_corpus.json");
        assert_eq!(corpus.counts.studies, 3);
        assert_eq!(corpus.projects.len(), corpus.counts.studies);

        // The two-pass paper resolution: publications carry `paper_ids` on the
        // wire and own a `Paper` afterwards. Checking the id alone would pass
        // even if the map lookup silently handed back the wrong entry, so the
        // text is checked too.
        let multi = project("DRP006604");
        assert_eq!(multi.publications.len(), 3);
        for publication in &multi.publications {
            let paper = publication.paper.as_ref().expect("paper should resolve");
            assert_eq!(paper.id, publication.id);
            assert_eq!(paper.char_count, 280);
            assert_eq!(paper.text.as_ref().map(|t| t.chars().count()), Some(280));
            assert!(paper.truncated);
        }

        // A publication with no paper_ids resolves to no paper, rather than to
        // an empty one.
        let no_ids = project("DRP003937");
        assert_eq!(no_ids.publications.len(), 1);
        assert!(no_ids.publications[0].paper.is_none());

        // The DOI publication proves `type` is parsed, not passed through: the
        // wire says "eDOI" and the other two say "ePubmed".
        let dois = multi
            .publications
            .iter()
            .filter(|p| p.db_type == PublicationDb::EDoi)
            .count();
        assert_eq!(dois, 1);

        // Nested structure survives the trim, so the fixture still exercises
        // the deep DTO conversions rather than just the top level.
        assert_eq!(multi.samples.len(), 2);
        assert_eq!(multi.experiments.len(), 2);
        assert!(multi.experiments.iter().all(|e| !e.runs.is_empty()));
        // Every experiment's sample_ids resolve; a fixture that lost this would
        // make the target-schema tests pass for the wrong reason.
        for experiment in &multi.experiments {
            for id in &experiment.sample_ids {
                assert!(multi.samples.contains_key(id), "dangling sample id {id:?}");
            }
        }

        // Corpus-format bookkeeping is copied onto every project, not just kept
        // at the root, so a Project separated from its Corpus still knows how
        // stale it is.
        assert_eq!(multi.source.corpus_format_version, SUPPORTED_FORMAT_VERSION);
        assert_eq!(multi.source.fetched_at.raw, corpus.created.raw);
    }

    #[test]
    fn test_from_json_rejects_unsupported_version() {
        // Only the first occurrence: `format_version` appears again inside
        // `params`, and the check reads the root one.
        let bumped = MINI.replacen(
            &format!("\"format_version\": {SUPPORTED_FORMAT_VERSION}"),
            "\"format_version\": 99",
            1,
        );
        match Corpus::from_json(&bumped, false) {
            Err(CorpusError::UnsupportedVersion { found, supported }) => {
                assert_eq!(found, 99);
                assert_eq!(supported, SUPPORTED_FORMAT_VERSION);
            }
            other => panic!("expected UnsupportedVersion, got {other:?}"),
        }
    }

    #[test]
    fn test_from_json_rejects_malformed() {
        // Truncation rather than garbage: a half-written corpus is the failure
        // that actually happens, and it is the one that could plausibly parse
        // into a short-but-valid-looking Corpus.
        let truncated = &MINI[..MINI.len() / 2];
        assert!(matches!(
            Corpus::from_json(truncated, false),
            Err(CorpusError::Json(_))
        ));
    }

    #[test]
    fn test_with_paper_text() {
        let corpus = corpus();
        let with_text: Vec<_> = corpus
            .with_paper_text()
            .map(|p| p.accession.0.as_str())
            .collect();

        // One of three. The other two are the reason this method exists: both
        // carry a publication classified `oa`, and neither yields any text.
        assert_eq!(with_text, vec!["DRP006604"]);

        // Having publications is not the same as having text, which is the
        // distinction the whole method turns on.
        assert!(corpus.projects.iter().all(|p| !p.publications.is_empty()));
    }

    #[test]
    fn test_papers_of() {
        // Every publication resolves, so this is 3 and not 1: the single-paper
        // short-circuit that used to cap a study at one paper is gone.
        assert_eq!(papers_of(&project("DRP006604")).len(), 3);

        // No paper_ids at all - nothing to filter, nothing to return.
        assert!(papers_of(&project("DRP003937")).is_empty());

        // The paper resolves and is attached, but its text is empty, so it is
        // filtered out. This is the branch the synthetic study exists for.
        let empty = project("SRP999999");
        assert!(empty.publications[0].paper.is_some());
        assert!(papers_of(&empty).is_empty());

        // Order follows the publication order, which the model's evidence
        // string depends on: a set here would reintroduce the nondeterminism
        // the BTreeMaps were chosen to avoid.
        let multi = project("DRP006604");
        let ids: Vec<_> = papers_of(&multi).iter().map(|p| p.id.as_str()).collect();
        assert_eq!(ids, vec!["10.1038/s41598-020-72888-6", "32973264", "36223455"]);
    }
}