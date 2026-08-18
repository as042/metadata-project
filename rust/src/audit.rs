use std::fmt;

use serde_json::Value;

use crate::export::Export;
use crate::target_schema::{Directness, TargetSchema, EVIDENCE_HEADER};

// Checking the one claim a model makes that can be falsified.
//
// `quoted` says a value appears in the evidence word for word. That is not an
// opinion and not a confidence — it is a statement about a string, and a string
// match can say whether it holds. `rephrased` and `inferred` claim the opposite
// and are not checkable this way, which is the whole reason the directness axis
// replaced a confidence score: it made one bucket auditable for free.
//
// No model, no network, no spend. The input is a saved run.

// How much of the record's field-slots this could speak to.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Audit {
    // `quoted` claims that were checked.
    pub checked: usize,
    pub verified: usize,
    pub unsupported: Vec<Finding>,
    // Values labelled `rephrased` or `inferred` that turn out to be verbatim
    // anyway. Not errors — under-claiming is the safe direction — but a run
    // where this is large is a run whose labels are not tracking what was done.
    pub understated: Vec<Finding>,
    // Inferred answers with no evidence stored for their record. Reported
    // rather than counted as passes: an unaudited claim is not a verified one.
    pub unauditable: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Finding {
    pub record: String,
    pub field: String,
    pub value: String,
    pub claimed: Directness,
}

impl Audit {
    // Of the claims that could be checked, how many held.
    pub fn rate(&self) -> f64 {
        if self.checked == 0 {
            return 1.0;
        }
        self.verified as f64 / self.checked as f64
    }
}

impl fmt::Display for Audit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "verbatim audit: {}/{} quoted claims verified ({:.0}%), {} unauditable",
            self.verified,
            self.checked,
            self.rate() * 100.0,
            self.unauditable
        )?;
        for finding in &self.unsupported {
            writeln!(
                f,
                "  NOT VERBATIM  {}  {} = {:?}",
                finding.record, finding.field, finding.value
            )?;
        }
        for finding in &self.understated {
            writeln!(
                f,
                "  understated   {}  {} = {:?} (labelled {:?}, but verbatim)",
                finding.record, finding.field, finding.value, finding.claimed
            )?;
        }
        Ok(())
    }
}

// Casefold, reduce every run of non-alphanumeric characters to one space, and
// pad with spaces.
//
// The padding is what makes this a token-boundary test rather than a substring
// one: without it "USA" matches inside "USAGE". Collapsing punctuation is what
// lets a value survive the attribute bag arriving as JSON — `{"age": "8 weeks"}`
// normalises to ` age 8 weeks `, so `8 weeks` is found.
fn normalize(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push(' ');
    let mut space = true;
    for c in text.chars() {
        if c.is_alphanumeric() {
            for lower in c.to_lowercase() {
                out.push(lower);
            }
            space = false;
        } else if !space {
            out.push(' ');
            space = true;
        }
    }
    // Always closes with a space, so the result is bounded on both sides and a
    // needle can only match on token boundaries. An all-punctuation input ends
    // as two spaces, which matches nothing.
    if !space || out.len() == 1 {
        out.push(' ');
    }
    out
}

fn contains(haystack: &str, needle: &str) -> bool {
    let needle = normalize(needle);
    // A value that normalises to nothing cannot be looked for.
    if needle.trim().is_empty() {
        return false;
    }
    normalize(haystack).contains(&needle)
}

// The part of a record's stored evidence that one layer showed it.
//
// A record's entry is the concatenation of every block it was shown, each under
// a `=== evidence: <layer> ===` line. Slicing by layer is what keeps a `quoted`
// claim falsifiable once more than one layer runs: layer 3 saw an attribute bag
// and layer 4 saw a publication, and checking the first against the second would
// pass claims that were never supported.
//
// A run saved before the headers existed has none, and reads as one block for
// whichever layer asks. That is right for those files — they were all layer 3.
fn section<'a>(evidence: &'a str, layer: &str) -> std::borrow::Cow<'a, str> {
    if !evidence.contains(EVIDENCE_HEADER) {
        return evidence.into();
    }
    let header = format!("{EVIDENCE_HEADER}{layer} ===");
    let mut out = String::new();
    let mut keeping = false;
    for line in evidence.lines() {
        if line.starts_with(EVIDENCE_HEADER) {
            keeping = line == header;
            continue;
        }
        if keeping {
            out.push_str(line);
            out.push('\n');
        }
    }
    out.into()
}

// Which layer's evidence a provenance points at. The two model layers are the
// only ones that claim a directness at all.
fn layer_of(provenance: &Value) -> Option<(&'static str, Directness)> {
    for (key, layer) in [
        ("InferredFromText", "llm_naive"),
        ("InferredFromPaper", "llm_paper"),
    ] {
        if let Some(Ok(directness)) = provenance
            .get(key)
            .map(|value| serde_json::from_value::<Directness>(value.clone()))
        {
            return Some((layer, directness));
        }
    }
    None
}

// Every model-layer answer in a record, as (field, text, layer, directness).
//
// Read off the serialized form rather than through a 62-arm match: this is a
// diagnostic over a saved run, the export is JSON, and the shapes below are the
// shapes the file actually has. The test at the bottom pins them, so a change to
// the representation fails here rather than silently auditing nothing.
fn inferred_answers(record: &TargetSchema) -> Vec<(String, String, &'static str, Directness)> {
    let Ok(Value::Object(fields)) = serde_json::to_value(record) else {
        return Vec::new();
    };
    let mut out = Vec::new();

    for (name, value) in fields {
        let Value::Object(variant) = value else { continue };
        let Some((kind, payload)) = variant.into_iter().next() else { continue };
        let Value::Array(parts) = payload else { continue };
        let [held, provenance] = parts.as_slice() else { continue };

        // Only a model layer's answers carry a directness worth checking.
        let Some((layer, directness)) = layer_of(provenance) else {
            continue;
        };

        let text = match kind.as_str() {
            // A stated absence claims the INSDC term appears in the evidence.
            "Missing" => serde_json::from_value::<crate::target_schema::MissingReason>(held.clone())
                .ok()
                .map(|reason| reason.as_str().to_string()),
            // Typed fields were parsed from text that is no longer kept, so only
            // the ones that are still strings can be matched against evidence.
            "Known" => held.as_str().map(str::to_string),
            _ => None,
        };
        if let Some(text) = text {
            out.push((name, text, layer, directness));
        }
    }
    out
}

// Audits every `quoted` claim in a saved run against the evidence it stored.
pub fn verbatim(export: &Export) -> Audit {
    let mut audit = Audit::default();

    for record in &export.records {
        let id = &record.id.0;
        let evidence = export.evidence.as_ref().and_then(|e| e.get(id));

        for (field, value, layer, claimed) in inferred_answers(record) {
            // Checked against what *that* layer showed this record, not against
            // the union: a paper is a 30,000-character haystack, and letting a
            // layer-3 claim match inside it would verify something the call was
            // never given.
            let evidence = evidence.map(|text| section(text, layer));
            let Some(evidence) = evidence else {
                audit.unauditable += 1;
                continue;
            };
            let found = contains(&evidence, &value);
            let finding = || Finding {
                record: id.clone(),
                field: field.clone(),
                value: value.clone(),
                claimed,
            };
            match claimed {
                Directness::Quoted => {
                    audit.checked += 1;
                    if found {
                        audit.verified += 1;
                    } else {
                        audit.unsupported.push(finding());
                    }
                }
                // Under-claiming is safe, but worth surfacing: a run where most
                // `inferred` answers are verbatim is a run whose labels are not
                // describing what was done.
                _ if found => audit.understated.push(finding()),
                _ => {}
            }
        }
    }
    audit
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::export::{Counts, Params, FORMAT_VERSION};
    use crate::target_schema::{Field, MissingReason, Provenance};
    use std::collections::BTreeMap;

    fn export(record: TargetSchema, evidence: Option<&str>) -> Export {
        let id = record.id.0.clone();
        Export {
            format_version: FORMAT_VERSION,
            created: "2026-08-17T12:00:00+00:00".into(),
            params: Params::default(),
            counts: Counts::default(),
            evidence: evidence.map(|e| BTreeMap::from([(id, e.to_string())])),
            records: vec![record],
        }
    }

    fn quoted(value: &str) -> Field<String> {
        Field::Known(value.into(), Provenance::InferredFromText(Directness::Quoted))
    }

    fn record() -> TargetSchema {
        TargetSchema {
            id: crate::project::ExperimentAccession("DRX1".into()),
            ..Default::default()
        }
    }

    // -- whose evidence a claim is checked against ---------------------------

    // What the store actually holds once both model layers have run.
    fn tagged(blocks: &[(&str, &str)]) -> String {
        blocks
            .iter()
            .map(|(layer, text)| format!("{EVIDENCE_HEADER}{layer} ===\n{text}"))
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn a_text_claim_is_not_verified_against_the_paper_the_call_never_saw() {
        // The failure adding layer 4 would otherwise introduce. The paper is a
        // 30,000-character haystack appended to the same record, and a layer-3
        // claim matching inside it is a claim verified against evidence that
        // call was not given — an unfalsifiable claim wearing a pass.
        let mut r = record();
        r.host = quoted("Apis mellifera");
        let evidence = tagged(&[
            ("llm_naive", r#"SAMPLE ATTRIBUTES: {"strain": "W303"}"#),
            ("llm_paper", "We sampled the gut of Apis mellifera workers."),
        ]);
        let audit = verbatim(&export(r, Some(&evidence)));
        assert_eq!((audit.checked, audit.verified), (1, 0));
        assert_eq!(audit.unsupported.len(), 1, "the paper verified a layer-3 claim");
    }

    #[test]
    fn a_paper_claim_is_verified_against_the_paper() {
        // And the converse: layer 4 is audited on the same terms as layer 3,
        // which it can be here because the evidence is kept. Python could not —
        // it never persisted the paper text.
        let mut r = record();
        r.host = Field::Known(
            "Apis mellifera".into(),
            Provenance::InferredFromPaper(Directness::Quoted),
        );
        let evidence = tagged(&[
            ("llm_naive", r#"SAMPLE ATTRIBUTES: {"strain": "W303"}"#),
            ("llm_paper", "We sampled the gut of Apis mellifera workers."),
        ]);
        let audit = verbatim(&export(r, Some(&evidence)));
        assert_eq!((audit.checked, audit.verified), (1, 1));
    }

    #[test]
    fn every_block_a_layer_wrote_is_searched_not_only_the_first() {
        // A record is covered by its study-level call and its own, and both are
        // its layer's evidence.
        let mut r = record();
        r.host = quoted("Apis mellifera");
        let evidence = tagged(&[
            ("llm_naive", "STUDY TITLE: bees"),
            ("llm_paper", "irrelevant"),
            ("llm_naive", r#"{"host": "Apis mellifera"}"#),
        ]);
        let audit = verbatim(&export(r, Some(&evidence)));
        assert_eq!((audit.checked, audit.verified), (1, 1));
    }

    #[test]
    fn a_run_saved_before_the_headers_existed_still_audits() {
        // The 15 saved runs have untagged evidence and were all layer 3, so the
        // whole entry is that layer's — reading it any other way would silently
        // turn every one of those claims unauditable.
        let mut r = record();
        r.host = quoted("Apis mellifera");
        let audit = verbatim(&export(r, Some(r#"{"host": "Apis mellifera"}"#)));
        assert_eq!((audit.checked, audit.verified), (1, 1));
    }

    // -- normalisation -------------------------------------------------------

    #[test]
    fn a_value_survives_the_attribute_bag_arriving_as_json() {
        // The bag is serialised into the evidence, so the value is surrounded by
        // quotes, colons and braces rather than spaces.
        let evidence = r#"SAMPLE ATTRIBUTES: {"host": "Apis mellifera", "age": "8 weeks"}"#;
        assert!(contains(evidence, "Apis mellifera"));
        assert!(contains(evidence, "8 weeks"));
    }

    #[test]
    fn matching_is_on_token_boundaries() {
        // Without the padding, "USA" matches inside "USAGE" and the audit
        // reports a verbatim quote that is not one.
        assert!(!contains("annual USAGE report", "USA"));
        assert!(contains("collected in USA: California", "USA"));
    }

    #[test]
    fn case_and_punctuation_do_not_matter() {
        assert!(contains("Geo Loc Name: China: Beijing", "china beijing"));
        assert!(contains("tissue: leaf", "LEAF"));
    }

    #[test]
    fn an_empty_value_is_never_a_match() {
        assert!(!contains("anything at all", ""));
        assert!(!contains("anything at all", "   "));
    }

    // -- what gets audited ---------------------------------------------------

    #[test]
    fn only_the_text_layers_answers_are_checked() {
        // A direct or harmonized value was not claimed to be quoted by anyone.
        let mut r = record();
        r.host = Field::Known("Apis mellifera".into(), Provenance::Direct);
        r.country = Field::Known("USA".into(), Provenance::Harmonized);
        let audit = verbatim(&export(r, Some("nothing relevant here")));
        assert_eq!(audit.checked, 0);
        assert_eq!(audit.unsupported.len(), 0);
    }

    #[test]
    fn a_quoted_claim_that_holds_is_verified() {
        let mut r = record();
        r.host = quoted("Apis mellifera");
        let audit = verbatim(&export(r, Some(r#"{"host": "Apis mellifera"}"#)));
        assert_eq!((audit.checked, audit.verified), (1, 1));
        assert!(audit.unsupported.is_empty());
    }

    #[test]
    fn a_quoted_claim_that_does_not_hold_is_reported() {
        // The real finding from the live runs, in miniature.
        let mut r = record();
        r.collected_by = Field::Missing(
            MissingReason::NotProvided,
            Provenance::InferredFromText(Directness::Quoted),
        );
        let audit = verbatim(&export(r, Some("STUDY TITLE: honey bee gut microbiome")));
        assert_eq!((audit.checked, audit.verified), (1, 0));
        assert_eq!(audit.unsupported.len(), 1);
        assert_eq!(audit.unsupported[0].field, "collected_by");
        assert_eq!(audit.unsupported[0].value, "not provided");
    }

    #[test]
    fn a_stated_absence_the_evidence_does_say_is_verified() {
        // The other half of the rule: if the bag literally reads "not
        // collected", quoting it is correct.
        let mut r = record();
        r.collected_by = Field::Missing(
            MissingReason::NotCollected,
            Provenance::InferredFromText(Directness::Quoted),
        );
        let audit = verbatim(&export(r, Some(r#"{"collected_by": "not collected"}"#)));
        assert_eq!((audit.checked, audit.verified), (1, 1));
    }

    #[test]
    fn rephrased_and_inferred_are_not_counted_as_quoted_claims() {
        let mut r = record();
        r.host = Field::Known(
            "Apis mellifera".into(),
            Provenance::InferredFromText(Directness::Rephrased),
        );
        r.sex = Field::Missing(
            MissingReason::NotApplicable,
            Provenance::InferredFromText(Directness::Inferred),
        );
        let audit = verbatim(&export(r, Some("nothing matching")));
        assert_eq!(audit.checked, 0);
    }

    #[test]
    fn an_under_claim_is_surfaced_without_being_an_error() {
        // Labelled inferred, but the value is right there in the evidence.
        let mut r = record();
        r.host = Field::Known(
            "Apis mellifera".into(),
            Provenance::InferredFromText(Directness::Inferred),
        );
        let audit = verbatim(&export(r, Some(r#"{"host": "Apis mellifera"}"#)));
        assert_eq!(audit.checked, 0, "not a quoted claim");
        assert_eq!(audit.understated.len(), 1);
        assert_eq!(audit.understated[0].claimed, Directness::Inferred);
    }

    #[test]
    fn a_run_with_no_stored_evidence_reports_unauditable_rather_than_passing() {
        // The failure mode to avoid: an export saved without evidence quietly
        // auditing clean and looking like a perfect score.
        let mut r = record();
        r.host = quoted("Apis mellifera");
        let audit = verbatim(&export(r, None));
        assert_eq!(audit.checked, 0);
        assert_eq!(audit.verified, 0);
        assert_eq!(audit.unauditable, 1);
        assert!(audit.to_string().contains("1 unauditable"));
    }

    #[test]
    fn a_perfect_run_reports_a_full_rate_and_an_empty_one_does_not_divide_by_zero() {
        let mut r = record();
        r.host = quoted("Apis mellifera");
        assert_eq!(verbatim(&export(r, Some("host Apis mellifera"))).rate(), 1.0);
        assert_eq!(verbatim(&export(record(), Some("x"))).rate(), 1.0);
    }

    #[test]
    fn typed_fields_are_skipped_rather_than_stringified() {
        // taxon_id was parsed from text the export no longer keeps, so Debug-
        // formatting the integer and matching that would be checking a string
        // the model never wrote.
        let mut r = record();
        r.taxon_id = Field::Known(749906, Provenance::InferredFromText(Directness::Quoted));
        let audit = verbatim(&export(r, Some("taxon 749906")));
        assert_eq!(audit.checked, 0, "not auditable, so not counted either way");
    }

    #[test]
    fn the_serialized_shape_this_reads_is_the_shape_the_export_has() {
        // `inferred_answers` walks the JSON rather than matching 62 fields. If
        // the representation changes, this fails loudly instead of the audit
        // silently finding nothing to check.
        let mut r = record();
        r.host = quoted("Apis mellifera");
        let json = serde_json::to_value(&r).unwrap();
        assert_eq!(
            json["host"],
            serde_json::json!({"Known": ["Apis mellifera", {"InferredFromText": "Quoted"}]})
        );
        assert_eq!(json["age"], serde_json::json!("Unknown"));
        assert_eq!(inferred_answers(&r).len(), 1);
    }
}
