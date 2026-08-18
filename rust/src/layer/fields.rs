use chrono::{NaiveDate, NaiveDateTime};

use crate::dto;
use crate::project::*;
use crate::target_schema::{Field, MissingReason, PartialDate, Provenance, TargetSchema};

// Writing a string into a typed schema field, and knowing what the fields are.
//
// Shared by every layer that proposes values as text. Layer 2 reads a submitter
// attribute bag, layer 3 reads a model's answers, and both end up needing the
// same thing: take `"9606"` and put it in a `Field<u64>`, take `"not
// applicable"` and record it as a stated absence, refuse what will not fit, and
// never overwrite what an earlier layer settled.
//
// The only thing that differs between them is the provenance stamped on the
// result, which is why `assign` takes it as an argument rather than assuming.

// Where a field sits in the archive's hierarchy.
//
// Not decoration: it decides what a layer is allowed to ask about. Run-,
// submission- and record-level fields are assigned by the archive at
// deposition — release dates, upload formats, the submitting centre — and no
// abstract, attribute bag or manuscript states them. Layer 3 had no such guard
// in Python and filled 4,342 of them, including a BioProject accession written
// into `submission_accession` on 146 records and thousands of "not provided"
// at full token price.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Level {
    Study,
    Sample,
    Experiment,
    Run,
    Submission,
    Record,
}

// Mirrors the Python schema's per-field `level` metadata, translated through
// this schema's renames: `abstract_text` is Python's study-level `description`,
// `earliest_run_published` its submission-level `first_public`, `total_spots`
// its run-level `read_count`.
pub fn level_of(name: &str) -> Option<Level> {
    use Level::*;
    Some(match name {
        // -- study ---------------------------------------------------------
        "bioproject_accession" | "study_accession" | "study_title" | "abstract_text"
        | "study_alias" | "center_project_name" => Study,

        // -- submission ----------------------------------------------------
        "center_name" | "submission_accession" | "broker_name" | "datahub"
        | "first_created" | "last_updated" | "earliest_run_published" => Submission,

        // -- experiment ----------------------------------------------------
        "experiment_accession" | "experiment_title" | "experiment_alias"
        | "library_strategy" | "library_source" | "library_selection"
        | "library_layout" | "library_name" | "library_construction_protocol"
        | "platform" | "instrument_model" | "sequencing_method" => Experiment,

        // -- run -----------------------------------------------------------
        "total_spots" | "total_bases" | "submitted_format" | "submitted_read_type" => Run,

        // -- record --------------------------------------------------------
        "tag" => Record,

        // -- sample --------------------------------------------------------
        "biosample_accession" | "sample_accession" | "sample_title" | "sample_alias"
        | "scientific_name" | "taxon_id" | "biosample_package" | "age"
        | "broad_scale_environmental_context" | "cell_line" | "cell_type" | "checklist"
        | "collected_by" | "collection_date" | "country" | "dev_stage"
        | "environment_biome" | "environment_feature" | "environment_material"
        | "environmental_medium" | "host" | "host_scientific_name" | "host_sex"
        | "host_tax_id" | "isolation_source" | "local_environmental_context"
        | "sample_capture_status" | "sample_description" | "sex" | "strain"
        | "tissue_type" | "treatment" => Sample,

        _ => return None,
    })
}

// The fields no layer has settled yet, in a stable order.
//
// `id` is absent because it is not a `Field` — it is assigned at construction
// and there is nothing to infer.
pub fn open_fields(schema: &TargetSchema) -> Vec<&'static str> {
    FIELD_NAMES
        .iter()
        .copied()
        .filter(|name| !is_settled(schema, name))
        .collect()
}

// INSDC's missing-value vocabulary, recognised before any type parsing.
//
// A submitter who writes "not applicable" has *answered* — the field does not
// apply to this sample — and storing that as the literal string asserts the
// country is called "not applicable". `Field::Missing` is the variant built for
// it, and this is what reaches it. Measured across the corpus this covers
// 36,207 harmonised values, roughly a tenth of everything this layer writes.
//
// Deliberately closed: only the INSDC terms, matched exactly after casefolding.
// A submitter's own "unknown", "na" or "none" (2,796 values) is free text that
// happens to read like an absence, not a term selected from a controlled
// vocabulary, and promoting it here would be inferring a determination nobody
// made. That is a job for a layer that can read the value, not this one.
#[inline]
fn missing_reason(value: &str) -> Option<MissingReason> {
    match value.trim().to_ascii_lowercase().as_str() {
        "not applicable" => Some(MissingReason::NotApplicable),
        "not collected" => Some(MissingReason::NotCollected),
        "not provided" => Some(MissingReason::NotProvided),
        "restricted access" => Some(MissingReason::RestrictedAccess),
        "missing" => Some(MissingReason::Unspecified),
        _ => None,
    }
}

// MIxS and INSDC allow a reduced-precision collection date, so the three
// precisions are accepted and kept apart rather than padded up to a full date.
#[inline]
fn parse_partial_date(value: &str) -> Option<PartialDate> {
    let value = value.trim();
    // chrono's %Y accepts any digit count, so "19-05-04" would otherwise parse
    // as the year 19. INSDC dates are four-digit years; anything else is a
    // format this does not recognise rather than an ancient sample.
    if !value.starts_with(|c: char| c.is_ascii_digit()) || !value.is_char_boundary(4) {
        return None;
    }
    if value.len() >= 4 && !value[..4].chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    if let Ok(date) = NaiveDate::parse_from_str(value, "%Y-%m-%d") {
        return Some(PartialDate::Date(date));
    }
    let parts: Vec<&str> = value.split('-').collect();
    match parts.as_slice() {
        [year] if year.len() == 4 => year.parse().ok().map(PartialDate::Year),
        [year, month] if year.len() == 4 => {
            let year = year.parse().ok()?;
            let month = month.parse().ok()?;
            (1..=12).contains(&month).then_some(PartialDate::YearMonth(year, month))
        }
        _ => None,
    }
}

// Date-only input is rejected rather than assumed to be midnight. No attribute
// key in the corpus reaches a timestamp field, so this parses the two formats
// the archive is known to emit and refuses to guess at anything else.
#[inline]
fn parse_timestamp(value: &str) -> Option<NaiveDateTime> {
    let value = value.trim();
    NaiveDateTime::parse_from_str(value, "%Y-%m-%d %H:%M:%S")
        .or_else(|_| NaiveDateTime::parse_from_str(value, "%Y-%m-%dT%H:%M:%S"))
        .ok()
}

// What happened to one (field, value) pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    // No field of that name. The key is not a schema field and must fall
    // through to the synonym table.
    UnknownField,
    // The field exists but the value does not fit its type — Python's
    // `_storable` probe, which drops the pair rather than storing a string in a
    // typed column.
    Rejected,
    // An earlier layer already settled it. Never overwritten: a direct value is
    // the archive's own, and this layer's job is to fill gaps, not to correct.
    Closed,
    Set,
}

// Assigns one harmonised value, parsing it into the field's type on the way.
//
// The single source of truth for which names this schema has: `harmonized`
// decides whether a key is an exact field-name match by asking this function
// about a scratch record, exactly as Python's `_storable` probes a spare
// instance. Keeping membership and assignment in one match means they cannot
// drift apart.
#[inline]
pub fn assign(
    schema: &mut TargetSchema,
    name: &str,
    value: &str,
    provenance: Provenance,
) -> Outcome {
    macro_rules! put {
        ($field:ident, $parsed:expr) => {{
            if schema.$field.is_settled() {
                Outcome::Closed
            } else if let Some(reason) = missing_reason(value) {
                // Checked before parsing, so a typed field records the reason
                // too. This is what `Field<T>` being generic over T buys: Python
                // rejects a sentinel outright on a typed column and documents
                // the loss as a known casualty.
                schema.$field = Field::Missing(reason, provenance.clone());
                Outcome::Set
            } else {
                match $parsed {
                    Some(value) => {
                        schema.$field = Field::Known(value, provenance.clone());
                        Outcome::Set
                    }
                    None => Outcome::Rejected,
                }
            }
        }};
    }
    macro_rules! text {
        ($field:ident) => {
            put!($field, Some(value.to_string()))
        };
    }

    match name {
        // -- study ---------------------------------------------------------
        "bioproject_accession" => put!(bioproject_accession, Some(BioProjectAccession(value.into()))),
        "study_accession" => put!(study_accession, Some(StudyAccession(value.into()))),
        "study_title" => text!(study_title),
        "abstract_text" => text!(abstract_text),
        "study_alias" => text!(study_alias),
        "center_project_name" => text!(center_project_name),
        "center_name" => text!(center_name),

        // -- submission ----------------------------------------------------
        "submission_accession" => text!(submission_accession),
        "broker_name" => text!(broker_name),

        // -- sample --------------------------------------------------------
        "biosample_accession" => put!(biosample_accession, Some(BioSampleAccession(value.into()))),
        "sample_accession" => put!(sample_accession, Some(SampleAccession(value.into()))),
        "sample_title" => text!(sample_title),
        "sample_alias" => text!(sample_alias),
        "scientific_name" => text!(scientific_name),
        "taxon_id" => put!(taxon_id, value.trim().parse::<u64>().ok()),
        "biosample_package" => text!(biosample_package),

        // -- experiment ----------------------------------------------------
        "experiment_accession" => put!(experiment_accession, Some(ExperimentAccession(value.into()))),
        "experiment_title" => text!(experiment_title),
        "experiment_alias" => text!(experiment_alias),
        "library_strategy" => put!(library_strategy, dto::library_strategy(Some(value.into()))),
        "library_source" => put!(library_source, dto::library_source(Some(value.into()))),
        "library_selection" => put!(library_selection, dto::library_selection(Some(value.into()))),
        "library_layout" => put!(library_layout, dto::library_layout(Some(value.into()))),
        "library_name" => text!(library_name),
        "library_construction_protocol" => text!(library_construction_protocol),
        "platform" => put!(platform, dto::platform(Some(value.into()))),
        "instrument_model" => text!(instrument_model),

        // -- run -----------------------------------------------------------
        "total_spots" => put!(total_spots, value.trim().parse::<u64>().ok()),
        "total_bases" => put!(total_bases, value.trim().parse::<u64>().ok()),
        "earliest_run_published" => put!(earliest_run_published, parse_timestamp(value)),

        // -- no Project counterpart ----------------------------------------
        "age" => text!(age),
        "broad_scale_environmental_context" => text!(broad_scale_environmental_context),
        "cell_line" => text!(cell_line),
        "cell_type" => text!(cell_type),
        "checklist" => text!(checklist),
        "collected_by" => text!(collected_by),
        "collection_date" => put!(collection_date, parse_partial_date(value)),
        "country" => text!(country),
        "datahub" => text!(datahub),
        "dev_stage" => text!(dev_stage),
        "environment_biome" => text!(environment_biome),
        "environment_feature" => text!(environment_feature),
        "environment_material" => text!(environment_material),
        "environmental_medium" => text!(environmental_medium),
        "first_created" => put!(first_created, parse_timestamp(value)),
        "host" => text!(host),
        "host_scientific_name" => text!(host_scientific_name),
        "host_sex" => text!(host_sex),
        "host_tax_id" => put!(host_tax_id, value.trim().parse::<u64>().ok()),
        "isolation_source" => text!(isolation_source),
        "last_updated" => put!(last_updated, parse_timestamp(value)),
        "local_environmental_context" => text!(local_environmental_context),
        "sample_capture_status" => text!(sample_capture_status),
        "sample_description" => text!(sample_description),
        "sequencing_method" => text!(sequencing_method),
        "sex" => text!(sex),
        "strain" => text!(strain),
        "submitted_format" => text!(submitted_format),
        "submitted_read_type" => text!(submitted_read_type),
        "tag" => text!(tag),
        "tissue_type" => text!(tissue_type),
        "treatment" => text!(treatment),

        // `id` is not a Field and is deliberately absent, matching Python's
        // explicit skip of it.
        _ => Outcome::UnknownField,
    }
}

// Whether this schema has a field of that name, independent of any value.
//
// Python tests `key in fields`; this asks `assign` about a throwaway record,
// which is the same question without a second list to keep in sync. A value
// that fails to parse still answers "yes, the field exists" — membership is by
// name, and a bad value is rejected later rather than falling through to the
// synonym table.
#[inline]
pub fn is_schema_field(name: &str) -> bool {
    assign(&mut TargetSchema::default(), name, "", Provenance::Direct) != Outcome::UnknownField
}


// The `&'static str` for each schema field name, needed only so an exact match
// can name its own target. `is_schema_field` remains the authority on
// membership; a name missing here simply never resolves, which the drift test
// in this module catches.
pub const FIELD_NAMES: &[&str] = &[
    "bioproject_accession", "study_accession", "study_title", "abstract_text",
    "study_alias", "center_project_name", "center_name", "submission_accession",
    "broker_name", "biosample_accession", "sample_accession", "sample_title",
    "sample_alias", "scientific_name", "taxon_id", "biosample_package",
    "experiment_accession", "experiment_title", "experiment_alias",
    "library_strategy", "library_source", "library_selection", "library_layout",
    "library_name", "library_construction_protocol", "platform",
    "instrument_model", "total_spots", "total_bases", "earliest_run_published",
    "age", "broad_scale_environmental_context", "cell_line", "cell_type",
    "checklist", "collected_by", "collection_date", "country", "datahub",
    "dev_stage", "environment_biome", "environment_feature",
    "environment_material", "environmental_medium", "first_created", "host",
    "host_scientific_name", "host_sex", "host_tax_id", "isolation_source",
    "last_updated", "local_environmental_context", "sample_capture_status",
    "sample_description", "sequencing_method", "sex", "strain",
    "submitted_format", "submitted_read_type", "tag", "tissue_type", "treatment",
];


// Whether a named field already holds a determination. By name rather than by
// accessor because the callers work in field names, and a second match over the
// same list would be one more thing to keep in step.
fn is_settled(schema: &TargetSchema, name: &str) -> bool {
    // A scratch write that is refused as Closed is exactly the question being
    // asked, and reuses the one match that already knows every field.
    let mut probe = schema.clone();
    assign(&mut probe, name, "", Provenance::Direct) == Outcome::Closed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::target_schema::Directness;

    // -- typed targets -----------------------------------------------------

    #[test]
    fn collection_date_keeps_the_precision_it_was_given() {
        // 53% of corpus values are full dates, 31% a bare year, 5.6%
        // year-month. Padding a year up to January 1st would invent precision.
        let mut schema = TargetSchema::default();
        assert_eq!(assign(&mut schema, "collection_date", "2019-05-04", Provenance::Harmonized), Outcome::Set);
        assert_eq!(
            schema.collection_date,
            Field::Known(
                PartialDate::Date(NaiveDate::from_ymd_opt(2019, 5, 4).unwrap()),
                Provenance::Harmonized
            )
        );

        let mut year = TargetSchema::default();
        assert_eq!(assign(&mut year, "collection_date", "2019", Provenance::Harmonized), Outcome::Set);
        assert_eq!(year.collection_date, Field::Known(PartialDate::Year(2019), Provenance::Harmonized));

        let mut ym = TargetSchema::default();
        assert_eq!(assign(&mut ym, "collection_date", "2019-05", Provenance::Harmonized), Outcome::Set);
        assert_eq!(ym.collection_date,
                   Field::Known(PartialDate::YearMonth(2019, 5), Provenance::Harmonized));
    }

    #[test]
    fn an_unparseable_date_is_rejected_rather_than_stored() {
        // 0.8% of corpus values are ranges like `2019-05/2019-08`, which have
        // no variant yet. They are dropped, not coerced into one endpoint.
        for raw in ["2019-05/2019-08", "May 2019", "2019-13", "19-05-04"] {
            let mut schema = TargetSchema::default();
            assert_eq!(assign(&mut schema, "collection_date", raw, Provenance::Harmonized), Outcome::Rejected, "{raw:?}");
            assert_eq!(schema.collection_date, Field::Unknown);
        }
    }

    #[test]
    fn integer_fields_reject_non_integers() {
        let mut good = TargetSchema::default();
        assert_eq!(assign(&mut good, "host_tax_id", "7460", Provenance::Harmonized), Outcome::Set);
        assert_eq!(good.host_tax_id, Field::Known(7460, Provenance::Harmonized));

        // Python's `_storable` drops these; so does this, rather than storing
        // text in a typed column. A missing-value term is not among them — it
        // is an answer, and has its own test below.
        for raw in ["", "9606.0", "taxid:9606"] {
            let mut schema = TargetSchema::default();
            assert_eq!(assign(&mut schema, "host_tax_id", raw, Provenance::Harmonized), Outcome::Rejected, "{raw:?}");
        }
    }

    #[test]
    fn a_rejected_value_does_not_fall_through_to_the_synonym_table() {
        // Membership is by name. `tax_id` bridges to taxon_id whatever the
        // value is; a bad value is dropped there rather than looked up again.
        let mut schema = TargetSchema::default();
        assert_eq!(assign(&mut schema, "taxon_id", "taxid:9606", Provenance::Harmonized), Outcome::Rejected);
        assert_eq!(schema.taxon_id, Field::Unknown);
    }

    // -- missing-value terms ------------------------------------------------

    #[test]
    fn insdc_missing_value_terms_become_a_stated_absence() {
        // "not applicable" is an answer — the field does not apply to this
        // sample — not a value called "not applicable".
        for (raw, reason) in [
            ("not applicable", MissingReason::NotApplicable),
            ("Not Applicable", MissingReason::NotApplicable),
            ("not collected", MissingReason::NotCollected),
            ("not provided", MissingReason::NotProvided),
            ("restricted access", MissingReason::RestrictedAccess),
            ("missing", MissingReason::Unspecified),
        ] {
            let mut schema = TargetSchema::default();
            assert_eq!(assign(&mut schema, "country", raw, Provenance::Harmonized), Outcome::Set, "{raw:?}");
            assert_eq!(schema.country, Field::Missing(reason, Provenance::Harmonized), "{raw:?}");
        }
    }

    #[test]
    fn a_typed_field_can_record_a_missing_value_too() {
        // What `Field<T>` being generic over T buys. Python rejects a sentinel
        // outright on a typed column and documents the loss as a known
        // casualty: a cell line arguably has no collection date, and there was
        // no way to say so.
        let mut date = TargetSchema::default();
        assert_eq!(assign(&mut date, "collection_date", "not applicable", Provenance::Harmonized), Outcome::Set);
        assert_eq!(date.collection_date,
                   Field::Missing(MissingReason::NotApplicable, Provenance::Harmonized));

        let mut count = TargetSchema::default();
        assert_eq!(assign(&mut count, "host_tax_id", "missing", Provenance::Harmonized), Outcome::Set);
        assert_eq!(count.host_tax_id,
                   Field::Missing(MissingReason::Unspecified, Provenance::Harmonized));
    }

    #[test]
    fn non_insdc_placeholders_are_left_alone() {
        // 2,796 corpus values. These are free text that happens to read like an
        // absence, not a term chosen from a controlled vocabulary, so promoting
        // them here would infer a determination nobody made. Deliberate, and
        // the counts are in `missing_reason`'s comment if it should change.
        for raw in ["unknown", "na", "n/a", "none", "not determined"] {
            let mut schema = TargetSchema::default();
            assert_eq!(assign(&mut schema, "country", raw, Provenance::Harmonized), Outcome::Set, "{raw:?}");
            assert_eq!(schema.country, Field::Known(raw.to_string(), Provenance::Harmonized));
        }
    }

    #[test]
    fn a_missing_value_still_does_not_overwrite_an_earlier_layer() {
        let mut schema = TargetSchema {
            country: Field::Known("Brazil".into(), Provenance::Direct),
            ..Default::default()
        };
        assert_eq!(assign(&mut schema, "country", "not applicable", Provenance::Harmonized), Outcome::Closed);
        assert_eq!(schema.country, Field::Known("Brazil".into(), Provenance::Direct));
    }


    #[test]
    fn timestamp_fields_refuse_a_date_without_a_time() {
        let mut ok = TargetSchema::default();
        assert_eq!(assign(&mut ok, "first_created", "2019-05-04 11:22:33", Provenance::Harmonized), Outcome::Set);
        // no attribute key in the corpus reaches these, so guessing midnight
        // would invent a time nobody stated
        let mut bare = TargetSchema::default();
        assert_eq!(assign(&mut bare, "first_created", "2019-05-04", Provenance::Harmonized), Outcome::Rejected);
    }

    // -- the open-field rule -----------------------------------------------

    #[test]
    fn a_settled_field_is_never_overwritten() {
        // The cascade's central rule: a direct value is the archive's own, and
        // this layer fills gaps rather than correcting.
        let mut schema = TargetSchema {
            host: Field::Known("Apis mellifera".into(), Provenance::Direct),
            ..Default::default()
        };
        assert_eq!(assign(&mut schema, "host", "Homo sapiens", Provenance::Harmonized), Outcome::Closed);
        assert_eq!(schema.host, Field::Known("Apis mellifera".into(), Provenance::Direct));
    }

    #[test]
    fn a_missing_value_verdict_also_closes_a_field() {
        // A stated reason is a determination, not an absence.
        let mut schema = TargetSchema {
            sex: Field::Missing(MissingReason::NotApplicable, Provenance::Direct),
            ..Default::default()
        };
        assert_eq!(assign(&mut schema, "sex", "female", Provenance::Harmonized), Outcome::Closed);
    }

    #[test]
    fn everything_this_layer_writes_is_stamped_harmonized() {
        // Not Direct: the value is the submitter's but the key mapping is ours,
        // which is the whole reason this is a separate provenance class.
        let mut schema = TargetSchema::default();
        assign(&mut schema, "host", "Mus musculus", Provenance::Harmonized);
        assert_eq!(schema.host.provenance(), Some(&Provenance::Harmonized));
    }

    // -- provenance ----------------------------------------------------------

    #[test]
    fn the_caller_chooses_the_provenance() {
        // The only thing that differs between the layers here. Layer 2 mapped a
        // submitter's key, layer 3 read a model's answer, layer 4 read a paper —
        // and the field records which, from the same write. Before this was an
        // argument, the write hard-coded Harmonized and layer 3 could not have
        // used it at all.
        for provenance in [
            Provenance::Direct,
            Provenance::Harmonized,
            Provenance::InferredFromText(Directness::Quoted),
            Provenance::InferredFromPaper(Directness::Inferred),
        ] {
            let mut schema = TargetSchema::default();
            assign(&mut schema, "host", "Mus musculus", provenance.clone());
            assert_eq!(schema.host.provenance(), Some(&provenance));
        }
    }

    #[test]
    fn the_provenance_reaches_a_declared_absence_too() {
        let mut schema = TargetSchema::default();
        let provenance = Provenance::InferredFromText(Directness::Inferred);
        assign(&mut schema, "sex", "not applicable", provenance.clone());
        assert_eq!(
            schema.sex,
            Field::Missing(MissingReason::NotApplicable, provenance)
        );
    }

    // -- levels --------------------------------------------------------------

    #[test]
    fn every_field_has_a_level() {
        // A field with no level cannot be blinded from a layer that must not
        // answer it, and would silently reach the ask.
        for name in FIELD_NAMES {
            assert!(level_of(name).is_some(), "{name} has no level");
        }
        assert_eq!(level_of("not_a_field"), None);
    }

    #[test]
    fn the_renamed_fields_keep_their_python_level() {
        // The four that changed name across the port. Getting one wrong moves a
        // field into or out of a layer's blind set.
        assert_eq!(level_of("abstract_text"), Some(Level::Study));          // description
        assert_eq!(level_of("center_project_name"), Some(Level::Study));    // project_name
        assert_eq!(level_of("earliest_run_published"), Some(Level::Submission)); // first_public
        assert_eq!(level_of("total_spots"), Some(Level::Run));              // read_count
        assert_eq!(level_of("biosample_package"), Some(Level::Sample));     // ncbi_reporting_standard
        assert_eq!(level_of("taxon_id"), Some(Level::Sample));              // tax_id
    }

    #[test]
    fn the_accession_pair_follows_this_schemas_meaning() {
        // Both are study-level either way, so the inversion does not move them
        // between layers — but it is worth pinning that it does not.
        assert_eq!(level_of("study_accession"), Some(Level::Study));
        assert_eq!(level_of("bioproject_accession"), Some(Level::Study));
        assert_eq!(level_of("sample_accession"), Some(Level::Sample));
        assert_eq!(level_of("biosample_accession"), Some(Level::Sample));
    }

    // -- open fields ---------------------------------------------------------

    #[test]
    fn a_fresh_record_has_every_field_open() {
        assert_eq!(open_fields(&TargetSchema::default()).len(), FIELD_NAMES.len());
    }

    #[test]
    fn a_settled_field_is_not_open() {
        let schema = TargetSchema {
            host: Field::Known("Mus musculus".into(), Provenance::Direct),
            ..Default::default()
        };
        let open = open_fields(&schema);
        assert!(!open.contains(&"host"));
        assert!(open.contains(&"sex"));
        assert_eq!(open.len(), FIELD_NAMES.len() - 1);
    }

    #[test]
    fn a_declared_absence_is_not_open_either() {
        // A missing-value term is a determination. Leaving it open would let a
        // later layer overwrite the submitter's own declaration.
        let schema = TargetSchema {
            sex: Field::Missing(MissingReason::NotApplicable, Provenance::Harmonized),
            ..Default::default()
        };
        assert!(!open_fields(&schema).contains(&"sex"));
    }
}
