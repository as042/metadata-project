use std::collections::BTreeMap;

use crate::project::*;
use super::fields::{assign, is_schema_field, FIELD_NAMES};
use crate::target_schema::{Provenance, TargetSchema};

// Layer 2 — map submitter attribute keys onto schema fields.
//
// No model, no network: a normalisation pass plus the synonym table. The
// *value* is the submitter's and is trustworthy; only the key mapping is ours,
// which is exactly what `Harmonized` provenance records and why these carry no
// Directness — nobody chose the value, so there is nothing to attribute.
//
// Running this before the LLM layers pays twice. It fills fields the model was
// getting wrong by hand (a measured 24 of 60 records had the study organism
// written into `host` because the model was harmonising the bag itself), and
// every field it fills is closed to the model afterwards — fewer questions, on
// a smaller schema, for fewer output tokens.
//
// Reads every attribute bag on the record, in `BAG_PRIORITY` order. Python
// reads only the SRA one; this is a deliberate divergence, worth 27,053 fields
// the SRA bag cannot supply on its own.

// The attribute bags this layer reads, in the order they win.
//
// Precedence needs no machinery of its own: `assign` never overwrites a settled
// field, so whichever bag reaches a field first closes it against every bag
// after it. Ordering this list *is* the precedence rule.
//
// The order is deliberate but provisional — change the list, not the loop.
//
//   SraSample            The submitter's own bag on the SRA record. First
//                        because it is the one the archive serves alongside the
//                        experiment, and the one Python has always used.
//   BioSample            The submitter's own bag on the BioSample record. Same
//                        author, different registry. Overlaps the SRA bag
//                        heavily; where both carry a field they disagree 433
//                        times corpus-wide, which is what makes this an
//                        ordering decision rather than a free merge.
//   BioSampleHarmonized  NCBI's `harmonized_name` view of the BioSample bag.
//                        Last because the values are the submitter's but the
//                        key mapping is NCBI's — second-hand in exactly the way
//                        this layer's own mapping is, so it should not outrank
//                        a key someone actually typed.
//   Experiment           EXPERIMENT_ATTRIBUTES. Contributes nothing today: all
//                        four keys in the corpus are identifiers or status
//                        flags (`GEO Accession`, `ENA-STATUS`, `ENA-STATUS-ID`,
//                        `GNB`) and none resolves to a field. Wired anyway so a
//                        real attribute appearing there is not silently missed.
pub const BAG_PRIORITY: &[Bag] = &[
    Bag::SraSample,
    Bag::BioSample,
    Bag::BioSampleHarmonized,
    Bag::Experiment,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bag {
    SraSample,
    BioSample,
    BioSampleHarmonized,
    Experiment,
}

impl Bag {
    // None when the record carries no such bag at all, which is different from
    // carrying an empty one only in that neither yields anything.
    #[inline]
    fn of<'a>(
        &self,
        sample: Option<&'a Sample>,
        experiment: Option<&'a Experiment>,
    ) -> Option<&'a BTreeMap<String, String>> {
        match self {
            Bag::SraSample => sample.map(|s| &s.attributes),
            Bag::BioSample => sample.map(|s| &s.biosample_attributes),
            Bag::BioSampleHarmonized => {
                sample.and_then(|s| s.biosample.as_ref()).map(|b| &b.harmonized)
            }
            Bag::Experiment => experiment.map(|e| &e.attributes),
        }
    }
}

// Submitter attribute key (normalised) -> schema field. Only keys whose
// normalised form differs from the target field name need an entry; everything
// else — `isolation_source`, `host`, `sex`, `strain`, `collected_by`,
// `collection_date`, `host_sex`, `checklist`, `cell type` -> `cell_type` — is
// matched directly against the schema and needs no row.
//
// Ordered by how often the key actually appears, measured across a 40-study
// sample: geo_loc_name 22, collection_date 21, isolation_source 18, host 15,
// tissue 14, biome 9, the env_* triad 3 each.
//
// The first block is Python's `_SYNONYMS` verbatim, retargeted where this
// schema renamed the destination field. The second block exists only because of
// those renames — see RENAME BRIDGES below.
const SYNONYMS: &[(&str, &str)] = &[
    ("geo_loc_name", "country"),
    ("geographic_location_country_and_or_sea", "country"),
    ("tissue", "tissue_type"),
    ("biome", "environment_biome"),
    ("feature", "environment_feature"),
    ("material", "environment_material"),
    ("env_broad_scale", "broad_scale_environmental_context"),
    ("env_local_scale", "local_environmental_context"),
    ("env_medium", "environmental_medium"),
    ("env_biome", "environment_biome"),
    ("env_feature", "environment_feature"),
    ("env_material", "environment_material"),
    // Python target `ncbi_reporting_standard`; this schema calls it
    // biosample_package.
    ("biosamplemodel", "biosample_package"),
    ("host_taxid", "host_tax_id"),
    ("specific_host", "host_scientific_name"),
    ("description", "sample_description"),
    ("ena_checklist", "checklist"),
    // `age`, `cell line`, `treatment` and `dev_stage` normalise straight onto
    // their own field names, so only the spelled-out form of dev_stage needs a
    // row.
    ("developmental_stage", "dev_stage"),
    // The LLM layer was already mapping `agent` -> treatment correctly on its
    // own; moving it here makes it free and deterministic instead of a paid
    // guess.
    ("agent", "treatment"),
    ("development_stage", "dev_stage"),
    // -- RENAME BRIDGES ---------------------------------------------------
    // Not new mappings. In Python these keys need no table row because they
    // *are* schema field names and match exactly; this schema renamed the
    // fields, so without a row the same key would stop resolving. Each one
    // restores the Python behaviour rather than adding to it.
    //
    // Corpus hit counts: ncbi_reporting_standard 38,397, project_name 8,109,
    // tax_id 3, and zero for the rest.
    ("ncbi_reporting_standard", "biosample_package"),
    ("project_name", "center_project_name"),
    ("tax_id", "taxon_id"),
    ("instrument_platform", "platform"),
    ("read_count", "total_spots"),
    ("base_count", "total_bases"),
    ("first_public", "earliest_run_published"),
    ("secondary_study_accession", "study_accession"),
    ("secondary_sample_accession", "sample_accession"),
];

// ‼️ ONE DIVERGENCE FROM PYTHON, forced by the rename and not expressible here.
// `sample_accession` and `study_accession` name different columns in the two
// schemas: the BioSample and BioProject in Python, the SRS and SRP here. A
// bridge row cannot restore the Python meaning because both are field names of
// *this* schema, so the exact-match rule resolves them before the table is ever
// consulted. They therefore follow this schema's meaning. Neither key occurs as
// an attribute in the corpus, so no record is affected today.

// Deliberately absent: `source_name`, the single most common unmapped key
// (37,532 occurrences). It has no fixed target — observed values include
// "Fibroblast" (a cell type), "Hypothalamus" (a tissue), "whole worms" (an
// organism) and "liver parenchymal cells" (either). Pinning it to one field
// would be wrong roughly two thirds of the time, which is worse than leaving it
// to a layer that can read the value.

// Casefold an attribute key and collapse its separators.
//
// Does most of the work for free. Submitters write the same key every which way
// — `cell type` / `cell_type`, `strain` / `STRAIN`, `collection date` /
// `collection_date` — and all of those collapse onto one form here, before any
// synonym lookup. A normalised key that *is* a schema field name needs no table
// entry at all, which is why SYNONYMS only carries the genuinely different
// names.
#[inline]
fn normalize_key(key: &str) -> String {
    let mut out = String::with_capacity(key.len());
    let mut pending = false;
    for c in key.trim().chars() {
        if c.is_ascii_alphanumeric() {
            if pending && !out.is_empty() {
                out.push('_');
            }
            pending = false;
            out.extend(c.to_lowercase());
        } else {
            pending = true;
        }
    }
    out
}

// `geo_loc_name` is INSDC's `Country:Region` — take the country.
//
// The one value transform in this layer, and the reason `Harmonized` is its own
// provenance class rather than folding into `Direct`. "China:Beijing" dropped
// whole into `country` is simply wrong, and the split is a decision of ours
// that could be wrong in its own way (a submitter who writes only a region, or
// uses a different separator, gets mangled).
#[inline]
fn country_from_geo_loc(value: &str) -> Option<&str> {
    let country = value.split(':').next().unwrap_or("").trim();
    (!country.is_empty()).then_some(country)
}

// `{schema field: value}` for every attribute key this layer recognises.
//
// An exact (normalised) hit on a schema field name wins over a synonym, so a
// bag carrying both `tissue_type` and `tissue` keeps the submitter's own
// `tissue_type`. Blank values are skipped — the harvest records a TAG with no
// VALUE as an empty string, which is "present but empty", not an answer.
#[inline]
fn harmonized(attributes: &BTreeMap<String, String>) -> BTreeMap<&'static str, String> {
    let mut out: BTreeMap<&'static str, String> = BTreeMap::new();

    for (raw_key, value) in attributes {
        if value.trim().is_empty() {
            continue;
        }
        let key = normalize_key(raw_key);

        // An exact field-name match beats the table.
        let exact = is_schema_field(&key);
        let Some(target) = (if exact {
            FIELD_NAMES.iter().find(|f| **f == key).copied()
        } else {
            SYNONYMS.iter().find(|(k, _)| *k == key).map(|(_, t)| *t)
        }) else {
            continue;
        };

        let value = value.trim();
        let transformed = match target {
            "country" => country_from_geo_loc(value),
            _ => Some(value),
        };
        let Some(transformed) = transformed else { continue };

        // An exact key match beats a synonym that already claimed the field.
        // Measured on the corpus this decides 12 samples, all of them by the
        // exact-match rule; no collision is settled by iteration order, so
        // reading a sorted map here cannot diverge from Python's insertion
        // order.
        if out.contains_key(target) && !exact {
            continue;
        }
        out.insert(target, transformed.to_string());
    }
    out
}

// Fills open fields on records the direct layer already created.
//
// Each record is matched back to its sample through `sample_accession` and to
// its experiment through `experiment_accession`, both of which only the direct
// layer sets — so a schema list built without it is left untouched. A record
// whose sample or experiment does not resolve simply reads no bag from that
// side rather than being guessed at.
#[inline]
pub(crate) fn process(project: &Project, schemas: &mut [TargetSchema]) {
    // Built once per project rather than scanned per record: the largest study
    // in the corpus has 1,649 experiments, and a linear search inside the record
    // loop would make this quadratic for no reason.
    let experiments: BTreeMap<&ExperimentAccession, &Experiment> =
        project.experiments.iter().map(|e| (&e.accession, e)).collect();

    for schema in schemas.iter_mut() {
        let sample = schema
            .sample_accession
            .value()
            .and_then(|accession| project.samples.get(accession));
        let experiment = schema
            .experiment_accession
            .value()
            .and_then(|accession| experiments.get(accession).copied());

        for bag in BAG_PRIORITY {
            let Some(attributes) = bag.of(sample, experiment) else {
                continue;
            };
            if attributes.is_empty() {
                continue;
            }
            for (target, value) in harmonized(attributes) {
                assign(schema, target, &value, Provenance::Harmonized);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::target_schema::{Field, MissingReason};

    fn bag(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
        pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
    }

    fn only(pairs: &[(&str, &str)]) -> Vec<(&'static str, String)> {
        harmonized(&bag(pairs)).into_iter().collect()
    }

    // -- key normalisation -------------------------------------------------

    #[test]
    fn normalisation_collapses_the_ways_submitters_write_a_key() {
        for raw in ["cell type", "Cell Type", "cell_type", "CELL-TYPE", "  cell   type  "] {
            assert_eq!(normalize_key(raw), "cell_type", "{raw:?}");
        }
        assert_eq!(normalize_key("STRAIN"), "strain");
        assert_eq!(normalize_key("geo_loc_name"), "geo_loc_name");
        // leading and trailing separators do not become underscores
        assert_eq!(normalize_key("__host__"), "host");
        assert_eq!(normalize_key("---"), "");
    }

    #[test]
    fn a_normalised_key_that_is_a_field_name_needs_no_table_row() {
        // The reason the table only carries genuinely different names.
        assert_eq!(only(&[("Isolation Source", "soil")]), [("isolation_source", "soil".into())]);
        assert_eq!(only(&[("SEX", "female")]), [("sex", "female".into())]);
    }

    // -- the table ---------------------------------------------------------

    #[test]
    fn synonyms_resolve_to_their_target_field() {
        assert_eq!(only(&[("tissue", "liver")]), [("tissue_type", "liver".into())]);
        assert_eq!(only(&[("biome", "marine")]), [("environment_biome", "marine".into())]);
        assert_eq!(only(&[("specific_host", "Apis mellifera")]),
                   [("host_scientific_name", "Apis mellifera".into())]);
        assert_eq!(only(&[("developmental_stage", "larva")]), [("dev_stage", "larva".into())]);
        assert_eq!(only(&[("agent", "doxorubicin")]), [("treatment", "doxorubicin".into())]);
        assert_eq!(only(&[("ena_checklist", "ERC000011")]), [("checklist", "ERC000011".into())]);
    }

    #[test]
    fn rename_bridges_keep_python_keys_resolving() {
        // These need no row in Python because they are field names there. The
        // rows exist only because this schema renamed the destination.
        assert_eq!(only(&[("ncbi_reporting_standard", "MIMS.me")]),
                   [("biosample_package", "MIMS.me".into())]);
        assert_eq!(only(&[("project_name", "FOX-2019")]),
                   [("center_project_name", "FOX-2019".into())]);
        assert_eq!(only(&[("tax_id", "9606")]), [("taxon_id", "9606".into())]);
    }

    #[test]
    fn the_inverted_accession_keys_follow_this_schemas_meaning() {
        // The one divergence from Python. Both names are fields of this schema,
        // so the exact-match rule resolves them before the table is consulted
        // and no bridge row can redirect them. In Python the same keys land on
        // the BioSample and BioProject columns. Neither occurs in the corpus.
        assert_eq!(only(&[("sample_accession", "SRS000001")]),
                   [("sample_accession", "SRS000001".into())]);
        assert_eq!(only(&[("study_accession", "SRP000001")]),
                   [("study_accession", "SRP000001".into())]);
        assert_eq!(only(&[("secondary_sample_accession", "SRS000001")]),
                   [("sample_accession", "SRS000001".into())]);
    }

    #[test]
    fn source_name_is_deliberately_unmapped() {
        // 37,532 occurrences and no fixed target: the values are cell types,
        // tissues and organisms in roughly equal measure.
        assert!(only(&[("source_name", "Fibroblast")]).is_empty());
        assert!(only(&[("isolate", "strain-7")]).is_empty());
        assert!(only(&[("lat_lon", "35.6 N 139.7 E")]).is_empty());
    }

    // -- values ------------------------------------------------------------

    #[test]
    fn blank_values_are_skipped() {
        // A TAG with no VALUE is present-but-empty, not an answer.
        assert!(only(&[("host", "")]).is_empty());
        assert!(only(&[("host", "   ")]).is_empty());
    }

    #[test]
    fn values_are_trimmed() {
        assert_eq!(only(&[("host", "  Mus musculus  ")]), [("host", "Mus musculus".into())]);
    }

    #[test]
    fn geo_loc_name_keeps_only_the_country() {
        // The one value transform, and the reason this layer has its own
        // provenance rather than folding into Direct.
        assert_eq!(only(&[("geo_loc_name", "China:Beijing")]), [("country", "China".into())]);
        assert_eq!(only(&[("geo_loc_name", "  Japan : Tokyo ")]), [("country", "Japan".into())]);
        assert_eq!(only(&[("geo_loc_name", "Brazil")]), [("country", "Brazil".into())]);
        // a leading separator leaves no country, so nothing is stored
        assert!(only(&[("geo_loc_name", ":Beijing")]).is_empty());
    }

    // -- precedence --------------------------------------------------------

    #[test]
    fn an_exact_field_name_beats_a_synonym_for_the_same_field() {
        let out = only(&[("tissue", "liver"), ("tissue_type", "hepatocyte")]);
        assert_eq!(out, [("tissue_type", "hepatocyte".into())]);
    }

    #[test]
    fn a_synonym_cannot_displace_a_claim_already_made() {
        // Two synonyms for one field: the first to claim it keeps it. Measured
        // on the corpus this decides 12 samples, every one of them by the
        // exact-match rule above rather than by ordering.
        let out = only(&[("biome", "marine"), ("env_biome", "freshwater")]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, "environment_biome");
    }

    // -- missing values through the country transform -----------------------

    #[test]
    fn geo_loc_name_missing_terms_survive_the_country_split() {
        // The transform runs first and takes everything before the colon, so
        // the term still has to be recognised afterwards.
        let mut schema = TargetSchema::default();
        let out = harmonized(&bag(&[("geo_loc_name", "missing")]));
        assert_eq!(out.get("country").map(String::as_str), Some("missing"));
        assign(&mut schema, "country", &out["country"], Provenance::Harmonized);
        assert_eq!(schema.country,
                   Field::Missing(MissingReason::Unspecified, Provenance::Harmonized));
    }

    // -- process -----------------------------------------------------------

    fn project_with(attributes: &[(&str, &str)]) -> Project {
        let sample = Sample {
            accession: SampleAccession("SRS000001".into()),
            attributes: bag(attributes),
            ..Default::default()
        };
        Project {
            accession: StudyAccession("SRP000001".into()),
            samples: [(sample.accession.clone(), sample)].into_iter().collect(),
            experiments: vec![Experiment {
                accession: ExperimentAccession("SRX000001".into()),
                sample_ids: vec![SampleAccession("SRS000001".into())],
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    #[test]
    fn process_fills_records_the_direct_layer_built() {
        let project = project_with(&[
            ("geo_loc_name", "China:Beijing"),
            ("Collection Date", "2019-05-04"),
            ("tissue", "liver"),
            ("source_name", "Fibroblast"),
        ]);
        let mut schemas = Vec::new();
        crate::layer::direct::process(&project, &mut schemas);
        process(&project, &mut schemas);

        assert_eq!(schemas.len(), 1);
        let s = &schemas[0];
        assert_eq!(s.country, Field::Known("China".into(), Provenance::Harmonized));
        assert_eq!(s.tissue_type, Field::Known("liver".into(), Provenance::Harmonized));
        assert!(s.collection_date.is_settled());
        // unmapped, and left for a layer that can read the value
        assert_eq!(s.cell_type, Field::Unknown);
    }

    #[test]
    fn process_leaves_direct_values_alone() {
        // `scientific_name` is set by the direct layer from the sample record;
        // an attribute of the same name must not overwrite it.
        let mut project = project_with(&[("scientific_name", "Homo sapiens")]);
        project.samples.get_mut(&SampleAccession("SRS000001".into())).unwrap()
            .scientific_name = Some("Mus musculus".into());

        let mut schemas = Vec::new();
        crate::layer::direct::process(&project, &mut schemas);
        process(&project, &mut schemas);
        assert_eq!(
            schemas[0].scientific_name,
            Field::Known("Mus musculus".into(), Provenance::Direct)
        );
    }

    #[test]
    fn process_without_the_direct_layer_does_nothing() {
        // No records exist to fill, and the sample is found through
        // `sample_accession`, which only the direct layer sets.
        let project = project_with(&[("host", "Mus musculus")]);
        let mut schemas = Vec::new();
        process(&project, &mut schemas);
        assert!(schemas.is_empty());

        // A record with no sample is skipped rather than guessed at.
        let mut orphan = vec![TargetSchema::default()];
        process(&project, &mut orphan);
        assert_eq!(orphan[0].host, Field::Unknown);
    }

    #[test]
    fn process_skips_a_sample_the_project_does_not_carry() {
        let project = project_with(&[("host", "Mus musculus")]);
        let mut schemas = vec![TargetSchema {
            sample_accession: Field::Known(SampleAccession("SRS999999".into()), Provenance::Direct),
            ..Default::default()
        }];
        process(&project, &mut schemas);
        assert_eq!(schemas[0].host, Field::Unknown);
    }

    #[test]
    fn process_is_idempotent() {
        // Running it twice must change nothing: every field it filled is now
        // settled, so the second pass is all Closed.
        let project = project_with(&[("geo_loc_name", "China:Beijing"), ("tissue", "liver")]);
        let mut once = Vec::new();
        crate::layer::direct::process(&project, &mut once);
        process(&project, &mut once);
        let mut twice = once.clone();
        process(&project, &mut twice);
        assert_eq!(once, twice);
    }

    // -- bag priority -------------------------------------------------------

    fn project_with_bags(
        sra: &[(&str, &str)],
        biosample: &[(&str, &str)],
        harmonized_view: &[(&str, &str)],
        experiment_attrs: &[(&str, &str)],
    ) -> Project {
        let sample = Sample {
            accession: SampleAccession("SRS000001".into()),
            attributes: bag(sra),
            biosample_attributes: bag(biosample),
            biosample: Some(BioSample {
                harmonized: bag(harmonized_view),
                ..Default::default()
            }),
            ..Default::default()
        };
        Project {
            accession: StudyAccession("SRP000001".into()),
            samples: [(sample.accession.clone(), sample)].into_iter().collect(),
            experiments: vec![Experiment {
                accession: ExperimentAccession("SRX000001".into()),
                sample_ids: vec![SampleAccession("SRS000001".into())],
                attributes: bag(experiment_attrs),
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    fn built(project: &Project) -> Vec<TargetSchema> {
        let mut schemas = Vec::new();
        crate::layer::direct::process(project, &mut schemas);
        process(project, &mut schemas);
        schemas
    }

    #[test]
    fn the_sra_bag_outranks_the_biosample_bags() {
        // 433 fields corpus-wide are carried by both bags with different values,
        // so this order decides real records rather than a hypothetical.
        let project = project_with_bags(
            &[("treatment", "from SRA")],
            &[("treatment", "from BioSample")],
            &[("treatment", "from the harmonized view")],
            &[],
        );
        assert_eq!(
            built(&project)[0].treatment,
            Field::Known("from SRA".into(), Provenance::Harmonized)
        );
    }

    #[test]
    fn the_biosample_bag_outranks_the_harmonized_view() {
        // NCBI's view carries the submitter's values under NCBI's key mapping,
        // which is second-hand in the way this layer's own mapping is.
        let project = project_with_bags(
            &[],
            &[("treatment", "from BioSample")],
            &[("treatment", "from the harmonized view")],
            &[],
        );
        assert_eq!(
            built(&project)[0].treatment,
            Field::Known("from BioSample".into(), Provenance::Harmonized)
        );
    }

    #[test]
    fn a_later_bag_fills_what_an_earlier_one_leaves_open() {
        // The point of reading them all: 27,053 fields corpus-wide come from a
        // bag the SRA one cannot supply.
        let project = project_with_bags(
            &[("host", "Mus musculus")],
            &[("sex", "female")],
            &[("tissue", "liver")],
            &[],
        );
        let s = &built(&project)[0];
        assert_eq!(s.host, Field::Known("Mus musculus".into(), Provenance::Harmonized));
        assert_eq!(s.sex, Field::Known("female".into(), Provenance::Harmonized));
        assert_eq!(s.tissue_type, Field::Known("liver".into(), Provenance::Harmonized));
    }

    #[test]
    fn ena_checklist_now_resolves() {
        // The single largest gain, and previously dead: `ENA-CHECKLIST` occurs
        // 9,996 times and lives *only* in the BioSample bag, so the synonym row
        // for it could never fire while this layer read the SRA bag alone.
        let project = project_with_bags(&[], &[("ENA-CHECKLIST", "ERC000011")], &[], &[]);
        assert_eq!(
            built(&project)[0].checklist,
            Field::Known("ERC000011".into(), Provenance::Harmonized)
        );
    }

    #[test]
    fn the_experiment_bag_is_read_even_though_it_yields_nothing_today() {
        // All four corpus keys there are identifiers or status flags. This uses
        // a key that does resolve, so the wiring is proven rather than assumed —
        // otherwise the variant would look connected while being unreachable.
        let project = project_with_bags(&[], &[], &[], &[("cell type", "hepatocyte")]);
        assert_eq!(
            built(&project)[0].cell_type,
            Field::Known("hepatocyte".into(), Provenance::Harmonized)
        );
    }

    #[test]
    fn the_real_experiment_attribute_keys_still_resolve_to_nothing() {
        let project = project_with_bags(
            &[],
            &[],
            &[],
            &[("GEO Accession", "GSM1128675"), ("ENA-STATUS", "PUBLIC"),
              ("ENA-STATUS-ID", "4"), ("GNB", "G26287")],
        );
        let s = &built(&project)[0];
        assert_eq!(s.tag, Field::Unknown);
        assert_eq!(s.sample_description, Field::Unknown);
    }

    #[test]
    fn a_direct_value_still_beats_every_bag() {
        // Bag priority orders the bags against each other, not against layer 1.
        let mut project = project_with_bags(&[("scientific_name", "from SRA")], &[], &[], &[]);
        project.samples.get_mut(&SampleAccession("SRS000001".into())).unwrap()
            .scientific_name = Some("Mus musculus".into());
        assert_eq!(
            built(&project)[0].scientific_name,
            Field::Known("Mus musculus".into(), Provenance::Direct)
        );
    }

    #[test]
    fn bag_priority_lists_every_variant_once() {
        // The list is the precedence rule, so a variant missing from it is a
        // bag that is silently never read.
        let all = [Bag::SraSample, Bag::BioSample, Bag::BioSampleHarmonized, Bag::Experiment];
        assert_eq!(BAG_PRIORITY.len(), all.len());
        for bag in all {
            assert!(BAG_PRIORITY.contains(&bag), "{bag:?} is never read");
        }
        assert_eq!(BAG_PRIORITY[0], Bag::SraSample, "the SRA bag must stay first");
    }

    #[test]
    fn a_record_with_no_sample_still_reads_its_experiment_bag() {
        // The two lookups are independent; one failing must not suppress the other.
        let project = project_with_bags(&[], &[], &[], &[("cell type", "hepatocyte")]);
        let mut schemas = vec![TargetSchema {
            experiment_accession: Field::Known(
                ExperimentAccession("SRX000001".into()), Provenance::Direct),
            sample_accession: Field::Known(
                SampleAccession("SRS999999".into()), Provenance::Direct),
            ..Default::default()
        }];
        process(&project, &mut schemas);
        assert_eq!(
            schemas[0].cell_type,
            Field::Known("hepatocyte".into(), Provenance::Harmonized)
        );
    }

    // -- drift -------------------------------------------------------------

    #[test]
    fn every_field_name_is_assignable_and_every_target_exists() {
        // FIELD_NAMES is a second list and could drift from the `assign` match.
        // This is what catches it.
        for name in FIELD_NAMES {
            assert!(is_schema_field(name), "FIELD_NAMES has {name:?}, assign does not");
        }
        for (key, target) in SYNONYMS {
            assert!(
                is_schema_field(target),
                "synonym {key:?} points at {target:?}, which is not a field"
            );
            assert_eq!(normalize_key(key), *key, "synonym key {key:?} is not normalised");
        }
    }

    #[test]
    fn no_synonym_key_shadows_a_schema_field_name() {
        // A row whose key is itself a field name can never fire, because the
        // exact match wins first. That covers both kinds: a row redirecting
        // elsewhere (harmful — this is the `description` bug Python had) and an
        // identity row (harmless, but dead weight the exact match already
        // handles). Neither is allowed.
        for (key, _target) in SYNONYMS {
            assert!(
                !is_schema_field(key),
                "synonym {key:?} is also a field name, so the row can never fire"
            );
        }
    }
}
