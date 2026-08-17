"""The layered reconstruction cascade.

Four layers fill a :class:`schema.TargetSchema`, cheapest technology first, each
one seeing only what the layers before it left open:

    direct  ->  harmonized  ->  inferred_from_text  ->  inferred_from_paper

The layer names are the provenance classes on purpose. **Provenance is recorded
by the cascade, not reported by the layer** — a layer returns values, and
:func:`reconstruct` stamps them with the class of whichever layer produced them.
Asking a model to report where it got something would be exactly as unreliable
as asking it how confident it is; this way provenance is a fact about the code
that ran.

Layer 1 is :meth:`schema.TargetSchema.from_project` and is already built: it
fills only what SRA states outright and stamps every value ``direct``. It always
runs. The other three are enabled individually and are **not implemented yet** —
enabling one raises :class:`NotImplementedError` rather than quietly returning
nothing, so an empty result never gets mistaken for "the model found nothing".

    records, report = reconstruct(project)                    # direct only
    records, report = reconstruct(project, from_text=True)    # once layer 3 exists

Writing a layer
---------------
A layer is ``fn(project, records, open_by_id) -> {record_id: {field: Proposal}}``.
It receives the whole batch rather than one record at a time, because the useful
unit of work is not the record:

* **Study-level fields** — one call per *study*. Every record shares them.
* **Biological fields** — one call per *sample*, then applied to that sample's
  records. Records are grouped by ``secondary_sample_accession``.

Asking per record would pay for the same answer once per experiment.

A layer may only fill fields listed in ``open_by_id``; anything else is a bug and
:func:`reconstruct` raises rather than dropping it silently.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import claude
import project as project_module
from schema import (CONFIDENCE_LEVELS, MISSING_VALUES, RECORD, RUN,
                    STUDY, SUBMISSION, TargetSchema)


@dataclass
class Proposal:
    """One layer's answer for one field.

    ``confidence`` is one of :data:`schema.CONFIDENCE_LEVELS`, and only the two
    ``inferred_*`` layers should set it — nobody *chose* a direct or harmonized
    value, so there is nothing to be confident about. :func:`reconstruct` refuses
    a confidence from the other layers rather than writing one that
    :meth:`schema.TargetSchema.inconsistent_confidence` would immediately flag.

    ``value`` may be a :data:`schema.MISSING_VALUES` term — "not applicable",
    "not collected" — which is an answer like any other and carries confidence
    like any other. It is not the same as declining to answer: to leave a field
    for the next layer, simply omit it from the returned mapping.
    """

    value: Any
    confidence: str | None = None


# Provenance classes whose missing-value verdicts a later layer may revisit.
# Deliberately excludes `harmonized`: that term came from the submitter's own
# attribute bag, so it is *data* rather than a gap, and a later layer inferring
# over it is a regression however confident it sounds.
_REVISABLE_VERDICTS = ("inferred_from_text", "inferred_from_paper")


def open_fields(record: TargetSchema, overwrite_missing: bool = True) -> set[str]:
    """The fields a later layer is still allowed to fill.

    Always excludes ``id`` (synthetic) and any field already holding an ordinary
    value. The live question is the middle case, fields holding a
    :data:`schema.MISSING_VALUES` term, and the answer turns on *who declared it*.

    ``overwrite_missing=True`` (default) reopens a term a **model** wrote.
    "not collected" from the text layer means *not collected as far as the
    attributes and abstract show*, and the paper is the very thing that might
    say otherwise, so an early missing-verdict should not permanently block the
    layer most likely to overturn it.

    A term the **harmonization** layer wrote is never reopened, whatever
    ``overwrite_missing`` says. That value is the submitter's own declaration —
    ``geo_loc_name: "not provided"`` means the depositor stated the location is
    unavailable, which is a fact about the record and not a hole in it.

    Measured on 1,664 GenomeTrakr records: 380 samples declared
    ``geo_loc_name: "not provided"``, layer 2 recorded that faithfully, the old
    rule reopened all 380, and layer 3 overwrote them with a country guessed
    from the submitting lab's address — ``Brazil`` x167, ``Thailand`` x58,
    ``Senegal`` x39 (from "Institut Pasteur de Dakar"). 369 of 380 wrong. The
    same mechanism damaged 103 ``host`` fields. One rule, 472 records.

    ``overwrite_missing=False`` additionally treats a model's own verdict as
    final. Cheaper, and defensible if you trust the earlier layer, but it caps
    what the paper layer can ever contribute.
    """
    out = set()
    for name in record.field_names():
        if name == "id":
            continue
        value = getattr(record, name)
        if value is None:
            out.add(name)
        elif (
            overwrite_missing
            and value in MISSING_VALUES
            and record.provenance.get(name) in _REVISABLE_VERDICTS
        ):
            out.add(name)
    return out


# --------------------------------------------------------------------------- #
# Layer 2 — harmonization
# --------------------------------------------------------------------------- #
def _normalize_key(key: str) -> str:
    """Casefold an attribute key and collapse its separators.

    Does most of the work for free. Submitters write the same key every which
    way — ``cell type`` / ``cell_type``, ``strain`` / ``STRAIN``,
    ``collection date`` / ``collection_date`` — and all of those collapse onto
    one form here, before any synonym lookup. A normalized key that *is* a
    schema field name needs no table entry at all, which is why :data:`_SYNONYMS`
    below only has to carry the genuinely different names.
    """
    return re.sub(r"[^a-z0-9]+", "_", key.strip().lower()).strip("_")


def _country_from_geo_loc(value: str) -> str | None:
    """``geo_loc_name`` is INSDC's ``Country:Region`` — take the country.

    The one value transform in this layer, and the reason `harmonized` is its
    own provenance class rather than being folded into `direct`. "China:Beijing"
    dropped whole into ``country`` is simply wrong, and the split is a decision
    of ours that could be wrong in its own way (a submitter who writes only a
    region, or uses a different separator, gets mangled).
    """
    return value.split(":")[0].strip() or None


# Submitter attribute key (normalized) -> schema field. Only keys whose
# normalized form differs from the target field name need an entry; everything
# else — `isolation_source`, `host`, `sex`, `strain`, `collected_by`,
# `collection_date`, `host_sex`, `checklist`, `cell type` -> `cell_type` — is
# matched directly against the schema and needs no table row.
#
# Ordered by how often the key actually appears, measured across a 40-study
# sample: geo_loc_name 22, collection_date 21, isolation_source 18, host 15,
# tissue 14, biome 9, the env_* triad 3 each.
_SYNONYMS: dict[str, str] = {
    "geo_loc_name": "country",
    "geographic_location_country_and_or_sea": "country",
    "tissue": "tissue_type",
    "biome": "environment_biome",
    "feature": "environment_feature",
    "material": "environment_material",
    "env_broad_scale": "broad_scale_environmental_context",
    "env_local_scale": "local_environmental_context",
    "env_medium": "environmental_medium",
    "env_biome": "environment_biome",
    "env_feature": "environment_feature",
    "env_material": "environment_material",
    "biosamplemodel": "ncbi_reporting_standard",
    "host_taxid": "host_tax_id",
    "specific_host": "host_scientific_name",
    "description": "sample_description",
    "ena_checklist": "checklist",
    # `age`, `cell line`, `treatment` and `dev_stage` normalize straight onto
    # their own field names now that those exist; only the spelled-out form of
    # dev_stage needs a row.
    "developmental_stage": "dev_stage",
    # layer 3 was already mapping `agent` -> treatment correctly on its own;
    # moving it here makes it free and deterministic instead of a paid guess.
    "agent": "treatment",
    "development_stage": "dev_stage",
}

_TRANSFORMS = {"country": _country_from_geo_loc}

# Synonym rows whose key is *itself* a schema field name. These need calling out
# because the exact-name match in :func:`_harmonized` resolves first, so without
# this the row could never fire — the table would carry a mapping that silently
# did nothing.
#
# That is what happened to ``description``. It is a submitter's *sample*
# attribute, but ``description`` is also the name of the STUDY-level field (the
# abstract), so the exact match won and 87 sample descriptions across the corpus
# were being written into the study abstract instead of ``sample_description``.
#
# Derived from the table rather than listed by hand, so a future row cannot
# reintroduce the same silent shadowing. Identity rows (key == target) are
# excluded, which is what makes them fail the drift test rather than pass it
# unnoticed: such a row is unreachable dead weight, since the exact-name match
# already routes the key to the same place.
#
# Note this is deliberately *not* a blanket "a sample attribute may not fill a
# STUDY-level field" rule. ``project_name`` is STUDY-level too and is legitimately
# filled from the sample bag 8,109 times; the problem is only a key that collides
# with a *different* field's name.
_SHADOWED_SYNONYMS = {
    key
    for key, target in _SYNONYMS.items()
    if key != target and key in set(TargetSchema.field_names())
}

# Deliberately absent from _SYNONYMS: `source_name`, the single most common
# unmapped key (60 of 175 occurrences). It has no fixed target — observed values
# include "Fibroblast" (a cell type), "Hypothalamus" (a tissue), "whole worms"
# (an organism) and "liver parenchymal cells" (either). Pinning it to one field
# would be wrong roughly two thirds of the time, which is worse than leaving it
# to a layer that can read the value. TEXT_SYSTEM tells layer 3 how to route it.


def _harmonized(attributes: dict[str, str]) -> dict[str, str]:
    """``{schema field: value}`` for every attribute key this layer recognises.

    An exact (normalized) hit on a schema field name wins over a synonym, so a
    bag carrying both ``tissue_type`` and ``tissue`` keeps the submitter's own
    ``tissue_type``. The exception is :data:`_SHADOWED_SYNONYMS` — a key that is
    itself a field name *and* the table maps elsewhere, where deferring to the
    field name would make the row unreachable. Blank values are skipped —
    :func:`project._attrs_bag` records a TAG with no VALUE as ``None``, which is
    "present but empty", not an answer.
    """
    fields = set(TargetSchema.field_names())
    out: dict[str, str] = {}
    for raw_key, value in attributes.items():
        if not value or not str(value).strip():
            continue
        key = _normalize_key(raw_key)
        if key in _SHADOWED_SYNONYMS:
            target = _SYNONYMS[key]
        else:
            target = key if key in fields else _SYNONYMS.get(key)
        if target is None or target == "id":
            continue
        transformed = _TRANSFORMS.get(target, lambda v: v)(str(value).strip())
        if not transformed:
            continue
        # an exact key match beats a synonym that already claimed the field
        if target in out and _normalize_key(raw_key) != target:
            continue
        out[target] = transformed
    return out


def harmonize_attributes(project, records, open_by_id):
    """Layer 2 — map submitter attribute keys onto schema fields.

    No model, no network: a normalization pass plus :data:`_SYNONYMS`. The
    *value* is the submitter's and is trustworthy; only the key mapping is ours,
    which is exactly what `harmonized` provenance records and why these carry no
    confidence — nobody chose the value, so there is nothing to be confident
    about.

    Running this before layer 3 pays twice. It fills the fields the model was
    getting wrong by hand (a measured 24/60 records had the study organism
    written into ``host`` because the model was harmonizing the bag itself), and
    every field it fills is *closed* to layer 3 — so the model is asked fewer
    questions, on a smaller schema, for fewer output tokens.
    """
    proposals: dict[str, dict[str, Proposal]] = {}
    for record in records:
        sample = project.samples.get(record.secondary_sample_accession)
        if sample is None or not sample.attributes:
            continue
        filled = {
            name: Proposal(value)
            for name, value in _harmonized(sample.attributes).items()
            if name in open_by_id[record.id] and _storable(name, value)
        }
        if filled:
            proposals[record.id] = filled
    return proposals


# Deliberately minimal — the field semantics that would make this good live in
# the schema's own docs, not here yet. Expect to rewrite this.
#
# Keep it byte-identical across calls — any per-sample text spliced in here
# breaks caching for every sample in the run.
#
# The FIELD DEFINITIONS block is not padding. Three error classes measured over
# 52 records traced directly to the model not knowing what a field meant:
# `host_scientific_name` filled with the study organism on all 48 records it
# appeared in (all rated `high`), `strain` filled from a `genotype` attribute,
# and `sequencing_method` duplicating `library_construction_protocol`.
#
# The CONFIDENCE block exists because the level was carrying no information:
# 391 high / 41 medium / **0 low** across 432 inferences, with the largest error
# class uniformly `high`. A scale whose bottom rung is never used cannot
# separate right answers from wrong ones.
#
# Still short of Haiku 4.5's 4096-token cache floor (Opus 5: 512, Sonnet 5:
# 1024), so this prefix is re-billed on every call — `cache_control` below the
# floor is a silent no-op.
TEXT_SYSTEM = """You reconstruct sequencing metadata from the evidence supplied.

Answer only from the evidence. Do not fill a field from background knowledge
about the organism or the field of study.

If the evidence does not settle a field, leave it out of your answer entirely.
A later stage will try it. Only include fields you can actually answer.

CONFIDENCE
Every answer carries one of high, medium, or low. This records *where the value
came from*, not how likely you think it is to be right. It describes what you
did, not a belief about the outcome:

  high    you copied it. The value appears in the evidence word for word and you
          could point at the exact span it came from.
  medium  you rephrased it. The evidence carries this value in different words -
          a unit normalised, a phrase tidied, a synonym chosen.
  low     you inferred it. The evidence does not carry the value; you concluded
          it from what is there.

Two rules decide the common edge cases:

  Copied but chosen is medium. If the evidence offered more than one span you
  could have quoted and you picked one, that is a decision - label it medium
  even though your answer is word for word.

  A missing-value term is high only if the evidence says it. If the attributes
  literally read "not collected", that is a quote. If you concluded the field
  does not apply, that is an inference - label it low.

Do not use this to hedge. A value you copied exactly is high even if you suspect
the submitter was wrong: whether the archive is correct is not what this records.

MISSING VALUES
Where one of these applies, answer with the exact string rather than omitting
the field:

  "not applicable"     the field cannot apply to this sample
  "not collected"      nobody measured it
  "not provided"       it exists but was not deposited
  "restricted access"  it exists but is behind controlled access

These are answers, not defaults. Reach for one when the evidence positively
establishes it - a cell line genuinely has no developmental stage, a lab-housed
animal genuinely has no environmental context. Do not work down the field list
assigning "not applicable" to everything you could not fill: where you simply
could not tell, omitting the field is both cheaper and more honest. An omitted
field means "I could not tell"; a missing-value term means "I determined this".

FIELD DEFINITIONS

Study-level. These describe the whole study and are identical for every sample
in it.

  project_name    The short name the submitter gave the BioProject, if any.
                  This is not the study title. A study whose only name is its
                  title has "not provided" here.
  study_alias     The submitter's own identifier for the study - a lab code, an
                  internal accession, a short slug. Not the title.
  description     The study's own description or abstract.
  study_title     The full title of the study.

Sample-level. These describe the biological material and differ between the
samples of a study. They are the fields most worth getting right.

  age             Age of the organism when sampled, in whatever units the
                  evidence uses: "8 weeks", "E14.5", "58 years", "3rd instar".
                  Keep the units. Do not convert.
  cell_line       A named, established cell line: HeLa, NIH3T3, K562, CHO.
                  Freshly isolated primary cells are not a cell line - if the
                  sample is primary material, this is "not applicable".
  cell_type       The cell population or cell type: hepatocyte, CD4+ T cell,
                  fibroblast, magnocellular neurone. Not a tissue, not an
                  organ, and never a whole organism.
  tissue_type     An anatomical tissue or organ: liver, hypothalamus, leaf,
                  root, whole blood. Not a cell type. If the sample is a whole
                  organism rather than a dissected part, this is
                  "not applicable".
  dev_stage       Developmental stage: embryo, larva, L4, seedling, juvenile,
                  adult, "Day 3 adult", "8 dpf".
  sex             Sex of the organism the sample came from: male, female,
                  pooled male and female, hermaphrodite. For a sample with no
                  meaningful sex - a cell line, an environmental sample, a
                  microbial isolate - this is "not applicable".
  strain          A named strain, isolate, ecotype or cultivar: C57BL/6,
                  N2 Bristol, K-12 MG1655, Col-0. A genotype, a mutation, a
                  knockout, or the word "wildtype" is not a strain.
  treatment       The experimental perturbation applied to this sample: a drug,
                  a dose, a stress, an infection, a knockdown, a diet. An
                  untreated control is a treatment too - say so plainly.
  host            The organism the sample was taken FROM: the host of a
                  microbiome, a parasite, or a pathogen. If the sample simply
                  IS the organism under study, there is no host - answer
                  "not applicable". A mouse study is not hosted by a mouse.
  host_scientific_name
                  The host's binomial name. Same rule: only when a genuine host
                  exists, otherwise "not applicable".
  host_sex        The host's sex. Only when a genuine host exists. For the sex
                  of the studied organism itself, use sex.
  host_tax_id     The host's NCBI taxonomy id, as digits.
  isolation_source
                  Where the sample physically came from: "rumen contents",
                  "surface seawater", "tumour biopsy", "rhizosphere soil".
  collected_by    The person, laboratory or institution that collected the
                  physical sample.
  collection_date The date the physical sample was collected. This is not the
                  sequencing date, the submission date, or the release date -
                  if the evidence only gives you those, this is "not provided".
  country         The country of collection. Where the evidence gives
                  "Country:Region", take the country only.
  environment_biome, environment_feature, environment_material
                  MIxS descriptors of an environmental sample's setting.
  broad_scale_environmental_context, local_environmental_context,
  environmental_medium
                  The MIxS environmental triad, for samples taken from an
                  environment or a host. A laboratory, an incubator, a culture
                  flask, or an experimental treatment is NOT an environmental
                  context. For a cultured or lab-housed sample these are
                  "not applicable".
  checklist       The submission checklist or reporting standard the sample was
                  registered against, e.g. an ENA checklist identifier.
  ncbi_reporting_standard
                  The BioSample package the sample was registered under, e.g.
                  "Model organism or animal", "MIGS.eu", "Human", "Plant".
  sample_capture_status
                  How the sample came to be captured, e.g. active surveillance,
                  clinical presentation.
  sample_alias    The submitter's own short identifier for the sample. Often
                  the same string as the sample title; if the only thing you
                  have is a descriptive title, prefer "not provided".
  sample_title    The sample's title.
  sample_description
                  A description of the sample. Summarise only what the evidence
                  states; do not invent detail to fill the field.
  scientific_name The binomial name of the organism the sample is OF. For a
                  metagenome this is the metagenome name, not a species.
  tax_id          The NCBI taxonomy id of that organism, as digits.

Experiment-level. These describe how the library was made and sequenced, and
are shared by every run beneath one experiment.

  library_name    The submitter's name for the library.
  library_strategy
                  The assay: RNA-Seq, WGS, WXS, ChIP-Seq, AMPLICON, ATAC-seq.
  library_source  The material class: TRANSCRIPTOMIC, GENOMIC, METAGENOMIC,
                  METATRANSCRIPTOMIC, SYNTHETIC.
  library_selection
                  How the material was selected: RANDOM, PCR, cDNA, PolyA,
                  size fractionation, ChIP.
  library_layout  SINGLE or PAIRED.
  library_construction_protocol
                  How the library was prepared: a named kit, a published
                  protocol, "3'READS", "Smart-seq2", "10x Genomics 3' v3".
  instrument_model
                  The specific sequencer: "Illumina NovaSeq 6000",
                  "MGISEQ-2000", "PromethION".
  instrument_platform
                  The platform family: ILLUMINA, OXFORD_NANOPORE, PACBIO_SMRT,
                  DNBSEQ, CAPILLARY.
  sequencing_method
                  The sequencing approach or assay as the submitter describes
                  it. This is not the library preparation protocol - never put
                  the same value in both, and if you only know the prep, fill
                  library_construction_protocol and leave this out.
  experiment_alias, experiment_title
                  The submitter's identifier and title for the experiment.

Run and submission level. These are administrative and are almost never
recoverable from a study description - if the evidence does not name them
directly, leave them out rather than guessing.

  run_alias, submitted_format, submitted_read_type
                  The submitter's run identifier, the uploaded file format
                  (FASTQ, BAM, CRAM), and the read type (single, paired).
  broker_name, center_name, datahub
                  The broker, sequencing centre, or data hub that submitted.
  submission_accession, first_created, first_public, last_updated
                  Archive bookkeeping. Never infer these from a paper or an
                  abstract.
  tag             Free-form archive tags.

ROUTING FREE-TEXT ATTRIBUTES
Submitters use loose keys. Route by what the value actually names, not by the
key:

  "source_name": "Fibroblast"              -> cell_type
  "source_name": "Hypothalamus"            -> tissue_type
  "source_name": "whole worms"             -> the organism; tissue_type and
                                              cell_type are both "not applicable"
  "source_name": "rumen contents"          -> isolation_source
  "genotype": "lax188(skn-1gf)"            -> not a strain; leave strain out
  "agent": "paraquat"                      -> treatment
  "chip antibody": "Creb3l1"               -> library_construction_protocol
                                              context, not a treatment

COMMON MISTAKES TO AVOID
  - Copying the study organism into host or host_scientific_name. If there is
    no host relationship, those fields are "not applicable".
  - Putting the same value in cell_type and tissue_type.
  - Putting the same value in sequencing_method and
    library_construction_protocol.
  - Putting a title into an alias field.
  - Treating an experimental treatment as an environmental context.
  - Answering every field you are shown. A field you cannot settle should be
    omitted, and a field that cannot apply should say so.

WORKED EXAMPLES
Every field you are asked about gets one of three outcomes. Keep them distinct:

  FILL     the evidence gives you a value
  DECLARE  you can positively determine the field cannot apply, or that nobody
           collected it - answer with a missing-value term
  OMIT     the evidence does not reach the field - say nothing at all

OMIT is the common case. In each example below, count the fields in each
bucket: a handful filled, a few declared, and everything else omitted. If you
find yourself declaring more fields than you omit, you are guessing.

1. Lab animal experiment.
   Evidence: "Transcription factor Creb3l1 in rat hypothalamus"; organism
   Rattus norvegicus; attributes {"tissue": "Hypothalamus", "cell type":
   "Magnocellular neurone enriched", "Sex": "male", "treatment": "72 hours
   water deprivation"}.
   FILL     tissue_type "Hypothalamus" (high); cell_type "Magnocellular
            neurone enriched" (high); sex "male" (high); treatment "72 hours
            water deprivation" (high).
   DECLARE  host and host_scientific_name "not applicable" - the rat is the
            subject, not a host. cell_line "not applicable" - primary tissue.
   OMIT     everything else, and that is most of what you were asked: age,
            dev_stage, strain, isolation_source, country, collection_date,
            collected_by, checklist, the aliases, the environmental fields,
            library_construction_protocol, sequencing_method. The evidence is
            silent on all of them. Silence is the answer.

2. Immortalised cell line.
   Evidence: "PCF11 knockdown in NIH3T3"; organism Mus musculus; attributes
   {"cell line": "NIH3T3", "treatment": "siPcf11", "strain": "C57BL/6"}.
   FILL     cell_line "NIH3T3" (high); treatment "siPcf11" (high); strain
            "C57BL/6" (high); cell_type "fibroblast" (medium) - NIH3T3 is a
            fibroblast line, but the evidence does not say so outright.
   DECLARE  tissue_type "not applicable" - a passaged line is not a tissue.
            host "not applicable".
   OMIT     sex, age, dev_stage - a passaged line arguably has none of these,
            but the evidence does not establish that, so do not declare it.
            Also country, collection_date, isolation_source, the aliases, and
            the rest.

3. Host-associated microbiome. The case where host IS filled.
   Evidence: "16S survey of mouse gut microbiota"; organism "mouse gut
   metagenome"; attributes {"host": "Mus musculus", "isolation_source":
   "caecal contents", "host_sex": "female"}.
   FILL     host "Mus musculus" (high); host_scientific_name "Mus musculus"
            (high); host_sex "female" (high); isolation_source "caecal
            contents" (high).
   DECLARE  sex "not applicable" - the sample is the community, which has no
            sex; the host's sex went in host_sex.
   OMIT     cell_type, cell_line, tissue_type, age, dev_stage, strain,
            treatment, country, collection_date, the environmental triad, the
            aliases. A 16S survey description mentions none of them.

4. Environmental sample. The case where the environmental triad IS filled.
   Evidence: "Ammonia-oxidising archaea, Monterey Bay"; organism "marine
   metagenome"; attributes {"geo_loc_name": "USA: California",
   "env_broad_scale": "marine biome", "env_medium": "sea water",
   "collection_date": "2019-06-14"}.
   FILL     country "USA" (high) - country only, region dropped;
            broad_scale_environmental_context "marine biome" (high);
            environmental_medium "sea water" (high); collection_date
            "2019-06-14" (high); isolation_source "sea water" (medium).
   DECLARE  host "not applicable" - free-living. treatment "not applicable" -
            an observational survey applies no perturbation.
   OMIT     sex, age, dev_stage, cell_line, cell_type, tissue_type, strain,
            collected_by, the aliases. Plausibly inapplicable to a seawater
            metagenome, but the evidence does not establish it, so stay quiet
            rather than declaring.

5. Sparse study - the most common shape, and the one to imitate.
   Evidence: "RNA-seq of Arabidopsis thaliana seedlings"; organism Arabidopsis
   thaliana; attributes {"source_name": "seedling"}.
   FILL     dev_stage "seedling" (medium) - source_name names a stage here.
   DECLARE  nothing. One line of evidence supports no determination that a
            field cannot apply.
   OMIT     every other field. Not host, not sex, not tissue_type, not
            treatment - you do not know whether this plant was treated, what
            tissue was taken, or where it was grown. A one-line title is not
            grounds for forty answers.

Across the five: host is filled only in case 3, the environmental triad only
in case 4, and case 5 declares nothing at all. The declared fields are the
ones where the evidence positively rules the field out - never the ones you
merely could not fill."""

# Which model the text layer calls, and how hard it works. Separate from
# `claude.MODEL` so this layer can be switched without touching the transport,
# and so the two model layers can eventually run on different models.
#
# Haiku 4.5 by default: measured at **$0.0017/sample** against Opus 5 at medium
# effort's **$0.0164** — 9.6x cheaper. Output tokens are ~85% of the bill and
# Haiku bills them at $5/MTok against Opus's $25, so the model is the lever;
# dropping Opus from medium to low effort only buys 1.4x. At a third of a cent
# per sample, iterating on the prompt over a whole test set costs pocket change.
#
# **This is a cost default, not a quality verdict.** On one 6-sample study Haiku
# filled 32 fields where Opus filled 55 — but a re-run of that same Opus config
# filled 71, so run-to-run noise swallows most of the gap and neither figure is
# a measurement. Score both against the gold set before trusting either, and
# remember that filling fewer fields may be the *better* behaviour: declining
# where the evidence is thin is what the confidence levels are there to express.
#
# Haiku 4.5 predates adaptive thinking and the effort parameter and rejects
# both, hence None/False — :func:`dataset.validate_model_settings` refuses the
# combination locally rather than letting it 400 mid-run.
#
# Don't edit these to change a run: pass ``text_model=`` / ``paper_model=`` to
# the pipeline stages, or call :func:`configure_models`. Both routes reprice the
# spend estimate to match, which editing the constants directly does not.
#
# Measured on the 52-record set, layers 3 and 4 together: Haiku no-thinking
# $0.25, Sonnet 5 no-thinking ~$0.49, Sonnet 5 thinking $1.25. Thinking was the
# worst of the three — it cost 2.5x more than the same model without it, filled
# *fewer* real values, and pushed a third of its answers onto layer 4 where
# :mod:`audit` cannot check them. Measure before assuming more is better.
TEXT_MODEL = claude.HAIKU_4_5
TEXT_EFFORT: str | None = None
TEXT_THINKING = False

# Layer 4 gets its own three. They defaulted to layer 3's settings because the
# two layers started identical, but the work is not: layer 3 asks a small
# question about ~540 characters of archive evidence tens of thousands of times,
# while layer 4 asks one question per study carrying up to PAPER_MAX_CHARS of
# full text. That is the call where a better model has the most to read and the
# fewest chances to bill, so it is the one worth paying more for — and pinning
# them together meant upgrading the cheap layer to upgrade the expensive one.
PAPER_MODEL = TEXT_MODEL
PAPER_EFFORT: str | None = TEXT_EFFORT
PAPER_THINKING = TEXT_THINKING


def configure_models(text_model=None, text_effort=None, text_thinking=None,
                     paper_model=None, paper_effort=None, paper_thinking=None) -> None:
    """Override the per-layer model settings for this process.

    ``None`` leaves a setting alone, so a caller can change the paper model
    without restating layer 3's. Mirrors :func:`claude.set_api_key`: the
    pipeline stages take these as arguments and call this once, before any
    paid work, rather than reaching into module globals themselves.

    Validation lives in :func:`dataset.validate_model_settings`, which runs
    before the spend estimate — a rejected combination has to fail on a local
    check, not on a 400 partway through a run that has already billed.
    """
    global TEXT_MODEL, TEXT_EFFORT, TEXT_THINKING
    global PAPER_MODEL, PAPER_EFFORT, PAPER_THINKING
    if text_model is not None:
        TEXT_MODEL = text_model
    if text_effort is not None:
        TEXT_EFFORT = text_effort
    if text_thinking is not None:
        TEXT_THINKING = text_thinking
    if paper_model is not None:
        PAPER_MODEL = paper_model
    if paper_effort is not None:
        PAPER_EFFORT = paper_effort
    if paper_thinking is not None:
        PAPER_THINKING = paper_thinking

# Fields per call. `None` asks the whole open set at once, which is what the
# answer-list schema below makes possible. This is the dial for the
# one-call-per-field question: set it to 1 and the layer asks each field in
# isolation with everything else held constant. Smaller chunks split the joint
# reasoning that makes a batched call worth having — knowing the organism is a
# metagenome is what settles `sex` — so if you do chunk, keep related fields
# together rather than slicing the alphabetical order.
FIELDS_PER_CALL: int | None = None

# Submit this layer's calls through the Batch API instead of one at a time.
# Every token bills at 50%, with no change to the model, prompt or output.
#
# **Off by default, and the reason is control rather than cost.** A batch is
# submit-poll-collect: it streams no progress, and a run killed mid-batch
# abandons work that still completes and still bills. Sequential calls can be
# watched and stopped, and after a run that overspent by $7 that is worth more
# than the discount.
#
# Be clear that it *is* a discount being given up. Measured:
#
#   52-record run    batched $0.1786  vs live $0.2465   -> batching 28% cheaper
#   1,782-record run batched $6.16    vs live ~$6.98    -> batching 12% cheaper
#
# Batching fights prompt caching — a batch's requests run in parallel, so most
# write the shared prefix rather than reading it, and at 1,782 records 65% of
# calls wrote (see layer_3_haiku_findings.md section 6). That erodes the
# discount but never reversed it on anything measured. A batch whose cache
# actually read would have cost $3.49, so the headroom is in fixing that
# interaction, not in avoiding batches.
TEXT_BATCH = False

# A throwaway record used to test whether a value will survive assignment before
# it is proposed. The API constrains types, but not semantics: it will happily
# return "spring 2015" for a date field, which parses fine as JSON and then
# raises on the way into the schema. Checking here costs nothing and keeps one
# fuzzy date from ending a 305-study run.
_PROBE = TargetSchema(id="probe")


def _storable(name: str, value) -> bool:
    try:
        setattr(_PROBE, name, value)
    except ValueError:
        return False
    return True


def _answer_schema(names) -> dict:
    """A JSON Schema asking for a *list* of answers, enforced by the API.

    The obvious shape — one property per field, each an optional
    ``{value, confidence}`` object — is rejected outright. Structured outputs
    enforce three separate limits, and asking about ~41 open fields trips all
    three in turn: at most 16 union-typed parameters (so no
    ``["string", "null"]`` values), at most 24 optional parameters (so not one
    property per field), and an overall "schema is too complex" ceiling that 24
    nested objects exceeds on its own.

    A list sidesteps all of it. One property, one item shape, and the field name
    is an ``enum`` — enums are cheap where properties are not — so the whole open
    set fits in a single call with room to spare. Declining is simply not
    emitting an item, and nothing constrains the model to answer everything.

    ``value`` is a string for every field, including the integer and date ones:
    :class:`schema.TargetSchema` already coerces ``"9606"`` and ``"2015-03-04"``
    on assignment, and a per-field value type would reintroduce the union limit
    for no gain.
    """
    return claude.object_schema(
        {
            "answers": {
                "type": "array",
                "items": claude.object_schema(
                    {
                        "field": {"type": "string", "enum": list(names)},
                        "value": {"type": "string"},
                        "confidence": {"type": "string", "enum": list(CONFIDENCE_LEVELS)},
                    }
                ),
            }
        }
    )


def _evidence(project, sample, records) -> str:
    """Everything the archive offers about one sample, as plain text.

    The sample's raw ``attributes`` are the important part and go over
    un-harmonized — with layer 2 skipped they are the only place most biological
    values appear at all. See :func:`harmonize_attributes` for what that costs:
    the model maps those keys itself, and the cascade stamps the result
    `inferred_from_text` when it is really harmonization.
    """
    lines = [f"STUDY TITLE: {project.title}"]
    if project.abstract:
        lines.append(f"STUDY ABSTRACT: {project.abstract}")
    if sample is not None:
        if sample.scientific_name:
            lines.append(f"ORGANISM: {sample.scientific_name}")
        if sample.title:
            lines.append(f"SAMPLE TITLE: {sample.title}")
        if sample.attributes:
            lines.append(f"SAMPLE ATTRIBUTES: {json.dumps(sample.attributes)}")
    titles = sorted({r.experiment_title for r in records if r.experiment_title})
    if titles:
        lines.append("EXPERIMENT TITLES: " + "; ".join(titles))
    assays = sorted({r.library_strategy for r in records if r.library_strategy})
    if assays:
        lines.append("LIBRARY STRATEGY: " + ", ".join(assays))
    return "\n".join(lines)


def _study_evidence(project) -> str:
    """Just the study — no sample attributes. Feeds the one study-level call."""
    lines = [f"STUDY TITLE: {project.title}"]
    if project.abstract:
        lines.append(f"STUDY ABSTRACT: {project.abstract}")
    if project.study_type:
        lines.append(f"STUDY TYPE: {project.study_type}")
    return "\n".join(lines)


def _ask(evidence: str, wanted: list[str]) -> dict[str, dict]:
    """One or more calls covering `wanted`; returns ``{field: answer}``."""
    answers: dict[str, dict] = {}
    for chunk in _chunks(wanted):
        reply = claude.extract(
            evidence,
            _answer_schema(chunk),
            system=TEXT_SYSTEM,
            model=TEXT_MODEL,
            effort=TEXT_EFFORT,
            thinking=TEXT_THINKING,
        )
        for answer in reply.get("answers") or []:
            answers[answer["field"]] = answer
    return answers


def _propose(proposals, records, answers, open_by_id) -> None:
    """Merge `answers` into `proposals` for every record they are open on."""
    for record in records:
        for name, answer in answers.items():
            value = answer.get("value")
            if not value or name not in open_by_id[record.id]:
                continue
            if not _storable(name, value):
                continue
            proposals.setdefault(record.id, {})[name] = Proposal(
                value, answer.get("confidence")
            )


def infer_from_text(project, records, open_by_id):
    """Layer 3 — reason from titles, abstracts and the sample attribute bag.

    Two call shapes, split by what the evidence can actually settle:

    **One call per study** for the study-level fields (:data:`schema.STUDY`).
    ``project_name`` is a property of the study, not of a sample, so asking it
    once per sample was both redundant and wrong: over a 60-sample run it made
    ~106 duplicate asks and produced *two different* ``study_alias`` answers
    across one study's 15 samples. One study, one answer, by construction. This
    call sees only the study text — sample attributes cannot inform it.

    **One call per sample** for everything else. Every experiment on a sample
    shares its biology, so asking once and applying to all of them is cheaper
    and self-consistent. Records with no resolvable sample are grouped and asked
    once too.

    Each call asks the union of fields open across its records and proposes back
    only the ones open for each individual record — :func:`reconstruct` refuses a
    proposal outside the open set. A field the model omits is simply not
    proposed; that is how it declines and leaves the field for the paper layer.

    **Does not ask what the archive assigns.** Run-, submission- and
    record-level fields are dropped from the ask entirely (see
    :data:`_TEXT_BLIND`), mirroring what :func:`infer_from_paper` does with
    :data:`_PAPER_BLIND`. Without it this layer invented run and submission
    accessions and answered "not provided" thousands of times at full price.

    With :data:`TEXT_BATCH` the whole study's calls are submitted as one batch at
    half price; the work is planned identically either way, so the two paths
    differ only in how the requests are sent.
    """
    study_level = set(TargetSchema.fields_at_level(STUDY))

    # Drop what this layer cannot answer before anything is planned, so the
    # blind fields never reach a schema, a token budget, or an answer.
    open_by_id = {rid: open_ - _TEXT_BLIND for rid, open_ in open_by_id.items()}

    # Plan every call first — the same plan feeds the live and batched paths.
    jobs: list[tuple[str, list, str, list[str]]] = []
    study_open = sorted({f for r in records for f in open_by_id[r.id]} & study_level)
    if study_open:
        jobs.append(("study", records, _study_evidence(project), study_open))

    by_sample: dict[str | None, list] = {}
    for record in records:
        by_sample.setdefault(record.secondary_sample_accession, []).append(record)
    for sample_id, group in by_sample.items():
        wanted = sorted(set().union(*(open_by_id[r.id] for r in group)) - study_level)
        if not wanted:
            continue
        sample = project.samples.get(sample_id) if sample_id else None
        jobs.append((sample_id or "_", group, _evidence(project, sample, group), wanted))

    if not jobs:
        return {}
    answers = _run_batched(jobs) if TEXT_BATCH else _run_live(jobs)

    proposals: dict[str, dict[str, Proposal]] = {}
    for key, targets, _, _ in jobs:
        _propose(proposals, targets, answers.get(key, {}), open_by_id)
    return proposals


def _run_live(jobs) -> dict[str, dict[str, dict]]:
    """One request at a time, answer returned immediately."""
    return {key: _ask(evidence, wanted) for key, _, evidence, wanted in jobs}


def _run_batched(jobs) -> dict[str, dict[str, dict]]:
    """Every request for this study in one half-price batch.

    Chunks get their own batch entries and are merged back per job, so
    :data:`FIELDS_PER_CALL` behaves the same as on the live path.
    """
    requests: dict[str, tuple[str, dict]] = {}
    owners: dict[str, str] = {}
    for key, _, evidence, wanted in jobs:
        for n, chunk in enumerate(_chunks(wanted)):
            rid = f"{key}#{n}"
            requests[rid] = (evidence, _answer_schema(chunk))
            owners[rid] = key

    replies = claude.extract_batch(
        requests, system=TEXT_SYSTEM, model=TEXT_MODEL,
        effort=TEXT_EFFORT, thinking=TEXT_THINKING,
    )
    out: dict[str, dict[str, dict]] = {}
    for rid, reply in replies.items():
        for answer in reply.get("answers") or []:
            out.setdefault(owners[rid], {})[answer["field"]] = answer
    return out


def _chunks(names: list[str]):
    if not FIELDS_PER_CALL:
        yield names
        return
    for start in range(0, len(names), FIELDS_PER_CALL):
        yield names[start : start + FIELDS_PER_CALL]


PAPER_SYSTEM = TEXT_SYSTEM + """

READING A PAPER
You are now reading the study's publication rather than its archive record. Two
things change.

First, a paper describes the *study*. It will say "we sequenced tumour and
adjacent normal tissue" without telling you which sample is which. Do answer
study-wide facts; do not invent a per-sample distinction the paper does not
draw. If a value clearly differs between samples and the paper does not say
which is which, leave the field alone.

The paper text is "the evidence" for the confidence rules above. One addition:
a fact the paper states about the study as a whole, which you are applying to
this sample, is medium at best - you are generalising, not copying, even when
the words match.

Second, you are shown what earlier stages already established. Treat those as
settled and do not restate them - they are context to reason from, not fields to
answer. You are only being asked about what is still missing.
"""

# Characters of paper text sent per study. A full paper is tens of thousands of
# tokens against ~540 for a sample's archive evidence, so this layer's single
# call can cost more than every layer-3 call in the study combined. The text is
# Methods-first (see `project.fetch_open_access_text`), so a tight budget keeps
# the section where sample provenance actually lives.
PAPER_MAX_CHARS = 30000

# Fields a paper cannot speak to, and which layer 4 is therefore never asked
# about. Not a prompt instruction — TEXT_SYSTEM already says "never infer these
# from a paper or an abstract" and the model did it anyway, 14 times per field,
# because the field list it is shown outranks the prose telling it not to.
#
# Run- and submission-level fields are assigned by the archive at deposition:
# run accessions, upload formats, the submitting centre, release dates. A
# manuscript states none of them. The named extras below sit at sample or
# experiment level in the schema but are the same kind of thing — registration
# artefacts and submitter-chosen identifiers, not science.
#
# On the measured run these accounted for ~150 of layer 4's 262 "filled" fields,
# essentially all of them "not provided".
PAPER_BLIND_FIELDS = (
    "checklist",                # ENA checklist identifier
    "ncbi_reporting_standard",  # BioSample package
    "sample_alias",             # submitter-chosen identifiers, all three
    "experiment_alias",
    "study_alias",
    "library_name",
    "sample_capture_status",    # controlled INSDC vocabulary, not prose
)

_PAPER_BLIND = (
    set(TargetSchema.fields_at_level(RUN))
    | set(TargetSchema.fields_at_level(SUBMISSION))
    | set(TargetSchema.fields_at_level(RECORD))
    | set(PAPER_BLIND_FIELDS)
)

# The same three levels are equally unanswerable from a study's own text, and
# layer 3 had no guard at all. Run accessions, submission accessions, upload
# formats and the submitting centre are assigned by the archive at deposition —
# an abstract and a sample attribute bag state none of them.
#
# Measured on the 1,782-record and 52-record runs, layer 3 filled 4,342 of these
# and the results were exactly as good as that description predicts:
#
#   submission_accession  PRJNA293224 on 146 records — a *BioProject* accession,
#                         not a submission accession; wrong identifier entirely
#   run_accession         CFSAN100605 — a strain id, not an SRR
#   the rest              "not provided" x thousands, at full token price
#
# Those "not provided" answers are not harmless either: they are written as
# determinations with provenance, so they read as settled facts, inflate the
# coverage metric, and were a large share of the missing-value terms that the
# confidence audit found flooding the `high` bucket.
#
# Deliberately *not* including PAPER_BLIND_FIELDS: `checklist`,
# `ncbi_reporting_standard` and the aliases are sample- and experiment-level
# facts that the attribute bag genuinely can carry, even though a manuscript
# cannot. This layer reads that bag; layer 4 does not.
_TEXT_BLIND = (
    set(TargetSchema.fields_at_level(RUN))
    | set(TargetSchema.fields_at_level(SUBMISSION))
    | set(TargetSchema.fields_at_level(RECORD))
)


def _established(record) -> dict[str, str]:
    """What the earlier layers settled, as context for the paper call.

    Passed in so the model can reason from the organism, the tissue and the
    treatment rather than re-deriving them - and so it does not answer them
    again. There is no anchoring risk here by construction: this layer is only
    asked about fields that are still empty, so nothing in the context overlaps
    the question.
    """
    return {
        name: str(getattr(record, name))
        for name in record.field_names()
        if getattr(record, name) is not None and name != "id"
    }


def infer_from_paper(project, records, open_by_id):
    """Layer 4 - reason from the study's open-access publication.

    **One call per study, one paper.** The paper is study-level evidence, so it
    is read once and its answers copied to every record beneath it. Reading the
    same paper once per sample would be the single most expensive mistake
    available in this pipeline - a 30,000-character paper re-sent for each of a
    study's 24 samples.

    Only the *first* publication classified ``oa`` is used. Later ones cannot
    add much a paper on the same study has not already said, and each one is
    another full-text fetch and another large call.

    **Never overrides.** This layer only proposes fields whose current value is
    ``None``: not fields holding an ordinary value, and not fields holding a
    missing-value term either. A ``not applicable`` from layer 3 was a positive
    determination from the sample's own attributes, which are better evidence
    about one sample than a paper describing the whole cohort. The cascade's
    ``overwrite_missing`` knob would reopen those; this layer declines them
    regardless.

    **Does not ask what a paper cannot answer.** Run- and submission-level
    fields, plus the identifiers in :data:`PAPER_BLIND_FIELDS`, are dropped from
    the ask entirely (see :data:`_PAPER_BLIND`) — a smaller schema, fewer output
    tokens, and no "not provided" noise for fields the archive assigns.

    Does nothing - no fetch, no call - when the study has no ``oa``
    publication, when the text cannot be retrieved, or when nothing is open.
    """
    paper_id = next(
        (p.id for p in project.publications if p.accessibility_type == "oa"), None
    )
    if paper_id is None:
        return {}

    # Fields still genuinely empty on at least one record, minus the ones a
    # paper cannot answer. Asked once for the whole study.
    wanted = sorted({
        name
        for record in records
        for name in open_by_id[record.id]
        if getattr(record, name) is None and name not in _PAPER_BLIND
    })
    if not wanted:
        return {}

    text = project_module.fetch_open_access_text(paper_id, max_chars=PAPER_MAX_CHARS)
    if not text:
        return {}

    established = _established(records[0])
    evidence = (
        f"STUDY TITLE: {project.title}\n"
        f"ALREADY ESTABLISHED (do not answer these again): "
        f"{json.dumps(established, sort_keys=True)}\n\n"
        f"PUBLICATION (PMID/DOI {paper_id}):\n{text}"
    )

    answers: dict[str, dict] = {}
    for chunk in _chunks(wanted):
        reply = claude.extract(
            evidence, _answer_schema(chunk), system=PAPER_SYSTEM,
            model=PAPER_MODEL, effort=PAPER_EFFORT, thinking=PAPER_THINKING,
        )
        for answer in reply.get("answers") or []:
            answers[answer["field"]] = answer

    proposals: dict[str, dict[str, Proposal]] = {}
    for record in records:
        filled = {}
        for name, answer in answers.items():
            value = answer.get("value")
            if not value or name not in open_by_id[record.id]:
                continue
            if getattr(record, name) is not None:
                continue  # never override an earlier layer, even a declared one
            if not _storable(name, value):
                continue
            filled[name] = Proposal(value, answer.get("confidence"))
        if filled:
            proposals[record.id] = filled
    return proposals


# Order is the cascade order and is fixed: simplest technology first, so a later
# layer only ever pays for what the earlier ones could not answer. Enabling a
# layer is a flag rather than a list, which makes an out-of-order run
# unrepresentable.
LAYERS = (
    ("harmonize", "harmonized", harmonize_attributes),
    ("from_text", "inferred_from_text", infer_from_text),
    ("from_paper", "inferred_from_paper", infer_from_paper),
)

_INFERRED = ("inferred_from_text", "inferred_from_paper")


def reconstruct(
    project,
    harmonize: bool = False,
    from_text: bool = False,
    from_paper: bool = False,
    overwrite_missing: bool = True,
) -> tuple[list[TargetSchema], dict[str, int]]:
    """Run the cascade over one study; returns ``(records, report)``.

    ``records`` is one :class:`schema.TargetSchema` per experiment (see
    :meth:`schema.TargetSchema.from_project`). ``report`` counts the fields each
    layer filled — ``{"direct": 20, "inferred_from_text": 7}`` — which is the
    number worth watching as layers come online, since a layer that fires but
    fills nothing looks identical to one that never ran.

    The direct layer always runs; the rest are opt-in and currently raise
    :class:`NotImplementedError`. ``overwrite_missing`` is passed to
    :func:`open_fields` and decides whether a later layer may revisit an earlier
    layer's missing-value verdict.
    """
    records = TargetSchema.from_project(project)
    report = {"direct": sum(len(r.provenance) for r in records)}

    enabled = {"harmonize": harmonize, "from_text": from_text, "from_paper": from_paper}
    for flag, provenance_class, layer in LAYERS:
        if not enabled[flag]:
            continue
        open_by_id = {
            r.id: open_fields(r, overwrite_missing=overwrite_missing) for r in records
        }
        proposals = layer(project, records, open_by_id) or {}
        report[provenance_class] = _apply(records, proposals, provenance_class, open_by_id)
    return records, report


def _apply(records, proposals, provenance_class, open_by_id) -> int:
    """Write one layer's proposals, stamping provenance. Returns fields filled."""
    by_id = {r.id: r for r in records}
    filled = 0
    for record_id, fields in proposals.items():
        record = by_id.get(record_id)
        if record is None:
            raise ValueError(
                f"{provenance_class} layer proposed values for unknown record "
                f"{record_id!r}"
            )
        for name, proposal in fields.items():
            if name not in open_by_id[record_id]:
                # Silently dropping this would let a layer quietly overwrite an
                # earlier, cheaper answer — or think it had filled something it
                # had not. Both are layer bugs, and both are invisible without it.
                raise ValueError(
                    f"{provenance_class} layer proposed {name!r} on {record_id!r}, "
                    f"which is not open to it"
                )
            if proposal.confidence is not None and provenance_class not in _INFERRED:
                raise ValueError(
                    f"{provenance_class} layer set a confidence on {name!r}; only "
                    f"{' and '.join(_INFERRED)} may, since nobody chose the value"
                )
            setattr(record, name, proposal.value)
            record.provenance[name] = provenance_class
            if proposal.confidence is not None:
                record.confidence[name] = proposal.confidence
            elif name in record.confidence:
                del record.confidence[name]  # a stale level from an earlier layer
            filled += 1
    return filled