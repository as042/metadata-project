"""The target schema for reconstructed SRA metadata.

:class:`TargetSchema` is the *output* of the LLM stage, in contrast to
:mod:`project`, which is the *input*: `Project` holds SRA exactly as the archive
returned it—nested, sparse, with submitter attributes left as an untouched
key/value bag—while a `TargetSchema` record holds one flat, typed, queryable
row with those attributes harmonized into named fields.

The field set is ENA's own metadata vocabulary for the read portal, which buys
a free validation signal: wherever ENA publishes a real value for a field, a
reconstructed one can be scored against it without hand-labelling anything.
Names and spellings are kept verbatim even where they are inconsistent
(``environment_biome`` vs ``environmental_medium``, ``tissue_type`` vs
``cell_type``)—diverging would cost that comparison.

**Granularity: one record per experiment.** Study- and sample-level values are
denormalized onto every experiment beneath them, and each experiment's runs hang
off it as a list. A study therefore yields a *list* of records:

    study   = Project("SRP098789")
    records = TargetSchema.from_project(study)     # one per experiment

The experiment (SRX) is the grain because it is the level at which biology and
assay meet: "human liver RNA-seq" is a conjunction of a sample property and an
assay property, and SRX is the smallest entity where both are defined. It is
also what ``db=sra`` indexes and what an NCBI search returns — see PIPELINE.md
§2, which pages experiment records and de-dupes *up* to studies. Runs are the
download menu underneath a hit, not a search result: a 7-run experiment is one
library split across lanes, so seven records would repeat one biology seven
times, and any per-record inference could answer them inconsistently.

``from_project`` fills only what SRA states outright—accessions, library and
instrument fields, counts, dates, organism. It deliberately performs **no**
attribute harmonization and **no** inference: every value it sets is provably
the archive's own. Populating the biological fields (``host``, ``tissue_type``,
``isolation_source``, the environmental context triad …) is the reconstruction
stage's job, and each field it fills should be recorded in :attr:`provenance`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields as _dataclass_fields
from datetime import datetime

# Every field is one of three Python types. Values are coerced to these on write
# (see TargetSchema.__setattr__), so a record is always in its declared types.
FIELD_TYPES = (str, int, datetime)

# Which level of the SRA hierarchy a field describes. A record is one experiment,
# so everything above it is denormalized down (see the module docstring).
RECORD = "record"
STUDY = "study"
EXPERIMENT = "experiment"
SAMPLE = "sample"
RUN = "run"
SUBMISSION = "submission"

LEVELS = (RECORD, STUDY, EXPERIMENT, SAMPLE, RUN, SUBMISSION)

# How a reconstructed field was obtained. The split is by *who decided the value
# belongs in this field*, because that is what a wrong value tells you:
#
#   direct              read from an INSDC structured field; nobody chose, the
#                       archive's schema guarantees it. A failure is a parser bug.
#   harmonized          a submitter attribute key-matched onto a schema field.
#                       The value is the submitter's; the mapping is ours, and it
#                       can be wrong (`geo_loc_name` is often "USA: California",
#                       which is not just a country). A failure is a synonym gap.
#   inferred_from_text  reasoned by the model from titles/abstracts.
#   inferred_from_paper reasoned from the linked open-access publication. Kept
#                       apart from the above because a paper describes the
#                       *study* — "tumor and adjacent normal" is true of the
#                       whole cohort and does not say which experiment is which,
#                       so these carry an attribution risk the others don't.
#
# Scoring accuracy per class is the point: an aggregate figure can't tell you
# whether to fix the parser, the synonym table, or the prompt.
PROVENANCE_CLASSES = (
    "direct",
    "harmonized",
    "inferred_from_text",
    "inferred_from_paper",
)

# **Where the value came from**, not how likely it is to be right:
#
#   high    quoted     the value appears in the evidence word for word
#   medium  rephrased  the evidence carries it in different words, or the model
#                      picked one of several spans it could have quoted
#   low     inferred   the evidence does not carry the value at all
#
# This used to be an epistemic scale — how sure the model was of being correct —
# and it did not work. Measured over four runs the label never separated: `high`
# took 68–94% of every answer and `low` was emitted **zero** times in two of
# them, so there was no variance for accuracy to correlate with. The prompt was
# also asking for two things at once, defining the buckets mechanically while
# framing them as correctness, and the correctness half is the half models are
# worst at.
#
# The mechanical axis earns its keep by being *checkable*. "Appears in the
# evidence word for word" is a claim :mod:`audit` can verify by string matching —
# no gold set, no model, no spend — which is something no confidence scale could
# ever offer. Run :func:`audit.verbatim_report` over a finished dataset before
# trusting the label.
#
# Note what this does **not** measure: a quoted value can still be wrong (the
# model quoted the wrong span — see the `host` bug in `reconstruct`), and an
# inferred one can be right. Do not read `high` as "probably correct". If you
# want per-field error rates, score against a labelled set; this tells you how
# far the model reached, not whether it landed.
#
# Still ordinal, still three buckets, and deliberately not a 0–1 float:
# verbalized numeric confidence is a generated token rather than a read-off
# probability, and three named operations don't imply precision that isn't there.
CONFIDENCE_LEVELS = ("high", "medium", "low")

# INSDC's missing-value vocabulary. Each is a *stated reason* a field has no
# ordinary value — an answer, not an absence — so each is stored as the field's
# value and carries provenance and confidence like any other. Using INSDC's own
# terms rather than inventing sentinels means anything downstream that already
# understands them reads this output correctly.
#
#   not applicable     the field cannot apply here: `sex` on a soil metagenome,
#                      `host` on a free-living isolate
#   not collected      nobody measured it — the value does not exist anywhere
#   not provided       it exists but the submitter did not deposit it
#   restricted access  it exists and is deposited, but behind controlled access
#                      (dbGaP and friends)
#
# Not verified against the INSDC/BioSample specification this session; confirm
# the exact set and spellings before depending on them.
# Fields with no counterpart in ENA's portal vocabulary. Everything else here
# is a real ENA field name, so a value can in principle be checked against what
# ENA publishes for the same accession — a free correctness test on layer 2's
# synonym table, though not on the inferred layers, which target fields ENA
# leaves empty by definition.
#
# `age`, `cell_line` and `dev_stage` are ENA fields that simply were not in the
# 61-field starting subset; adding them widens ENA coverage rather than
# diverging from it. `treatment` is a deliberate extension — ENA has no field
# for it under any spelling, and it is one of the most common submitter
# attributes there is, so the alternative was watching the model dump it into
# whichever adjacent slot looked closest.
EXTENSION_FIELDS = ("treatment",)

NOT_APPLICABLE = "not applicable"
NOT_COLLECTED = "not collected"
NOT_PROVIDED = "not provided"
RESTRICTED_ACCESS = "restricted access"

MISSING_VALUES = (NOT_APPLICABLE, NOT_COLLECTED, NOT_PROVIDED, RESTRICTED_ACCESS)

# What none of them mean: ``None``. A field is None when *nothing has been
# concluded* — reconstruction has not run on it yet, or ran and reached no
# answer. That is a statement about this pipeline, whereas every term above is a
# statement about the archive, and the two must not be conflated: defaulting an
# untouched field to `not collected` would assert something about the submitter
# that nobody checked, and would report full coverage on a record no model has
# looked at. `not collected` is the right answer once reconstruction has looked
# and found nothing — it is a conclusion, not an initial state.


def _f(type_: type, level: str):
    """An optional schema field, tagged with its type and hierarchy level."""
    return field(default=None, metadata={"type": type_, "level": level})


# Accepted on input, most precise first. SRA/ENA emit the first three; the
# fourth is `Project.published`'s format; the last two are partial dates, which
# submitters use freely ("2015", "2015-03").
_DATE_FORMATS = (
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
    "%Y-%m",
    "%Y",
)


def _parse_date(value, name: str) -> datetime | None:
    """Coerce to datetime, accepting partial dates by padding to January 1st.

    Padding is lossy—"2015" and "2015-01-01" both land on the same instant, so
    the record no longer says which was submitted, and submitters write bare
    years and months constantly. That is the price of storing these as
    ``datetime``, which has no partial form; the alternative is keeping them as
    strings and giving up ordering and comparison. Worth revisiting if the
    precision turns out to matter more than the sorting does.
    """
    if value is None or isinstance(value, datetime):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:  # handles offsets and the trailing Z that strptime will not
            return datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            pass
        for fmt in _DATE_FORMATS:
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                continue
    raise ValueError(f"{name}: cannot parse {value!r} as a date")


def _parse_int(value, name: str) -> int | None:
    """Coerce to int. Raises rather than dropping a value it cannot read.

    An LLM asked for ``read_count`` will occasionally answer "about 4 million";
    failing loudly on that beats indexing a silent null that then reads as
    "the archive never said".
    """
    if value is None or isinstance(value, int):
        return value
    if isinstance(value, str):
        text = value.strip().replace(",", "")
        if not text:
            return None
        value = text
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name}: cannot parse {value!r} as an integer") from None


def _clean(value) -> str | None:
    """Strip a text value; blank becomes None so 'absent' has one representation."""
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in MISSING_VALUES:
        return text.lower()  # canonical case — a model will vary it
    return text or None


def _is_missing_value(value) -> bool:
    """True for an INSDC missing-value term in any casing.

    Only the exact terms count. Aliases a submitter might write — "N/A", "NA",
    "none", "unknown" — are deliberately *not* mapped: each is ambiguous with a
    real value ("NA" is a plausible strain, region or country code), and quietly
    reinterpreting data as a missing-value declaration is worse than leaving it
    as the text it was.
    """
    return isinstance(value, str) and value.strip().lower() in MISSING_VALUES


class _ValidatedMap(dict):
    """Base for the sidecar ``{field_name: token}`` maps hung off a record.

    Validation lives here rather than in ``TargetSchema.__setattr__`` because the
    usual way to write one is item assignment —
    ``record.provenance["country"] = "harmonized"`` — which never reaches
    ``__setattr__`` at all. Checking only whole-dict assignment would leave the
    common path unguarded, the same reasoning that put field coercion in
    ``__setattr__`` rather than ``__init__``.

    Keys are checked against the schema too, so ``provenance["contry"]`` fails
    now instead of surfacing as a field that mysteriously has no provenance.
    """

    _label = "value"
    _allowed: tuple[str, ...] = ()

    def __init__(self, initial=()):
        super().__init__()
        self.update(initial)

    def __setitem__(self, field_name, token):
        if field_name not in _FIELD_TYPES:
            raise ValueError(
                f"{self._label} key {field_name!r} is not a schema field; "
                f"expected one of {len(_FIELD_TYPES)} field names"
            )
        if token not in self._allowed:
            raise ValueError(
                f"{self._label}[{field_name!r}] = {token!r} is not valid; "
                f"expected one of {', '.join(self._allowed)}"
            )
        super().__setitem__(field_name, token)

    def update(self, other=(), /, **kw):
        for key, value in dict(other, **kw).items():
            self[key] = value

    def setdefault(self, key, default=None):
        if key not in self:
            self[key] = default
        return self[key]


class _ProvenanceMap(_ValidatedMap):
    _label = "provenance"
    _allowed = PROVENANCE_CLASSES


class _ConfidenceMap(_ValidatedMap):
    _label = "confidence"
    _allowed = CONFIDENCE_LEVELS


@dataclass
class TargetSchema:
    """One reconstructed metadata record—a single run, fully denormalized.

    Every field except ``id`` defaults to ``None``, which means *not known*.
    Values are coerced and validated on construction and on assignment, so a
    record is always in its declared types.
    """

    id: str = field(metadata={"type": str, "level": RECORD})

    age: str | None = _f(str, SAMPLE)
    base_count: int | None = _f(int, RUN)
    broad_scale_environmental_context: str | None = _f(str, SAMPLE)
    broker_name: str | None = _f(str, SUBMISSION)
    cell_line: str | None = _f(str, SAMPLE)
    cell_type: str | None = _f(str, SAMPLE)
    center_name: str | None = _f(str, SUBMISSION)
    checklist: str | None = _f(str, SAMPLE)
    collected_by: str | None = _f(str, SAMPLE)
    collection_date: datetime | None = _f(datetime, SAMPLE)
    country: str | None = _f(str, SAMPLE)
    datahub: str | None = _f(str, SUBMISSION)
    description: str | None = _f(str, STUDY)
    dev_stage: str | None = _f(str, SAMPLE)
    environment_biome: str | None = _f(str, SAMPLE)
    environment_feature: str | None = _f(str, SAMPLE)
    environment_material: str | None = _f(str, SAMPLE)
    environmental_medium: str | None = _f(str, SAMPLE)
    experiment_accession: str | None = _f(str, EXPERIMENT)
    experiment_alias: str | None = _f(str, EXPERIMENT)
    experiment_title: str | None = _f(str, EXPERIMENT)
    first_created: datetime | None = _f(datetime, SUBMISSION)
    first_public: datetime | None = _f(datetime, SUBMISSION)
    host: str | None = _f(str, SAMPLE)
    host_scientific_name: str | None = _f(str, SAMPLE)
    host_sex: str | None = _f(str, SAMPLE)
    host_tax_id: int | None = _f(int, SAMPLE)
    instrument_model: str | None = _f(str, EXPERIMENT)
    instrument_platform: str | None = _f(str, EXPERIMENT)
    isolation_source: str | None = _f(str, SAMPLE)
    last_updated: datetime | None = _f(datetime, SUBMISSION)
    library_construction_protocol: str | None = _f(str, EXPERIMENT)
    library_layout: str | None = _f(str, EXPERIMENT)
    library_name: str | None = _f(str, EXPERIMENT)
    library_selection: str | None = _f(str, EXPERIMENT)
    library_source: str | None = _f(str, EXPERIMENT)
    library_strategy: str | None = _f(str, EXPERIMENT)
    local_environmental_context: str | None = _f(str, SAMPLE)
    ncbi_reporting_standard: str | None = _f(str, SAMPLE)
    project_name: str | None = _f(str, STUDY)
    read_count: int | None = _f(int, RUN)
    run_accession: str | None = _f(str, RUN)
    run_alias: str | None = _f(str, RUN)
    sample_accession: str | None = _f(str, SAMPLE)
    sample_alias: str | None = _f(str, SAMPLE)
    sample_capture_status: str | None = _f(str, SAMPLE)
    sample_description: str | None = _f(str, SAMPLE)
    sample_title: str | None = _f(str, SAMPLE)
    scientific_name: str | None = _f(str, SAMPLE)
    secondary_sample_accession: str | None = _f(str, SAMPLE)
    secondary_study_accession: str | None = _f(str, STUDY)
    sequencing_method: str | None = _f(str, EXPERIMENT)
    sex: str | None = _f(str, SAMPLE)
    strain: str | None = _f(str, SAMPLE)
    study_accession: str | None = _f(str, STUDY)
    study_alias: str | None = _f(str, STUDY)
    study_title: str | None = _f(str, STUDY)
    submission_accession: str | None = _f(str, SUBMISSION)
    submitted_format: str | None = _f(str, RUN)
    submitted_read_type: str | None = _f(str, RUN)
    tag: str | None = _f(str, RECORD)
    tax_id: int | None = _f(int, SAMPLE)
    tissue_type: str | None = _f(str, SAMPLE)
    treatment: str | None = _f(str, SAMPLE)

    # Not part of the ENA field set: how each field above was obtained, keyed by
    # field name -> one of PROVENANCE_CLASSES. Validated on write (see
    # _ProvenanceMap). Not a schema field; kept by to_dict() so a
    # reconstruction can be audited and scored per provenance class.
    provenance: dict[str, str] = field(
        default_factory=dict, metadata={"type": None, "level": RECORD}
    )

    # Parallel to provenance: how sure the model is it picked the right answer,
    # one of CONFIDENCE_LEVELS. Only meaningful for an `inferred_*` field that
    # actually holds an answer — including a MISSING_VALUES term, which is one. A
    # `direct` field or an empty one carrying a confidence is a category error.
    # That invariant is *not* enforced on write, because the two maps are filled
    # independently and requiring provenance to land first would make correct
    # code fail on ordering alone; use :meth:`inconsistent_confidence` instead.
    confidence: dict[str, str] = field(
        default_factory=dict, metadata={"type": None, "level": RECORD}
    )

    # Also not part of the field set: this experiment's runs, the download menu
    # under a search hit. One dict per run, reusing the schema's own field names
    # (``run_accession``, ``read_count``, ``base_count``, ``first_public``), with
    # dates left as the archive's raw strings — this list is an attachment, not
    # an indexed field, so it stays plain JSON primitives.
    #
    # The scalar RUN-level fields above (``run_accession``, ``run_alias``,
    # ``submitted_format``, ``submitted_read_type``) have no single value at this
    # grain and from_project leaves them empty; ``read_count`` and ``base_count``
    # do, and carry the experiment's totals summed across these runs.
    runs: list[dict] = field(
        default_factory=list, metadata={"type": None, "level": RUN}
    )

    # -- validation ------------------------------------------------------- #
    def __post_init__(self):
        # Coercion itself happens in __setattr__, which the generated __init__
        # has already run for every field by this point.
        if not self.id:
            raise ValueError("id is required and cannot be blank")

    def __setattr__(self, name, value):
        """Coerce on every write, not just at construction.

        Reconstruction fills fields one at a time (``rec.collection_date = …``),
        so validating only in ``__init__`` would leave the common path unchecked.
        """
        declared = _FIELD_TYPES.get(name)
        if declared not in (None, str) and _is_missing_value(value):
            # A count or a date has no room for a sentinel: storing a string
            # there would break every comparison downstream. `collection_date`
            # is the one genuine casualty — a cell line arguably has no
            # collection date — so that reason cannot currently be recorded.
            raise ValueError(
                f"{name}: {value.strip().lower()!r} cannot be stored in a "
                f"{declared.__name__} field; leave it None"
            )
        if declared is int:
            value = _parse_int(value, name)
        elif declared is datetime:
            value = _parse_date(value, name)
        elif declared is str:
            value = _clean(value)
        elif name == "provenance":
            # wrap on assignment so item writes are validated from here on
            value = _ProvenanceMap(value or {})
        elif name == "confidence":
            value = _ConfidenceMap(value or {})
        object.__setattr__(self, name, value)

    # -- schema introspection --------------------------------------------- #
    @classmethod
    def field_names(cls) -> list[str]:
        """The schema field names, in order. Excludes the sidecar attributes."""
        return [f.name for f in _schema_fields()]

    @classmethod
    def field_type(cls, name: str) -> type:
        """The declared Python type of a field: ``str``, ``int`` or ``datetime``."""
        return _FIELD_TYPES[name]

    @classmethod
    def level(cls, name: str) -> str:
        return _LEVELS[name]

    @classmethod
    def fields_at_level(cls, level: str) -> list[str]:
        """Field names describing one level of the hierarchy.

        The reconstruction stage should batch by this: study-level fields are
        one LLM call per *study*, sample-level fields one call per *sample*.
        Asking per run would pay for the same answer once per run.
        """
        if level not in LEVELS:
            raise ValueError(f"unknown level {level!r}; expected one of {LEVELS}")
        return [f.name for f in _schema_fields() if f.metadata["level"] == level]

    # -- coverage --------------------------------------------------------- #
    def filled(self) -> list[str]:
        """Field names holding a value—the reconstruction's per-record yield."""
        return [n for n in self.field_names() if getattr(self, n) is not None]

    def missing(self) -> list[str]:
        return [n for n in self.field_names() if getattr(self, n) is None]

    def declared_missing(self) -> dict[str, str]:
        """``{field: term}`` for fields answered with a :data:`MISSING_VALUES` term.

        Resolved, but not data. These count as :meth:`filled`, because an answer
        was reached and no further reconstruction is owed — which is exactly what
        separates them from :meth:`missing`, where nothing has been concluded at
        all. Subtract them from :meth:`coverage` for data density rather than
        progress::

            density = (len(r.filled()) - len(r.declared_missing())) / len(r.field_names())
        """
        return {
            n: getattr(self, n)
            for n in self.field_names()
            if getattr(self, n) in MISSING_VALUES
        }

    def coverage(self) -> float:
        """Fraction of schema fields resolved, 0.0–1.0.

        Counts :data:`MISSING_VALUES` answers — they are settled, not missing.
        See :meth:`declared_missing` to net them out.
        """
        return len(self.filled()) / len(self.field_names())

    def inconsistent_confidence(self) -> list[str]:
        """Fields carrying a confidence that nothing justifies.

        Two ways to earn a place here:

        * **Nobody chose.** A confidence is only meaningful for the two
          ``inferred_*`` classes. On a ``direct`` or ``harmonized`` field the
          value came from the archive or a key mapping, and on a field with no
          provenance at all there is nothing to be confident *about*.
        * **Nothing was concluded.** A ``None`` value means reconstruction has
          not run on the field, or ran and reached no answer. A field the model
          did resolve carries a value — including a :data:`MISSING_VALUES` term
          such as ``not applicable`` or ``not collected``, each of which *is* an
          answer, and one that can be wrong, so each is worth a confidence.

        Run this over a finished batch. Enforcing it on write would punish
        setting confidence before provenance, which is only an ordering choice.
        """
        inferred = {"inferred_from_text", "inferred_from_paper"}
        return [
            name
            for name in self.confidence
            if self.provenance.get(name) not in inferred or getattr(self, name) is None
        ]

    # -- serialization ---------------------------------------------------- #
    def to_dict(self, omit_none: bool = True) -> dict:
        """JSON-ready mapping; datetimes become ISO-8601 strings.

        Includes ``provenance`` and ``runs``. Pass ``omit_none=False`` for a
        square table where every record carries every column.
        """
        out = {}
        for name in self.field_names():
            value = getattr(self, name)
            if value is None and omit_none:
                continue
            out[name] = value.isoformat() if isinstance(value, datetime) else value
        if self.runs or not omit_none:
            out["runs"] = [dict(run) for run in self.runs]
        if self.provenance or not omit_none:
            out["provenance"] = dict(self.provenance)
        if self.confidence or not omit_none:
            out["confidence"] = dict(self.confidence)
        return out

    @classmethod
    def from_dict(cls, d: dict, strict: bool = False) -> "TargetSchema":
        """Rebuild from a :meth:`to_dict` mapping.

        Unknown keys are ignored by default, so a file written by a later
        version of the schema still loads. ``strict=True`` raises on them
        instead—use it when parsing LLM output, where an unknown key means the
        model invented a field rather than that the schema moved on.
        """
        known = set(cls.field_names())
        if strict:
            unknown = set(d) - known - {"provenance", "confidence", "runs"}
            if unknown:
                raise ValueError(f"unknown field(s): {', '.join(sorted(unknown))}")
        record = cls(**{k: v for k, v in d.items() if k in known})
        record.provenance = dict(d.get("provenance") or {})
        record.confidence = dict(d.get("confidence") or {})
        record.runs = [dict(run) for run in (d.get("runs") or [])]
        return record

    # -- bridge from the raw archive object -------------------------------- #
    @classmethod
    def from_project(cls, project, include_provenance: bool = True) -> list["TargetSchema"]:
        """Seed one record per experiment from a built :class:`project.Project`.

        Sets **only** fields SRA states outright—no harmonization of the
        submitter attribute bag, no inference. Every value here is `direct`
        provenance, which makes these the authoritative anchors to pass into the
        LLM stage rather than values it is free to overwrite.

        Two ENA naming traps are handled here. ENA's ``study_accession`` is the
        *BioProject* (``PRJNA…``) and ``secondary_study_accession`` is the SRP;
        likewise ``sample_accession`` is the BioSample (``SAMN…``) and
        ``secondary_sample_accession`` is the SRS. `Project` stores them the
        other way round, so mapping them positionally would silently swap them.

        Each experiment's runs are attached as :attr:`runs`, with ``read_count``
        and ``base_count`` summed over them and ``first_public`` taken as the
        earliest run's release date. Release dates are compared as strings, which
        is chronological given SRA's fixed-width format — the same comparison
        :meth:`project.Project._note_published` makes.

        Falls back to sample- then study-level records when the Project was built
        without experiments (``include_experiments=False``, or summary mode), so
        a summary-only study still yields one usable stub. ``id`` follows it
        down: experiment accession, else sample, else study. Building without
        runs costs only the ``runs`` list and the two counts.

        **Pooled experiments get a composite id.** One SRX over several SRS
        (a multiplexed library, the reason ``Experiment.sample_ids`` is a list)
        expands to one record per sample, exactly as
        :meth:`project.Project.to_dataframe` does — but ``id`` has to stay
        unique, so reusing the experiment accession across them would silently
        drop every sample in the pool but one. Those records are keyed
        ``<experiment>.<sample>`` instead. Non-pooled studies, which are the
        overwhelming majority, keep the bare accession and so stay directly
        comparable against ENA's own ids.
        """
        records: list[TargetSchema] = []

        def _new(rec_id, sample=None, experiment=None):
            r = cls(id=rec_id)
            r.study_accession = project.bioproject
            r.secondary_study_accession = project.accession
            r.study_title = project.title
            r.description = project.abstract
            if experiment is not None:
                r.experiment_accession = experiment.accession
                r.experiment_title = experiment.title
                r.library_strategy = experiment.library_strategy
                r.library_source = experiment.library_source
                r.library_selection = experiment.library_selection
                r.library_layout = experiment.library_layout
                r.instrument_platform = experiment.platform
                r.instrument_model = experiment.instrument_model
                r.runs = [
                    {
                        "run_accession": run.accession,
                        "read_count": run.total_spots,
                        "base_count": run.total_bases,
                        "first_public": run.published,
                    }
                    for run in experiment.runs
                ]
                spots = [x.total_spots for x in experiment.runs if x.total_spots is not None]
                bases = [x.total_bases for x in experiment.runs if x.total_bases is not None]
                dates = [x.published for x in experiment.runs if x.published]
                # a run with no reported count must not read as a zero total
                r.read_count = sum(spots) if spots else None
                r.base_count = sum(bases) if bases else None
                r.first_public = min(dates) if dates else None
            if sample is not None:
                r.sample_accession = sample.biosample
                r.secondary_sample_accession = sample.accession
                r.sample_title = sample.title
                r.scientific_name = sample.scientific_name
                r.tax_id = sample.taxon_id
            if r.first_public is None and project.published:
                # study-level release date: the oldest indexed record's, which
                # is not provably the earliest run (see PIPELINE.md §3).
                r.first_public = project.published
            if include_provenance:
                r.provenance = {n: "direct" for n in r.filled() if n != "id"}
            return r

        for experiment in project.experiments:
            samples = project.samples_of(experiment) or [None]
            pooled = len(samples) > 1  # several SRS share this SRX's records

            def _id(base, sample):
                # suffix only when the base would otherwise repeat across the
                # pool; an unresolvable sample cannot disambiguate anything
                if not pooled or sample is None:
                    return base
                return f"{base}.{sample.accession}"

            for sample in samples:
                records.append(_new(_id(experiment.accession, sample), sample, experiment))
        if not records:
            for accession, sample in project.samples.items():
                records.append(_new(accession, sample))
        if not records:
            records.append(_new(project.accession))
        return records

    def __repr__(self):
        filled = len(self.filled())
        return (
            f"TargetSchema({self.id!r}, "
            f"study={self.secondary_study_accession!r}, "
            f"sample={self.secondary_sample_accession!r}, "
            f"runs={len(self.runs)}, "
            f"{filled}/{len(self.field_names())} fields)"
        )


def _schema_fields():
    """The declared schema fields — everything except the sidecar attributes."""
    return [f for f in _dataclass_fields(TargetSchema) if f.metadata.get("type")]


_FIELD_TYPES: dict[str, type] = {f.name: f.metadata["type"] for f in _schema_fields()}
_LEVELS: dict[str, str] = {f.name: f.metadata["level"] for f in _schema_fields()}