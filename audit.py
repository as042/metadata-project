"""Audit the `confidence` label against the evidence it claims to come from.

:data:`schema.CONFIDENCE_LEVELS` records *where a value came from* rather than
how likely it is to be right, and the top level makes a claim that can be
checked mechanically: `high` means the value appears in the evidence **word for
word**. This module checks it — no gold set, no model, no spend.

    import audit
    audit.verbatim_report("datasets/test3/test_reconstructed.json")

That is the point of the mechanical axis. An epistemic confidence could only be
validated against labelled data nobody has; "is this string in that string" can
be validated against the run's own inputs.

WHAT THIS DOES NOT TELL YOU
A quoted value can still be wrong — the model may have quoted the *wrong span*,
which is exactly how the `host` = study-organism bug looked. A high verbatim
rate means the label is honest about its own reach, not that the answers are
correct. Scoring accuracy still needs a labelled set.

TWO MODES, AND WHY THE DEFAULT IS THE WEAKER ONE
:func:`reconstruct._evidence` puts the sample's raw attribute bag into the
prompt, and that bag is **not** saved in the reconstructed output. So:

* **offline** (default) rebuilds what evidence the record itself still carries —
  titles, organism, strategy. It is blind to the attribute bag, so it can prove
  a value was quoted but cannot prove one was not. Non-matches are reported as
  ``unknown``, never as failures. Free, no network.
* **expand=True** re-fetches each study from SRA and rebuilds the exact string
  the model saw, so a non-match is a real ``not_verbatim``. Costs NCBI requests
  (free, no tokens) and assumes SRA still returns what it returned at run time.

Collapsing ``unknown`` into ``not_verbatim`` would invent failures wholesale —
most values legitimately come from the attribute bag — so the two are kept
apart everywhere, including in the totals.
"""

from __future__ import annotations

import collections
import json
import re

from schema import MISSING_VALUES, STUDY, TargetSchema

# Verdicts. `unknown` exists because the offline mode's evidence is partial:
# "I could not see where this came from" is not "this was not quoted".
VERBATIM = "verbatim"
NOT_VERBATIM = "not_verbatim"
UNKNOWN = "unknown"

# Only layer 3 can be audited. Layer 4 sends up to `reconstruct.PAPER_MAX_CHARS`
# of full text that is never persisted, so its answers have no recoverable
# evidence — they are counted and reported, not silently dropped.
AUDITABLE_PROVENANCE = "inferred_from_text"
UNAUDITABLE_PROVENANCE = "inferred_from_paper"

_NON_ALNUM = re.compile(r"[^0-9a-z]+")


def normalize(text: str) -> str:
    """Casefold, reduce every non-alphanumeric run to one space, pad with spaces.

    The padding is what makes a plain ``in`` test respect token boundaries, so
    ``"male"`` does not match ``"female"``. Punctuation is flattened rather than
    stripped because the attribute bag arrives as :func:`json.dumps` — the
    evidence holds ``{"age": "8 weeks"}`` and the answer is ``8 weeks``, and
    both normalize to the same tokens once quotes and colons become spaces.
    """
    return f" {_NON_ALNUM.sub(' ', str(text).casefold()).strip()} "


def is_verbatim(value, evidence: str) -> bool:
    """Whether ``value`` appears in ``evidence`` word for word, modulo formatting."""
    needle = normalize(value)
    if needle == " ":
        return False
    return needle in normalize(evidence)


# The record fields that also appear in `reconstruct._evidence`, and so survive
# to disk. The attribute bag — the bulk of the real evidence — does not, which
# is what makes the offline mode sound-positive only.
_RECOVERABLE = ("study_title", "sample_title", "scientific_name",
                "experiment_title", "library_strategy", "description")


def _record_evidence(record: TargetSchema, exclude: str | None = None) -> str:
    """The evidence still recoverable from a finished record, with no network.

    ``exclude`` drops the field being audited. Without it a field that is *part
    of* the evidence trivially quotes itself: ``description`` checked against a
    string containing ``description`` matches every time, and layer 3 does fill
    those fields, so the offline verbatim rate would be inflated by exactly the
    fields it can say least about.
    """
    return "\n".join(
        str(getattr(record, name))
        for name in _RECOVERABLE
        if name != exclude and getattr(record, name, None)
    )


def _verdict(value, evidence: str, exact: bool) -> str:
    """Classify one value against the evidence available for it."""
    if is_verbatim(value, evidence):
        return VERBATIM          # sound in both modes: a match is a match
    return NOT_VERBATIM if exact else UNKNOWN


def _evidence_by_record(records, studies_path, max_records):
    """``{record_id: evidence}`` rebuilt exactly, by re-fetching each study.

    Mirrors the grouping in :func:`reconstruct.infer_from_text`: study-level
    fields saw only :func:`reconstruct._study_evidence`, everything else saw the
    per-sample string, grouped by ``secondary_sample_accession``. Imported
    lazily so the offline path never pulls in the network stack.
    """
    import dataset
    import reconstruct
    from project import load_studies

    by_study = collections.defaultdict(list)
    for record in records:
        by_study[record.secondary_study_accession or record.study_accession].append(record)

    out: dict[str, tuple[str, str]] = {}
    for study in load_studies(studies_path):
        group = by_study.get(study.accession)
        if not group:
            continue
        project, note = dataset._expand(study, max_records)
        if note:
            print(f"  {study.accession}: {note.strip()}", flush=True)
        study_text = reconstruct._study_evidence(project)
        by_sample = collections.defaultdict(list)
        for record in group:
            by_sample[record.secondary_sample_accession].append(record)
        for sample_id, sample_group in by_sample.items():
            sample = project.samples.get(sample_id) if sample_id else None
            sample_text = reconstruct._evidence(project, sample, sample_group)
            for record in sample_group:
                out[record.id] = (study_text, sample_text)
    return out


def audit_records(records, evidence_by_record=None) -> dict:
    """Classify every auditable confidence label. Returns counts, prints nothing.

    ``evidence_by_record`` maps record id to ``(study_evidence, sample_evidence)``
    and switches on the exact mode; without it each record is checked against
    what it still carries, and non-matches come back ``unknown``.
    """
    exact = evidence_by_record is not None
    study_fields = set(TargetSchema.fields_at_level(STUDY))
    missing_terms = {normalize(term) for term in MISSING_VALUES}

    by_level = collections.Counter()
    verdicts = collections.Counter()
    real_verdicts = collections.Counter()
    real_by_level = collections.Counter()
    per_field = collections.defaultdict(collections.Counter)
    missing_value_verdicts = collections.Counter()
    unauditable = 0

    for record in records:
        for name, level in record.confidence.items():
            provenance = record.provenance.get(name)
            if provenance == UNAUDITABLE_PROVENANCE:
                unauditable += 1
                continue
            if provenance != AUDITABLE_PROVENANCE:
                continue
            value = getattr(record, name, None)
            if value is None:
                continue
            if exact:
                study_text, sample_text = evidence_by_record.get(record.id, ("", ""))
                evidence = study_text if name in study_fields else sample_text
            else:
                evidence = _record_evidence(record, exclude=name)

            verdict = _verdict(value, evidence, exact)
            by_level[level] += 1
            verdicts[(level, verdict)] += 1
            # A missing-value term is a *determination*, so "not quoted" is its
            # normal state and mixing it in drowns the real signal: on the first
            # run these were 274 of 364 `high` labels and every per-field
            # offender was simply a field answered "not applicable". Counted
            # separately so the field table shows genuine wrong-span quoting.
            if normalize(value) in missing_terms:
                missing_value_verdicts[(level, verdict)] += 1
            else:
                real_by_level[level] += 1
                real_verdicts[(level, verdict)] += 1
                per_field[name][(level, verdict)] += 1

    return {
        "exact": exact,
        "by_level": by_level,
        "verdicts": verdicts,
        "real_by_level": real_by_level,
        "real_verdicts": real_verdicts,
        "per_field": per_field,
        "missing_values": missing_value_verdicts,
        "unauditable": unauditable,
        "records": len(records),
    }


def format_report(result: dict, top_fields: int = 10) -> str:
    """Render :func:`audit_records` output as a plain-text report."""
    exact = result["exact"]
    by_level, verdicts = result["by_level"], result["verdicts"]
    total = sum(by_level.values())
    lines = []

    mode = "exact (evidence re-fetched)" if exact else "offline (partial evidence)"
    lines.append(f"Confidence audit — {result['records']:,} records, "
                 f"{total:,} auditable labels, mode: {mode}")
    if not exact:
        lines.append("  'unknown' means the attribute bag was not available to check "
                     "against — it is not a failure.")
    if result["unauditable"]:
        lines.append(f"  {result['unauditable']:,} labels from "
                     f"{UNAUDITABLE_PROVENANCE} skipped: paper text is not persisted.")
    if not total:
        lines.append("  nothing to audit.")
        return "\n".join(lines)

    # The histogram first: a label that never varies cannot correlate with
    # anything, and that was the original finding this whole change addresses.
    lines.append("\nDistribution")
    for level in ("high", "medium", "low"):
        n = by_level.get(level, 0)
        lines.append(f"  {level:7} {n:8,}  {100 * n / total:5.1f}%")

    def table(title, levels, counts):
        out = [title,
               f"  {'level':7} {'n':>7}   {VERBATIM:>16} {NOT_VERBATIM:>16} {UNKNOWN:>16}"]
        for level in ("high", "medium", "low"):
            n = levels.get(level, 0)
            if not n:
                out.append(f"  {level:7} {0:7,}")
                continue
            cells = " ".join(
                f"{counts.get((level, v), 0):,} "
                f"({100 * counts.get((level, v), 0) / n:.1f}%)".rjust(16)
                for v in (VERBATIM, NOT_VERBATIM, UNKNOWN)
            )
            out.append(f"  {level:7} {n:7,}   {cells}")
        return out

    # Real values first — this is the table that says whether `high` means what
    # it claims. The missing-value terms below are a different question.
    lines += table("\nVerbatim check — real values (missing-value terms excluded)",
                   result["real_by_level"], result["real_verdicts"])

    if exact:
        real_high = result["real_by_level"].get("high", 0)
        broken = result["real_verdicts"].get(("high", NOT_VERBATIM), 0)
        if real_high:
            lines.append(f"\n  'high' on a real value that is not actually quoted: "
                         f"{broken:,}/{real_high:,} ({100 * broken / real_high:.1f}%)")

    mv = result["missing_values"]
    if sum(mv.values()):
        mv_levels = collections.Counter()
        for (level, _), count in mv.items():
            mv_levels[level] += count
        lines += table(
            "\nMissing-value terms — 'not applicable' and friends."
            "\n  These are determinations, not quotes: 'not quoted' is their normal state."
            "\n  Under the directness rules they belong in 'low' unless the evidence says them.",
            mv_levels, mv)

    # Fields whose `high` labels least survive the check — the place a real
    # extraction bug shows up, the way `host` did at layer 2.
    if exact:
        offenders = []
        for name, counts in result["per_field"].items():
            high = sum(c for (lvl, _), c in counts.items() if lvl == "high")
            bad = counts.get(("high", NOT_VERBATIM), 0)
            if high >= 5 and bad:
                offenders.append((bad / high, bad, high, name))
        if offenders:
            lines.append(f"\nFields whose 'high' least survives the check "
                         f"(real values only, top {top_fields}, n>=5)")
            for rate, bad, high, name in sorted(offenders, reverse=True)[:top_fields]:
                lines.append(f"  {name:34} {bad:5,}/{high:<6,} not quoted  {100*rate:5.1f}%")

    return "\n".join(lines)


def verbatim_report(records_path, studies_path=None, expand=False,
                    max_records=None, top_fields: int = 10) -> dict:
    """Audit a reconstructed dataset and print the report. Returns the raw counts.

    ``expand=True`` needs ``studies_path`` — the filtered/studies JSON the run
    was built from — and re-fetches each study from SRA to rebuild the exact
    evidence. That costs NCBI requests and no tokens. Without it the audit is
    offline and non-matches are ``unknown``.
    """
    import dataset

    # Checked before the records are loaded: reading a large dataset only to
    # refuse the arguments wastes the one expensive step of an offline audit.
    if expand and not studies_path:
        raise ValueError(
            "expand=True needs studies_path — the studies JSON the records "
            "were reconstructed from — to rebuild the evidence"
        )

    records = dataset.load_records(records_path)
    evidence = None
    if expand:
        if max_records is None:
            max_records = dataset.MAX_RECORDS
        print("Re-fetching studies to rebuild evidence (NCBI only, no tokens)...",
              flush=True)
        evidence = _evidence_by_record(records, studies_path, max_records)

    result = audit_records(records, evidence)
    print(format_report(result, top_fields=top_fields))
    return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        sys.exit("usage: python audit.py <records.json> [studies.json]")
    verbatim_report(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None,
                    expand=len(sys.argv) > 2)
