"""Build the study shortlist that feeds LLM metadata reconstruction.

Three stages. The first two write a JSON array of ``Project`` dicts; the third
converts those into the flat target schema:

1. :func:`save_recent_studies`—sample studies from SRA and save their summaries.
2. :func:`filter_oa_studies`  —keep only the ones with an open-access paper,
   caching the accessibility classes it computes into the saved JSON.
3. :func:`save_reconstructed_records`—expand each study to full depth and emit
   one :class:`schema.TargetSchema` record per experiment.

Stages 1 and 3 checkpoint as they go, so a run over thousands of studies
survives a crash or a Ctrl-C: rerun the same call and it picks up where it
stopped.
"""

from datetime import date
import collections
import json
import os
import time

from project import classify_publication
from project import search_studies
from project import scan_iter
from project import load_studies
from project import Project
from project import _load_json
from schema import TargetSchema
import claude
import reconstruct


# Studies bigger than this are continuously-appended surveillance umbrellas
# (PulseNet, GenomeTrakr, growing cohorts). Their samples aren't described by the
# linked papers, so they're poor LLM context however accessible the paper is.
MAX_RECORDS = 5000

# Studies scanned between checkpoint writes. Small enough that a crash costs
# seconds of work, large enough that the rewrite cost stays negligible.
CHECKPOINT_EVERY = 25

# esummary pages between search-progress lines. search_studies reports every page,
# which is ~2,700 of them on a 5,000-study harvest—enough to bury everything else.
SEARCH_PROGRESS_EVERY = 20

# Pacing between classify_publication calls. Deliberately not the NCBI rate: most
# of that work hits Europe PMC and Unpaywall, who are third parties that never
# granted us a raised limit, so an NCBI api_key must not speed this up.
CLASSIFY_SLEEP = 0.34

# Measured dollar cost per unit of paid work on the current settings (Haiku
# 4.5, batched, the ~4,000-token prompt). Both figures held to within 1% across
# a 52-record run and a 1,782-record one, which is why an estimate is possible
# at all — the unit cost is stable, it is the *unit count* that surprises.
#
# Layer 3 makes roughly one call per sample, and samples track experiments
# closely (1,601 samples across 1,664 experiments in the run that overspent), so
# a study's `record_count` is a good proxy without expanding it first.
COST_PER_RECORD_TEXT = 0.0035   # layer 3
COST_PER_STUDY_PAPER = 0.013    # layer 4: one paper fetch + one call per study

# A run estimated above this refuses to start. The number that prompted it: a
# 20-study harvest yielding just 2 open-access studies looked cheap and cost $7,
# because one of those studies held 1,664 experiments. Studies are the unit you
# authorise in; samples are the unit that bills.
MAX_SPEND = 1.00

_CHECKPOINT_VERSION = 1


class SpendLimitExceeded(RuntimeError):
    """Raised before any paid work when the estimate exceeds ``max_spend``."""


def estimate_reconstruction_cost(studies, harmonize=False, from_text=False,
                                 from_paper=False, max_records=MAX_RECORDS):
    """``(dollars, report)`` for reconstructing ``studies``. Costs nothing to call.

    Everything it needs is already in the saved summaries: ``record_count`` per
    study and the publication classes stage 2 cached. No network, no model.

    Studies over ``max_records`` are counted as one stub record, matching what
    :func:`_expand` actually does with them, and layer 4 is counted only for
    studies that have a publication classified ``oa`` — the rest never fetch.
    """
    lines, records, oa_studies = [], 0, 0
    for study in studies:
        count = study.record_count or 0
        billable = 1 if (max_records is not None and count > max_records) else count
        has_oa = any(p.accessibility_type == "oa" for p in study.publications)
        records += billable
        oa_studies += bool(has_oa)
        per = billable * COST_PER_RECORD_TEXT * bool(from_text)
        per += COST_PER_STUDY_PAPER * bool(from_paper) * bool(has_oa)
        lines.append(
            f"  {study.accession:12} {billable:>7,} records"
            f"{'  (oversized -> stub)' if billable != count else '':<22}"
            f"{'  +paper' if has_oa and from_paper else '':<8} ${per:7.3f}"
        )
    total = (records * COST_PER_RECORD_TEXT * bool(from_text)
             + oa_studies * COST_PER_STUDY_PAPER * bool(from_paper))
    header = (f"Estimated cost: ${total:.2f}  "
              f"({records:,} records x ${COST_PER_RECORD_TEXT} layer 3"
              f"{f' + {oa_studies} papers x ${COST_PER_STUDY_PAPER} layer 4' if from_paper else ''})")
    if not (from_text or from_paper):
        header = "Estimated cost: $0.00 (no model layers enabled)"
        lines = []
    return total, "\n".join([header, *lines])


def _checkpoint_path(path):
    return f"{path}.partial"


def _write_json(path, payload):
    """Write JSON atomically—a crash mid-write must not destroy a checkpoint."""
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False)
    os.replace(tmp, path)  # atomic on POSIX


def _load_checkpoint(path, params):
    """Return a resumable checkpoint for these params, or None.

    A checkpoint is only valid for the exact search that produced it: the study
    list is a *random* sample, so resuming one harvest into a differently
    parameterised one would silently mix two samples.
    """
    try:
        with open(path, encoding="utf-8") as file:
            state = json.load(file)
    except (OSError, ValueError):
        return None
    if state.get("version") != _CHECKPOINT_VERSION or state.get("params") != params:
        return None
    return state


def save_recent_studies(
    path="./recent_studies.json",
    max_studies=50,
    after_date=date(2020, 1, 1),
    before_date=date(2024, 1, 1),
    sort="random",
    max_records=MAX_RECORDS,
    resume=True,
    checkpoint_every=CHECKPOINT_EVERY,
):
    """Sample studies from SRA and save their summaries as a JSON array.

    ``sort="random"`` spreads the sample across the whole date window, which is
    what turns up studies old enough to have a linked paper. ``max_records`` drops
    oversized umbrella studies (see :data:`MAX_RECORDS`); pass None to keep them.

    Progress is checkpointed to ``<path>.partial`` every ``checkpoint_every``
    studies and removed on success. With ``resume=True`` (the default) a rerun of
    the *same* call continues from that file instead of re-enumerating and
    re-scanning; a checkpoint written by different search parameters is ignored.
    """
    params = {
        "max_studies": max_studies,
        "after_date": str(after_date),
        "before_date": str(before_date),
        "sort": sort,
        "max_records": max_records,
    }
    ckpt = _checkpoint_path(path)
    state = _load_checkpoint(ckpt, params) if resume else None

    if state is not None:
        studies = state["accessions"]
        done = state["done"]
        print(f"Resuming {ckpt}: {len(done)} of {len(studies)} studies already scanned")
    else:
        print(f"Searching for {max_studies} studies ...")
        pages = 0

        def on_page(found, scanned):
            nonlocal pages
            pages += 1
            if pages % SEARCH_PROGRESS_EVERY == 0:
                print(f"  search: {found} studies from {scanned} records", flush=True)

        studies = search_studies(
            after_date=after_date,
            before_date=before_date,
            max_studies=max_studies,
            sort=sort,
            progress=on_page,
        )
        done = {}
        print("Number of studies searched:", len(studies))
        _write_json(ckpt, {"version": _CHECKPOINT_VERSION, "params": params,
                           "accessions": studies, "done": done})

    remaining = [a for a in studies if a not in done]
    kept = sum(1 for v in done.values() if v is not None)
    for acc, project, _error in scan_iter(
        remaining, include_publications=True, max_records=max_records
    ):
        done[acc] = project.to_dict() if project is not None else None
        if project is not None:
            kept += 1
        if len(done) % checkpoint_every == 0:
            _write_json(ckpt, {"version": _CHECKPOINT_VERSION, "params": params,
                               "accessions": studies, "done": done})
            print(f"  scanned {len(done)}/{len(studies)}, kept {kept}", flush=True)

    summaries = [d for d in done.values() if d is not None]
    print("Number of projects scanned and errors:",
          len(summaries), len(done) - len(summaries))
    for d in summaries:
        print(f"{d['accession']}: published {d['published']}, "
              f"{d['record_count']} records")

    text = json.dumps(summaries, indent=2, ensure_ascii=False)
    with open(path, "w", encoding="utf-8") as file:
        file.write(text)
    # Only once the real output is safely on disk.
    try:
        os.remove(ckpt)
    except OSError:
        pass


def filter_oa_studies(
    in_path="./recent_studies.json",
    out_path="./oa_studies.json",
    sleep=CLASSIFY_SLEEP,
):
    """Keep the studies with at least one open-access publication.

    Every class that gets computed is cached onto the publication and written to
    ``out_path``, so re-running over the output costs no network calls. Papers
    after the first open-access one in a study are left unclassified (``None``) —
    they can't change the outcome.

    ``sleep`` paces the classifier between studies (see :data:`CLASSIFY_SLEEP`).
    """
    projects = load_studies(in_path)
    print("Studies loaded:", len(projects))

    keep = []
    for p in projects:
        # Stop at the first OA paper: classifying the rest can't change the
        # outcome, and umbrella studies list a dozen publications.
        has_oa = False
        for pub in p.publications:
            if pub.accessibility_type is None:
                pub.accessibility_type = classify_publication(pub.id)
                time.sleep(sleep)
            if pub.accessibility_type == "oa":
                has_oa = True
                break
        print(f"{p.accession}: {[pub.accessibility_type for pub in p.publications]}")
        if has_oa:
            keep.append(p)

    print("Studies with an open-access paper:", len(keep), "of", len(projects))
    text = json.dumps([p.to_dict() for p in keep], indent=2, ensure_ascii=False)
    with open(out_path, "w", encoding="utf-8") as file:
        file.write(text)


def _expand(study, max_records):
    """Full-depth rebuild of one study; returns ``(project, note)``.

    Falls back to the summary already loaded from disk rather than failing the
    run: a summary still yields a usable study-level record, so a flaky fetch
    costs depth, not the study. ``note`` explains any fallback, or is blank.

    The size check runs *before* the build, not after. Expanding a study fetches
    every one of its records in batches of 300, so an oversized umbrella study
    is thousands of requests — the guard has to precede them (PIPELINE.md §3).
    """
    if max_records is not None and study.record_count and study.record_count > max_records:
        return study, f" [summary only: {study.record_count} records > {max_records}]"
    try:
        # include_publications=False saves a BioProject fetch per study: the
        # summary loaded from disk already carries the publications, complete
        # with the accessibility classes stage 2 paid to compute.
        full = Project(study.accession, include_publications=False, max_records=max_records)
    except Exception as exc:  # noqa: BLE001 - one bad study must not abort the run
        return study, f" [summary only: {type(exc).__name__}: {exc}]"
    if full.oversized:
        return study, f" [summary only: {full.record_count} records > {max_records}]"
    # ...but they have to be carried onto the expanded object, which is what the
    # cascade actually sees. Without this the paper layer finds no `oa`
    # publication on any study and silently contributes nothing — it did exactly
    # that on its first run, returning 0 fields and making 0 calls.
    full.publications = study.publications
    return full, ""


def _reconstruct(source, layers):
    """Run the cascade over one study; returns ``(records, report, note)``.

    Mirrors :func:`_expand`'s contract: a study whose model layers fail keeps
    its direct-only records rather than taking the run down with it. Over
    hundreds of studies and thousands of paid calls, a refusal or an exhausted
    retry budget somewhere is close to certain, and losing one study's inference
    is a far better outcome than losing the run's remaining progress.
    """
    try:
        records, report = reconstruct.reconstruct(source, **layers)
    except Exception as exc:  # noqa: BLE001 - one bad study must not abort the run
        if not any(layers.values()):
            raise  # nothing to fall back from: the direct layer itself failed
        records = TargetSchema.from_project(source)
        return records, {"direct": sum(len(r.provenance) for r in records)}, (
            f" [direct only: {type(exc).__name__}: {exc}]"
        )
    return records, report, ""


def _richness(study):
    """Sort key for choosing between two copies of the same study.

    Prefers the copy carrying the most *classified* publications. Those classes
    are the expensive part of stage 2 — each one cost up to three third-party
    lookups — so a copy that has them should never lose to one that does not.
    Falls back to publication count, then to record count.
    """
    classified = sum(1 for p in study.publications if p.accessibility_type)
    return (classified, len(study.publications), study.record_count or 0)


def combine_studies(sources, out_path=None, max_records=MAX_RECORDS):
    """Merge saved study files into one de-duplicated corpus.

    ``sources`` is a list of paths (or anything :func:`project.load_studies`
    accepts). Returns the merged list of :class:`project.Project` objects and,
    with ``out_path``, writes them as a JSON array in the same shape every other
    stage reads — so the result drops straight into
    :func:`save_reconstructed_records` and stages 1 and 2 never run again.

    **De-duplicated by accession.** Harvests are random samples over the same
    archive, so overlap between them is expected rather than exceptional. Where
    the same study appears twice the richer copy wins (see :func:`_richness`) —
    losing a cached publication class would mean paying for it again.

    Output is sorted by accession, so the file is stable across runs and diffs
    cleanly when you add a harvest to it.
    """
    merged, seen_in, dupes, conflicts = {}, collections.Counter(), 0, 0
    for source in sources:
        studies = load_studies(source)
        seen_in[str(source)] = len(studies)
        for study in studies:
            existing = merged.get(study.accession)
            if existing is None:
                merged[study.accession] = study
                continue
            dupes += 1
            if _richness(study) > _richness(existing):
                merged[study.accession] = study
                conflicts += 1

    out = [merged[a] for a in sorted(merged)]
    print(f"Combined {len(sources)} file(s):")
    for name, n in seen_in.items():
        print(f"  {n:>6} studies  {name}")
    print(f"  {dupes:>6} duplicate accession(s) dropped"
          f"{f' ({conflicts} replaced by a richer copy)' if conflicts else ''}")
    print(f"  {len(out):>6} unique studies")

    records = sum(s.record_count or 0 for s in out)
    oversized = [s for s in out if max_records and (s.record_count or 0) > max_records]
    print("\nWhat reconstruction would face:")
    print(f"  {records:>7,} records across {len(out)} studies "
          f"(largest: {max((s.record_count or 0) for s in out):,})")
    if oversized:
        print(f"  {len(oversized):>7} study(ies) over max_records={max_records:,} "
              f"-> stub records only: {', '.join(s.accession for s in oversized[:5])}")
    cost, report = estimate_reconstruction_cost(
        out, from_text=True, from_paper=True, max_records=max_records
    )
    print(f"  {report.splitlines()[0]}")

    if out_path:
        text = json.dumps([s.to_dict() for s in out], indent=2, ensure_ascii=False)
        with open(out_path, "w", encoding="utf-8") as file:
            file.write(text)
        print(f"\nWrote {out_path}")
    return out


def save_reconstructed_records(
    in_path="./oa_studies.json",
    out_path="./records.json",
    expand=True,
    max_records=MAX_RECORDS,
    resume=True,
    checkpoint_every=CHECKPOINT_EVERY,
    harmonize=False,
    from_text=False,
    from_paper=False,
    max_spend=MAX_SPEND,
):
    """Reconstruct saved studies into target-schema records and save them.

    Reads a JSON array of ``Project`` dicts (either stage's output), optionally
    re-fetches each study at full depth, and writes a flat JSON array of
    :class:`schema.TargetSchema` dicts — one record per experiment, keyed by a
    unique ``id``, with each experiment's runs attached. Load it back with
    :func:`load_records`.

    ``expand=True`` is where the cost is. The saved studies are *summaries*:
    they carry study-level fields only, so without expanding, each yields a
    single stub record and the sample/experiment/run fields stay empty. With it,
    every study is rebuilt from SRA, which is one request per 300 records.
    ``max_records`` skips studies too large to be worth that (see
    :data:`MAX_RECORDS`); they fall back to their summary rather than dropping
    out. Pass ``expand=False`` for a free, network-less pass over the file.

    Progress is checkpointed to ``<out_path>.partial`` every
    ``checkpoint_every`` studies and removed on success; ``resume=True`` (the
    default) continues from it, ignoring one written under different arguments.

    ``harmonize`` / ``from_text`` / ``from_paper`` enable the reconstruction
    layers (:func:`reconstruct.reconstruct`). All default to **off**, matching
    the cascade's own defaults: the direct layer always runs and costs nothing,
    while the model layers spend real money per sample, so turning one on is a
    decision the caller makes rather than a default it inherits. With all of
    them off this writes exactly what SRA states outright, and the coverage line
    is the size of the gap the model layers would have to fill.

    A layer that fails on one study — a refusal, an exhausted retry budget —
    costs that study its inference, not the run: the study falls back to its
    direct-only records and the failure is counted in the summary.

    **``max_spend`` is checked before any paid work.** The estimate is printed
    every run and, if it exceeds the limit, :class:`SpendLimitExceeded` is
    raised having spent nothing. Pass a higher number to authorise a larger run,
    or ``None`` to disable the check. It is a hard stop rather than a prompt so
    that an unattended run fails instead of hanging.
    """
    studies = load_studies(in_path)
    layers = {"harmonize": bool(harmonize), "from_text": bool(from_text),
              "from_paper": bool(from_paper)}
    params = {
        "in_path": str(in_path),
        "expand": bool(expand),
        "max_records": max_records,
        "study_count": len(studies),
        # a checkpoint from a direct-only run must not resume into a run with
        # model layers on, or the output splices two different reconstructions
        **layers,
    }
    estimate, report = estimate_reconstruction_cost(
        studies, harmonize=harmonize, from_text=from_text,
        from_paper=from_paper, max_records=max_records,
    )
    print(report, flush=True)
    if max_spend is not None and estimate > max_spend:
        raise SpendLimitExceeded(
            f"estimated ${estimate:.2f} exceeds max_spend=${max_spend:.2f} — "
            f"nothing has been spent. Re-run with max_spend={estimate:.2f} to "
            f"authorise it, max_spend=None to disable the check, or "
            f"from_text=False / from_paper=False to cut the cost."
        )

    ckpt = _checkpoint_path(out_path)
    state = _load_checkpoint(ckpt, params) if resume else None

    if state is not None:
        done = state["done"]
        print(f"Resuming {ckpt}: {len(done)} of {len(studies)} studies already reconstructed")
    else:
        done = {}
        print("Studies loaded:", len(studies))

    fallbacks = 0
    filled = collections.Counter()
    for study in studies:
        if study.accession in done:
            continue
        source, note = _expand(study, max_records) if expand else (study, "")
        if note:
            fallbacks += 1
        records, report, layer_note = _reconstruct(source, layers)
        filled.update(report)
        done[study.accession] = [r.to_dict() for r in records]
        print(f"{study.accession}: {len(records)} record(s){note}{layer_note}", flush=True)
        if len(done) % checkpoint_every == 0:
            _write_json(ckpt, {"version": _CHECKPOINT_VERSION, "params": params,
                               "done": done})
            print(f"  reconstructed {len(done)}/{len(studies)} studies", flush=True)

    rows = [row for records in done.values() for row in records]
    print(f"Records: {len(rows)} from {len(done)} studies", end="")
    print(f" ({fallbacks} not expanded)" if fallbacks else "")
    if filled:
        print("Fields filled per layer: "
              + ", ".join(f"{cls} {n:,}" for cls, n in sorted(filled.items())))
    if any(layers.values()):
        print("Claude usage:", claude.usage_report())
    if rows:
        n_fields = len(TargetSchema.field_names())
        # to_dict omits empty fields, so its length is the per-record yield —
        # but only after dropping all three sidecars. Counting `confidence` and
        # `runs` as schema fields inflated this by exactly 2 per record.
        sidecars = {"provenance", "confidence", "runs"}
        mean = sum(len(set(row) - sidecars) for row in rows) / len(rows)
        tail = ("" if any(layers.values())
                else " — the remainder is what the model layers would fill")
        print(f"Mean coverage: {mean:.1f}/{n_fields} fields ({mean / n_fields:.0%}){tail}")

    text = json.dumps(rows, indent=2, ensure_ascii=False)
    with open(out_path, "w", encoding="utf-8") as file:
        file.write(text)
    # Only once the real output is safely on disk.
    try:
        os.remove(ckpt)
    except OSError:
        pass


def load_records(source) -> list[TargetSchema]:
    """Load :func:`save_reconstructed_records` output back into TargetSchema objects.

    ``source`` is a file path, a JSON string, or an already-parsed list — the
    same shapes :func:`project.load_studies` accepts. No network calls.
    """
    data = _load_json(source)
    if isinstance(data, dict):
        data = [data]
    return [TargetSchema.from_dict(d) for d in data]