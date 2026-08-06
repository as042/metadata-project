"""Build the study shortlist that feeds LLM metadata reconstruction.

Two stages, each writing a JSON array of ``Project`` dicts:

1. :func:`save_recent_studies` — sample studies from SRA and save their summaries.
2. :func:`filter_oa_studies`   — keep only the ones with an open-access paper,
   caching the accessibility classes it computes into the saved JSON.

Stage 1 checkpoints as it goes, so a harvest of thousands of studies survives a
crash or a Ctrl-C: rerun the same call and it picks up where it stopped.
"""

from datetime import date
import json
import os
import time

from project import classify_publication
from project import search_studies
from project import scan_iter
from project import load_studies


# Studies bigger than this are continuously-appended surveillance umbrellas
# (PulseNet, GenomeTrakr, growing cohorts). Their samples aren't described by the
# linked papers, so they're poor LLM context however accessible the paper is.
MAX_RECORDS = 5000

# Studies scanned between checkpoint writes. Small enough that a crash costs
# seconds of work, large enough that the rewrite cost stays negligible.
CHECKPOINT_EVERY = 25

# esummary pages between search-progress lines. search_studies reports every page,
# which is ~2,700 of them on a 5,000-study harvest — enough to bury everything else.
SEARCH_PROGRESS_EVERY = 20

# Pacing between classify_publication calls. Deliberately not the NCBI rate: most
# of that work hits Europe PMC and Unpaywall, who are third parties that never
# granted us a raised limit, so an NCBI api_key must not speed this up.
CLASSIFY_SLEEP = 0.34

_CHECKPOINT_VERSION = 1


def _checkpoint_path(path):
    return f"{path}.partial"


def _write_json(path, payload):
    """Write JSON atomically — a crash mid-write must not destroy a checkpoint."""
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
