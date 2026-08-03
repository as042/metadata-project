"""Build the study shortlist that feeds LLM metadata reconstruction.

Two stages, each writing a JSON array of ``Project`` dicts:

1. :func:`save_recent_studies` — sample studies from SRA and save their summaries.
2. :func:`filter_oa_studies`   — keep only the ones with an open-access paper,
   caching the accessibility classes it computes into the saved JSON.
"""

from datetime import date
import json

from project import classify_publication
from project import search_studies
from project import scan
from project import load_studies


# Studies bigger than this are continuously-appended surveillance umbrellas
# (PulseNet, GenomeTrakr, growing cohorts). Their samples aren't described by the
# linked papers, so they're poor LLM context however accessible the paper is.
MAX_RECORDS = 5000


def save_recent_studies(
    path="./recent_studies.json",
    max_studies=50,
    after_date=date(2020, 1, 1),
    before_date=date(2024, 1, 1),
    sort="random",
    max_records=MAX_RECORDS,
):
    """Sample studies from SRA and save their summaries as a JSON array.

    ``sort="random"`` spreads the sample across the whole date window, which is
    what turns up studies old enough to have a linked paper. ``max_records`` drops
    oversized umbrella studies (see :data:`MAX_RECORDS`); pass None to keep them.
    """
    studies = search_studies(
        after_date=after_date,
        before_date=before_date,
        max_studies=max_studies,
        sort=sort,
    )
    projects, errors = scan(
        studies, include_publications=True, max_records=max_records
    )
    print("Number of studies searched:", len(studies))
    print("Number of projects scanned and errors:", len(projects), len(errors))

    summaries = list(projects.values())
    for p in summaries:
        print(f"{p.accession}: published {p.published}, {p.record_count} records")

    text = json.dumps(
        [p.to_dict() for p in summaries], indent=2, ensure_ascii=False
    )
    with open(path, "w", encoding="utf-8") as file:
        file.write(text)


def filter_oa_studies(in_path="./recent_studies.json", out_path="./oa_studies.json"):
    """Keep the studies with at least one open-access publication.

    Every class that gets computed is cached onto the publication and written to
    ``out_path``, so re-running over the output costs no network calls. Papers
    after the first open-access one in a study are left unclassified (``None``) —
    they can't change the outcome.
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