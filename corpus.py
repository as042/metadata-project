"""Build a fully-expanded, self-contained corpus file from a study shortlist.

The harvest writes *summaries*: accession, title, abstract, record count, and
the classified publications. Everything a model layer actually reads — the
per-sample attribute bags, the experiment and run rows, the paper text — is
re-fetched from SRA and Europe PMC at reconstruction time, on every run.

That is fine for one run and wrong for everything else:

* **Reproducibility.** Two runs a week apart are not the same experiment. A
  model comparison across them mixes the model's effect with whatever the
  archives changed in between.
* **A language boundary.** Anything that consumes reconstruction input has to
  speak SRA, Europe PMC and Unpaywall. With a corpus file it has to speak JSON.
* **Auditability.** :mod:`audit` cannot check layer 4 at all today, for the
  single reason that paper text is never persisted. Persisting it — measured at
  30 KB per paper, 9 MB for the whole reference corpus — closes that hole.

    import corpus
    corpus.build_full_corpus("datasets/oa_corpus.json",
                             "datasets/oa_corpus_full.json")

Costs no model tokens. It is NCBI, Europe PMC and Unpaywall only, all free, and
on the 346-study reference corpus it runs for roughly two and a half hours —
almost all of it waiting on SRA to return 102,227 records. Checkpointed every
``checkpoint_every`` studies so an interrupted build resumes instead of
restarting.

FILE SHAPE
``papers`` is keyed by publication id; each publication's ``paper_ids`` indexes
into it. Shared rather than inlined because 17 papers serve more than one study,
and a paper can then be re-fetched without rebuilding a study blob. A list per
publication, so a study with several open-access papers keeps all of them and a
second copy of one article can be added later without a format change.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os

from project import (Project, classify_publication, fetch_biosample_records,
                     fetch_bioproject_record, fetch_open_access_text,
                     publication_oa_status, load_studies)
import dataset
import reconstruct

# Bumped when the file shape changes in a way a reader must notice. Readers
# should refuse a version they do not know rather than guess — a corpus built
# a year from now will not be this one.
FORMAT_VERSION = 2

# Studies between checkpoint writes. Each study is minutes of fetching, so a
# small number costs little and saves a lot when a build is interrupted.
CHECKPOINT_EVERY = 10


def _checkpoint_path(path):
    return f"{path}.partial"


def _classify_all(publications, sleep=None):
    """Classify every publication, not just up to the first open-access one.

    :func:`dataset.filter_oa_studies` deliberately stops at the first ``oa``
    hit — for deciding whether a study qualifies, the rest cannot change the
    answer. The corpus is not making that decision, and inheriting the
    short-circuit left **80 of 432 publications unclassified** in the previous
    build: 50 studies have more than one paper and nobody had checked whether
    the extras were retrievable.

    Returns the number newly classified. ``accessibility_type`` stays None only
    if the lookup itself fails, which keeps "never checked" distinguishable from
    "checked, not open".
    """
    done = 0
    for pub in publications:
        if pub.accessibility_type is None:
            try:
                pub.accessibility_type = classify_publication(pub.id, sleep=sleep)
                done += 1
            except Exception:      # noqa: BLE001 - one bad id must not stop a build
                pass
    return done


def _oa_publications(study):
    """Every publication classified ``oa``, in order.

    All of them, not just the first: layer 4 may choose to read only one, but
    that is a *reconstruction* policy and baking it into the corpus would mean a
    rebuild to revisit it.
    """
    return [p for p in study.publications if p.accessibility_type == "oa"]


def build_full_corpus(
    in_path="datasets/oa_corpus.json",
    out_path="datasets/oa_corpus_full.json",
    max_records=dataset.MAX_RECORDS,
    biosample=True,
    bioproject=True,
    papers=True,
    resume=True,
    checkpoint_every=CHECKPOINT_EVERY,
    limit=None,
):
    """Expand every study in ``in_path`` and write a self-contained corpus.

    ``biosample`` additionally fetches each sample's BioSample record and keeps
    it in :attr:`project.Sample.biosample_attributes`, *beside* the SRA bag
    rather than merged into it — the two disagree often enough to be worth
    telling apart, and merging would change what layer 3 gets credited with
    inferring. Batched, so it adds minutes rather than hours.

    ``papers`` fetches the full text of each study's first open-access
    publication into the shared ``papers`` map. Text is capped by
    :func:`project.fetch_open_access_text` at
    :data:`reconstruct.PAPER_MAX_CHARS`, which is what layer 4 would have seen.

    ``limit`` builds only the first N studies — for trying the shape out before
    committing to the full run.

    A study that fails to expand is kept in summary form with a ``build_note``
    explaining why, exactly as :func:`dataset._expand` does mid-pipeline: one
    unreachable study must not cost the other 345.
    """
    studies = load_studies(in_path)
    if limit is not None:
        studies = studies[:limit]

    ckpt = _checkpoint_path(out_path)
    params = {"source": str(in_path), "max_records": max_records,
              "biosample": bool(biosample), "bioproject": bool(bioproject),
              "papers": bool(papers),
              "study_count": len(studies), "format_version": FORMAT_VERSION}
    state = dataset._load_checkpoint(ckpt, params) if resume else None
    done = state["done"] if state else {}
    paper_texts = state.get("papers", {}) if state else {}
    if state:
        print(f"Resuming {ckpt}: {len(done)} of {len(studies)} studies already built")
    else:
        print(f"Building {len(studies)} studies -> {out_path}")

    notes = 0
    for n, study in enumerate(studies, 1):
        if study.accession in done:
            continue
        full, note = dataset._expand(study, max_records)
        if note:
            notes += 1
        # _expand deliberately drops publications (it skips the BioProject
        # fetch); they carry the `oa` classes stage 2 paid for, so restore them.
        full.publications = study.publications

        n_bio = 0
        if biosample and not note:
            ids = [s.biosample for s in full.samples.values() if s.biosample]
            if ids:
                try:
                    got = fetch_biosample_records(ids)
                    for sample in full.samples.values():
                        entry = got.get(sample.biosample or "")
                        if entry:
                            sample.biosample_attributes, sample.biosample_record = entry
                            n_bio += 1
                except Exception as exc:      # noqa: BLE001 - never fail the build
                    note += f" [biosample: {type(exc).__name__}: {exc}]"

        # The BioProject record proper. `_expand` skips it to save a fetch per
        # study, so the publications restored above carry only what stage 2
        # classified — this adds the project's own description, target and dates.
        if bioproject and not note and full.bioproject:
            try:
                got = fetch_bioproject_record(full.bioproject)
                if got:
                    full.bioproject_record, bp_pubs = got
                    # Merge the BioProject's own publication metadata (date,
                    # status, reference) into the classified list, matched by id.
                    by_id = {p.id: p for p in bp_pubs}
                    for pub in full.publications:
                        extra = by_id.get(pub.id)
                        if extra:
                            pub.date = pub.date or extra.date
                            pub.status = pub.status or extra.status
                            pub.reference = pub.reference or extra.reference
            except Exception as exc:          # noqa: BLE001
                note += f" [bioproject: {type(exc).__name__}: {exc}]"

        # Classify the whole bibliography before choosing what to fetch, or the
        # short-circuit in stage 2 silently caps this study at one paper.
        n_new = _classify_all(full.publications) if papers else 0

        for pub in (_oa_publications(full) if papers else []):
            if pub.id not in paper_texts:
                try:
                    text = fetch_open_access_text(pub.id)
                except Exception as exc:      # noqa: BLE001
                    text = None
                    note += f" [paper {pub.id}: {type(exc).__name__}: {exc}]"
                # An `oa` paper whose text will not come back is normal, not an
                # error: bronze and green are genuinely open access and still
                # deposit nothing to Europe PMC, the only source the fetch reads.
                try:
                    oa_status = publication_oa_status(pub.id)
                except Exception:             # noqa: BLE001 - never fail the build
                    oa_status = None
                paper_texts[pub.id] = {
                    "id": pub.id, "type": pub.type,
                    "chars": len(text) if text else 0,
                    "text": text or None,
                    "oa_status": oa_status,
                    # 91% of stored papers hit the cap; flag it so nothing
                    # downstream treats a fragment as a whole document.
                    "truncated": bool(text) and
                                 len(text) >= reconstruct.PAPER_MAX_CHARS,
                }
            # Point the publication at its text. Keyed by publication id today;
            # a list so a second copy of the same article can be added later
            # without changing the format.
            if paper_texts.get(pub.id, {}).get("text"):
                pub.paper_ids = [pub.id]

        blob = full.to_dict()
        blob["build_note"] = note.strip() or None
        done[study.accession] = blob

        print(f"  [{n}/{len(studies)}] {study.accession}: "
              f"{len(full.samples):,} samples, {len(full.experiments):,} experiments"
              f"{f', {n_bio} biosample' if n_bio else ''}"
              f"{f', +{n_new} classified' if n_new else ''}"
              f"{'  ' + note.strip() if note else ''}", flush=True)

        if len(done) % checkpoint_every == 0:
            dataset._write_json(ckpt, {"version": dataset._CHECKPOINT_VERSION,
                                       "params": params, "done": done,
                                       "papers": paper_texts})

    payload = {
        "format_version": FORMAT_VERSION,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "params": params,
        "counts": {
            "studies": len(done),
            "records": sum(len(b.get("experiments") or []) for b in done.values()),
            "samples": sum(len(b.get("samples") or {}) for b in done.values()),
            "papers": sum(1 for p in paper_texts.values() if p["text"]),
            "papers_empty": sum(1 for p in paper_texts.values() if not p["text"]),
            "summary_only": notes,
        },
        "papers": paper_texts,
        "studies": [done[s.accession] for s in studies if s.accession in done],
    }
    dataset._write_json(out_path, payload)
    if os.path.exists(ckpt):
        os.remove(ckpt)

    c = payload["counts"]
    print(f"\nWrote {out_path}  ({os.path.getsize(out_path)/1e6:.1f} MB)")
    print(f"  studies {c['studies']:,} | samples {c['samples']:,} | "
          f"experiments {c['records']:,}")
    print(f"  papers {c['papers']:,} with text, {c['papers_empty']:,} classified oa "
          f"but empty | {c['summary_only']:,} studies summary-only")
    return payload


def load_full_corpus(path):
    """Read a corpus file back as ``(projects, papers, meta)`` — no network.

    Refuses a ``format_version`` it does not know rather than guessing at the
    shape: a reader that silently mis-parses a future corpus is worse than one
    that stops.

    ``build_note`` is re-attached to each project;
    :meth:`project.Project.from_dict` only reads the keys it declares.

    The link from a study to its text lives on each publication's
    ``paper_ids``, which index into the returned ``papers`` map. A study can
    therefore carry several papers, or none, without a format change.
    """
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    version = payload.get("format_version")
    if version != FORMAT_VERSION:
        raise ValueError(
            f"{path} is format_version {version!r}; this build understands "
            f"{FORMAT_VERSION}. Rebuild it, or read it with the matching version."
        )
    projects = []
    for blob in payload["studies"]:
        p = Project.from_dict(blob)
        p.build_note = blob.get("build_note")
        projects.append(p)
    return projects, payload["papers"], {k: v for k, v in payload.items()
                                         if k not in ("studies", "papers")}


def paper_texts(project, papers) -> list[str]:
    """Every retrieved text for this study, in publication order. Often empty.

    Empty covers two cases a caller usually treats alike: no open-access
    publication at all, and one that is genuinely open but whose text could not
    be retrieved. See :func:`split_by_paper` for why the second happens.
    """
    out = []
    for pub in project.publications:
        for pid in getattr(pub, "paper_ids", None) or []:
            text = (papers.get(pid) or {}).get("text")
            if text:
                out.append(text)
    return out


def paper_text(project, papers) -> str | None:
    """The first retrieved text, or None — what layer 4 actually reads today."""
    texts = paper_texts(project, papers)
    return texts[0] if texts else None


def split_by_paper(projects, papers):
    """``(with_text, without_text)`` — the split layer 4 actually cares about.

    **A study without usable paper text does not look "bad" in the data.** Its
    ``publications`` list is populated, the publication's
    ``accessibility_type`` is ``"oa"``, and ``build_note`` is None — because
    none of that is wrong. The paper genuinely is open
    access; it is open by a route (*bronze* at the publisher, or *green* in a
    repository) that deposits nothing to Europe PMC, which is the only source
    :func:`project.fetch_open_access_text` reads.

    So filtering on publications or on ``accessibility_type`` will not find
    them. The only signal is an empty ``paper_ids``, or an empty ``text`` behind
    one, which is what this checks.
    """
    with_text, without = [], []
    for p in projects:
        (with_text if paper_text(p, papers) else without).append(p)
    return with_text, without


if __name__ == "__main__":
    import sys
    from project import set_entrez_credentials

    def _cred(path):
        with open(path, encoding="utf-8") as fh:
            return fh.read().strip()

    set_entrez_credentials(api_key=_cred("api_key.txt"), email=_cred("email.txt"))
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    build_full_corpus(limit=limit)
