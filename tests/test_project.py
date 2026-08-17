"""Integration tests for project.Project — these hit NCBI E-utilities live.

Run either way:

    python tests/test_project.py      # standalone runner (no pytest needed)
    pytest tests/test_project.py      # if pytest is installed

Fixtures were verified 2026-07. record_count values are stable for these
published studies, but if NCBI revises a study the pinned counts may need a bump.
"""

from __future__ import annotations

import json
import os
import sys
from functools import lru_cache

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project import (  # noqa: E402
    EUTILS,
    Project,
    Publication,
    Run,
    Sample,
    _common,
    _record_count,
    _request_with_retry,
    _to_entrez_date,
    classify_publication,
    filter_by_publication,
    filter_projects_by_publication,
    load_studies,
    scan,
    search_studies,
    set_entrez_credentials,
)


# --------------------------------------------------------------------------- #
# cached builds so the whole suite stays to a handful of network round-trips
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=None)
def summary(acc, pubs=False):
    return Project.summary(acc, include_publications=pubs)


@lru_cache(maxsize=None)
def full(acc):
    return Project(acc)


# --------------------------------------------------------------------------- #
# summary (lightweight) build
# --------------------------------------------------------------------------- #
def test_summary_small_study():
    p = summary("SRP098789")
    assert p.record_count == 26
    assert p.title.startswith("Selective stalling")
    assert p.bioproject == "PRJNA369742"
    assert p.study_type is not None
    assert p.external_ids.get("GEO") == "GSE94454"
    assert p.published.startswith("2017-06-07")  # release date, even in summary mode
    # summary is study-only: no heavy collections
    assert len(p.samples) == 0
    assert len(p.experiments) == 0


def test_search_within_years_narrows_the_result_set_server_side():
    """``within_years`` reaches Entrez and genuinely restricts the pool.

    An earlier version of this test range-checked ``Project.published`` — the
    earliest RUN\u0040published — against the window. That was wrong rather than
    merely flaky: the filter is ``datetype=pdat``, the Entrez *record* date,
    which is unrelated to when the data was released and which NCBI bumps on
    re-index. ``SRP223534`` has RUN\u0040published ``2019-09-28`` and an Entrez
    ``createdate`` of today, so it legitimately matches a five-year window while
    reporting a 2019 release. The old assertion held only while the cutoff year
    sat below 2019.

    Checking ``createdate`` instead does not work either: NCBI has bumped it to
    the current month for effectively every record, so it is recent whatever the
    filter does. Nor can this go through ``search_studies``, whose ``max_scanned``
    bound returns the same accessions with and without a date window — verified
    by deleting the date params and watching the assertions still pass.

    What is left, and what actually holds, is the count: the same query with a
    narrower window must match strictly fewer records.
    """
    def count(**extra):
        res = _request_with_retry(
            f"{EUTILS}/esearch.fcgi", db="sra", term="Homo sapiens[Organism]",
            retmode="json", retmax=0, **extra,
        )
        return int(json.loads(res.text)["esearchresult"]["count"])

    unfiltered = count()
    five_years = count(datetype="pdat", reldate=int(round(5 * 365.25)))
    one_year = count(datetype="pdat", reldate=365)

    assert unfiltered > 0
    # Strict: a five-year window must exclude something, and one year must
    # exclude more still. Measured 2026-08: 7.15M / 4.46M / 0.90M.
    assert five_years < unfiltered, f"5-year window matched everything ({five_years})"
    assert one_year < five_years, f"1-year {one_year} not below 5-year {five_years}"


def test_search_within_years_returns_usable_accessions():
    # The end-to-end smoke test the one above deliberately does not attempt:
    # that the filter does not break the search, only that it runs and yields
    # real study accessions. Their dates are not asserted, for the reasons in
    # the docstring above.
    accessions = search_studies(organism="Homo sapiens", within_years=5, max_studies=4)
    assert accessions
    assert all(a[:3] in ("SRP", "ERP", "DRP") for a in accessions), accessions
    assert len(set(accessions)) == len(accessions), "duplicate accessions returned"


def test_summary_is_o1_on_large_study():
    # 727 runs, but summary still fetches one package and stays study-only
    p = summary("SRP157974")
    assert p.record_count == 727
    assert p.title
    assert len(p.samples) == 0
    assert len(p.experiments) == 0


def test_summary_published_is_earliest_not_newest():
    # Regression: summary mode fetched only the first uid, and the esearch result
    # set is newest-first, so `published` was the study's NEWEST release date.
    # SRP049009 is appended to continuously — its newest record is from 2026 while
    # the study opened in 2014 — which put studies a decade outside a requested
    # date window. Small studies like SRP098789 can't catch this: all 26 of their
    # runs share one timestamp.
    p = summary("SRP049009")
    assert p.record_count > 2000
    assert p.published.startswith("2014-10-17"), p.published


def test_summary_published_matches_full_build():
    # the cheap summary date must agree with the date a full build computes
    acc = "SRP098789"
    assert summary(acc).published == full(acc).published


def test_summary_publications_present():
    pubs = summary("SRP098789", pubs=True).publications
    assert [(x.id, x.type) for x in pubs] == [("28323820", "ePubmed")]


def test_summary_publications_absent():
    assert summary("SRP157974", pubs=True).publications == []


# --------------------------------------------------------------------------- #
# full build
# --------------------------------------------------------------------------- #
def test_full_build_counts():
    p = full("SRP098789")
    assert p.record_count == 26
    assert len(p.samples) == 26
    assert len(p.experiments) == 26
    assert sum(len(e.runs) for e in p.experiments) == 26


def test_full_build_sample_attributes():
    s = full("SRP098789").samples["SRS1956378"]
    assert s.biosample == "SAMN06293494"
    assert s.scientific_name == "Homo sapiens"
    assert s.attributes.get("cell line") == "Huh7"


def test_experiment_shape():
    p = full("SRP098789")
    for e in p.experiments:
        assert e.accession.startswith("SRX")
        assert isinstance(e.sample_ids, list)  # pool-safe list, never a bare str
        assert e.sample_ids and e.sample_ids[0].startswith("SRS")
        assert e.library_layout == "SINGLE"  # this study is single-end
        assert e.instrument_model.startswith("Illumina")
        assert e.runs and e.runs[0].accession.startswith("SRR")
        assert e.runs[0].total_bases and e.runs[0].total_bases > 0


def test_resolver_helpers():
    p = full("SRP098789")
    e = p.experiments[0]
    resolved = p.samples_of(e)
    assert resolved and resolved[0].accession == e.sample_ids[0]
    assert len(p.runs_of_sample(e.sample_ids[0])) >= 1


# --------------------------------------------------------------------------- #
# serialization / flattening
# --------------------------------------------------------------------------- #
def test_to_dataframe():
    df = full("SRP098789").to_dataframe()
    assert len(df) == 26  # one row per run
    assert {"study_accession", "run_accession", "sample_accession", "bioproject"} <= set(
        df.columns
    )
    assert df["study_accession"].nunique() == 1
    assert df["run_accession"].nunique() == 26


def test_to_json_preserves_shape():
    data = json.loads(full("SRP098789").to_json())
    assert {
        "accession",
        "bioproject",
        "title",
        "samples",
        "experiments",
        "publications",
    } <= set(data.keys())
    assert len(data["samples"]) == 26
    assert len(data["experiments"]) == 26
    assert isinstance(data["experiments"][0]["runs"], list)  # nesting intact


def test_from_dict_roundtrip():
    original = full("SRP098789")
    clone = Project.from_dict(original.to_dict())
    assert clone.accession == original.accession
    assert clone.bioproject == original.bioproject
    assert clone.published == original.published
    assert len(clone.samples) == 26
    assert len(clone.experiments) == 26
    # nested dataclasses are rebuilt, not left as dicts
    assert isinstance(next(iter(clone.samples.values())), Sample)
    assert isinstance(clone.experiments[0].runs[0], Run)
    # exact round-trip
    assert clone.to_dict() == original.to_dict()


def test_load_studies_from_string_and_file():
    import os
    import tempfile

    payload = [full("SRP098789").to_dict(), full("SRP098789").to_dict()]

    studies = load_studies(json.dumps(payload))  # from a JSON string
    assert len(studies) == 2
    assert all(isinstance(p, Project) for p in studies)
    assert studies[0].accession == "SRP098789"

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        fh.write(json.dumps(payload))
        path = fh.name
    try:
        from_file = load_studies(path)  # from a file path
        assert len(from_file) == 2
        assert from_file[0].to_dict() == full("SRP098789").to_dict()
    finally:
        os.unlink(path)


# --------------------------------------------------------------------------- #
# include_* flags
# --------------------------------------------------------------------------- #
def test_include_runs_false():
    p = Project("SRP098789", include_runs=False)
    assert len(p.experiments) == 26
    assert all(e.runs == [] for e in p.experiments)


# --------------------------------------------------------------------------- #
# scan()
# --------------------------------------------------------------------------- #
def test_scan_collects_results_and_failures():
    projects, errors = scan(["SRP098789", "NOT_A_REAL_ACCESSION_ZZZ"])
    assert "SRP098789" in projects
    assert projects["SRP098789"].record_count == 26
    assert "NOT_A_REAL_ACCESSION_ZZZ" in errors
    assert isinstance(errors["NOT_A_REAL_ACCESSION_ZZZ"], Exception)


def test_record_count_helper():
    assert _record_count("SRP098789") == 26
    big = _record_count("SRP040281")  # PulseNet Salmonella surveillance umbrella
    assert big is not None and big > 1000


def test_scan_max_records_guard():
    # SRP098789 (26 records) kept; SRP040281 (hundreds of thousands) skipped
    projects, errors = scan(["SRP098789", "SRP040281"], max_records=1000)
    assert "SRP098789" in projects
    assert "SRP040281" not in projects
    assert "SRP040281" in errors
    assert "record_count" in str(errors["SRP040281"])


# --------------------------------------------------------------------------- #
# search_studies()
# --------------------------------------------------------------------------- #
def test_search_studies_enumerates_distinct_in_window():
    accs = search_studies(
        organism="Homo sapiens",
        strategy="RNA-Seq",
        within_years=10,
        max_studies=3,
    )
    assert 1 <= len(accs) <= 3
    assert len(set(accs)) == len(accs)  # distinct studies, not per-experiment dups
    assert all(a[:3] in ("SRP", "ERP", "DRP") and a[3:].isdigit() for a in accs)


def test_entrez_credentials_fallback_and_override():
    try:
        set_entrez_credentials(email="me@example.org", api_key="KEY123")
        c = _common()
        assert c["tool"] == "metadata-project"
        assert c["email"] == "me@example.org"
        assert c["api_key"] == "KEY123"
        # an explicit per-call value wins over the process default
        over = _common(email="other@example.org")
        assert over["email"] == "other@example.org"
        assert over["api_key"] == "KEY123"  # unspecified -> still the default
    finally:
        set_entrez_credentials()  # reset so later live tests aren't sent a fake key
    assert "email" not in _common() and "api_key" not in _common()


def test_to_entrez_date():
    from datetime import date

    assert _to_entrez_date(date(2022, 3, 4)) == "2022/03/04"
    assert _to_entrez_date("2022-03-04") == "2022/03/04"  # ISO string ok
    assert _to_entrez_date(None) is None
    try:
        _to_entrez_date("03/04/2022")  # ambiguous locale-ordered string -> rejected
    except ValueError:
        return
    raise AssertionError("ambiguous date string should be rejected")


def test_search_date_range_with_date_objects():
    from datetime import date

    accs = search_studies(
        organism="Homo sapiens",
        strategy="RNA-Seq",
        after_date=date(2015, 1, 1),
        before_date=date(2015, 12, 31),
        max_studies=3,
    )
    assert 1 <= len(accs) <= 3
    assert all(a[:3] in ("SRP", "ERP", "DRP") and a[3:].isdigit() for a in accs)


def test_search_sort_modes():
    kw = dict(organism="Homo sapiens", strategy="RNA-Seq", max_studies=5)
    recent = search_studies(sort="recent", **kw)
    oldest = search_studies(sort="oldest", **kw)
    rand = search_studies(sort="random", **kw)
    assert len(recent) == 5 and len(oldest) == 5 and len(rand) == 5
    # recent and oldest sample opposite ends of a huge result set -> disjoint
    assert set(recent).isdisjoint(oldest)


def test_search_sort_invalid():
    try:
        search_studies(sort="banana", max_studies=1)
    except ValueError:
        return
    raise AssertionError("expected ValueError for invalid sort")


def test_search_then_scan_pipeline():
    accs = search_studies(organism="Homo sapiens", within_years=10, max_studies=2)
    projects, errors = scan(accs)
    assert not errors
    assert set(projects) == set(accs)
    assert all(p.record_count and p.record_count > 0 for p in projects.values())


# --------------------------------------------------------------------------- #
# classify_publication()
# --------------------------------------------------------------------------- #
def test_classify_publication_oa():
    assert classify_publication("28323820") == "oa"  # PLoS Biol (SRP098789)


def test_classify_publication_partial():
    # PNAS (SRP066834): full text in PMC but not in the OA subset
    assert classify_publication("26644564") == "partial"


def test_classify_publication_paywall():
    # J Pathology (SRP074349): indexed, no PMCID -> abstract only
    assert classify_publication("29282718") == "paywall"


def test_classify_publication_unknown():
    assert classify_publication("99999999999") == "unknown"


def test_classify_publication_by_doi():
    assert classify_publication("10.1371/journal.pbio.2001882") == "oa"


def test_filter_by_publication():
    # SRP098789 -> OA paper; SRP066834 -> partial; SRP157974 -> no paper
    accs = ["SRP098789", "SRP066834", "SRP157974"]
    matched, classes = filter_by_publication(accs, "oa")
    assert matched == ["SRP098789"]
    assert "oa" in classes["SRP098789"]
    assert classes["SRP066834"] == ["partial"]
    assert classes["SRP157974"] == []  # no linked publication
    # every input accession is represented in the returned classification map
    assert set(classes) == set(accs)


def test_publication_classes_method():
    p = Project.from_dict(
        {"accession": "X", "publications": [{"id": "28323820", "type": "ePubmed"}]}
    )
    assert p.publication_classes() == ["oa"]


def test_publication_accessibility_type_default():
    assert Publication(id="1").accessibility_type is None


def test_publication_classes_uses_cached_type():
    # accessibility_type pre-set with a fake id: a real lookup would give "unknown",
    # so returning "oa" proves the cached value is used with no network call
    p = Project.from_dict(
        {
            "accession": "X",
            "publications": [
                {"id": "0000000", "type": "ePubmed", "accessibility_type": "oa"}
            ],
        }
    )
    assert p.publication_classes() == ["oa"]
    assert p.publication_classes(refresh=True) == ["unknown"]  # forces real lookup


def test_publication_classes_caches_and_roundtrips():
    p = Project.from_dict(
        {"accession": "X", "publications": [{"id": "28323820", "type": "ePubmed"}]}
    )
    assert p.publications[0].accessibility_type is None
    assert p.publication_classes() == ["oa"]
    assert p.publications[0].accessibility_type == "oa"  # cached on the object
    d = p.to_dict()
    assert d["publications"][0]["accessibility_type"] == "oa"  # persisted
    reloaded = Project.from_dict(d)
    assert reloaded.publications[0].accessibility_type == "oa"  # restored
    assert reloaded.publication_classes() == ["oa"]  # served from cache


def test_filter_projects_by_publication():
    # build Projects offline with known publications; only classify hits network
    oa = Project.from_dict(
        {"accession": "STUDY_OA", "publications": [{"id": "28323820", "type": "ePubmed"}]}
    )
    paywall = Project.from_dict(
        {"accession": "STUDY_PW", "publications": [{"id": "29282718", "type": "ePubmed"}]}
    )
    nopaper = Project.from_dict({"accession": "STUDY_NONE", "publications": []})

    matched, classes = filter_projects_by_publication([oa, paywall, nopaper], "oa")
    assert [p.accession for p in matched] == ["STUDY_OA"]
    assert classes == {
        "STUDY_OA": ["oa"],
        "STUDY_PW": ["paywall"],
        "STUDY_NONE": [],
    }


def test_search_studies_publication_filter():
    # Every returned study must have at least one OA-classified publication.
    accs = search_studies(
        organism="Homo sapiens",
        within_years=15,
        max_studies=1,
        publication="OA",
        max_scanned=120,
    )
    assert len(accs) <= 1
    for a in accs:
        p = Project.summary(a, include_publications=True)
        assert any(classify_publication(pub.id) == "oa" for pub in p.publications)


# --------------------------------------------------------------------------- #
# standalone runner (used when pytest is not installed)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import traceback

    tests = [
        (name, fn)
        for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
            passed += 1
        except Exception:
            print(f"FAIL  {name}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    raise SystemExit(1 if failed else 0)