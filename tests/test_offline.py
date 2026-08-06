"""Unit tests for project.py that never touch the network.

The live suite in test_project.py is the real end-to-end check, but it can't run
in CI or offline and a flaky NCBI response fails it. Everything here is pure logic
or a stubbed request, so it runs anywhere in well under a second:

    python tests/test_offline.py
    pytest tests/test_offline.py
"""

from __future__ import annotations

import os
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import project as P  # noqa: E402
from project import (  # noqa: E402
    Project,
    Sample,
    _attrs_bag,
    _common,
    _int,
    _load_json,
    _page_starts,
    _sleep_for,
    _to_entrez_date,
    set_entrez_credentials,
)


# --------------------------------------------------------------------------- #
# pure helpers
# --------------------------------------------------------------------------- #
def test_page_starts_recent_and_oldest():
    assert _page_starts("recent", 1000, 500) == [0, 500]
    assert _page_starts("oldest", 1000, 500) == [500, 0]
    # a partial last page still covers record 0, which needs an overlapping page
    assert _page_starts("oldest", 1001, 500) == [501, 1, 0]
    assert _page_starts("recent", 0, 500) == [0]


def test_page_starts_random_is_a_permutation_of_pages():
    starts = _page_starts("random", 1000, 100)
    assert sorted(starts) == [i * 100 for i in range(10)]


def test_page_starts_max_pages_caps_without_materializing_everything():
    # a realistic unfiltered SRA count: only the pages we will read are produced,
    # but they are still drawn from across the whole range
    starts = _page_starts("random", 5_000_000, 100, max_pages=5)
    assert len(starts) == 5
    assert len(set(starts)) == 5
    assert all(0 <= s < 5_000_000 and s % 100 == 0 for s in starts)
    assert _page_starts("recent", 5_000_000, 100, max_pages=3) == [0, 100, 200]
    assert _page_starts("oldest", 1000, 500, max_pages=1) == [500]


def test_to_entrez_date():
    from datetime import date

    assert _to_entrez_date(date(2022, 3, 4)) == "2022/03/04"
    assert _to_entrez_date("2022-03-04") == "2022/03/04"
    assert _to_entrez_date(None) is None
    for bad in ("03/04/2022", "2022-13-01"):
        try:
            _to_entrez_date(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad!r} should be rejected")
    try:
        _to_entrez_date(20220304)
    except TypeError:
        pass
    else:
        raise AssertionError("an int should be rejected")


def test_int_and_attrs_bag():
    assert _int("12") == 12 and _int(None) is None and _int("x") is None
    el = ET.fromstring(
        "<S><SAMPLE_ATTRIBUTES>"
        "<SAMPLE_ATTRIBUTE><TAG>age</TAG></SAMPLE_ATTRIBUTE>"
        "<SAMPLE_ATTRIBUTE><TAG>sex</TAG><VALUE>F</VALUE></SAMPLE_ATTRIBUTE>"
        "</SAMPLE_ATTRIBUTES></S>"
    )
    # a TAG with no VALUE is present-but-blank, not absent
    assert _attrs_bag(el, "SAMPLE_ATTRIBUTES", "SAMPLE_ATTRIBUTE") == {
        "age": None,
        "sex": "F",
    }
    assert _attrs_bag(el, "MISSING", "X") == {}


def test_load_json_accepts_dict_string_and_long_string():
    import json

    assert _load_json({"a": 1}) == {"a": 1}
    assert _load_json('[{"accession": "X"}]') == [{"accession": "X"}]
    # a JSON string far longer than a legal filename must not trip the path check
    big = json.dumps([{"accession": "X"}] * 3000)
    assert len(_load_json(big)) == 3000


def test_sleep_for_tracks_credentials():
    try:
        set_entrez_credentials()
        assert _sleep_for(None, None) == P._NO_KEY_SLEEP
        assert _sleep_for(None, "KEY") == P._KEY_SLEEP
        assert _sleep_for(1.5, None) == 1.5  # explicit wins
        set_entrez_credentials(api_key="KEY")
        assert _sleep_for(None, None) == P._KEY_SLEEP  # process-wide key counts
    finally:
        set_entrez_credentials()


def _retry_harness(errors, sleep_calls=None):
    """Run _request_with_retry against a stub that raises `errors` then succeeds."""
    seq = list(errors)
    attempts = []

    def fake_get(url, params=None, timeout=None, **kw):
        attempts.append(url)
        if len(attempts) <= len(seq):
            raise seq[len(attempts) - 1]
        return _Resp("done")

    orig_get, orig_sleep = P._SESSION.get, P.time.sleep
    P._SESSION.get = fake_get
    P.time.sleep = lambda s: (sleep_calls.append(s) if sleep_calls is not None else None)
    try:
        return P._request_with_retry("https://example.test/x", sleep=0.11), attempts
    finally:
        P._SESSION.get, P.time.sleep = orig_get, orig_sleep


def test_retry_recovers_from_a_truncated_response_body():
    # regression: a body that dies mid-stream raises ChunkedEncodingError, which is a
    # *sibling* of ConnectionError under RequestException — not a subclass. It used to
    # escape the retry loop and kill a whole harvest.
    err = P.requests.exceptions.ChunkedEncodingError("Response ended prematurely")
    backoff = []
    r, attempts = _retry_harness([err, err], sleep_calls=backoff)
    assert r.text == "done"
    assert len(attempts) == 3  # two failures, then success
    assert backoff == [0.34, 0.68]  # exponential, floored at the no-key pace


def test_retry_covers_the_other_transient_body_failures():
    for exc in (
        P.requests.exceptions.ContentDecodingError("bad gzip"),
        P.requests.ConnectionError("reset"),
        P.requests.Timeout("slow"),
    ):
        r, attempts = _retry_harness([exc])
        assert r.text == "done" and len(attempts) == 2, exc


def test_retry_gives_up_after_five_attempts_and_reraises():
    err = P.requests.exceptions.ChunkedEncodingError("always")
    try:
        _retry_harness([err] * 5)
    except P.requests.exceptions.ChunkedEncodingError:
        pass
    else:
        raise AssertionError("should re-raise once the attempts are exhausted")


def test_retry_does_not_swallow_a_non_transient_error():
    # a malformed URL is a bug, not a hiccup — fail immediately, don't burn 5 attempts
    try:
        _retry_harness([P.requests.exceptions.TooManyRedirects("loop")])
    except P.requests.exceptions.TooManyRedirects:
        pass
    else:
        raise AssertionError("non-transient errors must propagate on the first attempt")


def test_common_omits_unset_credentials():
    try:
        set_entrez_credentials()
        assert _common() == {"tool": "metadata-project"}
        set_entrez_credentials(email="a@b.c", api_key="K")
        assert _common()["email"] == "a@b.c"
        assert _common(email="x@y.z")["email"] == "x@y.z"  # per-call override
    finally:
        set_entrez_credentials()


# --------------------------------------------------------------------------- #
# parsing / from_dict
# --------------------------------------------------------------------------- #
_PACKAGE = """
<EXPERIMENT_PACKAGE_SET><EXPERIMENT_PACKAGE>
  <EXPERIMENT accession="SRX1">
    <TITLE>exp one</TITLE>
    <DESIGN>
      <SAMPLE_DESCRIPTOR accession="SRS1"/>
      <LIBRARY_DESCRIPTOR>
        <LIBRARY_STRATEGY>RNA-Seq</LIBRARY_STRATEGY>
        <LIBRARY_LAYOUT><PAIRED/></LIBRARY_LAYOUT>
      </LIBRARY_DESCRIPTOR>
    </DESIGN>
    <PLATFORM><ILLUMINA><INSTRUMENT_MODEL>NovaSeq</INSTRUMENT_MODEL></ILLUMINA></PLATFORM>
  </EXPERIMENT>
  <STUDY accession="SRP1">
    <IDENTIFIERS><EXTERNAL_ID namespace="BioProject">PRJNA1</EXTERNAL_ID>
      <EXTERNAL_ID namespace="GEO">GSE1</EXTERNAL_ID></IDENTIFIERS>
    <DESCRIPTOR><STUDY_TITLE>a study</STUDY_TITLE>
      <STUDY_ABSTRACT>an abstract</STUDY_ABSTRACT>
      <STUDY_TYPE existing_study_type="Transcriptome Analysis"/></DESCRIPTOR>
  </STUDY>
  <SAMPLE accession="SRS1">
    <IDENTIFIERS><EXTERNAL_ID namespace="BioSample">SAMN1</EXTERNAL_ID></IDENTIFIERS>
    <TITLE>sample one</TITLE>
    <SAMPLE_NAME><TAXON_ID>9606</TAXON_ID>
      <SCIENTIFIC_NAME>Homo sapiens</SCIENTIFIC_NAME></SAMPLE_NAME>
    <SAMPLE_ATTRIBUTES><SAMPLE_ATTRIBUTE><TAG>tissue</TAG>
      <VALUE>liver</VALUE></SAMPLE_ATTRIBUTE></SAMPLE_ATTRIBUTES>
  </SAMPLE>
  <RUN_SET>
    <RUN accession="SRR2" total_spots="20" total_bases="200" published="2020-05-05 00:00:00"/>
    <RUN accession="SRR1" total_spots="10" total_bases="100" published="2014-01-01 00:00:00"/>
  </RUN_SET>
</EXPERIMENT_PACKAGE></EXPERIMENT_PACKAGE_SET>
"""


def _blank_project() -> Project:
    """A Project with build flags set but no network access."""
    p = Project.from_dict({"accession": "SRP1"})
    p._include_samples = p._include_experiments = p._include_runs = True
    p._study_parsed = False
    return p


def test_parse_package_set_offline():
    p = _blank_project()
    p._parse_package_set(_PACKAGE)
    assert p.title == "a study"
    assert p.abstract == "an abstract"
    assert p.bioproject == "PRJNA1"
    assert p.external_ids == {"GEO": "GSE1"}
    assert p.study_type == "Transcriptome Analysis"
    assert list(p.samples) == ["SRS1"]
    assert p.samples["SRS1"].attributes == {"tissue": "liver"}
    assert p.samples["SRS1"].biosample == "SAMN1"
    e = p.experiments[0]
    assert e.library_strategy == "RNA-Seq" and e.library_layout == "PAIRED"
    assert e.platform == "ILLUMINA" and e.instrument_model == "NovaSeq"
    assert e.sample_ids == ["SRS1"]


def test_note_published_takes_earliest_run_in_the_package():
    # regression: it used to read only the first RUN element, so a package whose
    # runs were released on different days reported whichever came first in the XML
    p = _blank_project()
    p._parse_package_set(_PACKAGE)
    assert p.published == "2014-01-01 00:00:00"


def test_from_dict_restores_build_state():
    p = Project.from_dict(
        {
            "accession": "SRP1",
            "title": "t",
            "samples": {"SRS1": {"accession": "SRS1"}},
            "experiments": [{"accession": "SRX1", "runs": [{"accession": "SRR1"}]}],
            "publications": [{"id": "1", "type": "ePubmed"}],
        }
    )
    assert p._study_parsed is True
    assert p._include_samples and p._include_experiments and p._include_runs
    assert p._include_publications and p._summary_only is False
    # pacing follows the process-wide credentials rather than a hardcoded 0.34
    try:
        set_entrez_credentials(api_key="KEY")
        assert Project.from_dict({"accession": "X"})._sleep == P._KEY_SLEEP
    finally:
        set_entrez_credentials()


def test_to_dataframe_keeps_every_pooled_sample():
    p = Project.from_dict(
        {
            "accession": "SRP1",
            "samples": {
                "SRS1": {"accession": "SRS1", "attributes": {"tissue": "liver"}},
                "SRS2": {"accession": "SRS2", "attributes": {"tissue": "lung"}},
            },
            "experiments": [
                {
                    "accession": "SRX1",
                    "sample_ids": ["SRS1", "SRS2"],  # pooled / multiplexed
                    "runs": [{"accession": "SRR1"}],
                }
            ],
        }
    )
    df = p.to_dataframe()
    # regression: only SRS1 used to survive, dropping SRS2's attribute bag
    assert sorted(df["sample_accession"]) == ["SRS1", "SRS2"]
    assert sorted(df["sample.tissue"]) == ["liver", "lung"]


def test_to_dataframe_without_samples_still_emits_run_rows():
    p = Project.from_dict(
        {
            "accession": "SRP1",
            "experiments": [{"accession": "SRX1", "runs": [{"accession": "SRR1"}]}],
        }
    )
    df = p.to_dataframe()
    assert len(df) == 1 and df["sample_accession"].isna().all()


def test_fetch_publications_dedupes():
    xml = (
        "<R><Publication id='111'><DbType>ePubmed</DbType></Publication>"
        "<Publication id='222'><DbType>ePubmed</DbType></Publication>"
        "<Publication id='111'><DbType>ePubmed</DbType></Publication></R>"
    )
    p = Project.from_dict({"accession": "SRP1"})
    p._get = lambda *a, **k: type("R", (), {"text": xml})()
    assert [x.id for x in p._fetch_publications("PRJNA1")] == ["111", "222"]


class _BioProjectStub:
    """Answer esearch with a uid and efetch with a canned record, logging calls."""

    def __init__(self, uid, record):
        self.uid, self.record, self.calls = uid, record, []

    def __call__(self, endpoint, **params):
        self.calls.append((endpoint, params))
        if endpoint == "esearch.fcgi":
            idlist = [self.uid] if self.uid else []
            return type(
                "R", (), {"json": lambda _s: {"esearchresult": {"idlist": idlist}}}
            )()
        return type("R", (), {"text": self.record})()


def _bioproject_project(stub):
    p = Project.from_dict({"accession": "ERP1"})
    p._get = stub
    p._sleep = 0  # no pacing needed against a stub
    return p


def test_fetch_publications_resolves_non_ncbi_accession():
    # regression: efetch db=bioproject doesn't resolve accessions, it strips the
    # PRJ?? prefix and uses the digits as a uid. PRJEB47383 (really uid 778158)
    # used to fetch unrelated uid 47383 and adopt whatever papers it found there.
    stub = _BioProjectStub(
        "778158",
        "<R><DocumentSummary><ArchiveID accession='PRJEB47383'/>"
        "<Publication id='999'><DbType>ePubmed</DbType></Publication>"
        "</DocumentSummary></R>",
    )
    p = _bioproject_project(stub)
    assert [x.id for x in p._fetch_publications("PRJEB47383")] == ["999"]
    # the accession is resolved first, exactly, and the uid is what gets fetched
    assert stub.calls[0][0] == "esearch.fcgi"
    assert stub.calls[0][1]["term"] == "PRJEB47383[Project Accession]"
    assert stub.calls[1][1]["id"] == "778158"


def test_fetch_publications_skips_ncbi_accession_resolution():
    # PRJNA accessions *are* their uid, so the extra round-trip is skipped
    stub = _BioProjectStub(None, "<R><DocumentSummary/></R>")
    _bioproject_project(stub)._fetch_publications("PRJNA646996")
    assert [c[0] for c in stub.calls] == ["efetch.fcgi"]
    assert stub.calls[0][1]["id"] == "PRJNA646996"


def test_fetch_publications_rejects_a_record_for_another_project():
    # a record naming only some other project describes a different study
    stub = _BioProjectStub(
        "361249",
        "<R><DocumentSummary><ArchiveID accession='PRJNA13694'/>"
        "<Publication id='15001713'><DbType>ePubmed</DbType></Publication>"
        "</DocumentSummary></R>",
    )
    assert _bioproject_project(stub)._fetch_publications("PRJEB13694") == []


def test_fetch_publications_empty_when_accession_does_not_resolve():
    stub = _BioProjectStub(None, "<R/>")
    assert _bioproject_project(stub)._fetch_publications("PRJEB47383") == []
    assert [c[0] for c in stub.calls] == ["esearch.fcgi"]  # no pointless efetch


# --------------------------------------------------------------------------- #
# classify_publication: stubbed HTTP, no network
# --------------------------------------------------------------------------- #
class _Resp:
    def __init__(self, payload="", data=None):
        self.text = payload
        self._data = data
        self.status_code = 200

    def json(self):
        return self._data

    def raise_for_status(self):
        pass


def _epmc(hits):
    return {"resultList": {"result": hits}}


def _stub_requests(monkey: dict):
    """Route the shared Session's GET by URL prefix to a canned response."""
    def fake_get(url, params=None, timeout=None, **kw):
        for prefix, resp in monkey.items():
            if url.startswith(prefix):
                if callable(resp):
                    return resp(params or {})
                return resp
        raise AssertionError(f"unexpected request to {url}")

    return fake_get


def _with_stub(monkey, fn):
    # network goes through the module-level Session, so that is what gets stubbed
    original = P._SESSION.get
    P._SESSION.get = _stub_requests(monkey)
    try:
        return fn()
    finally:
        P._SESSION.get = original


def test_classify_unindexed_doi_falls_back_to_unpaywall():
    # regression: an OA paper outside PubMed (e.g. Frontiers in Marine Science,
    # 10.3389/fmars.2022.930017) used to come back "unknown" and be dropped
    calls = []

    def unpaywall(params):
        calls.append(params)
        return _Resp(data={"is_oa": True})

    got = _with_stub(
        {P.EPMC_SEARCH: _Resp(data=_epmc([])), P.UNPAYWALL: unpaywall},
        lambda: P.classify_publication("10.3389/fmars.2022.930017", email="a@b.c"),
    )
    assert got == "oa"
    assert calls[0]["email"] == "a@b.c"  # Unpaywall requires a contact address


def test_classify_unindexed_doi_without_oa_copy_is_unknown():
    got = _with_stub(
        {P.EPMC_SEARCH: _Resp(data=_epmc([])), P.UNPAYWALL: _Resp(data={"is_oa": False})},
        lambda: P.classify_publication("10.1/nope", email="a@b.c"),
    )
    assert got == "unknown"


def test_classify_unindexed_pmid_is_unknown_without_a_doi_to_check():
    got = _with_stub(
        {P.EPMC_SEARCH: _Resp(data=_epmc([]))},  # any Unpaywall call would assert
        lambda: P.classify_publication("99999999999", email="a@b.c"),
    )
    assert got == "unknown"


def test_classify_falls_back_to_unpaywall_when_indexed_but_not_open():
    # e.g. 10.1111/age.13334: indexed, no PMC copy, free PDF at the publisher
    hits = [{"pmcid": None, "isOpenAccess": "N", "doi": "10.1111/age.13334"}]
    got = _with_stub(
        {P.EPMC_SEARCH: _Resp(data=_epmc(hits)), P.UNPAYWALL: _Resp(data={"is_oa": True})},
        lambda: P.classify_publication("29282718", email="a@b.c"),
    )
    assert got == "oa"


def test_classify_paywall_when_nothing_hosts_it_freely():
    hits = [{"pmcid": None, "isOpenAccess": "N", "doi": "10.1002/path.5026"}]
    got = _with_stub(
        {P.EPMC_SEARCH: _Resp(data=_epmc(hits)), P.UNPAYWALL: _Resp(data={"is_oa": False})},
        lambda: P.classify_publication("29282718", email="a@b.c"),
    )
    assert got == "paywall"


def test_classify_pmc_oa_subset_is_authoritative():
    hits = [{"pmcid": "PMC1", "isOpenAccess": "N"}]
    got = _with_stub(
        {
            P.EPMC_SEARCH: _Resp(data=_epmc(hits)),
            P.PMC_OA: _Resp("<OA><records><record id='PMC1'/></records></OA>"),
        },
        lambda: P.classify_publication("1", email="a@b.c"),
    )
    assert got == "oa"


def test_classify_partial_when_in_pmc_but_not_oa_licensed():
    hits = [{"pmcid": "PMC1", "isOpenAccess": "N"}]
    got = _with_stub(
        {
            P.EPMC_SEARCH: _Resp(data=_epmc(hits)),
            P.PMC_OA: _Resp("<OA><error code='idIsNotOpenAccess'/></OA>"),
        },
        lambda: P.classify_publication("1", email="a@b.c"),
    )
    assert got == "partial"


def test_ncbi_credentials_are_not_sent_to_third_party_hosts():
    # regression: the NCBI api_key was splatted into the Europe PMC query string
    seen: dict[str, dict] = {}

    def record(host):
        def _f(params):
            seen[host] = params
            if host == "epmc":
                return _Resp(data=_epmc([{"pmcid": "PMC1", "isOpenAccess": "Y"}]))
            return _Resp("<OA><records><record id='PMC1'/></records></OA>")

        return _f

    try:
        set_entrez_credentials(email="me@psu.edu", api_key="SECRET")
        _with_stub(
            {P.EPMC_SEARCH: record("epmc"), P.PMC_OA: record("pmc")},
            lambda: P.classify_publication("1"),
        )
    finally:
        set_entrez_credentials()
    assert "api_key" not in seen["epmc"] and "email" not in seen["epmc"]
    assert seen["pmc"]["api_key"] == "SECRET"  # NCBI host still gets them


def test_unpaywall_skipped_without_a_contact_email():
    try:
        set_entrez_credentials()  # no email configured
        # any Unpaywall request would hit the assert in the stub router
        got = _with_stub(
            {P.EPMC_SEARCH: _Resp(data=_epmc([]))},
            lambda: P.classify_publication("10.1/x"),
        )
    finally:
        set_entrez_credentials()
    assert got == "unknown"


def test_publication_classes_caches_without_network():
    p = Project.from_dict(
        {
            "accession": "X",
            "publications": [{"id": "1", "accessibility_type": "oa"}],
        }
    )
    # no stub installed: a network call here would raise
    assert p.publication_classes() == ["oa"]


# --------------------------------------------------------------------------- #
# search_studies page-failure handling (stubbed HTTP, no network)
# --------------------------------------------------------------------------- #
def _esummary_xml(*srps):
    items = "".join(
        f"<Item Name='ExpXml'>&lt;Study acc=\"{s}\" /&gt;</Item>" for s in srps
    )
    return f"<eSummaryResult><DocSum>{items}</DocSum></eSummaryResult>"


def _search_stub(page_results):
    """esearch returns a fixed history handle; esummary answers per page offset.

    Results are keyed by page offset, not by call, so a page listed as an
    exception fails *every* attempt. That matters: _request_with_retry sits
    underneath and retries transient errors five times, so a page that fails only
    once never reaches search_studies' own handler at all.
    """
    pages = list(page_results)
    order = []

    def fake(url, params=None, data=None, timeout=None, **kw):
        p = params or data or {}
        if "esearch" in url:
            # a large count so _page_starts produces plenty of page offsets;
            # a small one silently caps the loop and hides what we're testing
            return _Resp(data={"esearchresult": {
                "count": "1000000", "webenv": "W1", "querykey": "1"}})
        start = p.get("retstart")
        if start not in order:
            order.append(start)
        idx = order.index(start)
        nxt = pages[idx] if idx < len(pages) else RuntimeError("no more pages")
        if isinstance(nxt, Exception):
            raise nxt
        return _Resp(nxt)

    return fake


def _with_search_stub(page_results, fn):
    orig_get, orig_post, orig_sleep = P._SESSION.get, P._SESSION.post, P.time.sleep
    stub = _search_stub(page_results)
    P._SESSION.get = P._SESSION.post = stub
    P.time.sleep = lambda s: None
    try:
        return fn()
    finally:
        P._SESSION.get, P._SESSION.post, P.time.sleep = orig_get, orig_post, orig_sleep


def test_search_studies_skips_a_failed_page_and_warns():
    import warnings as _w

    boom = P.requests.exceptions.ChunkedEncodingError("truncated")
    pages = [_esummary_xml("SRP1", "SRP2"), boom, _esummary_xml("SRP3", "SRP4")]
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        got = _with_search_stub(
            pages,
            # max_scanned raised so max_pages doesn't cap the loop below 3 pages
            lambda: P.search_studies(max_studies=4, sort="recent", max_scanned=100000),
        )
    # the surviving pages still yield their studies
    assert got == ["SRP1", "SRP2", "SRP3", "SRP4"]
    assert len(caught) == 1 and "1 esummary page(s) failed" in str(caught[0].message)


def test_search_studies_stops_after_consecutive_page_failures():
    import warnings as _w

    boom = P.requests.exceptions.ChunkedEncodingError("truncated")
    # one good page, then a wall of failures (an expired WebEnv looks like this)
    pages = [_esummary_xml("SRP1")] + [boom] * 20
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        got = _with_search_stub(
            pages, lambda: P.search_studies(max_studies=50, sort="recent")
        )
    # gives up rather than grinding through every remaining page, but keeps what
    # it already enumerated instead of raising it all away
    assert got == ["SRP1"]
    assert len(caught) == 1
    assert f"{P._MAX_CONSECUTIVE_PAGE_FAILURES} esummary page(s) failed" in str(
        caught[0].message
    )


def test_a_page_that_fails_once_is_absorbed_by_the_retry_layer():
    # the two mechanisms are layered: _request_with_retry handles the common case,
    # and search_studies' page handler is only the last resort behind it. A page
    # that recovers on retry must not be reported as a lost page.
    import warnings as _w

    calls = []

    def flaky(url, params=None, data=None, timeout=None, **kw):
        p = params or data or {}
        if "esearch" in url:
            return _Resp(data={"esearchresult": {
                "count": "1000000", "webenv": "W1", "querykey": "1"}})
        calls.append(p.get("retstart"))
        if len(calls) == 1:  # first attempt at the first page dies mid-body
            raise P.requests.exceptions.ChunkedEncodingError("truncated")
        return _Resp(_esummary_xml("SRP1", "SRP2"))

    orig = (P._SESSION.get, P._SESSION.post, P.time.sleep)
    P._SESSION.get = P._SESSION.post = flaky
    P.time.sleep = lambda s: None
    try:
        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            got = P.search_studies(max_studies=2, sort="recent")
    finally:
        P._SESSION.get, P._SESSION.post, P.time.sleep = orig
    assert got == ["SRP1", "SRP2"]
    assert calls == [0, 0]  # retried the same offset, not skipped to the next
    assert caught == []


def test_search_studies_is_silent_when_no_page_fails():
    import warnings as _w

    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        got = _with_search_stub(
            [_esummary_xml("SRP1", "SRP2")],
            lambda: P.search_studies(max_studies=2, sort="recent"),
        )
    assert got == ["SRP1", "SRP2"] and caught == []


def test_scan_iter_yields_one_triple_per_accession():
    calls = []

    def fake_summary(acc, **kw):
        calls.append(acc)
        if acc == "BAD":
            raise ValueError("no such study")
        p = Project.from_dict({"accession": acc, "record_count": 10})
        return p

    orig, orig_sleep = P.Project.summary, P.time.sleep
    P.Project.summary = staticmethod(fake_summary)
    P.time.sleep = lambda s: None
    try:
        out = list(P.scan_iter(["A", "BAD", "B"]))
    finally:
        P.Project.summary, P.time.sleep = orig, orig_sleep
    assert [acc for acc, _, _ in out] == ["A", "BAD", "B"]
    assert out[0][1] is not None and out[0][2] is None
    assert out[1][1] is None and isinstance(out[1][2], ValueError)
    # and scan() is still exactly this collected into two dicts
    assert calls == ["A", "BAD", "B"]


# --------------------------------------------------------------------------- #
# standalone runner (mirrors test_project.py)
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