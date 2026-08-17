"""Unit tests for project.py that never touch the network.

The live suite in test_project.py is the real end-to-end check, but it can't run
in CI or offline and a flaky NCBI response fails it. Everything here is pure logic
or a stubbed request, so it runs anywhere in well under a second:

    python tests/test_offline.py
    pytest tests/test_offline.py
"""

from __future__ import annotations

import collections
import json
import os
import pytest
import sys
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import audit as A  # noqa: E402
import claude as C  # noqa: E402
import dataset as D  # noqa: E402
import main as M  # noqa: E402
import project as P  # noqa: E402
import reconstruct as R  # noqa: E402
import schema  # noqa: E402
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
from schema import TargetSchema  # noqa: E402


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


def test_within_years_reaches_entrez_as_a_pdat_reldate(monkeypatch):
    """The date window must actually be sent, not merely accepted as an argument.

    Live tests cannot cover this. ``search_studies`` bounds how far it scans, so
    it returns the same accessions with and without a date window; and the field
    the filter applies to — the Entrez record date — now reads as the current
    month for effectively every record, because NCBI bumps it on re-index. Both
    were checked by deleting the date parameters and watching the live
    assertions still pass. Capturing the outgoing request is what is left.
    """
    calls = []

    class _Res:
        def json(self):
            return {"esearchresult": {"count": "0"}}  # stop after the first call

    def fake_eutils(endpoint, **params):
        calls.append((endpoint, params))
        return _Res()

    monkeypatch.setattr(P, "_eutils", fake_eutils)

    P.search_studies(organism="Homo sapiens", within_years=5, max_studies=1)
    assert calls, "search_studies made no request at all"
    endpoint, params = calls[0]
    assert endpoint == "esearch.fcgi"
    assert params["datetype"] == "pdat"
    # 5 years in days, which is what reldate takes - passing 5 would ask for a
    # five-day window and silently return almost nothing
    assert params["reldate"] == 1826

    calls.clear()
    P.search_studies(organism="Homo sapiens", max_studies=1)
    assert "reldate" not in calls[0][1], "an unfiltered search must send no window"
    assert "datetype" not in calls[0][1]


def test_explicit_date_range_reaches_entrez_as_min_and_maxdate(monkeypatch):
    # The other branch: after_date/before_date become an explicit pdat window,
    # with open ends filled in rather than omitted.
    from datetime import date

    calls = []

    class _Res:
        def json(self):
            return {"esearchresult": {"count": "0"}}

    monkeypatch.setattr(P, "_eutils",
                        lambda endpoint, **params: (calls.append(params), _Res())[1])

    P.search_studies(after_date=date(2022, 1, 1), before_date=date(2024, 6, 30),
                     max_studies=1)
    assert calls[0]["datetype"] == "pdat"
    assert calls[0]["mindate"] == "2022/01/01"
    assert calls[0]["maxdate"] == "2024/06/30"

    calls.clear()
    P.search_studies(after_date=date(2022, 1, 1), max_studies=1)
    assert calls[0]["mindate"] == "2022/01/01"
    assert calls[0]["maxdate"] == "3000/01/01"  # open-ended, not absent


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


def _xml_retry_harness(bodies, sleep_calls=None):
    """Run _request_with_retry(xml=True) against a stub serving `bodies` in turn."""
    seq = list(bodies)
    attempts = []

    def fake_get(url, params=None, timeout=None, **kw):
        attempts.append(url)
        return _Resp(seq[min(len(attempts) - 1, len(seq) - 1)])

    orig_get, orig_sleep = P._SESSION.get, P.time.sleep
    P._SESSION.get = fake_get
    P.time.sleep = lambda s: (sleep_calls.append(s) if sleep_calls is not None else None)
    try:
        return P._request_with_retry("https://example.test/x", sleep=0.11, xml=True), attempts
    finally:
        P._SESSION.get, P.time.sleep = orig_get, orig_sleep


def test_retry_recovers_from_a_body_that_is_complete_but_corrupt_as_xml():
    # regression: E-utilities serves 200s whose body is truncated mid-token. That
    # raises no requests error at all, so parsing at the call site let ParseError
    # escape the retry layer and downgrade a whole study (SRP071759, 3002 records,
    # where the identical request succeeded on a later attempt).
    truncated = "<EXPERIMENT_PACKAGE_SET><EXPERIMENT_PACKAGE><TITLE>half a doc"
    backoff = []
    root, attempts = _xml_retry_harness([truncated, truncated, "<OK/>"], backoff)
    assert root.tag == "OK"
    assert len(attempts) == 3
    assert backoff == [0.34, 0.68]  # same exponential schedule as a dropped body


def test_retry_reraises_a_parse_error_that_never_resolves():
    # a body that is *always* malformed is a real failure, not a hiccup
    try:
        _xml_retry_harness(["<not-xml"])
    except ET.ParseError:
        pass
    else:
        raise AssertionError("a persistent ParseError must propagate")


def test_xml_responses_are_parsed_exactly_once():
    # parsing in the retry layer must not double-parse: full builds download
    # hundreds of multi-megabyte packages per study
    parses = []
    orig = P.ET.fromstring

    def counting(text, *a, **kw):
        parses.append(len(text))
        return orig(text, *a, **kw)

    P.ET.fromstring = counting
    try:
        root, _ = _xml_retry_harness(["<OK/>"])
    finally:
        P.ET.fromstring = orig
    assert root.tag == "OK" and len(parses) == 1


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
    p._parse_package_set(ET.fromstring(_PACKAGE))
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
    p._parse_package_set(ET.fromstring(_PACKAGE))
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
    p._get = lambda *a, **k: ET.fromstring(xml)  # xml=True -> a parsed root
    assert [x.id for x in p._fetch_publications("PRJNA1")] == ["111", "222"]


class _BioProjectStub:
    """Answer esearch with a uid and efetch with a canned record, logging calls."""

    def __init__(self, uid, record):
        self.uid, self.record, self.calls = uid, record, []

    def __call__(self, endpoint, xml=False, **params):
        self.calls.append((endpoint, params))
        if endpoint == "esearch.fcgi":
            idlist = [self.uid] if self.uid else []
            return type(
                "R", (), {"json": lambda _s: {"esearchresult": {"idlist": idlist}}}
            )()
        assert xml, "efetch db=bioproject must parse inside the retry layer"
        return ET.fromstring(self.record)


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
# schema.TargetSchema (pure logic — nothing here may touch the network)
# --------------------------------------------------------------------------- #
# ENA's field vocabulary, verbatim and in order. Hard-coded rather than derived
# from TargetSchema so the two can disagree: a field renamed, reordered or
# dropped on the Python side has to be a deliberate edit here too.
_SCHEMA_FIELDS = (
    "id", "age", "base_count", "broad_scale_environmental_context", "broker_name",
    "cell_line", "cell_type", "center_name", "checklist", "collected_by", "collection_date",
    "country", "datahub", "description", "dev_stage", "environment_biome",
    "environment_feature",
    "environment_material", "environmental_medium", "experiment_accession",
    "experiment_alias", "experiment_title", "first_created", "first_public", "host",
    "host_scientific_name", "host_sex", "host_tax_id", "instrument_model",
    "instrument_platform", "isolation_source", "last_updated",
    "library_construction_protocol", "library_layout", "library_name",
    "library_selection", "library_source", "library_strategy",
    "local_environmental_context", "ncbi_reporting_standard", "project_name",
    "read_count", "run_accession", "run_alias", "sample_accession", "sample_alias",
    "sample_capture_status", "sample_description", "sample_title", "scientific_name",
    "secondary_sample_accession", "secondary_study_accession", "sequencing_method",
    "sex", "strain", "study_accession", "study_alias", "study_title",
    "submission_accession", "submitted_format", "submitted_read_type", "tag", "tax_id",
    "tissue_type", "treatment",
)


def _study(**overrides):
    """A minimal full-depth Project: 1 experiment, 1 sample, 2 runs.

    Two runs, not one, so the experiment grain is actually exercised: they must
    collapse to a single record with their counts summed. SRR2 is both listed
    second and released *earlier*, so ``first_public`` has to take the earliest
    rather than the first.
    """
    d = {
        "accession": "SRP1",
        "bioproject": "PRJNA1",
        "title": "study title",
        "abstract": "study abstract",
        "published": "2020-03-04 05:06:07",
        "samples": {
            "SRS1": {
                "accession": "SRS1",
                "biosample": "SAMN1",
                "taxon_id": "9606",
                "scientific_name": "Homo sapiens",
                "title": "sample title",
            }
        },
        "experiments": [
            {
                "accession": "SRX1",
                "sample_ids": ["SRS1"],
                "title": "experiment title",
                "library_strategy": "RNA-Seq",
                "platform": "ILLUMINA",
                "runs": [
                    {
                        "accession": "SRR1",
                        "total_spots": 10,
                        "total_bases": 100,
                        "published": "2020-03-04 05:06:07",
                    },
                    {
                        "accession": "SRR2",
                        "total_spots": 5,
                        "total_bases": 50,
                        "published": "2019-12-31 00:00:00",
                    },
                ],
            }
        ],
    }
    d.update(overrides)
    return Project.from_dict(d)


def test_schema_field_set_matches_enas_vocabulary():
    assert TargetSchema.field_names() == list(_SCHEMA_FIELDS)
    # types are load-bearing: they drive coercion, so guard the non-text ones
    assert TargetSchema.field_type("base_count") is int
    assert TargetSchema.field_type("tax_id") is int
    assert TargetSchema.field_type("collection_date") is datetime
    assert TargetSchema.field_type("run_accession") is str
    assert TargetSchema.field_type("host") is str
    assert all(TargetSchema.field_type(n) in schema.FIELD_TYPES
               for n in TargetSchema.field_names())
    # the sidecars are ours, not ENA's, and must stay out of the field set
    for sidecar in ("provenance", "confidence", "runs"):
        assert sidecar not in TargetSchema.field_names()


def test_every_field_is_assigned_a_hierarchy_level():
    by_level = {lvl: TargetSchema.fields_at_level(lvl) for lvl in schema.LEVELS}
    assert sum(len(v) for v in by_level.values()) == len(_SCHEMA_FIELDS)
    assert TargetSchema.level("study_title") == schema.STUDY
    assert TargetSchema.level("library_strategy") == schema.EXPERIMENT
    assert TargetSchema.level("collection_date") == schema.SAMPLE
    assert TargetSchema.level("read_count") == schema.RUN


def test_values_are_coerced_on_assignment_not_just_construction():
    r = TargetSchema(id="SRR1")
    r.read_count = " 1,234 "  # LLM/CSV formatting
    r.collection_date = "2015"  # partial date -> padded to Jan 1st
    r.host = "  Homo sapiens  "
    r.strain = "   "  # blank is absent, not empty-string
    assert r.read_count == 1234
    assert r.collection_date == datetime(2015, 1, 1)
    assert r.host == "Homo sapiens" and r.strain is None
    # constructor path coerces identically
    assert TargetSchema(id="x", tax_id="9606").tax_id == 9606


def test_date_parsing_accepts_every_shape_the_pipeline_emits():
    r = TargetSchema(id="x")
    for value, expected in [
        ("2020-03-04 05:06:07", datetime(2020, 3, 4, 5, 6, 7)),  # Project.published
        ("2020-03-04T05:06:07", datetime(2020, 3, 4, 5, 6, 7)),
        ("2020-03-04", datetime(2020, 3, 4)),
        ("2020-03", datetime(2020, 3, 1)),
        ("2020", datetime(2020, 1, 1)),
    ]:
        r.first_public = value
        assert r.first_public.replace(tzinfo=None) == expected, value
    r.first_public = "2020-03-04T05:06:07Z"  # a UTC-stamped ISO string
    assert r.first_public.replace(tzinfo=None) == datetime(2020, 3, 4, 5, 6, 7)


def test_unparseable_values_raise_rather_than_becoming_null():
    # a silent null would read downstream as "the archive never said"
    for field_name, bad in [("read_count", "about 4 million"), ("collection_date", "spring")]:
        try:
            setattr(TargetSchema(id="x"), field_name, bad)
        except ValueError as exc:
            assert field_name in str(exc)
        else:
            raise AssertionError(f"{field_name}={bad!r} should not have been accepted")


def test_id_is_required():
    for bad in ("", None, "   "):
        try:
            TargetSchema(id=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"id={bad!r} should have been rejected")


def test_dict_round_trip_preserves_values_and_provenance():
    r = TargetSchema(id="SRR1", read_count=5, collection_date="2015-06-02")
    r.provenance["collection_date"] = "inferred_from_text"
    back = TargetSchema.from_dict(json.loads(json.dumps(r.to_dict())))
    assert back == r
    assert back.provenance == {"collection_date": "inferred_from_text"}
    # unknown keys are tolerated (a newer schema may add some) unless strict
    assert TargetSchema.from_dict({"id": "x", "not_a_field": 1}).id == "x"
    try:
        TargetSchema.from_dict({"id": "x", "not_a_field": 1}, strict=True)
    except ValueError as exc:
        assert "not_a_field" in str(exc)
    else:
        raise AssertionError("strict=True should reject an invented field")


def test_to_dict_drops_nulls_and_squares_the_table_on_request():
    r = TargetSchema(id="SRX1", first_public="2020-03-04 05:06:07")
    r.provenance["id"] = "direct"
    r.runs = [{"run_accession": "SRR1"}]
    # omit_none drops every unset field; dates become ISO strings
    assert r.to_dict()["first_public"] == "2020-03-04T05:06:07"
    assert "host" not in r.to_dict()
    # the three sidecars ride along, and omit_none=False squares the table
    assert {"provenance", "runs", "confidence"} <= set(r.to_dict(omit_none=False))
    assert len(r.to_dict(omit_none=False)) == len(_SCHEMA_FIELDS) + 3


def test_coverage_counts_populated_fields():
    r = TargetSchema(id="SRR1")
    assert r.filled() == ["id"] and len(r.missing()) == len(_SCHEMA_FIELDS) - 1
    assert r.coverage() == 1 / len(_SCHEMA_FIELDS)


def test_from_project_maps_the_ena_accession_pairs_without_swapping_them():
    (r,) = TargetSchema.from_project(_study())
    # ENA's *_accession is the BioProject/BioSample; secondary_* is the SRP/SRS.
    # Project stores them the other way round, so a positional map would swap.
    assert (r.study_accession, r.secondary_study_accession) == ("PRJNA1", "SRP1")
    assert (r.sample_accession, r.secondary_sample_accession) == ("SAMN1", "SRS1")
    assert (r.id, r.experiment_accession) == ("SRX1", "SRX1")
    assert r.instrument_platform == "ILLUMINA"  # Project calls this `platform`
    assert r.description == "study abstract"
    assert all(v == "direct" for v in r.provenance.values())
    assert "id" not in r.provenance


def test_from_project_emits_one_record_per_experiment_with_its_runs_attached():
    (r,) = TargetSchema.from_project(_study())  # 2 runs -> 1 record
    assert [run["run_accession"] for run in r.runs] == ["SRR1", "SRR2"]
    assert (r.read_count, r.base_count) == (15, 150)  # summed over both runs
    # earliest run, not the first listed
    assert r.first_public == datetime(2019, 12, 31)
    # the scalar RUN fields have no single value at this grain
    assert r.run_accession is None and r.submitted_format is None
    # per-run detail survives for the download menu
    assert r.runs[1] == {"run_accession": "SRR2", "read_count": 5,
                         "base_count": 50, "first_public": "2019-12-31 00:00:00"}


def test_run_counts_are_absent_not_zero_when_unreported():
    # summing an empty list would report 0 reads, which reads as a real measurement
    quiet = _study(experiments=[{"accession": "SRX1", "sample_ids": ["SRS1"],
                                 "runs": [{"accession": "SRR1"}]}])
    (r,) = TargetSchema.from_project(quiet)
    assert r.read_count is None and r.base_count is None
    assert len(r.runs) == 1
    # no run date either -> falls back to the study's release date
    assert r.first_public == datetime(2020, 3, 4, 5, 6, 7)


def test_from_project_degrades_one_level_at_a_time():
    full = _study()
    # experiments stripped -> keyed on the sample; samples too -> the study
    no_runs = _study(experiments=[{"accession": "SRX1", "sample_ids": ["SRS1"]}])
    no_exps = _study(experiments=[])
    no_samples = _study(samples={})
    summary = _study(samples={}, experiments=[])

    assert [r.id for r in TargetSchema.from_project(full)] == ["SRX1"]
    assert [r.id for r in TargetSchema.from_project(no_runs)] == ["SRX1"]
    assert [r.id for r in TargetSchema.from_project(no_exps)] == ["SRS1"]
    assert [r.id for r in TargetSchema.from_project(no_samples)] == ["SRX1"]
    assert [r.id for r in TargetSchema.from_project(summary)] == ["SRP1"]

    # dropping runs costs only the run list and the counts, not the record
    stripped = TargetSchema.from_project(no_runs)[0]
    assert stripped.runs == [] and stripped.read_count is None
    assert stripped.library_strategy is None  # this fixture omits it too

    # each rung drops only the levels it lost; study fields survive throughout
    assert TargetSchema.from_project(no_samples)[0].scientific_name is None
    assert TargetSchema.from_project(no_exps)[0].library_strategy is None
    assert all(
        TargetSchema.from_project(p)[0].study_title == "study title"
        for p in (full, no_runs, no_exps, no_samples, summary)
    )
    # with no run to date it, first_public falls back to the study release date
    assert TargetSchema.from_project(summary)[0].first_public == datetime(2020, 3, 4, 5, 6, 7)


def test_from_project_gives_pooled_samples_unique_ids():
    # one SRX over two SRS: expands per sample, and `id` has to stay unique,
    # so a bare experiment accession would collapse the pool into one record
    pooled = _study(
        samples={"SRS1": {"accession": "SRS1"}, "SRS2": {"accession": "SRS2"}},
        experiments=[
            {
                "accession": "SRX1",
                "sample_ids": ["SRS1", "SRS2"],
                "runs": [{"accession": "SRR1"}],
            }
        ],
    )
    records = TargetSchema.from_project(pooled)
    assert [r.id for r in records] == ["SRX1.SRS1", "SRX1.SRS2"]
    assert len({r.id for r in records}) == len(records)
    # only `id` is composite — experiment_accession stays the archive's own value
    assert {r.experiment_accession for r in records} == {"SRX1"}
    assert [r.secondary_sample_accession for r in records] == ["SRS1", "SRS2"]
    # both records carry the shared run, since the pool was sequenced together
    assert all([run["run_accession"] for run in r.runs] == ["SRR1"] for r in records)


def test_from_project_leaves_unpooled_ids_bare():
    # the overwhelming majority: ids must stay directly comparable with ENA's
    two_samples = _study(
        samples={"SRS1": {"accession": "SRS1"}, "SRS2": {"accession": "SRS2"}},
        experiments=[
            {"accession": "SRX1", "sample_ids": ["SRS1"], "runs": [{"accession": "SRR1"}]},
            {"accession": "SRX2", "sample_ids": ["SRS2"], "runs": [{"accession": "SRR2"}]},
        ],
    )
    assert [r.id for r in TargetSchema.from_project(two_samples)] == ["SRX1", "SRX2"]
    # an unresolvable sample cannot disambiguate, so no suffix is invented
    orphan = _study(samples={}, experiments=[
        {"accession": "SRX1", "sample_ids": ["SRS1", "SRS2"], "runs": [{"accession": "SRR1"}]}
    ])
    assert [r.id for r in TargetSchema.from_project(orphan)] == ["SRX1"]


class _NoNetwork:
    """Fail the test if anything inside the block attempts an HTTP request."""

    def __enter__(self):
        def boom(*a, **kw):
            raise AssertionError("unexpected network call")

        self._saved = (P._request_with_retry, P._SESSION.get, P._SESSION.post)
        P._request_with_retry = P._SESSION.get = P._SESSION.post = boom
        return self

    def __exit__(self, *exc):
        P._request_with_retry, P._SESSION.get, P._SESSION.post = self._saved
        return False


def test_from_project_never_expands_an_incomplete_project():
    """A thin Project yields a thin record set — it must not silently re-fetch."""
    with _NoNetwork():
        summary = _study(samples={}, experiments=[])
        (r,) = TargetSchema.from_project(summary)
        assert r.id == "SRP1" and r.library_strategy is None


def test_from_project_survives_an_oversized_stub():
    # max_records aborts the build with only accession + record_count set, and
    # record_count has no home in the ENA schema, so the record is nearly empty
    stub = Project.from_dict({"accession": "SRP1", "record_count": 99999})
    (r,) = TargetSchema.from_project(stub)
    assert r.id == "SRP1" and r.secondary_study_accession == "SRP1"
    assert r.study_title is None and r.filled() == ["id", "secondary_study_accession"]


def test_provenance_rejects_an_unknown_class():
    r = TargetSchema(id="SRX1")
    for bad in ("guessed", "inferred", "DIRECT", "", None):
        try:
            r.provenance["country"] = bad
        except ValueError as exc:
            assert "country" in str(exc)
        else:
            raise AssertionError(f"provenance class {bad!r} should be rejected")
    for good in schema.PROVENANCE_CLASSES:
        r.provenance["country"] = good
    assert r.provenance["country"] == "inferred_from_paper"


def test_provenance_rejects_a_key_that_is_not_a_schema_field():
    # a typo'd key would otherwise surface as a field mysteriously lacking
    # provenance, which is invisible rather than loud
    r = TargetSchema(id="SRX1")
    for bad in ("contry", "provenance", "runs"):
        try:
            r.provenance[bad] = "direct"
        except ValueError as exc:
            assert bad in str(exc)
        else:
            raise AssertionError(f"provenance key {bad!r} should be rejected")


def test_provenance_is_validated_however_it_is_written():
    # item assignment, whole-dict assignment, update() and setdefault() all go
    # through the same check — validating only __setattr__ would miss the first
    r = TargetSchema(id="SRX1")
    try:
        r.provenance.update({"country": "nonsense"})
    except ValueError:
        pass
    else:
        raise AssertionError("update() must validate")
    try:
        r.provenance.setdefault("country", "nonsense")
    except ValueError:
        pass
    else:
        raise AssertionError("setdefault() must validate")
    try:
        r.provenance = {"country": "nonsense"}
    except ValueError:
        pass
    else:
        raise AssertionError("whole-dict assignment must validate")
    # and a bad class in a saved file is caught on load
    try:
        TargetSchema.from_dict({"id": "SRX1", "provenance": {"country": "nonsense"}})
    except ValueError:
        pass
    else:
        raise AssertionError("from_dict must validate provenance")


def test_provenance_still_behaves_like_a_dict():
    r = TargetSchema(id="SRX1")
    r.provenance["country"] = "harmonized"
    assert r.provenance == {"country": "harmonized"}  # equality with a plain dict
    assert dict(r.provenance) == {"country": "harmonized"}
    assert json.loads(json.dumps(r.to_dict()))["provenance"] == {"country": "harmonized"}
    assert TargetSchema.from_dict(r.to_dict()) == r


def test_confidence_accepts_only_the_declared_levels():
    r = TargetSchema(id="SRX1")
    for good in schema.CONFIDENCE_LEVELS:
        r.confidence["sex"] = good
    for bad in ("HIGH", "very high", 0.9, 1, "", None):
        try:
            r.confidence["sex"] = bad
        except ValueError as exc:
            assert "sex" in str(exc)
        else:
            raise AssertionError(f"confidence {bad!r} should be rejected")
    # same key check as provenance, and the same coverage on every write path
    for write in (
        lambda: r.confidence.__setitem__("contry", "high"),
        lambda: r.confidence.update({"sex": "nope"}),
        lambda: r.confidence.setdefault("strain", "nope"),  # a key not already set
        lambda: setattr(r, "confidence", {"sex": "nope"}),
        lambda: TargetSchema.from_dict({"id": "X", "confidence": {"sex": "nope"}}),
    ):
        try:
            write()
        except ValueError:
            pass
        else:
            raise AssertionError("every confidence write path must validate")


def test_not_applicable_is_an_answer_that_carries_confidence():
    # confidence is in the answer, and "this cannot apply here" is one: sex on a
    # soil metagenome is confidently not applicable, not confidently missing
    r = TargetSchema(id="SRX1", isolation_source="soil")
    r.sex = schema.NOT_APPLICABLE
    r.provenance["sex"] = "inferred_from_text"
    r.confidence["sex"] = "high"
    assert r.sex == "not applicable"
    assert r.inconsistent_confidence() == []
    assert r.declared_missing() == {"sex": "not applicable"}
    assert "sex" in r.filled()  # resolved, so it counts as progress


def test_not_applicable_is_normalized_and_survives_a_round_trip():
    r = TargetSchema(id="SRX1", host="Not Applicable")  # a model will vary case
    assert r.host == schema.NOT_APPLICABLE
    r.tissue_type = "  NOT APPLICABLE  "
    assert r.tissue_type == schema.NOT_APPLICABLE
    back = TargetSchema.from_dict(json.loads(json.dumps(r.to_dict())))
    assert back == r
    assert back.declared_missing() == {"host": schema.NOT_APPLICABLE,
                                       "tissue_type": schema.NOT_APPLICABLE}
    assert r.to_dict()["host"] == schema.NOT_APPLICABLE  # a real value, not a null


def test_every_insdc_missing_value_is_accepted_and_normalized():
    r = TargetSchema(id="SRX1")
    assert schema.MISSING_VALUES == ("not applicable", "not collected",
                                     "not provided", "restricted access",
                                     "missing")
    for term in schema.MISSING_VALUES:
        r.host = term.upper()
        assert r.host == term, term
    # each is an answer, so each can carry provenance and confidence
    r.host = schema.NOT_COLLECTED
    r.provenance["host"] = "inferred_from_text"
    r.confidence["host"] = "medium"
    assert r.inconsistent_confidence() == []
    assert r.declared_missing() == {"host": "not collected"}


def test_missing_value_aliases_are_left_as_ordinary_text():
    # "NA" is a plausible strain, region or country code; reinterpreting it as a
    # missing-value declaration would destroy real data
    r = TargetSchema(id="SRX1")
    # `missing` was on this list and is not any more: it is an INSDC vocabulary
    # term, not a submitter's own shorthand, and unlike "NA" there is no
    # plausible strain, region or country called "missing". It was the single
    # commonest value layer 2 stored — 20,804 across the corpus — recorded as
    # though a sample's country were named "missing".
    for alias in ("N/A", "NA", "none", "unknown", "not reported"):
        r.strain = alias
        assert r.strain == alias, alias
        assert r.declared_missing() == {}


def test_bare_missing_is_a_declared_absence_not_a_value():
    # The commonest term in the corpus by a wide margin: 20,804 harmonized
    # values, against 15,403 for the four reasoned terms combined. Storing it as
    # ordinary text asserted that a sample's country was called "missing".
    r = TargetSchema(id="SRX1")
    r.country = "Missing"
    assert r.country == "missing"                      # canonical case
    assert schema._is_missing_value(r.country)
    assert r.declared_missing() == {"country": "missing"}


def test_a_harmonized_missing_term_is_never_reopened():
    # The GenomeTrakr rule. A term layer 2 recorded is the submitter's own
    # declaration, so no later layer may infer over it however confident it
    # sounds — 369 of 380 countries were wrong the last time that happened.
    # Adding `missing` to the vocabulary must not put those records back at risk.
    r = TargetSchema(id="SRX1")
    r.country = "missing"
    r.provenance["country"] = "harmonized"
    assert "country" not in R.open_fields(r, overwrite_missing=True)


def test_a_model_written_missing_term_stays_revisable():
    # The other half of the same rule: a model's verdict is not the submitter's,
    # and the paper layer is the thing most likely to overturn it.
    r = TargetSchema(id="SRX1")
    r.country = "missing"
    r.provenance["country"] = "inferred_from_text"
    assert "country" in R.open_fields(r, overwrite_missing=True)
    assert "country" not in R.open_fields(r, overwrite_missing=False)


def test_missing_cannot_be_stored_in_a_typed_field():
    # Unchanged, and the reason the Rust port needed a different shape: a typed
    # column has no room for a sentinel, so `collection_date: missing` is still
    # dropped rather than recorded. Rust's Field<T> carries the reason instead.
    r = TargetSchema(id="SRX1")
    with pytest.raises(ValueError, match="cannot be stored"):
        r.collection_date = "missing"
    assert R._storable("collection_date", "missing") is False


def test_none_is_not_a_missing_value_term():
    # None means nothing was concluded — a statement about this pipeline, not
    # about the archive. Defaulting untouched fields to a term would report full
    # coverage on a record no model has looked at.
    r = TargetSchema(id="SRX1")
    assert r.declared_missing() == {}
    assert len(r.missing()) == len(_SCHEMA_FIELDS) - 1
    assert r.coverage() == 1 / len(_SCHEMA_FIELDS)


def test_not_applicable_is_refused_by_counts_and_dates():
    # a run always has a read count; storing a string there would break every
    # comparison downstream, so this is a caller error, not a representation gap
    for field_name in ("read_count", "base_count", "tax_id", "collection_date"):
        for term in schema.MISSING_VALUES:
            try:
                setattr(TargetSchema(id="SRX1"), field_name, term)
            except ValueError as exc:
                assert field_name in str(exc) and "None" in str(exc)
            else:
                raise AssertionError(f"{field_name} should refuse {term!r}")


def test_inconsistent_confidence_flags_unjustified_levels():
    r = TargetSchema(id="SRX1", country="Brazil", strain="K12", host="mouse")
    r.provenance["country"] = "direct"       # nobody chose -> confidence is moot
    r.confidence["country"] = "high"
    r.confidence["strain"] = "low"           # no provenance at all
    r.provenance["host"] = "inferred_from_paper"
    r.confidence["host"] = "medium"          # legitimate
    assert sorted(r.inconsistent_confidence()) == ["country", "strain"]


def test_confidence_needs_an_actual_answer():
    # None means not-reported or not-yet-attempted; neither is a conclusion, so
    # neither earns a confidence. NOT_APPLICABLE is the way to say "I decided".
    r = TargetSchema(id="SRX1")
    r.provenance["sex"] = "inferred_from_text"
    r.confidence["sex"] = "high"
    assert r.sex is None
    assert r.inconsistent_confidence() == ["sex"]
    r.sex = schema.NOT_APPLICABLE
    assert r.inconsistent_confidence() == []


def test_confidence_round_trips_and_stays_out_of_the_index():
    r = TargetSchema(id="SRX1")
    r.provenance["host"] = "inferred_from_text"
    r.confidence["host"] = "low"
    back = TargetSchema.from_dict(json.loads(json.dumps(r.to_dict())))
    assert back == r and back.confidence == {"host": "low"}
    assert "confidence" not in TargetSchema.field_names()
    assert "confidence" not in TargetSchema.field_names()


def test_from_project_records_no_confidence():
    # everything from_project sets is `direct` — no model chose anything
    (r,) = TargetSchema.from_project(_study())
    assert r.confidence == {}
    assert r.inconsistent_confidence() == []


def test_from_project_can_skip_provenance():
    (r,) = TargetSchema.from_project(_study(), include_provenance=False)
    assert r.provenance == {} and r.study_title == "study title"


# --------------------------------------------------------------------------- #
# dataset.save_reconstructed_records (stubbed expansion, no network)
# --------------------------------------------------------------------------- #
_FULL_STUDY = {
    "accession": "SRP1",
    "title": "study title",
    "record_count": 2,
    "samples": {"SRS1": {"accession": "SRS1", "scientific_name": "Homo sapiens"}},
    "experiments": [
        {
            "accession": "SRX1",
            "sample_ids": ["SRS1"],
            "library_strategy": "RNA-Seq",
            "runs": [{"accession": "SRR1"}, {"accession": "SRR2"}],
        }
    ],
}


def _summaries_file(tmp, *studies):
    """Write a stage-1/2 style JSON array of study dicts and return its path."""
    path = os.path.join(tmp, "studies.json")
    with open(path, "w", encoding="utf-8") as file:
        json.dump(list(studies) or [{"accession": "SRP1", "title": "study title",
                                     "record_count": 2}], file)
    return path


class _StubExpansion:
    """Replace dataset's ``Project`` with a recorded stub. Records the calls."""

    def __init__(self, result=None, error=None):
        self.result, self.error, self.calls = result, error, []

    def __enter__(self):
        def fake(srp, **kw):
            self.calls.append((srp, kw))
            if self.error is not None:
                raise self.error
            return Project.from_dict(self.result)

        self._saved = D.Project
        D.Project = fake
        return self

    def __exit__(self, *exc):
        D.Project = self._saved
        return False


def test_reconstruct_without_expanding_touches_no_network():
    with tempfile.TemporaryDirectory() as tmp:
        in_path = _summaries_file(tmp)
        out_path = os.path.join(tmp, "records.json")
        with _NoNetwork():
            D.save_reconstructed_records(in_path, out_path, expand=False)
        # a summary carries study-level fields only -> one stub record
        (r,) = D.load_records(out_path)
        assert r.id == "SRP1" and r.study_title == "study title"
        assert r.run_accession is None and r.library_strategy is None


def test_reconstruct_expands_each_study_to_full_depth():
    with tempfile.TemporaryDirectory() as tmp:
        in_path, out_path = _summaries_file(tmp), os.path.join(tmp, "records.json")
        with _StubExpansion(result=_FULL_STUDY) as stub:
            D.save_reconstructed_records(in_path, out_path)
        assert [acc for acc, _ in stub.calls] == ["SRP1"]
        # the schema has no publication field, so that BioProject fetch is waste
        assert stub.calls[0][1]["include_publications"] is False
        records = D.load_records(out_path)
        # SRX1 owns both runs, so the two collapse into one record
        assert [r.id for r in records] == ["SRX1"]
        assert [run["run_accession"] for run in records[0].runs] == ["SRR1", "SRR2"]
        assert records[0].library_strategy == "RNA-Seq"
        assert records[0].scientific_name == "Homo sapiens"


def _cost_study(accession, records, oa=True):
    return Project.from_dict({
        "accession": accession, "record_count": records,
        "publications": [{"id": "1", "type": "ePubmed",
                          "accessibility_type": "oa" if oa else "paywall"}],
    })


def test_cost_estimate_matches_the_runs_it_was_calibrated_on():
    # 52 records over 5 studies actually cost $0.18 (layer 3) and $0.066 (layer 4)
    small = [_cost_study(f"SRP{i}", 10) for i in range(5)] + [_cost_study("SRP9", 2)]
    text, _ = D.estimate_reconstruction_cost(small, from_text=True)
    assert 0.15 < text < 0.22, text
    paper, _ = D.estimate_reconstruction_cost(small, from_paper=True)
    assert 0.05 < paper < 0.10, paper
    # ...and the 1,782-record run that cost ~$6.2
    big = [_cost_study("SRP070120", 1664), _cost_study("SRP293759", 118)]
    total, report = D.estimate_reconstruction_cost(big, from_text=True, from_paper=True)
    assert 5.5 < total < 7.0, total
    assert "1,782 records" in report


def test_cost_estimate_is_zero_without_model_layers():
    studies = [_cost_study("SRP1", 5000)]
    total, report = D.estimate_reconstruction_cost(studies, harmonize=True)
    assert total == 0.0 and "no model layers" in report


def test_cost_estimate_counts_oversized_studies_as_one_stub():
    # _expand abandons them and emits a single summary record; the estimate has
    # to agree or a PulseNet-sized umbrella would dominate a figure it never costs
    studies = [_cost_study("SRP1", 500_000)]
    total, report = D.estimate_reconstruction_cost(studies, from_text=True,
                                                   max_records=D.MAX_RECORDS)
    assert total == pytest.approx(D.COST_PER_RECORD_TEXT)
    assert "oversized" in report


def test_cost_estimate_charges_layer_four_only_for_open_access_studies():
    studies = [_cost_study("SRP1", 10, oa=True), _cost_study("SRP2", 10, oa=False)]
    total, _ = D.estimate_reconstruction_cost(studies, from_paper=True)
    assert total == pytest.approx(D.COST_PER_STUDY_PAPER)   # one study, not two


def test_spend_limit_blocks_before_any_paid_work():
    # the $7 run: two open-access studies, 1,782 records, nothing to warn you
    with tempfile.TemporaryDirectory() as tmp:
        in_path = _summaries_file(tmp, {"accession": "SRP070120", "record_count": 1664,
                                        "publications": [{"id": "1", "type": "ePubmed",
                                                          "accessibility_type": "oa"}]})
        out_path = os.path.join(tmp, "records.json")
        with _StubExpansion(result=_FULL_STUDY) as stub:
            try:
                D.save_reconstructed_records(in_path, out_path, from_text=True,
                                             from_paper=True)
            except D.SpendLimitExceeded as exc:
                assert "nothing has been spent" in str(exc)
                assert "max_spend=" in str(exc)          # tells you how to proceed
            else:
                raise AssertionError("a $5.8 run must not start under a $1 cap")
        assert stub.calls == []                          # no study was even expanded
        assert not os.path.exists(out_path)


def test_spend_limit_lets_an_affordable_run_through():
    with tempfile.TemporaryDirectory() as tmp:
        in_path, out_path = _summaries_file(tmp), os.path.join(tmp, "records.json")
        with _StubExpansion(result=_FULL_STUDY):
            D.save_reconstructed_records(in_path, out_path)   # free: no model layers
        assert os.path.exists(out_path)


def test_spend_limit_can_be_raised_or_disabled():
    studies = [_cost_study("SRP1", 1664)]
    total, _ = D.estimate_reconstruction_cost(studies, from_text=True)
    with tempfile.TemporaryDirectory() as tmp:
        in_path = _summaries_file(tmp, {"accession": "SRP1", "record_count": 1664})
        for limit in (total + 1, None):
            out_path = os.path.join(tmp, f"r{limit}.json")
            layer = _FakeLayer({})
            saved = _with_layer("from_text", "inferred_from_text", layer)
            try:
                with _StubExpansion(result=_FULL_STUDY):
                    D.save_reconstructed_records(in_path, out_path, from_text=True,
                                                 max_spend=limit)
            finally:
                R.LAYERS = saved
            assert os.path.exists(out_path), limit


def _study_file(tmp, name, studies):
    path = os.path.join(tmp, name)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(studies, file)
    return path


def test_combine_studies_merges_and_sorts():
    with tempfile.TemporaryDirectory() as tmp:
        a = _study_file(tmp, "a.json", [{"accession": "SRP2", "record_count": 5},
                                        {"accession": "SRP1", "record_count": 3}])
        b = _study_file(tmp, "b.json", [{"accession": "SRP3", "record_count": 7}])
        out = os.path.join(tmp, "corpus.json")
        merged = D.combine_studies([a, b], out_path=out)
        assert [s.accession for s in merged] == ["SRP1", "SRP2", "SRP3"]   # sorted
        assert [s["accession"] for s in json.load(open(out))] == ["SRP1", "SRP2", "SRP3"]


def test_combine_studies_dedupes_and_keeps_the_richer_copy():
    # harvests are random samples over one archive, so overlap is expected. The
    # copy carrying classified publications must win — those classes cost up to
    # three third-party lookups each and would otherwise be paid for twice.
    thin = {"accession": "SRP1", "record_count": 5,
            "publications": [{"id": "1", "type": "ePubmed", "accessibility_type": None}]}
    rich = {"accession": "SRP1", "record_count": 5,
            "publications": [{"id": "1", "type": "ePubmed", "accessibility_type": "oa"}]}
    for order, label in (([thin], [rich]), ([rich], [thin])):
        with tempfile.TemporaryDirectory() as tmp:
            a = _study_file(tmp, "a.json", order)
            b = _study_file(tmp, "b.json", label)
            (merged,) = D.combine_studies([a, b])
            assert merged.publications[0].accessibility_type == "oa"


def test_combine_studies_output_feeds_straight_back_in():
    # the whole point: skip stages 1-2 on a reuse. The file must load as studies
    # and be safe to re-combine without changing.
    with tempfile.TemporaryDirectory() as tmp:
        src = _study_file(tmp, "a.json", [
            {"accession": "SRP1", "record_count": 3,
             "publications": [{"id": "9", "type": "ePubmed", "accessibility_type": "oa"}]}])
        out = os.path.join(tmp, "corpus.json")
        D.combine_studies([src], out_path=out)
        reloaded = P.load_studies(out)
        assert reloaded[0].accession == "SRP1"
        assert reloaded[0].publications[0].accessibility_type == "oa"
        again = os.path.join(tmp, "corpus2.json")
        D.combine_studies([out], out_path=again)
        assert json.load(open(out)) == json.load(open(again))   # idempotent


def test_reconstruct_passes_the_layer_flags_through_to_the_cascade():
    # regression: save_reconstructed_records called TargetSchema.from_project
    # directly, so every record came out `direct` no matter which layers were on
    layer = _FakeLayer({"SRX1": {"host": R.Proposal("mouse", "high")}})
    saved_layers = _with_layer("from_text", "inferred_from_text", layer)
    with tempfile.TemporaryDirectory() as tmp:
        in_path, out_path = _summaries_file(tmp), os.path.join(tmp, "records.json")
        try:
            with _StubExpansion(result=_FULL_STUDY):
                D.save_reconstructed_records(in_path, out_path, from_text=True)
        finally:
            R.LAYERS = saved_layers
        (r,) = D.load_records(out_path)
        assert r.provenance["host"] == "inferred_from_text"
        assert r.confidence["host"] == "high"
        assert r.provenance["library_strategy"] == "direct"


def test_reconstruct_defaults_to_direct_only_so_a_run_cannot_bill_by_accident():
    # the model layers cost money per sample; turning one on is the caller's
    # decision, so the default has to be off
    with tempfile.TemporaryDirectory() as tmp:
        in_path, out_path = _summaries_file(tmp), os.path.join(tmp, "records.json")
        with _StubExpansion(result=_FULL_STUDY):
            D.save_reconstructed_records(in_path, out_path)   # no layer flags
        (r,) = D.load_records(out_path)
        assert set(r.provenance.values()) == {"direct"}
        assert r.confidence == {}


def test_a_failing_model_layer_costs_the_study_its_inference_not_the_run():
    def explode(project, records, open_by_id):
        raise RuntimeError("simulated refusal")

    saved_layers = _with_layer("from_text", "inferred_from_text", explode)
    with tempfile.TemporaryDirectory() as tmp:
        in_path, out_path = _summaries_file(tmp), os.path.join(tmp, "records.json")
        try:
            with _StubExpansion(result=_FULL_STUDY):
                D.save_reconstructed_records(in_path, out_path, from_text=True)
        finally:
            R.LAYERS = saved_layers
        (r,) = D.load_records(out_path)          # study survives, direct-only
        assert set(r.provenance.values()) == {"direct"}


def test_expansion_carries_publications_onto_the_expanded_study():
    # regression: _expand rebuilds the study with include_publications=False to
    # save a BioProject fetch, so the expanded object had none — and the paper
    # layer, which looks for an `oa` publication there, silently did nothing
    summary = Project.from_dict({
        "accession": "SRP1",
        "publications": [{"id": "9", "type": "ePubmed", "accessibility_type": "oa"}],
    })
    with _StubExpansion(result=_FULL_STUDY):
        expanded, note = D._expand(summary, D.MAX_RECORDS)
    assert note == ""
    assert [(p.id, p.accessibility_type) for p in expanded.publications] == [("9", "oa")]


def test_reconstruct_falls_back_to_the_summary_when_expansion_fails():
    # one flaky study must cost depth, not the study itself
    with tempfile.TemporaryDirectory() as tmp:
        in_path, out_path = _summaries_file(tmp), os.path.join(tmp, "records.json")
        with _StubExpansion(error=RuntimeError("simulated NCBI outage")):
            D.save_reconstructed_records(in_path, out_path)
        (r,) = D.load_records(out_path)
        assert r.id == "SRP1" and r.study_title == "study title"


def test_reconstruct_skips_oversized_studies_before_fetching_them():
    # expanding is one request per 300 records, so the guard must precede it
    with tempfile.TemporaryDirectory() as tmp:
        in_path = _summaries_file(tmp, {"accession": "SRP1", "record_count": 99})
        out_path = os.path.join(tmp, "records.json")
        with _StubExpansion(result=_FULL_STUDY) as stub:
            D.save_reconstructed_records(in_path, out_path, max_records=10)
        assert stub.calls == []  # never fetched
        (r,) = D.load_records(out_path)
        assert r.id == "SRP1"  # kept as a summary rather than dropped


def test_reconstruct_resumes_from_its_checkpoint():
    with tempfile.TemporaryDirectory() as tmp:
        in_path = _summaries_file(
            tmp, {"accession": "SRP1"}, {"accession": "SRP2"}, {"accession": "SRP3"}
        )
        out_path = os.path.join(tmp, "records.json")
        ckpt = out_path + ".partial"

        # Ctrl-C after the first study: BaseException is not swallowed by the
        # per-study fallback, so the run stops with a checkpoint on disk
        calls = []

        def flaky(study, max_records):
            calls.append(study.accession)
            if len(calls) > 1:
                raise KeyboardInterrupt
            return study, ""

        saved = D._expand
        try:
            D._expand = flaky
            try:
                D.save_reconstructed_records(in_path, out_path, checkpoint_every=1)
            except KeyboardInterrupt:
                pass
            assert os.path.exists(ckpt) and not os.path.exists(out_path)

            # resume with a working stub — restoring the real _expand here would
            # send this offline test to NCBI
            D._expand = lambda study, max_records: (study, "")
            D.save_reconstructed_records(in_path, out_path, checkpoint_every=1)
        finally:
            D._expand = saved

        assert calls == ["SRP1", "SRP2"]  # SRP1 was not re-done on resume

        assert [r.id for r in D.load_records(out_path)] == ["SRP1", "SRP2", "SRP3"]
        assert not os.path.exists(ckpt)  # removed only once the output landed


def test_reconstruct_ignores_a_checkpoint_from_different_arguments():
    with tempfile.TemporaryDirectory() as tmp:
        in_path = _summaries_file(tmp, {"accession": "SRP1"}, {"accession": "SRP2"})
        out_path = os.path.join(tmp, "records.json")
        with open(out_path + ".partial", "w", encoding="utf-8") as file:
            json.dump(
                {
                    "version": D._CHECKPOINT_VERSION,
                    "params": {"in_path": in_path, "expand": True,
                               "max_records": 5000, "study_count": 999},
                    "done": {"SRP1": [{"id": "stale"}]},
                },
                file,
            )
        with _NoNetwork():
            D.save_reconstructed_records(in_path, out_path, expand=False)
        assert [r.id for r in D.load_records(out_path)] == ["SRP1", "SRP2"]


def test_load_records_accepts_a_path_a_string_or_a_parsed_list():
    rows = [TargetSchema(id="SRR1", read_count=5).to_dict()]
    assert D.load_records(rows)[0].read_count == 5
    assert D.load_records(json.dumps(rows))[0].id == "SRR1"
    assert D.load_records(rows[0])[0].id == "SRR1"  # a lone record dict
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "r.json")
        with open(path, "w", encoding="utf-8") as file:
            json.dump(rows, file)
        assert D.load_records(path)[0].id == "SRR1"


# --------------------------------------------------------------------------- #
# reconstruct.py — the layer cascade (layers stubbed, no network, no model)
# --------------------------------------------------------------------------- #
class _FakeLayer:
    """Stand in for an unbuilt layer, recording what the cascade handed it."""

    def __init__(self, proposals):
        self.proposals, self.seen_open = proposals, None

    def __call__(self, project, records, open_by_id):
        self.seen_open = {k: set(v) for k, v in open_by_id.items()}
        return self.proposals


def _with_layer(flag, provenance_class, fn):
    """Swap one layer into LAYERS for the duration of a call."""
    saved = R.LAYERS
    R.LAYERS = tuple(
        (f, c, fn if f == flag else old) for f, c, old in saved
    )
    return saved


def test_reconstruct_runs_the_direct_layer_and_reports_it():
    records, report = R.reconstruct(_study())
    assert [r.id for r in records] == ["SRX1"]
    assert report == {"direct": len(records[0].provenance)}
    assert all(v == "direct" for v in records[0].provenance.values())


def _paper_study(**over):
    """A study with an OA publication and two experiments on two samples."""
    d = dict(
        accession="SRP1", bioproject="PRJNA1", title="study title",
        abstract="study abstract",
        publications=[{"id": "111", "type": "ePubmed", "accessibility_type": "paywall"},
                      {"id": "222", "type": "ePubmed", "accessibility_type": "oa"},
                      {"id": "333", "type": "ePubmed", "accessibility_type": "oa"}],
        samples={"SRS1": {"accession": "SRS1", "scientific_name": "Mus musculus"},
                 "SRS2": {"accession": "SRS2", "scientific_name": "Mus musculus"}},
        experiments=[{"accession": "SRX1", "sample_ids": ["SRS1"]},
                     {"accession": "SRX2", "sample_ids": ["SRS2"]}],
    )
    d.update(over)
    return Project.from_dict(d)


class _PaperStub:
    """Replace the fetch and the model call; record how often each ran."""

    def __init__(self, text="TITLE: A paper", answers=()):
        self.text, self.answers = text, list(answers)
        self.fetches, self.calls, self.prompts = 0, 0, []

    def fetch(self, pub_id, max_chars=None, **kw):
        self.fetches += 1
        self.fetched_id, self.max_chars = pub_id, max_chars
        return self.text

    def extract(self, prompt, schema, **kw):
        self.calls += 1
        self.prompts.append(prompt)
        self.asked = set(schema["properties"]["answers"]["items"]["properties"]["field"]["enum"])
        return {"answers": self.answers}


def _with_paper(stub):
    saved = R.project_module.fetch_open_access_text, C.extract
    R.project_module.fetch_open_access_text = stub.fetch
    C.extract = stub.extract
    return saved


def test_paper_layer_reads_one_paper_once_per_study():
    # reading the same paper once per sample is the most expensive mistake
    # available here: a 30,000-char paper resent for every sample in the study
    stub = _PaperStub(answers=[{"field": "treatment", "value": "LPS", "confidence": "medium"}])
    saved = _with_paper(stub)
    try:
        project = _paper_study()
        records = TargetSchema.from_project(project)
        proposals = R.infer_from_paper(project, records,
                                       {r.id: R.open_fields(r) for r in records})
    finally:
        R.project_module.fetch_open_access_text, C.extract = saved

    assert stub.fetches == 1 and stub.calls == 1          # two samples, one read
    assert stub.fetched_id == "222"                       # the first `oa` one, not "111"
    assert stub.max_chars == R.PAPER_MAX_CHARS
    # and the one answer is copied to every record beneath the study
    assert sorted(proposals) == ["SRX1", "SRX2"]
    assert all(f["treatment"] == R.Proposal("LPS", "medium") for f in proposals.values())


def test_paper_layer_does_nothing_without_an_open_access_paper():
    # no fetch, no call — a study without a readable paper is a normal outcome
    for pubs in ([], [{"id": "1", "type": "ePubmed", "accessibility_type": "paywall"}]):
        stub = _PaperStub()
        saved = _with_paper(stub)
        try:
            project = _paper_study(publications=pubs)
            records = TargetSchema.from_project(project)
            assert R.infer_from_paper(project, records,
                                      {r.id: R.open_fields(r) for r in records}) == {}
        finally:
            R.project_module.fetch_open_access_text, C.extract = saved
        assert stub.fetches == 0 and stub.calls == 0


def test_paper_layer_skips_the_fetch_when_the_text_is_unavailable():
    stub = _PaperStub(text=None)
    saved = _with_paper(stub)
    try:
        project = _paper_study()
        records = TargetSchema.from_project(project)
        assert R.infer_from_paper(project, records,
                                  {r.id: R.open_fields(r) for r in records}) == {}
    finally:
        R.project_module.fetch_open_access_text, C.extract = saved
    assert stub.fetches == 1 and stub.calls == 0   # fetched, found nothing, stopped


def test_paper_layer_never_overrides_an_earlier_layer():
    # including a missing-value term: `not applicable` from layer 3 came from the
    # sample's own attributes, which beat a paper describing the whole cohort
    stub = _PaperStub(answers=[
        {"field": "tissue_type", "value": "kidney", "confidence": "high"},
        {"field": "host", "value": "Mus musculus", "confidence": "high"},
        {"field": "treatment", "value": "LPS", "confidence": "high"},
    ])
    saved = _with_paper(stub)
    try:
        project = _paper_study()
        records = TargetSchema.from_project(project)
        records[0].tissue_type = "liver"                 # an ordinary value
        records[0].host = schema.NOT_APPLICABLE          # a declared verdict...
        records[0].provenance["host"] = "inferred_from_text"   # ...by a model,
        open_by_id = {r.id: R.open_fields(r) for r in records}  # so it reopens
        proposals = R.infer_from_paper(project, records, open_by_id)
    finally:
        R.project_module.fetch_open_access_text, C.extract = saved

    # the two are protected by different mechanisms, and both must hold:
    #   tissue_type holds an ordinary value -> open_fields already excludes it
    #   host holds a declared verdict       -> open_fields REOPENS it, and the
    #                                          cascade would have allowed a write
    assert "tissue_type" not in open_by_id["SRX1"]
    assert "host" in open_by_id["SRX1"]
    # SRX1 keeps what earlier layers put there; only the genuinely empty field lands
    assert set(proposals["SRX1"]) == {"treatment"}
    assert records[0].tissue_type == "liver" and records[0].host == schema.NOT_APPLICABLE
    # SRX2 had all three empty, so all three are proposed to it
    assert set(proposals["SRX2"]) == {"treatment", "tissue_type", "host"}


def test_paper_layer_does_not_ask_about_fields_no_record_needs():
    # asked once for the whole study, so a field every record already has is
    # dropped from the schema entirely rather than filtered after the fact
    stub = _PaperStub()
    saved = _with_paper(stub)
    try:
        project = _paper_study()
        records = TargetSchema.from_project(project)
        for record in records:
            record.tissue_type = "liver"
        R.infer_from_paper(project, records, {r.id: R.open_fields(r) for r in records})
    finally:
        R.project_module.fetch_open_access_text, C.extract = saved
    assert "tissue_type" not in stub.asked
    assert "treatment" in stub.asked


def test_paper_layer_never_asks_about_archive_bookkeeping():
    # a manuscript does not state run accessions, upload formats, the submitting
    # centre, or submitter-chosen aliases. TEXT_SYSTEM already says so in prose
    # and the model answered them anyway — 157 of 262 filled fields on the
    # measured run, every one "not provided" — so they are excluded in code.
    stub = _PaperStub()
    saved = _with_paper(stub)
    try:
        project = _paper_study()
        records = TargetSchema.from_project(project)
        R.infer_from_paper(project, records, {r.id: R.open_fields(r) for r in records})
    finally:
        R.project_module.fetch_open_access_text, C.extract = saved

    for blind in ("run_accession", "submitted_format", "broker_name", "center_name",
                  "datahub", "checklist", "sample_alias", "library_name"):
        assert blind not in stub.asked, blind
    # ...while everything a paper genuinely can speak to is still asked
    for reachable in ("treatment", "tissue_type", "dev_stage", "age",
                      "library_construction_protocol", "country"):
        assert reachable in stub.asked, reachable


def test_text_layer_never_asks_about_archive_bookkeeping():
    # layer 3 had no guard at all and filled 4,342 of these across the measured
    # runs: `PRJNA293224` as a submission_accession on 146 records (a BioProject
    # accession, the wrong identifier entirely), `CFSAN100605` as a run
    # accession, and "not provided" thousands of times at full token price.
    asked = []

    def stub(prompt, schema_, system=None, **kw):
        asked.extend(
            schema_["properties"]["answers"]["items"]["properties"]["field"]["enum"]
        )
        return {"answers": []}

    saved = C.extract, R.claude.extract
    C.extract = R.claude.extract = stub
    try:
        project = P.Project.from_dict({
            "accession": "SRP1", "title": "t", "abstract": "a",
            "samples": {"SRS1": {"accession": "SRS1",
                                 "attributes": {"tissue": "liver"}}},
            "experiments": [{"accession": "SRX1", "sample_ids": ["SRS1"], "runs": []}],
        })
        records = TargetSchema.from_project(project)
        R.infer_from_text(project, records,
                          {r.id: R.open_fields(r) for r in records})
    finally:
        C.extract, R.claude.extract = saved

    for blind in ("run_accession", "run_alias", "submitted_format",
                  "submitted_read_type", "submission_accession", "broker_name",
                  "center_name", "datahub", "tag", "read_count", "base_count"):
        assert blind not in asked, blind
    # ...while the sample biology this layer exists for is still asked
    for reachable in ("tissue_type", "treatment", "dev_stage", "age", "sex",
                      "isolation_source"):
        assert reachable in asked, reachable


def test_text_blind_set_is_built_from_the_schema_not_hand_listed():
    from schema import RECORD, RUN, SUBMISSION
    assert set(TargetSchema.fields_at_level(RUN)) <= R._TEXT_BLIND
    assert set(TargetSchema.fields_at_level(SUBMISSION)) <= R._TEXT_BLIND
    assert set(TargetSchema.fields_at_level(RECORD)) <= R._TEXT_BLIND
    # the sample- and experiment-level identifiers a paper cannot state but an
    # attribute bag genuinely can — layer 3 reads that bag, so it still asks
    for reachable in ("checklist", "ncbi_reporting_standard", "sample_alias",
                      "experiment_alias", "tissue_type"):
        assert reachable not in R._TEXT_BLIND, reachable


def test_paper_blind_set_is_built_from_the_schema_not_hand_listed():
    # levels move as fields are added; deriving the set means a new run- or
    # submission-level field is excluded automatically
    from schema import RUN, SUBMISSION
    assert set(TargetSchema.fields_at_level(RUN)) <= R._PAPER_BLIND
    assert set(TargetSchema.fields_at_level(SUBMISSION)) <= R._PAPER_BLIND
    assert "tissue_type" not in R._PAPER_BLIND


def test_paper_layer_passes_established_fields_as_context():
    stub = _PaperStub()
    saved = _with_paper(stub)
    try:
        project = _paper_study()
        records = TargetSchema.from_project(project)
        R.infer_from_paper(project, records, {r.id: R.open_fields(r) for r in records})
    finally:
        R.project_module.fetch_open_access_text, C.extract = saved
    prompt = stub.prompts[0]
    assert "ALREADY ESTABLISHED" in prompt
    assert "Mus musculus" in prompt          # scientific_name, filled by direct
    assert "study abstract" in prompt        # description, filled by direct


def test_a_layer_only_sees_fields_the_earlier_layers_left_open():
    record_id = "SRX1"
    layer = _FakeLayer({record_id: {"host": R.Proposal("mouse", "medium")}})
    saved = _with_layer("from_text", "inferred_from_text", layer)
    try:
        records, report = R.reconstruct(_study(), from_text=True)
    finally:
        R.LAYERS = saved

    open_to_it = layer.seen_open[record_id]
    assert "host" in open_to_it and "id" not in open_to_it
    # direct already filled these, so they are closed
    assert "library_strategy" not in open_to_it
    assert "experiment_accession" not in open_to_it

    (r,) = records
    assert r.host == "mouse"
    assert r.provenance["host"] == "inferred_from_text"  # stamped by the cascade
    assert r.confidence["host"] == "medium"
    assert r.provenance["library_strategy"] == "direct"  # untouched
    by_class = collections.Counter(r.provenance.values())
    assert report == {"direct": by_class["direct"], "inferred_from_text": 1}
    assert r.inconsistent_confidence() == []


def test_a_layer_reaching_outside_its_open_set_is_an_error():
    # overwriting an earlier, cheaper answer must not pass silently
    layer = _FakeLayer({"SRX1": {"library_strategy": R.Proposal("WGS")}})
    saved = _with_layer("from_text", "inferred_from_text", layer)
    try:
        R.reconstruct(_study(), from_text=True)
    except ValueError as exc:
        assert "library_strategy" in str(exc) and "not open" in str(exc)
    else:
        raise AssertionError("a closed field should be refused")
    finally:
        R.LAYERS = saved


def test_only_the_inferred_layers_may_set_a_confidence():
    layer = _FakeLayer({"SRX1": {"host": R.Proposal("mouse", "high")}})
    saved = _with_layer("harmonize", "harmonized", layer)
    try:
        R.reconstruct(_study(), harmonize=True)
    except ValueError as exc:
        assert "confidence" in str(exc) and "harmonized" in str(exc)
    else:
        raise AssertionError("a harmonized value cannot carry a confidence")
    finally:
        R.LAYERS = saved


def test_open_fields_reopens_a_missing_verdict_only_for_the_model_that_made_it():
    # who declared it decides. A model's "not collected" means "not collected as
    # far as I could see" and the paper layer should get a chance at it; the
    # submitter's own "not provided" is data about the record, not a gap.
    def record(field, value, provenance=None):
        r = TargetSchema(id="SRX1")
        setattr(r, field, value)
        if provenance:
            r.provenance[field] = provenance
        return r

    assert "host" in R.open_fields(record("host", schema.NOT_COLLECTED, "inferred_from_text"))
    assert "host" in R.open_fields(record("host", schema.NOT_COLLECTED, "inferred_from_paper"))
    # the regression this exists for: 380 GenomeTrakr samples declared
    # `geo_loc_name: "not provided"`, layer 2 recorded it, the old rule reopened
    # it, and layer 3 guessed a country from the submitting lab's address
    assert "country" not in R.open_fields(record("country", "not provided", "harmonized"))
    # unknown provenance stays closed — the conservative default
    assert "host" not in R.open_fields(record("host", schema.NOT_COLLECTED))
    # an ordinary value is never reopened, whoever wrote it
    for prov in ("direct", "harmonized", "inferred_from_text", None):
        assert "country" not in R.open_fields(record("country", "Brazil", prov))


def test_open_fields_can_treat_every_verdict_as_final():
    r = TargetSchema(id="SRX1")
    r.host = schema.NOT_COLLECTED
    r.provenance["host"] = "inferred_from_text"
    assert "host" in R.open_fields(r)                              # default
    assert "host" not in R.open_fields(r, overwrite_missing=False)  # opted out


def test_harmonized_verdict_survives_the_whole_cascade():
    # end to end: layer 2 records the submitter's "not provided" and layer 3 is
    # never even asked about it. Measured impact of getting this wrong: 369 of
    # 380 country values wrong, plus 103 host, across one study.
    layer = _FakeLayer({})          # a real layer only answers what it is offered
    saved = _with_layer("from_text", "inferred_from_text", layer)
    try:
        project = _study(samples={"SRS1": {"accession": "SRS1", "attributes": {
            "geo_loc_name": "not provided", "collected_by": "Institut Pasteur de Dakar"}}})
        records, _ = R.reconstruct(project, harmonize=True, from_text=True)
    finally:
        R.LAYERS = saved
    (r,) = records
    assert r.country == "not provided"
    assert r.provenance["country"] == "harmonized"
    # the guarantee: layer 3 is never even shown the field. If it proposed one
    # anyway, _apply refuses it — see
    # test_a_layer_reaching_outside_its_open_set_is_an_error.
    assert "country" not in layer.seen_open["SRX1"]
    assert "collected_by" not in layer.seen_open["SRX1"]   # harmonized too


def test_a_later_layer_overtakes_an_earlier_missing_verdict():
    layer = _FakeLayer({"SRX1": {"host": R.Proposal("mouse", "high")}})
    saved = _with_layer("from_paper", "inferred_from_paper", layer)
    try:
        records, _ = R.reconstruct(_study(), from_paper=True)
        (r,) = records
        # simulate the text layer having concluded "not collected" first
        r.host = schema.NOT_COLLECTED
        r.provenance["host"] = "inferred_from_text"
        r.confidence["host"] = "low"
        R._apply(records, layer.proposals, "inferred_from_paper",
                 {"SRX1": R.open_fields(r)})
        assert r.host == "mouse"
        # value, provenance and confidence all move to the layer that answered
        assert r.provenance["host"] == "inferred_from_paper"
        assert r.confidence["host"] == "high"
    finally:
        R.LAYERS = saved


def test_a_proposal_without_confidence_clears_a_stale_one():
    r = TargetSchema(id="SRX1")
    r.host = schema.NOT_COLLECTED
    r.provenance["host"] = "inferred_from_text"
    r.confidence["host"] = "low"
    R._apply([r], {"SRX1": {"host": R.Proposal("mouse")}},
             "inferred_from_paper", {"SRX1": R.open_fields(r)})
    assert r.host == "mouse" and "host" not in r.confidence
    assert r.inconsistent_confidence() == []


def test_a_layer_proposing_an_unknown_record_is_an_error():
    layer = _FakeLayer({"SRX-nope": {"host": R.Proposal("mouse")}})
    saved = _with_layer("from_text", "inferred_from_text", layer)
    try:
        R.reconstruct(_study(), from_text=True)
    except ValueError as exc:
        assert "SRX-nope" in str(exc)
    else:
        raise AssertionError("an unknown record id should be refused")
    finally:
        R.LAYERS = saved


# --------------------------------------------------------------------------- #
# claude.py — request shaping and bookkeeping (stubbed SDK, no network, no key)
# --------------------------------------------------------------------------- #
class _Usage:
    def __init__(self, **kw):
        for name in ("input_tokens", "output_tokens",
                     "cache_creation_input_tokens", "cache_read_input_tokens"):
            setattr(self, name, kw.get(name, 0))


class _Block:
    def __init__(self, type_, text=""):
        self.type, self.text = type_, text


class _Reply:
    def __init__(self, blocks, stop_reason="end_turn", stop_details=None, **usage):
        self.content, self.stop_reason = blocks, stop_reason
        self.stop_details, self.usage = stop_details, _Usage(**usage)


class _StubClient:
    """Capture the request params instead of sending them.

    Exposes both `.messages` and `.beta.messages`, and records which one was
    used — claude.py picks the beta path only for models that accept the
    `fallbacks` parameter, and sending it to any other model is a 400.
    """

    def __init__(self, reply):
        self.reply, self.params, self.path = reply, None, None
        outer = self

        class _Messages:
            def __init__(_s, path):
                _s.path = path

            def create(_s, **params):
                outer.params, outer.path = params, _s.path
                return outer.reply

        self.messages = _Messages("plain")

        class _Beta:
            messages = _Messages("beta")

        self.beta = _Beta()


class _NoNetworkClient:
    """Stands in for the Anthropic client so no test can spend real tokens.

    Claude calls cost money, and this suite runs on every commit — a single
    forgotten stub would bill silently and indefinitely rather than fail. This
    is installed at import (see below) and is what every stubbing test restores
    when it finishes, so the default state of the suite is "cannot reach the
    API" rather than "happens not to".
    """

    def __getattr__(self, name):
        raise AssertionError(
            "a test reached the real Claude API — those calls cost money, so the "
            "offline suite must stub claude._client_instance (see _with_claude_stub)"
        )


def _offline_guard():
    """Make claude unusable until a test deliberately stubs it."""
    C._api_key = "offline-tests-never-authenticate"  # so _client() skips the key file
    C._client_instance = _NoNetworkClient()


_offline_guard()


def _with_claude_stub(reply):
    """Install a stub client; returns it. Caller restores via _offline_guard()."""
    stub = _StubClient(reply)
    C._client_instance = stub
    C._api_key = "test-key"  # so _client() never reads the key file
    return stub


def test_claude_imports_without_a_key_on_disk():
    # the module must be importable in CI and in this suite, where no credential
    # exists — the client is built on first call, not at import
    assert C.MODEL == "claude-opus-5"
    assert C.DEFAULT_EFFORT in ("low", "medium", "high", "xhigh", "max")


def test_object_schema_requires_every_field_and_forbids_extras():
    # structured outputs reject open objects, and a property left out of
    # `required` may simply be omitted from the reply
    s = C.object_schema({"country": {"type": "string"}, "host": {"type": "string"}})
    assert s["additionalProperties"] is False
    assert sorted(s["required"]) == ["country", "host"]
    # an explicit subset is still honoured
    assert C.object_schema({"a": {}, "b": {}}, required=["a"])["required"] == ["a"]


def test_extract_sends_the_schema_and_parses_the_reply():
    saved = C._client_instance, C._api_key
    try:
        stub = _with_claude_stub(_Reply([_Block("thinking"), _Block("text", '{"country": "Brazil"}')]))
        schema = C.object_schema({"country": {"type": ["string", "null"]}})
        assert C.extract("prompt", schema, system="rules") == {"country": "Brazil"}

        p = stub.params
        assert p["output_config"]["format"] == {"type": "json_schema", "schema": schema}
        assert p["model"] == "claude-opus-5"
        assert p["thinking"] == {"type": "adaptive"}
        assert p["output_config"]["effort"] == C.DEFAULT_EFFORT
        # the system prompt is marked cacheable — it is identical on every call
        # and dwarfs the per-sample payload
        assert p["system"][0]["cache_control"] == {"type": "ephemeral"}
        assert p["system"][0]["text"] == "rules"
    finally:
        C._client_instance, C._api_key = saved


def test_fallbacks_are_sent_only_to_models_that_accept_them():
    # regression: `fallbacks` was sent unconditionally, so every model except
    # Opus 5 / Fable 5 / Mythos 5 returned 400 "does not support the
    # `fallbacks` parameter" — which quietly made this module Opus-5-only
    saved = C._client_instance, C._api_key
    try:
        stub = _with_claude_stub(_Reply([_Block("text", "hi")]))
        C.complete("p", model="claude-opus-5")
        assert stub.params["fallbacks"] == "default"
        assert stub.params["betas"] == [C._FALLBACK_BETA]
        assert stub.path == "beta"

        stub = _with_claude_stub(_Reply([_Block("text", "hi")]))
        C.complete("p", model="claude-haiku-4-5", effort=None, thinking=False)
        assert "fallbacks" not in stub.params and "betas" not in stub.params
        assert stub.path == "plain"  # no betas -> stay off the beta endpoint
        # and neither knob is sent when the model does not support it
        assert "thinking" not in stub.params
        assert "effort" not in stub.params.get("output_config", {})
    finally:
        C._client_instance, C._api_key = saved


def test_a_refusal_raises_instead_of_looking_like_an_empty_answer():
    # a refusal is a successful HTTP 200 with empty content; reading content[0]
    # without checking stop_reason would treat it as a valid extraction
    saved = C._client_instance, C._api_key
    try:
        details = type("D", (), {"category": "bio", "explanation": "declined"})()
        _with_claude_stub(_Reply([], stop_reason="refusal", stop_details=details))
        try:
            C.complete("prompt")
        except C.RefusalError as exc:
            assert exc.category == "bio" and "declined" in str(exc)
        else:
            raise AssertionError("a refusal must not be returned as an answer")
    finally:
        C._client_instance, C._api_key = saved


def test_usage_accounting_separates_cached_from_fresh_tokens():
    # cache reads bill at ~10% of the input rate and writes at ~125%, so folding
    # them into one number makes a bulk run's cost unreadable
    saved = C._client_instance, C._api_key
    C.reset_usage()
    try:
        _with_claude_stub(_Reply([_Block("text", "hi")], input_tokens=100, output_tokens=20,
                          cache_creation_input_tokens=900, cache_read_input_tokens=0))
        C.complete("a")
        _with_claude_stub(_Reply([_Block("text", "hi")], input_tokens=100, output_tokens=20,
                          cache_creation_input_tokens=0, cache_read_input_tokens=900))
        C.complete("b")
        tokens = {k: v for k, v in C.USAGE.items()
                  if not k.startswith("batch") and k != "cost"}
        assert tokens == {"calls": 2, "input": 200, "output": 40,
                          "cache_write": 900, "cache_read": 900}
        # The dollar figure is the point of tracking any of this: cache writes
        # bill at 125% and reads at 10%, so the same 900 tokens cost 12.5x more
        # to write than to read and the token totals alone say nothing.
        rate_in, rate_out = C.PRICES[C.MODEL]
        expected = (
            (200 * rate_in + 900 * rate_in * C.CACHE_WRITE_RATE
             + 900 * rate_in * C.CACHE_READ_RATE + 40 * rate_out) / 1e6
        )
        assert abs(C.USAGE["cost"] - expected) < 1e-9
        # batched tokens are counted apart — the 50% discount is a billing rate,
        # not a token reduction, so mixing them would overstate a batch run 2x
        assert C.USAGE["batch_calls"] == 0
        report = C.usage_report()
        assert "2 calls" in report and "900" in report and "$" in report
    finally:
        C.reset_usage()
        C._client_instance, C._api_key = saved


def test_missing_key_file_names_the_file_it_wants():
    try:
        C.set_api_key(path="/nonexistent/claude_api_key.txt")
    except FileNotFoundError as exc:
        assert "claude_api_key.txt" in str(exc)
    else:
        raise AssertionError("a missing credential file should say so")
    finally:
        _offline_guard()  # this test clears the key — put the guard back


def test_harmonization_maps_attribute_keys_onto_schema_fields():
    # no model, no network — a normalization pass plus the synonym table
    attrs = {
        "tissue": "liver",                    # synonym  -> tissue_type
        "Sex": "male",                        # casing   -> sex
        "cell type": "hepatocyte",            # spacing  -> cell_type
        "geo_loc_name": "China:Beijing",      # synonym + value transform
        "isolation_source": "liver biopsy",   # exact field name, no table row
        "env_broad_scale": "terrestrial biome",
        "BioSampleModel": "Model organism or animal",
        "host_subject_id": "M12",             # no schema field -> dropped
        "collection_date": "",                # blank -> not an answer
    }
    assert R._harmonized(attrs) == {
        "tissue_type": "liver",
        "sex": "male",
        "cell_type": "hepatocyte",
        "country": "China",                   # not "China:Beijing"
        "isolation_source": "liver biopsy",
        "broad_scale_environmental_context": "terrestrial biome",
        "ncbi_reporting_standard": "Model organism or animal",
    }


def test_harmonization_prefers_the_submitters_own_field_name():
    # a bag carrying both `tissue_type` and `tissue` keeps the exact match
    assert R._harmonized({"tissue": "liver", "tissue_type": "hepatic"}) == {
        "tissue_type": "hepatic"
    }
    assert R._harmonized({"tissue_type": "hepatic", "tissue": "liver"}) == {
        "tissue_type": "hepatic"
    }


def test_a_sample_description_is_not_written_into_the_study_abstract():
    # `description` is a submitter's *sample* attribute, but it is also the name
    # of the STUDY-level field (the abstract). The exact-name match used to win,
    # so the synonym row `description -> sample_description` could never fire and
    # 87 sample descriptions across the corpus landed in the study abstract.
    assert R._harmonized({"description": "Venezuelan Equine Encephalitis virus/BE508"}) == {
        "sample_description": "Venezuelan Equine Encephalitis virus/BE508"
    }


def test_a_shadowed_row_still_loses_to_its_own_exact_field_name():
    # The exception is narrow: it redirects `description` to the field the table
    # names, but an explicit `sample_description` key is still the submitter's
    # own and wins, in either bag order.
    assert R._harmonized(
        {"description": "from the synonym", "sample_description": "from the field"}
    ) == {"sample_description": "from the field"}
    assert R._harmonized(
        {"sample_description": "from the field", "description": "from the synonym"}
    ) == {"sample_description": "from the field"}


def test_no_synonym_row_is_unreachable():
    # The shadow set is derived from the table so a future row cannot silently
    # do nothing. Every row must be able to fire.
    fields = set(schema.TargetSchema.field_names())
    for key, target in R._SYNONYMS.items():
        # No identity-row exemption. A row whose key is its own target is
        # unreachable too - the exact-name match gets there first - and is dead
        # weight rather than documentation.
        assert key not in fields or key in R._SHADOWED_SYNONYMS, (
            f"synonym {key!r} is also a field name and is not shadow-handled, "
            f"so the row can never fire"
        )
    assert R._SHADOWED_SYNONYMS == {"description"}


def test_harmonized_values_carry_no_confidence():
    # nobody chose the value — the submitter supplied it — so there is nothing
    # to be confident about, and reconstruct() refuses a confidence from layer 2
    project = _study(samples={"SRS1": {"accession": "SRS1",
                                       "attributes": {"tissue": "liver"}}})
    records = TargetSchema.from_project(project)
    proposals = R.harmonize_attributes(project, records,
                                       {r.id: R.open_fields(r) for r in records})
    (fields,) = proposals.values()
    assert fields["tissue_type"] == R.Proposal("liver", None)


def test_harmonized_fields_are_closed_to_the_text_layer():
    # the point of running layer 2 first: it shrinks what layer 3 is asked
    project = _study(samples={"SRS1": {"accession": "SRS1",
                                       "attributes": {"tissue": "liver", "Sex": "male"}}})
    records, _ = R.reconstruct(project, harmonize=True)
    (r,) = records
    assert r.tissue_type == "liver" and r.sex == "male"
    assert r.provenance["tissue_type"] == "harmonized"
    assert "tissue_type" not in R.open_fields(r) and "sex" not in R.open_fields(r)


def test_study_level_fields_are_asked_once_per_study():
    # regression: study-level fields were asked once per *sample*, which over a
    # 60-sample run made ~106 duplicate asks and gave one study two different
    # `study_alias` answers
    saved = C._client_instance, C._api_key
    saved_batch, R.TEXT_BATCH = R.TEXT_BATCH, False   # pinned: spies on live extract
    try:
        stub = _with_claude_stub(_Reply([_Block("text", '{"answers": []}')]))
        asks = []
        real = C.extract

        def spy(prompt, schema, **kw):
            asks.append(set(schema["properties"]["answers"]["items"]
                            ["properties"]["field"]["enum"]))
            return real(prompt, schema, **kw)

        C.extract = spy
        try:
            project = _study(samples={"SRS1": {"accession": "SRS1"},
                                      "SRS2": {"accession": "SRS2"}},
                             experiments=[
                                 {"accession": "SRX1", "sample_ids": ["SRS1"]},
                                 {"accession": "SRX2", "sample_ids": ["SRS2"]}])
            records = TargetSchema.from_project(project)
            R.infer_from_text(project, records,
                              {r.id: R.open_fields(r) for r in records})
        finally:
            C.extract = real

        study_level = set(TargetSchema.fields_at_level(schema.STUDY))
        # one study call carrying the study-level fields, then one call per sample
        assert len(asks) == 3, asks
        assert asks[0] <= study_level                    # study call: only those
        assert all(not (a & study_level) for a in asks[1:])  # sample calls: none
    finally:
        C._client_instance, C._api_key = saved
        R.TEXT_BATCH = saved_batch


def test_the_text_layer_sends_its_own_model_settings():
    # the fake-layer swap used by the cascade tests never runs the real
    # infer_from_text, so a knob that isn't plumbed through would pass unnoticed
    saved = C._client_instance, C._api_key
    saved_batch, R.TEXT_BATCH = R.TEXT_BATCH, False   # pinned: this is the live path
    try:
        stub = _with_claude_stub(_Reply([_Block("text", '{"answers": []}')]))
        project = _study()
        records = TargetSchema.from_project(project)
        R.infer_from_text(project, records, {r.id: R.open_fields(r) for r in records})

        assert stub.params["model"] == R.TEXT_MODEL == "claude-haiku-4-5"
        # Haiku 4.5 predates both and rejects them — the knobs must not leak
        assert "thinking" not in stub.params
        assert "effort" not in stub.params.get("output_config", {})
        # ...and it must not take the beta path, which would 400 on `fallbacks`
        assert stub.path == "plain"
    finally:
        C._client_instance, C._api_key = saved
        R.TEXT_BATCH = saved_batch


class _BatchStub:
    """Stand in for client.messages.batches — records what was submitted."""

    def __init__(self, replies):
        self.replies, self.submitted = replies, None
        outer = self

        class _Batches:
            def create(_s, requests):
                outer.submitted = requests
                return type("B", (), {"id": "batch_test"})()

            def retrieve(_s, _id):
                return type("B", (), {"processing_status": "ended",
                                      "request_counts": type("C", (), {"succeeded": len(outer.submitted)})()})()

            def results(_s, _id):
                # one result per submitted request, not per canned reply
                for i, text in enumerate(outer.replies[:len(outer.submitted)]):
                    msg = _Reply([_Block("text", text)], input_tokens=10, output_tokens=5)
                    yield type("R", (), {
                        "custom_id": f"r{i}",
                        "result": type("X", (), {"type": "succeeded", "message": msg})(),
                    })()

        outer.live = []

        class _Live:
            def create(_s, **params):
                outer.live.append(params)
                return _Reply([_Block("text", outer.replies[0])],
                              input_tokens=10, output_tokens=5)

        self.messages = type("M", (), {"batches": _Batches(), "create": _Live().create})()
        self.beta = type("Bt", (), {"messages": None})()


def test_batched_and_live_paths_build_identical_request_bodies():
    # a batch that differs from the live call by any byte is a different cache
    # prefix and a different measurement — the two must not drift apart
    live = C._body("prompt", "sys", {"type": "object"}, "claude-haiku-4-5",
                   None, 16000, False, True, "1h")
    saved = C._client_instance, C._api_key
    try:
        stub = _BatchStub(['{"answers": []}'])
        C._client_instance, C._api_key = stub, "k"
        C.extract_batch({"a": ("prompt", {"type": "object"}),
                         "b": ("prompt", {"type": "object"})}, system="sys",
                        model="claude-haiku-4-5", effort=None, thinking=False,
                        prewarm=False)
        assert dict(stub.submitted[0]["params"]) == live
        # `fallbacks` is rejected by the Batches API and must never be sent
        assert "fallbacks" not in stub.submitted[0]["params"]
    finally:
        C._client_instance, C._api_key = saved


def test_batch_keys_survive_characters_the_api_rejects():
    # pooled-experiment ids contain a dot; custom_id charset does not allow it
    saved = C._client_instance, C._api_key
    try:
        stub = _BatchStub(['{"answers": [{"field": "host", "value": "mouse"}]}'])
        C._client_instance, C._api_key = stub, "k"
        out = C.extract_batch({"SRX1.SRS1": ("p", {})}, thinking=False, effort=None,
                              prewarm=False)
        assert stub.submitted[0]["custom_id"] == "r0"      # positional on the wire
        assert list(out) == ["SRX1.SRS1"]                  # caller key on the way back
    finally:
        C._client_instance, C._api_key = saved


def test_batched_tokens_are_billed_apart_from_live_ones():
    saved = C._client_instance, C._api_key
    C.reset_usage()
    try:
        stub = _BatchStub(['{"answers": []}', '{"answers": []}'])
        C._client_instance, C._api_key = stub, "k"
        C.extract_batch({"a": ("p", {}), "b": ("p", {})}, thinking=False, effort=None,
                        prewarm=False)
        assert C.USAGE["batch_calls"] == 2 and C.USAGE["calls"] == 0
        assert C.USAGE["batch_input"] == 20 and C.USAGE["input"] == 0
        # 50% discount applied to the batched half only
        live_equivalent = (20 * 1.0 + 10 * 5.0) / 1e6
        assert abs(C.estimated_cost("claude-haiku-4-5") - live_equivalent * 0.5) < 1e-12
    finally:
        C.reset_usage()
        C._client_instance, C._api_key = saved


def test_layer_three_runs_live_by_default():
    # batching is 12-28% cheaper but gives up progress and interruptibility: a
    # run killed mid-batch abandons work that still completes and still bills.
    # After a $7 overspend that trade is not worth the discount.
    assert R.TEXT_BATCH is False


def test_prewarm_is_off_by_default():
    # it improves the cache read/write ratio (0.62 -> 1.19) but still bills more
    # ($0.1786 -> $0.1858): the warm calls lose the 50% discount and one warm
    # call does not serialise a parallel batch
    import inspect
    assert inspect.signature(C.extract_batch).parameters["prewarm"].default is False


def test_prewarm_sends_one_live_call_per_distinct_prefix():
    # a batch runs its requests in parallel, so without a warm cache most of
    # them WRITE the shared prefix rather than reading it — measured at 4.4x
    # the writes and 55% of the reads, which ate half the batch discount
    saved = C._client_instance, C._api_key
    try:
        stub = _BatchStub(['{"answers": []}'] * 4)
        C._client_instance, C._api_key = stub, "k"
        schema_a, schema_b = {"type": "object", "x": 1}, {"type": "object", "x": 2}
        out = C.extract_batch(
            {"a1": ("p", schema_a), "a2": ("p", schema_a),
             "a3": ("p", schema_a), "b1": ("p", schema_b)},
            system="sys", model="claude-haiku-4-5", effort=None, thinking=False,
            prewarm=True,   # off by default — it measured more expensive
        )
        # two distinct schemas -> two live warm calls, the other two batched
        assert len(stub.live) == 2
        assert len(stub.submitted) == 2
        # every caller key is answered, warmed or batched
        assert sorted(out) == ["a1", "a2", "a3", "b1"]
        # the warm calls carry the same 1h ttl as the batch, or they would warm
        # a different cache entry than the batch reads
        assert stub.live[0]["system"][0]["cache_control"]["ttl"] == "1h"
        assert stub.submitted[0]["params"]["system"][0]["cache_control"]["ttl"] == "1h"
    finally:
        C._client_instance, C._api_key = saved


def test_the_real_layer_three_cannot_reach_the_api_under_test():
    # the guard exists because a stray from_text=True in a future test would bill
    # real tokens on every commit, silently. This asserts it actually bites: no
    # layer stub here, so reconstruct() runs the genuine infer_from_text.
    try:
        R.reconstruct(_study(), from_text=True)
    except AssertionError as exc:
        assert "cost money" in str(exc)
    else:
        raise AssertionError("layer 3 must not be able to reach the network in tests")


# --------------------------------------------------------------------------- #
# main.full_pipeline — wiring only (every stage stubbed, no network, no tokens)
# --------------------------------------------------------------------------- #
class _StagesStub:
    """Replace the three pipeline stages and the credential read."""

    def __init__(self):
        self.calls = {}

    def __enter__(self):
        self._saved = (M.set_entrez_credentials, M.read_credential,
                       M.save_recent_studies, M.filter_oa_studies,
                       M.save_reconstructed_records)
        M.set_entrez_credentials = lambda **kw: None
        M.read_credential = lambda path: "stub"
        M.save_recent_studies = lambda path, n, **kw: self.calls.__setitem__("harvest", path)
        M.filter_oa_studies = lambda **kw: self.calls.__setitem__("filter", kw)
        M.save_reconstructed_records = lambda **kw: self.calls.__setitem__("reconstruct", kw)
        return self

    def __exit__(self, *exc):
        (M.set_entrez_credentials, M.read_credential, M.save_recent_studies,
         M.filter_oa_studies, M.save_reconstructed_records) = self._saved
        return False


def test_pipeline_creates_the_output_directory():
    # so a new run prefix needs no setup — `full_pipeline(".../test3")` just works
    with tempfile.TemporaryDirectory() as tmp:
        target = os.path.join(tmp, "datasets", "test3")
        assert not os.path.exists(target)
        with _StagesStub() as stub:
            M.full_pipeline(file_location=target, dataset_prefix="test3")
        assert os.path.isdir(target)
        assert stub.calls["harvest"] == f"{target}/test3_studies.json"


def test_pipeline_refuses_a_missing_directory_before_doing_any_work():
    # without create_dirs the failure must land here, not after stage 1 has
    # finished harvesting and tries to write — that would throw away every
    # request the harvest just paid for
    with tempfile.TemporaryDirectory() as tmp:
        target = os.path.join(tmp, "nope")
        with _StagesStub() as stub:
            try:
                M.full_pipeline(file_location=target, create_dirs=False)
            except FileNotFoundError as exc:
                assert "create_dirs=True" in str(exc)
            else:
                raise AssertionError("a missing directory should be refused")
        assert stub.calls == {}          # nothing ran
        assert not os.path.exists(target)


def test_pipeline_is_idempotent_on_an_existing_directory():
    with tempfile.TemporaryDirectory() as tmp:
        with _StagesStub():
            M.full_pipeline(file_location=tmp)      # already exists
            M.full_pipeline(file_location=tmp)      # and again
        assert os.path.isdir(tmp)


def test_pipeline_passes_all_four_layer_flags_through():
    with tempfile.TemporaryDirectory() as tmp:
        with _StagesStub() as stub:
            M.full_pipeline(file_location=tmp, harmonize=True,
                            from_text=True, from_paper=True)
        kw = stub.calls["reconstruct"]
        assert (kw["harmonize"], kw["from_text"], kw["from_paper"]) == (True, True, True)
        # and the paper layer can be switched off on its own — it is the
        # expensive one, at ~$0.013/study against layer 3's ~$0.0034/record
        with _StagesStub() as stub:
            M.full_pipeline(file_location=tmp, from_paper=False)
        assert stub.calls["reconstruct"]["from_paper"] is False


def test_pipeline_bills_the_default_key_file_unless_told_otherwise():
    # the common case must not touch the credential machinery at all — a run
    # that says nothing about keys keeps whatever key is already loaded
    with tempfile.TemporaryDirectory() as tmp:
        saved = (C._api_key, C._client_instance)
        try:
            with _StagesStub():
                M.full_pipeline(file_location=tmp)
            assert (C._api_key, C._client_instance) == saved   # untouched
        finally:
            C._api_key, C._client_instance = saved


def test_pipeline_can_be_pointed_at_a_different_claude_key():
    # layers 3 and 4 are the only things that spend, so which account pays is a
    # per-run choice, not a constant in claude.py
    with tempfile.TemporaryDirectory() as tmp:
        key_file = os.path.join(tmp, "other_key.txt")
        with open(key_file, "w", encoding="utf-8") as fh:
            fh.write("  sk-ant-other\n")             # padded: stripping is the point
        saved = (C._api_key, C._client_instance)
        try:
            with _StagesStub():
                M.full_pipeline(file_location=tmp, claude_key_file=key_file)
            assert _fp(C._api_key) == _fp("sk-ant-other")
            # and the cached client is dropped, or the *old* key would keep
            # billing for the rest of the process
            assert C._client_instance is None
        finally:
            C._api_key, C._client_instance = saved


def _fp(text):
    """Fingerprint a credential for assertions.

    Tests compare keys by hash, never by value: an assertion on the raw string
    prints both sides when it fails, and a test that accidentally reads the
    real key file would then dump a live credential into the output. That is
    not hypothetical — it is how this file's first draft failed.
    """
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def test_there_is_no_default_credential_file():
    # the whole point: an implicit default meant "I named no key" and "no key
    # was used" were different things, and a stale claude_api_key.txt in the
    # repo root silently became the account that paid
    assert C.API_KEY_FILE is None
    saved = (C._api_key, C._client_instance, C._api_key_source)
    try:
        C._api_key, C._client_instance, C._api_key_source = None, None, None
        with pytest.raises(C.MissingAPIKeyError):
            C.set_api_key()                    # neither key= nor path=
        assert C._api_key is None              # and nothing was resolved
        assert C.key_source() == "(none configured)"
    finally:
        C._api_key, C._client_instance, C._api_key_source = saved
        _offline_guard()


def test_client_refuses_to_build_without_a_key_instead_of_hunting_for_one():
    saved = (C._api_key, C._client_instance, C._api_key_source)
    try:
        C._api_key, C._client_instance, C._api_key_source = None, None, None
        with pytest.raises(C.MissingAPIKeyError) as exc:
            C._client()
        assert "nothing has been spent" in str(exc.value).lower()
    finally:
        C._api_key, C._client_instance, C._api_key_source = saved
        _offline_guard()


def test_a_malformed_credential_is_rejected_at_load_not_at_first_request():
    # catching it here costs a file read; catching it at the first call costs
    # the whole harvest that ran before it
    cases = {
        "sk-ant-has space": "whitespace",       # authenticates as a different string
        "not-an-anthropic-key": "Anthropic",    # wrong file entirely (NCBI key? email?)
        "": "empty",
    }
    with tempfile.TemporaryDirectory() as tmp:
        saved = (C._api_key, C._client_instance, C._api_key_source)
        try:
            for content, expected in cases.items():
                path = os.path.join(tmp, "k.txt")
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(content + "\n")
                with pytest.raises(C.MissingAPIKeyError) as exc:
                    C.set_api_key(path=path)
                assert expected in str(exc.value)
        finally:
            C._api_key, C._client_instance, C._api_key_source = saved
            _offline_guard()


def test_naming_both_a_key_and_a_file_is_refused_as_ambiguous():
    with pytest.raises(ValueError):
        C.set_api_key(key="sk-ant-x", path="some_file.txt")


def test_api_key_file_is_read_at_call_time_not_import_time():
    # binding path=API_KEY_FILE as a parameter default froze it at import, so
    # reassigning claude.API_KEY_FILE silently kept billing the original account
    with tempfile.TemporaryDirectory() as tmp:
        redirected = os.path.join(tmp, "redirected.txt")
        with open(redirected, "w", encoding="utf-8") as fh:
            fh.write("sk-ant-redirected\n")
        saved = (C._api_key, C._client_instance, C._api_key_source, C.API_KEY_FILE)
        try:
            C.API_KEY_FILE = redirected
            C.set_api_key()
            assert _fp(C._api_key) == _fp("sk-ant-redirected")
        finally:
            (C._api_key, C._client_instance, C._api_key_source,
             C.API_KEY_FILE) = saved
            _offline_guard()


def test_key_source_reports_without_leaking_the_key():
    saved = (C._api_key, C._client_instance, C._api_key_source)
    try:
        C.set_api_key(key="sk-ant-secret-value")
        assert "secret" not in C.key_source()     # never the credential itself
        assert C.key_source() == "(passed directly)"
        # with nothing loaded it says so, rather than naming a file that may
        # not be the one that ends up paying
        C._api_key_source = None
        assert C.key_source() == "(none configured)"
    finally:
        C._api_key, C._client_instance, C._api_key_source = saved
        _offline_guard()


def test_reconstruct_stage_can_select_its_own_key():
    # main.full_pipeline is not the only entry point — running the stage
    # directly (as README suggests) must be able to choose the account too
    with tempfile.TemporaryDirectory() as tmp:
        key_file = os.path.join(tmp, "other_key.txt")
        with open(key_file, "w", encoding="utf-8") as fh:
            fh.write("sk-ant-stage-key\n")
        in_path = os.path.join(tmp, "studies.json")
        with open(in_path, "w", encoding="utf-8") as fh:
            fh.write("[]")
        saved = (C._api_key, C._client_instance, C._api_key_source)
        try:
            D.save_reconstructed_records(
                in_path=in_path, out_path=os.path.join(tmp, "out.json"),
                expand=False, claude_key_file=key_file,
            )
            assert _fp(C._api_key) == _fp("sk-ant-stage-key")
        finally:
            C._api_key, C._client_instance, C._api_key_source = saved
            _offline_guard()


def test_paid_layers_refuse_to_start_without_a_key():
    # the guard must fire before load_studies, so a keyless run costs nothing
    # at all — not a harvest, not an expansion, not a request
    with tempfile.TemporaryDirectory() as tmp:
        in_path = os.path.join(tmp, "studies.json")
        with open(in_path, "w", encoding="utf-8") as fh:
            fh.write("[]")
        saved = (C._api_key, C._client_instance, C._api_key_source)
        try:
            C._api_key, C._client_instance, C._api_key_source = None, None, None
            for layers in ({"from_text": True}, {"from_paper": True},
                           {"from_text": True, "from_paper": True}):
                with pytest.raises(C.MissingAPIKeyError):
                    D.save_reconstructed_records(
                        in_path=in_path, out_path=os.path.join(tmp, "out.json"),
                        expand=False, **layers,
                    )
                assert not os.path.exists(os.path.join(tmp, "out.json"))
        finally:
            C._api_key, C._client_instance, C._api_key_source = saved
            _offline_guard()


def test_a_free_run_does_not_claim_to_be_billing_a_key(capsys):
    # harmonize is layer 2 — a synonym table, no model. Testing `any(layers)`
    # instead of the paid layers made a free run print
    # "Billing Claude key: (none configured)", which reads as though a run with
    # no credential were about to bill one.
    with tempfile.TemporaryDirectory() as tmp:
        in_path = os.path.join(tmp, "studies.json")
        with open(in_path, "w", encoding="utf-8") as fh:
            fh.write("[]")
        saved = (C._api_key, C._client_instance, C._api_key_source)
        try:
            C._api_key, C._client_instance, C._api_key_source = None, None, None
            D.save_reconstructed_records(
                in_path=in_path, out_path=os.path.join(tmp, "out.json"),
                expand=False, harmonize=True, from_text=False, from_paper=False)
            out = capsys.readouterr().out
            assert "Billing Claude key" not in out
            assert "Claude usage" not in out
        finally:
            C._api_key, C._client_instance, C._api_key_source = saved
            _offline_guard()


def test_a_free_run_still_needs_no_key_at_all():
    # layers 1 and 2 cost nothing, so requiring a credential for them would
    # break the one mode that is safe to run with no account configured
    with tempfile.TemporaryDirectory() as tmp:
        in_path = os.path.join(tmp, "studies.json")
        with open(in_path, "w", encoding="utf-8") as fh:
            fh.write("[]")
        saved = (C._api_key, C._client_instance, C._api_key_source)
        try:
            C._api_key, C._client_instance, C._api_key_source = None, None, None
            D.save_reconstructed_records(              # must not raise
                in_path=in_path, out_path=os.path.join(tmp, "out.json"),
                expand=False, harmonize=True, from_text=False, from_paper=False,
            )
        finally:
            C._api_key, C._client_instance, C._api_key_source = saved
            _offline_guard()


def test_full_pipeline_refuses_a_keyless_paid_run_before_touching_anything():
    # not even the output directory should be created — the credential is the
    # first thing checked, ahead of NCBI and ahead of stage 1
    with tempfile.TemporaryDirectory() as tmp:
        target = os.path.join(tmp, "datasets", "nokey")
        saved = (C._api_key, C._client_instance, C._api_key_source)
        try:
            C._api_key, C._client_instance, C._api_key_source = None, None, None
            with _StagesStub() as stub:
                with pytest.raises(C.MissingAPIKeyError):
                    M.full_pipeline(file_location=target, from_text=True)
            assert stub.calls == {}                 # no harvest, no filter
            assert not os.path.exists(target)       # not even a mkdir
        finally:
            C._api_key, C._client_instance, C._api_key_source = saved
            _offline_guard()


def test_pipeline_refuses_a_missing_claude_key_before_harvesting():
    # same reasoning as the missing-directory check: a typo'd key path must cost
    # nothing, not surface after stage 1 has spent its requests
    with tempfile.TemporaryDirectory() as tmp:
        saved = (C._api_key, C._client_instance)
        try:
            with _StagesStub() as stub:
                try:
                    M.full_pipeline(file_location=tmp, claude_key_file="/nonexistent/nope.txt")
                except FileNotFoundError as exc:
                    assert "nope.txt" in str(exc)
                else:
                    raise AssertionError("a missing key file should be refused")
            assert stub.calls == {}          # nothing harvested
        finally:
            C._api_key, C._client_instance = saved


# --------------------------------------------------------------------------- #
# corpus — the expanded, self-contained dump (no network)
# --------------------------------------------------------------------------- #
class _Esearch:
    """Minimal stand-in for the esearch JSON response."""

    def __init__(self, uids):
        self._uids = uids

    def json(self):
        return {"esearchresult": {"idlist": self._uids}}


def test_biosample_fetch_discards_accessions_it_did_not_ask_for(monkeypatch):
    # E-utilities matches BioSample ids on the numeric part and ignores the
    # archive prefix: asking for DDBJ's SAMD00041293 returns NCBI's
    # SAMN00041293 — a different sample, from a different study, with a 200 and
    # no warning. Observed returning Human Microbiome Project metadata for a
    # honey-bee study. Anything keyed by request order would have stored it.
    xml = ET.fromstring(
        '<BioSampleSet><BioSample accession="SAMN00041293">'
        '<Attributes><Attribute attribute_name="host_sex">female</Attribute>'
        '</Attributes></BioSample></BioSampleSet>')
    # esearch resolves the uid; efetch returns the wrong-prefix record
    monkeypatch.setattr(P, "_request_with_retry",
                        lambda url, **k: _Esearch(["4112774"]) if "esearch" in url else xml)
    got = P.fetch_biosample_attributes(["SAMD00041293"])
    assert got == {}, "a mismatched accession must never be stored"


def test_biosample_fetch_keeps_accessions_it_did_ask_for(monkeypatch):
    xml = ET.fromstring(
        '<BioSampleSet><BioSample accession="SAMN10538065">'
        '<Attributes><Attribute attribute_name="host_sex">female</Attribute>'
        '<Attribute attribute_name="blank"> </Attribute></Attributes>'
        '</BioSample></BioSampleSet>')
    monkeypatch.setattr(P, "_request_with_retry",
                        lambda url, **k: _Esearch(["10538065"]) if "esearch" in url else xml)
    got = P.fetch_biosample_attributes(["SAMN10538065"])
    assert got == {"SAMN10538065": {"host_sex": "female"}}   # blank value dropped


def test_biosample_attributes_stay_separate_from_the_sra_bag():
    # merging would erase which archive said what — the distinction a benchmark
    # needs — and would change what layer 3 is credited with inferring
    s = P.Sample(accession="SRS1", biosample="SAMN1")
    s.attributes = {"tissue": "liver"}
    s.biosample_attributes = {"tissue": "liver", "host_sex": "female"}
    rebuilt = P.Sample(**{**s.__dict__})
    assert rebuilt.attributes == {"tissue": "liver"}
    assert "host_sex" in rebuilt.biosample_attributes


def test_corpus_round_trips_and_refuses_an_unknown_format_version():
    import corpus
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "c.json")
        payload = {
            "format_version": corpus.FORMAT_VERSION, "created": "x", "params": {},
            "counts": {}, "papers": {"1": {"id": "1", "type": "ePubmed",
                                           "chars": 3, "text": "abc"}},
            "studies": [{"accession": "SRP1", "title": "t",
                         "samples": {"SRS1": {"accession": "SRS1",
                                              "biosample_attributes": {"sex": "male"}}},
                         "experiments": [], "publications": [], "paper_id": "1"}],
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        projects, papers, meta = corpus.load_full_corpus(path)
        assert projects[0].accession == "SRP1"
        assert projects[0].samples["SRS1"].biosample_attributes == {"sex": "male"}
        assert papers["1"]["text"] == "abc"

        payload["format_version"] = corpus.FORMAT_VERSION + 99
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        with pytest.raises(ValueError) as exc:
            corpus.load_full_corpus(path)
        assert "format_version" in str(exc.value)


# --------------------------------------------------------------------------- #
# per-layer model settings and the model-aware cost estimate
# --------------------------------------------------------------------------- #
class _ModelSettings:
    """Restore reconstruct's model globals — configure_models mutates them."""

    def __enter__(self):
        self._saved = (R.TEXT_MODEL, R.TEXT_EFFORT, R.TEXT_THINKING,
                       R.PAPER_MODEL, R.PAPER_EFFORT, R.PAPER_THINKING)
        return self

    def __exit__(self, *exc):
        (R.TEXT_MODEL, R.TEXT_EFFORT, R.TEXT_THINKING,
         R.PAPER_MODEL, R.PAPER_EFFORT, R.PAPER_THINKING) = self._saved
        return False


def test_thinking_false_actually_disables_thinking_on_models_that_default_to_it():
    # the $1.25-on-a-$0.49-estimate bug: omitting `thinking` means *adaptive*
    # on Sonnet 5 and Opus 5, so thinking=False has to say "disabled" out loud
    for model in C.THINKS_BY_DEFAULT:
        body = C._body("p", "sys", None, model, None, 1000, False, False)
        assert body["thinking"] == {"type": "disabled"}, model
    # ...and must not be sent to models that predate the field and reject it
    body = C._body("p", "sys", None, C.HAIKU_4_5, None, 1000, False, False)
    assert "thinking" not in body
    # thinking=True is adaptive everywhere it is accepted
    body = C._body("p", "sys", None, C.SONNET_5, None, 1000, True, False)
    assert body["thinking"] == {"type": "adaptive"}


def test_an_unset_effort_is_priced_as_the_api_default_not_as_cheap():
    # `None` means the API's own default, which is `high`. Pricing it at the
    # `low` rate is what let a $1.25 run through a $1.00 cap.
    assert D.THINKING_MULTIPLIER[None] == D.THINKING_MULTIPLIER[C.EFFORT_HIGH]
    assert D.THINKING_MULTIPLIER[None] > D.THINKING_MULTIPLIER[C.EFFORT_LOW]


def test_the_estimate_now_covers_the_run_that_overspent():
    # the exact configuration that billed $1.25: Sonnet 5, effort unset,
    # thinking on. The estimate must land at or above that, not under it.
    studies = [_cost_study(f"SRP{i}", n) for i, n in
               enumerate((2, 6, 15, 22, 7))]          # the real test2 shape
    total, _ = D.estimate_reconstruction_cost(
        studies, from_text=True, from_paper=True,
        text_model=C.SONNET_5, text_thinking=True,
        paper_model=C.SONNET_5, paper_thinking=True)
    assert total >= 1.25, f"estimate {total:.2f} is below the measured $1.25"


def test_call_cost_prices_cache_reads_and_writes_apart():
    class _U:
        input_tokens = 1_000_000
        output_tokens = 0
        cache_read_input_tokens = 0
        cache_creation_input_tokens = 0

    rate_in, _ = C.PRICES[C.SONNET_5]
    assert abs(C.call_cost(_U(), C.SONNET_5) - rate_in) < 1e-9
    _U.input_tokens, _U.cache_read_input_tokens = 0, 1_000_000
    assert abs(C.call_cost(_U(), C.SONNET_5) - rate_in * C.CACHE_READ_RATE) < 1e-9
    _U.cache_read_input_tokens, _U.cache_creation_input_tokens = 0, 1_000_000
    assert abs(C.call_cost(_U(), C.SONNET_5) - rate_in * C.CACHE_WRITE_RATE) < 1e-9


def test_every_named_model_is_priced():
    # the constants exist so a model is chosen by name, and a named model that
    # cannot be costed would fail at the spend guard instead of at import —
    # this keeps claude.MODELS and claude.PRICES from drifting apart
    assert set(C.MODELS) == set(C.PRICES)
    assert C.MODEL in C.PRICES                       # the transport default too
    assert R.TEXT_MODEL in C.PRICES and R.PAPER_MODEL in C.PRICES


def test_named_effort_levels_match_the_cost_table():
    # a level with no multiplier would raise a KeyError deep inside the
    # estimate rather than being rejected by validate_model_settings
    assert set(C.EFFORT_LEVELS) <= set(D.THINKING_MULTIPLIER)
    assert C.DEFAULT_EFFORT in C.EFFORT_LEVELS


def test_the_constants_are_the_api_ids_so_raw_strings_still_work():
    # the constants are for the caller; the wire format is unchanged, and every
    # dataset and checkpoint already written names models as plain strings
    assert C.HAIKU_4_5 == "claude-haiku-4-5"
    assert C.OPUS_5 == "claude-opus-5"
    assert D.cost_multiplier(C.OPUS_5, C.EFFORT_MEDIUM, True) == \
           D.cost_multiplier("claude-opus-5", "medium", True)


def test_the_estimate_follows_the_model_instead_of_assuming_the_baseline():
    # the whole point: a hardcoded per-record cost meant switching to a pricier
    # model left max_spend guarding a number with no relation to the bill
    studies = [_cost_study("SRP1", 1000)]
    cheap, _ = D.estimate_reconstruction_cost(
        studies, from_text=True, text_model="claude-haiku-4-5")
    dear, _ = D.estimate_reconstruction_cost(
        studies, from_text=True, text_model="claude-opus-5",
        text_effort="medium", text_thinking=True)
    assert dear > cheap * 9        # 5x price, ~1.9x thinking
    assert D.cost_multiplier("claude-haiku-4-5", None, False) == 1.0


def test_the_two_layers_are_priced_on_their_own_models():
    # layer 4 on Opus while layer 3 stays on Haiku is the whole reason the
    # settings are split; the estimate has to reflect that, not average them
    studies = [_cost_study("SRP1", 100)]
    total, report = D.estimate_reconstruction_cost(
        studies, from_text=True, from_paper=True,
        text_model="claude-haiku-4-5",
        paper_model="claude-opus-5", paper_effort="medium", paper_thinking=True)
    expected = (100 * D.COST_PER_RECORD_TEXT
                + D.COST_PER_STUDY_PAPER * D.cost_multiplier("claude-opus-5", "medium", True))
    assert abs(total - expected) < 1e-9
    assert "claude-haiku-4-5" in report and "claude-opus-5" in report


def test_an_unpriced_model_stops_the_run_rather_than_guessing():
    # falling back to the baseline price would hand max_spend a meaningless
    # number — the same way a $1.00 cap once let a $7.00 run through
    with pytest.raises(D.UnpricedModelError) as exc:
        D.estimate_reconstruction_cost(
            [_cost_study("SRP1", 10)], from_text=True, text_model="claude-made-up-9")
    assert "max_spend cannot protect it" in str(exc.value)


def test_combinations_the_api_rejects_are_refused_before_any_work():
    # each of these is a 400 at request time, which would land after the
    # harvest and after earlier studies had already billed
    with pytest.raises(ValueError) as exc:      # Haiku predates both parameters
        D.validate_model_settings("claude-haiku-4-5", "medium", False,
                                  "claude-haiku-4-5", None, False)
    assert "400" in str(exc.value)
    with pytest.raises(ValueError):             # Opus 5 won't disable thinking that high
        D.validate_model_settings("claude-opus-5", "xhigh", False,
                                  "claude-opus-5", None, True)
    with pytest.raises(ValueError):             # not an effort level at all
        D.validate_model_settings("claude-opus-5", "turbo", True,
                                  "claude-opus-5", None, True)
    # ...and the same settings are fine on a layer that is switched off
    D.validate_model_settings("claude-haiku-4-5", "medium", True,
                              "claude-opus-5", None, False,
                              from_text=False, from_paper=False)


def test_configure_models_leaves_unset_layers_alone():
    # so changing the paper model doesn't silently reset layer 3's settings
    with _ModelSettings():
        R.configure_models(text_model="claude-sonnet-5", text_thinking=True)
        R.configure_models(paper_model="claude-opus-5")
        assert R.TEXT_MODEL == "claude-sonnet-5" and R.TEXT_THINKING is True
        assert R.PAPER_MODEL == "claude-opus-5"


def test_the_stage_refuses_a_bad_combination_before_loading_studies():
    with tempfile.TemporaryDirectory() as tmp:
        in_path = os.path.join(tmp, "studies.json")
        with open(in_path, "w", encoding="utf-8") as fh:
            fh.write("[]")
        with _ModelSettings():
            with pytest.raises(ValueError):
                D.save_reconstructed_records(
                    in_path=in_path, out_path=os.path.join(tmp, "out.json"),
                    expand=False, from_text=True,
                    text_model="claude-haiku-4-5", text_effort="max",
                )
            assert not os.path.exists(os.path.join(tmp, "out.json"))


# --------------------------------------------------------------------------- #
# audit — the verbatim checker (no network, no tokens)
# --------------------------------------------------------------------------- #
def _audited(**fields):
    """A record whose fields all came from layer 3, so all are auditable."""
    record = TargetSchema(id=fields.pop("id", "SRX1"))
    for name, value in fields.items():
        setattr(record, name, value)
        record.provenance[name] = "inferred_from_text"
        record.confidence[name] = "high"
    return record


def test_normalize_survives_the_json_attribute_bag():
    # the evidence carries the bag as json.dumps, so the answer "8 weeks" has to
    # match `{"age": "8 weeks"}` through the quoting and punctuation
    assert A.is_verbatim("8 weeks", 'SAMPLE ATTRIBUTES: {"age": "8 weeks"}')
    assert A.is_verbatim("CD4+ T cell", "SAMPLE TITLE: sorted CD4+ T cell, donor 3")
    assert A.is_verbatim("Mus musculus", "ORGANISM: mus musculus")   # casefolded


def test_normalize_respects_token_boundaries():
    # the padding in normalize() is what stops a substring hit inside a word
    assert not A.is_verbatim("male", "SAMPLE TITLE: female liver")
    assert A.is_verbatim("male", "SAMPLE TITLE: adult male liver")
    assert not A.is_verbatim("", "anything")          # an empty answer is not a quote


def test_offline_mode_reports_unknown_never_not_verbatim():
    # offline evidence is missing the attribute bag, so a non-match proves
    # nothing — calling it a failure would invent one for most of the dataset
    record = _audited(tissue_type="liver", sex="male", sample_title="mouse liver")
    result = A.audit_records([record])
    assert result["exact"] is False
    assert result["real_verdicts"][("high", A.NOT_VERBATIM)] == 0
    assert result["real_verdicts"][("high", A.VERBATIM)] == 1        # 'liver' is in the title
    # 'male' is not in the recoverable evidence, and sample_title is excluded
    # from its own evidence rather than trivially quoting itself
    assert result["real_verdicts"][("high", A.UNKNOWN)] == 2


def test_a_field_cannot_quote_itself_in_offline_mode():
    # description is part of the evidence string *and* a field layer 3 fills,
    # so without the exclusion it would verify against itself every time and
    # inflate the offline rate on exactly the fields it can say least about
    record = _audited(description="a study of mouse liver")
    assert A.audit_records([record])["real_verdicts"][("high", A.VERBATIM)] == 0


def test_exact_mode_can_fail_a_value():
    # with the real evidence in hand, a non-match *is* a finding
    record = _audited(tissue_type="liver", sex="male")
    evidence = {record.id: ("", "SAMPLE ATTRIBUTES: {\"tissue\": \"liver\"}")}
    result = A.audit_records([record], evidence)
    assert result["exact"] is True
    assert result["real_verdicts"][("high", A.VERBATIM)] == 1
    assert result["real_verdicts"][("high", A.NOT_VERBATIM)] == 1
    assert result["real_verdicts"][("high", A.UNKNOWN)] == 0


def test_study_level_fields_are_checked_against_the_study_evidence():
    # infer_from_text asks study fields in their own call, against text that
    # never includes sample attributes — auditing them against the sample
    # string would credit a quote the model could not have seen
    record = _audited(study_title="Liver atlas", tissue_type="liver")
    evidence = {record.id: ("STUDY TITLE: Liver atlas", "SAMPLE ATTRIBUTES: {}")}
    result = A.audit_records([record], evidence)
    assert result["real_verdicts"][("high", A.VERBATIM)] == 1        # study_title
    assert result["real_verdicts"][("high", A.NOT_VERBATIM)] == 1    # tissue_type


def test_missing_value_terms_are_counted_apart_from_real_values():
    # they are determinations, not quotes: mixing them in made every per-field
    # offender simply "a field answered not applicable"
    record = _audited(cell_line="not applicable", tissue_type="liver")
    evidence = {record.id: ("", "SAMPLE ATTRIBUTES: {\"tissue\": \"liver\"}")}
    result = A.audit_records([record], evidence)
    assert result["missing_values"][("high", A.NOT_VERBATIM)] == 1
    assert sum(result["real_by_level"].values()) == 1                # only tissue_type
    assert "cell_line" not in result["per_field"]


def test_a_quoted_missing_value_term_is_recognised_as_a_quote():
    # if the bag literally says "not collected", that is a quote and the new
    # rules call it high — the checker has to agree
    record = _audited(sex="not collected")
    evidence = {record.id: ("", 'SAMPLE ATTRIBUTES: {"sex": "not collected"}')}
    result = A.audit_records([record], evidence)
    assert result["missing_values"][("high", A.VERBATIM)] == 1


def test_paper_layer_is_reported_unauditable_not_dropped():
    # PAPER_MAX_CHARS of full text is never persisted, so layer 4 cannot be
    # checked — saying so is different from quietly excluding it
    record = TargetSchema(id="SRX2")
    record.tissue_type = "liver"
    record.provenance["tissue_type"] = "inferred_from_paper"
    record.confidence["tissue_type"] = "high"
    result = A.audit_records([record])
    assert result["unauditable"] == 1
    assert sum(result["by_level"].values()) == 0
    assert "not persisted" in A.format_report(result)


def test_direct_and_harmonized_fields_are_not_audited():
    # neither carries a confidence to check; only the inferred_* classes do
    record = TargetSchema(id="SRX3")
    record.tissue_type = "liver"
    record.provenance["tissue_type"] = "harmonized"
    result = A.audit_records([record])
    assert sum(result["by_level"].values()) == 0
    assert result["unauditable"] == 0


def test_report_renders_without_a_crash_on_an_empty_audit():
    assert "nothing to audit" in A.format_report(A.audit_records([]))


def test_expand_without_a_studies_path_is_refused():
    with pytest.raises(ValueError) as exc:
        A.verbatim_report("whatever.json", expand=True)
    assert "studies_path" in str(exc.value)


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