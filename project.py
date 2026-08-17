"""Object model for a single SRA study (SRP) and everything under it.

The classes mirror SRA's real reference-graph model:

    Study (SRP)  ── owns ──>  Experiment (SRX)  ── owns ──>  Run (SRR)
        │                          │
        │ owns (canonical store)   └── references ──> Sample (SRS) by id
        └──> Sample (SRS)

Study also carries the cross-database handles (BioProject / BioSample) and the
publications parsed from the BioProject record.

Build one straight from an accession:

    study = Project("SRP098789")
    print(study.title, len(study.samples), len(study.experiments))
    df = study.to_dataframe()          # flatten back to one-row-per-run

Everything is fetched live from NCBI E-utilities; nothing here depends on pysradb.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import time
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import date

import requests

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
EPMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
PMC_OA = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"
UNPAYWALL = "https://api.unpaywall.org/v2"
EPMC_REST = "https://www.ebi.ac.uk/europepmc/webservices/rest"

# Publication accessibility classes returned by classify_publication().
PUBLICATION_CLASSES = ("oa", "partial", "paywall", "unknown")

# Consecutive esummary page failures in search_studies before enumeration gives up.
# Isolated failures are normal over a long harvest; a run of them means the shared
# WebEnv session is gone, and no later page can succeed either.
_MAX_CONSECUTIVE_PAGE_FAILURES = 5

# --------------------------------------------------------------------------- #
# Process-wide NCBI credentials
#
# Credentials are session config, not per-study data, so they live here rather
# than on each Project. Every network call falls back to these when not given an
# explicit value — set them once and everything (including studies reloaded from
# JSON via load_studies) uses them. Explicit per-call arguments still win.
# --------------------------------------------------------------------------- #
_DEFAULT_EMAIL: str | None = None
_DEFAULT_API_KEY: str | None = None


def set_entrez_credentials(email: str | None = None, api_key: str | None = None) -> None:
    """Set the process-wide default NCBI credentials used by every network call.

    * ``email``   — contact address NCBI can warn before blocking your IP.
    * ``api_key`` — raises the E-utilities limit from 3 to 10 requests/second.

    Call once at startup. Both are replaced on every call (pass no args to clear).
    """
    global _DEFAULT_EMAIL, _DEFAULT_API_KEY
    _DEFAULT_EMAIL = email
    _DEFAULT_API_KEY = api_key


def _common(email=None, api_key=None, **extra) -> dict:
    """Shared request params, falling back to the process-wide credentials."""
    p = {"tool": "metadata-project"}
    e = email if email is not None else _DEFAULT_EMAIL
    k = api_key if api_key is not None else _DEFAULT_API_KEY
    if e:
        p["email"] = e
    if k:
        p["api_key"] = k
    p.update(extra)
    return p


_NO_KEY_SLEEP = 0.34  # ~3 requests/sec (NCBI's limit without an api_key)
_KEY_SLEEP = 0.11  # ~9 requests/sec (just under the 10/sec limit an api_key grants)


def _sleep_for(sleep=None, api_key=None) -> float:
    """Resolve the inter-request pacing delay.

    Honors an explicit ``sleep``; otherwise picks from the credential state — with
    an api_key (explicit or process-wide) NCBI allows 10 req/s so we pace at
    ~0.11s, without one 3 req/s so ~0.34s.

    Don't expect the api_key's 3x rate cap to make a bulk run 3x faster: the run is
    latency-bound, not rate-limited. Pacing is roughly a quarter of a scan's wall
    time (the rest is spent waiting on responses), so the key is worth ~20%.
    """
    if sleep is not None:
        return sleep
    return _KEY_SLEEP if (api_key or _DEFAULT_API_KEY) else _NO_KEY_SLEEP


# One connection pool for the whole process. A bulk run makes thousands of calls
# to a handful of hosts, and a fresh requests.get() re-does the TCP + TLS handshake
# every time — measured at ~0.18s per request against eutils, which is most of a
# small request's cost. Keep-alive through a shared Session removes it.
_SESSION = requests.Session()

# Transient failures worth another attempt. ChunkedEncodingError/ContentDecodingError
# are *siblings* of ConnectionError under RequestException, not subclasses, so they
# slip past the obvious `except (ConnectionError, Timeout)` — a body that dies
# mid-stream ("Response ended prematurely") used to kill the whole run. E-utilities
# truncates responses under load often enough that a multi-hour harvest will hit it.
# Everything else (a bad URL, too many redirects) is a real error: fail fast.
_TRANSIENT = (
    requests.ConnectionError,
    requests.Timeout,
    requests.exceptions.ChunkedEncodingError,
    requests.exceptions.ContentDecodingError,
)


def _request_with_retry(
    url, *, sleep: float = 0.34, post: bool = False, xml: bool = False,
    attempts: int = 5, **params
):
    """GET/POST with exponential backoff on rate-limit and transient server errors.

    Every network call in this module goes through here: NCBI allows only 3
    requests/second without an api_key (10 with one) and E-utilities also returns
    sporadic 5xx under load. Dropped connections and DNS hiccups are retried too —
    over a run of hundreds of studies one is near-certain, and it shouldn't end the
    run. Use ``post=True`` when passing a long id list (a GET URL would 414). Any
    non-retryable error status raises.

    ``xml=True`` parses the body **inside** the retry loop and returns the parsed
    root instead of the Response. E-utilities occasionally serves a body that is
    complete at the HTTP level — 200, no ``ChunkedEncodingError`` — yet truncated
    mid-token, and a full study build downloads hundreds of these. Parsed at the
    call site, that ``ParseError`` escapes the retry layer entirely and kills the
    build; observed once in 305 studies, where the same request succeeded on a
    later attempt. Parsing here also keeps it to exactly one parse per response.
    """
    delay = max(sleep, 0.34)
    last = attempts - 1
    for attempt in range(attempts):
        try:
            if post:
                r = _SESSION.post(url, data=params, timeout=60)
            else:
                r = _SESSION.get(url, params=params, timeout=60)
        except _TRANSIENT:
            if attempt == last:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 5.0)
            continue
        if r.status_code in (429, 500, 502, 503, 504) and attempt < last:
            time.sleep(delay)
            delay = min(delay * 2, 5.0)
            continue
        r.raise_for_status()
        if not xml:
            return r
        try:
            return ET.fromstring(r.text)
        except ET.ParseError:
            if attempt == last:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 5.0)


# --------------------------------------------------------------------------- #
# Leaf data classes
# --------------------------------------------------------------------------- #
@dataclass
class FileAlternative:
    """The same file at a different host, with its own access terms.

    Not redundant with :attr:`FileRef.url`: a submitter's original upload often
    has no `url` at all and exists only via requester-pays cloud delivery, so
    collapsing alternatives into the parent would lose the only way to reach it.
    """

    url: str | None = None
    org: str | None = None            # NCBI / AWS / GCP
    access_type: str | None = None    # anonymous / aws identity / Use Cloud Data Delivery
    free_egress: str | None = None    # worldwide / s3.us-east-1 / -


@dataclass
class FileRef:
    url: str | None = None
    md5: str | None = None
    size: int | None = None
    source: str | None = None  # e.g. "sra", "s3", "gs"
    filename: str | None = None
    file_date: str | None = None      # SRAFile@date, raw
    semantic_name: str | None = None
    supertype: str | None = None      # "Original" / "Primary ETL"
    sratoolkit: str | None = None
    alternatives: list[FileAlternative] = field(default_factory=list)


@dataclass
class CloudFile:
    """RUN/CloudFiles/CloudFile — a compact index of cloud copies by provider."""

    provider: str | None = None       # s3 / gs
    location: str | None = None
    filetype: str | None = None


@dataclass
class ReadStat:
    """One read's statistics within a run (Statistics/Read)."""

    index: int | None = None
    count: int | None = None
    average: float | None = None
    stdev: float | None = None


@dataclass
class RunStatistics:
    n_reads: int | None = None
    n_spots: int | None = None
    reads: list[ReadStat] = field(default_factory=list)


@dataclass
class Run:
    """SRR—belongs to exactly one Experiment."""

    accession: str
    total_spots: int | None = None
    total_bases: int | None = None
    published: str | None = None  # release date, e.g. "2017-06-07 10:30:09"
    files: list[FileRef] = field(default_factory=list)
    alias: str | None = None
    is_public: bool | None = None
    size_bytes: int | None = None
    cluster_name: str | None = None
    submitter_id: str | None = None
    statistics: RunStatistics | None = None
    # Bases/Base: {@value: @count} — nucleotide composition of the run.
    base_composition: dict[str, int] = field(default_factory=dict)
    cloud_files: list[CloudFile] = field(default_factory=list)


@dataclass
class Experiment:
    """SRX—references a study + one-or-more samples, owns its runs."""

    accession: str
    sample_ids: list[str] = field(default_factory=list)  # list for Pool support
    title: str | None = None
    library_strategy: str | None = None
    library_source: str | None = None
    library_selection: str | None = None
    library_layout: str | None = None  # SINGLE / PAIRED
    platform: str | None = None
    instrument_model: str | None = None
    attributes: dict[str, str] = field(default_factory=dict)  # EXPERIMENT_ATTRIBUTES
    runs: list[Run] = field(default_factory=list)
    alias: str | None = None
    design_description: str | None = None
    library_name: str | None = None
    library_construction_protocol: str | None = None
    xrefs: dict[str, str] = field(default_factory=dict)  # EXPERIMENT_LINKS
    pool_members: list["PoolMember"] = field(default_factory=list)


@dataclass
class PoolMember:
    """Pool/Member — how reads split across samples in a multiplexed run.

    Present on every experiment, with exactly one member when nothing is pooled.
    It is why :attr:`Experiment.sample_ids` is a list: sample-to-experiment is
    many-to-many in SRA, and collapsing it would lose a pooled study's split.
    """

    accession: str | None = None
    member_name: str | None = None
    sample_name: str | None = None
    sample_title: str | None = None
    organism: str | None = None
    tax_id: str | None = None
    spots: int | None = None
    bases: int | None = None


@dataclass
class Sample:
    """SRS—the biological sample; its attributes are the open EAV bag."""

    accession: str
    biosample: str | None = None  # SAMN cross-ref (BioSample DB)
    taxon_id: str | None = None
    scientific_name: str | None = None
    title: str | None = None
    attributes: dict[str, str] = field(default_factory=dict)  # SAMPLE_ATTRIBUTES
    external_ids: dict[str, str] = field(default_factory=dict)  # e.g. {"GEO": "GSM..."}
    # The BioSample record's own attribute bag, when it has been fetched
    # (:func:`fetch_biosample_attributes`). **Kept apart from `attributes`
    # deliberately.** The two overlap heavily but not completely — measured
    # across five studies, BioSample carried `host_sex`, `sex`,
    # `isolation_source`, `body site`, `is tumor` and `histological type` that
    # the SRA SAMPLE block had dropped. Merging them would erase which archive
    # said what, which is exactly the distinction a benchmark needs, and would
    # silently change what layer 3 is credited with inferring.
    biosample_attributes: dict[str, str] = field(default_factory=dict)
    alias: str | None = None
    xrefs: dict[str, str] = field(default_factory=dict)  # SAMPLE_LINKS
    # The BioSample record's non-attribute fields, when fetched.
    biosample_record: "BioSampleRecord | None" = None


@dataclass
class BioSampleRecord:
    """Everything on a BioSample record except its attribute bag.

    The bag stays on :attr:`Sample.biosample_attributes` so the two attribute
    views (SRA's and BioSample's) sit side by side. `package` is the one most
    worth having: it names the checklist governing the record, which decides
    which attributes were *mandatory* — signal about why one is absent.
    """

    accession: str
    title: str | None = None
    organism_name: str | None = None
    taxonomy_id: str | None = None
    package: str | None = None
    models: list[str] = field(default_factory=list)
    owner: str | None = None
    contact_first: str | None = None
    contact_last: str | None = None
    status: str | None = None
    status_when: str | None = None
    access: str | None = None
    submission_date: str | None = None
    publication_date: str | None = None
    last_update: str | None = None
    ids: dict[str, str] = field(default_factory=dict)
    links: dict[str, str] = field(default_factory=dict)
    # Attribute@harmonized_name — NCBI's normalisation, a separate view of the
    # same data rather than a replacement for the submitter's spelling.
    harmonized: dict[str, str] = field(default_factory=dict)


@dataclass
class Publication:
    id: str
    type: str | None = None  # ePubmed (PMID) / eDOI (DOI)
    # Accessibility class ("oa"/"partial"/"paywall"/"unknown"), filled in on demand
    # by classification and persisted so it need not be recomputed. None = not yet
    # classified (determining it costs a network lookup).
    accessibility_type: str | None = None
    date: str | None = None       # BioProject Publication@date — the one zoned date
    status: str | None = None
    reference: str | None = None
    # Keys into the corpus's shared `papers` map for text retrieved for *this*
    # publication. A list, not a single id, so a second copy of the same article
    # (a repository version alongside the PMC one) can be added without a
    # format change. Usually empty: most publications yield no retrievable text.
    paper_ids: list[str] = field(default_factory=list)
    # Unpaywall's route: gold/hybrid reach PMC, bronze and green do not. Explains
    # why a publication can be genuinely open access and still yield no text.
    oa_status: str | None = None


@dataclass
class Organization:
    """EXPERIMENT_PACKAGE/Organization — who deposited the submission."""

    name: str | None = None
    abbreviation: str | None = None
    org_type: str | None = None
    contact_email: str | None = None
    contact_first: str | None = None
    contact_last: str | None = None


@dataclass
class Submission:
    """The SRA SUBMISSION block. Carries `broker_name`, which maps to a target field."""

    accession: str | None = None
    alias: str | None = None
    center_name: str | None = None
    broker_name: str | None = None
    lab_name: str | None = None
    comment: str | None = None
    organization: Organization | None = None


@dataclass
class BioProjectRecord:
    """The BioProject record beyond its publication list.

    ``uid`` is stored alongside the accession because efetch does not resolve
    accessions — it strips the PRJ?? prefix and treats the digits as an internal
    uid, which coincides for PRJNA and is wrong for PRJEB/PRJDB. Keeping the real
    uid is what makes that mis-resolution detectable later.
    """

    accession: str
    uid: str | None = None
    title: str | None = None
    description: str | None = None
    name: str | None = None
    relevance: str | None = None
    model_organism: str | None = None
    target_organism: str | None = None
    target_taxid: str | None = None
    target_sample_scope: str | None = None
    target_material: str | None = None
    target_capture: str | None = None
    method_type: str | None = None
    data_types: list[str] = field(default_factory=list)
    objectives: list[str] = field(default_factory=list)
    submitting_organization: str | None = None
    submitted: str | None = None
    last_update: str | None = None
    external_links: dict[str, str] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# small XML helpers
# --------------------------------------------------------------------------- #
def _text(el, path):
    node = el.find(path) if el is not None else None
    return node.text.strip() if node is not None and node.text else None


def _attrs_bag(el, container_tag, item_tag):
    """Flatten a *_ATTRIBUTES block of TAG/VALUE pairs into a dict."""
    out: dict[str, str] = {}
    block = el.find(container_tag) if el is not None else None
    if block is None:
        return out
    for item in block.findall(item_tag):
        tag = _text(item, "TAG")
        val = _text(item, "VALUE")
        if tag:
            out[tag] = val
    return out


def _xref_bag(el, container_tag):
    """Flatten a *_LINKS block of XREF_LINK DB/ID pairs into a dict."""
    out: dict[str, str] = {}
    block = el.find(container_tag) if el is not None else None
    if block is None:
        return out
    for xref in block.iter("XREF_LINK"):
        db = _text(xref, "DB")
        ident = _text(xref, "ID")
        if db:
            out[db] = ident
    return out


def _float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# The top-level object
# --------------------------------------------------------------------------- #
class Project:
    """An SRA study (SRP) with all of its samples, experiments and runs.

    Passing just the SRP accession triggers the full fetch + assembly.
    """

    def __init__(
        self,
        srp: str,
        include_samples: bool = True,
        include_experiments: bool = True,
        include_runs: bool = True,
        include_publications: bool = True,
        email: str | None = None,
        api_key: str | None = None,
        sleep: float | None = None,
        batch_size: int = 300,
        max_records: int | None = None,
        _summary_only: bool = False,
    ):
        """Build a Project from just an SRP accession.

        ``max_records`` abandons the build as soon as the record count is known to
        exceed it, leaving a stub carrying only ``accession`` and ``record_count``
        (see :meth:`_build_summary`). Callers that intend to discard oversized
        umbrella studies should pass it rather than filtering afterwards.

        The ``include_*`` flags let you skip the heavy parts you don't need,
        keeping the object (and its JSON) small:

        * ``include_samples``     —the per-sample records + attribute bags
        * ``include_experiments`` —the experiment records (and, with them, runs)
        * ``include_runs``        —the run/file data under each experiment
        * ``include_publications``—the extra BioProject fetch for PMIDs

        ``include_runs`` has no effect when ``include_experiments`` is False
        (runs live under experiments). Note these gate what is *retained*, not
        what is fetched: study/sample/experiment data all arrive in one efetch,
        so skipping them shrinks the object but does not save that download.
        Skipping publications *does* save a network round-trip.
        """
        self.accession = srp
        self.bioproject: str | None = None
        self.title: str | None = None
        self.abstract: str | None = None
        self.study_type: str | None = None
        self.published: str | None = None  # earliest run release date in the study
        self.external_ids: dict[str, str] = {}
        self.samples: dict[str, Sample] = {}  # keyed by SRS (canonical store)
        self.experiments: list[Experiment] = []
        self.publications: list[Publication] = []
        # STUDY@alias / @center_name / CENTER_PROJECT_NAME — stated in the XML and
        # mapping onto target fields the model was otherwise inferring.
        self.study_alias: str | None = None
        self.center_name: str | None = None
        self.center_project_name: str | None = None
        self.xrefs: dict[str, str] = {}          # STUDY_LINKS
        self.submission: Submission | None = None
        self.bioproject_record: BioProjectRecord | None = None
        self.record_count: int | None = None  # # of SRA (experiment) records in the study

        self._study_parsed = False  # STUDY block is repeated in every package
        self._summary_only = _summary_only
        self._include_samples = include_samples
        self._include_experiments = include_experiments
        self._include_runs = include_runs and include_experiments
        self._include_publications = include_publications
        self._email = email
        self._api_key = api_key
        self._sleep = _sleep_for(sleep, api_key)  # pace faster when an api_key is set
        self._batch = batch_size
        self._max_records = max_records
        self.oversized = False  # set when max_records aborted the build

        self._build()

    @classmethod
    def summary(
        cls,
        srp: str,
        include_publications: bool = False,
        email: str | None = None,
        api_key: str | None = None,
        sleep: float | None = None,
        max_records: int | None = None,
    ) -> "Project":
        """Lightweight study-only build for scanning many studies cheaply.

        Fetches only the study's first and last experiment packages (every package
        embeds the full STUDY block), so the download is O(1) in the number of
        runs—three small requests regardless of whether the study has 26 or 26,000
        runs. Populates study-level fields (title, abstract, bioproject,
        study_type, external_ids), ``record_count`` and ``published``; leaves
        samples/experiments/runs empty.

        Pass ``include_publications=True`` to also do the BioProject PMID fetch,
        and ``max_records`` to stop early on oversized studies (``oversized`` is
        then True and only ``record_count`` is populated).
        """
        return cls(
            srp,
            include_samples=False,
            include_experiments=False,
            include_runs=False,
            include_publications=include_publications,
            email=email,
            api_key=api_key,
            sleep=sleep,
            max_records=max_records,
            _summary_only=True,
        )

    # -- networking ------------------------------------------------------- #
    def _common_params(self, **extra):
        # self._email/_api_key are per-instance overrides (usually None); _common
        # falls back to the process-wide credentials when they are None.
        return _common(self._email, self._api_key, **extra)

    def _get(self, endpoint, xml: bool = False, **params):
        return _request_with_retry(
            f"{EUTILS}/{endpoint}",
            sleep=self._sleep,
            xml=xml,
            **self._common_params(**params),
        )

    # -- build pipeline --------------------------------------------------- #
    def _build(self):
        if self._summary_only:
            self._build_summary()
        else:
            self._build_full()
        if self.oversized:
            return  # study is being discarded; don't pay for its BioProject record
        if self._include_publications and self.bioproject:
            self.publications = self._fetch_publications(self.bioproject)

    def _build_full(self):
        uids = self._esearch_uids()
        if not uids:
            raise ValueError(f"No SRA records found for {self.accession}")
        self.record_count = len(uids)
        if self._over_max_records():
            return
        webenv, qkey = self._epost(uids)
        for start in range(0, len(uids), self._batch):
            self._parse_package_set(self._efetch_batch(webenv, qkey, start))
            time.sleep(self._sleep)

    def _build_summary(self):
        # esearch gives the total record count for free; grab a single uid and
        # efetch just that one package to read the shared STUDY block. usehistory
        # lets _note_oldest_published index straight to the far end of the set.
        res = self._get(
            "esearch.fcgi",
            db="sra",
            term=self.accession,
            usehistory="y",
            retmax=1,
            retmode="json",
        ).json()["esearchresult"]
        self.record_count = _int(res.get("count"))
        # esearch has already answered the only question an oversized study needs
        # answered, so stop here. The umbrella studies max_records exists to drop
        # (PulseNet et al.) have BioProject records tens of MB wide that take ~20s
        # and often 500 — fetching one only to discard the study dominated bulk
        # runs. Two small efetches are saved as well.
        if self._over_max_records():
            return
        uids = res.get("idlist", [])
        if not uids:
            raise ValueError(f"No SRA records found for {self.accession}")
        root = self._efetch_ids(uids)
        pkg = next(root.iter("EXPERIMENT_PACKAGE"), None)
        if pkg is not None:
            self._parse_study(pkg.find("STUDY"))
            self._note_published(pkg)
        self._note_oldest_published(res)

    def _over_max_records(self) -> bool:
        """True once ``record_count`` is known to exceed ``max_records``.

        Latches ``oversized`` so the rest of the build can bail out.
        """
        if self._max_records is None or not self.record_count:
            return False
        self.oversized = self.record_count > self._max_records
        return self.oversized

    def _note_oldest_published(self, esearch_result: dict):
        """Read the study's last record so ``published`` is really the earliest date.

        The esearch result set is newest-first, so the uid used by
        :meth:`_build_summary` is the study's *newest* record. Studies that are
        appended to over time (surveillance umbrellas, growing cohorts) can have a
        decade between their first and last release — SRP049009 spans 2014 to 2026 —
        so reading only the newest record reports a date years off. One extra small
        request pins down the real earliest release date.
        """
        count = self.record_count or 0
        webenv = esearch_result.get("webenv")
        qkey = esearch_result.get("querykey")
        if count < 2 or not webenv or not qkey:
            return  # single-record study: its one date is already the earliest
        time.sleep(self._sleep)
        try:
            root = self._efetch_batch(webenv, qkey, count - 1)
        except (requests.RequestException, ET.ParseError):
            return  # keep the date we have rather than failing the whole build
        pkg = next(root.iter("EXPERIMENT_PACKAGE"), None)
        if pkg is not None:
            self._note_published(pkg)

    def _esearch_uids(self) -> list[str]:
        r = self._get(
            "esearch.fcgi",
            db="sra",
            term=self.accession,
            retmax=100000,
            retmode="json",
        )
        return r.json()["esearchresult"]["idlist"]

    def _efetch_ids(self, ids):
        return self._get(
            "efetch.fcgi", db="sra", id=",".join(ids), retmode="xml", xml=True
        )

    def _epost(self, uids):
        # POST, not GET: this is the whole point of epost — the id list is one per
        # record in the study, so a GET URL blows the ~8KB server limit and 414s
        # somewhere around a thousand records (SRP094905, 1800 records, failed).
        root = _request_with_retry(
            f"{EUTILS}/epost.fcgi",
            sleep=self._sleep,
            post=True,
            xml=True,
            **self._common_params(db="sra", id=",".join(uids)),
        )
        return _text(root, "WebEnv"), _text(root, "QueryKey")

    def _efetch_batch(self, webenv, qkey, start):
        return self._get(
            "efetch.fcgi",
            db="sra",
            WebEnv=webenv,
            query_key=qkey,
            retstart=start,
            retmax=self._batch,
            retmode="xml",
            xml=True,
        )

    # -- parsing ---------------------------------------------------------- #
    def _parse_package_set(self, root):
        for pkg in root.iter("EXPERIMENT_PACKAGE"):
            self._parse_study(pkg.find("STUDY"))
            self._parse_submission(pkg.find("SUBMISSION"))
            self._parse_organization(pkg.find("Organization"))
            self._note_published(pkg)  # cheap; independent of the include_* flags
            if self._include_samples:
                sample = self._parse_sample(pkg.find("SAMPLE"))
                if sample:
                    self.samples.setdefault(sample.accession, sample)
            if self._include_experiments:
                self.experiments.append(self._parse_experiment(pkg))

    def _note_published(self, pkg):
        """Track the earliest RUN release date as the study's ``published`` date.

        Checks every RUN in the package, not just the first — a single experiment
        can hold runs released on different days.
        """
        for run in pkg.iter("RUN"):
            released = run.get("published")
            if released and (self.published is None or released < self.published):
                self.published = released

    def _parse_study(self, study_el):
        if study_el is None or self._study_parsed:
            return  # STUDY is repeated in every package; parse once
        self._study_parsed = True
        for ext in study_el.iter("EXTERNAL_ID"):
            ns = ext.get("namespace")
            if ns == "BioProject":
                self.bioproject = ext.text
            elif ns:
                self.external_ids[ns] = ext.text
        descr = study_el.find("DESCRIPTOR")
        self.title = _text(descr, "STUDY_TITLE")
        self.abstract = _text(descr, "STUDY_ABSTRACT")
        st = descr.find("STUDY_TYPE") if descr is not None else None
        if st is not None:
            self.study_type = st.get("existing_study_type") or (st.text or None)
        # These three are stated outright in the XML and map to target fields the
        # model was otherwise paying to guess: study_alias, project_name and
        # center_name. The audit found `project_name` scored "high" on 44/44
        # records with zero of them quoted — it was inventing what is written here.
        self.study_alias = study_el.get("alias")
        self.center_name = study_el.get("center_name")
        self.center_project_name = _text(descr, "CENTER_PROJECT_NAME")
        self.xrefs.update(_xref_bag(study_el, "STUDY_LINKS"))

    def _parse_submission(self, sub_el):
        if sub_el is None or self.submission is not None:
            return  # repeated in every package, like STUDY
        self.submission = Submission(
            accession=sub_el.get("accession"),
            alias=sub_el.get("alias"),
            center_name=sub_el.get("center_name"),
            broker_name=sub_el.get("broker_name"),
            lab_name=sub_el.get("lab_name") or None,
            comment=sub_el.get("submission_comment"),
        )

    def _parse_organization(self, org_el):
        if org_el is None or self.submission is None:
            return
        if self.submission.organization is not None:
            return
        name_el = org_el.find("Name")
        contact = org_el.find("Contact")
        self.submission.organization = Organization(
            name=(name_el.text or "").strip() or None if name_el is not None else None,
            abbreviation=name_el.get("abbr") if name_el is not None else None,
            org_type=org_el.get("type"),
            contact_email=contact.get("email") if contact is not None else None,
            contact_first=_text(contact, "Name/First"),
            contact_last=_text(contact, "Name/Last"),
        )

    def _parse_sample(self, sample_el) -> Sample | None:
        if sample_el is None:
            return None
        s = Sample(accession=sample_el.get("accession"))
        s.title = _text(sample_el, "TITLE")
        s.taxon_id = _text(sample_el, "SAMPLE_NAME/TAXON_ID")
        s.scientific_name = _text(sample_el, "SAMPLE_NAME/SCIENTIFIC_NAME")
        for ext in sample_el.iter("EXTERNAL_ID"):
            ns = ext.get("namespace")
            if ns == "BioSample":
                s.biosample = ext.text
            elif ns:
                s.external_ids[ns] = ext.text
        s.attributes = _attrs_bag(sample_el, "SAMPLE_ATTRIBUTES", "SAMPLE_ATTRIBUTE")
        s.alias = sample_el.get("alias")
        s.xrefs = _xref_bag(sample_el, "SAMPLE_LINKS")
        return s

    def _parse_experiment(self, pkg) -> Experiment:
        exp_el = pkg.find("EXPERIMENT")
        e = Experiment(accession=exp_el.get("accession"))
        e.title = _text(exp_el, "TITLE")

        e.alias = exp_el.get("alias")
        e.design_description = _text(exp_el, "DESIGN/DESIGN_DESCRIPTION")

        lib = exp_el.find("DESIGN/LIBRARY_DESCRIPTOR")
        if lib is not None:
            e.library_strategy = _text(lib, "LIBRARY_STRATEGY")
            e.library_source = _text(lib, "LIBRARY_SOURCE")
            e.library_selection = _text(lib, "LIBRARY_SELECTION")
            e.library_name = _text(lib, "LIBRARY_NAME")
            # Maps to the library_construction_protocol target field; the audit
            # flagged the model quoting this one wrong when it could not see it.
            e.library_construction_protocol = _text(
                lib, "LIBRARY_CONSTRUCTION_PROTOCOL")
            layout = lib.find("LIBRARY_LAYOUT")
            if layout is not None and len(layout):
                e.library_layout = layout[0].tag  # SINGLE / PAIRED

        platform = exp_el.find("PLATFORM")
        if platform is not None and len(platform):
            e.platform = platform[0].tag
            e.instrument_model = _text(platform[0], "INSTRUMENT_MODEL")

        e.attributes = _attrs_bag(
            exp_el, "EXPERIMENT_ATTRIBUTES", "EXPERIMENT_ATTRIBUTE"
        )
        e.xrefs = _xref_bag(exp_el, "EXPERIMENT_LINKS")
        e.sample_ids = self._sample_ids(exp_el, pkg)
        e.pool_members = self._parse_pool(pkg.find("Pool"))
        e.runs = self._parse_runs(pkg.find("RUN_SET")) if self._include_runs else []
        return e

    @staticmethod
    def _parse_pool(pool_el) -> list[PoolMember]:
        if pool_el is None:
            return []
        return [
            PoolMember(
                accession=m.get("accession"),
                member_name=m.get("member_name"),
                sample_name=m.get("sample_name"),
                sample_title=m.get("sample_title"),
                organism=m.get("organism"),
                tax_id=m.get("tax_id"),
                spots=_int(m.get("spots")),
                bases=_int(m.get("bases")),
            )
            for m in pool_el.findall("Member")
        ]

    @staticmethod
    def _sample_ids(exp_el, pkg) -> list[str]:
        ids: list[str] = []
        sd = exp_el.find("DESIGN/SAMPLE_DESCRIPTOR")
        if sd is not None and sd.get("accession"):
            ids.append(sd.get("accession"))
        # Pool members (multiplexed libraries), from either the descriptor or package
        for src in (sd, pkg.find("Pool")):
            if src is None:
                continue
            for m in src.iter("Member"):
                if m.get("accession"):
                    ids.append(m.get("accession"))
        # dedupe, preserve order
        seen: set[str] = set()
        return [i for i in ids if not (i in seen or seen.add(i))]

    @staticmethod
    def _parse_runs(run_set_el) -> list[Run]:
        runs: list[Run] = []
        if run_set_el is None:
            return runs
        for run_el in run_set_el.findall("RUN"):
            r = Run(accession=run_el.get("accession"))
            r.total_spots = _int(run_el.get("total_spots"))
            r.total_bases = _int(run_el.get("total_bases"))
            r.published = run_el.get("published")
            r.alias = run_el.get("alias")
            r.cluster_name = run_el.get("cluster_name")
            r.size_bytes = _int(run_el.get("size"))
            pub = run_el.get("is_public")
            r.is_public = None if pub is None else pub.lower() == "true"
            r.submitter_id = _text(run_el, "IDENTIFIERS/SUBMITTER_ID")

            stats_el = run_el.find("Statistics")
            if stats_el is not None:
                stats = RunStatistics(
                    n_reads=_int(stats_el.get("nreads")),
                    n_spots=_int(stats_el.get("nspots")),
                )
                for read in stats_el.findall("Read"):
                    stats.reads.append(ReadStat(
                        index=_int(read.get("index")),
                        count=_int(read.get("count")),
                        average=_float(read.get("average")),
                        stdev=_float(read.get("stdev")),
                    ))
                r.statistics = stats

            bases_el = run_el.find("Bases")
            if bases_el is not None:
                for b in bases_el.findall("Base"):
                    if b.get("value"):
                        r.base_composition[b.get("value")] = _int(b.get("count"))

            sra_files = run_el.find("SRAFiles")
            if sra_files is not None:
                for f in sra_files.findall("SRAFile"):
                    ref = FileRef(
                        url=f.get("url"),
                        md5=f.get("md5"),
                        size=_int(f.get("size")),
                        source=f.get("cluster") or f.get("semantic_name"),
                        filename=f.get("filename"),
                        file_date=f.get("date"),
                        semantic_name=f.get("semantic_name"),
                        supertype=f.get("supertype"),
                        sratoolkit=f.get("sratoolkit"),
                    )
                    for alt in f.findall("Alternatives"):
                        ref.alternatives.append(FileAlternative(
                            url=alt.get("url"),
                            org=alt.get("org"),
                            access_type=alt.get("access_type"),
                            free_egress=alt.get("free_egress"),
                        ))
                    r.files.append(ref)

            cloud_el = run_el.find("CloudFiles")
            if cloud_el is not None:
                for c in cloud_el.findall("CloudFile"):
                    r.cloud_files.append(CloudFile(
                        provider=c.get("provider"),
                        location=c.get("location"),
                        filetype=c.get("filetype"),
                    ))
            runs.append(r)
        return runs

    def _bioproject_uid(self, bioproject: str) -> str | None:
        """Resolve a BioProject accession to the UID ``efetch`` actually needs.

        ``efetch db=bioproject`` does not look accessions up — it strips the
        ``PRJ??`` prefix and uses the remaining digits as a raw internal UID
        (``id=PRJEB47383`` and ``id=47383`` return byte-identical responses). For
        NCBI-archived projects that is correct by construction, since the accession
        is *formed from* the UID: PRJNA646996 is UID 646996. EBI and DDBJ number
        their accessions independently, so PRJEB47383 (really UID 778158) fetches
        unrelated UID 47383 — usually a non-public record, which reads as "no
        publications", but sometimes a real project whose papers then get attached
        to the wrong study. Those have to go through esearch.
        """
        if bioproject.startswith("PRJNA"):
            return bioproject  # accession is the uid; skip the extra round-trip
        try:
            # Field-qualified: a bare term is relevance-ranked over all fields and
            # can rank another project first (PRJEB1787 returns 3 hits, its own
            # record third). [Project Accession] matches exactly one.
            res = self._get(
                "esearch.fcgi",
                db="bioproject",
                term=f"{bioproject}[Project Accession]",
                retmax=1,
                retmode="json",
            ).json()["esearchresult"]
        except (requests.RequestException, ValueError, KeyError):
            return None
        time.sleep(self._sleep)
        uids = res.get("idlist") or []
        return uids[0] if uids else None

    def _fetch_publications(self, bioproject: str) -> list[Publication]:
        uid = self._bioproject_uid(bioproject)
        if uid is None:
            return []
        try:
            root = self._get(
                "efetch.fcgi", db="bioproject", id=uid, retmode="xml", xml=True
            )
        except (requests.RequestException, ET.ParseError):
            return []
        # Belt-and-braces against the misresolution above: if the record names
        # projects and ours isn't among them, it describes a different study, so
        # its publications are not ours to claim.
        declared = {
            a.get("accession") for a in root.iter("ArchiveID") if a.get("accession")
        }
        if declared and bioproject not in declared:
            return []
        pubs = []
        seen: set[str] = set()
        for p in root.iter("Publication"):
            pid = p.get("id")
            # BioProject records sometimes list the same paper twice (SRP049009
            # repeats PMID 25999578); dedupe so it isn't classified twice.
            if pid and pid not in seen:
                seen.add(pid)
                pubs.append(Publication(id=pid, type=_text(p, "DbType")))
        return pubs

    # -- convenience accessors ------------------------------------------- #
    def samples_of(self, exp: Experiment) -> list[Sample]:
        return [self.samples[sid] for sid in exp.sample_ids if sid in self.samples]

    def runs_of_sample(self, srs: str) -> list[Run]:
        return [r for e in self.experiments if srs in e.sample_ids for r in e.runs]

    def publication_classes(
        self, sleep: float | None = None, refresh: bool = False
    ) -> list[str]:
        """Distinct accessibility classes of this study's publications.

        Classifies the publications already on the object (see
        :func:`classify_publication`) — no BioProject re-resolution. The result is
        **cached** on each ``Publication.accessibility_type``, so once classified
        (and saved via :meth:`to_dict`/:meth:`to_json`) a reloaded study returns
        instantly with no network calls. Pass ``refresh=True`` to re-classify
        (e.g. if a paper's OA status may have changed). Empty list if no paper.

        Filter a loaded list of studies with a comprehension::

            oa = [p for p in studies if "oa" in p.publication_classes()]
        """
        out: list[str] = []
        for pub in self.publications:
            if pub.accessibility_type is None or refresh:
                pub.accessibility_type = classify_publication(pub.id, sleep=sleep)
            if pub.accessibility_type not in out:
                out.append(pub.accessibility_type)
        return out

    def to_dataframe(self):
        """Flatten back to one row per run (the pysradb-style square table).

        A *pooled* experiment (multiplexed library, several SRS under one SRX)
        yields one row per run **per sample** rather than silently reporting only
        the first sample — otherwise the other samples' attribute bags, the reason
        ``Experiment.sample_ids`` is a list, would be dropped. Non-pooled studies
        (the overwhelming majority) are unaffected: one sample means one row per run.
        """
        import pandas as pd

        rows = []
        for e in self.experiments:
            # no resolvable sample (e.g. include_samples=False) -> one blank row
            for s in self.samples_of(e) or [Sample(accession=None)]:
                for run in e.runs:
                    rows.append(
                        {
                            "study_accession": self.accession,
                            "bioproject": self.bioproject,
                            "experiment_accession": e.accession,
                            "library_strategy": e.library_strategy,
                            "library_selection": e.library_selection,
                            "library_layout": e.library_layout,
                            "platform": e.platform,
                            "instrument_model": e.instrument_model,
                            "sample_accession": s.accession,
                            "biosample": s.biosample,
                            "run_accession": run.accession,
                            "run_total_spots": run.total_spots,
                            "run_total_bases": run.total_bases,
                            **{f"sample.{k}": v for k, v in s.attributes.items()},
                        }
                    )
        return pd.DataFrame(rows)

    def to_dict(self) -> dict:
        """Nested dict mirroring the object exactly (no flattening / reshaping)."""
        from dataclasses import asdict

        return {
            "accession": self.accession,
            "bioproject": self.bioproject,
            "title": self.title,
            "abstract": self.abstract,
            "study_type": self.study_type,
            "published": self.published,
            "record_count": self.record_count,
            "external_ids": self.external_ids,
            "study_alias": self.study_alias,
            "center_name": self.center_name,
            "center_project_name": self.center_project_name,
            "xrefs": self.xrefs,
            "submission": asdict(self.submission) if self.submission else None,
            "bioproject_record": (asdict(self.bioproject_record)
                                  if self.bioproject_record else None),
            "samples": {k: asdict(v) for k, v in self.samples.items()},
            "experiments": [asdict(e) for e in self.experiments],
            "publications": [asdict(p) for p in self.publications],
        }

    def to_json(self, path: str | None = None, indent: int = 2) -> str:
        """Serialize the whole Project to JSON, preserving its exact shape.

        With ``path`` -> writes the file and returns the path.
        Without ``path`` -> returns the JSON string (e.g. to print).
        """
        text = json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
        if path is None:
            return text
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        return path

    @classmethod
    def from_dict(cls, d: dict) -> "Project":
        """Rebuild a Project from a :meth:`to_dict` mapping — no network calls.

        Reconstructs the nested Sample/Experiment/Run/FileRef/Publication
        dataclasses. Missing keys default sensibly, so summary-mode dicts (empty
        samples/experiments) and older JSON without ``record_count`` both load.
        """

        def _file(fd):
            fd = dict(fd)
            alts = [FileAlternative(**a) for a in fd.pop("alternatives", []) or []]
            return FileRef(**fd, alternatives=alts)

        def _run(rd):
            rd = dict(rd)
            files = [_file(f) for f in rd.pop("files", []) or []]
            clouds = [CloudFile(**c) for c in rd.pop("cloud_files", []) or []]
            stats = rd.pop("statistics", None)
            if stats:
                stats = dict(stats)
                reads = [ReadStat(**r) for r in stats.pop("reads", []) or []]
                stats = RunStatistics(**stats, reads=reads)
            return Run(**rd, files=files, cloud_files=clouds, statistics=stats)

        def _experiment(ed):
            ed = dict(ed)
            runs = [_run(r) for r in ed.pop("runs", []) or []]
            pool = [PoolMember(**m) for m in ed.pop("pool_members", []) or []]
            return Experiment(**ed, runs=runs, pool_members=pool)

        def _sample(sd):
            sd = dict(sd)
            rec = sd.pop("biosample_record", None)
            return Sample(**sd, biosample_record=BioSampleRecord(**rec) if rec else None)

        p = cls.__new__(cls)  # bypass __init__ so nothing is fetched
        p.accession = d["accession"]
        p.bioproject = d.get("bioproject")
        p.title = d.get("title")
        p.abstract = d.get("abstract")
        p.study_type = d.get("study_type")
        p.published = d.get("published")
        p.record_count = d.get("record_count")
        p.external_ids = dict(d.get("external_ids") or {})
        p.study_alias = d.get("study_alias")
        p.center_name = d.get("center_name")
        p.center_project_name = d.get("center_project_name")
        p.xrefs = dict(d.get("xrefs") or {})
        sub = d.get("submission")
        if sub:
            sub = dict(sub)
            org = sub.pop("organization", None)
            p.submission = Submission(**sub,
                                      organization=Organization(**org) if org else None)
        else:
            p.submission = None
        bpr = d.get("bioproject_record")
        p.bioproject_record = BioProjectRecord(**bpr) if bpr else None
        p.samples = {k: _sample(v) for k, v in (d.get("samples") or {}).items()}
        p.experiments = [_experiment(e) for e in (d.get("experiments") or [])]
        p.publications = [Publication(**pub) for pub in (d.get("publications") or [])]
        # networking config: no per-instance override -> use process-wide creds,
        # including their pacing (an api_key allows 10 req/s rather than 3).
        p._email = None
        p._api_key = None
        p._sleep = _sleep_for(None, None)
        p._batch = 300
        p._max_records = None
        p.oversized = False  # a saved study was, by definition, not discarded
        # build-time state: nothing was fetched, and the study block is already set
        p._study_parsed = p.title is not None
        p._summary_only = not (p.samples or p.experiments)
        p._include_samples = bool(p.samples)
        p._include_experiments = bool(p.experiments)
        p._include_runs = any(e.runs for e in p.experiments)
        p._include_publications = bool(p.publications)
        return p

    @classmethod
    def from_json(cls, source) -> "Project":
        """Rebuild one Project from a JSON file path, JSON string, or parsed dict."""
        return cls.from_dict(_load_json(source))

    def __repr__(self):
        return (
            f"Project({self.accession!r}, bioproject={self.bioproject!r}, "
            f"record_count={self.record_count}, published={self.published!r}, "
            f"samples={len(self.samples)}, experiments={len(self.experiments)}, "
            f"runs={sum(len(e.runs) for e in self.experiments)}, "
            f"publications={[p.id for p in self.publications]})"
        )


def _eutils(endpoint, sleep: float = 0.34, post: bool = False, **params):
    """Module-level E-utilities call (see :func:`_request_with_retry`)."""
    return _request_with_retry(
        f"{EUTILS}/{endpoint}", sleep=sleep, post=post, **params
    )


# Study accession (SRP/ERP/DRP) inside each esummary ExpXml blob. Regex, not XML,
# because ExpXml titles can contain stray '&' that break a re-parse.
_STUDY_ACC_RE = re.compile(r'<Study\b[^>]*\bacc="([^"]+)"')


def _srps_from_esummary(root) -> list[str]:
    # ``root`` is the outer esummary envelope, already parsed by the retry layer.
    # Only the nested ExpXml blobs need the regex above.
    out = []
    for item in root.iter("Item"):
        if item.get("Name") == "ExpXml" and item.text:
            m = _STUDY_ACC_RE.search(item.text)
            if m:
                out.append(m.group(1))
    return out


def fetch_biosample_attributes(
    biosample_ids, *, batch: int = 300, sleep: float | None = None,
    email=None, api_key=None, progress=None,
) -> dict[str, dict[str, str]]:
    """``{SAMN accession: {attribute: value}}`` straight from the BioSample DB.

    The SRA SAMPLE block mirrors *most* of a BioSample record, so this is not a
    replacement for :attr:`Sample.attributes` — it is the part SRA drops.
    Measured over five studies from three archives, four carried extra fields
    and the extras were the ones this project most wants: ``host_sex``, ``sex``,
    ``isolation_source``, ``body site``, ``is tumor``, ``histological type``.
    (One study matched exactly, so the gain is real but not universal.) Many of
    the rest are dbGaP registry artefacts — ``gap_consent_code``,
    ``submitted_subject_id`` — which are bookkeeping rather than biology.

    **Batched, because per-sample fetching would dominate a corpus build.** At
    300 ids per request the whole 102,227-record reference corpus costs minutes
    rather than hours; one id at a time measured 8 ms/record even on a warm
    connection, and that is before per-request pacing.

    ``progress(done, total)`` is called after each batch. Ids that BioSample
    does not return are simply absent from the result rather than raising —
    a withdrawn or suppressed sample must not end a corpus build.
    """
    return {acc: attrs for acc, (attrs, _rec)
            in fetch_biosample_records(biosample_ids, batch=batch, sleep=sleep,
                                       email=email, api_key=api_key,
                                       progress=progress).items()}


def fetch_biosample_records(
    biosample_ids, *, batch: int = 300, sleep: float | None = None,
    email=None, api_key=None, progress=None,
) -> dict[str, tuple[dict[str, str], "BioSampleRecord"]]:
    """``{SAMN: (attributes, BioSampleRecord)}`` — the bag and everything else.

    Split because the two are used differently: the attribute bag sits beside
    SRA's own bag on the sample, while the record's other fields (package,
    owner, status, dates) describe the deposit rather than the biology.

    Same batching and same accession-prefix defence as described above.
    """
    ids = [i for i in dict.fromkeys(biosample_ids) if i]
    wanted = set(ids)
    out: dict[str, tuple[dict[str, str], BioSampleRecord]] = {}
    pause = _sleep_for(sleep, api_key)
    for start in range(0, len(ids), batch):
        chunk = ids[start:start + batch]
        # Accession -> uid first, then fetch by uid. **Fetching by accession is
        # not safe**: efetch strips the archive prefix and treats the digits as
        # an internal uid, which coincides only for SAMN. Asking for DDBJ's
        # SAMD00041293 that way returns NCBI's SAMN00041293 — a different sample
        # from a different study, 200 OK, no warning; a honey-bee request came
        # back with Human Microbiome Project metadata. esearch resolves all three
        # archives correctly, which is why there is no separate EBI path here.
        uid_root = _request_with_retry(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            sleep=pause, post=True,
            **_common(email, api_key, db="biosample", retmode="json",
                      retmax=len(chunk),
                      term=" OR ".join(f"{a}[Accession]" for a in chunk)),
        )
        uids = uid_root.json().get("esearchresult", {}).get("idlist") or []
        time.sleep(pause)
        if uids:
            # POST: a 300-id GET URL runs past what E-utilities accepts.
            root = _request_with_retry(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                sleep=pause, post=True, xml=True,
                **_common(email, api_key, db="biosample", id=",".join(uids),
                          rettype="full", retmode="xml"),
            )
            for doc in root.iter("BioSample"):
                acc = doc.get("accession")
                # Key on the *returned* accession and keep only what was asked
                # for. esearch can rank in an order unrelated to the query, so
                # never zip(chunk, docs) — and this is the second line of defence
                # against the prefix collapse described above.
                if not acc or acc not in wanted:
                    continue
                bag, harmonized = {}, {}
                for attr in doc.iter("Attribute"):
                    # `attribute_name` is the submitter's spelling;
                    # `harmonized_name` is NCBI's normalisation. Keep both —
                    # they are different views, not alternatives.
                    value = (attr.text or "").strip()
                    if not value:
                        continue
                    if attr.get("attribute_name"):
                        bag[attr.get("attribute_name")] = value
                    if attr.get("harmonized_name"):
                        harmonized[attr.get("harmonized_name")] = value
                status = doc.find("Status")
                org = doc.find("Description/Organism")
                owner = doc.find("Owner")
                contact = doc.find("Owner/Contacts/Contact")
                pkg_el = doc.find("Package")
                rec = BioSampleRecord(
                    accession=acc,
                    title=_text(doc, "Description/Title"),
                    organism_name=(org.get("taxonomy_name") if org is not None else None)
                    or _text(org, "OrganismName"),
                    taxonomy_id=org.get("taxonomy_id") if org is not None else None,
                    package=(pkg_el.get("display_name") or (pkg_el.text or "").strip()
                             if pkg_el is not None else None),
                    models=[(m.text or "").strip() for m in doc.iter("Model")
                            if (m.text or "").strip()],
                    owner=_text(owner, "Name"),
                    contact_first=_text(contact, "Name/First"),
                    contact_last=_text(contact, "Name/Last"),
                    status=status.get("status") if status is not None else None,
                    status_when=status.get("when") if status is not None else None,
                    access=doc.get("access"),
                    submission_date=doc.get("submission_date"),
                    publication_date=doc.get("publication_date"),
                    last_update=doc.get("last_update"),
                    ids={i.get("db"): (i.text or "").strip()
                         for i in doc.iter("Id") if i.get("db")},
                    links={l.get("type") or l.get("target") or "link":
                           (l.get("label") or (l.text or "").strip())
                           for l in doc.iter("Link")},
                    harmonized=harmonized,
                )
                out[acc] = (bag, rec)
            time.sleep(pause)
        if progress:
            progress(min(start + batch, len(ids)), len(ids))

    return out


def fetch_bioproject_record(
    accession: str, sleep: float | None = None, email=None, api_key=None
):
    """``(BioProjectRecord, [Publication])`` for a BioProject accession, or None.

    Resolves accession -> uid via esearch before fetching, because
    ``efetch db=bioproject`` does **not** resolve accessions: it strips the
    ``PRJ??`` prefix and uses the digits as an internal uid. That coincides for
    PRJNA (PRJNA646996 really is uid 646996) and is wrong for PRJEB/PRJDB —
    PRJEB13694 fetched by accession returns PRJNA13694, an unrelated project.
    The returned record's own ArchiveID is checked against what was asked for.
    """
    pause = _sleep_for(sleep, api_key)
    try:
        r = _request_with_retry(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi", sleep=pause,
            **_common(email, api_key, db="bioproject",
                      term=f"{accession}[Project Accession]", retmode="json"))
        uids = r.json().get("esearchresult", {}).get("idlist") or []
    except Exception:  # noqa: BLE001
        return None
    if not uids:
        return None
    time.sleep(pause)
    try:
        root = _request_with_retry(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            sleep=pause, xml=True,
            **_common(email, api_key, db="bioproject", id=uids[0], retmode="xml"))
    except Exception:  # noqa: BLE001
        return None

    archive_id = root.find(".//Project/ProjectID/ArchiveID")
    if archive_id is None or archive_id.get("accession") != accession:
        return None                      # mis-resolved; better nothing than wrong
    descr = root.find(".//Project/ProjectDescr")
    target = root.find(".//ProjectTypeSubmission/Target")
    method = root.find(".//ProjectTypeSubmission/Method")
    sub = root.find(".//Submission")
    org = root.find(".//Submission/Description/Organization/Name")
    organism = root.find(".//ProjectTypeSubmission/Target/Organism")

    rec = BioProjectRecord(
        accession=accession,
        uid=archive_id.get("id") or uids[0],
        title=_text(descr, "Title"),
        description=_text(descr, "Description"),
        name=_text(descr, "Name"),
        relevance=(", ".join((e.tag for e in descr.find("Relevance")))
                   if descr is not None and descr.find("Relevance") is not None else None),
        model_organism=_text(descr, "Relevance/ModelOrganism"),
        target_organism=_text(organism, "OrganismName"),
        target_taxid=organism.get("taxID") if organism is not None else None,
        target_sample_scope=target.get("sample_scope") if target is not None else None,
        target_material=target.get("material") if target is not None else None,
        target_capture=target.get("capture") if target is not None else None,
        method_type=method.get("method_type") if method is not None else None,
        data_types=[(d.text or "").strip() for d in root.iter("DataType")
                    if (d.text or "").strip()],
        objectives=[d.get("data_type") for d in root.iter("Data") if d.get("data_type")],
        submitting_organization=(org.text or "").strip() if org is not None else None,
        submitted=sub.get("submitted") if sub is not None else None,
        last_update=sub.get("last_update") if sub is not None else None,
        external_links={x.get("db"): _text(x, "ID")
                        for x in root.iter("dbXREF") if x.get("db")},
    )
    pubs = []
    for p in root.iter("Publication"):
        if not p.get("id"):
            continue
        pubs.append(Publication(
            id=p.get("id"),
            type=_text(p, "DbType"),
            date=p.get("date"),
            status=p.get("status"),
            reference=_text(p, "Reference"),
        ))
    return rec, pubs


def classify_publication(pub_id, sleep: float | None = None, email=None, api_key=None) -> str:
    """Classify a paper's public accessibility from a PMID or DOI.

    ``pub_id`` may be a PMID (digits) or a DOI (contains ``/``). Returns one of:

    * ``"oa"``      — open-access full text: in the PMC Open Access subset
      (downloadable, license-tagged), flagged open in Europe PMC, or reported free
      by Unpaywall (publisher- or repository-hosted, outside PMC).
    * ``"partial"`` — full text is in PMC but *not* OA-licensed: readable on the
      web, but not downloadable / not text-mineable (e.g. author manuscripts).
    * ``"paywall"`` — the article is indexed but only its abstract is public.
    * ``"unknown"`` — the id could not be resolved anywhere.

    Europe PMC gives existence + PMCID + the open flag in one call; when a PMCID
    exists the PMC OA service is the authoritative check for downloadable OA. PMC
    indexing is *not* the same thing as open access, though, so a DOI that PMC
    doesn't know or doesn't flag open gets a second opinion from Unpaywall (see
    :func:`_unpaywall_is_oa`) before being written off.

    ``email``/``api_key`` default to the process-wide credentials (see
    :func:`set_entrez_credentials`). They are sent only to NCBI hosts — Europe PMC
    and Unpaywall are third parties with no use for an NCBI api_key.
    """
    pub_id = str(pub_id).strip()
    sleep = _sleep_for(sleep, api_key)
    is_doi = "/" in pub_id
    query = f'DOI:"{pub_id}"' if is_doi else f"EXT_ID:{pub_id} AND SRC:MED"
    hits = (
        _request_with_retry(
            EPMC_SEARCH, sleep=sleep, query=query, format="json", resultType="core"
        )
        .json()
        .get("resultList", {})
        .get("result", [])
    )
    if not hits:
        # Not indexed in PubMed/Europe PMC. Plenty of fully OA venues aren't
        # (Frontiers' non-biomedical titles, preprint servers, much of MDPI), so
        # don't equate "unindexed" with "unavailable" when we have a DOI to check.
        if is_doi and _unpaywall_is_oa(pub_id, sleep=sleep, email=email):
            return "oa"
        return "unknown"
    art = hits[0]
    pmcid = art.get("pmcid")
    is_oa = art.get("isOpenAccess") == "Y"
    if pmcid:
        root = _request_with_retry(
            PMC_OA, sleep=sleep, id=pmcid, xml=True, **_common(email, api_key)
        )
        if root.find(".//record") is not None:
            return "oa"  # PMC Open Access subset — authoritative
        return "oa" if is_oa else "partial"
    if is_oa:
        return "oa"
    # Indexed, no PMC copy, not flagged open — the publisher or a repository may
    # still host it freely (e.g. 10.1111/age.13334, free at Wiley but not in PMC).
    doi = art.get("doi") or (pub_id if is_doi else None)
    if doi and _unpaywall_is_oa(doi, sleep=sleep, email=email):
        return "oa"
    return "paywall"


# Sections worth sending to a model reconstructing sample metadata. Methods is
# where the material, the strains, the treatments and the collection details
# live; results and discussion are about findings, not provenance, and cost the
# same per token.
_METHODS_RE = re.compile(r"method|material|experimental|procedure", re.I)


def fetch_open_access_text(
    pub_id, max_chars: int = 30000, sleep: float | None = None, email=None, api_key=None
) -> str | None:
    """Plain text of an open-access paper, or None if it cannot be retrieved.

    Resolves a PMID or DOI to a PMCID through Europe PMC, then pulls the JATS
    full text from its REST service. Returns None — never raises — when the
    paper is not in Europe PMC, has no PMC copy, or is not in the open subset;
    a study without a retrievable paper is a normal outcome, not an error.

    **This is the expensive input in the pipeline.** A full paper runs tens of
    thousands of tokens, against ~540 for a sample's archive evidence, so the
    text is trimmed rather than sent whole: title, abstract, then Methods-like
    sections, then whatever else fits under ``max_chars``. Methods is where
    sample provenance actually lives; results and discussion bill the same per
    token and describe findings instead. Widen ``max_chars`` deliberately, not
    by default.
    """
    pub_id = str(pub_id).strip()
    sleep = _sleep_for(sleep, api_key)
    is_doi = "/" in pub_id
    query = f'DOI:"{pub_id}"' if is_doi else f"EXT_ID:{pub_id} AND SRC:MED"
    try:
        hits = (
            _request_with_retry(
                EPMC_SEARCH, sleep=sleep, query=query, format="json", resultType="core"
            )
            .json()
            .get("resultList", {})
            .get("result", [])
        )
    except (requests.RequestException, ValueError):
        return None
    if not hits:
        return None
    pmcid = hits[0].get("pmcid")
    if not pmcid:
        return None  # indexed but no PMC copy — nothing to fetch full text from
    try:
        root = _request_with_retry(
            f"{EPMC_REST}/{pmcid}/fullTextXML", sleep=sleep, xml=True
        )
    except (requests.RequestException, ET.ParseError):
        return None
    return _jats_text(root, max_chars)


def publication_oa_status(
    pub_id, sleep: float | None = None, email=None, api_key=None
) -> str | None:
    """Unpaywall's open-access *route* for a PMID or DOI, or None.

    ``bronze`` (free at the publisher, no licence) and ``green`` (repository
    copy) almost never reach Europe PMC, which is the only source
    :func:`fetch_open_access_text` reads; ``gold``/``hybrid`` usually do. See
    :func:`_unpaywall_status` for the measured breakdown — the signal is strong
    but one-directional.

    Two of the 54 unretrievable papers come back ``closed``, meaning Unpaywall
    sees no free copy at all. Those are candidates for a mis-classification by
    our own publication classifier rather than a retrieval-route problem.

    Resolves a PMID to its DOI through Europe PMC first, because Unpaywall is
    DOI-only. Returns None when the DOI cannot be found or the lookup fails.
    """
    sleep = _sleep_for(sleep, api_key)
    doi = pub_id if "/" in pub_id else None
    if doi is None:
        try:
            hits = (
                _request_with_retry(
                    EPMC_SEARCH, sleep=sleep, query=f"EXT_ID:{pub_id} AND SRC:MED",
                    format="json", resultType="core")
                .json().get("resultList", {}).get("result", [])
            )
        except (requests.RequestException, ValueError):
            return None
        doi = hits[0].get("doi") if hits else None
    if not doi:
        return None
    return _unpaywall_status(doi, sleep=sleep, email=email)[1]


def _jats_text(root, max_chars: int) -> str | None:
    """Flatten a JATS article to plain text, Methods-first, under `max_chars`."""

    def flat(el):
        return " ".join("".join(el.itertext()).split()) if el is not None else ""

    parts: list[str] = []
    title = flat(root.find(".//article-title"))
    if title:
        parts.append(f"TITLE: {title}")
    abstract = flat(root.find(".//abstract"))
    if abstract:
        parts.append(f"ABSTRACT: {abstract}")

    body = root.find(".//body")
    methods, other = [], []
    for sec in body.findall("sec") if body is not None else []:
        heading = flat(sec.find("title"))
        text = flat(sec)
        if not text:
            continue
        (methods if _METHODS_RE.search(heading) else other).append(text)

    budget = max_chars - sum(len(p) for p in parts)
    for text in methods + other:  # Methods first, so a tight budget keeps it
        if budget <= 0:
            break
        parts.append(text[:budget])
        budget -= len(text)
    joined = "\n\n".join(parts).strip()
    return joined or None


def _unpaywall_is_oa(doi: str, sleep: float = 0.34, email: str | None = None) -> bool:
    """True if Unpaywall reports a free full-text copy of ``doi``.

    Covers the OA that PMC can't see: publisher-hosted (gold/hybrid) and
    repository-hosted (green) copies of articles outside PubMed's index. Unpaywall
    requires a contact email, so this is skipped—returning False—when neither an
    explicit ``email`` nor a process-wide one is set (see
    :func:`set_entrez_credentials`). Any lookup failure is treated as "not proven
    OA" rather than raising: this is a best-effort second opinion.
    """
    return _unpaywall_status(doi, sleep=sleep, email=email)[0]


def _unpaywall_status(doi: str, sleep: float = 0.34, email: str | None = None):
    """``(is_oa, oa_status)`` from Unpaywall — the route, not just the verdict.

    ``oa_status`` is Unpaywall's classification of *how* an article is open.
    Measured against retrieval on the reference corpus (all 54 papers that
    yielded no text, plus 30 sampled from the 272 that did):

        bronze   0 with text / 20 without
        green    2 / 19
        gold    20 /  6
        hybrid   8 /  7
        closed   0 /  2

    So it predicts well in one direction and not the other: ``bronze`` and
    ``green`` almost never reach Europe PMC, while ``gold``/``hybrid`` usually
    but not always do. Informative, not determinative — do not use it as a
    substitute for checking whether text actually came back.

    ``(False, None)`` on any failure: a best-effort second opinion must not raise.
    """
    contact = email if email is not None else _DEFAULT_EMAIL
    if not contact:
        return False, None
    try:
        r = _request_with_retry(f"{UNPAYWALL}/{doi}", sleep=sleep, email=contact)
        data = r.json()
        return bool(data.get("is_oa")), data.get("oa_status")
    except (requests.RequestException, ValueError):
        return False, None


def _publication_classes(srp, email=None, api_key=None, sleep=None) -> list[str]:
    """Distinct accessibility classes of a study's linked publications, by accession.

    Fetches a summary (resolving BioProject PMIDs) then classifies. Empty list
    means no linked paper (or the study could not be resolved). When you already
    have a Project, call ``project.publication_classes()`` instead — no re-fetch.
    """
    try:
        p = Project.summary(
            srp, include_publications=True, email=email, api_key=api_key, sleep=sleep
        )
    except Exception:  # noqa: BLE001 - a bad/private study just has no classes
        return []
    return p.publication_classes(sleep=sleep)


def _study_has_publication_class(
    srp, wanted, email=None, api_key=None, sleep=None
) -> bool:
    """True if any publication linked to ``srp`` classifies as ``wanted``."""
    return wanted in _publication_classes(
        srp, email=email, api_key=api_key, sleep=sleep
    )


def _to_entrez_date(d) -> str | None:
    """Format a date as Entrez expects (YYYY/MM/DD).

    Accepts a ``datetime.date``/``datetime.datetime`` (unambiguous), or an ISO
    ``"YYYY-MM-DD"`` string. Rejects anything else so a locale-ordered string like
    ``"03/04/2022"`` can't slip through and be misread.
    """
    if d is None:
        return None
    if isinstance(d, str):
        d = date.fromisoformat(d)  # only unambiguous ISO 'YYYY-MM-DD' is allowed
    if not isinstance(d, date):
        raise TypeError(
            "date must be a datetime.date or an ISO 'YYYY-MM-DD' string, "
            f"got {type(d).__name__}"
        )
    return d.strftime("%Y/%m/%d")


def _record_count(accession, email=None, api_key=None, sleep=None) -> int | None:
    """Number of SRA (experiment) records for a study accession, cheaply.

    Used to spot giant continuously-appended umbrella studies (e.g. PulseNet
    surveillance) whose record_count runs into the hundreds of thousands.
    """
    sleep = _sleep_for(sleep, api_key)
    try:
        res = _eutils(
            "esearch.fcgi", sleep=sleep, db="sra", term=accession, retmax=0,
            retmode="json", **_common(email, api_key),
        ).json()["esearchresult"]
        return _int(res.get("count"))
    except Exception:  # noqa: BLE001 - treat an unknowable count as "no guard hit"
        return None


def _page_starts(
    sort: str, count: int, page_size: int, max_pages: int | None = None
) -> list[int]:
    """Ordered esummary retstart offsets for a sort mode over `count` records.

    The esearch result set is newest-first, so offset 0 is the newest record and
    ``count-1`` the oldest. "recent" reads front-to-back, "oldest" back-to-front,
    "random" visits page-aligned blocks in shuffled order (spreading the sample
    across the whole date range).

    ``max_pages`` caps how many offsets are produced. It matters for "random" over
    a large result set: an unfiltered SRA date range runs to millions of records,
    and only the handful of pages the caller will actually read should be
    materialized. ``random.sample`` still draws them from the full range, so the
    spread across the date range is unchanged.
    """
    total_pages = max(1, math.ceil(count / page_size))
    pages = total_pages if max_pages is None else min(total_pages, max(1, max_pages))
    if sort == "random":
        return [i * page_size for i in random.sample(range(total_pages), pages)]
    if sort == "oldest":
        return [max(0, count - (i + 1) * page_size) for i in range(pages)]
    return [i * page_size for i in range(pages)]  # "recent"


def search_studies(
    query: str = "",
    max_studies: int = 50,
    sort: str = "recent",
    within_years: float | None = None,
    after_date: date | None = None,
    before_date: date | None = None,
    organism: str | None = None,
    strategy: str | None = None,
    source: str | None = None,
    publication: str | None = None,
    max_records: int | None = None,
    email: str | None = None,
    api_key: str | None = None,
    sleep: float | None = None,
    page_size: int = 500,
    max_scanned: int | None = None,
    progress=None,
) -> list[str]:
    """Enumerate up to ``max_studies`` distinct study accessions (SRP/ERP/DRP).

    ``sort`` controls which part of the (newest-first) result set is sampled:

    * ``"recent"`` (default) — newest studies first
    * ``"oldest"``           — oldest studies first
    * ``"random"``           — a spread across the whole result set; best when
      seeking studies old enough to already have a linked publication (the newest
      studies almost never do — data is deposited well before the paper appears)

    All filters are optional and AND-combined into an Entrez query:

    * ``query``   —raw query string, e.g. ``"cancer AND single cell"``
    * ``organism``—e.g. ``"Homo sapiens"``      -> ``[Organism]``
    * ``strategy``—e.g. ``"RNA-Seq"``           -> ``[Strategy]``
    * ``source``  —e.g. ``"TRANSCRIPTOMIC"``    -> ``[Source]``

    Time window (by release date, ``datetype=pdat``):

    * ``within_years=10`` -> studies released in the last ~10 years
    * ``after_date`` / ``before_date`` -> an explicit range given as
      ``datetime.date`` objects, e.g.
      ``after_date=date(2022, 1, 1), before_date=date(2024, 6, 30)``. Using
      ``date`` (or an ISO ``"YYYY-MM-DD"`` string) avoids the day/month ambiguity
      of locale-ordered date strings.

    ``publication`` filters by the accessibility of each study's linked paper
    (see :func:`classify_publication`): ``"oa"``, ``"partial"``, or ``"paywall"``.
    A study is kept only if *some* linked publication matches. This is expensive
    — it resolves BioProject PMIDs and classifies each per candidate study, and
    studies with no linked paper are always dropped — so use a small
    ``max_studies`` (and raise ``max_scanned`` if you get fewer than requested).

    ``max_records`` skips studies with more than that many SRA records — the guard
    against giant surveillance umbrella studies (PulseNet, GenomeTrakr, ...). When
    set it costs one extra count lookup per candidate.

    SRA search is per-experiment, so this pages through hits and de-dupes down to
    the underlying studies (``max_scanned`` bounds how many experiment records are
    examined). Returns accessions in the order implied by ``sort``—feed them
    straight to :func:`scan`.

    ``progress`` is an optional ``callable(found, scanned)`` invoked after each
    page; enumerating thousands of studies reads thousands of pages over many
    minutes, and this is the only window into that phase.

    A page whose request fails after its retries is skipped rather than aborting
    the enumeration, and a :mod:`warnings` warning is raised at the end reporting
    how many were lost. Enumeration stops early after
    ``_MAX_CONSECUTIVE_PAGE_FAILURES`` failures in a row.
    """
    if sort not in ("recent", "oldest", "random"):
        raise ValueError(
            f"sort must be 'recent', 'oldest', or 'random', got {sort!r}"
        )
    sleep = _sleep_for(sleep, api_key)
    parts = []
    if query:
        parts.append(f"({query})")
    if organism:
        parts.append(f'"{organism}"[Organism]')
    if strategy:
        parts.append(f"{strategy}[Strategy]")
    if source:
        parts.append(f"{source}[Source]")
    term = " AND ".join(parts) if parts else "all[sb]"

    common = _common(email, api_key)

    date_params: dict = {}
    if within_years is not None:
        date_params = {"datetype": "pdat", "reldate": int(round(within_years * 365.25))}
    elif after_date or before_date:
        date_params = {
            "datetype": "pdat",
            "mindate": _to_entrez_date(after_date) or "1900/01/01",
            "maxdate": _to_entrez_date(before_date) or "3000/01/01",
        }

    if max_scanned is None:
        max_scanned = max_studies * 200  # bound the experiment records examined
    wanted = publication.lower() if publication else None
    page_failures: list[tuple[int, Exception]] = []

    def _candidates():
        """Yield distinct study accessions in `sort` order, bounded by max_scanned."""
        # Use the history server so retstart can index anywhere in the result set
        # (direct pagination caps out near 10k records; studies can number millions).
        res = _eutils(
            "esearch.fcgi",
            sleep=sleep,
            db="sra",
            term=term,
            usehistory="y",
            retmax=0,
            retmode="json",
            **common,
            **date_params,
        ).json()["esearchresult"]
        count = _int(res.get("count")) or 0
        webenv = res.get("webenv")
        qkey = res.get("querykey")
        if not count or not webenv:
            return
        # smaller pages for random so the sample spreads across more time points
        eff_page = min(page_size, 100) if sort == "random" else page_size
        seen: set[str] = set()
        scanned = 0
        max_pages = math.ceil(max_scanned / eff_page)
        consecutive = 0
        for start in _page_starts(sort, count, eff_page, max_pages=max_pages):
            if scanned >= max_scanned:
                return
            time.sleep(sleep)
            try:
                xml = _eutils(
                    "esummary.fcgi",
                    sleep=sleep,
                    post=True,  # WebEnv/id payload -> POST to avoid a 414
                    db="sra",
                    WebEnv=webenv,
                    query_key=qkey,
                    retstart=start,
                    retmax=eff_page,
                    retmode="xml",
                    xml=True,
                    **common,
                )
                srps = _srps_from_esummary(xml)
            except (requests.RequestException, ET.ParseError) as exc:
                # A harvest of thousands reads thousands of pages over hours; one
                # page failing its retries must not throw away everything already
                # enumerated. Skip it — the sample is a random spread anyway, so a
                # missing page costs a little breadth, not correctness.
                page_failures.append((start, exc))
                consecutive += 1
                if consecutive >= _MAX_CONSECUTIVE_PAGE_FAILURES:
                    # Not bad luck any more: the history session has almost
                    # certainly expired (WebEnv is held across the whole
                    # enumeration) or NCBI is down. Every later page would fail the
                    # same way, so stop and let the caller keep what it has.
                    return
                scanned += eff_page
                continue
            consecutive = 0
            for srp in srps:
                if srp not in seen:
                    seen.add(srp)
                    yield srp
            scanned += eff_page
            if progress is not None:
                progress(len(seen), scanned)

    studies: list[str] = []
    for srp in _candidates():
        # cheap guard first: drop giant umbrella studies before the costly checks
        if max_records is not None:
            rc = _record_count(srp, email=email, api_key=api_key, sleep=sleep)
            if rc is not None and rc > max_records:
                continue
        if wanted is not None and not _study_has_publication_class(
            srp, wanted, email=email, api_key=api_key, sleep=sleep
        ):
            continue
        studies.append(srp)
        if len(studies) >= max_studies:
            break
    if page_failures:
        # Never silent: fewer studies than asked for is a legitimate result here,
        # but the caller has to be able to tell "SRA has no more" from "we lost
        # pages", because only the second is worth retrying.
        warnings.warn(
            f"search_studies: {len(page_failures)} esummary page(s) failed and were "
            f"skipped; returned {len(studies)} of {max_studies} requested studies. "
            f"First failure at retstart={page_failures[0][0]}: {page_failures[0][1]!r}",
            stacklevel=2,
        )
    return studies


def _load_json(source):
    """Return parsed JSON from a dict/list (as-is), a file path, or a JSON string."""
    if isinstance(source, (dict, list)):
        return source
    try:
        if os.path.exists(source):
            with open(source, encoding="utf-8") as fh:
                return json.load(fh)
    except (OSError, ValueError):
        pass  # not a usable path (e.g. a long JSON string) -> parse as JSON text
    return json.loads(source)


def load_studies(source) -> list[Project]:
    """Load a JSON array of study dicts into ``Project`` objects (no network).

    ``source`` is a file path (e.g. ``"recent_studies.json"``), a JSON string, or
    an already-parsed list/dict. Accepts a single study dict too. Use this to map
    a saved study list back into Projects::

        studies = load_studies("recent_studies.json")
        oa = [p for p in studies if any(x.type for x in p.publications)]
    """
    data = _load_json(source)
    if isinstance(data, dict):
        data = [data]
    return [Project.from_dict(d) for d in data]


def scan(
    accessions,
    include_publications: bool = False,
    max_records: int | None = None,
    email: str | None = None,
    api_key: str | None = None,
    sleep: float | None = None,
) -> tuple[dict[str, Project], dict[str, Exception]]:
    """Lightweight ``Project.summary`` over many accessions.

    Returns ``(projects, errors)``:

    * ``projects``—``{accession: Project}`` for every study that built
    * ``errors``  —``{accession: Exception}`` for every one that failed
      (a bad accession or persistent network error never aborts the whole run)

    ``max_records`` skips studies whose ``record_count`` exceeds it — a guard
    against giant continuously-appended umbrella studies (e.g. PulseNet
    surveillance, hundreds of thousands of runs) that would blow up a later full
    ``Project(acc)`` build. Skipped studies land in ``errors`` with a reason.
    The check is nearly free: it is pushed down into the build, which abandons an
    oversized study right after the esearch that reveals its count — before the
    (expensive) BioProject fetch, not after.

    Requests are paced by ``sleep`` between studies; per-request rate-limit
    retries are handled inside ``Project``. Supply an ``api_key`` to raise
    NCBI's limit from 3 to 10 requests/second for faster bulk scans.
    """
    projects: dict[str, Project] = {}
    errors: dict[str, Exception] = {}
    for acc, project, error in scan_iter(
        accessions,
        include_publications=include_publications,
        max_records=max_records,
        email=email,
        api_key=api_key,
        sleep=sleep,
    ):
        if error is not None:
            errors[acc] = error
        else:
            projects[acc] = project
    return projects, errors


def scan_iter(
    accessions,
    include_publications: bool = False,
    max_records: int | None = None,
    email: str | None = None,
    api_key: str | None = None,
    sleep: float | None = None,
):
    """Streaming :func:`scan` — yields ``(accession, project, error)`` per study.

    Exactly one of ``project``/``error`` is None. :func:`scan` is this collected
    into two dicts; use this form directly when a long run needs to checkpoint or
    report progress as it goes rather than only at the end::

        for acc, project, error in scan_iter(accessions, max_records=5000):
            ...
    """
    sleep = _sleep_for(sleep, api_key)
    for acc in accessions:
        project: Project | None = None
        error: Exception | None = None
        try:
            p = Project.summary(
                acc,
                include_publications=include_publications,
                email=email,
                api_key=api_key,
                sleep=sleep,
                max_records=max_records,
            )
        except Exception as exc:  # noqa: BLE001 - collect, don't abort the batch
            error = exc
        else:
            if max_records is not None and p.record_count and p.record_count > max_records:
                error = ValueError(
                    f"record_count {p.record_count} exceeds max_records {max_records}"
                )
            else:
                project = p
        yield acc, project, error
        time.sleep(sleep)


def filter_by_publication(
    accessions,
    wanted: str,
    email: str | None = None,
    api_key: str | None = None,
    sleep: float | None = None,
) -> tuple[list[str], dict[str, list[str]]]:
    """Post-hoc filter: keep studies whose linked paper classifies as ``wanted``.

    The "scan first, classify the survivors" complement to
    ``search_studies(publication=...)`` — cheaper and cacheable because you
    control which (already-shortlisted) accessions get classified.

    ``wanted`` is one of ``"oa"``, ``"partial"``, ``"paywall"`` (case-insensitive).
    Returns ``(matched, classes)``:

    * ``matched``—accessions with at least one publication of class ``wanted``
    * ``classes``—``{accession: [classes...]}`` for *every* input accession, so
      the classification work isn't thrown away (empty list = no linked paper)
    """
    want = wanted.lower()
    sleep = _sleep_for(sleep, api_key)
    classes: dict[str, list[str]] = {}
    matched: list[str] = []
    for acc in accessions:
        cls = _publication_classes(acc, email=email, api_key=api_key, sleep=sleep)
        classes[acc] = cls
        if want in cls:
            matched.append(acc)
        time.sleep(sleep)
    return matched, classes


def filter_projects_by_publication(
    projects, wanted: str, sleep: float | None = None
) -> tuple[list[Project], dict[str, list[str]]]:
    """Filter existing ``Project`` objects by linked-paper accessibility.

    Use this on Projects you already have (e.g. from :func:`load_studies`) rather
    than converting back to accessions — it classifies the publications already on
    each object and never re-resolves BioProject PMIDs.

    ``wanted`` is one of ``"oa"``, ``"partial"``, ``"paywall"`` (case-insensitive).
    Returns ``(matched, classes)``:

    * ``matched``—the Projects with at least one publication of class ``wanted``
    * ``classes``—``{accession: [classes...]}`` for every input Project
    """
    want = wanted.lower()
    classes: dict[str, list[str]] = {}
    matched: list[Project] = []
    for p in projects:
        cls = p.publication_classes(sleep=sleep)
        classes[p.accession] = cls
        if want in cls:
            matched.append(p)
    return matched, classes