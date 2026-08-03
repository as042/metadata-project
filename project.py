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
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import date

import requests

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
EPMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
PMC_OA = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"
UNPAYWALL = "https://api.unpaywall.org/v2"

# Publication accessibility classes returned by classify_publication().
PUBLICATION_CLASSES = ("oa", "partial", "paywall", "unknown")

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


def _request_with_retry(url, *, sleep: float = 0.34, post: bool = False, **params):
    """GET/POST with exponential backoff on rate-limit and transient server errors.

    Every network call in this module goes through here: NCBI allows only 3
    requests/second without an api_key (10 with one) and E-utilities also returns
    sporadic 5xx under load. Dropped connections and DNS hiccups are retried too —
    over a run of hundreds of studies one is near-certain, and it shouldn't end the
    run. Use ``post=True`` when passing a long id list (a GET URL would 414). Any
    non-retryable error status raises.
    """
    delay = max(sleep, 0.34)
    for attempt in range(5):
        try:
            if post:
                r = _SESSION.post(url, data=params, timeout=60)
            else:
                r = _SESSION.get(url, params=params, timeout=60)
        except (requests.ConnectionError, requests.Timeout):
            if attempt == 4:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 5.0)
            continue
        if r.status_code in (429, 500, 502, 503, 504) and attempt < 4:
            time.sleep(delay)
            delay = min(delay * 2, 5.0)
            continue
        r.raise_for_status()
        return r


# --------------------------------------------------------------------------- #
# Leaf data classes
# --------------------------------------------------------------------------- #
@dataclass
class FileRef:
    url: str
    md5: str | None = None
    size: int | None = None
    source: str | None = None  # e.g. "sra", "s3", "gs"


@dataclass
class Run:
    """SRR—belongs to exactly one Experiment."""

    accession: str
    total_spots: int | None = None
    total_bases: int | None = None
    published: str | None = None  # release date, e.g. "2017-06-07 10:30:09"
    files: list[FileRef] = field(default_factory=list)


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


@dataclass
class Publication:
    id: str
    type: str | None = None  # ePubmed (PMID) / eDOI (DOI)
    # Accessibility class ("oa"/"partial"/"paywall"/"unknown"), filled in on demand
    # by classification and persisted so it need not be recomputed. None = not yet
    # classified (determining it costs a network lookup).
    accessibility_type: str | None = None


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

    def _get(self, endpoint, **params):
        return _request_with_retry(
            f"{EUTILS}/{endpoint}", sleep=self._sleep, **self._common_params(**params)
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
            xml = self._efetch_batch(webenv, qkey, start).text
            self._parse_package_set(xml)
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
        root = ET.fromstring(self._efetch_ids(uids))
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
            root = ET.fromstring(self._efetch_batch(webenv, qkey, count - 1).text)
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

    def _efetch_ids(self, ids) -> str:
        return self._get(
            "efetch.fcgi", db="sra", id=",".join(ids), retmode="xml"
        ).text

    def _epost(self, uids):
        # POST, not GET: this is the whole point of epost — the id list is one per
        # record in the study, so a GET URL blows the ~8KB server limit and 414s
        # somewhere around a thousand records (SRP094905, 1800 records, failed).
        r = _request_with_retry(
            f"{EUTILS}/epost.fcgi",
            sleep=self._sleep,
            post=True,
            **self._common_params(db="sra", id=",".join(uids)),
        )
        root = ET.fromstring(r.text)
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
        )

    # -- parsing ---------------------------------------------------------- #
    def _parse_package_set(self, xml_text: str):
        root = ET.fromstring(xml_text)
        for pkg in root.iter("EXPERIMENT_PACKAGE"):
            self._parse_study(pkg.find("STUDY"))
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
        return s

    def _parse_experiment(self, pkg) -> Experiment:
        exp_el = pkg.find("EXPERIMENT")
        e = Experiment(accession=exp_el.get("accession"))
        e.title = _text(exp_el, "TITLE")

        lib = exp_el.find("DESIGN/LIBRARY_DESCRIPTOR")
        if lib is not None:
            e.library_strategy = _text(lib, "LIBRARY_STRATEGY")
            e.library_source = _text(lib, "LIBRARY_SOURCE")
            e.library_selection = _text(lib, "LIBRARY_SELECTION")
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
        e.sample_ids = self._sample_ids(exp_el, pkg)
        e.runs = self._parse_runs(pkg.find("RUN_SET")) if self._include_runs else []
        return e

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
            sra_files = run_el.find("SRAFiles")
            if sra_files is not None:
                for f in sra_files.findall("SRAFile"):
                    r.files.append(
                        FileRef(
                            url=f.get("url"),
                            md5=f.get("md5"),
                            size=_int(f.get("size")),
                            source=f.get("cluster") or f.get("semantic_name"),
                        )
                    )
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
            r = self._get("efetch.fcgi", db="bioproject", id=uid, retmode="xml")
            root = ET.fromstring(r.text)
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

        def _run(rd):
            rd = dict(rd)
            files = [FileRef(**f) for f in rd.pop("files", []) or []]
            return Run(**rd, files=files)

        def _experiment(ed):
            ed = dict(ed)
            runs = [_run(r) for r in ed.pop("runs", []) or []]
            return Experiment(**ed, runs=runs)

        p = cls.__new__(cls)  # bypass __init__ so nothing is fetched
        p.accession = d["accession"]
        p.bioproject = d.get("bioproject")
        p.title = d.get("title")
        p.abstract = d.get("abstract")
        p.study_type = d.get("study_type")
        p.published = d.get("published")
        p.record_count = d.get("record_count")
        p.external_ids = dict(d.get("external_ids") or {})
        p.samples = {k: Sample(**v) for k, v in (d.get("samples") or {}).items()}
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


def _srps_from_esummary(xml_text: str) -> list[str]:
    root = ET.fromstring(xml_text)  # the outer esummary envelope is well-formed
    out = []
    for item in root.iter("Item"):
        if item.get("Name") == "ExpXml" and item.text:
            m = _STUDY_ACC_RE.search(item.text)
            if m:
                out.append(m.group(1))
    return out


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
        root = ET.fromstring(
            _request_with_retry(
                PMC_OA, sleep=sleep, id=pmcid, **_common(email, api_key)
            ).text
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


def _unpaywall_is_oa(doi: str, sleep: float = 0.34, email: str | None = None) -> bool:
    """True if Unpaywall reports a free full-text copy of ``doi``.

    Covers the OA that PMC can't see: publisher-hosted (gold/hybrid) and
    repository-hosted (green) copies of articles outside PubMed's index. Unpaywall
    requires a contact email, so this is skipped—returning False—when neither an
    explicit ``email`` nor a process-wide one is set (see
    :func:`set_entrez_credentials`). Any lookup failure is treated as "not proven
    OA" rather than raising: this is a best-effort second opinion.
    """
    contact = email if email is not None else _DEFAULT_EMAIL
    if not contact:
        return False
    try:
        r = _request_with_retry(
            f"{UNPAYWALL}/{doi}", sleep=sleep, email=contact
        )
        return bool(r.json().get("is_oa"))
    except (requests.RequestException, ValueError):
        return False


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
        for start in _page_starts(sort, count, eff_page, max_pages=max_pages):
            if scanned >= max_scanned:
                return
            time.sleep(sleep)
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
                **common,
            ).text
            for srp in _srps_from_esummary(xml):
                if srp not in seen:
                    seen.add(srp)
                    yield srp
            scanned += eff_page

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
    sleep = _sleep_for(sleep, api_key)
    projects: dict[str, Project] = {}
    errors: dict[str, Exception] = {}
    for acc in accessions:
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
            errors[acc] = exc
        else:
            if max_records is not None and p.record_count and p.record_count > max_records:
                errors[acc] = ValueError(
                    f"record_count {p.record_count} exceeds max_records {max_records}"
                )
            else:
                projects[acc] = p
        time.sleep(sleep)
    return projects, errors


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