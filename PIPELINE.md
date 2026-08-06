# The Harvest Pipeline — Every API Call, In Order

How the shortlist that feeds LLM metadata reconstruction gets built: from an empty
query to a set of studies that each have a linked, openly-readable paper.

Three stages, two files. `dataset.py` orchestrates and checkpoints; `project.py` owns
every HTTP call. `main.py` is the entry point and does nothing but supply credentials
and parameters.

```
search_studies()          scan_iter()               filter_oa_studies()
page a date-filtered  ->  build a study-level   ->  classify each linked paper,
SRA query, de-dupe        summary + its linked      keep studies with an
to studies                publications             open-access one

2,000 accessions          1,858 kept                305 open-access
```

Stages 1 and 2 run inside `save_recent_studies()` and write `recent_studies.json`.
Stage 3 reads that file back and writes `oa_studies.json`. Counts above are from the
2,000-study harvest over the 2016–2024 release window.

---

## 1. The services and databases involved

Four services. Only the two NCBI hosts ever receive the `api_key` — Europe PMC and
Unpaywall are third parties with no use for it, and sending it would leak the key into
their access logs.

| Service | Host | What it answers |
|---|---|---|
| **NCBI E-utilities** | `eutils.ncbi.nlm.nih.gov` | All discovery and metadata. Serves two Entrez databases: `db=sra` and `db=bioproject` |
| **PMC Open Access service** | `www.ncbi.nlm.nih.gov/pmc` | Is this paper in the downloadable, licence-tagged OA subset? |
| **Europe PMC** | `www.ebi.ac.uk` | Resolve a PMID or DOI to an article record with its PMCID and open-access flag |
| **Unpaywall** | `api.unpaywall.org` | OA that PMC cannot see — publisher-hosted and repository-hosted copies. Requires a contact email |

Two Entrez databases are used, and the split matters:

* **`db=sra`** — the study, experiment, run and sample records themselves.
* **`db=bioproject`** — the project registry. **The link to a publication lives here, not
  in the SRA record.** This is the only way to get from a study to its paper.

---

## 2. Stage 1 — finding studies

`project.py` · `search_studies()`

SRA is indexed per *experiment* (SRX), not per study, so this pages through experiment
records and de-dupes down to the studies underneath. Roughly **2.65 distinct studies per
100-record page**.

| # | Method | Endpoint | Database | Called |
|---|---|---|---|---|
| 1 | GET | `esearch.fcgi` | `sra` | once per run |
| 2 | POST | `esummary.fcgi` | `sra` | once per page |

**1. Open a history session.**

```
GET esearch.fcgi
    db=sra  term=all[sb]  datetype=pdat  mindate=…  maxdate=…
    usehistory=y  retmax=0  retmode=json
```

Returns the total hit count plus a `WebEnv` + `query_key` handle. The history server is
what lets `retstart` index anywhere in the result set — plain pagination caps out near
10,000 records, and a multi-year window holds tens of millions.

**2. Read one page of experiment summaries.**

```
POST esummary.fcgi
     db=sra  WebEnv=…  query_key=…  retstart=<page offset>  retmax=100  retmode=xml
```

Each `DocSum` carries an `ExpXml` blob; a regex pulls `<Study acc="…">` out of it
(regex rather than a nested XML parse, because those blobs can contain a stray `&` that
breaks re-parsing). **POST, not GET** — the WebEnv payload would overflow the URL limit.

`sort="random"` shuffles the page offsets so the sample spreads across the whole date
window instead of clustering in the newest records. This matters: the newest studies
almost never have a linked paper yet, because data is deposited well before publication.

---

## 3. Stage 2 — scanning each study

`project.py` · `scan_iter()` → `Project.summary()`

One to five requests per accession, depending on three branch conditions. The cost is
deliberately **O(1) in the number of runs** — a 26-run study and a 26,000-run study take
the same number of calls, because every `EXPERIMENT_PACKAGE` embeds the full `STUDY`
block, so one package is enough.

| # | Method | Endpoint | Database | Called |
|---|---|---|---|---|
| 1 | GET | `esearch.fcgi` | `sra` | always |
| 2 | GET | `efetch.fcgi` | `sra` | unless oversized |
| 3 | GET | `efetch.fcgi` | `sra` | if ≥ 2 records |
| 4 | GET | `esearch.fcgi` | `bioproject` | if **not** `PRJNA` |
| 5 | GET | `efetch.fcgi` | `bioproject` | if a BioProject exists |

**1. Count the study's records and grab one UID.**

```
GET esearch.fcgi  db=sra  term=<SRP>  usehistory=y  retmax=1  retmode=json
```

The count becomes `record_count`; the single UID is the study's *newest* record.

> **Stop here if `record_count > max_records`** (default 5,000). Oversized
> continuously-appended surveillance umbrellas — PulseNet, GenomeTrakr — cost exactly one
> request and nothing else. This guard lives inside the build rather than in `scan()`,
> because their BioProject records are tens of megabytes and take ~20s: fetching one only
> to discard the study used to dominate bulk runs.

**2. Read the study block.**

```
GET efetch.fcgi  db=sra  id=<uid>  retmode=xml
```

Yields title, abstract, study type, the BioProject accession, and cross-references
(GEO, dbGaP — NCBI-side registries, so these appear on `SRP` studies only).

**3. Read the oldest record, for the release date.**

```
GET efetch.fcgi  db=sra  WebEnv=…  query_key=…  retstart=<count-1>  retmax=300  retmode=xml
```

`esearch` returns its `idlist` **newest-first**, so the UID from step 1 is the study's
newest record. For a study appended to over years that date can be badly wrong — one
example reported 2026 when its true earliest run was 2014. Skipped when `record_count < 2`.

Note this makes `published` the release date of the study's oldest *indexed* record, which
is not provably the earliest run: esearch's ordering is not the same as the run
`@published` ordering. Sampling 15 studies against full builds found 13 exact, one +0.1h,
one +33h. Good enough for date-window selection; don't document it as exactly "the
earliest run".

**4. Resolve a non-NCBI BioProject accession to a UID.**

```
GET esearch.fcgi  db=bioproject  term=<PRJEB…>[Project Accession]  retmax=1  retmode=json
```

**Skipped for `PRJNA`.** `efetch db=bioproject` does *not* resolve accessions — it strips
the `PRJ??` prefix and uses the remaining digits as a raw internal UID. That is correct for
NCBI-archived projects by construction, since the accession is formed from the UID
(`PRJNA646996` *is* UID 646996). EBI and DDBJ number independently, so `PRJEB47383` (really
UID 778158) would silently fetch unrelated UID 47383 — usually a non-public record reading
as "no publications", occasionally a real project whose papers then get attached to the
wrong study.

The field qualifier is load-bearing: a bare `term=PRJEB1787` is relevance-ranked across all
fields and returns that project's own record *third*. `[Project Accession]` matches exactly
one.

**5. Fetch the linked publications.**

```
GET efetch.fcgi  db=bioproject  id=<uid or PRJNA…>  retmode=xml
```

Returns `<Publication id= DbType=>` elements — `ePubmed` gives a PMID, `eDOI` a DOI. The
same paper is sometimes listed twice, so ids are de-duplicated. The record's `ArchiveID`
is checked against the requested accession before its publications are trusted; if it
names some other project, the result is discarded.

Skipped when the study has no BioProject at all (12 of 1,858 in the reference run).

---

## 4. Stage 3 — filtering by publication

`dataset.py` · `filter_oa_studies()` → `project.py` · `classify_publication()`

One to three requests per paper, across three services. A study stops at its **first**
open-access paper — later ones cannot change the outcome and are left unclassified
(`None`) in the output.

| # | Method | Endpoint | Service | Called |
|---|---|---|---|---|
| 1 | GET | `europepmc/webservices/rest/search` | Europe PMC | always |
| 2 | GET | `pmc/utils/oa/oa.fcgi` | PMC OA | if a PMCID exists |
| 3 | GET | `v2/{doi}` | Unpaywall | as a fallback |

```
GET  …/europepmc/webservices/rest/search
     query=EXT_ID:<pmid> AND SRC:MED     (or)  query=DOI:"<doi>"
     format=json  resultType=core

GET  …/pmc/utils/oa/oa.fcgi   id=<PMCID>

GET  …/v2/<doi>   email=<contact>
```

### The decision tree

```
Europe PMC: does the paper resolve?
│
├─ no ──┬─ DOI available ─── Unpaywall ─┬─ is_oa ────────── oa
│       │                               └─ not ──────────── unknown
│       └─ PMID only ────────────────────────────────────── unknown
│
└─ yes ─┬─ has PMCID ─── PMC OA ─┬─ in the OA subset ────── oa
        │                        └─ not ─┬─ flagged open ── oa
        │                                └─ not ─────────── partial
        └─ no PMCID ─┬─ flagged open ────────────────────── oa
                     └─ not ─── Unpaywall ─┬─ is_oa ─────── oa
                                           └─ not ───────── paywall
```

| Class | Meaning |
|---|---|
| `oa` | Open-access full text — downloadable and text-mineable, which is what the LLM stage needs |
| `partial` | Full text is in PMC but not OA-licensed. Readable on the web, not downloadable (e.g. author manuscripts) |
| `paywall` | Indexed, but only the abstract is public |
| `unknown` | The id could not be resolved anywhere |

**Why three services.** Being indexed in PMC is not the same as being open, and it fails in
both directions. Fully-OA venues outside PubMed score `unknown` on a PMC-only check —
`10.3389/fmars.2022.930017` is CC-BY with live full text and returns zero hits from both
Europe PMC and `esearch db=pmc`. Conversely, papers free at the publisher but absent from
PMC score `paywall` — `10.1111/age.13334` is free at Wiley. Unpaywall closes both gaps.
Note also that publisher pages often return HTTP 403 to bots, so fetching the page is not a
valid paywall test.

---

## 5. Complete call reference

| Stage | Method | Service | Endpoint | Database | Called | Returns |
|---|---|---|---|---|---|---|
| 1 | GET | E-utilities | `esearch.fcgi` | `sra` | 1× / run | Hit count + WebEnv history handle |
| 1 | POST | E-utilities | `esummary.fcgi` | `sra` | 1× / page | 100 experiment DocSums → study accessions |
| 2 | GET | E-utilities | `esearch.fcgi` | `sra` | 1× / study | `record_count` + newest record UID |
| 2 | GET | E-utilities | `efetch.fcgi` | `sra` | 1× / study | STUDY block: title, abstract, BioProject, cross-refs |
| 2 | GET | E-utilities | `efetch.fcgi` | `sra` | if ≥2 records | Oldest record → earliest release date |
| 2 | GET | E-utilities | `esearch.fcgi` | `bioproject` | if not `PRJNA` | NCBI UID for a `PRJEB` / `PRJDB` accession |
| 2 | GET | E-utilities | `efetch.fcgi` | `bioproject` | if BioProject | Publication elements → PMIDs and DOIs |
| 3 | GET | Europe PMC | `rest/search` | — | 1× / paper | PMCID, `isOpenAccess` flag, DOI |
| 3 | GET | PMC OA | `oa.fcgi` | — | if PMCID | Membership of the OA subset |
| 3 | GET | Unpaywall | `v2/{doi}` | — | fallback | `is_oa` across publisher and repository copies |

### What that cost

Scan-phase requests from the 2,000-accession run (1,858 kept, 305 open-access):

| Call | Requests | Share | Note |
|---|---:|---:|---|
| `esearch db=sra` | 2,000 | 26.5% | Every candidate, including the 142 rejected as oversized |
| `efetch db=sra` (study) | 1,858 | 24.6% | One per kept study |
| `efetch db=sra` (oldest) | 1,550 | 20.5% | Skipped for 308 single-record studies |
| `efetch db=bioproject` | 1,846 | 24.5% | Skipped for 12 studies with no BioProject |
| `esearch db=bioproject` | 296 | 3.9% | The 237 `ERP` and 59 `DRP` studies only |
| **Total** | **7,550** | | **4.06 requests per kept study**, against a floor of 3 |

Measured throughput: search **0.234 s/study** (linear — it does not degrade with harvest
size), scan **~0.23 s/request**. A 2,000-study harvest runs in roughly 45 minutes.

---

## 6. The path not taken — a full build

A full `Project(accession)` build fetches every sample, experiment and run. **The harvest
never calls it** — the shortlist only needs study-level summaries — but it is what the LLM
stage will use on the studies that survive.

```
GET  esearch.fcgi  db=sra  term=<SRP>  retmax=100000  retmode=json
POST epost.fcgi    db=sra  id=<every record UID>
GET  efetch.fcgi   db=sra  WebEnv=…  query_key=…  retstart=…  retmax=300  retmode=xml
     ... once per batch of 300 records
```

`epost` must be a **POST**: the id list is one entry per record, and a GET URL 414s past
roughly a thousand of them.

---

## 7. Operational notes

**Accession prefixes.** The first letter is the INSDC archive of origin — `S` = NCBI SRA,
`E` = EBI/ENA, `D` = DDBJ — and the second and third encode the object type: `RP` study,
`RS` sample, `RX` experiment, `RR` run. All three archives mirror each other, so ENA and
DDBJ studies are fully retrievable through NCBI. Their BioProject accessions follow the
same split (`PRJNA` / `PRJEB` / `PRJDB`), which is what step 4 of stage 2 exists to handle.

**Retries and pacing.** Every call goes through `_request_with_retry()`, which retries
429/5xx and transient connection failures five times with exponential backoff. Note that
`ChunkedEncodingError` and `ContentDecodingError` — a response body that dies mid-stream —
are *siblings* of `ConnectionError` under `RequestException`, not subclasses, so they must
be named explicitly in the retry tuple. Pacing is 0.11s with an api_key, 0.34s without.
Third-party calls in stage 3 are paced separately at `CLASSIFY_SLEEP`, deliberately not at
the NCBI rate, since Europe PMC and Unpaywall never granted a raised limit.

**The api_key is worth ~20%, not 3×.** The run is latency-bound, not rate-limited: roughly
three quarters of wall time is spent waiting inside HTTP responses, and the key only changes
the pacing sleeps.

**Checkpointing.** `save_recent_studies()` writes `<path>.partial` every 25 studies and
removes it only after the real output lands. Re-running the same call resumes from it.
A checkpoint written under different search parameters is ignored — the study list is a
random sample, so resuming one harvest into a differently-parameterised one would splice two
different samples together.

---

## 8. Verifying this document

API behaviour drifts and docs go stale. To re-trace the real calls, wrap the shared session
and print what goes over the wire:

```python
import project as P

_get, _post = P._SESSION.get, P._SESSION.post

def trace(fn, verb):
    def inner(url, **kw):
        q = kw.get("params") or kw.get("data") or {}
        print(f"{verb:4} {url.rsplit('/', 1)[-1]:14} db={q.get('db', ''):12} "
              f"{ {k: v for k, v in q.items() if k in ('term', 'id', 'retmax', 'retstart')} }")
        return fn(url, **kw)
    return inner

P._SESSION.get, P._SESSION.post = trace(_get, "GET"), trace(_post, "POST")

P.Project.summary("ERP134525", include_publications=True, max_records=5000)
P.classify_publication("28323820")
```

The call sequences in sections 2–4 were produced this way.
