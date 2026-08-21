# Reconstruction Cost — What We Measured

What the model layers actually bill, on Claude Sonnet 5 with thinking disabled, and what
that projects to at corpus and archive scale. All figures are from the **Rust
implementation** (`rust/`) over the five-study / 52-record comparison set unless noted,
with saved runs in `rust/runs/`. Ordering effects on cost live in
[layer_order_findings.md](layer_order_findings.md); Haiku-era prompt and caching work is
in [layer_3_haiku_findings.md](layer_3_haiku_findings.md).

**The one-line summary:** the per-sample layer is 97.5% of the bill, so cost scales with
*samples* and not with studies or papers; the whole 346-study corpus prices at **$284.63**
and all of SRA at **$130k–$886k** depending entirely on which volume assumption you make.
No per-layer model split beat pure Sonnet at this scale — Opus on the paper layer returned
nothing for +60–70%. And the pre-flight estimator modelled the cached prefix as written **once
per run** on the sequential path when it is really written **once per study**, which put two
Opus runs 18–21% *under* actual before it was fixed.

---

## 1. Measured unit costs

Whole-corpus plan, Sonnet 5, thinking disabled, batched, no caps:

| layer | calls | cost | share |
|---|---:|---:|---:|
| `LLMNaive` | 88,608 | **$277.52** | 97.5% |
| `LLMPaper` | 353 | $7.11 | 2.5% |
| `Direct`, `Harmonized` | 0 | $0.00 | — |
| **total (346 studies, 102,240 records)** | 88,961 | **$284.63** | |

Derived units, which are what any projection should actually use:

* **$0.003134 per sample** — `LLMNaive` plans one call per distinct sample, so this is the
  number that scales.
* **$0.02154 per paper** — `LLMPaper` plans one call per paper, once per study.

**Do not extrapolate from a per-record rate off a small run.** The five-study comparison
set averages ~10 samples per study; the corpus averages 256. A rate fitted to the former
misprices the latter by an order of magnitude, because the naive layer bills per sample
while the paper layer bills per study.

### Effort: no cheaper, no better, and worse on coverage

**Scope, first: every figure in this section is with `thinking: Disabled`.** The two are not
independent — the API itself refuses Opus 5 at `XHigh`/`Max` effort when thinking is disabled
(`unsupported_combination` in `claude.rs` mirrors that rule). Effort *with thinking enabled*
was attempted separately and never produced a usable measurement, for the reason in
"Adaptive thinking cannot be given a workable ceiling" below.

`effort` reaches the wire as `output_config.effort` on every model that accepts it, whatever
the thinking setting, so it was reasonable to suspect it changed output volume. It does not,
at the scale that matters.

**The probe that said otherwise.** Against the real layer-3 request (same prompt, same JSON
schema, evidence straight from `plan`), Sonnet 5, thinking disabled, 8 calls per level across
four hand-picked jobs:

| effort | output tokens per field | vs medium |
|---|---:|---:|
| low | 5.8 | 0.87× |
| medium | 6.7 | 1.00× |
| high | 6.6 | 0.99× |
| xhigh | 6.3 | 0.94× |
| max | 11.0 | 1.64× |

On that basis a 1.65× multiplier was added to the estimator.

**The full runs that overturned it.** Two 52-record runs at `Max` against four at `Medium`,
pure Sonnet, sequential, effort the only variable:

| effort | order | output tokens | cost |
|---|---|---:|---:|
| medium | DHNP | 10,608 | $0.5187 |
| medium | DHNP | 10,190 | $0.4851 |
| medium | DHPN | 7,703 | $0.4308 |
| medium | DHPN | 5,999 | $0.4137 |
| **max** | DHNP | 10,252 | $0.5151 |
| **max** | DHPN | 8,934 | $0.3992 |

**1.11× on output, 0.99× on cost** — and the within-effort spread (5,999 to 10,608 output
tokens at identical settings) is far larger than the difference between efforts. The
multiplier was reverted; the estimator has no effort term.

With it in place, the two Max runs were quoted at $0.7422 against actuals of $0.5151 and
$0.3992 — **+44% and +86%**. Removing it returns them to $0.6007, or +17% and +50%, in line
with every other sequential run.

### The coverage and accuracy side

Cost was the question; it turned out not to be the interesting axis. The same six runs, scored
on what they actually filled:

| order | effort | real/rec | text | paper | missing |
|---|---|---:|---:|---:|---:|
| DHNP | medium | 35.71 | 2.54 | 1.67 | 299 |
| DHNP | medium | 35.29 | 2.50 | 1.29 | 314 |
| DHPN | medium | 36.25 | 2.04 | 2.71 | 312 |
| DHPN | medium | 36.06 | 1.54 | 3.02 | 275 |
| DHNP | **max** | 34.58 | 2.29 | **0.79** | 384 |
| DHPN | **max** | 35.10 | 1.85 | **1.75** | 228 |

**−0.99 real values per record** against a medium sd of 0.37, with complete separation: both
max runs fall below all four medium runs.

**The loss is concentrated in `LLMPaper`** — 1.67/1.29 → 0.79 on DHNP, 2.71/3.02 → 1.75 on
DHPN, while the text layer barely moves. Max effort roughly halves what the study-wide layer
contributes, and the mechanism for that is unexplained.

**It does not convert them into declared absences.** Missing is flat (300 against 306), so
total settled slots fall from ~2,163 to ~2,118: max effort simply answers less and leaves more
fields `Unknown`.

**And it is no more accurate by any proxy available:**

| | medium (n=4) | max (n=2) |
|---|---|---|
| verbatim rate | 44%, 58%, 65%, 40% | 51%, 56% |
| understated | 25, 26, 2, 14 | 22, 43 |
| literal `"placeholder"` / `"n/a"` values | 7, 4, 2, 3 | 5, 4 |

Max sits inside the medium range on all three. One defect appears at max and in no medium run:
`age = "6\ntdash8 week old"` on 7 records, a mangled en-dash — one run, so an observation
rather than a finding.

**Conclusion: `Effort::Medium` is the right default.** Max costs the same, fills about one
field per record less, and is not more honest about what it quoted. True *accuracy* remains
unmeasured — that needs the gold set — so this rules max out on the axes we can see, not on
correctness.

**The lesson is the one this repo keeps relearning.** Eight calls on four jobs is not a
measurement of a fifty-two-call run. The probe was run on the real request shape, which made
it *feel* like a measurement of the real thing, and it still generalised from four samples to
fifty-two and was wrong. Where a run-scale figure is available, it is the only one that counts.

---

### Adaptive thinking cannot be given a workable ceiling on the paper layer

Three attempts to measure effort *with* thinking enabled, all on `LLMPaper` at `Max` effort,
sequential, Sonnet 5. None produced a usable run — and the failure is structural, not a
matter of picking a bigger number.

| `PAPER_MAX_TOKENS` | Thinking tokens used | Paper layer output | Outcome |
|---:|---:|---|---|
| 16,000 | 16,000 (the whole ceiling) | **0 answer tokens** | `Truncated` — `stop_reason max_tokens, 0 characters of reply` |
| 32,000 | 32,000 (the whole ceiling) | **0 fields** — `inferred_from_paper: 0`, not even declared absences | Run completed, $0.4184 billed, paper layer contributed nothing |
| 128,000 | — | — | Never returned: 600 s client timeout, retried as a `Transport` error |

**Adaptive thinking expands to fill whatever ceiling it is given.** Doubling 16k to 32k
doubled the thinking and still left zero tokens for the answer. The 30,000-character paper
prompt gives it more than enough to reason about; there is no ceiling at which it
spontaneously stops and writes.

**And the ceiling that might let it finish is unreachable sequentially.** Generating 128,000
tokens takes far longer than the client's 600-second request timeout (`claude.rs:145`), so
every attempt times out and — because a timeout is a retryable `Transport` error — retries up
to five times, ~50 minutes, billing server-side each time with nothing recorded locally. The
ceiling required for the answer to survive is above the ceiling that can be generated inside
the timeout. **Sequential and `Max`-effort adaptive thinking are therefore incompatible on
this layer at any setting.**

**Batching is the only remaining path** and is untested. `PAPER_BATCH = true` removes the
per-request client timeout entirely — the API's own window is 24 hours — so a long-thinking
call can complete. It is also 45% cheaper and, at five studies, only four calls, so batch
latency costs little. Until that runs, effort × thinking remains unmeasured.

**One thing this did confirm.** The 32k run billed **$0.4184 across 6 calls, and the ledger
recorded all of it** — including the paper call that produced nothing. Before the
billed-failure fix in section 6, that call would have returned early and been invisible.

---

### Splitting the model per layer does not pay at this scale

`ModelConfig` is per layer, so the obvious moves are a cheap model on the per-sample layer and
a strong one on the per-study layer. Both were measured over the five-study set. **All rows
below are sequential**, so transport is held constant:

| order | naive / paper | n | real/rec | text | paper | missing | verbatim | cost |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `D H N P` | Sonnet / Sonnet | 2 | 35.50 ±0.30 | 2.52 | 1.48 | 306 | 44%, 58% | $0.5019 |
| `D H P N` | Sonnet / Sonnet | 2 | 36.15 ±0.14 | 1.79 | 2.87 | 294 | 65%, 40% | **$0.4222** |
| `D H N P` | Sonnet / **Opus** | 1 | 35.25 | 2.25 | 1.50 | 297 | 66% | $0.8029 |
| `D H P N` | Sonnet / **Opus** | 1 | 36.02 | 1.94 | 2.58 | 321 | **81%** | $0.7181 |
| `D H N P` | **Haiku** / Opus | 1 | 34.35 | 2.19 | 0.65 | 369 | 45% | $0.6234 |
| `D H P N` | **Haiku** / Opus | 1 | 36.92 | 2.92 | 2.50 | 421 | 51% | $0.6482 |

**Opus on the paper layer buys nothing and costs 60–70% more.** Against pure Sonnet it scores
−0.25 and −0.13 real values per record — both *inside* the pure-Sonnet run-to-run spread, so
indistinguishable. It slightly underperformed on its own layer, 2.58 against Sonnet's 2.87.

**Haiku on the naive layer still costs more here, +24% and +54%**, because Opus is 2.5× Sonnet
on the paper layer and pays the repeated 6,707-token prefix writes at Opus's input rate. At 52
naive calls against 4 paper calls, the paper premium beats the naive saving. It also declares
far more absences — 369/421 against ~300 — filling the MIxS environmental triad and `host`
with "not applicable" on cell-line studies at full token price.

**The one real improvement, with a catch.** `D H P N` Sonnet/Opus hit **81% verbatim**, the
best of any run. But the denominator moved: **96 `quoted` claims against ~197** for pure
Sonnet. Opus claims `quoted` about half as often, so it gets a higher fraction right while
producing *fewer* verified-verbatim values outright — 78 against ~104. That is better-
calibrated labelling, not better data, and the deferred demotion pass targets the same problem
for free.

**None of this extrapolates to corpus scale.** Here the mix is 52 naive : 4 paper calls; at
corpus scale it is 88,608 : 353, where the naive layer is 97.5% of the bill and halving it
dominates everything else. The corpus projection still favours Haiku on the naive layer ($157
against $285). The five-study set is the wrong scale to answer a question about the layer that
scales.

### The ordering result survives every model choice

| naive / paper | `D H P N` − `D H N P` |
|---|---:|
| Sonnet / Sonnet | +0.65 |
| Sonnet / Opus | +0.77 |
| Haiku / Opus | +2.58 |

Three independent model configurations, same direction each time — a stronger case for the
ordering than the replicate p-value alone, since it now holds across conditions rather than
only across repeats. See [layer_order_findings.md](layer_order_findings.md) §5.

Two caveats on the verbatim column throughout: n=1 for every Opus row against a pure-Sonnet
spread of 40–65%, wide enough to swallow most of the gap; and the audit artifact in section 4
of [layer_order_findings.md](layer_order_findings.md) — 65% of `NOT VERBATIM` findings are
declared absences mislabelled as quoted — penalises the high-`missing` rows hardest, which is
exactly the Haiku/Opus pair.

---

## 2. How big SRA is — and two counting traps

| source | count | what it actually is |
|---|---:|---|
| ENA portal `result=study` | **1,076,910** | INSDC study objects (SRP/ERP/DRP) |
| NCBI `db=bioproject` | 1,083,295 | BioProjects — agrees to within 0.6% |
| 46,059,823 ÷ 37.7 exp/study | ~1,221,700 | implied by our own harvest ratio |

**Trap 1: `esearch db=sra` counts experiments, not studies.** It returns
**46,059,823** — one UID per SRX. Treating that as a study count overstates by ~43×. This
is the same property that makes `search_studies` cluster and de-duplicate experiment hits
down to studies.

**Trap 2: ENA's `read_study` count is not studies either.** It reports 43,527,467; a shape
check shows the result returns one row per *run* (`study_accession`, `run_accession`
pairs). Use `result=study`.

Working figure: **~1.08 million studies, ~46 million experiments.**

---

## 3. Archive-scale projection

Scaling the corpus plan by 1,076,910 ÷ 346 = ×3,112:

| scenario | naive | paper | total |
|---|---:|---:|---:|
| every study = our corpus average | $863,769 | $22,124 | **~$886,000** |
| SRA's real volume, every study OA | $125,025 | $22,124 | **~$147,000** |
| SRA's real volume, real ~17% OA rate | $125,025 | $4,013 | **~$129,000** |

### The corpus average is not SRA's average

| | records per study |
|---|---:|
| our OA corpus | **295.5** |
| SRA overall (46.06M experiments ÷ 1.08M studies) | **42.8** |

The corpus is **6.9× heavier per study** than SRA at large. Assuming every SRA study looks
like ours implies 318 million records against the ~46 million SRA actually holds, which is
the entire gap between the $886k and $147k rows. The OA filter selected for substantial
studies — unsurprising, since studies that produce papers tend to be larger.

Three corrections all push the real figure below every number in that table: the estimator
runs high on the batched path (section 4), `D H P N` measured 9–16% cheaper than the
`D H N P` these figures assume, and the estimator prices both orderings identically because
it cannot see paid-layer ordering at all.

---

## 4. The estimator wrote the cached prefix once per run; it is once per study

**Fixed 2026-08-21 in `estimate.rs::with_cache`.** The sequential branch budgeted a single
cache write for an entire run:

```rust
let writes = if batch { (self.groups + self.wide_groups) as u64 } else { 1 };
//                                                                      ^ wrong
```

Measured across eight five-study runs, the prefix is written **3.4–6.4 times, never once** —
about one per study. `Layer::process` runs per project, so the other layer's calls sit
between this layer's and the five-minute cache TTL lapses in the gap. The batch branch was
already fitted to a measured run and needed no change; `wide_groups` stays batch-only,
because sequential calls cannot race each other for the same cold entry.

**Why it surfaced on Opus and not before.** A cache write bills at **1.25× the input rate** —
$6.25/MTok on Opus against $2.50 on Sonnet. The same modelling error costs 2.5× more there,
enough to break through the deliberately loose output over-estimate that had been hiding it.
Decomposed on the run that exposed it:

| component | estimated | actual | ratio |
|---|---:|---:|---:|
| input | 90,454 | 79,535 | 0.88× |
| **cache_write** | **12,876** | **54,196** | **4.21×** |
| cache_read | 334,692 | 213,450 | 0.64× |
| output | 21,760 | 22,301 | 1.02× |

The 41,320 excess write tokens account for essentially the whole $0.135 shortfall.

### Calibration, before and after the fix

| run | mode | old est | new est | actual | old err | new err |
|---|---|---:|---:|---:|---:|---:|
| `D H N P` Sonnet | seq | 0.4976 | 0.6007 | 0.5187 | −4.1% | +15.8% |
| `D H N P` Sonnet | seq | 0.4976 | 0.6007 | 0.4851 | +2.6% | +23.8% |
| `D H P N` Sonnet | seq | 0.4976 | 0.6007 | 0.4308 | +15.5% | +39.4% |
| `D H P N` Sonnet | seq | 0.4976 | 0.6007 | 0.4137 | +20.3% | +45.2% |
| `D H N P` Opus/Haiku | seq | 0.5131 | 0.6571 | 0.6234 | **−17.7%** | +5.4% |
| `D H P N` Opus/Haiku | seq | 0.5131 | 0.6571 | 0.6482 | **−20.8%** | +1.4% |
| `D H N P` Sonnet, `Max` | seq | — | 0.6007 | 0.5151 | — | +16.6% |
| `D H P N` Sonnet, `Max` | seq | — | 0.6007 | 0.3992 | — | +50.5% |

Every sequential run is now over-estimated. **The batched figures are unchanged** — that
branch was already correct — which means every corpus and SRA projection in sections 1–3
stands as written, since all of them are batched.

`effort` was investigated the same day and found *not* to need modelling — the estimator
ignores it, correctly. See section 1 for the probe that briefly suggested otherwise.

**One batched run still came in 1.1% under** (`N D H P`, $0.0892 estimated against $0.0902),
so "never under" remains a claim this estimator cannot make. Treat a pre-flight number as a
good forecast with `Budget` as the thing that actually stops a run.

### The full batched picture

Every measured run against its own pre-flight estimate:

| run | transport | estimate | actual | error |
|---|---|---:|---:|---:|
| `D H N P` | batched | $0.3145 | $0.2695 | +16.7% |
| `D H P N` | batched | $0.3145 | $0.2430 | +29.4% |
| `D N H P` | batched | $0.3254 | $0.2644 | +23.1% |
| `D N P H` | batched | $0.3262 | $0.2516 | +29.7% |
| `D P H N` | batched | $0.3153 | $0.2291 | +37.6% |
| `D P N H` | batched | $0.3262 | $0.2343 | +39.2% |
| `H D N P` | batched | $0.3262 | $0.2402 | +35.8% |
| **`N D H P`** | batched | $0.0892 | $0.0902 | **−1.1%** |
| `D H N P N` | batched | $0.5398 | $0.5202 | +3.8% |
| **`D H N P` rep2** | sequential | $0.4976 | $0.5187 | **−4.1%** |
| `D H P N` rep2 | sequential | $0.4976 | $0.4308 | +15.5% |
| `D H N P` rep3 | sequential | $0.4976 | $0.4851 | +2.6% |
| `D H P N` rep3 | sequential | $0.4976 | $0.4137 | +20.3% |

These are the pre-fix figures, kept because they are what the sequential rows were measured
against. Post-fix, sequential ranges +1.4% to +45.2% and batched is unchanged at −1.1% to
+39.2%.

**Consequence for the spend guard.** `confirmation_prompt` prices a plan and asks for a typed
`y`. Even corrected, that price is a forecast and not a ceiling — `Budget` remains the thing
that actually stops a run.

---

## 5. Batching costs 1.7–1.9× less, and hours more

Same orderings, same records, both transports:

| ordering | batched | sequential | ratio |
|---|---:|---:|---:|
| `D H N P` | $0.2661 | $0.5019 | 1.89× |
| `D H P N` | $0.2430 | $0.4222 | 1.74× |

Against wall time: four sequential runs finished in **12 minutes total** (~3.5 min each);
single batched runs of the same 56 calls ranged **43 minutes to 4 hours 7 minutes**. Batch
turnaround is queue-dependent and unrelated to the work — the 90-call ordering finished in
47 minutes while a 56-call one took over four hours. Measured on the process: 37 seconds of
CPU across 12 hours of wall time.

**So: sequential while iterating, batched at scale.** The discount only matters once the
run is large enough that its wall time no longer gates the work.

---

## 6. What the guards actually bound

Four independent limits exist. Their strength is not equal, and it depends on transport.

| guard | bounds | weakness |
|---|---|---|
| `SPEND=1` | paid layers do not exist without it | none — verified no-op with it unset and with `SPEND=0` |
| `confirmation_prompt` | prices the real plan, requires typed `y`, EOF declines | the price can be ~4% low (section 4) |
| `Budget` / `max_spend` | refuses the next call once the ledger is full | see below |
| `max_studies` / `max_total_records` | volume, before layers run | none |

**Failed calls were invisible to the ledger until 2026-08-21.** `Budgeted::complete` did
`self.inner.complete(request)?` — the `?` returned before `record`, so a call that the
provider billed but that failed to parse was never counted. A refusal, a malformed reply and
a clipped one all arrive as HTTP 200 with tokens already generated and charged. Measured: a
run reported **$0.0520 against roughly $1.00 actually spent**, because five paper-layer calls
clipped at `max_tokens` and none reached the ledger — `max_spend` of $1.00 never fired because
it could not see the money. `parse` now carries usage out on every billed error,
`ModelError::billed_usage()` exposes it, `Budgeted` records it, and `Retrying` sums it across
attempts. The batch path already did this correctly; it was a sequential-path hole, like the
cache-write bug.

**`Budget` is materially weaker on the batched path.** `Budgeted::complete_many` calls
`Budget::check`, which refuses only when the ledger is *already* full. `Budget::reserve` —
written precisely to stop "a $50 batch on a $1 budget with $0.99 remaining" — is not
reachable through `Budgeted`; only a direct `run_batch` caller uses it. So the batch that
crosses a ceiling is bounded by its own size, not by the limit. One batch is one project's
jobs for one layer: about **$0.045** (naive) or **$0.022** (paper) at this study size, but
it scales with the project.

Sequential has no such gap: `Budgeted::complete` checks before every individual call, so
overshoot is bounded by one call.

**Practical rule.** For a hard aggregate ceiling across several runs, share one `Budget`
across every client and set it *below* the authorised figure by at least one batch. Set a
per-run limit above the expected cost, never on top of it — a limit that sits at the
expected cost stops runs part-way and wastes them.

---

## 7. Wall time is the binding constraint, not money

SRA-wide reconstruction needs ~276 million model calls, but the model layers are not the
blocker. Everything upstream is:

* the corpus build ran **~2.5 hours for 346 studies**, almost all of it waiting on SRA —
  extrapolating to **~320 days** for 1.08M studies;
* harvesting measured **1.13s per study requested** (latency-bound, ~75–90% of wall time
  inside HTTP responses), so the shortlist alone is ~2 weeks;
* NCBI's rate limits cap this, and no budget relaxes them.

A bigger budget buys nothing until the ingestion side is parallelised or replaced with bulk
downloads.

---

## What is not measured

* **Whether any of this is worth it.** Every figure is cost and coverage. Per-field
  accuracy needs the hand-labelled gold set that does not exist yet.
* **Cost on any study set but this one.** Unit costs come from five studies of three
  organisms; the corpus projection assumes they generalise across 346, and the SRA
  projection assumes 346 generalise across a million.
* **Whether the ~17% OA rate holds archive-wide.** It is measured on the 1,858-study
  reference harvest (305 OA), and publication linkage varies sharply by archive — SRP
  17.3% OA, ERP 8.4%, DRP 23.7%.
* **Effort with thinking enabled.** Still unmeasured, but no longer for want of trying: three
  ceilings (16k / 32k / 128k) all failed, and sequential runs cannot reach a workable one at
  all (section 1). A batched attempt is the open experiment. Thinking itself was measured
  separately and lost on both models tried, but that was at default effort, so the
  *combination* remains a blank.
* **Model choice at scale.** Every model comparison in section 1 is n=1 per configuration on
  five studies with a 52:4 call mix. The layer that actually scales is the naive one, and no
  configuration has been measured at a mix resembling the corpus's 88,608:353.
* **Whether Opus would help a layer it was given more to do.** It was measured only on the
  paper layer, which is 4 calls and 2.5% of the bill.
* **Non-Anthropic providers.** The `Model` trait is provider-neutral and nothing here has
  been priced against a local or third-party model.
