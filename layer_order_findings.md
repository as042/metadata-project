# Layer Ordering — What We Measured

Findings from running every permutation of the reconstruction cascade against real
studies. All nine runs are the **Rust implementation** (`rust/`), over the same five
studies and 52 experiment records as the `datasets/test2` comparison set, on Claude
Sonnet 5 with thinking disabled and both model layers batched. Saved runs are in
`rust/runs/2026082*.json`, each carrying its ordering in `params.note`.

Layers are named rather than numbered — `Direct`, `Harmonized`, `LLMNaive`, `LLMPaper`,
abbreviated `D H N P` — because the ordering is a property of the layer list and the
numbers stop meaning anything the moment one is inserted or moved. See
[rust/README.md](rust/README.md).

**The one-line summary:** `D H P N` — asking the paper before the per-sample text layer
— beats the current `D H N P` ordering by **+0.76 real values per record (p = 0.014)**
while costing 9–16% less, replicated seven times; running `Harmonized` after any model
layer destroys it while leaving the totals almost unchanged; and paid-layer ordering
moves the bill in a direction the pre-flight estimator is structurally blind to.

---

## 1. The experiment

Nine orderings: the six with `Direct` first, plus three degenerate cases. `Direct` is
the only layer that creates records, so anything scheduled before it sees an empty slice
and does nothing — that leaves 3! = 6 real permutations, not 4! = 24.

Total spend **$2.3426 over 486 calls** for the nine-way sweep, against a $4.00 aggregate
ceiling, plus **$1.8482 over 224 calls** for the replicates in section 5 (unbatched,
$2.50 ceiling) — **$4.19 in all**. Guards used on both, all four independent:

* `SPEND=1`, verified to no-op with the variable unset and with `SPEND=0`.
* A **global** `Budget` shared by every client in every ordering — the real ceiling.
* A **per-ordering** `Budget` of $0.50 nested beneath it.
* A pre-flight `estimate::for_corpus` gate per ordering, skipping any run the global
  ledger lacked room for. None were skipped.

**The per-run budget alone would not have been enough.** On the batched path
`Budgeted::complete_many` calls `Budget::check`, which refuses only when the ledger is
*already* full; `Budget::reserve`, written precisely to stop "a $50 batch on a $1 budget
with $0.99 remaining", is unreachable through `Budgeted` and is only used by a direct
`run_batch` caller. So the batch that crosses a ceiling is bounded by its own size, not
by the limit. Here that is one project's jobs for one layer — about $0.045 (naive) or
$0.022 (paper) — which is why the global ceiling was set $0.50 below the authorised
figure.

---

## 2. Results

| order | Direct | Harm. | text | paper | real/rec | missing | verbatim | cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `D H N P` baseline | 1508 | **130** | 125 | 69 | 35.23 | 271 | 65% | $0.2695 |
| `D H P N` | 1508 | **130** | 85 | 140 | **35.83** | **240** | 60% | **$0.2430** |
| `D N H P` | 1508 | 2 | 249 | 48 | 34.75 | 252 | 93% | $0.2644 |
| `D N P H` | 1508 | 0 | 251 | 91 | 35.58 | 336 | 94% | $0.2516 |
| `D P H N` | 1508 | 22 | 95 | 251 | **36.08** | 311 | 50% | **$0.2291** |
| `D P N H` | 1508 | 0 | 105 | 215 | 35.15 | 290 | 43% | $0.2343 |
| `H D N P` | 1508 | 0 | 249 | 85 | 35.42 | 311 | 72% | $0.2402 |
| `N D H P` | 1508 | 130 | **0** | 169 | 34.75 | 139 | 29% | $0.0902 |
| `D H N P N` | 1508 | 130 | 184 | 37 | 35.75 | 274 | 51% | $0.5202 |

**Every row here is a single run.** Section 5 replicates the top two and shows which of
these columns survive: the `real/rec` ordering does, the `missing` column does not.

Counts are **real values only**; declared absences are the separate `missing` column.
Mixing the two is how a coverage comparison ends up measuring noise: in the
Python-vs-Rust comparison, 273 of 276 apparently-unfilled slots were declared-missing
terms rather than gaps.

---

## 3. Ordering destroys `Harmonized`, and the totals hide it

Run `Harmonized` after any model layer and it collapses from **130 fields to 2, 0, 22,
0**. Tracing where those fields went, on `D N H P` against the baseline:

| field | baseline H / text | `D N H P` H / text |
|---|---:|---:|
| strain | 44 / 0 | 1 / 43 |
| cell_line | 22 / 3 | 0 / 22 |
| tissue_type | 22 / 1 | 1 / 8 |
| dev_stage | 15 / 0 | 0 / 15 |
| treatment | 14 / 24 | 0 / 44 |
| cell_type | 13 / 24 | 0 / 26 |
| **total** | **130 / 52** | **2 / 158** |

The model is paid to transcribe attribute-bag values the synonym table reads for free.

**Total coverage barely moves** — 34.75 against the baseline's 35.23 — so a comparison
that only counts filled fields calls this harmless. It is not. Those values now carry
model provenance instead of `Harmonized`: a deterministic value from the submitter has
been swapped for an inferred one, at token cost, with no gain in count. This is the
strongest argument for the current ordering and it is invisible in any aggregate.

The effect is deterministic, not statistical. It reproduced in all four orderings that
place `Harmonized` after a model layer.

---

## 4. A high verbatim rate can mark wasted spend

`D N H P` and `D N P H` score **93% and 94%** — the best of the nine — for the reason in
section 3. The model is quoting attribute-bag values word for word, which is genuinely
verbatim and entirely unnecessary. The baseline's 65% is *better work* at a worse score.

So the verbatim rate is a measure of whether a `quoted` label is honest, and not of
whether the layer should have been asked at all. Read it beside the provenance split,
never on its own.

---

## 5. `D H P N` is a real improvement, replicated

Seven runs of the top two orderings — four of `D H N P`, three of `D H P N`, each with
both batched and sequential transports represented:

| ordering | n | real/rec | sd | range |
|---|---:|---:|---:|---|
| `D H N P` | 4 | 35.28 | 0.33 | 34.90 – 35.71 |
| `D H P N` | 3 | **36.04** | 0.21 | 35.83 – 36.25 |

**+0.76 real values per record. Welch t = 3.69, df = 5, two-tailed p = 0.014.** The
ranges do not overlap: every `D H N P` run scored below every `D H P N` run.

The mechanism is visible in the layer split — asking the paper first moves answers from
the 52-call per-sample layer onto the 4-call study-wide one, so the study-level facts a
paper states plainly get recorded once instead of being partially reconstructed per
sample.

**The transport does not explain it.** These runs mix batched and sequential calls, so
the gap was re-checked within each:

| transport | `D H N P` | `D H P N` | gap |
|---|---:|---:|---:|
| batched | 35.07 | 35.83 | +0.76 |
| sequential | 35.50 | 36.15 | +0.65 |

Both match the pooled +0.76.

**It is also cheaper, in both transports:**

| transport | `D H N P` | `D H P N` | saving |
|---|---:|---:|---:|
| batched | $0.2661 | $0.2430 | 8.7% |
| sequential | $0.5019 | $0.4222 | 15.9% |

### What did *not* replicate

The single-run sweep showed `D H P N` with 31 fewer declared absences (240 against 271),
and that was noise. Across replicates the two are indistinguishable — **277.8 against
275.7** — with a within-ordering range of 227–314 that swamps any difference between
them. The gain is in real values only.

Verbatim rate is likewise unresolvable at this sample size: `D H N P` scored 44%, 58%,
65% and `D H P N` 40%, 60%, 65%. The within-ordering spread is larger than any gap
between them.

### It also holds across model choices (added 2026-08-21)

The replicates above vary only the random seed. Varying the *models* instead, sequential
throughout, gives the same answer three times:

| naive / paper | `D H P N` − `D H N P` |
|---|---:|
| Sonnet / Sonnet | +0.65 |
| Sonnet / Opus | +0.77 |
| Haiku / Opus | +2.58 |

The gap widens as the paper layer's model gets stronger relative to the naive layer's, which
is what the mechanism predicts: a study-wide answer is worth more when it is asked first and
the model asking it is the better one. Each of these is n=1, so the *sizes* are not
established — only the direction, which is consistent across every condition tried. Costs and
the full table are in [cost_findings.md](cost_findings.md) §1.

### An unresolved observation

Sequential runs scored higher than batched ones in *both* orderings — +0.43 and +0.32.
Batching should not change a model's output; it defers the identical request. But the
batched runs happened earlier than the sequential ones, so transport is confounded with
time here and this experiment cannot separate them. Not a finding, and worth remembering
before pooling runs of different transports for anything finer-grained than the effect
above.

---

## 6. The estimator cannot see paid-layer ordering

A free pre-flight sweep priced `D H N P` and `D H P N` at **exactly the same $0.3145**,
byte-identical in input, cache and output tokens. They actually billed **$0.2695 and
$0.2430**.

The cause is structural. In `estimate::for_corpus`, a paid layer gets
`layer.estimate(&project, &schemas)`, which does not mutate `schemas`, while only free
layers get `layer.process(…, &mut schemas, …)`. So **every paid layer is planned as
though no other paid layer ran**, against whatever state the free layers left.

This is the right conservative bound for a spend guard — it cannot know what a model
will answer, and assuming a later layer gets no help errs high, never low. But it means:

* a free estimate sweep answers "does this ordering work" and prices *free*-layer
  placement, and cannot rank paid-layer orderings at all;
* the real spread across orderings is **$0.2291–$0.2695, about 15%**, entirely invisible
  before the run.

What the estimate *can* see is `Harmonized`'s position: placing it before `LLMNaive`
saves $0.0109, and before `LLMPaper` only $0.0008 — and the paper case has an
interesting shape, gaining input tokens (48,750 → 48,781, a larger `established()`
context block) while losing more output tokens (1,536 → 1,376).

---

## 7. Degenerate orderings fail visibly

| case | result |
|---|---|
| `H D N P` — free layer before `Direct` | Silent no-op. 0 harmonized fields, **no error raised**. Cost $0.0117 more than the baseline in the estimate, and the models absorbed the work. |
| `N D H P` — **paid** layer before `Direct` | The naive layer planned **zero calls**, $0.0000, and still printed its own line rather than vanishing from the report. Total run 4 calls, $0.0902. |
| `D H N P N` — repeated paid layer | Really does re-bill. See section 8. |

`N D H P` is worth a second look: with no text layer at all it still reached 34.75
real/rec, because `LLMPaper` absorbed the work and produced 169 values — its highest
anywhere. The layers substitute for each other more than the cascade's framing suggests.

`H D N P` raising no error is the one place the ordering list's "describes the pipeline
rather than policing it" stance has a cost. If that is ever to be caught, the pre-flight
estimate is where it would have to surface, since it is the only stage that sees the
whole plan before anything runs.

---

## 8. A repeated layer re-bills; the "conservative" estimate was nearly exact

`D H N P N` billed **$0.5202 against a $0.5398 estimate** — 90 calls against the
baseline's 56.

The prediction going in was that the second `LLMNaive` would find nothing open and cost
almost nothing, making the estimate a large over-count. That was wrong. The second pass
found **34 of 52 samples still had open fields** and asked about them, for a net gain of
27 real values at roughly double the spend. Do not assume a repeated layer degrades to a
no-op; the cascade leaves more open after one model pass than it looks.

---

## 9. Two audit artifacts to correct before quoting a verbatim rate

**65% of `NOT VERBATIM` findings are not quote failures.** Of 851 across the nine runs,
**552** are the model labelling a declared absence as `quoted` — `host = "not
applicable"`, `checklist = "not provided"`. A missing-value declaration is not a span
that can appear in the evidence, so scoring it as a broken quote is a category error
that depresses every rate in section 2 by an unknown amount. Excluding them is the first
thing to fix in `audit::verbatim`.

**The `placeholder` hallucination class is not confined to one study.** The literal
string `"placeholder"` was written as a value **19 times** across the nine runs, labelled
`quoted`, into `cell_line`, `treatment`, `checklist` and `cell_type`. Also observed:
`library_name = "Rample_name"` (an apparently mangled "Sample_name"). This is the same
class as the earlier `host_scientific_name = "Apis mellifera name for"` case, at higher
volume than one study suggested — and it is the argument for the deferred demotion pass
on unreliable `Quoted` labels.

---

## 10. Batch latency is not a function of the work

Wall time per ordering, all doing 56 calls except where noted:

| order | wall | order | wall |
|---|---:|---|---:|
| `D H N P` | 2h06m | `D P N H` | 43m |
| `D H P N` | 2h05m | `H D N P` | 38m |
| `D N H P` | **4h07m** | `N D H P` (4 calls) | **14m** |
| `D N P H` | 44m | `D H N P N` (90 calls) | 47m |
| `D P H N` | 48m | | |

**Identical work, 43 minutes to 4 hours.** The 90-call ordering finished in 47 minutes
while a 56-call one took over four hours. Measured on the process: **37 seconds of CPU
across 12 hours of wall time, 0.2% CPU** — it is not compute, not the debug build and
not corpus cloning; it is queue wait.

Two structural notes. `Layer::process` runs per project and `process_batched` builds one
batch per project, so an ordering is **nine sequential batch round-trips** (five naive,
four paper) and its wall time is their *sum*, not their max. And `Claude::complete_many`
passes a no-op progress callback, so nothing is observable between submission and
collection — a long run gives no signal that it is alive.

An earlier `main.rs` run of the same config finished in ~10 minutes. That was a lucky
draw, not a baseline.

**Sequential is the fix when latency matters.** The four replicate runs in section 5 were
unbatched and took **~3.5 minutes each — 12 minutes for all four**, against 43 minutes to
4 hours for one batched run of the same 56 calls. The trade is the discount: those runs
billed $0.42–$0.52 against $0.24–$0.27 batched, roughly 1.8×. For an experiment being
iterated on, sequential is usually the better deal; for a corpus run, batching is.

---

## What is not measured

* **Correctness.** Every number here is coverage, cost or label honesty. Whether a
  reordered run's values are *right* needs the hand-labelled gold set that does not
  exist yet. `D P H N`'s extra fields could be better answers or more confident wrong
  ones.
* **n = 1 for seven of the nine orderings.** Only `D H N P` (n=4) and `D H P N` (n=3)
  are replicated. For the other seven, any difference under ~0.7 real values/record
  should be treated as unresolved.
* **One study set, three organisms** — *Mus musculus* (31 records), *C. elegans* (15),
  *H. sapiens* (6). The MIxS environmental triad never fires here, so nothing about
  environmental-field behaviour under reordering is established.
* **Whether `D H P N` holds up beyond this study set.** The ordering gain is solid on
  these five studies; nothing establishes it on a different corpus slice, and the
  per-study spread on the model layers is several times the effect size.
