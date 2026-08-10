# Layer 3 on Haiku — What We Measured

Findings from building and tuning the text-inference layer (`reconstruct.infer_from_text`)
against Claude Haiku 4.5. Everything here is measured on real runs over the
`datasets/test2` study set (5 studies, 52 experiment records) unless noted.

**The one-line summary:** output tokens are the entire cost story, they cannot be
cached, and nothing we tried on the prompt or in layer 2 reduced them — every
lever that improved *quality* cost money rather than saving it. The one thing
that cut the bill was the Batch API (section 6), at 28% rather than its
advertised 50%, because batching and prompt caching work against each other.

---

## 1. Model choice

A sweep over one 6-sample study, same prompt and schema throughout:

| config | $/sample | output tok | vs Opus |
|---|---:|---:|---:|
| opus-5, medium effort, thinking | $0.0164 | 3,203 | 1× |
| opus-5, low effort, thinking | $0.0115 | 2,018 | 1.4× |
| sonnet-5, low effort, thinking | $0.0045 | 1,165 | 3.6× |
| sonnet-5, low, no thinking | $0.0047 | 1,290 | 3.5× |
| **haiku-4.5, no thinking** | **$0.0017** | 896 | **9.6×** |

**Effort alone buys 1.4×; the model buys 9.6×.** Output bills at 5× input on
every model, and Opus charges $25/MTok against Haiku's $5 — so the output *rate*
matters more than the output *volume*, and only a model change moves it.

Two constraints on Haiku 4.5 that force settings elsewhere in the stack:

- It **rejects adaptive thinking** (`{"type": "adaptive"}` is a 400). It predates
  it — `budget_tokens` is its only thinking mode. Hence `TEXT_THINKING = False`.
- It **rejects the `effort` parameter** entirely. Hence `TEXT_EFFORT = None`.
- It **rejects `fallbacks`** — that parameter is Opus 5 / Fable 5 / Mythos 5 only.
  `claude.py` gates on `_FALLBACK_MODELS` for this reason; sending it
  unconditionally made the whole transport module silently Opus-only.

**Unmeasured:** whether Haiku's answers are as *good*. Field counts on identical
configs swung 32/35 (Haiku) and 55/71 (Opus) across reruns of the same study, so
the noise band is wider than most of the differences we'd want to read. Cost
numbers are stable to ~10%; quality numbers are not measurements yet.

---

## 2. Caching

### The minimum cacheable prefix is per-model, and not monotonic

| model | minimum prefix |
|---|---:|
| Claude Opus 5 | 512 tokens |
| Claude Sonnet 5 | 1,024 |
| **Claude Haiku 4.5** | **4,096** |
| Opus 4.6 / 4.5 | 4,096 |

**Below the floor, `cache_control` is a silent no-op.** No error, no warning —
just `cache_read: 0` forever. This bit us directly: the original ~923-token
prefix cached cleanly on Opus, and when we moved to Haiku for cost it silently
stopped, with nothing in the output to say so.

### What is and isn't in the cacheable prefix

Per call, measured:

| component | tokens | cacheable |
|---|---:|---|
| `TEXT_SYSTEM` | 173 → 3,974 (grew over the work) | yes |
| compiled JSON schema (~44 open fields) | ~525 | yes |
| per-sample evidence (title + abstract + attributes) | ~540–780 | **no** |

The compiled schema counts toward the cached prefix, which is why the very first
Opus run cached 923 tokens when `TEXT_SYSTEM` was only 173.

### Growing the prompt past the floor works — and barely helps

Taking `TEXT_SYSTEM` from 644 → 3,974 tokens put the prefix at ~4,499 and caching
engaged, confirmed by trace:

```
call   input  cache_w  cache_r
   1     259     4350        0
   2     259        0     4350
   3     259        0     4350
```

Across the full 57-call run: **36,089 written, 221,597 read**, uncached input down
from 72,525 → 29,486. And yet the *billed* input per call only fell ~5%, for two
reasons we did not anticipate:

1. **Per-sample evidence dominates.** It's uncached by construction and runs
   540–780 tokens/call, swamping the prefix saving.
2. **Each distinct open-field set is a distinct prefix** with its own cache write.
   Five studies × two call shapes (study-level and sample-level) means several
   writes, not one.

**Methodology warning:** a 4-call proof of this reported "MORE EXPENSIVE" because
the one-time cache write amortised over 4 calls is ~1,359 tokens/call of
overhead. Over 50 calls it's ~109. **Never validate caching on a handful of
calls** — the thing being measured is amortisation.

### Still unexploited

- **Second cache breakpoint for per-sample evidence.** The study abstract is
  re-sent uncached for every sample in the study — 24× over for a 24-sample
  study. A breakpoint after the evidence block would let a study's samples share
  it. `claude.py` supports one breakpoint (system only); this needs a second.
- **Batch API** — built and measured; see section 6. It is the only lever that
  moved cost, and it interacts badly with caching, so it belongs there rather
  than here.

---

## 3. Prompt construction

### Structured outputs enforce three separate limits

Asking about ~41 open fields trips all three, in this order:

| limit | what it kills |
|---|---|
| ≤ 16 union-typed parameters | `{"type": ["string", "null"]}` on every field |
| ≤ 24 optional parameters | one property per field |
| "schema is too complex" | 24 nested `{value, confidence}` objects |

**The shape that works is a list, not a property per field:**

```json
{"answers": [{"field": <enum>, "value": "…", "confidence": "high"}]}
```

One property, one item shape, field name as an `enum` — enums are cheap where
properties are not. The whole open set fits in one call with room to spare, and
declining a field is just not emitting an item. `value` is a string for every
field including the int and date ones, since `TargetSchema` coerces `"9606"` and
`"2015-03-04"` on assignment; per-field value types would reintroduce the union
limit for nothing.

This also removed the need for chunking. `FIELDS_PER_CALL` survives only as the
dial for the one-call-per-field experiment (set it to `1`).

### Field definitions work. They are the highest-value thing in the prompt.

Adding a FIELD DEFINITIONS block to `TEXT_SYSTEM`:

| error class | before | after |
|---|---:|---:|
| `host_scientific_name` filled with the study organism | 48/52 | **0/52** |
| `cell_type` duplicating `tissue_type` | 14 | 6 |
| environmental-context fields on lab animals | 4 | 0 |

The `host` error was the single largest, ran at 100% `high` confidence, and was
eliminated by one paragraph explaining that `host` means *the organism a sample
was taken from*. It was a definition problem, not a capability limit.

### Examples outweigh prose

Attempts to reduce output volume, in order:

| change | out/call |
|---|---:|
| short prompt (173 tok) | 163 |
| long prompt, "prefer a missing-value term over silence" | 821 |
| softened that instruction in prose | 591 |
| restructured examples around FILL / DECLARE / OMIT | 525 |

Softening the *instruction* while leaving five worked examples that each
enumerated four or five `not applicable` verdicts barely moved anything. The
examples were teaching the behaviour regardless of the surrounding prose.
Restructuring them — adding an explicit OMIT bucket, adding a sparse study that
declares nothing, adding "if you're declaring more than you omit, you're
guessing" — bought a further 11%, and **missing-value declarations still went up**
(325 → 346).

**Conclusion: prompt-side control of output volume plateaued.** Three attempts
took 821 → 525 against a 163 baseline. A longer prompt that defines 65 fields and
supplies a rich vocabulary for declaring things produces more answers, fairly
directly. The long prompt's value and its cost are the same property.

### Confidence is not yet a usable signal

| | high | medium | low |
|---|---:|---:|---:|
| before definitions | 391 | 41 | **0** |
| after | 362 | 100 | 4 |

Zero `low` in 432 inferences originally; 4 in 466 after defining the levels by
*what you did to get the answer* (copying / interpreting / choosing). The scale
went from 2 values to ~2.5.

Worse than uninformative in one place: **`host_scientific_name` was wrong on all
48 records and rated `high` on all 48.** Any threshold on confidence would have
kept exactly the wrong answers. Treat it as unvalidated until scored against
labelled data.

---

## 4. Token efficiency

Cost decomposition, Haiku 4.5 at $1 / $5 per MTok:

| run | out/call | $/record |
|---|---:|---:|
| A — short prompt | 178 | $0.00237 |
| B — short prompt + field definitions + `agent` synonym | 163 | **$0.00277** |
| D — long prompt, softened | 591 | $0.00507 |
| E — long prompt, restructured examples | 525 | $0.00474 |
| **E via Batch API** | 419 | **$0.00344** |

At corpus scale (~109k records): **$300 (B), $520 (E), $375 (E batched)** —
batching buys back most of the long prompt's cost, so all of E's quality lands
at 1.24x B's price rather than 1.7x.

**Output is 64–85% of the bill in every configuration, and cannot be cached.**
Input is effectively a solved problem once caching engages; it stops being where
the money is. Any future optimisation has to reduce answers emitted, not tokens
sent.

Two things that do *not* reduce output, both measured:

- Lowering effort (Haiku doesn't support it anyway).
- **Closing fields before the layer runs** — see below.

---

## 5. How layer 2 interacts with layer 3

### Layer 2 is free and accurate; layer 3 is neither

Hand-judged against live SRA on three samples:

| | correct | wrong | defensible |
|---|---:|---:|---:|
| harmonized (layer 2) | **6/6** | 0 | 0 |
| inferred (layer 3) | 9 | 8 | 8 |

Layer 2 is a normalisation pass plus a synonym table — no model, no network.
Casefolding and collapsing separators does most of the work for free
(`cell type` → `cell_type`, `STRAIN` → `strain`), so the table only carries the
genuinely different names.

### Closing fields does NOT reduce layer 3's cost

This was predicted twice and wrong twice. Enabling layer 2:

| | harmonized | inferred |
|---|---:|---:|
| layer 3 alone | 0 | 324 |
| + layer 2 (73 fields) | 73 | **436** |
| + layer 2 (148 fields) | 148 | **453** |

**Inferred count went up every time.** The model does not answer fewer questions
when given fewer slots — it redistributes the same effort onto whatever remains
open, including the fields it gets wrong.

**Layer 2 is a correctness and quality lever, not a cost lever.** Plan layer 4
gating on that basis.

### Schema gaps in layer 3's target cause layer 3's errors

The ENA field subset had no `age`, `treatment`, `cell_line`, or `dev_stage` —
four of the seven most common unmapped attribute keys, 79 of 175 occurrences.
With nowhere to put `treatment: "72 hours water deprivation"`, the model put it
in `broad_scale_environmental_context`.

Adding those four fields:

- environmental-context errors: **4 → 0**
- layer 2 yield: 73 → 148 harmonized
- `age` 24 harmonized, `treatment` 20, `cell_line` 16, `dev_stage` 15

`age`, `cell_line` and `dev_stage` are real ENA fields that simply weren't in the
starting 61-field subset (ENA exposes 195 for `read_run`; 60 of our 61 are
genuine ENA names). Only `treatment` is a true extension — recorded in
`schema.EXTENSION_FIELDS`.

### Migrate mappings downward as you find them

Layer 3 was correctly mapping `agent` → `treatment` on its own. Moving it into
`_SYNONYMS` made it free and deterministic instead of a paid guess. **Watch layer
3's correct inferences for synonyms layer 2 should own.**

The inverse also holds. `source_name` is the most common unmapped key (60 of 175
occurrences) and is deliberately *not* in the table: observed values include
`"Fibroblast"` (a cell type), `"Hypothalamus"` (a tissue), `"whole worms"` (an
organism) and `"rumen contents"` (an isolation source). Pinning it to one field
would be wrong about two thirds of the time. Value-dependent routing belongs in
the layer that can read the value.

### Study-level split: correctness, not tokens

Asking study-level fields once per *sample* produced ~106 duplicate asks over 60
samples and gave one study **two different `study_alias` answers** across its 15
records. Splitting them into one call per study fixes that by construction.

It is roughly **token-neutral**, not a saving: it removes the duplicate asks but
adds one call per study, and each added call re-pays the fixed prefix. It only
clearly wins on studies with many samples.

---

## 6. Batch API

**The 50% discount is real but nets 28%, because batching degrades caching.**

| | cache_write | cache_read | output | billed |
|---|---:|---:|---:|---:|
| live | 36,089 | 221,597 | 29,942 | $0.2465 |
| **batch** | 158,767 | 98,919 | 23,856 | **$0.1786** |
| batch + pre-warm | 117,889 | 139,797 | 22,974 | $0.1858 |

A cache entry only becomes readable once the request that wrote it has
returned, and a batch runs its requests in parallel — so most of them *write*
the shared prefix rather than reading it. Cache writes went up 4.4x and reads
fell 55%, which made the batch **45% more expensive before the discount** and
only 28% cheaper after it.

Same quality either way: coverage 33.1 vs 32.5, `host` error 0 in both. Wall
clock 9.1 min for five per-study batches (11.0 with pre-warm).

### Pre-warming does not pay here

The textbook remedy for parallel fan-out is to send one request, let it return,
then fire the rest. Implemented as one live call per distinct prefix, with its
answer kept so the call is not wasted.

**It works mechanically and loses on the economics.** Read/write ratio improved
0.62 → 1.19 and writes fell 26% — but the bill went *up*, $0.1786 → $0.1858:

- The 12 warm calls move off the 50% discount, and they are precisely the
  requests that pay the 1.25x cache write.
- One warm call does not serialise a parallel batch. The remaining 45 requests
  still wrote 81,800 tokens between them, so writes fell 26% rather than to
  near-zero.

`prewarm` therefore defaults to **False**. The option is kept because it is the
documented technique and would otherwise be re-attempted; re-measure before
enabling it on a different workload shape.

The `max_tokens=0` warm-up trick is unusable here regardless: the API rejects it
alongside `output_config.format`, and dropping the schema to get around that
would change the prefix and warm the wrong entry.

### Other batch notes

- **1-hour cache TTL** on this path rather than the default five minutes. A
  batch can take longer than 5 min to drain, and an expired prefix is re-written
  by every request still in flight. 2x on write beats fifty writes at 1.25x.
  Kept regardless of the pre-warm decision.
- **`fallbacks` is rejected by the Batches API.** Not an issue on Haiku, which
  does not accept it anyway, but it is why the live and batched request bodies
  must be built by one shared function.
- **`custom_id` has a restricted charset.** Pooled-experiment ids contain dots
  (`SRX1.SRS1`), so caller keys are replaced with positional ids and mapped back.
- **Batches are per study**, since `reconstruct()` is called per study. Batching
  across studies would need `dataset.save_reconstructed_records` to collect
  first; the discount is the same either way.

## 7. Open questions

- **Is Haiku good enough?** Unmeasured. Needs 50–100 hand-labelled samples and
  per-field accuracy scored per model. Run-to-run variance makes anything less
  uninterpretable.
- **Is confidence calibrated at all?** Currently ~90% `high` with the largest
  error class uniformly `high`. Same gold set answers it.
- **Second cache breakpoint** for per-sample evidence — still the largest
  untouched lever, and it gets *more* valuable under batching, where the
  uncached evidence is a bigger share of what is billed at full rate.
- **One call per field?** `FIELDS_PER_CALL = 1` runs it. Cost scales with the
  *output* side, which caching cannot offset, so expect it to be expensive.

---

## Appendix: settings and where they live

| setting | location | value | why |
|---|---|---|---|
| `TEXT_MODEL` | `reconstruct.py` | `claude-haiku-4-5` | 9.6× cheaper than Opus 5 |
| `TEXT_EFFORT` | `reconstruct.py` | `None` | Haiku rejects the parameter |
| `TEXT_THINKING` | `reconstruct.py` | `False` | Haiku rejects adaptive thinking |
| `FIELDS_PER_CALL` | `reconstruct.py` | `None` | array schema removed the need to chunk |
| `DEFAULT_EFFORT` | `claude.py` | `"medium"` | only applies if a model that supports it is used |
| `_FALLBACK_MODELS` | `claude.py` | Opus 5 / Fable 5 / Mythos 5 | others 400 on `fallbacks` |
| `TEXT_BATCH` | `reconstruct.py` | `False` | 12-28% cheaper when on, but gives up progress and interruptibility |
| `prewarm` | `claude.extract_batch` | `False` | measured more expensive, not less |
| `cache_ttl` | `claude.extract_batch` | `"1h"` | a batch can outlive a 5-minute prefix |
