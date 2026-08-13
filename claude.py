"""Claude API transport for the reconstruction layers.

A thin wrapper over the official ``anthropic`` SDK, holding the credential, the
shared client, and the two call shapes :mod:`reconstruct` needs:

    text = claude.complete("Summarise this study", system=INSTRUCTIONS)
    data = claude.extract(prompt, schema, system=INSTRUCTIONS)

:func:`extract` is the one the model layers want. It uses **structured outputs**
— the schema is enforced by the API, not requested in the prompt — so the reply
is always valid JSON with exactly the declared keys. That removes the whole
class of "the model wrapped it in a code fence" parsing failures, and it is why
this module takes a JSON Schema rather than asking for JSON politely.

Deliberately knows nothing about :class:`schema.TargetSchema`. Building the JSON
Schema for a batch of open fields is :mod:`reconstruct`'s job — it is the module
that knows which fields are open, what they mean, and which vocabulary each one
accepts. This module only moves bytes.

There is **no default credential file**: the account that pays is named per run,
via ``set_api_key(path=...)`` or the ``claude_key_file`` argument the pipeline
stages take. :func:`require_api_key` is the guard clause paid work calls first,
so a run with no key fails immediately instead of at the first request — by
which point a harvest has already happened.

Nothing happens at import time, so the module imports fine with no key present —
tests can stub :func:`_client` without a credential on disk, and the free layers
(1 and 2) never need one.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import anthropic

# The models this project can run, by name rather than by hand-typed string.
# Use these anywhere a model is selected — `reconstruct.TEXT_MODEL`, the
# `text_model=` / `paper_model=` arguments, a direct `complete()` call — so a
# typo is a NameError your editor catches rather than a string that travels
# until :func:`dataset.cost_multiplier` refuses to price it.
#
# The values are the API's own IDs and are what actually goes over the wire, so
# the raw strings keep working unchanged; these are for the caller's benefit,
# not the transport's. Every one of them must have a price in PRICES below —
# `test_every_named_model_is_priced` fails if the two lists drift apart.
HAIKU_4_5 = "claude-haiku-4-5"
SONNET_5 = "claude-sonnet-5"
OPUS_5 = "claude-opus-5"
OPUS_4_8 = "claude-opus-4-8"

MODELS = (HAIKU_4_5, SONNET_5, OPUS_5, OPUS_4_8)

# Models that think unless explicitly told not to. **Omitting the `thinking`
# field does not mean "off" on these** — it means adaptive, which is the
# opposite of what the older models do with the same omission.
#
# This cost real money to learn. A Sonnet 5 run configured `thinking=False`
# simply left the field out, thought anyway at ~1,341 output tokens/call, and
# billed $1.25 against a $0.49 estimate and a $1.00 cap. `_body` now sends an
# explicit `{"type": "disabled"}` for these, so the flag means what it says.
THINKS_BY_DEFAULT = (OPUS_5, SONNET_5)

# Claude Opus 5 — the current frontier model, and the one to keep unless there
# is a measured reason to move. Cost per study is dominated by how much context
# each call carries, not by the model tier.
#
# **This is the transport default only** — the fallback for a direct
# `complete()` / `extract()` call that names no model. Both pipeline layers pass
# their model explicitly, so changing this does *not* change what a pipeline run
# uses; `reconstruct.TEXT_MODEL` and `reconstruct.PAPER_MODEL` do that.
MODEL = OPUS_5

# How hard the model works per call, lowest to highest. Named for the same
# reason the models are: `EFFORT_XHIGH` beats remembering whether the string is
# "xhigh", "x-high", or "extra_high" (it is the first, and the other two are a
# 400). `None` means "send no effort at all", which is what the models that
# predate the parameter require.
EFFORT_LOW = "low"
EFFORT_MEDIUM = "medium"
EFFORT_HIGH = "high"
EFFORT_XHIGH = "xhigh"
EFFORT_MAX = "max"

EFFORT_LEVELS = (EFFORT_LOW, EFFORT_MEDIUM, EFFORT_HIGH, EFFORT_XHIGH, EFFORT_MAX)

# `medium` rather than the API's `high` default because reconstruction is one
# small call per sample over tens of thousands of samples, and effort is the
# main cost lever. **Sweep this against a labelled sample before trusting it** —
# if extraction accuracy is materially better at `high`, the extra spend is
# worth it, and this constant is the one place to change.
DEFAULT_EFFORT = EFFORT_MEDIUM

# Thinking tokens count against max_tokens, so this needs headroom well past the
# size of the JSON being returned. 16k stays under the SDK's non-streaming
# timeout guard; pass stream=True for anything larger.
DEFAULT_MAX_TOKENS = 16000

# Every Anthropic key starts with this. Cheap way to catch the wrong file being
# pointed at the credential — an NCBI key or a contact address fails it.
_KEY_PREFIX = "sk-ant-"

# There is deliberately **no default credential file**. An implicit default is
# silent in exactly the wrong direction: a run that named no key still found
# claude_api_key.txt sitting in the repo root and billed that account, so
# "I provided no key" and "no key was used" quietly stopped meaning the same
# thing. The only way to spend now is to say which credential pays.
API_KEY_FILE = None

# Opus 5 runs safety classifiers that can decline a request outright — a 200
# with stop_reason "refusal", not an exception. Unlikely on sequencing metadata,
# but a study abstract about a pathogen or a toxin is exactly the kind of benign
# text that can trip one. With this beta enabled, a declined request is re-run
# server-side on Anthropic's recommended fallback model inside the same call, so
# the pipeline gets an answer instead of a hole. Drop both this and `fallbacks`
# below if you would rather see the refusal.
_FALLBACK_BETA = "server-side-fallback-2026-07-01"

# ...but only the frontier models run those classifiers, and only they accept
# the parameter — every other model rejects it outright with a 400 ("does not
# support the `fallbacks` parameter"). Sending it unconditionally made this
# module silently Opus-5-only, which is exactly wrong for a layer whose cost is
# dominated by output tokens and whose obvious lever is a smaller model.
_FALLBACK_MODELS = ("claude-opus-5", "claude-fable-5", "claude-mythos-5")

_api_key: str | None = None
_client_instance: anthropic.Anthropic | None = None

# Where the live key came from, for reporting. Worth tracking because the key
# itself must never be printed, so without this there is no way to answer "which
# account is this run about to bill?" — and the default resolution is silent: if
# API_KEY_FILE happens to exist, a run that named no credential still gets one.
_api_key_source: str | None = None

# Cumulative token spend for this process, so a bulk run can report what it
# cost. Cache reads bill at ~10% of the input rate and cache writes at ~125%,
# so they are tracked apart rather than folded into `input`.
USAGE = {"calls": 0, "input": 0, "output": 0, "cache_write": 0, "cache_read": 0,
         # Batched tokens are counted apart because the Batch API's 50% discount
         # is a billing rate, not a token reduction — the counts look identical,
         # so folding them together would silently overstate the cost of a
         # batched run by 2x.
         "batch_calls": 0, "batch_input": 0, "batch_output": 0,
         "batch_cache_write": 0, "batch_cache_read": 0,
         # Dollars, accumulated per call at that call's own model rate.
         "cost": 0.0}

# Cache reads bill at ~10% of the input rate, cache writes at ~125%.
CACHE_READ_RATE = 0.10
CACHE_WRITE_RATE = 1.25

# $ per million tokens, (input, output), for the models this project uses.
# Keyed by the named constants above; the two must stay in step, because a model
# missing here cannot be costed and `dataset.cost_multiplier` refuses to run it.
#
# **SONNET_5 is on introductory pricing that ends 2026-08-31**, after which it
# is (3.0, 15.0) and every Sonnet estimate here is a third low. Update it then.
PRICES = {HAIKU_4_5: (1.0, 5.0), SONNET_5: (2.0, 10.0),
          OPUS_5: (5.0, 25.0), OPUS_4_8: (5.0, 25.0)}
BATCH_DISCOUNT = 0.5


class RefusalError(RuntimeError):
    """The request was declined by safety classifiers rather than answered.

    Not an SDK exception: a refusal arrives as a perfectly successful HTTP 200
    whose ``content`` is empty or partial, so code that reads ``content[0]``
    without checking ``stop_reason`` silently treats a refusal as an answer.
    Raised here so a caller cannot miss it.
    """

    def __init__(self, category: str | None, explanation: str | None):
        self.category, self.explanation = category, explanation
        super().__init__(f"request declined ({category or 'unspecified'}): {explanation or ''}")


class MissingAPIKeyError(RuntimeError):
    """No credential was named, or the one named is not usable.

    Its own type so callers can guard on it: a paid layer needs to refuse
    *before* it starts working, and a missing key is a caller mistake to fix,
    not a transient failure to retry.
    """


def set_api_key(key: str | None = None, path: str | None = None) -> None:
    """Name the credential to bill: either ``key`` directly or a file at ``path``.

    One of the two is required — there is no default file to fall back on, and
    calling this with neither is an error rather than a silent no-op. Resets the
    cached client so a new key takes effect immediately.
    """
    global _api_key, _client_instance, _api_key_source
    if key is not None and path is not None:
        raise ValueError("pass key= or path=, not both — which one pays is ambiguous")
    if key is None and path is None:
        path = API_KEY_FILE
        if path is None:
            raise MissingAPIKeyError(
                "no Claude API key named. Pass set_api_key(path='your_key.txt') or "
                "set_api_key(key=...), or set claude.API_KEY_FILE. There is no "
                "default credential file — layers 3 and 4 spend real money, so the "
                "account that pays has to be named explicitly."
            )
    resolved = key if key is not None else _read_key(path)
    _validate_key(resolved, "key=" if key is not None else path)
    _api_key = resolved
    _api_key_source = "(passed directly)" if key is not None else path
    _client_instance = None


def have_api_key() -> bool:
    """Whether a credential is loaded, without raising or reading anything.

    Lets a caller decide (free layers need no key) rather than discovering it
    from an exception partway through a run.
    """
    return _api_key is not None


def require_api_key() -> None:
    """Guard clause: raise unless a usable credential is already loaded.

    Call this before doing any paid work. It never reads a file and never falls
    back — if nothing has been named by now, that is the error.
    """
    if _api_key is None:
        raise MissingAPIKeyError(
            "no Claude API key configured, and the model layers cannot run without "
            "one. Name a credential file (claude_key_file='your_key.txt'), or turn "
            "the paid layers off with from_text=False, from_paper=False for a free "
            "run. Nothing has been spent."
        )


def key_source() -> str:
    """Which credential this process will bill, as a printable string.

    Never the key itself, and never a guess: if nothing is loaded this says so
    rather than naming a file that might not be the one that ends up paying.
    """
    return _api_key_source or "(none configured)"


def _validate_key(key: str, source: str) -> None:
    """Reject a credential that cannot possibly authenticate.

    Caught here rather than at the first API call because by then a harvest has
    already run. Whitespace is fatal (the key authenticates as a different
    string and every request 401s) and so is the wrong file entirely — pointing
    this at an NCBI key or an email is the mistake worth catching, and both fail
    the prefix. Relax ``_KEY_PREFIX`` if Anthropic ever changes the format.
    """
    if not key:
        raise MissingAPIKeyError(f"{source} is empty — no key to use")
    if any(char.isspace() for char in key):
        raise MissingAPIKeyError(
            f"{source} contains whitespace inside the key — it would authenticate "
            f"as a different string and 401 on every request"
        )
    if not key.startswith(_KEY_PREFIX):
        raise MissingAPIKeyError(
            f"{source} does not look like an Anthropic API key (expected it to "
            f"start with {_KEY_PREFIX!r}) — is it the NCBI key, or the wrong file?"
        )


def _read_key(path: str) -> str:
    """Read a one-line credential file.

    Stripped because a trailing newline is invisible in an editor but fatal —
    a key with whitespace in it authenticates as a different (invalid) string
    and every request 401s.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found — put the Claude API key there (keep it gitignored), "
            f"or call set_api_key(key=...) directly"
        )
    with open(path, encoding="utf-8") as file:
        return file.read().strip()


def _client() -> anthropic.Anthropic:
    """The shared client, built on first use.

    ``max_retries=5`` rather than the SDK's 2, matching the retry budget
    :func:`project._request_with_retry` uses against NCBI: a run of thousands of
    calls will hit a 429 or a 5xx, and one transient failure should not cost a
    study. The SDK backs off exponentially and only retries what is retryable.
    """
    global _client_instance
    if _client_instance is None:
        require_api_key()   # never resolves a key itself — see set_api_key
        _client_instance = anthropic.Anthropic(api_key=_api_key, max_retries=5)
    return _client_instance


def object_schema(properties: dict[str, Any], required: list[str] | None = None) -> dict:
    """Wrap field definitions into a JSON Schema the API will accept.

    Structured outputs reject a schema whose objects allow extra keys, and any
    property not in ``required`` may simply be omitted — so ``required`` defaults
    to *every* property. That is what makes the reply shape predictable: the
    caller gets the keys it asked for, and a field the model has no answer for
    comes back as an explicit null rather than a missing key.
    """
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties) if required is None else required,
        "additionalProperties": False,
    }


def _system_blocks(system: str | None, cache: bool, ttl: str | None = None):
    """Render the system prompt, marking it cacheable.

    The reconstruction layers send one large, unchanging instruction block —
    the field definitions and the vocabularies — followed by a small per-sample
    payload. Caching that prefix costs ~1.25x on the first call and ~0.1x on
    every one after, which on a per-sample run is the difference between paying
    for the instructions once and paying for them tens of thousands of times.
    Keep the system text **byte-identical** across calls or the cache misses.
    """
    if not system:
        return anthropic.NOT_GIVEN
    block: dict[str, Any] = {"type": "text", "text": system}
    if cache:
        block["cache_control"] = {"type": "ephemeral"}
        if ttl:
            # 1h costs 2x on write against 5m's 1.25x, but a batch can take
            # longer than five minutes to drain — and a prefix that expires
            # mid-batch is re-written by every remaining request, which is the
            # whole problem pre-warming exists to avoid.
            block["cache_control"]["ttl"] = ttl
    return [block]


def _body(prompt, system, schema, model, effort, max_tokens, thinking,
          cache_system, cache_ttl=None) -> dict:
    """The request body, identical for the live and batched paths.

    Shared deliberately: a batch that differs from the live call by so much as a
    whitespace change is a different cache prefix and a different measurement.
    """
    params: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": _system_blocks(system, cache_system, cache_ttl),
        "messages": [{"role": "user", "content": prompt}],
    }
    if thinking:
        # Adaptive only — the frontier models reject a fixed budget_tokens, and
        # older ones (Haiku 4.5) reject `adaptive`. Pass thinking=False there.
        params["thinking"] = {"type": "adaptive"}
    elif model in THINKS_BY_DEFAULT:
        # Leaving the field out would run adaptive thinking on these, so
        # `thinking=False` has to say so out loud. Only sent to the models that
        # need it: the older ones predate the field and reject it.
        params["thinking"] = {"type": "disabled"}
    output_config: dict[str, Any] = {}
    if effort:
        output_config["effort"] = effort
    if schema is not None:
        output_config["format"] = {"type": "json_schema", "schema": schema}
    if output_config:
        params["output_config"] = output_config
    return params


def _message(
    prompt: str,
    *,
    system: str | None,
    schema: dict | None,
    model: str,
    effort: str | None,
    max_tokens: int,
    thinking: bool,
    cache_system: bool,
    stream: bool,
    cache_ttl: str | None = None,
):
    """One request. Returns the response object; callers pull text out of it."""
    params = _body(prompt, system, schema, model, effort, max_tokens, thinking,
                   cache_system, cache_ttl)

    if model in _FALLBACK_MODELS:
        params["betas"] = [_FALLBACK_BETA]
        params["fallbacks"] = "default"
        messages = _client().beta.messages
    else:
        messages = _client().messages  # no betas needed; stay off the beta path

    if stream:
        with messages.stream(**params) as response_stream:
            response = response_stream.get_final_message()
    else:
        response = messages.create(**params)

    _record(response.usage, model=params["model"])
    if response.stop_reason == "refusal":
        details = getattr(response, "stop_details", None)
        raise RefusalError(
            getattr(details, "category", None), getattr(details, "explanation", None)
        )
    return response


def _record(usage, batch: bool = False, model: str | None = None) -> None:
    p = "batch_" if batch else ""
    USAGE[p + "calls"] += 1
    USAGE[p + "input"] += getattr(usage, "input_tokens", 0) or 0
    USAGE[p + "output"] += getattr(usage, "output_tokens", 0) or 0
    USAGE[p + "cache_write"] += getattr(usage, "cache_creation_input_tokens", 0) or 0
    USAGE[p + "cache_read"] += getattr(usage, "cache_read_input_tokens", 0) or 0
    # Priced per call, because a run can mix models across layers and the token
    # totals alone cannot be costed afterwards. Accumulated here so the figure
    # is the *billed* one rather than a second estimate — a run that overshoots
    # its estimate should say so on its own, not wait to be reconstructed by
    # hand from a pasted terminal line.
    USAGE["cost"] += call_cost(usage, model or MODEL, batch=batch)


def call_cost(usage, model: str, batch: bool = False) -> float:
    """Dollars for one response's usage, at ``model``'s rates.

    Cache reads bill at ~10% of the input rate and cache writes at ~125%, which
    is why they cannot be folded into the input count: on the Sonnet 5 run,
    cache writes alone were 15% of the bill.
    """
    rate_in, rate_out = PRICES.get(model, PRICES[MODEL])
    tokens = lambda name: getattr(usage, name, 0) or 0
    dollars = (
        tokens("input_tokens") * rate_in
        + tokens("cache_read_input_tokens") * rate_in * CACHE_READ_RATE
        + tokens("cache_creation_input_tokens") * rate_in * CACHE_WRITE_RATE
        + tokens("output_tokens") * rate_out
    ) / 1e6
    return dollars * (BATCH_DISCOUNT if batch else 1.0)


def _text(response) -> str:
    """Concatenate the text blocks, skipping thinking and any other block type."""
    return "".join(b.text for b in response.content if b.type == "text")


def complete(
    prompt: str,
    *,
    system: str | None = None,
    model: str = MODEL,
    effort: str | None = DEFAULT_EFFORT,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    thinking: bool = True,
    cache_system: bool = True,
    stream: bool = False,
    cache_ttl: str | None = None,
) -> str:
    """Send a prompt, return the reply as text.

    For anything whose answer has a shape, prefer :func:`extract` — the API can
    enforce the shape, which beats asking for it and parsing hopefully.

    ``thinking=False`` genuinely disables thinking, including on the models in
    :data:`THINKS_BY_DEFAULT` where merely omitting the parameter would leave
    adaptive thinking on. ``effort`` of ``None`` sends nothing, which means the
    API's own default (``high``) rather than "no effort" — the two are easy to
    confuse and cost very different amounts.
    """
    return _text(
        _message(
            prompt,
            system=system,
            schema=None,
            model=model,
            effort=effort,
            max_tokens=max_tokens,
            thinking=thinking,
            cache_system=cache_system,
            stream=stream,
            cache_ttl=cache_ttl,
        )
    )


def extract(
    prompt: str,
    schema: dict,
    *,
    system: str | None = None,
    model: str = MODEL,
    effort: str | None = DEFAULT_EFFORT,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    thinking: bool = True,
    cache_system: bool = True,
    stream: bool = False,
    cache_ttl: str | None = None,
) -> dict:
    """Send a prompt, return a reply conforming to ``schema`` as a dict.

    ``schema`` is a JSON Schema object — build it with :func:`object_schema`.
    The API constrains generation to it, so the result parses and carries
    exactly the declared keys. A first call with a new schema pays a one-off
    compilation cost; identical schemas are cached for 24 hours after that,
    which is free reuse across a whole run.

    Note the guarantee is *shape*, not *truth*: the model can still put a wrong
    value in a correctly-typed field. That is what provenance and confidence in
    :mod:`schema` are for.
    """
    return json.loads(
        _text(
            _message(
                prompt,
                system=system,
                schema=schema,
                model=model,
                effort=effort,
                max_tokens=max_tokens,
                thinking=thinking,
                cache_system=cache_system,
                stream=stream,
                cache_ttl=cache_ttl,
            )
        )
    )


def extract_batch(
    requests: dict[str, tuple[str, dict]],
    *,
    system: str | None = None,
    model: str = MODEL,
    effort: str | None = DEFAULT_EFFORT,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    thinking: bool = True,
    cache_system: bool = True,
    poll_seconds: float = 10.0,
    timeout_seconds: float = 86400.0,
    prewarm: bool = False,
    cache_ttl: str | None = "1h",
    progress=None,
) -> dict[str, dict]:
    """Run many :func:`extract` calls as one batch. **Half price, asynchronous.**

    ``requests`` maps a caller key to ``(prompt, schema)``; the return maps the
    same keys to parsed replies. Keys that errored, expired or were cancelled
    are simply absent — one bad request must not cost the whole batch, and the
    caller can see what is missing by comparing keys.

    Every token in a batch bills at 50%, with no change to the model, the
    prompt, or the output. The cost is latency: submit, poll, collect. Most
    batches finish well inside an hour; the API's own ceiling is 24h, which is
    what ``timeout_seconds`` defaults to.

    Caller keys are *not* sent as-is. The API constrains ``custom_id`` charset,
    and record ids here include dots (``SRX1.SRS1`` for a pooled experiment), so
    keys are replaced with positional ids and mapped back on the way out.

    ``fallbacks`` is never sent: the Batches API rejects it outright, which is
    another reason the live and batched paths must not diverge silently.

    **Pre-warming — off by default, and measured that way.** A cache entry only
    becomes readable once the request that wrote it has returned, and a batch
    runs its requests in parallel, so most of them write the shared prefix
    instead of reading it: on a 57-request run, cache writes were 4.4x higher
    and reads 55% lower than the same work sent live, making the batch 45%
    *more* expensive before the discount and only 28% cheaper after it.

    ``prewarm=True`` sends the first request of each distinct prefix live and
    keeps its answer, so the rest of the batch can read what it wrote. **On this
    workload that costs more than it saves.** Measured on the same 57 requests:
    the read/write ratio improved from 0.62 to 1.19 and writes fell 26% — the
    mechanism works — but the bill went *up*, $0.1786 to $0.1858. Two reasons:
    the 12 warm calls move off the 50% discount and are the very requests that
    pay the 1.25x cache write, and one warm call does not serialise a parallel
    batch, so the remaining 45 requests still wrote 81,800 tokens between them.

    Keep the option, since it is the textbook remedy for parallel fan-out and
    will otherwise be re-attempted; just do not switch it on without measuring
    it again on a different workload shape. (The documented ``max_tokens=0``
    warm-up is not usable here at all — the API rejects it alongside
    ``output_config.format``, and dropping the schema to get around that would
    change the prefix and warm the wrong entry.)

    ``cache_ttl`` defaults to one hour on this path rather than the standard
    five minutes, because a batch can take longer than five minutes to drain
    and an expired prefix is re-written by every request still in flight.
    """
    if not requests:
        return {}
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    out: dict[str, dict] = {}
    pending = dict(requests)
    if prewarm:
        # One live call per distinct prefix. The prefix is system + schema, so
        # requests sharing a schema share a cache entry; warming one of each is
        # enough for the rest of the batch to read instead of write.
        warmed: set[str] = set()
        for key, (prompt, schema) in list(pending.items()):
            signature = json.dumps(schema, sort_keys=True)
            if signature in warmed:
                continue
            warmed.add(signature)
            try:
                out[key] = extract(prompt, schema, system=system, model=model,
                                   effort=effort, max_tokens=max_tokens,
                                   thinking=thinking, cache_system=cache_system,
                                   cache_ttl=cache_ttl)
            except (RefusalError, json.JSONDecodeError):
                pass          # same tolerance the batch path gives a bad reply
            del pending[key]  # answered live; do not send it twice
        if progress:
            progress(f"pre-warmed {len(warmed)} prefix(es) with live calls")
    if not pending:
        return out

    keys = list(pending)
    payload = [
        Request(
            custom_id=f"r{i}",
            params=MessageCreateParamsNonStreaming(
                **_body(pending[k][0], system, pending[k][1], model, effort,
                        max_tokens, thinking, cache_system, cache_ttl)
            ),
        )
        for i, k in enumerate(keys)
    ]

    batch = _client().messages.batches.create(requests=payload)
    if progress:
        progress(f"batch {batch.id}: {len(payload)} requests submitted")

    deadline = time.monotonic() + timeout_seconds
    while True:
        state = _client().messages.batches.retrieve(batch.id)
        if state.processing_status == "ended":
            break
        if time.monotonic() > deadline:
            raise TimeoutError(f"batch {batch.id} still {state.processing_status}")
        if progress:
            progress(f"batch {batch.id}: {state.processing_status} "
                     f"{state.request_counts.succeeded} done")
        time.sleep(poll_seconds)

    for result in _client().messages.batches.results(batch.id):
        key = keys[int(result.custom_id[1:])]
        if result.result.type != "succeeded":
            continue  # errored / expired / cancelled — absent from the result
        message = result.result.message
        _record(message.usage, batch=True, model=model)
        if message.stop_reason == "refusal":
            continue
        try:
            out[key] = json.loads(_text(message))
        except json.JSONDecodeError:
            continue
    return out


def estimated_cost(model: str = MODEL) -> float:
    """Dollar cost of this process's usage so far, batch discount applied."""
    rate_in, rate_out = PRICES.get(model, PRICES[MODEL])
    u = USAGE
    live = (u["input"] * rate_in + u["cache_write"] * rate_in * 1.25
            + u["cache_read"] * rate_in * 0.10 + u["output"] * rate_out)
    batched = (u["batch_input"] * rate_in + u["batch_cache_write"] * rate_in * 1.25
               + u["batch_cache_read"] * rate_in * 0.10
               + u["batch_output"] * rate_out) * BATCH_DISCOUNT
    return (live + batched) / 1e6


def usage_report() -> str:
    """One-line summary of this process's token spend."""
    u = USAGE
    live = (
        f"${u['cost']:.2f} | {u['calls']} calls | in {u['input']:,} "
        f"(+{u['cache_read']:,} cached, {u['cache_write']:,} written) "
        f"| out {u['output']:,}"
    )
    if not u["batch_calls"]:
        return live
    return (
        f"{live}  ||  batched (50% rate): {u['batch_calls']} calls "
        f"| in {u['batch_input']:,} (+{u['batch_cache_read']:,} cached, "
        f"{u['batch_cache_write']:,} written) | out {u['batch_output']:,}"
    )


def reset_usage() -> None:
    for key in USAGE:
        USAGE[key] = 0