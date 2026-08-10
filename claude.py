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

The credential is read from ``claude_api_key.txt`` (gitignored) on first use.
Nothing happens at import time, so the module imports fine with no key present —
tests can stub :func:`_client` without a credential on disk.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import anthropic

# Claude Opus 5 — the current frontier model, and the one to keep unless there
# is a measured reason to move. Cost per study is dominated by how much context
# each call carries, not by the model tier.
MODEL = "claude-opus-5"

# How hard the model works per call: low | medium | high | xhigh | max.
# `medium` rather than the API's `high` default because reconstruction is one
# small call per sample over tens of thousands of samples, and effort is the
# main cost lever. **Sweep this against a labelled sample before trusting it** —
# if extraction accuracy is materially better at `high`, the extra spend is
# worth it, and this constant is the one place to change.
DEFAULT_EFFORT = "medium"

# Thinking tokens count against max_tokens, so this needs headroom well past the
# size of the JSON being returned. 16k stays under the SDK's non-streaming
# timeout guard; pass stream=True for anything larger.
DEFAULT_MAX_TOKENS = 16000

API_KEY_FILE = "claude_api_key.txt"

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

# Cumulative token spend for this process, so a bulk run can report what it
# cost. Cache reads bill at ~10% of the input rate and cache writes at ~125%,
# so they are tracked apart rather than folded into `input`.
USAGE = {"calls": 0, "input": 0, "output": 0, "cache_write": 0, "cache_read": 0,
         # Batched tokens are counted apart because the Batch API's 50% discount
         # is a billing rate, not a token reduction — the counts look identical,
         # so folding them together would silently overstate the cost of a
         # batched run by 2x.
         "batch_calls": 0, "batch_input": 0, "batch_output": 0,
         "batch_cache_write": 0, "batch_cache_read": 0}

# $ per million tokens, (input, output), for the models this project uses.
PRICES = {"claude-haiku-4-5": (1.0, 5.0), "claude-sonnet-5": (2.0, 10.0),
          "claude-opus-5": (5.0, 25.0), "claude-opus-4-8": (5.0, 25.0)}
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


def set_api_key(key: str | None = None, path: str = API_KEY_FILE) -> None:
    """Set the credential explicitly, or force a re-read from ``path``.

    Called automatically on first use, so most callers never need it. Resets the
    cached client so a new key takes effect immediately.
    """
    global _api_key, _client_instance
    _api_key = key if key is not None else _read_key(path)
    _client_instance = None


def _read_key(path: str) -> str:
    """Read a one-line credential file.

    Stripped because a trailing newline is invisible in an editor but fatal —
    a key with whitespace in it authenticates as a different (invalid) string
    and every request 401s.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found — put the Claude API key there (it is gitignored), "
            f"or call set_api_key(key=...) directly"
        )
    with open(path, encoding="utf-8") as file:
        key = file.read().strip()
    if not key:
        raise ValueError(f"{path} is empty")
    return key


def _client() -> anthropic.Anthropic:
    """The shared client, built on first use.

    ``max_retries=5`` rather than the SDK's 2, matching the retry budget
    :func:`project._request_with_retry` uses against NCBI: a run of thousands of
    calls will hit a 429 or a 5xx, and one transient failure should not cost a
    study. The SDK backs off exponentially and only retries what is retryable.
    """
    global _client_instance
    if _client_instance is None:
        if _api_key is None:
            set_api_key()
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

    _record(response.usage)
    if response.stop_reason == "refusal":
        details = getattr(response, "stop_details", None)
        raise RefusalError(
            getattr(details, "category", None), getattr(details, "explanation", None)
        )
    return response


def _record(usage, batch: bool = False) -> None:
    p = "batch_" if batch else ""
    USAGE[p + "calls"] += 1
    USAGE[p + "input"] += getattr(usage, "input_tokens", 0) or 0
    USAGE[p + "output"] += getattr(usage, "output_tokens", 0) or 0
    USAGE[p + "cache_write"] += getattr(usage, "cache_creation_input_tokens", 0) or 0
    USAGE[p + "cache_read"] += getattr(usage, "cache_read_input_tokens", 0) or 0


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
        _record(message.usage, batch=True)
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
        f"{u['calls']} calls | in {u['input']:,} "
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