use std::time::Duration;

use serde::Deserialize;
use serde_json::{json, Value};

use crate::model::{Effort, Model, ModelError, Request, Response, Thinking, Usage};

pub mod batch;

// The Anthropic Messages API.
//
// Hand-rolled against the HTTP endpoint rather than built on a crate. Three of
// the features this pipeline depends on — structured outputs, prompt caching,
// and the effort control — are not exposed by any published Rust client, and
// the OpenAI-compatible layer documents that it ignores all three. Against raw
// JSON they are three fields.
//
// The split below is what makes this testable without a network or a key:
// `body()` and `parse()` are pure, and `complete()` is the only function that
// performs I/O. Every test in this module exercises the pure halves.

pub const API_BASE: &str = "https://api.anthropic.com/v1";
pub const API_VERSION: &str = "2023-06-01";

// A key that does not start with this is not an Anthropic key, and sending it
// would be a request that cannot succeed.
const KEY_PREFIX: &str = "sk-ant-";

// Server-side fallback: if the named model is saturated, the API may serve a
// sibling rather than 529. Only offered on the models that support it.
const FALLBACK_BETA: &str = "server-side-fallback-2026-07-01";

// The API rejects anything smaller.
pub const MIN_THINKING_BUDGET: u32 = 1024;

const CACHE_READ_RATE: f64 = 0.10;
const CACHE_WRITE_RATE: f64 = 1.25;
pub const BATCH_DISCOUNT: f64 = 0.5;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ModelId {
    Haiku45,
    Sonnet5,
    #[default]
    Opus5,
    Opus48,
}

impl ModelId {
    #[inline]
    pub fn as_str(self) -> &'static str {
        match self {
            ModelId::Haiku45 => "claude-haiku-4-5",
            ModelId::Sonnet5 => "claude-sonnet-5",
            ModelId::Opus5 => "claude-opus-5",
            ModelId::Opus48 => "claude-opus-4-8",
        }
    }

    // Models that think unless told otherwise. On these, omitting the parameter
    // is not neutral — it leaves adaptive thinking on — so `Thinking::Disabled`
    // has to be sent explicitly. The others predate the field and reject it.
    #[inline]
    pub fn thinks_by_default(self) -> bool {
        matches!(self, ModelId::Opus5 | ModelId::Sonnet5)
    }

    // Whether the model accepts `thinking: {"type": "adaptive"}` at all.
    //
    // A separate question from `thinks_by_default`, and the two do not coincide:
    // Opus 4.8 accepts adaptive but does not think unless asked. Haiku 4.5
    // predates the parameter and answers a 400 — "adaptive thinking is not
    // supported on this model" — which fails the whole request.
    #[inline]
    pub fn supports_adaptive_thinking(self) -> bool {
        !matches!(self, ModelId::Haiku45)
    }

    // Whether the model accepts `output_config.effort`. Haiku 4.5 predates it
    // and answers a 400, the same way it does for adaptive thinking — two
    // separate parameters, one model, two lost requests.
    // Whether the model takes `thinking: {type: "enabled", budget_tokens: N}`.
    // The complement of `supports_adaptive_thinking`: 4.6-and-later models
    // answer a 400 for `budget_tokens`, and older ones answer a 400 for
    // `adaptive`. Exactly one mode works per model.
    #[inline]
    pub fn supports_thinking_budget(self) -> bool {
        matches!(self, ModelId::Haiku45)
    }

    #[inline]
    pub fn supports_effort(self) -> bool {
        !matches!(self, ModelId::Haiku45)
    }

    #[inline]
    pub fn supports_fallback(self) -> bool {
        matches!(self, ModelId::Opus5)
    }

    // Dollars per million tokens, (input, output).
    #[inline]
    pub fn prices(self) -> (f64, f64) {
        match self {
            ModelId::Haiku45 => (1.0, 5.0),
            ModelId::Sonnet5 => (2.0, 10.0),
            ModelId::Opus5 | ModelId::Opus48 => (5.0, 25.0),
        }
    }
}

// Dollars for one response's usage at this model's rates.
//
// Cache reads bill at ~10% of the input rate and writes at ~125%, which is why
// they cannot be folded into the input count.
#[inline]
pub fn call_cost(usage: Usage, model: ModelId, batch: bool) -> f64 {
    let (rate_in, rate_out) = model.prices();
    let dollars = (usage.input as f64 * rate_in
        + usage.cache_read as f64 * rate_in * CACHE_READ_RATE
        + usage.cache_write as f64 * rate_in * CACHE_WRITE_RATE
        + usage.output as f64 * rate_out)
        / 1e6;
    dollars * if batch { BATCH_DISCOUNT } else { 1.0 }
}

pub struct Claude {
    api_key: String,
    model: ModelId,
    http: reqwest::blocking::Client,
    // Overridable so a caller can point at a proxy. Not a test seam — the tests
    // in this module never reach `complete()` at all.
    base_url: String,
}

impl Claude {
    // There is no default key file and no environment fallback, deliberately.
    // A keyless run must fail loudly at construction rather than quietly pick
    // up whichever key happens to be lying around: that is exactly how a run
    // once billed a key its author believed was uninvolved.
    pub fn new(api_key: impl Into<String>, model: ModelId) -> Result<Self, ModelError> {
        let api_key = api_key.into();
        validate_key(&api_key)?;
        let http = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(600))
            .build()
            .map_err(|e| ModelError::Transport(e.to_string()))?;
        Ok(Self { api_key, model, http, base_url: API_BASE.to_string() })
    }

    // Reads the key from a file, trimming the trailing newline a file always
    // has. Separate from `new` so the no-default-key rule stays visible: the
    // path is still named by the caller, and there is still no fallback.
    pub fn from_key_file(path: impl AsRef<std::path::Path>, model: ModelId) -> Result<Self, ModelError> {
        let path = path.as_ref();
        let key = std::fs::read_to_string(path).map_err(|e| {
            ModelError::InvalidApiKey(format!("could not read {}: {e}", path.display()))
        })?;
        Self::new(key.trim(), model)
    }

    #[inline]
    pub fn model(&self) -> ModelId {
        self.model
    }

    #[inline]
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    // The request body. Pure, so the parameter interactions that cost money are
    // testable without spending any.
    #[inline]
    pub fn body(&self, request: &Request) -> Value {
        body_for(self.model, request)
    }

    // Crate-visible so the batch module can reach the same client and
    // credentials without a second copy of either.
    #[inline]
    pub(crate) fn api_key(&self) -> &str {
        &self.api_key
    }

    #[inline]
    pub(crate) fn http(&self) -> &reqwest::blocking::Client {
        &self.http
    }

    #[inline]
    pub(crate) fn url(&self, path: &str) -> String {
        format!("{}{path}", self.base_url)
    }
}

// The beta this model needs enabled, as the header value the Messages API
// reads it from.
//
// Separate from `body_for` because the two endpoints disagree about where a
// beta goes: the Messages API takes a header, while a batch carries per-request
// params and no headers of its own. Keeping it a pure function means the choice
// is testable without a network, like everything else here.
#[inline]
pub fn beta_header(model: ModelId) -> Option<&'static str> {
    model.supports_fallback().then_some(FALLBACK_BETA)
}

// Free function rather than a method so the tests can build a body for every
// model without constructing a client (which would need a key).
#[inline]
pub fn body_for(model: ModelId, request: &Request) -> Value {
    let mut body = json!({
        "model": model.as_str(),
        "max_tokens": request.max_tokens,
        "messages": [{ "role": "user", "content": request.prompt }],
    });
    let map = body.as_object_mut().expect("object literal");

    if let Some(system) = &request.system {
        let mut block = json!({ "type": "text", "text": system });
        if request.cache_system {
            let mut cache = json!({ "type": "ephemeral" });
            if let Some(ttl) = request.cache_ttl {
                cache["ttl"] = json!(ttl.as_str());
            }
            block["cache_control"] = cache;
        }
        // An array of blocks, not a bare string: only a block can carry
        // cache_control.
        map.insert("system".into(), json!([block]));
    }

    if let Some(thinking) = thinking_for(request.thinking, model) {
        map.insert("thinking".into(), thinking);
    }

    // Omitted entirely when empty. Sending `output_config: {}` is not the same
    // request as sending nothing.
    let mut output_config = serde_json::Map::new();
    // Omitted on a model that rejects it, for the same reason adaptive thinking
    // is: a `Request` is provider-neutral, so asking for something a model
    // cannot do must degrade rather than fail the whole call.
    if let Some(effort) = request.effort.filter(|_| model.supports_effort()) {
        output_config.insert("effort".into(), json!(effort.as_str()));
    }
    if let Some(schema) = &request.schema {
        output_config.insert(
            "format".into(),
            json!({ "type": "json_schema", "schema": schema }),
        );
    }
    if !output_config.is_empty() {
        map.insert("output_config".into(), Value::Object(output_config));
    }

    if model.supports_fallback() {
        // `fallbacks` only. The beta itself is enabled by a header on the
        // Messages API, which rejects a `betas` *field* outright — verified
        // live: with the header set and `betas` in the body the call still
        // 400s with "betas: Extra inputs are not permitted". See `beta_header`.
        map.insert("fallbacks".into(), json!("default"));
    }

    body
}

// The three-way outcome that a bool cannot express.
#[inline]
fn thinking_for(thinking: Thinking, model: ModelId) -> Option<Value> {
    match thinking {
        // "Think if you can". A `Request` is provider-neutral by design — the
        // layer that builds one does not know which model will answer it — so
        // asking a model that cannot has to mean something. Omitted rather than
        // sent: sending it is a 400 that fails the request outright, which is
        // worse than an answer produced without thinking.
        //
        // Python does not guard this and documents it as the caller's problem;
        // that is exactly how a live run died on `claude-haiku-4-5`.
        Thinking::Adaptive if model.supports_adaptive_thinking() => {
            Some(json!({ "type": "adaptive" }))
        }
        // Asked for adaptive on a model that only does budgets. Both mean
        // "think", so this is expressible — but the caller named no budget, so
        // there is nothing to send and thinking stays off.
        Thinking::Adaptive => None,

        Thinking::Enabled { budget_tokens } if model.supports_thinking_budget() => {
            Some(json!({ "type": "enabled", "budget_tokens": budget_tokens }))
        }
        // A budget on a model that rejects budgets. Degraded to adaptive rather
        // than dropped: the caller asked to think, that model can think, and the
        // budget is simply not expressible there.
        Thinking::Enabled { .. } if model.supports_adaptive_thinking() => {
            Some(json!({ "type": "adaptive" }))
        }
        Thinking::Enabled { .. } => None,

        // Say it out loud, but only to a model that can hear it.
        Thinking::Disabled if model.thinks_by_default() => Some(json!({ "type": "disabled" })),
        Thinking::Disabled => None,
        Thinking::Unset => None,
    }
}

#[inline]
fn validate_key(key: &str) -> Result<(), ModelError> {
    if key.trim().is_empty() {
        return Err(ModelError::MissingApiKey);
    }
    if key != key.trim() {
        return Err(ModelError::InvalidApiKey(
            "surrounding whitespace — a key read from a file needs trimming".into(),
        ));
    }
    if !key.starts_with(KEY_PREFIX) {
        return Err(ModelError::InvalidApiKey(format!(
            "does not start with {KEY_PREFIX:?}"
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Response
// ---------------------------------------------------------------------------
// The one place a wire type earns its keep. `content` is a heterogeneous tagged
// union and `stop_reason` drives control flow; neither shape should reach a
// caller of `Model::complete`.

#[derive(Debug, Deserialize)]
struct WireResponse {
    #[serde(default)]
    content: Vec<ContentBlock>,
    #[serde(default)]
    stop_reason: Option<String>,
    #[serde(default)]
    stop_details: Option<StopDetails>,
    #[serde(default)]
    usage: WireUsage,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ContentBlock {
    Text { text: String },
    // Thinking and tool blocks are dropped, and an unrecognised one is dropped
    // rather than fatal — the API adds block types, and a new one must not turn
    // a good response into an error.
    #[serde(other)]
    Other,
}

#[derive(Debug, Default, Deserialize)]
struct StopDetails {
    #[serde(default)]
    category: Option<String>,
    #[serde(default)]
    explanation: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct WireUsage {
    #[serde(default)]
    input_tokens: u64,
    #[serde(default)]
    output_tokens: u64,
    #[serde(default)]
    cache_creation_input_tokens: u64,
    #[serde(default)]
    cache_read_input_tokens: u64,
    #[serde(default)]
    output_tokens_details: OutputDetails,
}

#[derive(Debug, Default, Deserialize)]
struct OutputDetails {
    #[serde(default)]
    thinking_tokens: u64,
}

// Turns a 200 body into a Response, or into the error it actually represents.
//
// `wants_json` rather than reading the schema off the request: a refusal must
// be reported as a refusal, not as malformed JSON, and that ordering is easier
// to see when the flag is explicit.
#[inline]
pub fn parse(raw: &str, wants_json: bool) -> Result<Response, ModelError> {
    let wire: WireResponse =
        serde_json::from_str(raw).map_err(|e| ModelError::Decode(e.to_string()))?;

    // Built before any error path. A refusal, a malformed reply and a clipped
    // one are all HTTP 200s whose tokens were generated and charged, so an
    // error that drops this figure hides a billed call from the ledger.
    let usage = Usage {
        input: wire.usage.input_tokens,
        output: wire.usage.output_tokens,
        cache_write: wire.usage.cache_creation_input_tokens,
        cache_read: wire.usage.cache_read_input_tokens,
        thinking: wire.usage.output_tokens_details.thinking_tokens,
    };

    // Checked before reading content: a refusal arrives as a successful 200
    // whose content is empty, so reading it first yields an empty string and
    // hides the cause.
    if wire.stop_reason.as_deref() == Some("refusal") {
        let details = wire.stop_details.unwrap_or_default();
        return Err(ModelError::Refused {
            category: details.category,
            explanation: details.explanation,
            usage,
        });
    }

    let text: String = wire
        .content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text { text } => Some(text.as_str()),
            ContentBlock::Other => None,
        })
        .collect();

    let json = if wants_json {
        // The stop reason travels with the error, because without it the two
        // causes are indistinguishable and one of them is actionable. A reply
        // clipped at `max_tokens` fails here as "EOF while parsing a string" —
        // the same message a genuinely malformed reply gives — and the fix for
        // one (raise the ceiling) is nothing like the fix for the other. Seen
        // live once: a 16,754-character answer to a 2,062-character ask, cut
        // mid-string, reported only as bad JSON.
        Some(serde_json::from_str(&text).map_err(|e| {
            let clipped = wire.stop_reason.as_deref() == Some("max_tokens");
            // Three outcomes, not two. The ceiling can be reached with a long
            // answer cut mid-string, or with *no answer at all* — the whole
            // budget spent reasoning. Thinking is billed as output and counted
            // against `max_tokens`, so an adaptive budget on a long prompt can
            // consume the lot before a single answer token is written. It is
            // named separately because the fix is specific: raise max_tokens,
            // or turn thinking off on this layer. Seen live on the paper layer
            // at adaptive thinking and a 16,000-token ceiling.
            let detail = if clipped && text.is_empty() {
                format!(
                    "no answer tokens at all — {} of {} output tokens were reasoning, and \
                     thinking counts toward max_tokens. Raise max_tokens or disable \
                     thinking on this layer.",
                    usage.thinking, usage.output
                )
            } else {
                format!(
                    "{e} — stop_reason {}, {} characters of reply",
                    wire.stop_reason.as_deref().unwrap_or("absent"),
                    text.len()
                )
            };
            // Split on the stop reason rather than on the parse error, which is
            // identical either way.
            if clipped {
                ModelError::Truncated { detail, usage }
            } else {
                ModelError::MalformedJson { detail, usage }
            }
        })?)
    } else {
        None
    };

    Ok(Response { text, json, stop_reason: wire.stop_reason, usage })
}

// A non-2xx status, read as the error it is. Also pure.
#[inline]
pub fn error_for(status: u16, retry_after: Option<u64>, raw: &str) -> ModelError {
    #[derive(Deserialize)]
    struct Envelope {
        error: Option<Inner>,
    }
    #[derive(Deserialize)]
    struct Inner {
        #[serde(rename = "type")]
        kind: Option<String>,
        message: Option<String>,
    }

    let parsed: Option<Envelope> = serde_json::from_str(raw).ok();
    let inner = parsed.and_then(|e| e.error);
    let kind = inner.as_ref().and_then(|i| i.kind.clone());
    let message = inner
        .and_then(|i| i.message)
        .unwrap_or_else(|| raw.chars().take(400).collect());

    match status {
        429 => ModelError::RateLimited { retry_after },
        529 => ModelError::Overloaded,
        _ => ModelError::Api { status, kind, message },
    }
}

// Combinations the API rejects, caught before a request is sent.
//
// Distinct from an unsupported parameter, which is dropped: there the caller
// asked for something the model cannot do, and the request still means what it
// meant. Here two things were asked for that cannot both hold, and quietly
// dropping either one changes the bill — turning thinking back on at max effort
// is the expensive direction — so the run is told instead.
pub fn unsupported_combination(model: ModelId, request: &Request) -> Option<ModelError> {
    // `budget_tokens` has two hard rules, both 400s. Checked here so a run
    // fails before spending rather than on the first request of a paid layer.
    if let Thinking::Enabled { budget_tokens } = request.thinking
        && model.supports_thinking_budget()
    {
        {
            if budget_tokens < MIN_THINKING_BUDGET {
                return Some(ModelError::Api {
                    status: 0,
                    kind: Some("thinking_budget_too_small".into()),
                    message: format!(
                        "budget_tokens must be at least {MIN_THINKING_BUDGET}; got {budget_tokens}"
                    ),
                });
            }
            // Thinking counts toward max_tokens, so a budget at or above it
            // leaves no room for the answer.
            if budget_tokens >= request.max_tokens {
                return Some(ModelError::Api {
                    status: 0,
                    kind: Some("thinking_budget_too_large".into()),
                    message: format!(
                        "budget_tokens ({budget_tokens}) must be below max_tokens ({}) — \
                         thinking counts toward the same ceiling, so the answer needs room",
                        request.max_tokens
                    ),
                });
            }
        }
    }
    let deep = matches!(request.effort, Some(Effort::XHigh) | Some(Effort::Max));
    if model == ModelId::Opus5 && deep && request.thinking == Thinking::Disabled {
        return Some(ModelError::Api {
            status: 0,
            kind: Some("unsupported_combination".into()),
            message: format!(
                "{} rejects thinking:disabled at {:?} effort. Raise thinking or \
                 lower the effort — dropping either here would change what the \
                 call costs.",
                model.as_str(),
                request.effort.unwrap()
            ),
        });
    }
    None
}

impl Model for Claude {
    #[inline]
    fn price(&self, usage: Usage) -> f64 {
        call_cost(usage, self.model, false)
    }

    #[inline]
    fn supports_batch(&self) -> bool {
        true
    }

    #[inline]
    fn price_many(&self, usage: Usage) -> f64 {
        call_cost(usage, self.model, true)
    }

    // Submit, poll, collect — at half price, minutes to hours later.
    //
    // The budget passed here is unlimited on purpose: whatever wraps this is
    // what meters, and a second ceiling inside would refuse against a ledger it
    // cannot see. A submission that fails at all fails every key, because none
    // of them were sent.
    fn complete_many(
        &self,
        requests: &std::collections::BTreeMap<String, Request>,
    ) -> std::collections::BTreeMap<String, Result<Response, ModelError>> {
        use crate::model::budget::Budget;
        let unmetered = Budget::unlimited();
        match self.run_batch(requests, batch::BatchOptions::default(), &unmetered, |_| {}) {
            Ok(outcome) => requests
                .keys()
                .map(|key| {
                    let result = match outcome.results.get(key) {
                        Some(response) => Ok(response.clone()),
                        None => Err(match outcome.failures.get(key) {
                            Some(failure) => ModelError::Api {
                                status: 0,
                                kind: Some("batch_request_failed".into()),
                                message: format!("{failure:?}"),
                            },
                            None => ModelError::Api {
                                status: 0,
                                kind: Some("batch_result_missing".into()),
                                message: "the batch returned no result for this key".into(),
                            },
                        }),
                    };
                    (key.clone(), result)
                })
                .collect(),
            Err(error) => requests
                .keys()
                .map(|key| {
                    (
                        key.clone(),
                        Err(ModelError::Api {
                            status: 0,
                            kind: Some("batch_submission_failed".into()),
                            message: error.to_string(),
                        }),
                    )
                })
                .collect(),
        }
    }

    // The only function here that touches the network.
    #[inline]
    fn complete(&self, request: &Request) -> Result<Response, ModelError> {
        // Locally, before the socket: a run should refuse at the top rather
        // than on the first request of a paid layer.
        if let Some(error) = unsupported_combination(self.model, request) {
            return Err(error);
        }
        let body = self.body(request);
        let mut post = self
            .http
            .post(self.url("/messages"))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", API_VERSION)
            .header("content-type", "application/json");
        // Where the beta goes on this endpoint. The batch path needs none: it
        // strips `fallbacks` from every request before submitting.
        if let Some(beta) = beta_header(self.model) {
            post = post.header("anthropic-beta", beta);
        }
        let response = post
            .json(&body)
            .send()
            .map_err(|e| ModelError::Transport(e.to_string()))?;

        let status = response.status().as_u16();
        let retry_after = response
            .headers()
            .get("retry-after")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse().ok());
        let raw = response
            .text()
            .map_err(|e| ModelError::Transport(e.to_string()))?;

        if !(200..300).contains(&status) {
            return Err(error_for(status, retry_after, &raw));
        }
        parse(&raw, request.schema.is_some())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{CacheTtl, Effort};

    // Every test here is offline and keyless by construction. `complete()` is
    // the only function that opens a socket and nothing below calls it, so this
    // module cannot spend money however it is run.

    fn req() -> Request {
        Request::new("what is the collection date?")
    }

    // -- the thinking tri-state --------------------------------------------

    #[test]
    fn adaptive_thinking_is_sent_only_to_models_that_accept_it() {
        // This test used to assert the opposite — that adaptive goes to every
        // model — and a live run died on it: `claude-haiku-4-5` answers
        // "adaptive thinking is not supported on this model" with a 400, and
        // the whole request is lost. A test can pin a bug as easily as a rule.
        for model in [ModelId::Sonnet5, ModelId::Opus5, ModelId::Opus48] {
            let body = body_for(model, &req().thinking(Thinking::Adaptive));
            assert_eq!(body["thinking"]["type"], "adaptive", "{model:?}");
        }
        let haiku = body_for(ModelId::Haiku45, &req().thinking(Thinking::Adaptive));
        assert!(haiku.get("thinking").is_none(), "Haiku 4.5 rejects the parameter");
    }

    #[test]
    fn accepting_adaptive_is_not_the_same_as_thinking_by_default() {
        // Opus 4.8 accepts the parameter but does not think unless asked, so one
        // predicate cannot answer both questions.
        assert!(ModelId::Opus48.supports_adaptive_thinking());
        assert!(!ModelId::Opus48.thinks_by_default());
        assert!(!ModelId::Haiku45.supports_adaptive_thinking());
        assert!(!ModelId::Haiku45.thinks_by_default());
        assert!(ModelId::Opus5.supports_adaptive_thinking());
        assert!(ModelId::Opus5.thinks_by_default());
    }

    #[test]
    fn a_provider_neutral_request_works_against_every_model() {
        // The property that matters: a layer builds one Request without knowing
        // which model answers it, so the default must not 400 on any of them.
        for model in [ModelId::Haiku45, ModelId::Sonnet5, ModelId::Opus5, ModelId::Opus48] {
            let body = body_for(model, &Request::new("p").system("s"));
            match body.get("thinking") {
                None => assert!(!model.supports_adaptive_thinking(), "{model:?}"),
                Some(t) => assert_eq!(t["type"], "adaptive", "{model:?}"),
            }
        }
    }

    #[test]
    fn disabling_thinking_is_said_out_loud_only_where_it_must_be() {
        // The distinction that cost 2.5x an estimate. On a model that thinks by
        // default, omitting the field leaves thinking ON, so "off" has to be
        // stated. On the others the field does not exist and sending it 400s.
        for model in [ModelId::Opus5, ModelId::Sonnet5] {
            let body = body_for(model, &req().thinking(Thinking::Disabled));
            assert_eq!(body["thinking"]["type"], "disabled", "{model:?}");
        }
        for model in [ModelId::Haiku45, ModelId::Opus48] {
            let body = body_for(model, &req().thinking(Thinking::Disabled));
            assert!(body.get("thinking").is_none(), "{model:?} must not be sent the field");
        }
    }

    #[test]
    fn a_thinking_budget_is_the_only_way_a_pre_4_6_model_thinks() {
        // Haiku 4.5 rejects `adaptive`, so before this variant existed it could
        // not think at all through this client — Adaptive was simply dropped.
        let body = body_for(
            ModelId::Haiku45,
            &req().thinking(Thinking::Enabled { budget_tokens: 4000 }),
        );
        assert_eq!(body["thinking"]["type"], "enabled");
        assert_eq!(body["thinking"]["budget_tokens"], 4000);
    }

    #[test]
    fn a_budget_on_a_model_that_rejects_budgets_becomes_adaptive() {
        // Both mean "think". The budget is not expressible on 4.6-and-later, so
        // the request degrades rather than 400ing — but it does still think.
        for model in [ModelId::Sonnet5, ModelId::Opus5, ModelId::Opus48] {
            let body = body_for(model, &req().thinking(Thinking::Enabled { budget_tokens: 4000 }));
            assert_eq!(body["thinking"]["type"], "adaptive", "{model:?}");
            assert!(body["thinking"].get("budget_tokens").is_none(),
                    "{model:?} 400s on budget_tokens");
        }
    }

    #[test]
    fn exactly_one_thinking_mode_works_per_model() {
        // The two are complements, not alternatives: a model takes adaptive or
        // a budget, never both and never neither.
        for model in [ModelId::Haiku45, ModelId::Sonnet5, ModelId::Opus5, ModelId::Opus48] {
            assert_ne!(
                model.supports_adaptive_thinking(),
                model.supports_thinking_budget(),
                "{model:?} claims both modes or neither"
            );
        }
    }

    #[test]
    fn unset_thinking_sends_nothing_at_all() {
        // Not the same as Disabled: this is what the parameter did before the
        // three-state enum, and on Opus 5 or Sonnet 5 it means thinking runs.
        for model in [ModelId::Haiku45, ModelId::Sonnet5, ModelId::Opus5, ModelId::Opus48] {
            let body = body_for(model, &req().thinking(Thinking::Unset));
            assert!(body.get("thinking").is_none(), "{model:?}");
        }
    }

    #[test]
    fn disabled_and_unset_are_different_requests() {
        let disabled = body_for(ModelId::Opus5, &req().thinking(Thinking::Disabled));
        let unset = body_for(ModelId::Opus5, &req().thinking(Thinking::Unset));
        assert_ne!(disabled, unset);
    }

    // -- system block and caching ------------------------------------------

    #[test]
    fn the_system_prompt_is_a_block_so_it_can_carry_cache_control() {
        // A bare string cannot be marked cacheable, which is the whole reason
        // this is an array.
        let body = body_for(ModelId::Opus5, &req().system("FIELD DEFINITIONS ..."));
        assert!(body["system"].is_array());
        assert_eq!(body["system"][0]["type"], "text");
        assert_eq!(body["system"][0]["text"], "FIELD DEFINITIONS ...");
        assert_eq!(body["system"][0]["cache_control"]["type"], "ephemeral");
        assert!(body["system"][0]["cache_control"].get("ttl").is_none());
    }

    #[test]
    fn a_one_hour_ttl_is_sent_only_when_asked_for() {
        let body = body_for(
            ModelId::Opus5,
            &req().system("prefix").cache_ttl(CacheTtl::OneHour),
        );
        assert_eq!(body["system"][0]["cache_control"]["ttl"], "1h");

        let five = body_for(
            ModelId::Opus5,
            &req().system("prefix").cache_ttl(CacheTtl::FiveMinutes),
        );
        assert_eq!(five["system"][0]["cache_control"]["ttl"], "5m");
    }

    #[test]
    fn caching_can_be_switched_off_without_losing_the_prompt() {
        let mut request = req().system("prefix");
        request.cache_system = false;
        let body = body_for(ModelId::Opus5, &request);
        assert_eq!(body["system"][0]["text"], "prefix");
        assert!(body["system"][0].get("cache_control").is_none());
    }

    #[test]
    fn no_system_prompt_means_no_system_key() {
        // Sending `system: null` is not the same request as sending nothing.
        let body = body_for(ModelId::Opus5, &req());
        assert!(body.get("system").is_none());
    }

    #[test]
    fn the_cached_prefix_is_byte_identical_across_calls() {
        // The cache is a prefix match, so any per-call text in the system block
        // loses every hit. Two requests differing only in prompt must produce
        // the same system block.
        let a = body_for(ModelId::Opus5, &Request::new("sample A").system("PREFIX"));
        let b = body_for(ModelId::Opus5, &Request::new("sample B").system("PREFIX"));
        assert_eq!(a["system"], b["system"]);
        assert_ne!(a["messages"], b["messages"]);
    }

    // -- output_config ------------------------------------------------------

    #[test]
    fn effort_and_schema_share_one_output_config() {
        let schema = json!({"type": "object", "properties": {"x": {"type": "string"}}});
        let body = body_for(
            ModelId::Opus5,
            &req().effort(Effort::High).schema(schema.clone()),
        );
        assert_eq!(body["output_config"]["effort"], "high");
        assert_eq!(body["output_config"]["format"]["type"], "json_schema");
        assert_eq!(body["output_config"]["format"]["schema"], schema);
    }

    #[test]
    fn an_empty_output_config_is_omitted_rather_than_sent_empty() {
        let mut request = req();
        request.effort = None;
        let body = body_for(ModelId::Opus5, &request);
        assert!(body.get("output_config").is_none());
    }

    #[test]
    fn effort_is_sent_only_to_models_that_accept_it() {
        // The second 400 on the same model. Haiku 4.5 predates `effort` as well
        // as adaptive thinking, and the default Request carries both.
        for model in [ModelId::Sonnet5, ModelId::Opus5, ModelId::Opus48] {
            let body = body_for(model, &req().effort(Effort::High));
            assert_eq!(body["output_config"]["effort"], "high", "{model:?}");
        }
        let haiku = body_for(ModelId::Haiku45, &req().effort(Effort::High));
        assert!(haiku.get("output_config").is_none(),
                "with effort dropped and no schema there is nothing left to send");
    }

    #[test]
    fn dropping_effort_still_leaves_the_schema() {
        // output_config carries two unrelated things; losing one must not lose
        // the other, or a structured request silently becomes freeform.
        let schema = json!({"type": "object"});
        let body = body_for(ModelId::Haiku45, &req().effort(Effort::High).schema(schema));
        assert!(body["output_config"].get("effort").is_none());
        assert_eq!(body["output_config"]["format"]["type"], "json_schema");
    }

    #[test]
    fn the_default_request_is_safe_on_every_model() {
        // What the live run failed on: a layer builds one provider-neutral
        // Request and it has to be sendable to whichever model answers it.
        for model in [ModelId::Haiku45, ModelId::Sonnet5, ModelId::Opus5, ModelId::Opus48] {
            let request = Request::new("p").system("s");
            assert!(unsupported_combination(model, &request).is_none(), "{model:?}");
            let body = body_for(model, &request);
            if !model.supports_adaptive_thinking() {
                assert!(body.get("thinking").is_none(), "{model:?}");
            }
            if !model.supports_effort() {
                assert!(body["output_config"].get("effort").is_none(), "{model:?}");
            }
        }
    }

    #[test]
    fn a_thinking_budget_below_the_minimum_is_refused_locally() {
        let request = req().thinking(Thinking::Enabled { budget_tokens: 512 });
        match unsupported_combination(ModelId::Haiku45, &request) {
            Some(ModelError::Api { kind, .. }) => {
                assert_eq!(kind.as_deref(), Some("thinking_budget_too_small"))
            }
            other => panic!("expected a refusal, got {other:?}"),
        }
        assert!(unsupported_combination(
            ModelId::Haiku45,
            &req().thinking(Thinking::Enabled { budget_tokens: MIN_THINKING_BUDGET })
        ).is_none());
    }

    #[test]
    fn a_thinking_budget_that_crowds_out_the_answer_is_refused() {
        // Thinking counts toward max_tokens, so a budget at or above it leaves
        // nothing for the response. The API 400s; this catches it first.
        let mut request = req().thinking(Thinking::Enabled { budget_tokens: 16_000 });
        request.max_tokens = 16_000;
        match unsupported_combination(ModelId::Haiku45, &request) {
            Some(ModelError::Api { kind, message, .. }) => {
                assert_eq!(kind.as_deref(), Some("thinking_budget_too_large"));
                assert!(message.contains("same ceiling"));
            }
            other => panic!("expected a refusal, got {other:?}"),
        }
        request.max_tokens = 16_001;
        assert!(unsupported_combination(ModelId::Haiku45, &request).is_none());
    }

    #[test]
    fn budget_rules_are_not_applied_to_models_that_never_see_the_budget() {
        // On Sonnet 5 the budget degrades to adaptive and is never sent, so
        // refusing on its value would block a request that would have worked.
        let mut request = req().thinking(Thinking::Enabled { budget_tokens: 1 });
        request.max_tokens = 100;
        assert!(unsupported_combination(ModelId::Sonnet5, &request).is_none());
    }

    #[test]
    fn thinking_tokens_are_read_as_a_subset_of_output_not_an_addition() {
        // Double-counting them would overstate every thinking run's bill.
        let raw = r#"{
            "content": [{"type": "text", "text": "ok"}],
            "usage": {"input_tokens": 100, "output_tokens": 900,
                      "output_tokens_details": {"thinking_tokens": 800}}
        }"#;
        let usage = parse(raw, false).unwrap().usage;
        assert_eq!(usage.output, 900);
        assert_eq!(usage.thinking, 800);
        // priced on `output` alone, so the thinking is in there exactly once
        let cost = call_cost(usage, ModelId::Sonnet5, false);
        let expected = (100.0 * 2.0 + 900.0 * 10.0) / 1e6;
        assert!((cost - expected).abs() < 1e-12);
    }

    #[test]
    fn a_response_with_no_thinking_detail_reads_as_zero() {
        let raw = r#"{"content": [{"type":"text","text":"ok"}],
                      "usage": {"input_tokens": 1, "output_tokens": 2}}"#;
        assert_eq!(parse(raw, false).unwrap().usage.thinking, 0);
    }

    #[test]
    fn a_contradictory_combination_is_refused_before_it_is_sent() {
        // Opus 5 rejects thinking:disabled at xhigh/max. Dropping either side
        // silently would change the bill — turning thinking back on at max
        // effort is the expensive direction — so the caller is told.
        for effort in [Effort::XHigh, Effort::Max] {
            let request = req().effort(effort).thinking(Thinking::Disabled);
            match unsupported_combination(ModelId::Opus5, &request) {
                Some(ModelError::Api { kind, message, .. }) => {
                    assert_eq!(kind.as_deref(), Some("unsupported_combination"));
                    assert!(message.contains("thinking:disabled"));
                }
                other => panic!("expected a refusal for {effort:?}, got {other:?}"),
            }
        }
        // and the same combination is fine at lower effort, and on other models
        assert!(unsupported_combination(
            ModelId::Opus5,
            &req().effort(Effort::High).thinking(Thinking::Disabled)
        ).is_none());
        assert!(unsupported_combination(
            ModelId::Sonnet5,
            &req().effort(Effort::Max).thinking(Thinking::Disabled)
        ).is_none());
    }

    #[test]
    fn every_effort_level_has_a_wire_name() {
        for (effort, expected) in [
            (Effort::Low, "low"),
            (Effort::Medium, "medium"),
            (Effort::High, "high"),
            (Effort::XHigh, "xhigh"),
            (Effort::Max, "max"),
        ] {
            let body = body_for(ModelId::Opus5, &req().effort(effort));
            assert_eq!(body["output_config"]["effort"], expected);
        }
    }

    #[test]
    fn the_default_request_matches_the_python_defaults() {
        // Changing any of these changes what every layer pays, so they are
        // pinned rather than left to drift: medium effort, 16k output, adaptive
        // thinking, caching on.
        let body = body_for(ModelId::Opus5, &Request::new("hello").system("prefix"));
        assert_eq!(body["max_tokens"], 16_000);
        assert_eq!(body["output_config"]["effort"], "medium");
        assert_eq!(body["thinking"]["type"], "adaptive");
        assert_eq!(body["system"][0]["cache_control"]["type"], "ephemeral");
    }

    // -- fallback beta ------------------------------------------------------

    #[test]
    fn the_fallback_beta_is_offered_only_where_it_is_supported() {
        let opus = body_for(ModelId::Opus5, &req());
        assert_eq!(opus["fallbacks"], "default");
        assert_eq!(beta_header(ModelId::Opus5), Some(FALLBACK_BETA));

        for model in [ModelId::Haiku45, ModelId::Sonnet5, ModelId::Opus48] {
            let body = body_for(model, &req());
            assert!(body.get("fallbacks").is_none(), "{model:?}");
            assert_eq!(beta_header(model), None, "{model:?}");
        }
    }

    #[test]
    fn the_beta_never_travels_in_the_body() {
        // Verified live: the Messages API rejects a `betas` field even with the
        // header set — "betas: Extra inputs are not permitted". Sending it made
        // every Opus call a 400, and no other model reaches this branch, so the
        // whole class was invisible until the first Opus run.
        for model in [ModelId::Haiku45, ModelId::Sonnet5, ModelId::Opus5, ModelId::Opus48] {
            assert!(body_for(model, &req()).get("betas").is_none(), "{model:?}");
        }
    }

    #[test]
    fn the_model_name_reaches_the_wire() {
        for model in [ModelId::Haiku45, ModelId::Sonnet5, ModelId::Opus5, ModelId::Opus48] {
            assert_eq!(body_for(model, &req())["model"], model.as_str());
        }
    }

    #[test]
    fn the_prompt_is_the_only_user_message() {
        let body = body_for(ModelId::Opus5, &req());
        assert_eq!(body["messages"].as_array().unwrap().len(), 1);
        assert_eq!(body["messages"][0]["role"], "user");
        assert_eq!(body["messages"][0]["content"], "what is the collection date?");
    }

    // -- response parsing ---------------------------------------------------

    #[test]
    fn text_blocks_are_concatenated_and_others_dropped() {
        let raw = r#"{
            "content": [
                {"type": "thinking", "thinking": "hmm"},
                {"type": "text", "text": "the answer"},
                {"type": "text", "text": " continues"}
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 3}
        }"#;
        let response = parse(raw, false).unwrap();
        assert_eq!(response.text, "the answer continues");
        assert!(response.json.is_none());
    }

    #[test]
    fn an_unknown_block_type_is_skipped_rather_than_fatal() {
        // The API adds block types. A new one must not turn a good response
        // into an error.
        let raw = r#"{
            "content": [
                {"type": "some_block_invented_next_year", "payload": {"a": 1}},
                {"type": "text", "text": "still fine"}
            ],
            "usage": {"input_tokens": 1, "output_tokens": 1}
        }"#;
        assert_eq!(parse(raw, false).unwrap().text, "still fine");
    }

    #[test]
    fn usage_keeps_the_four_counters_apart() {
        // Folding cache reads into input would understate a cached run: they
        // bill at a tenth of the rate, and writes at 1.25x.
        let raw = r#"{
            "content": [{"type": "text", "text": "ok"}],
            "usage": {"input_tokens": 100, "output_tokens": 20,
                      "cache_creation_input_tokens": 5000,
                      "cache_read_input_tokens": 40000}
        }"#;
        let usage = parse(raw, false).unwrap().usage;
        assert_eq!(usage, Usage { input: 100, output: 20, cache_write: 5000, cache_read: 40000, thinking: 0 });
    }

    #[test]
    fn absent_usage_counters_read_as_zero_not_as_an_error() {
        let raw = r#"{"content": [{"type": "text", "text": "ok"}]}"#;
        assert_eq!(parse(raw, false).unwrap().usage, Usage::default());
    }

    #[test]
    fn a_refusal_is_an_error_not_an_empty_string() {
        // It arrives as a perfectly successful 200 whose content is empty, so
        // reading content first would yield "" and hide the cause entirely.
        let raw = r#"{
            "content": [],
            "stop_reason": "refusal",
            "stop_details": {"category": "harmful_content",
                             "explanation": "declined to answer"},
            "usage": {"input_tokens": 12, "output_tokens": 0}
        }"#;
        match parse(raw, false) {
            Err(ModelError::Refused { category, explanation, usage }) => {
                // A refusal is billed; the ledger has to be able to see it.
                assert_eq!(usage.input, 12);
                assert_eq!(category.as_deref(), Some("harmful_content"));
                assert_eq!(explanation.as_deref(), Some("declined to answer"));
            }
            other => panic!("expected a refusal, got {other:?}"),
        }
    }

    #[test]
    fn a_refusal_without_details_is_still_a_refusal() {
        let raw = r#"{"content": [], "stop_reason": "refusal"}"#;
        assert!(matches!(parse(raw, false), Err(ModelError::Refused { .. })));
    }

    #[test]
    fn a_refusal_beats_a_schema_when_both_apply() {
        // Ordering matters: content is empty, so checking the schema first
        // would report "not JSON" for what is really a declined request.
        let raw = r#"{"content": [], "stop_reason": "refusal"}"#;
        assert!(matches!(parse(raw, true), Err(ModelError::Refused { .. })));
    }

    #[test]
    fn a_schema_request_returns_parsed_json() {
        let raw = r#"{
            "content": [{"type": "text", "text": "{\"answers\": [{\"field\": \"sex\"}]}"}],
            "usage": {"input_tokens": 1, "output_tokens": 1}
        }"#;
        let response = parse(raw, true).unwrap();
        assert_eq!(response.json.unwrap()["answers"][0]["field"], "sex");
        // the raw text is kept too, so an audit can still see what came back
        assert!(response.text.starts_with('{'));
    }

    #[test]
    fn a_schema_request_that_returns_prose_is_an_error() {
        let raw = r#"{"content": [{"type": "text", "text": "I think it was May."}]}"#;
        assert!(matches!(parse(raw, true), Err(ModelError::MalformedJson { .. })));
    }

    #[test]
    fn a_reply_clipped_at_the_ceiling_says_so_rather_than_only_bad_json() {
        // Both causes surface as "EOF while parsing a string", and the fix for
        // one — raise max_tokens — is nothing like the fix for the other. This
        // happened live and the report named only the JSON error, which is not
        // something anyone can act on.
        let raw = r#"{"stop_reason": "max_tokens",
                      "content": [{"type": "text", "text": "{\"answers\": [{\"value\": \"trunc"}]}"#;
        let Err(ModelError::Truncated { detail, .. }) = parse(raw, true) else {
            panic!("a clipped reply must not parse");
        };
        assert!(detail.contains("max_tokens"), "{detail}");
        assert!(detail.contains("characters of reply"), "{detail}");
    }

    #[test]
    fn a_reply_that_spent_the_whole_ceiling_thinking_says_that_specifically() {
        // The failure that cost a real run: adaptive thinking on the paper
        // layer consumed all 16,000 tokens and returned no answer at all. It
        // reads as a truncation, but "raise max_tokens or turn thinking off"
        // is a different instruction from "the answer was long", and the
        // generic message pointed at neither.
        let raw = r#"{"stop_reason": "max_tokens",
                      "content": [],
                      "usage": {"input_tokens": 12211, "output_tokens": 16000,
                                "output_tokens_details": {"thinking_tokens": 16000}}}"#;
        let Err(ModelError::Truncated { detail, usage }) = parse(raw, true) else {
            panic!("an empty clipped reply must not parse");
        };
        assert!(detail.contains("no answer tokens at all"), "{detail}");
        assert!(detail.contains("thinking counts toward max_tokens"), "{detail}");
        // And the tokens it burned are carried out, so the ledger sees them.
        assert_eq!(usage.output, 16_000);
        assert_eq!(usage.thinking, 16_000);
    }

    #[test]
    fn a_reply_that_stopped_normally_says_that_too() {
        // The other half: same failure, different cause, and the message has to
        // separate them or it has not helped.
        let raw = r#"{"stop_reason": "end_turn",
                      "content": [{"type": "text", "text": "I think it was May."}]}"#;
        let Err(ModelError::MalformedJson { detail, .. }) = parse(raw, true) else {
            panic!("prose is not JSON");
        };
        assert!(detail.contains("end_turn"), "{detail}");
        assert!(!detail.contains("max_tokens"), "{detail}");
    }

    #[test]
    fn a_body_that_is_not_a_response_is_a_decode_error() {
        assert!(matches!(parse("not json at all", false), Err(ModelError::Decode(_))));
    }

    // -- error statuses -----------------------------------------------------

    #[test]
    fn rate_limits_and_overload_are_their_own_variants() {
        let body = r#"{"error": {"type": "rate_limit_error", "message": "slow down"}}"#;
        match error_for(429, Some(30), body) {
            ModelError::RateLimited { retry_after } => assert_eq!(retry_after, Some(30)),
            other => panic!("expected RateLimited, got {other:?}"),
        }
        assert!(matches!(error_for(529, None, "{}"), ModelError::Overloaded));
    }

    #[test]
    fn other_statuses_keep_the_code_and_the_api_message() {
        // Flattening this to a string would lose the ability to branch on it,
        // and a 400 from a bad parameter combination is exactly what needs
        // reading.
        let body = r#"{"error": {"type": "invalid_request_error",
                                 "message": "thinking: unexpected field"}}"#;
        match error_for(400, None, body) {
            ModelError::Api { status, kind, message } => {
                assert_eq!(status, 400);
                assert_eq!(kind.as_deref(), Some("invalid_request_error"));
                assert_eq!(message, "thinking: unexpected field");
            }
            other => panic!("expected Api, got {other:?}"),
        }
    }

    #[test]
    fn an_unparseable_error_body_still_produces_an_error() {
        // A gateway can return HTML. That must not become a decode panic on top
        // of whatever the real failure was.
        match error_for(502, None, "<html>bad gateway</html>") {
            ModelError::Api { status, message, .. } => {
                assert_eq!(status, 502);
                assert!(message.contains("bad gateway"));
            }
            other => panic!("expected Api, got {other:?}"),
        }
    }

    #[test]
    fn only_the_transient_failures_are_retryable() {
        assert!(ModelError::RateLimited { retry_after: None }.is_retryable());
        assert!(ModelError::Overloaded.is_retryable());
        assert!(ModelError::Transport("reset".into()).is_retryable());
        assert!(ModelError::Api { status: 500, kind: None, message: String::new() }.is_retryable());

        // Retrying these identically cannot help.
        assert!(!ModelError::Refused { category: None, explanation: None, usage: Usage::default() }.is_retryable());
        assert!(!ModelError::MissingApiKey.is_retryable());
        assert!(!ModelError::MalformedJson { detail: "x".into(), usage: Usage::default() }.is_retryable());
        // Truncation is a configuration problem, not a sampling accident: it
        // recurred five attempts running when thinking ate the whole ceiling.
        assert!(!ModelError::Truncated { detail: "x".into(), usage: Usage::default() }.is_retryable());
        assert!(!ModelError::Api { status: 400, kind: None, message: String::new() }.is_retryable());
    }

    // -- the key guard ------------------------------------------------------

    #[test]
    fn a_client_cannot_be_built_without_a_plausible_key() {
        // There is no default key file and no environment fallback. A keyless
        // run fails here, before anything can be spent — which is the whole
        // point: a run once billed a key its author believed was uninvolved.
        assert!(matches!(
            Claude::new("", ModelId::Opus5),
            Err(ModelError::MissingApiKey)
        ));
        assert!(matches!(
            Claude::new("   ", ModelId::Opus5),
            Err(ModelError::MissingApiKey)
        ));
        assert!(matches!(
            Claude::new("not-a-key", ModelId::Opus5),
            Err(ModelError::InvalidApiKey(_))
        ));
        // a key read from a file, newline and all
        assert!(matches!(
            Claude::new("sk-ant-abc123\n", ModelId::Opus5),
            Err(ModelError::InvalidApiKey(_))
        ));
    }

    #[test]
    fn a_well_formed_key_is_accepted_without_contacting_anything() {
        // Not a real key: it only has to satisfy the shape check, and no
        // request is made here.
        let client = Claude::new("sk-ant-0000000000000000", ModelId::Sonnet5).unwrap();
        assert_eq!(client.model(), ModelId::Sonnet5);
        assert_eq!(client.body(&req())["model"], "claude-sonnet-5");
    }

    #[test]
    fn the_error_messages_say_what_to_do() {
        assert!(ModelError::MissingApiKey.to_string().contains("no default key file"));
        let refused = ModelError::Refused {
            category: Some("x".into()),
            explanation: Some("y".into()),
            usage: Usage::default(),
        };
        assert!(refused.to_string().contains("declined"));
    }

    // -- cost ---------------------------------------------------------------

    #[test]
    fn cost_prices_the_four_counters_at_their_own_rates() {
        // 1M plain input on Opus 5 is $5; 1M cache reads is a tenth of that;
        // 1M cache writes is 1.25x; 1M output is $25.
        let opus = ModelId::Opus5;
        let one_m = 1_000_000;
        assert!((call_cost(Usage { input: one_m, ..Default::default() }, opus, false) - 5.0).abs() < 1e-9);
        assert!((call_cost(Usage { cache_read: one_m, ..Default::default() }, opus, false) - 0.5).abs() < 1e-9);
        assert!((call_cost(Usage { cache_write: one_m, ..Default::default() }, opus, false) - 6.25).abs() < 1e-9);
        assert!((call_cost(Usage { output: one_m, ..Default::default() }, opus, false) - 25.0).abs() < 1e-9);
    }

    #[test]
    fn a_batch_call_costs_half() {
        let usage = Usage { input: 1_000_000, output: 1_000_000, ..Default::default() };
        let live = call_cost(usage, ModelId::Opus5, false);
        let batched = call_cost(usage, ModelId::Opus5, true);
        assert!((live - 30.0).abs() < 1e-9);
        assert!((batched - live * BATCH_DISCOUNT).abs() < 1e-9);
    }

    #[test]
    fn every_model_prices_separately() {
        let usage = Usage { input: 1_000_000, ..Default::default() };
        assert!((call_cost(usage, ModelId::Haiku45, false) - 1.0).abs() < 1e-9);
        assert!((call_cost(usage, ModelId::Sonnet5, false) - 2.0).abs() < 1e-9);
        assert!((call_cost(usage, ModelId::Opus5, false) - 5.0).abs() < 1e-9);
        assert!((call_cost(usage, ModelId::Opus48, false) - 5.0).abs() < 1e-9);
    }

    #[test]
    fn a_cached_run_costs_a_fraction_of_a_cold_one() {
        // The measurement the caching exists for: the same 40k-token prefix
        // read from cache rather than sent fresh.
        let cold = call_cost(Usage { input: 40_000, output: 500, ..Default::default() },
                             ModelId::Opus5, false);
        let warm = call_cost(Usage { input: 0, cache_read: 40_000, output: 500,
                                     ..Default::default() }, ModelId::Opus5, false);
        assert!(warm < cold);
        assert!(warm < cold * 0.5, "cached {warm} should be far below cold {cold}");
    }

    #[test]
    fn usage_accumulates() {
        let mut total = Usage::default();
        total.add(Usage { input: 10, output: 1, cache_write: 2, cache_read: 3, thinking: 0 });
        total.add(Usage { input: 20, output: 2, cache_write: 0, cache_read: 7, thinking: 0 });
        assert_eq!(total, Usage { input: 30, output: 3, cache_write: 2, cache_read: 10, thinking: 0 });
    }
}
