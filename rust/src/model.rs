use std::fmt;

use serde::{Deserialize, Serialize};

pub mod budget;
pub mod claude;
pub mod retry;

// One call to a language model.
//
// Synchronous on purpose. The only I/O in this pipeline is here, and making it
// async would colour every caller above it — `Layer::process`, `from_project`,
// `from_corpus` — none of which touch the network. The useful concurrency is
// bounded by the provider's rate limit rather than by threads, and the batch
// path turns thousands of records into three requests and some polling, so
// there is nothing here that parked threads cannot do. If that changes, this
// trait is the single seam to change.
//
// `Send + Sync` so a caller can still fan out over records with a thread pool
// without the trait having to know about it.
pub trait Model: Send + Sync {
    fn complete(&self, request: &Request) -> Result<Response, ModelError>;

    // Dollars this usage cost at this provider's rates. On the trait rather
    // than beside the Anthropic prices so a budget can meter any provider it
    // is handed — a local model answers zero and the accounting still works.
    fn price(&self, usage: Usage) -> f64;

    // Many requests at once, keyed by the caller's own keys.
    //
    // The default sends them one at a time, so a provider with no batch
    // endpoint works unchanged — OpenRouter's upstream has none. A provider
    // that does have one overrides this and the caller never learns which it
    // got, which is the point: the layer asks for many completions and the
    // transport decides how.
    //
    // Per-key `Result` rather than a failing whole: one bad request must not
    // cost the rest, which is the same rule the batch endpoint itself applies.
    fn complete_many(
        &self,
        requests: &std::collections::BTreeMap<String, Request>,
    ) -> std::collections::BTreeMap<String, Result<Response, ModelError>> {
        requests
            .iter()
            .map(|(key, request)| (key.clone(), self.complete(request)))
            .collect()
    }

    // Whether `complete_many` is anything other than a loop. Callers use it to
    // decide whether batching is worth asking for, and budgets use it to know
    // which rate applies.
    fn supports_batch(&self) -> bool {
        false
    }

    // Dollars for usage billed through `complete_many`. Defaults to the live
    // rate because the default implementation *is* live calls; a real batch
    // endpoint overrides it with its discount.
    fn price_many(&self, usage: Usage) -> f64 {
        self.price(usage)
    }
}

// So a boxed model can be wrapped by `Retrying` and `Budgeted` like any other.
// Without this, a caller holding a `Box<dyn Model>` — which is what a layer
// holds — cannot put a budget around it.
impl Model for Box<dyn Model> {
    #[inline]
    fn complete(&self, request: &Request) -> Result<Response, ModelError> {
        (**self).complete(request)
    }

    #[inline]
    fn price(&self, usage: Usage) -> f64 {
        (**self).price(usage)
    }

    #[inline]
    fn complete_many(
        &self,
        requests: &std::collections::BTreeMap<String, Request>,
    ) -> std::collections::BTreeMap<String, Result<Response, ModelError>> {
        (**self).complete_many(requests)
    }

    #[inline]
    fn supports_batch(&self) -> bool {
        (**self).supports_batch()
    }

    #[inline]
    fn price_many(&self, usage: Usage) -> f64 {
        (**self).price_many(usage)
    }
}

// What a layer asks for, in provider-neutral terms.
//
// The *model* is deliberately not here: it belongs to the client, so
// `Layer::LLMNaive { model, .. }` carries a thing already configured to speak to
// one model, and the layer does not have to know which providers exist. That is
// also what lets two layers in the same run hold two different models.
#[derive(Clone, Debug, PartialEq)]
pub struct Request {
    // The cached prefix. Must stay byte-identical across calls or the cache
    // misses and every call pays for the instructions again.
    pub system: Option<String>,
    pub prompt: String,
    // A JSON Schema. When set, the provider constrains generation to it, which
    // beats asking for a shape and parsing hopefully.
    pub schema: Option<serde_json::Value>,
    pub max_tokens: u32,
    pub effort: Option<Effort>,
    pub thinking: Thinking,
    pub cache_system: bool,
    pub cache_ttl: Option<CacheTtl>,
}

// Defaults mirror `claude.complete`: adaptive thinking, medium effort, caching
// on, 16k output. Changing one of these changes what every layer pays.
impl Default for Request {
    #[inline]
    fn default() -> Self {
        Self {
            system: None,
            prompt: String::new(),
            schema: None,
            max_tokens: 16_000,
            effort: Some(Effort::Medium),
            thinking: Thinking::Adaptive,
            cache_system: true,
            cache_ttl: None,
        }
    }
}

impl Request {
    #[inline]
    pub fn new(prompt: impl Into<String>) -> Self {
        Self { prompt: prompt.into(), ..Default::default() }
    }

    #[inline]
    pub fn system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }

    #[inline]
    pub fn schema(mut self, schema: serde_json::Value) -> Self {
        self.schema = Some(schema);
        self
    }

    #[inline]
    pub fn effort(mut self, effort: Effort) -> Self {
        self.effort = Some(effort);
        self
    }

    #[inline]
    pub fn thinking(mut self, thinking: Thinking) -> Self {
        self.thinking = thinking;
        self
    }

    #[inline]
    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    #[inline]
    pub fn cache_ttl(mut self, ttl: CacheTtl) -> Self {
        self.cache_ttl = Some(ttl);
        self
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Effort {
    Low,
    Medium,
    High,
    XHigh,
    Max,
}

impl Effort {
    #[inline]
    pub fn as_str(self) -> &'static str {
        match self {
            Effort::Low => "low",
            Effort::Medium => "medium",
            Effort::High => "high",
            Effort::XHigh => "xhigh",
            Effort::Max => "max",
        }
    }
}

// Four states, because the API has two different ways of saying "think".
//
// `Adaptive` is the newer one: Claude decides per request whether to think and
// how deeply. `Enabled` is *extended thinking*, the older manual mode, where the
// caller sets the budget. They are not interchangeable per model — 4.6-and-later
// models reject `budget_tokens` with a 400, and models older than that reject
// `adaptive` — so "think" cannot be a single value.
//
// `Disabled` and `Unset` differ in bill, not in wording: omitting the parameter
// leaves thinking *on* for the models that default to it, so switching it off
// has to be said out loud, while the models that predate the field reject being
// told anything at all. A bool cannot hold any of this, which is how a run once
// cost 2.5x its estimate.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Thinking {
    #[default]
    Adaptive,
    // Extended thinking with a fixed budget, in tokens. The only way to make a
    // pre-4.6 model think at all — without it, Haiku 4.5 cannot think in this
    // pipeline, because `Adaptive` is dropped there.
    Enabled { budget_tokens: u32 },
    Disabled,
    Unset,
}

impl Thinking {
    #[inline]
    pub fn as_str(self) -> &'static str {
        match self {
            Thinking::Adaptive => "adaptive",
            Thinking::Enabled { .. } => "enabled",
            Thinking::Disabled => "disabled",
            Thinking::Unset => "unset",
        }
    }
}

// How long a cached prefix survives. An hour costs 2x on write against five
// minutes' 1.25x, and is worth it only when a run can take longer than five
// minutes to drain — a prefix that expires mid-batch is rewritten by every
// remaining request, which is the problem caching exists to avoid.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CacheTtl {
    FiveMinutes,
    OneHour,
}

impl CacheTtl {
    #[inline]
    pub fn as_str(self) -> &'static str {
        match self {
            CacheTtl::FiveMinutes => "5m",
            CacheTtl::OneHour => "1h",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Response {
    // Every text block concatenated. Thinking and tool blocks are dropped.
    pub text: String,
    // Set when the request carried a schema. Parsed once here so callers do not
    // each re-parse the same string.
    pub json: Option<serde_json::Value>,
    // Why generation ended. Kept because "stopped at the ceiling" and "chose to
    // answer briefly" are indistinguishable from the usage alone, and thinking
    // tokens count toward `max_tokens` — so a short answer after a long think
    // is exactly the case worth telling apart.
    pub stop_reason: Option<String>,
    pub usage: Usage,
}

// The four counters, kept apart because they bill at different rates. Folding
// cache reads into `input` would understate a cached run and overstate a cold
// one; on the Sonnet 5 comparison, cache writes alone were 15% of the bill.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    pub input: u64,
    pub output: u64,
    pub cache_write: u64,
    pub cache_read: u64,
    // How many of the `output` tokens were internal reasoning.
    //
    // A SUBSET of `output`, not an addition to it — thinking is billed as
    // output and is already counted there. Pricing it again would double-bill.
    // Kept because a run cannot otherwise tell an expensive answer from an
    // expensive think: Sonnet spent 843 output tokens a call to fill 2 fields,
    // and only this field says how much of that was reasoning.
    //
    // `default` so exports written before this field existed still load. Adding
    // a non-optional field to a serialized type silently orphans every file
    // that predates it, which is how a twelve-run comparison quietly became a
    // seven-run one.
    #[serde(default)]
    pub thinking: u64,
}

impl Usage {
    #[inline]
    pub fn add(&mut self, other: Usage) {
        self.input += other.input;
        self.output += other.output;
        self.cache_write += other.cache_write;
        self.cache_read += other.cache_read;
        self.thinking += other.thinking;
    }
}

#[derive(Debug)]
pub enum ModelError {
    // No key was supplied, or it does not look like one. Deliberately its own
    // variant and checked before any request is built: a run that is going to
    // fail on credentials should fail before it can spend anything.
    MissingApiKey,
    InvalidApiKey(String),
    // Declined by safety classifiers rather than answered. Arrives as a
    // perfectly successful HTTP 200 whose content is empty or partial, so code
    // reading `content[0]` sees an index error rather than the real cause.
    Refused {
        category: Option<String>,
        explanation: Option<String>,
    },
    // Retryable. `retry_after` is the server's own answer to "how long".
    RateLimited {
        retry_after: Option<u64>,
    },
    // Retryable: the provider is up but shedding load.
    Overloaded,
    // Any other non-success status, kept whole rather than flattened to a
    // string so a caller can branch on the code.
    Api {
        status: u16,
        kind: Option<String>,
        message: String,
    },
    Transport(String),
    // The body arrived but did not have the shape a response has.
    Decode(String),
    // A schema was requested and the reply did not parse as JSON.
    MalformedJson(String),
    // The running ledger reached its ceiling. Raised *before* the call that
    // would have crossed it, so the figure reported is what has actually been
    // billed and not an estimate.
    BudgetExceeded {
        spent: f64,
        limit: f64,
    },
    // Every retry was used and the last attempt still failed. The cause is
    // kept rather than replaced: "gave up after 5 attempts" without saying at
    // what is not a diagnosis.
    RetriesExhausted {
        attempts: u32,
        last: Box<ModelError>,
    },
}

impl fmt::Display for ModelError {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelError::MissingApiKey => write!(
                f,
                "no API key supplied; there is no default key file, so one must \
                 be passed explicitly"
            ),
            ModelError::InvalidApiKey(why) => write!(f, "API key rejected: {why}"),
            ModelError::Refused { category, explanation } => {
                write!(f, "the model declined to answer")?;
                if let Some(category) = category {
                    write!(f, " ({category})")?;
                }
                if let Some(explanation) = explanation {
                    write!(f, ": {explanation}")?;
                }
                Ok(())
            }
            ModelError::RateLimited { retry_after } => match retry_after {
                Some(secs) => write!(f, "rate limited; retry after {secs}s"),
                None => write!(f, "rate limited"),
            },
            ModelError::Overloaded => write!(f, "the provider is overloaded"),
            ModelError::Api { status, kind, message } => match kind {
                Some(kind) => write!(f, "API error {status} ({kind}): {message}"),
                None => write!(f, "API error {status}: {message}"),
            },
            ModelError::Transport(e) => write!(f, "transport error: {e}"),
            ModelError::Decode(e) => write!(f, "could not read the response: {e}"),
            ModelError::MalformedJson(e) => {
                write!(f, "a schema was requested but the reply is not JSON: {e}")
            }
            ModelError::BudgetExceeded { spent, limit } => write!(
                f,
                "spend ceiling reached: ${spent:.4} of ${limit:.2} billed, so this \
                 call was not made. Raise the ceiling to authorise more, or pass no \
                 ceiling to disable the check."
            ),
            ModelError::RetriesExhausted { attempts, last } => {
                write!(f, "gave up after {attempts} attempts; last failure: {last}")
            }
        }
    }
}

impl std::error::Error for ModelError {}

impl ModelError {
    // Whether retrying the identical request could succeed. A refusal or a bad
    // schema will not change on a second attempt; a 429 or a 529 might.
    #[inline]
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            ModelError::RateLimited { .. }
                | ModelError::Overloaded
                | ModelError::Transport(_)
                | ModelError::Api { status: 500..=599, .. }
        )
    }
}
