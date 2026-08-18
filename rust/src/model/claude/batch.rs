use std::collections::BTreeMap;
use std::thread::sleep;
use std::time::{Duration, Instant};

use serde::Deserialize;
use serde_json::{json, Value};

use super::{body_for, error_for, parse, Claude, ModelId};
use crate::model::budget::Budget;
use crate::model::{ModelError, Request, Response, Usage};

// The Message Batches API — the same work at half price, paid for in latency.
//
// Every token in a batch bills at 50%, with no change to the model, the prompt
// or the output. Submit, poll, collect. Most batches finish well inside an
// hour; the API's own ceiling is 24 hours.
//
// This is deliberately not part of the `Model` trait. Its shape is different —
// many requests in, keyed results out, minutes or hours later — and no other
// provider in the plan offers it: OpenRouter's upstream has no batch endpoint
// at all. A trait can come later if that changes; inventing one now would mean
// designing against a single implementation.
//
// As in `claude.rs`, the pure parts are separated from the I/O so the payload
// and the result handling are testable with no network and no key.

// The API's own ceiling. Exceeding it is an error rather than a silent split:
// a caller who thinks they submitted one batch and got two has lost track of
// what is in flight, and the halves finish at different times.
pub const MAX_BATCH_REQUESTS: usize = 100_000;

const DEFAULT_POLL: Duration = Duration::from_secs(10);
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(86_400);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchId(pub String);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BatchCounts {
    pub processing: u64,
    pub succeeded: u64,
    pub errored: u64,
    pub canceled: u64,
    pub expired: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BatchStatus {
    pub id: BatchId,
    // `processing_status == "ended"`. Ended does not mean every request
    // succeeded — it means none is still running.
    pub ended: bool,
    pub raw_status: String,
    pub counts: BatchCounts,
}

// Why one request in a batch produced no answer.
//
// Python drops these silently and leaves the caller to diff the key sets. They
// are reported here instead: a batch that half-failed and a batch that half-
// refused need different responses, and "absent" cannot tell them apart.
#[derive(Clone, Debug, PartialEq)]
pub enum BatchFailure {
    Errored { kind: Option<String>, message: String },
    Canceled,
    Expired,
    Refused { category: Option<String>, explanation: Option<String> },
    MalformedJson(String),
    // A custom_id came back that was never sent, or came back twice.
    UnknownId(String),
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct BatchOutcome {
    // Keyed by the caller's own keys, not by the ids sent over the wire.
    pub results: BTreeMap<String, Response>,
    pub failures: BTreeMap<String, BatchFailure>,
    // Summed across every request that reported usage, including the ones that
    // failed after being billed. Price it with `call_cost(.., batch: true)`.
    pub usage: Usage,
}

#[derive(Clone, Copy, Debug)]
pub struct BatchOptions {
    pub poll: Duration,
    pub timeout: Duration,
    // What the caller believes this batch will cost, checked against the budget
    // before anything is submitted. `None` checks only that the ledger is not
    // already full, which is the weaker guarantee: a batch commits everything at
    // once, so by the time it is running there is nothing left to stop.
    pub estimate: Option<f64>,
}

impl Default for BatchOptions {
    fn default() -> Self {
        Self { poll: DEFAULT_POLL, timeout: DEFAULT_TIMEOUT, estimate: None }
    }
}

// The submission payload, plus the wire ids in the order they were sent.
//
// Caller keys are not sent as-is. The API constrains the `custom_id` charset,
// and record ids here contain dots — `SRX1.SRS1` for a pooled experiment — so
// each key becomes a positional `r{i}` and is mapped back on the way out. The
// returned Vec is that mapping.
pub fn batch_body(
    model: ModelId,
    requests: &BTreeMap<String, Request>,
) -> Result<(Value, Vec<String>), ModelError> {
    if requests.len() > MAX_BATCH_REQUESTS {
        return Err(ModelError::Api {
            status: 0,
            kind: Some("batch_too_large".into()),
            message: format!(
                "{} requests exceeds the API ceiling of {MAX_BATCH_REQUESTS}; \
                 split the work deliberately rather than relying on this",
                requests.len()
            ),
        });
    }

    let keys: Vec<String> = requests.keys().cloned().collect();
    let items: Vec<Value> = keys
        .iter()
        .enumerate()
        .map(|(i, key)| {
            let mut params = body_for(model, &requests[key]);
            // The Batches API rejects `fallbacks` outright. Stripped here rather
            // than by building a different body, so the live and batched paths
            // cannot drift: a batch differing from the live call by so much as a
            // whitespace change is a different cache prefix and a different
            // measurement.
            if let Some(map) = params.as_object_mut() {
                map.remove("fallbacks");
                map.remove("betas");
            }
            json!({ "custom_id": format!("r{i}"), "params": params })
        })
        .collect();

    Ok((json!({ "requests": items }), keys))
}

pub fn parse_status(raw: &str) -> Result<BatchStatus, ModelError> {
    #[derive(Deserialize)]
    struct Wire {
        id: String,
        processing_status: String,
        #[serde(default)]
        request_counts: Counts,
    }
    #[derive(Default, Deserialize)]
    struct Counts {
        #[serde(default)]
        processing: u64,
        #[serde(default)]
        succeeded: u64,
        #[serde(default)]
        errored: u64,
        #[serde(default)]
        canceled: u64,
        #[serde(default)]
        expired: u64,
    }

    let wire: Wire =
        serde_json::from_str(raw).map_err(|e| ModelError::Decode(e.to_string()))?;
    Ok(BatchStatus {
        id: BatchId(wire.id),
        ended: wire.processing_status == "ended",
        raw_status: wire.processing_status,
        counts: BatchCounts {
            processing: wire.request_counts.processing,
            succeeded: wire.request_counts.succeeded,
            errored: wire.request_counts.errored,
            canceled: wire.request_counts.canceled,
            expired: wire.request_counts.expired,
        },
    })
}

// Results arrive as JSONL — one result object per line, not a JSON array.
//
// A line that will not parse is recorded against the key it claims rather than
// aborting the collection: one malformed result must not cost the whole batch,
// which is the same reasoning that makes a refusal a per-key failure.
pub fn parse_results(jsonl: &str, keys: &[String]) -> BatchOutcome {
    #[derive(Deserialize)]
    struct Line {
        custom_id: String,
        result: Outcome,
    }
    #[derive(Deserialize)]
    #[serde(tag = "type", rename_all = "snake_case")]
    enum Outcome {
        Succeeded { message: Value },
        Errored { error: Option<ApiError> },
        Canceled,
        Expired,
        #[serde(other)]
        Unknown,
    }
    #[derive(Deserialize)]
    struct ApiError {
        #[serde(rename = "type")]
        kind: Option<String>,
        message: Option<String>,
    }

    let mut outcome = BatchOutcome::default();

    for raw_line in jsonl.lines() {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        let parsed: Line = match serde_json::from_str(line) {
            Ok(parsed) => parsed,
            Err(e) => {
                outcome.failures.insert(
                    format!("<unparseable line {}>", outcome.failures.len()),
                    BatchFailure::MalformedJson(e.to_string()),
                );
                continue;
            }
        };

        let Some(key) = key_for(&parsed.custom_id, keys) else {
            outcome.failures.insert(
                parsed.custom_id.clone(),
                BatchFailure::UnknownId(parsed.custom_id),
            );
            continue;
        };

        match parsed.result {
            Outcome::Succeeded { message } => {
                // The message body is the same shape a live call returns, so it
                // goes through the same parser — including the refusal check,
                // which arrives here as a successful result whose content is
                // empty.
                let text = message.to_string();
                // A batched request carries a schema whenever the live one did;
                // asking the parser for JSON is what surfaces a reply that did
                // not conform.
                match parse(&text, true) {
                    Ok(response) => {
                        outcome.usage.add(response.usage);
                        outcome.results.insert(key, response);
                    }
                    Err(ModelError::Refused { category, explanation }) => {
                        // Still billed, so still counted. Usage is re-read
                        // directly because the parse returned early.
                        outcome.usage.add(usage_of(&message));
                        outcome
                            .failures
                            .insert(key, BatchFailure::Refused { category, explanation });
                    }
                    Err(ModelError::MalformedJson(e)) => {
                        outcome.usage.add(usage_of(&message));
                        outcome.failures.insert(key, BatchFailure::MalformedJson(e));
                    }
                    Err(e) => {
                        outcome
                            .failures
                            .insert(key, BatchFailure::MalformedJson(e.to_string()));
                    }
                }
            }
            Outcome::Errored { error } => {
                let (kind, message) = match error {
                    Some(e) => (e.kind, e.message.unwrap_or_default()),
                    None => (None, String::new()),
                };
                outcome
                    .failures
                    .insert(key, BatchFailure::Errored { kind, message });
            }
            Outcome::Canceled => {
                outcome.failures.insert(key, BatchFailure::Canceled);
            }
            Outcome::Expired => {
                outcome.failures.insert(key, BatchFailure::Expired);
            }
            Outcome::Unknown => {
                outcome.failures.insert(
                    key,
                    BatchFailure::Errored {
                        kind: Some("unknown_result_type".into()),
                        message: "the API reported a result type this build does not \
                                  recognise"
                            .into(),
                    },
                );
            }
        }
    }
    outcome
}

// `r12` -> keys[12]. Anything else is not an id this code sent.
fn key_for(custom_id: &str, keys: &[String]) -> Option<String> {
    let index: usize = custom_id.strip_prefix('r')?.parse().ok()?;
    keys.get(index).cloned()
}

fn usage_of(message: &Value) -> Usage {
    let at = |name: &str| message["usage"][name].as_u64().unwrap_or(0);
    Usage {
        input: at("input_tokens"),
        output: at("output_tokens"),
        cache_write: at("cache_creation_input_tokens"),
        cache_read: at("cache_read_input_tokens"),
        thinking: message["usage"]["output_tokens_details"]["thinking_tokens"]
            .as_u64()
            .unwrap_or(0),
    }
}

// ---------------------------------------------------------------------------
// I/O
// ---------------------------------------------------------------------------
// Three primitives rather than one blocking call, because a batch can outlive
// the process that submitted it: the id is worth persisting, and a resumed run
// should be able to poll and collect without re-submitting.

impl Claude {
    pub fn submit_batch(
        &self,
        requests: &BTreeMap<String, Request>,
    ) -> Result<(BatchId, Vec<String>), ModelError> {
        let (body, keys) = batch_body(self.model(), requests)?;
        let raw = self.send("POST", &self.url("/messages/batches"), Some(&body))?;
        let status = parse_status(&raw)?;
        Ok((status.id, keys))
    }

    pub fn batch_status(&self, id: &BatchId) -> Result<BatchStatus, ModelError> {
        let url = self.url(&format!("/messages/batches/{}", id.0));
        parse_status(&self.send("GET", &url, None)?)
    }

    pub fn batch_results(
        &self,
        id: &BatchId,
        keys: &[String],
    ) -> Result<BatchOutcome, ModelError> {
        let url = self.url(&format!("/messages/batches/{}/results", id.0));
        Ok(parse_results(&self.send("GET", &url, None)?, keys))
    }

    // Submit, poll to completion, collect. The convenience path.
    //
    // Returns as soon as the batch ends, whatever the mix of outcomes: a batch
    // where every request errored is a successful call that reports failures,
    // not an error.
    //
    // The budget is a required argument rather than an option, because
    // `Budgeted` cannot reach this path — it wraps `Model::complete`, and this
    // is not that. A batch is the largest single commitment this client can
    // make, so it is the last place that should be metered by accident. Pass
    // `Budget::unlimited()` to opt out deliberately.
    pub fn run_batch(
        &self,
        requests: &BTreeMap<String, Request>,
        options: BatchOptions,
        budget: &Budget,
        mut progress: impl FnMut(&BatchStatus),
    ) -> Result<BatchOutcome, ModelError> {
        if requests.is_empty() {
            return Ok(BatchOutcome::default());
        }
        // Before submission, so a refusal costs nothing.
        match options.estimate {
            Some(estimate) => budget.reserve(estimate)?,
            None => budget.check()?,
        }
        let (id, keys) = self.submit_batch(requests)?;
        let deadline = Instant::now() + options.timeout;

        loop {
            let status = self.batch_status(&id)?;
            progress(&status);
            if status.ended {
                break;
            }
            if Instant::now() >= deadline {
                return Err(ModelError::Api {
                    status: 0,
                    kind: Some("batch_timeout".into()),
                    message: format!(
                        "batch {} still {} after {:?}; it is not lost — poll it with \
                         batch_status and collect with batch_results",
                        id.0, status.raw_status, options.timeout
                    ),
                });
            }
            sleep(options.poll);
        }
        let outcome = self.batch_results(&id, &keys)?;
        // Recorded at the batch rate from the usage actually reported, so the
        // ledger holds the billed figure rather than the estimate above.
        budget.record(self.price_batch(outcome.usage), outcome.usage);
        Ok(outcome)
    }

    // Dollars for a batch's usage. Separate from `Model::price`, which prices
    // the live path: the same tokens bill at half here.
    #[inline]
    pub fn price_batch(&self, usage: Usage) -> f64 {
        super::call_cost(usage, self.model(), true)
    }

    fn send(&self, method: &str, url: &str, body: Option<&Value>) -> Result<String, ModelError> {
        let mut req = match method {
            "POST" => self.http().post(url),
            _ => self.http().get(url),
        }
        .header("x-api-key", self.api_key())
        .header("anthropic-version", super::API_VERSION);

        if let Some(body) = body {
            req = req.json(body);
        }
        let response = req
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
        Ok(raw)
    }
}

// Pre-warming is deliberately absent, and this comment is the reason it should
// stay absent rather than be re-added on sight.
//
// The textbook remedy for parallel fan-out is to send one live request per
// distinct cache prefix so the rest of the batch reads what it wrote. It was
// implemented and measured on the Python side over 57 requests: the mechanism
// worked — read/write ratio rose from 0.62 to 1.19 and writes fell 26% — and
// the bill still went *up*, $0.1786 to $0.1858. The warm calls move off the
// 50% discount and are the very ones paying the 1.25x cache write, and one warm
// call does not serialise a parallel batch, so the remaining 45 requests still
// wrote 81,800 tokens between them. Measure again before adding it.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{CacheTtl, Thinking};

    // Offline and keyless, like the tests in `claude.rs`. The three functions
    // that open a socket are `submit_batch`, `batch_status` and
    // `batch_results`; nothing below calls any of them.

    fn requests(keys: &[&str]) -> BTreeMap<String, Request> {
        keys.iter()
            .map(|k| (k.to_string(), Request::new(format!("prompt for {k}"))))
            .collect()
    }

    fn schema() -> Value {
        json!({"type": "object", "properties": {"sex": {"type": "string"}}})
    }

    // -- the payload --------------------------------------------------------

    #[test]
    fn caller_keys_are_replaced_with_positional_ids() {
        // A pooled experiment's id contains a dot, which the API's custom_id
        // charset does not accept. Sending the key as-is would fail the whole
        // batch on a record shape that is entirely legitimate.
        let requests = requests(&["SRX000001.SRS000001", "SRX000002"]);
        let (body, keys) = batch_body(ModelId::Opus5, &requests).unwrap();

        let items = body["requests"].as_array().unwrap();
        assert_eq!(items.len(), 2);
        assert_eq!(items[0]["custom_id"], "r0");
        assert_eq!(items[1]["custom_id"], "r1");
        // and the mapping back is the returned order
        assert_eq!(keys, vec!["SRX000001.SRS000001", "SRX000002"]);
        assert!(body.to_string().contains("prompt for SRX000001.SRS000001"));
    }

    #[test]
    fn fallbacks_are_stripped_because_the_batches_api_rejects_them() {
        // Opus 5 is offered the server-side fallback on the live path. Sending
        // the same body to the batches endpoint is rejected outright, so it has
        // to come off here — and only here, or the two paths would build
        // different cache prefixes and stop being comparable.
        let live = body_for(ModelId::Opus5, &Request::new("p"));
        assert!(live.get("fallbacks").is_some(), "precondition: live path sends it");

        let (body, _) = batch_body(ModelId::Opus5, &requests(&["a"])).unwrap();
        let params = &body["requests"][0]["params"];
        assert!(params.get("fallbacks").is_none());
        assert!(params.get("betas").is_none());
    }

    #[test]
    fn everything_else_matches_the_live_body_exactly() {
        // A batch that differs from the live call by more than the fallback is
        // a different measurement. Compared field by field rather than as a
        // whole, so a future divergence names itself.
        let request = Request::new("p")
            .system("PREFIX")
            .schema(schema())
            .cache_ttl(CacheTtl::OneHour)
            .thinking(Thinking::Disabled);
        let mut live = body_for(ModelId::Sonnet5, &request);
        let map: BTreeMap<String, Request> = [("k".to_string(), request)].into();
        let (batched, _) = batch_body(ModelId::Sonnet5, &map).unwrap();

        live.as_object_mut().unwrap().remove("fallbacks");
        live.as_object_mut().unwrap().remove("betas");
        assert_eq!(&live, &batched["requests"][0]["params"]);
    }

    #[test]
    fn an_oversized_batch_is_refused_rather_than_split() {
        // Silently splitting would leave the caller believing one batch is in
        // flight when two are, finishing at different times.
        let requests: BTreeMap<String, Request> = (0..MAX_BATCH_REQUESTS + 1)
            .map(|i| (format!("k{i}"), Request::new("p")))
            .collect();
        match batch_body(ModelId::Opus5, &requests) {
            Err(ModelError::Api { kind, .. }) => {
                assert_eq!(kind.as_deref(), Some("batch_too_large"));
            }
            other => panic!("expected a refusal, got {other:?}"),
        }
    }

    #[test]
    fn an_empty_batch_produces_an_empty_payload() {
        let (body, keys) = batch_body(ModelId::Opus5, &BTreeMap::new()).unwrap();
        assert!(body["requests"].as_array().unwrap().is_empty());
        assert!(keys.is_empty());
    }

    // -- status -------------------------------------------------------------

    #[test]
    fn status_reports_ended_and_the_counts() {
        let raw = r#"{
            "id": "msgbatch_01",
            "processing_status": "ended",
            "request_counts": {"processing": 0, "succeeded": 55,
                               "errored": 2, "canceled": 0, "expired": 0}
        }"#;
        let status = parse_status(raw).unwrap();
        assert_eq!(status.id, BatchId("msgbatch_01".into()));
        assert!(status.ended);
        assert_eq!(status.counts.succeeded, 55);
        assert_eq!(status.counts.errored, 2);
    }

    #[test]
    fn in_progress_is_not_ended() {
        let raw = r#"{"id": "msgbatch_01", "processing_status": "in_progress",
                      "request_counts": {"processing": 40, "succeeded": 17}}"#;
        let status = parse_status(raw).unwrap();
        assert!(!status.ended);
        assert_eq!(status.raw_status, "in_progress");
        assert_eq!(status.counts.processing, 40);
    }

    #[test]
    fn ended_does_not_mean_everything_succeeded() {
        // The polling loop stops on `ended`; the outcome is read separately.
        // Conflating the two would report a wholly-failed batch as a success.
        let raw = r#"{"id": "b", "processing_status": "ended",
                      "request_counts": {"succeeded": 0, "errored": 12}}"#;
        let status = parse_status(raw).unwrap();
        assert!(status.ended);
        assert_eq!(status.counts.succeeded, 0);
        assert_eq!(status.counts.errored, 12);
    }

    #[test]
    fn missing_counts_read_as_zero() {
        let status = parse_status(r#"{"id": "b", "processing_status": "ended"}"#).unwrap();
        assert_eq!(status.counts, BatchCounts::default());
    }

    #[test]
    fn a_body_that_is_not_a_batch_is_a_decode_error() {
        assert!(matches!(parse_status("{}"), Err(ModelError::Decode(_))));
    }

    // -- results ------------------------------------------------------------

    fn succeeded_line(id: &str, text: &str, input: u64, output: u64) -> String {
        json!({
            "custom_id": id,
            "result": {
                "type": "succeeded",
                "message": {
                    "content": [{"type": "text", "text": text}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": input, "output_tokens": output}
                }
            }
        })
        .to_string()
    }

    #[test]
    fn results_are_keyed_back_to_the_callers_own_keys() {
        let keys = vec!["SRX1.SRS1".to_string(), "SRX2".to_string()];
        let jsonl = format!(
            "{}\n{}",
            succeeded_line("r0", r#"{"sex":"female"}"#, 10, 2),
            succeeded_line("r1", r#"{"sex":"male"}"#, 11, 3),
        );
        let outcome = parse_results(&jsonl, &keys);

        assert_eq!(outcome.results.len(), 2);
        assert_eq!(outcome.results["SRX1.SRS1"].json.as_ref().unwrap()["sex"], "female");
        assert_eq!(outcome.results["SRX2"].json.as_ref().unwrap()["sex"], "male");
        assert!(outcome.failures.is_empty());
    }

    #[test]
    fn usage_is_summed_across_the_batch() {
        // The number the 50% discount is applied to. Summed here rather than
        // per-request so a caller prices the batch once.
        let keys = vec!["a".to_string(), "b".to_string()];
        let jsonl = format!(
            "{}\n{}",
            succeeded_line("r0", "{}", 100, 10),
            succeeded_line("r1", "{}", 200, 20),
        );
        let outcome = parse_results(&jsonl, &keys);
        assert_eq!(outcome.usage.input, 300);
        assert_eq!(outcome.usage.output, 30);
    }

    #[test]
    fn every_failure_kind_is_reported_rather_than_dropped() {
        // Python leaves these absent and lets the caller diff the key sets. A
        // half-errored batch and a half-refused one need different responses,
        // and absence cannot tell them apart.
        let keys: Vec<String> = (0..4).map(|i| format!("k{i}")).collect();
        let jsonl = [
            json!({"custom_id": "r0", "result": {"type": "errored",
                   "error": {"type": "invalid_request_error", "message": "bad schema"}}}),
            json!({"custom_id": "r1", "result": {"type": "canceled"}}),
            json!({"custom_id": "r2", "result": {"type": "expired"}}),
            json!({"custom_id": "r3", "result": {"type": "succeeded", "message": {
                   "content": [], "stop_reason": "refusal",
                   "stop_details": {"category": "harmful_content"},
                   "usage": {"input_tokens": 5, "output_tokens": 0}}}}),
        ]
        .map(|v| v.to_string())
        .join("\n");

        let outcome = parse_results(&jsonl, &keys);
        assert!(outcome.results.is_empty());
        assert_eq!(outcome.failures.len(), 4);
        assert!(matches!(
            &outcome.failures["k0"],
            BatchFailure::Errored { kind, message }
                if kind.as_deref() == Some("invalid_request_error") && message == "bad schema"
        ));
        assert_eq!(outcome.failures["k1"], BatchFailure::Canceled);
        assert_eq!(outcome.failures["k2"], BatchFailure::Expired);
        assert!(matches!(
            &outcome.failures["k3"],
            BatchFailure::Refused { category, .. } if category.as_deref() == Some("harmful_content")
        ));
        // a refusal is billed, so it still counts toward the total
        assert_eq!(outcome.usage.input, 5);
    }

    #[test]
    fn one_bad_reply_does_not_cost_the_whole_batch() {
        let keys = vec!["good".to_string(), "bad".to_string()];
        let jsonl = format!(
            "{}\n{}",
            succeeded_line("r0", r#"{"sex":"female"}"#, 1, 1),
            succeeded_line("r1", "I think it was May.", 1, 1),
        );
        let outcome = parse_results(&jsonl, &keys);
        assert_eq!(outcome.results.len(), 1);
        assert!(outcome.results.contains_key("good"));
        assert!(matches!(outcome.failures["bad"], BatchFailure::MalformedJson(_)));
    }

    #[test]
    fn an_unparseable_line_does_not_abort_the_collection() {
        let keys = vec!["a".to_string()];
        let jsonl = format!("not json\n{}", succeeded_line("r0", "{}", 1, 1));
        let outcome = parse_results(&jsonl, &keys);
        assert_eq!(outcome.results.len(), 1);
        assert_eq!(outcome.failures.len(), 1);
    }

    #[test]
    fn a_custom_id_that_was_never_sent_is_reported() {
        // Would otherwise index past the key list, or silently attach a result
        // to the wrong record.
        let keys = vec!["only".to_string()];
        let jsonl = succeeded_line("r7", "{}", 1, 1);
        let outcome = parse_results(&jsonl, &keys);
        assert!(outcome.results.is_empty());
        assert!(matches!(outcome.failures["r7"], BatchFailure::UnknownId(_)));
    }

    #[test]
    fn a_result_type_from_the_future_is_reported_not_ignored() {
        let keys = vec!["a".to_string()];
        let jsonl = json!({"custom_id": "r0", "result": {"type": "deferred"}}).to_string();
        let outcome = parse_results(&jsonl, &keys);
        assert!(matches!(
            &outcome.failures["a"],
            BatchFailure::Errored { kind, .. } if kind.as_deref() == Some("unknown_result_type")
        ));
    }

    #[test]
    fn blank_lines_are_skipped() {
        let keys = vec!["a".to_string()];
        let jsonl = format!("\n{}\n\n", succeeded_line("r0", "{}", 1, 1));
        let outcome = parse_results(&jsonl, &keys);
        assert_eq!(outcome.results.len(), 1);
        assert!(outcome.failures.is_empty());
    }

    #[test]
    fn empty_results_are_an_empty_outcome_not_an_error() {
        let outcome = parse_results("", &["a".to_string()]);
        assert_eq!(outcome, BatchOutcome::default());
    }

    // -- pricing ------------------------------------------------------------

    #[test]
    fn a_batch_outcome_prices_at_half() {
        use super::super::call_cost;
        let keys = vec!["a".to_string()];
        let jsonl = succeeded_line("r0", "{}", 1_000_000, 1_000_000);
        let usage = parse_results(&jsonl, &keys).usage;
        let live = call_cost(usage, ModelId::Opus5, false);
        let batched = call_cost(usage, ModelId::Opus5, true);
        assert!((live - 30.0).abs() < 1e-9);
        assert!((batched - 15.0).abs() < 1e-9);
    }

    // -- the spend guard ----------------------------------------------------

    // Pointed at a closed port. The guard is supposed to return before any
    // request is attempted, so these never open a socket — and if the guard
    // ever stops working, the test fails against localhost rather than
    // reaching Anthropic.
    fn unroutable_client() -> Claude {
        Claude::new("sk-ant-0000000000000000", ModelId::Opus5)
            .unwrap()
            .with_base_url("http://127.0.0.1:1")
    }

    #[test]
    fn a_full_ledger_refuses_the_batch_before_submitting_it() {
        let budget = Budget::new(1.00);
        budget.record(1.50, Usage::default());
        let result = unroutable_client().run_batch(
            &requests(&["a", "b"]),
            BatchOptions::default(),
            &budget,
            |_| {},
        );
        match result {
            Err(ModelError::BudgetExceeded { spent, limit }) => {
                assert!((spent - 1.50).abs() < 1e-9);
                assert!((limit - 1.00).abs() < 1e-9);
            }
            other => panic!("expected BudgetExceeded before submission, got {other:?}"),
        }
    }

    #[test]
    fn an_estimate_that_would_cross_the_ceiling_is_refused() {
        // The reason the estimate exists. Without it the ledger is merely "not
        // yet full", and a batch commits everything at once — by the time it is
        // running there is nothing left to stop.
        let budget = Budget::new(1.00);
        budget.record(0.01, Usage::default());
        assert!(budget.check().is_ok(), "precondition: the ledger is not full");

        let options = BatchOptions { estimate: Some(50.0), ..Default::default() };
        assert!(matches!(
            unroutable_client().run_batch(&requests(&["a"]), options, &budget, |_| {}),
            Err(ModelError::BudgetExceeded { .. })
        ));
    }

    #[test]
    fn an_empty_batch_commits_nothing_even_on_a_full_ledger() {
        // Nothing is submitted, so there is nothing to refuse.
        let budget = Budget::new(1.00);
        budget.record(99.0, Usage::default());
        let outcome = unroutable_client()
            .run_batch(&BTreeMap::new(), BatchOptions::default(), &budget, |_| {})
            .unwrap();
        assert_eq!(outcome, BatchOutcome::default());
    }

    #[test]
    fn batch_usage_prices_at_half_the_live_rate() {
        // `Model::price` prices the live path; the same tokens bill at half
        // here, so the batch path needs its own.
        use crate::model::Model;
        let client = unroutable_client();
        let usage = Usage { input: 1_000_000, output: 1_000_000, ..Default::default() };
        assert!((client.price(usage) - 30.0).abs() < 1e-9);
        assert!((client.price_batch(usage) - 15.0).abs() < 1e-9);
    }

    #[test]
    fn the_default_options_match_the_apis_own_ceiling() {
        let options = BatchOptions::default();
        assert_eq!(options.poll, Duration::from_secs(10));
        assert_eq!(options.timeout, Duration::from_secs(86_400));
    }
}
