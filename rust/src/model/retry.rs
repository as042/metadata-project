use std::thread::sleep;
use std::time::Duration;

use crate::model::{Model, ModelError, Request, Response, Usage};

// Retries the failures that a second attempt could fix, and only those.
//
// A decorator rather than something baked into the Anthropic client: the
// policy is the same whichever provider is underneath, and a local model that
// occasionally drops a connection deserves it as much as a rate-limited API.
//
// What is *not* retried matters as much as what is. A refusal, a malformed
// schema reply, a 400 from a bad parameter combination and a spend ceiling are
// all deterministic — sending the identical request again changes nothing and
// bills again for the privilege. `ModelError::is_retryable` draws that line.
pub struct Retrying<M> {
    inner: M,
    policy: RetryPolicy,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RetryPolicy {
    // Total attempts, not retries after the first: 1 means never retry.
    pub attempts: u32,
    pub base_delay: Duration,
    // Also caps a server-supplied `retry-after`. Waiting an unbounded time
    // because a header said so is how an unattended run hangs instead of
    // failing.
    pub max_delay: Duration,
}

impl Default for RetryPolicy {
    // Five attempts with exponential backoff, matching the harvest client that
    // has run against NCBI for the life of this project: over a run of hundreds
    // of studies a transient failure is near-certain, and it should not end the
    // run.
    fn default() -> Self {
        Self {
            attempts: 5,
            base_delay: Duration::from_millis(500),
            max_delay: Duration::from_secs(60),
        }
    }
}

impl RetryPolicy {
    // How long to wait before attempt `attempt` (0-based, so 0 is the wait
    // after the first failure).
    //
    // A server that says `retry-after` knows better than any backoff curve, so
    // that wins where present — capped, because the cap is what keeps a run
    // failing rather than hanging.
    pub fn delay(&self, attempt: u32, error: &ModelError) -> Duration {
        if let ModelError::RateLimited { retry_after: Some(secs) } = error {
            return Duration::from_secs(*secs).min(self.max_delay);
        }
        let factor = 2u32.saturating_pow(attempt);
        self.base_delay
            .saturating_mul(factor)
            .min(self.max_delay)
    }
}

impl<M> Retrying<M> {
    #[inline]
    pub fn new(inner: M, policy: RetryPolicy) -> Self {
        Self { inner, policy }
    }

    #[inline]
    pub fn policy(&self) -> RetryPolicy {
        self.policy
    }
}

impl<M: Model> Model for Retrying<M> {
    #[inline]
    fn price(&self, usage: Usage) -> f64 {
        self.inner.price(usage)
    }

    #[inline]
    fn supports_batch(&self) -> bool {
        self.inner.supports_batch()
    }

    #[inline]
    fn price_many(&self, usage: Usage) -> f64 {
        self.inner.price_many(usage)
    }

    // Forwarded without retrying. A batch is hours of work and its per-key
    // failures are already reported individually; resubmitting the whole thing
    // would re-bill every request that succeeded.
    #[inline]
    fn complete_many(
        &self,
        requests: &std::collections::BTreeMap<String, Request>,
    ) -> std::collections::BTreeMap<String, Result<Response, ModelError>> {
        self.inner.complete_many(requests)
    }

    fn complete(&self, request: &Request) -> Result<Response, ModelError> {
        let attempts = self.policy.attempts.max(1);
        let mut last = None;

        for attempt in 0..attempts {
            match self.inner.complete(request) {
                Ok(response) => return Ok(response),
                Err(error) => {
                    // Returned as itself rather than wrapped: a refusal should
                    // read as a refusal, not as "retries exhausted" on a
                    // request that was never going to be retried.
                    if !error.is_retryable() {
                        return Err(error);
                    }
                    if attempt + 1 < attempts {
                        sleep(self.policy.delay(attempt, &error));
                    }
                    last = Some(error);
                }
            }
        }

        Err(ModelError::RetriesExhausted {
            attempts,
            last: Box::new(last.expect("a failed loop records its last error")),
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;

    // Replays a script of outcomes. No network, and the policies below use zero
    // delays so the suite does not sleep.
    struct Script {
        outcomes: Mutex<Vec<Result<(), ModelError>>>,
        calls: Mutex<u32>,
    }

    impl Script {
        fn new(outcomes: Vec<Result<(), ModelError>>) -> Self {
            // reversed so `pop` walks it forwards
            Self {
                outcomes: Mutex::new(outcomes.into_iter().rev().collect()),
                calls: Mutex::new(0),
            }
        }
        fn calls(&self) -> u32 {
            *self.calls.lock().unwrap()
        }
    }

    impl Model for Script {
        fn price(&self, _usage: Usage) -> f64 {
            0.0
        }
        fn complete(&self, _request: &Request) -> Result<Response, ModelError> {
            *self.calls.lock().unwrap() += 1;
            match self.outcomes.lock().unwrap().pop() {
                Some(Ok(())) => Ok(Response {
                    text: "ok".into(),
                    json: None,
                    stop_reason: None,
                    usage: Usage::default(),
                }),
                Some(Err(e)) => Err(e),
                None => panic!("the script ran out: more calls than outcomes"),
            }
        }
    }

    fn instant(attempts: u32) -> RetryPolicy {
        RetryPolicy {
            attempts,
            base_delay: Duration::ZERO,
            max_delay: Duration::ZERO,
        }
    }

    #[test]
    fn a_transient_failure_is_retried_until_it_succeeds() {
        let script = Script::new(vec![
            Err(ModelError::Overloaded),
            Err(ModelError::RateLimited { retry_after: None }),
            Ok(()),
        ]);
        let model = Retrying::new(script, instant(5));
        assert!(model.complete(&Request::new("p")).is_ok());
        assert_eq!(model.inner.calls(), 3);
    }

    #[test]
    fn a_deterministic_failure_is_not_retried_at_all() {
        // Sending the identical request again cannot change a refusal, and on a
        // real provider it bills again for finding that out.
        for error in [
            ModelError::Refused { category: None, explanation: None },
            ModelError::MalformedJson("nope".into()),
            ModelError::Api { status: 400, kind: None, message: String::new() },
            ModelError::MissingApiKey,
            ModelError::BudgetExceeded { spent: 2.0, limit: 1.0 },
        ] {
            let script = Script::new(vec![Err(error)]);
            let model = Retrying::new(script, instant(5));
            assert!(model.complete(&Request::new("p")).is_err());
            assert_eq!(model.inner.calls(), 1, "must not have been retried");
        }
    }

    #[test]
    fn the_original_error_survives_when_it_is_not_retryable() {
        // Wrapping a refusal in RetriesExhausted would misreport it.
        let script = Script::new(vec![Err(ModelError::Refused {
            category: Some("harmful_content".into()),
            explanation: None,
        })]);
        let model = Retrying::new(script, instant(5));
        assert!(matches!(
            model.complete(&Request::new("p")),
            Err(ModelError::Refused { .. })
        ));
    }

    #[test]
    fn exhausting_the_attempts_reports_the_last_cause() {
        // "gave up after 5 attempts" without saying at what is not a diagnosis.
        let script = Script::new(vec![
            Err(ModelError::Overloaded),
            Err(ModelError::Overloaded),
            Err(ModelError::RateLimited { retry_after: Some(1) }),
        ]);
        let model = Retrying::new(script, instant(3));
        match model.complete(&Request::new("p")) {
            Err(ModelError::RetriesExhausted { attempts, last }) => {
                assert_eq!(attempts, 3);
                assert!(matches!(*last, ModelError::RateLimited { .. }));
            }
            other => panic!("expected RetriesExhausted, got {other:?}"),
        }
        assert_eq!(model.inner.calls(), 3);
    }

    #[test]
    fn one_attempt_means_no_retry() {
        let script = Script::new(vec![Err(ModelError::Overloaded)]);
        let model = Retrying::new(script, instant(1));
        assert!(model.complete(&Request::new("p")).is_err());
        assert_eq!(model.inner.calls(), 1);
    }

    #[test]
    fn zero_attempts_is_treated_as_one_rather_than_as_never_calling() {
        // A policy of 0 should be a misconfiguration that still does the work,
        // not a model that silently answers nothing.
        let script = Script::new(vec![Ok(())]);
        let model = Retrying::new(script, instant(0));
        assert!(model.complete(&Request::new("p")).is_ok());
        assert_eq!(model.inner.calls(), 1);
    }

    #[test]
    fn a_success_first_time_makes_no_extra_calls() {
        let script = Script::new(vec![Ok(())]);
        let model = Retrying::new(script, instant(5));
        assert!(model.complete(&Request::new("p")).is_ok());
        assert_eq!(model.inner.calls(), 1);
    }

    // -- backoff ------------------------------------------------------------

    #[test]
    fn backoff_doubles_and_then_stops_at_the_cap() {
        let policy = RetryPolicy {
            attempts: 10,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_millis(500),
        };
        let e = ModelError::Overloaded;
        assert_eq!(policy.delay(0, &e), Duration::from_millis(100));
        assert_eq!(policy.delay(1, &e), Duration::from_millis(200));
        assert_eq!(policy.delay(2, &e), Duration::from_millis(400));
        assert_eq!(policy.delay(3, &e), Duration::from_millis(500)); // capped
        assert_eq!(policy.delay(40, &e), Duration::from_millis(500)); // no overflow
    }

    #[test]
    fn a_server_supplied_retry_after_wins_over_the_curve() {
        let policy = RetryPolicy {
            attempts: 5,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(60),
        };
        let told = ModelError::RateLimited { retry_after: Some(30) };
        assert_eq!(policy.delay(0, &told), Duration::from_secs(30));
        // and without one, the curve applies
        let untold = ModelError::RateLimited { retry_after: None };
        assert_eq!(policy.delay(0, &untold), Duration::from_millis(100));
    }

    #[test]
    fn retry_after_is_capped_so_a_run_fails_rather_than_hangs() {
        let policy = RetryPolicy {
            attempts: 5,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(60),
        };
        let e = ModelError::RateLimited { retry_after: Some(86_400) };
        assert_eq!(policy.delay(0, &e), Duration::from_secs(60));
    }

    #[test]
    fn the_default_policy_matches_the_harvest_client() {
        let policy = RetryPolicy::default();
        assert_eq!(policy.attempts, 5);
        assert_eq!(policy.base_delay, Duration::from_millis(500));
        assert_eq!(policy.max_delay, Duration::from_secs(60));
    }

    // -- composition --------------------------------------------------------

    #[test]
    fn retries_inside_a_budget_are_not_charged_as_separate_calls() {
        // The recommended arrangement, Budgeted<Retrying<M>>: the budget meters
        // one logical call, and the retries beneath it are invisible to the
        // ledger because a failed attempt returns no usage to bill.
        use crate::model::budget::{Budget, Budgeted};
        use std::sync::Arc;

        let script = Script::new(vec![
            Err(ModelError::Overloaded),
            Err(ModelError::Overloaded),
            Ok(()),
        ]);
        let budget = Arc::new(Budget::new(10.0));
        let model = Budgeted::new(Retrying::new(script, instant(5)), Arc::clone(&budget));

        assert!(model.complete(&Request::new("p")).is_ok());
        assert_eq!(budget.ledger().calls, 1, "three attempts, one billed call");
    }
}
