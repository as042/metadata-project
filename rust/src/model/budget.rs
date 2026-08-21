use std::sync::Mutex;

use crate::model::{Model, ModelError, Request, Response, Usage};

// ModelError is not Clone — it carries a boxed cause — and a refused batch has
// to hand the same refusal to every key.
fn clone_budget_error(error: &ModelError) -> ModelError {
    match error {
        ModelError::BudgetExceeded { spent, limit } => ModelError::BudgetExceeded {
            spent: *spent,
            limit: *limit,
        },
        other => ModelError::Api {
            status: 0,
            kind: Some("budget_check_failed".into()),
            message: other.to_string(),
        },
    }
}

// A running ceiling on what a run may bill.
//
// Python checks an *estimate* once, before any paid work, and refuses if it
// exceeds `max_spend`. That is worth having and is not what this is. An
// estimate can be wrong, and when it was — a run whose thinking parameter was
// not doing what the estimate assumed billed $1.25 against a $0.49 estimate —
// the pre-flight check waved it through, because the check had already passed
// before the first token was billed.
//
// This meters what has *actually* been billed and refuses the next call once
// the ceiling is reached. It cannot un-spend the call that crossed the line,
// so the ledger can end slightly above the limit; what it guarantees is that a
// run cannot keep going after that, which is the failure that mattered.
//
// Wrap a model with `Budgeted`, or share the `Budget` directly with the batch
// path, which commits far more in a single call than any one message does.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct Ledger {
    pub spent: f64,
    pub calls: u64,
    pub usage: Usage,
}

#[derive(Debug)]
pub struct Budget {
    // None disables the ceiling without disabling the accounting: a run that
    // wants no limit still wants to know what it spent.
    limit: Option<f64>,
    ledger: Mutex<Ledger>,
}

impl Budget {
    #[inline]
    pub fn new(limit: f64) -> Self {
        Self { limit: Some(limit), ledger: Mutex::new(Ledger::default()) }
    }

    // Meters but never refuses.
    #[inline]
    pub fn unlimited() -> Self {
        Self { limit: None, ledger: Mutex::new(Ledger::default()) }
    }

    #[inline]
    pub fn limit(&self) -> Option<f64> {
        self.limit
    }

    #[inline]
    pub fn ledger(&self) -> Ledger {
        *self.ledger.lock().expect("budget ledger poisoned")
    }

    #[inline]
    pub fn spent(&self) -> f64 {
        self.ledger().spent
    }

    #[inline]
    pub fn remaining(&self) -> Option<f64> {
        self.limit.map(|limit| (limit - self.spent()).max(0.0))
    }

    // Refuse if the ceiling has already been reached. Called before a request
    // is sent, so a run that is out of budget spends nothing further.
    pub fn check(&self) -> Result<(), ModelError> {
        let Some(limit) = self.limit else { return Ok(()) };
        let spent = self.spent();
        if spent >= limit {
            return Err(ModelError::BudgetExceeded { spent, limit });
        }
        Ok(())
    }

    // Refuse if a known-size commitment would cross the ceiling.
    //
    // `check` is too weak for a batch: a single submission can commit thousands
    // of requests at once, and asking only whether the ledger is already full
    // would let a $50 batch through on a $1 budget with $0.99 remaining.
    pub fn reserve(&self, estimate: f64) -> Result<(), ModelError> {
        let Some(limit) = self.limit else { return Ok(()) };
        let spent = self.spent();
        if spent + estimate > limit {
            return Err(ModelError::BudgetExceeded { spent, limit });
        }
        Ok(())
    }

    pub fn record(&self, dollars: f64, usage: Usage) {
        let mut ledger = self.ledger.lock().expect("budget ledger poisoned");
        ledger.spent += dollars;
        ledger.calls += 1;
        ledger.usage.add(usage);
    }
}

// Meters every call made through it and stops once the ceiling is reached.
pub struct Budgeted<M> {
    inner: M,
    budget: std::sync::Arc<Budget>,
}

impl<M> Budgeted<M> {
    #[inline]
    pub fn new(inner: M, budget: std::sync::Arc<Budget>) -> Self {
        Self { inner, budget }
    }

    #[inline]
    pub fn budget(&self) -> &std::sync::Arc<Budget> {
        &self.budget
    }
}

impl<M: Model> Model for Budgeted<M> {
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

    // Checked once, not per key: a batch is submitted whole, so there is no
    // moment between requests at which to stop. Recorded at the batch rate,
    // because that is what the provider charged.
    fn complete_many(
        &self,
        requests: &std::collections::BTreeMap<String, Request>,
    ) -> std::collections::BTreeMap<String, Result<Response, ModelError>> {
        if let Err(error) = self.budget.check() {
            return requests
                .keys()
                .map(|key| (key.clone(), Err(clone_budget_error(&error))))
                .collect();
        }
        let results = self.inner.complete_many(requests);
        for response in results.values().flatten() {
            self.budget
                .record(self.price_many(response.usage), response.usage);
        }
        results
    }

    fn complete(&self, request: &Request) -> Result<Response, ModelError> {
        self.budget.check()?;
        match self.inner.complete(request) {
            Ok(response) => {
                // Recorded from the response's own usage rather than an
                // estimate, so the ledger is the billed figure and not a
                // second guess at it.
                self.budget.record(self.price(response.usage), response.usage);
                Ok(response)
            }
            Err(error) => {
                // Failures are billed too. A refusal, a malformed reply and a
                // clipped one all generated tokens the provider charged for,
                // and a ledger that only counts successes is not a spend
                // ceiling. This was measured: a run reported $0.05 against
                // roughly $1.00 actually spent, because five failed paper calls
                // never reached `record`.
                if let Some(usage) = error.billed_usage() {
                    self.budget.record(self.price(usage), usage);
                }
                Err(error)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    // A model whose every call fails *after* the provider has billed it.
    struct BilledFailure {
        usage: Usage,
        calls: Mutex<u32>,
    }

    impl Model for BilledFailure {
        fn price(&self, usage: Usage) -> f64 {
            usage.output as f64 / 1000.0
        }
        fn complete(&self, _request: &Request) -> Result<Response, ModelError> {
            *self.calls.lock().unwrap() += 1;
            Err(ModelError::Truncated {
                detail: "clipped".into(),
                usage: self.usage,
            })
        }
    }

    // A model that bills a fixed amount per call and never touches a network.
    struct Meter {
        dollars_per_call: f64,
        calls: Mutex<u32>,
    }

    impl Meter {
        fn new(dollars_per_call: f64) -> Self {
            Self { dollars_per_call, calls: Mutex::new(0) }
        }
        fn calls(&self) -> u32 {
            *self.calls.lock().unwrap()
        }
    }

    impl Model for Meter {
        fn price(&self, _usage: Usage) -> f64 {
            self.dollars_per_call
        }
        fn complete(&self, _request: &Request) -> Result<Response, ModelError> {
            *self.calls.lock().unwrap() += 1;
            Ok(Response {
                text: "ok".into(),
                json: None,
                stop_reason: None,
                usage: Usage { input: 10, output: 1, ..Default::default() },
            })
        }
    }

    fn budgeted(per_call: f64, limit: f64) -> (Budgeted<Meter>, Arc<Budget>) {
        let budget = Arc::new(Budget::new(limit));
        (Budgeted::new(Meter::new(per_call), Arc::clone(&budget)), budget)
    }

    #[test]
    fn calls_are_metered_as_they_are_made() {
        let (model, budget) = budgeted(0.25, 10.0);
        for _ in 0..3 {
            model.complete(&Request::new("p")).unwrap();
        }
        let ledger = budget.ledger();
        assert_eq!(ledger.calls, 3);
        assert!((ledger.spent - 0.75).abs() < 1e-9);
        assert_eq!(ledger.usage.input, 30);
        assert_eq!(ledger.usage.output, 3);
    }

    #[test]
    fn the_ceiling_stops_the_run_rather_than_the_call_that_crossed_it() {
        // The honest guarantee: the ledger can finish slightly over, because a
        // call's cost is only known once it has been billed. What cannot happen
        // is the run continuing afterwards.
        let (model, budget) = budgeted(0.60, 1.00);
        assert!(model.complete(&Request::new("p")).is_ok()); // 0.60
        assert!(model.complete(&Request::new("p")).is_ok()); // 1.20 — over
        match model.complete(&Request::new("p")) {
            Err(ModelError::BudgetExceeded { spent, limit }) => {
                assert!((spent - 1.20).abs() < 1e-9);
                assert!((limit - 1.00).abs() < 1e-9);
            }
            other => panic!("expected BudgetExceeded, got {other:?}"),
        }
        assert_eq!(budget.ledger().calls, 2, "the refused call must not be counted");
    }

    #[test]
    fn a_refused_call_never_reaches_the_model() {
        // The whole point: nothing is sent, so nothing more is billed.
        let budget = Arc::new(Budget::new(0.10));
        budget.record(1.00, Usage::default());
        let inner = Meter::new(0.25);
        let model = Budgeted::new(inner, Arc::clone(&budget));
        assert!(model.complete(&Request::new("p")).is_err());
        assert_eq!(model.inner.calls(), 0);
    }

    #[test]
    fn an_unlimited_budget_still_accounts() {
        // Disabling the ceiling must not disable the reporting — a run that
        // wants no limit still wants to know what it spent.
        let budget = Arc::new(Budget::unlimited());
        let model = Budgeted::new(Meter::new(5.0), Arc::clone(&budget));
        for _ in 0..4 {
            model.complete(&Request::new("p")).unwrap();
        }
        assert_eq!(budget.limit(), None);
        assert_eq!(budget.remaining(), None);
        assert!((budget.spent() - 20.0).abs() < 1e-9);
        assert!(budget.check().is_ok());
    }

    #[test]
    fn reserve_refuses_a_commitment_that_would_cross_the_ceiling() {
        // A batch commits everything at once. `check` alone would wave through
        // a $50 batch on a $1 budget with $0.99 left.
        let budget = Budget::new(1.00);
        budget.record(0.01, Usage::default());
        assert!(budget.check().is_ok(), "the ledger is not full, so check passes");
        assert!(budget.reserve(0.50).is_ok());
        assert!(matches!(
            budget.reserve(50.0),
            Err(ModelError::BudgetExceeded { .. })
        ));
    }

    #[test]
    fn reserve_is_a_no_op_without_a_ceiling() {
        assert!(Budget::unlimited().reserve(1_000_000.0).is_ok());
    }

    #[test]
    fn remaining_never_goes_negative() {
        let budget = Budget::new(1.00);
        budget.record(2.50, Usage::default());
        assert_eq!(budget.remaining(), Some(0.0));
    }

    #[test]
    fn the_error_says_what_was_billed_and_what_to_do() {
        let message = ModelError::BudgetExceeded { spent: 1.2345, limit: 1.0 }.to_string();
        assert!(message.contains("1.234"));
        assert!(message.contains("Raise the ceiling"));
        // and it is not something to retry
        assert!(!ModelError::BudgetExceeded { spent: 1.0, limit: 1.0 }.is_retryable());
    }

    #[test]
    fn a_budget_is_shareable_across_threads() {
        // `Model` is Send + Sync so a caller can fan out over records; the
        // ledger has to survive that.
        let budget = Arc::new(Budget::new(1000.0));
        let model = Budgeted::new(Meter::new(1.0), Arc::clone(&budget));
        std::thread::scope(|scope| {
            for _ in 0..8 {
                scope.spawn(|| {
                    for _ in 0..10 {
                        let _ = model.complete(&Request::new("p"));
                    }
                });
            }
        });
        assert_eq!(budget.ledger().calls, 80);
        assert!((budget.spent() - 80.0).abs() < 1e-9);
    }

    #[test]
    fn a_call_that_failed_after_being_billed_still_reaches_the_ledger() {
        // Measured the hard way: a run reported $0.05 against roughly $1.00
        // actually spent, because five paper calls that clipped at max_tokens
        // returned early and never reached `record`. A ceiling that only counts
        // successes is not a ceiling.
        let usage = Usage { output: 16_000, ..Usage::default() };
        let inner = BilledFailure { usage, calls: Mutex::new(0) };
        let budget = Arc::new(Budget::new(100.0));
        let model = Budgeted::new(inner, Arc::clone(&budget));

        assert!(model.complete(&Request::new("p")).is_err());

        let ledger = budget.ledger();
        assert_eq!(ledger.calls, 1, "the failed call must be counted");
        assert_eq!(ledger.usage.output, 16_000);
        assert!((ledger.spent - 16.0).abs() < 1e-9, "spent {}", ledger.spent);
    }

    #[test]
    fn a_failure_the_provider_did_not_bill_is_not_charged() {
        // The complement: a refused *connection* generated no tokens, so
        // recording one would invent spend.
        struct Unbilled;
        impl Model for Unbilled {
            fn price(&self, _u: Usage) -> f64 { 1.0 }
            fn complete(&self, _r: &Request) -> Result<Response, ModelError> {
                Err(ModelError::Transport("connection reset".into()))
            }
        }
        let budget = Arc::new(Budget::new(100.0));
        let model = Budgeted::new(Unbilled, Arc::clone(&budget));
        assert!(model.complete(&Request::new("p")).is_err());
        assert_eq!(budget.ledger().calls, 0);
        assert_eq!(budget.spent(), 0.0);
    }
}
