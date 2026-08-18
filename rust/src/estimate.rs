use std::fmt;

use crate::corpus::Corpus;
use crate::model::budget::Budget;
use crate::model::{Model, Thinking, Usage};
use crate::target_schema::{SchemaSettings, TargetSchema};

// What a run would cost, worked out before anything is sent.
//
// The other half of the spend guard. `Budget` is an in-the-moment stop: it
// knows what a call cost only once the call has been paid for, so the cheapest
// mistake it can catch has already cost one call — and on a study planning
// 2,150 of them, "one call" is not the unit that hurts. This runs first, over
// the same plan the run will execute, and answers the question a ledger cannot:
// *should this begin at all*.
//
// Nothing here sends anything or reads a key. The layers' `plan` functions are
// pure over the records, which is what makes the estimate the real workload
// rather than a model of it.

// How the token counts are arrived at.
//
// Measured, not assumed. Run 20260818T003942Z sent 16,652 characters of
// instructions as its cached prefix and was billed 6,148 tokens for them —
// 2.708 characters per token. The folklore figure of four would have
// under-counted that prefix by a third, in the direction that matters least for
// a guard whose job is to prevent a surprise.
pub const CHARS_PER_TOKEN: f64 = 2.7;

// The answer schema goes over on every call and is not in the evidence. Its
// size tracks the field list: on that same run the uncounted remainder was
// 2,999 tokens across 30 calls carrying 660 field slots.
const SCHEMA_TOKENS_PER_FIELD: u64 = 4;
const SCHEMA_TOKENS_FIXED: u64 = 16;

// Output tokens per field *asked*. An upper bound, not a fit — the one number
// here that is deliberately loose, and in one direction.
//
// Output actually tracks fields *answered*: an asked field the model declines
// costs nothing to emit. Per answer it is stable (37 and 29 tokens across the
// two measured runs); per ask it is not, because the answer rate is a property
// of the study. A honey bee gut study answered 43% of what it was asked and
// billed 16.0 tokens per ask; five human and C. elegans studies answered 30%
// and billed 8.5. Nothing known before the run separates those two cases.
//
// So this is set at the high end. A guard may be wrong; it may not be wrong
// downwards, and the previous version's apparent accuracy came from this error
// cancelling an equal and opposite one in the cache model rather than from
// either being right.
const OUTPUT_TOKENS_PER_FIELD: u64 = 16;

// How many requests in one batch before its fan-out races an extra copy of the
// instructions into the cache.
//
// The prefix is written once and read back thereafter — unless several workers
// start before the first write lands, and then it is written again. Measured:
// a batch of 30 wrote twice (12,296 tokens) on one run and once (6,148) on
// another, so the race is real and nondeterministic; batches of 22 and 15 raced
// and batches of 7, 6, 2 and 1 did not. Ten is the dividing line those five
// groups imply, and assuming the race happens is the conservative reading of a
// coin flip.
const RACE_THRESHOLD: usize = 10;

// What adaptive thinking spends when the model decides to use it.
//
// The one number here with no cross-model measurement behind it. Haiku 4.5 on
// adaptive averaged 4,193 thinking tokens per call over eight runs; Sonnet 5
// was never run adaptive with an output count worth trusting. Estimates for an
// adaptive layer are therefore the roughest ones this produces, which is itself
// a reason to send `Thinking::Disabled` when the run is being budgeted.
const ADAPTIVE_THINKING_TOKENS: u64 = 4_200;

fn tokens(chars: usize) -> u64 {
    (chars as f64 / CHARS_PER_TOKEN).ceil() as u64
}

// One model layer's projected bill.
#[derive(Clone, Debug, PartialEq)]
pub struct Estimate {
    pub layer: &'static str,
    pub model: String,
    pub batch: bool,
    pub calls: usize,
    // How the calls are sent, which is what decides the cache bill. A layer
    // batches once per project, so a five-study run is five fan-outs and pays
    // for the instructions five times over, not once — the single biggest error
    // in the first version of this model.
    pub groups: usize,
    pub wide_groups: usize,
    pub usage: Usage,
    pub cost: f64,
}

// Every model layer in a run, plus the ceiling it will be checked against.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Estimates {
    pub layers: Vec<Estimate>,
    pub records: usize,
    pub studies: usize,
    pub max_spend: Option<f64>,
}

impl Estimates {
    pub fn cost(&self) -> f64 {
        self.layers.iter().map(|l| l.cost).sum()
    }

    pub fn calls(&self) -> usize {
        self.layers.iter().map(|l| l.calls).sum()
    }

    // Whether the run is expected to hit the in-the-moment guard. Worth saying
    // out loud: a run that trips `max_spend` stops mid-layer, and a half-done
    // layer is a paid result nobody can compare against anything.
    pub fn exceeds_ceiling(&self) -> bool {
        self.max_spend.is_some_and(|limit| self.cost() > limit)
    }
}

impl fmt::Display for Estimates {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "estimated spend over {} record{} in {} stud{}:",
            self.records,
            if self.records == 1 { "" } else { "s" },
            self.studies,
            if self.studies == 1 { "y" } else { "ies" },
        )?;
        for layer in &self.layers {
            writeln!(
                f,
                "  {:<10} {:<18} {:>6} call{}{}  in {:>7} cached {:>7}+{:<8} out {:>7}  ${:.4}",
                layer.layer,
                layer.model,
                layer.calls,
                if layer.calls == 1 { " " } else { "s" },
                if layer.batch { " batched" } else { "        " },
                layer.usage.input,
                layer.usage.cache_write,
                layer.usage.cache_read,
                layer.usage.output,
                layer.cost,
            )?;
        }
        write!(f, "  {:<10} {:>62} ${:.4}", "TOTAL", "", self.cost())?;
        if let Some(limit) = self.max_spend {
            write!(f, "  (max_spend ${limit:.4})")?;
            if self.exceeds_ceiling() {
                write!(f, " — the run is expected to stop part-way")?;
            }
        }
        writeln!(f)
    }
}

// The bill for one layer over one project, given what is still open.
//
// Called from `Layer::estimate`, which is where the dispatch lives so the plan
// used here cannot drift from the plan the run executes.
pub(crate) fn of_jobs(
    layer: &'static str,
    jobs: &[crate::layer::llm_naive::Job],
    model: &dyn Model,
    config: &crate::layer::ModelConfig,
) -> Estimate {
    let mut usage = Usage::default();
    for job in jobs {
        usage.input += tokens(job.evidence.len())
            + SCHEMA_TOKENS_FIXED
            + SCHEMA_TOKENS_PER_FIELD * job.wanted.len() as u64;
        usage.output += OUTPUT_TOKENS_PER_FIELD * job.wanted.len() as u64;
        usage.thinking += match config.thinking {
            Thinking::Adaptive => ADAPTIVE_THINKING_TOKENS,
            Thinking::Enabled { budget_tokens } => budget_tokens as u64,
            Thinking::Disabled | Thinking::Unset => 0,
        };
    }
    // Thinking is billed as output and counted inside it, which is how the
    // saved runs report it — 9,205 output tokens of which 8,427 were thinking.
    usage.output += usage.thinking;

    Estimate {
        layer,
        model: config.label.clone(),
        batch: config.batch,
        calls: jobs.len(),
        // One project, one fan-out.
        groups: usize::from(!jobs.is_empty()),
        wide_groups: usize::from(jobs.len() >= RACE_THRESHOLD),
        usage,
        cost: 0.0,
    }
    .with_cache(config.prompt.len(), config.batch && model.supports_batch())
    .priced(model)
}

impl Estimate {
    // The instructions are sent once per fan-out and read back on every other
    // call, which is the whole reason the prompt is a constant.
    //
    // Sequential calls share one warm cache entry for the whole run, so they pay
    // for the instructions once. Batched calls do not: a layer submits one batch
    // per project, and each batch is a cold start that writes the prefix again —
    // plus once more when the batch is wide enough to race (see RACE_THRESHOLD).
    // Modelling this as a property of the *run* rather than of the batches
    // under-counted a five-study run by 2.7×.
    //
    // The sequential case assumes the run outpaces the cache's five-minute life.
    // A slow enough run re-writes the prefix and this under-counts; no measured
    // run has been slow enough for that to show.
    fn with_cache(mut self, prompt_chars: usize, batch: bool) -> Self {
        if self.calls == 0 {
            return self;
        }
        let prefix = tokens(prompt_chars);
        let writes = if batch {
            (self.groups + self.wide_groups) as u64
        } else {
            1
        }
        .clamp(1, self.calls as u64);
        self.usage.cache_write = prefix * writes;
        self.usage.cache_read = prefix * (self.calls as u64 - writes);
        self
    }

    // Priced by the model itself, so a provider with different rates or a
    // different batch discount needs no change here.
    fn priced(mut self, model: &dyn Model) -> Self {
        self.cost = if self.batch && model.supports_batch() {
            model.price_many(self.usage)
        } else {
            model.price(self.usage)
        };
        self
    }
}

// What the configured run would cost, without running it.
//
// Walks the same projects, under the same caps, in the same layer order. Free
// layers really run — what they settle is what the paid layers will not be
// asked, and an estimate that skipped them would be wrong in the expensive
// direction by roughly a thousand field slots per study.
//
// Paid layers are planned but not applied, so a second paid layer sees every
// field the first one would have filled as still open. That over-counts, never
// under-counts, which is the right way for a guard to be wrong.
//
// The free layers therefore run twice per run — once here and once for real.
// That is seconds on a capped run and a few seconds on the whole corpus, paid
// to avoid finding out what a layer costs by being charged for it.
pub fn for_corpus(corpus: &Corpus, settings: &SchemaSettings, budget: &Budget) -> Estimates {
    // Its own settings: the caller's evidence store and issue sink belong to the
    // real run, and a dry pass writing into them would leave a run reporting
    // evidence for calls that never happened.
    let scratch = SchemaSettings::default();

    let mut out = Estimates {
        max_spend: budget.limit(),
        ..Default::default()
    };
    let mut totals: Vec<Option<Estimate>> = vec![None; settings.layers().len()];
    let studies = settings.study_limit().unwrap_or(usize::MAX);

    // Same predicate as the run, for the same reason the caps are honoured here:
    // an estimate over studies that are not the ones about to run is not a guard.
    for project in corpus
        .projects
        .iter()
        .filter(|p| settings.selects(p))
        .take(studies)
    {
        let mut project = project.clone();
        if let Some(limit) = settings.record_limit() {
            let room = limit.saturating_sub(out.records);
            if room == 0 {
                break;
            }
            if project.experiments.len() > room {
                project.experiments.truncate(room);
            }
        }

        let mut schemas: Vec<TargetSchema> = Vec::new();
        for (index, layer) in settings.layers().iter().enumerate() {
            match layer.estimate(&project, &schemas) {
                Some(estimate) => merge(&mut totals[index], estimate),
                None => layer.process(&project, &mut schemas, &scratch),
            }
        }
        if let Some(limit) = settings.record_limit() {
            schemas.truncate(limit.saturating_sub(out.records));
        }

        out.records += schemas.len();
        out.studies += 1;
    }

    // Re-cached and re-priced once the call count is final: the prefix is
    // written once for the whole run, not once per study.
    out.layers = totals
        .into_iter()
        .flatten()
        .zip(settings.layers().iter().filter(|l| l.is_paid()))
        .map(|(estimate, layer)| {
            let config = layer.config().expect("a paid layer carries a config");
            let model = layer.model().expect("a paid layer carries a model");
            estimate
                .with_cache(config.prompt.len(), config.batch && model.supports_batch())
                .priced(model)
        })
        .collect();
    out
}

// Sums one layer's per-project estimates. Cache figures are left alone here and
// recomputed at the end, because they are a property of the run, not the study.
fn merge(slot: &mut Option<Estimate>, next: Estimate) {
    match slot {
        Some(total) => {
            total.calls += next.calls;
            total.groups += next.groups;
            total.wide_groups += next.wide_groups;
            total.usage.input += next.usage.input;
            total.usage.output += next.usage.output;
            total.usage.thinking += next.usage.thinking;
        }
        None => *slot = Some(next),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layer::Layer;
    use crate::layer::{llm_naive::TEXT_SYSTEM_FULL, ModelConfig};
    use crate::model::claude::{call_cost, ModelId};
    use crate::model::{ModelError, Request, Response};

    // Prices like the real client without holding a key, so an estimate can be
    // checked against the recorded dollar figures.
    struct Priced(ModelId);
    impl Model for Priced {
        fn complete(&self, _r: &Request) -> Result<Response, ModelError> {
            unreachable!("an estimate sends nothing")
        }
        fn price(&self, usage: Usage) -> f64 {
            call_cost(usage, self.0, false)
        }
        fn price_many(&self, usage: Usage) -> f64 {
            call_cost(usage, self.0, true)
        }
        fn supports_batch(&self) -> bool {
            true
        }
    }

    fn jobs(count: usize, evidence_chars: usize, wanted: usize) -> Vec<crate::layer::llm_naive::Job> {
        (0..count)
            .map(|_| crate::layer::llm_naive::Job {
                key: crate::layer::llm_naive::JobKey::Study,
                evidence: "x".repeat(evidence_chars),
                wanted: vec!["host"; wanted],
                targets: vec![0],
            })
            .collect()
    }

    // -- calibration ---------------------------------------------------------

    // The two 30-record Sonnet runs, replanned. Their evidence totals 23,454
    // characters over 30 calls carrying 660 field slots — reproduced exactly by
    // `llm_naive::plan`, which is what makes this a check of the token model
    // rather than of the planner.
    fn measured_layer3(batch: bool) -> Estimate {
        of_jobs(
            "llm_naive",
            &jobs(30, 23_454 / 30, 22),
            &Priced(ModelId::Sonnet5),
            &ModelConfig::new("claude-sonnet-5", TEXT_SYSTEM_FULL)
                .thinking(Thinking::Disabled)
                .batch(batch),
        )
    }

    #[test]
    fn the_cached_prefix_matches_what_the_api_billed() {
        // 16,652 characters of instructions were billed as 6,148 tokens.
        let estimate = measured_layer3(false);
        let off = (estimate.usage.cache_write as f64 - 6_148.0).abs() / 6_148.0;
        assert!(off < 0.02, "cache_write {} vs 6148", estimate.usage.cache_write);
        assert_eq!(estimate.usage.cache_read, estimate.usage.cache_write * 29);
    }

    #[test]
    fn the_input_estimate_matches_what_the_api_billed() {
        // Recorded: 11,660 uncached input tokens.
        let estimate = measured_layer3(false);
        let off = (estimate.usage.input as f64 - 11_660.0).abs() / 11_660.0;
        assert!(off < 0.10, "input {} vs 11660", estimate.usage.input);
    }

    #[test]
    fn the_output_estimate_matches_what_the_api_billed() {
        // Recorded: 10,256 and 10,571 output tokens on the two runs.
        let estimate = measured_layer3(false);
        let off = (estimate.usage.output as f64 - 10_400.0).abs() / 10_400.0;
        assert!(off < 0.10, "output {} vs ~10400", estimate.usage.output);
    }

    #[test]
    fn the_dollar_estimate_matches_both_recorded_runs() {
        // The number the guard actually shows. Sequential billed $0.1769,
        // batched $0.0971 — the same work at a 45% discount, because the
        // fan-out pays for a second cache write.
        let live = measured_layer3(false).cost;
        let batched = measured_layer3(true).cost;
        assert!((live - 0.1769).abs() / 0.1769 < 0.15, "${live:.4} vs $0.1769");
        assert!((batched - 0.0971).abs() / 0.0971 < 0.15, "${batched:.4} vs $0.0971");
        assert!(batched < live);
    }

    // One layer's shape as the five-study run actually sent it, so the cache
    // model can be checked against a measured multi-study bill rather than only
    // against the single-study one it was first fitted to.
    fn shaped(calls: usize, groups: usize, wide: usize, prompt: usize, batch: bool) -> Estimate {
        Estimate {
            layer: "l",
            model: "m".into(),
            batch,
            calls,
            groups,
            wide_groups: wide,
            usage: Usage::default(),
            cost: 0.0,
        }
        .with_cache(prompt, batch)
    }

    #[test]
    fn the_cache_model_matches_the_measured_five_study_run() {
        // Run 20260818T044453Z: layer 3 sent 52 calls as five batches (22, 15,
        // 7, 6, 2) and layer 4 sent four batches of one. Billed 69,491 written
        // and 271,544 read. The previous model said 25,752 written — 2.7× under,
        // because it treated the cache as a property of the run.
        let text = shaped(52, 5, 2, 16_652, true);
        let paper = shaped(4, 4, 0, 18_110, true);
        let written = text.usage.cache_write + paper.usage.cache_write;
        let read = text.usage.cache_read + paper.usage.cache_read;

        assert!(
            (written as f64 - 69_491.0).abs() / 69_491.0 < 0.02,
            "cache_write {written} vs 69491"
        );
        assert!(
            (read as f64 - 271_544.0).abs() / 271_544.0 < 0.03,
            "cache_read {read} vs 271544"
        );
    }

    #[test]
    fn a_narrow_batch_does_not_race_a_second_write() {
        // Layer 4 sends one request per batch and was billed exactly one write
        // per study — four batches, four writes. A model that charged every
        // batch twice would overstate the expensive layer most.
        let paper = shaped(4, 4, 0, 18_110, true);
        assert_eq!(paper.usage.cache_write, tokens(18_110) * 4);
        assert_eq!(paper.usage.cache_read, 0, "each batch of one is its own cold start");
    }

    #[test]
    fn sequential_calls_share_one_warm_prefix_across_studies() {
        // No fan-out, so the entry written by the first call serves every call
        // after it whichever study it belongs to. Measured at 6,148 written for
        // a 30-call sequential run.
        let live = shaped(52, 5, 2, 16_652, false);
        assert_eq!(live.usage.cache_write, tokens(16_652));
        assert_eq!(live.usage.cache_read, tokens(16_652) * 51);
    }

    #[test]
    fn a_batched_layer_pays_for_the_prefix_once_per_study() {
        // The whole correction, end to end: three studies is three fan-outs.
        let mut layers = free();
        layers.push(naive(true));
        let batched = for_corpus(&mini(), &SchemaSettings::new(layers), &Budget::new(0.0));

        let mut layers = free();
        layers.push(naive(false));
        let live = for_corpus(&mini(), &SchemaSettings::new(layers), &Budget::new(0.0));

        assert!(batched.studies > 1);
        assert_eq!(batched.layers[0].groups, batched.studies);
        assert_eq!(
            batched.layers[0].usage.cache_write,
            live.layers[0].usage.cache_write * batched.studies as u64,
            "a batched run pays for the instructions once per study"
        );
    }

    #[test]
    fn batching_pays_for_the_second_cache_write_the_measurement_showed() {
        // 12,296 written and 172,144 read over 30 calls, against 6,148 and
        // 178,292 sequential. Modelling one write would put the batch discount
        // at 50% and it is 45%.
        let batched = measured_layer3(true);
        assert_eq!(batched.usage.cache_write, 2 * measured_layer3(false).usage.cache_write);
        assert_eq!(batched.usage.cache_read, batched.usage.cache_write / 2 * 28);
    }

    // -- the token model -----------------------------------------------------

    #[test]
    fn thinking_is_counted_as_output_because_that_is_how_it_bills() {
        // 9,205 output tokens of which 8,427 were thinking: the API reports
        // thinking inside output, so an estimate adding them separately would
        // price a thinking run as if it were free.
        let config = |thinking| {
            ModelConfig::new("m", TEXT_SYSTEM_FULL).thinking(thinking).effort(None)
        };
        let off = of_jobs("l", &jobs(2, 500, 10), &Priced(ModelId::Haiku45), &config(Thinking::Disabled));
        let on = of_jobs(
            "l",
            &jobs(2, 500, 10),
            &Priced(ModelId::Haiku45),
            &config(Thinking::Enabled { budget_tokens: 4000 }),
        );
        assert_eq!(on.usage.thinking, 8_000);
        assert_eq!(on.usage.output, off.usage.output + 8_000);
        assert!(on.cost > off.cost, "a thinking run must not estimate as cheaper");
    }

    #[test]
    fn an_adaptive_layer_is_estimated_rather_than_treated_as_free() {
        // The roughest number this produces, and the one most worth not
        // rounding to zero: Haiku averaged 4,193 thinking tokens a call.
        let adaptive = of_jobs(
            "l",
            &jobs(4, 500, 10),
            &Priced(ModelId::Haiku45),
            &ModelConfig::new("m", TEXT_SYSTEM_FULL).thinking(Thinking::Adaptive),
        );
        assert!(adaptive.usage.thinking > 4 * 4_000);
    }

    #[test]
    fn a_layer_with_nothing_to_ask_costs_nothing() {
        let empty = of_jobs(
            "l",
            &[],
            &Priced(ModelId::Sonnet5),
            &ModelConfig::new("m", TEXT_SYSTEM_FULL),
        );
        assert_eq!(empty.calls, 0);
        assert_eq!(empty.cost, 0.0);
        assert_eq!(empty.usage, Usage::default(), "no call means no cached prefix either");
    }

    #[test]
    fn a_single_call_writes_the_prefix_and_reads_it_back_never() {
        let one = of_jobs(
            "l",
            &jobs(1, 500, 10),
            &Priced(ModelId::Sonnet5),
            &ModelConfig::new("m", TEXT_SYSTEM_FULL).batch(true),
        );
        assert_eq!(one.usage.cache_read, 0);
        assert!(one.usage.cache_write > 0);
    }

    #[test]
    fn the_estimate_scales_with_the_work() {
        let small = of_jobs("l", &jobs(2, 500, 10), &Priced(ModelId::Sonnet5), &ModelConfig::new("m", ""));
        let big = of_jobs("l", &jobs(20, 500, 10), &Priced(ModelId::Sonnet5), &ModelConfig::new("m", ""));
        assert!(big.cost > small.cost * 5.0);
    }

    // -- walking a corpus ----------------------------------------------------

    // The same real slice the corpus and sequencing tests use: 3 studies, 4
    // experiments, and one study carrying 3 papers — enough to exercise both
    // caps and both paid layers.
    const MINI: &str = include_str!("../test_data/mini_corpus.json");

    fn mini() -> Corpus {
        Corpus::from_json(MINI, false).expect("fixture should parse")
    }

    fn naive(batch: bool) -> Layer {
        Layer::LLMNaive {
            model: Box::new(Priced(ModelId::Sonnet5)),
            config: ModelConfig::new("claude-sonnet-5", TEXT_SYSTEM_FULL)
                .thinking(Thinking::Disabled)
                .batch(batch),
        }
    }

    fn free() -> Vec<Layer> {
        vec![Layer::Direct, Layer::Harmonized]
    }

    #[test]
    fn a_free_run_estimates_nothing_to_confirm() {
        let corpus = mini();
        let expected: usize = corpus.projects.iter().map(|p| p.experiments.len()).sum();
        let estimate = for_corpus(&corpus, &SchemaSettings::new(free()), &Budget::new(0.0));
        assert_eq!(estimate.cost(), 0.0);
        assert_eq!(estimate.calls(), 0);
        assert!(estimate.layers.is_empty());
        assert_eq!(estimate.records, expected, "the free layers still ran");
    }

    #[test]
    fn the_estimate_obeys_the_same_caps_as_the_run() {
        // The point of estimating rather than extrapolating. A guard that
        // priced the whole corpus when the run is capped would be ignored
        // within a week.
        //
        // Counted in *calls*, not records. The record cap's job is to bound
        // what the paid layer is asked to do, and a run that truncates its
        // output after paying for it is exactly the failure the cap exists to
        // prevent — so an estimate that trimmed only the record list would
        // agree with this one and still be wrong.
        let estimate = |settings| for_corpus(&mini(), &settings, &Budget::new(0.0));

        let mut layers = free();
        layers.push(naive(false));
        let one_study = estimate(SchemaSettings::new(layers).max_studies(1));

        let mut layers = free();
        layers.push(naive(false));
        let one_record =
            estimate(SchemaSettings::new(layers).max_studies(1).max_total_records(1));

        // The first study has two experiments on two samples: capped to one
        // record it is a study call and one sample call instead of two.
        assert_eq!((one_study.studies, one_study.records), (1, 2));
        assert_eq!((one_record.studies, one_record.records), (1, 1));
        assert!(
            one_record.calls() < one_study.calls(),
            "the record cap did not reach the plan: {} vs {} calls",
            one_record.calls(),
            one_study.calls()
        );
        assert!(one_record.cost() < one_study.cost());

        // and the study cap bounds it too
        let mut layers = free();
        layers.push(naive(false));
        let everything = estimate(SchemaSettings::new(layers));
        assert!(everything.studies > 1);
        assert!(everything.calls() > one_study.calls());
    }

    #[test]
    fn the_estimate_counts_the_calls_the_run_would_make() {
        // Not a model of the workload — the workload. Both go through
        // `llm_naive::plan`, so a change to what gets asked moves both.
        let corpus = mini();
        let mut layers = free();
        layers.push(naive(false));
        let estimate = for_corpus(&corpus, &SchemaSettings::new(layers), &Budget::new(0.0));

        let planned: usize = corpus
            .projects
            .iter()
            .map(|project| {
                let schemas = TargetSchema::from_project(
                    project.clone(),
                    &SchemaSettings::new(free()),
                );
                crate::layer::llm_naive::plan(project, &schemas).len()
            })
            .sum();
        assert_eq!(estimate.calls(), planned);
        assert!(planned > 0);
    }

    #[test]
    fn the_estimate_prices_the_studies_that_were_selected() {
        // The guard is only a guard if it prices what is about to run. Selection
        // and the caps are separate mechanisms and both have to reach here.
        let mut layers = free();
        layers.push(naive(false));
        let one = for_corpus(
            &mini(),
            &SchemaSettings::new(layers).only_studies(["DRP003937"]),
            &Budget::new(0.0),
        );

        let mut layers = free();
        layers.push(naive(false));
        let all = for_corpus(&mini(), &SchemaSettings::new(layers), &Budget::new(0.0));

        assert_eq!(one.studies, 1);
        assert!(all.studies > 1);
        assert!(one.calls() < all.calls(), "{} vs {} calls", one.calls(), all.calls());
        assert!(one.cost() < all.cost());

        // and the priced plan is the selected study's, not the corpus's first
        let selected = mini()
            .projects
            .into_iter()
            .find(|p| p.accession.0 == "DRP003937")
            .unwrap();
        let schemas =
            TargetSchema::from_project(selected.clone(), &SchemaSettings::new(free()));
        assert_eq!(
            one.calls(),
            crate::layer::llm_naive::plan(&selected, &schemas).len()
        );
    }

    #[test]
    fn an_empty_selection_estimates_nothing_to_confirm() {
        // The configuration that would be most dangerous to misread: an empty
        // list means "no studies", and reading it as "unset" would quote — and
        // then run — the whole corpus.
        let none: [&str; 0] = [];
        let mut layers = free();
        layers.push(naive(false));
        let estimate = for_corpus(
            &mini(),
            &SchemaSettings::new(layers).only_studies(none),
            &Budget::new(0.0),
        );
        assert_eq!(estimate.calls(), 0);
        assert_eq!(estimate.cost(), 0.0);
        assert_eq!(estimate.records, 0);
    }

    #[test]
    fn the_cached_prefix_is_priced_once_for_the_run_not_once_per_study() {
        // The instructions are written to the cache once and read back on every
        // later call, across studies. Accumulating the cache per project would
        // keep whichever figure the first study produced and under-count every
        // read after it — an understatement, which is the direction a spend
        // guard must never be wrong in.
        let mut layers = free();
        layers.push(naive(false));
        let estimate = for_corpus(&mini(), &SchemaSettings::new(layers), &Budget::new(0.0));

        let layer = &estimate.layers[0];
        assert!(estimate.studies > 1, "needs more than one study to be a test");
        assert!(layer.calls > estimate.studies, "needs more calls than studies");
        assert_eq!(
            layer.usage.cache_read,
            layer.usage.cache_write * (layer.calls as u64 - 1),
            "{} calls over {} studies",
            layer.calls,
            estimate.studies
        );
    }

    #[test]
    fn the_paper_layer_is_estimated_at_one_call_per_retrievable_paper() {
        // Layer 4 is the expensive one per call — 30,000 characters against a
        // sample bag's few hundred — so a guard that quoted it at layer 3's
        // rate would understate the run it exists to gate.
        let corpus = mini();
        let mut layers = free();
        layers.push(Layer::LLMPaper {
            model: Box::new(Priced(ModelId::Sonnet5)),
            config: ModelConfig::new("claude-sonnet-5", crate::layer::llm_paper::PAPER_SYSTEM)
                .thinking(Thinking::Disabled),
        });
        let estimate = for_corpus(&corpus, &SchemaSettings::new(layers), &Budget::new(0.0));

        let papers: usize = corpus.projects.iter().map(|p| crate::corpus::papers_of(p).len()).sum();
        assert_eq!(estimate.calls(), papers);
        assert!(papers > 0);

        // The fixture's papers are 280-character stubs, so its *size* proves
        // nothing. What a real paper costs is asserted where the size is under
        // the test's control: a 30,000-character call against a 540-character
        // one, which is the ratio the corpus actually has.
        let model = Priced(ModelId::Sonnet5);
        let config = ModelConfig::new("m", crate::layer::llm_paper::PAPER_SYSTEM)
            .thinking(Thinking::Disabled);
        let paper = of_jobs("llm_paper", &jobs(1, 30_000, 19), &model, &config);
        let bag = of_jobs("llm_naive", &jobs(1, 540, 22), &model, &config);
        assert!(
            paper.usage.input > bag.usage.input * 20,
            "{} vs {}",
            paper.usage.input,
            bag.usage.input
        );
    }

    #[test]
    fn each_layer_is_estimated_against_its_own_model_and_settings() {
        // The reason per-layer configs exist: a run putting a cheap model on
        // layer 3 and an expensive one on layer 4 must not be quoted one price.
        let mut layers = free();
        layers.push(Layer::LLMNaive {
            model: Box::new(Priced(ModelId::Haiku45)),
            config: ModelConfig::new("claude-haiku-4-5", TEXT_SYSTEM_FULL)
                .thinking(Thinking::Disabled),
        });
        layers.push(Layer::LLMPaper {
            model: Box::new(Priced(ModelId::Opus5)),
            config: ModelConfig::new("claude-opus-5", crate::layer::llm_paper::PAPER_SYSTEM)
                .thinking(Thinking::Disabled),
        });
        let estimate = for_corpus(&mini(), &SchemaSettings::new(layers), &Budget::new(0.0));

        assert_eq!(estimate.layers.len(), 2);
        assert_eq!(estimate.layers[0].layer, "llm_naive");
        assert_eq!(estimate.layers[0].model, "claude-haiku-4-5");
        assert_eq!(estimate.layers[1].layer, "llm_paper");
        assert_eq!(estimate.layers[1].model, "claude-opus-5");
        assert!(
            estimate.layers[1].cost > estimate.layers[0].cost,
            "Opus over a paper cannot be quoted below Haiku over a bag: {:?}",
            estimate.layers
        );
    }

    #[test]
    fn the_ceiling_is_reported_when_the_estimate_would_trip_it() {
        // Worth saying before the run rather than after: tripping `max_spend`
        // stops a layer part-way, and half a layer is a paid result that cannot
        // be compared with anything.
        let mut layers = free();
        layers.push(naive(false));
        let settings = SchemaSettings::new(layers);

        let tight = for_corpus(&mini(), &settings, &Budget::new(0.000_1));
        assert!(tight.exceeds_ceiling());
        assert!(tight.to_string().contains("stop part-way"), "{tight}");

        assert!(!for_corpus(&mini(), &settings, &Budget::new(100.0)).exceeds_ceiling());
    }

    #[test]
    fn estimating_leaves_the_callers_evidence_store_alone() {
        // A dry pass writing into the real store would leave the saved run
        // reporting evidence for calls that never happened.
        let mut layers = free();
        layers.push(naive(false));
        let settings = SchemaSettings::new(layers).keep_evidence(true);
        let _ = for_corpus(&mini(), &settings, &Budget::new(0.0));
        assert_eq!(settings.evidence(), Some(Default::default()));
    }

    #[test]
    fn the_summary_names_every_layer_and_the_total() {
        let mut layers = free();
        layers.push(naive(true));
        let text = for_corpus(&mini(), &SchemaSettings::new(layers), &Budget::new(1.0)).to_string();
        assert!(text.contains("llm_naive"));
        assert!(text.contains("claude-sonnet-5"));
        assert!(text.contains("batched"));
        assert!(text.contains("TOTAL"));
        assert!(text.contains("max_spend"));
    }
}
