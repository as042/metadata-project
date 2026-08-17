use crate::{model::Model, project::Project, target_schema::TargetSchema};

pub mod direct;
pub mod harmonized;

// One stage of the reconstruction cascade.
//
// A layer only ever fills fields no earlier layer settled, which is what makes
// the order in `SchemaSettings` meaningful rather than merely cosmetic: run the
// cheap deterministic layers first and a later one pays only for what they
// could not answer. Nothing enforces that ordering, deliberately — the point of
// the list is that a colleague can reorder it, including into arrangements that
// cost more, without editing this enum.
//
// The variants divide on cost, not on technique. Direct and Harmonized are free
// and offline; both LLM variants bill per record against whichever `Model` they
// are handed.
pub enum Layer {
    // Fields the archive states outright. The only layer that creates records.
    Direct,
    // A synonym table mapping submitter attribute keys onto schema fields. Free
    // and offline, but unlike Direct the *key* mapping is ours, which is why it
    // gets its own provenance rather than folding into Direct.
    Harmonized,
    // Reads the archive's own text and attribute bags.
    LLMNaive(Box<dyn Model>),
    // Reads the linked publication. Only 346 of the studies have retrievable
    // text at all, so this layer is a no-op for most records while still
    // costing a request for the ones it can answer.
    LLMPaper(Box<dyn Model>),
}

impl Layer {
    // Runs one layer over one project.
    //
    // `schemas` is in-out because layers are not uniform: Direct appends the
    // records, and every other layer fills fields on records already there.
    // A layer scheduled before Direct therefore sees an empty slice and does
    // nothing — it is not an error, just wasted work.
    //
    // Takes `&self` so one SchemaSettings can drive every project in a corpus;
    // consuming the layer would make the settings single-use.
    #[inline]
    pub(crate) fn process(&self, project: &Project, schemas: &mut Vec<TargetSchema>) {
        match self {
            Layer::Direct => direct::process(project, schemas),
            Layer::Harmonized => harmonized::process(project, schemas),
            Layer::LLMNaive(_model) => todo!(),
            Layer::LLMPaper(_model) => todo!(),
        }
    }

    // Whether running this layer can bill an API. Not currently consulted by
    // anything; it exists so a spend guard has one place to ask rather than
    // matching on the enum at each call site.
    #[inline]
    pub fn is_paid(&self) -> bool {
        matches!(self, Layer::LLMNaive(_) | Layer::LLMPaper(_))
    }
}