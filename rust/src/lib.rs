pub mod audit;
pub mod corpus;
pub mod estimate;
pub mod export;
pub mod dto;
pub mod project;
pub mod target_schema;
pub mod layer;
pub mod model;
pub mod utils;

// Everything a run needs, in one import.
//
// Re-exports reach past the module tree where the tree is an implementation
// detail: `ModelId` lives in `model::claude` because that is whose ids they
// are, and the prompt constants live beside the layer that sends them, but a
// caller configuring a run wants all of them at once.
pub mod prelude {
    pub use crate::audit::{self, verbatim, Audit};
    pub use crate::corpus::*;
    pub use crate::estimate::{self, Estimate, Estimates};
    pub use crate::export::{Export, LayerParams, Params};
    pub use crate::project::*;
    pub use crate::target_schema::*;
    pub use crate::layer::*;
    pub use crate::layer::llm_naive::{TEXT_SYSTEM_FULL, TEXT_SYSTEM_SHORT, TEXT_SYSTEM_TARGETED};
    pub use crate::layer::llm_paper::PAPER_SYSTEM;
    pub use crate::model::*;
    pub use crate::model::budget::{Budget, Budgeted, Ledger};
    pub use crate::model::claude::{Claude, ModelId};
    pub use crate::model::retry::{RetryPolicy, Retrying};
    pub use crate::utils::*;
}
