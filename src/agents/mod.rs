//! Multi-phase agentic pipelines behind the complex commands.
//!
//! Each pipeline decomposes a command into bounded phases (plan → act →
//! reflect → synthesize) whose intermediate state lives in Rust structs
//! instead of an ever-growing model transcript. Cheap orchestration steps run
//! on the configured step model (`AGENT_STEP_MODEL`, derived when unset);
//! the user-facing final answer keeps using the command's configured model.

pub mod factcheck;
pub mod qc;
pub mod step;
pub mod tldr;
