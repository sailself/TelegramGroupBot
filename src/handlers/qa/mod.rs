mod trigger;
pub use trigger::{build_auto_q_query, should_auto_q_trigger};

mod prompt;

mod selection_ui;
pub use selection_ui::{model_selection_callback, MODEL_CALLBACK_PREFIX};

mod model_resolution;

mod chat_search;

mod process;

mod handler;
pub use handler::{q_handler, qc_handler, qq_handler, s_handler};

#[cfg(test)]
mod tests;
