use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentProvider {
    OpenRouter,
    Gemini,
}

impl AgentProvider {
    pub fn from_str(value: &str) -> Self {
        match value.trim().to_lowercase().as_str() {
            "gemini" => AgentProvider::Gemini,
            _ => AgentProvider::OpenRouter,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            AgentProvider::OpenRouter => "openrouter",
            AgentProvider::Gemini => "gemini",
        }
    }
}

#[derive(Debug, Clone)]
pub enum AgentRunOutcome {
    Completed {
        session_id: i64,
        response_text: String,
        selected_skills: Vec<String>,
    },
    AwaitingConfirmation {
        confirmation_key: String,
        notice_text: String,
    },
}

#[derive(Debug, Clone)]
pub struct PendingAgentAction {
    pub provider: AgentProvider,
    pub system_prompt: String,
    pub user_id: i64,
    pub chat_id: i64,
    pub session_id: i64,
    pub processing_message_id: i64,
    pub tool_call_record_id: i64,
    pub tool_call_id: String,
    pub tool_name: String,
    pub tool_args: Value,
    pub model_name: String,
    pub allowed_tools: Vec<String>,
    pub selected_skills: Vec<String>,
    pub messages: Vec<Value>,
}
