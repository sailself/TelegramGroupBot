use std::time::Duration;

use tracing::{info, warn};

use crate::config::CONFIG;
use crate::state::AppState;

pub async fn run_agent_hygiene_once(state: &AppState) {
    let memories_deleted = match state
        .db
        .prune_agent_memories_older_than(CONFIG.agent_memory_retention_days)
        .await
    {
        Ok(count) => count,
        Err(err) => {
            warn!("Agent hygiene failed pruning memories: {}", err);
            0
        }
    };

    let (sessions_deleted, steps_deleted, calls_deleted, skills_deleted) = match state
        .db
        .prune_agent_sessions_older_than(CONFIG.agent_session_retention_days)
        .await
    {
        Ok(counts) => counts,
        Err(err) => {
            warn!("Agent hygiene failed pruning sessions: {}", err);
            (0, 0, 0, 0)
        }
    };

    if memories_deleted > 0
        || sessions_deleted > 0
        || steps_deleted > 0
        || calls_deleted > 0
        || skills_deleted > 0
    {
        info!(
            "Agent hygiene pruned rows: memories={} sessions={} steps={} tool_calls={} session_skills={}",
            memories_deleted, sessions_deleted, steps_deleted, calls_deleted, skills_deleted
        );
    }
}

pub fn spawn_agent_hygiene_task(state: AppState) {
    if !CONFIG.agent_hygiene_enabled {
        info!("Agent hygiene loop disabled via AGENT_HYGIENE_ENABLED=false");
        return;
    }

    let interval_secs = CONFIG.agent_hygiene_interval_seconds.max(60);
    tokio::spawn(async move {
        run_agent_hygiene_once(&state).await;

        let mut ticker = tokio::time::interval(Duration::from_secs(interval_secs));
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            ticker.tick().await;
            run_agent_hygiene_once(&state).await;
        }
    });
}
