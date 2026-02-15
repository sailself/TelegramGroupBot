use std::collections::{HashMap, HashSet};

use tracing::{debug, warn};

use crate::config::CONFIG;
use crate::llm::call_gemini;
use crate::skills::types::{ActiveSkillSet, SkillDoc};

fn tokenize(text: &str) -> HashSet<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .map(str::trim)
        .filter(|token| !token.is_empty())
        .map(str::to_string)
        .collect()
}

fn heuristic_score(prompt_tokens: &HashSet<String>, skill: &SkillDoc) -> i32 {
    let mut score = 0;
    let lower_name = skill.meta.name.to_lowercase();
    let lower_desc = skill.meta.description.to_lowercase();
    let joined_prompt = prompt_tokens.iter().cloned().collect::<Vec<_>>().join(" ");

    for trigger in &skill.meta.triggers {
        let trigger = trigger.trim().to_lowercase();
        if trigger.is_empty() {
            continue;
        }
        if joined_prompt.contains(&trigger) {
            score += 5;
        }
    }

    for token in prompt_tokens {
        if lower_name.contains(token) {
            score += 3;
        }
        if lower_desc.contains(token) {
            score += 1;
        }
        if skill.meta.tags.iter().any(|tag| tag == token) {
            score += 2;
        }
    }

    score
}

fn pick_heuristic_candidates(
    prompt: &str,
    skills: &[SkillDoc],
    candidate_limit: usize,
) -> Vec<SkillDoc> {
    let prompt_tokens = tokenize(prompt);
    let mut scored = skills
        .iter()
        .cloned()
        .map(|skill| {
            let score = heuristic_score(&prompt_tokens, &skill);
            (score, skill)
        })
        .collect::<Vec<_>>();
    scored.sort_by(|a, b| {
        b.0.cmp(&a.0)
            .then_with(|| a.1.meta.name.cmp(&b.1.meta.name))
    });

    let mut selected = Vec::new();
    for (score, skill) in scored {
        if selected.len() >= candidate_limit {
            break;
        }
        if score <= 0 && !selected.is_empty() {
            break;
        }
        selected.push(skill);
    }

    if selected.is_empty() {
        return skills.iter().take(candidate_limit).cloned().collect();
    }
    selected
}

fn parse_json_array_from_text(raw: &str) -> Option<Vec<String>> {
    let trimmed = raw.trim();
    if let Ok(arr) = serde_json::from_str::<Vec<String>>(trimmed) {
        return Some(arr);
    }

    let start = trimmed.find('[')?;
    let end = trimmed.rfind(']')?;
    if end <= start {
        return None;
    }
    let candidate = &trimmed[start..=end];
    serde_json::from_str::<Vec<String>>(candidate).ok()
}

async fn llm_select_skills(
    prompt: &str,
    candidates: &[SkillDoc],
    max_active_skills: usize,
) -> Option<Vec<String>> {
    if candidates.is_empty() || CONFIG.gemini_api_key.trim().is_empty() {
        return None;
    }

    let mut candidate_lines = Vec::new();
    for skill in candidates {
        candidate_lines.push(format!("- {}: {}", skill.meta.name, skill.meta.description));
    }

    let selection_prompt = format!(
        "User request:\n{}\n\nChoose up to {} skills from the candidate list that best fit the request.\nReturn only a JSON array of skill names.\n\nCandidates:\n{}",
        prompt,
        max_active_skills,
        candidate_lines.join("\n")
    );
    let system_prompt = "You select the best skills for a coding agent. Return only valid JSON array like [\"skill-a\", \"skill-b\"].";

    let response = call_gemini(
        system_prompt,
        &selection_prompt,
        None,
        false,
        false,
        None,
        None,
        false,
        None,
        None,
        Some("agent_skill_selection"),
    )
    .await;

    match response {
        Ok(text) => parse_json_array_from_text(&text),
        Err(err) => {
            warn!("Skill selection model call failed: {}", err);
            None
        }
    }
}

pub async fn select_active_skills(
    prompt: &str,
    all_skills: &[SkillDoc],
    candidate_limit: usize,
    max_active_skills: usize,
) -> ActiveSkillSet {
    let always_active = all_skills
        .iter()
        .filter(|skill| skill.always_active)
        .cloned()
        .collect::<Vec<_>>();
    let selectable = all_skills
        .iter()
        .filter(|skill| !skill.always_active)
        .cloned()
        .collect::<Vec<_>>();

    let heuristic_candidates = pick_heuristic_candidates(prompt, &selectable, candidate_limit);
    let llm_selected_names = llm_select_skills(prompt, &heuristic_candidates, max_active_skills)
        .await
        .unwrap_or_default()
        .into_iter()
        .map(|name| name.trim().to_lowercase())
        .collect::<Vec<_>>();

    let candidate_map = heuristic_candidates
        .iter()
        .cloned()
        .map(|skill| (skill.meta.name.to_lowercase(), skill))
        .collect::<HashMap<_, _>>();

    let mut selected = Vec::new();
    selected.extend(always_active.clone());

    let mut selected_name_set = selected
        .iter()
        .map(|skill| skill.meta.name.to_lowercase())
        .collect::<HashSet<_>>();

    if !llm_selected_names.is_empty() {
        for name in llm_selected_names {
            if selected.len() >= max_active_skills + always_active.len() {
                break;
            }
            if let Some(skill) = candidate_map.get(&name) {
                if selected_name_set.insert(skill.meta.name.to_lowercase()) {
                    selected.push(skill.clone());
                }
            }
        }
    }

    if selected.len() <= always_active.len() {
        for skill in heuristic_candidates {
            if selected.len() >= max_active_skills + always_active.len() {
                break;
            }
            if selected_name_set.insert(skill.meta.name.to_lowercase()) {
                selected.push(skill);
            }
        }
    }

    let mut allowed_tools_set = HashSet::new();
    for skill in &selected {
        for tool in &skill.meta.allowed_tools {
            allowed_tools_set.insert(tool.clone());
        }
    }
    let mut allowed_tools = allowed_tools_set.into_iter().collect::<Vec<_>>();
    allowed_tools.sort();

    let selected_names = selected
        .iter()
        .map(|skill| skill.meta.name.clone())
        .collect::<Vec<_>>();
    debug!(
        "Selected skills: [{}], allowed_tools=[{}]",
        selected_names.join(", "),
        allowed_tools.join(", ")
    );

    ActiveSkillSet {
        selected,
        selected_names,
        allowed_tools,
    }
}
