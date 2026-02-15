use crate::skills::types::SkillDoc;

pub fn build_skill_index(skills: &[SkillDoc]) -> String {
    let mut lines = Vec::new();
    lines.push("Skill catalog:".to_string());

    for skill in skills {
        let tags = if skill.meta.tags.is_empty() {
            "-".to_string()
        } else {
            skill.meta.tags.join(", ")
        };
        lines.push(format!(
            "- {}: {} (tags: {}; risk: {})",
            skill.meta.name, skill.meta.description, tags, skill.meta.risk_level
        ));
    }

    lines.join("\n")
}

pub fn build_selected_skill_context(skills: &[SkillDoc]) -> String {
    let mut blocks = Vec::new();
    for skill in skills {
        blocks.push(format!(
            "Skill: {}\nDescription: {}\nAllowed tools: {}\nInstructions:\n{}",
            skill.meta.name,
            skill.meta.description,
            if skill.meta.allowed_tools.is_empty() {
                "-".to_string()
            } else {
                skill.meta.allowed_tools.join(", ")
            },
            skill.body
        ));
    }
    blocks.join("\n\n")
}
