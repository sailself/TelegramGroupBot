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
        let version = skill.meta.version.as_deref().unwrap_or("-");
        let source = skill
            .source_path
            .as_ref()
            .and_then(|path| path.file_name())
            .and_then(|name| name.to_str())
            .unwrap_or("built-in");
        lines.push(format!(
            "- {}: {} (version: {}; source: {}; tags: {}; risk: {})",
            skill.meta.name, skill.meta.description, version, source, tags, skill.meta.risk_level
        ));
    }

    lines.join("\n")
}

pub fn build_selected_skill_context(skills: &[SkillDoc]) -> String {
    let mut blocks = Vec::new();
    for skill in skills {
        let version = skill.meta.version.as_deref().unwrap_or("-");
        let source = skill
            .source_path
            .as_ref()
            .and_then(|path| path.file_name())
            .and_then(|name| name.to_str())
            .unwrap_or("built-in");
        blocks.push(format!(
            "Skill: {}\nDescription: {}\nVersion: {}\nSource: {}\nAllowed tools: {}\nInstructions:\n{}",
            skill.meta.name,
            skill.meta.description,
            version,
            source,
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
